"""Fixed multibudget E2 development calibration; no confirmation, N1 or training."""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np
import torch
from . import e2_pilot as entry
from .diagnostic_archive import write_archive
from .frozen_forward import FrozenForward, PlanStreams
from .frozen_stats import ScalarStream, TangentStream, outward_directions, promote_points
from .geometry import dot, exp, log
from .mechanisms import dependency_hashes, source_identity
from .protocols import digest
from .train import synchronize
from .vector_structure import METRICS, RadialPanel


FANOUTS=[4,8,16,32,64]
CONDITIONS={'local_L1':0,'F/S':1,'S/F':1,'S/S':1}
BASE_SEED=2026091701
PILOT={**entry.PILOT,'namespace':'E2-multibudget-development-v1','sampling_seed':BASE_SEED,
       'repetitions':128,'internal_seconds':3300,'outer_seconds':3600,
       'conditions':list(CONDITIONS),'task_repetitions':8,'fanouts':FANOUTS,
       'execution_layout':'one_checkpoint_one_fanout_per_job'}
PILOT.pop('fanout')


def validate_config(config):
    if (config.get('protocol')!='E2-multibudget-development-v1' or config.get('pilot')!=PILOT
            or config.get('numerical_limits')!=entry.LIMITS):raise ValueError('fixed multibudget development configuration required')
    local=json.loads((Path(__file__).resolve().parents[2]/'configs/e2_entry_local_pilot.json').read_text(encoding='utf-8'))
    if config.get('checkpoints')!=local['checkpoints']:raise ValueError('unchanged two selected checkpoints required')
    return digest(config)


def require_cuda(device):
    if (device.type!='cuda' or device.index not in (None,0) or not os.environ.get('SLURM_JOB_ID')
            or not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count()!=1):
        raise ValueError('new CUDA paths require one visible Slurm GPU with binding preserved')


def verify_cuda_evidence(path,approval,identity):
    if entry.file_sha256(path)!=approval.get('cuda_fixture_artifact_sha256'):
        raise ValueError('CUDA fixture artifact hash mismatch')
    row=json.loads(Path(path).read_text(encoding='utf-8'))
    if (row.get('cuda_fixture_passed') is not True or row.get('status')!='complete'
            or row.get('source',{}).get('source_commit')!=identity['source_commit']):
        raise ValueError('passed CUDA fixture for this fixed source required')
    hashes=row.get('source_sha256_normalized_lf',{})
    required={'acl_hct/'+name for name in ('e2_development.py','e2_pilot.py','vector_structure.py','diagnostic_archive.py')}
    if not required.issubset(hashes):raise ValueError('CUDA fixture source evidence incomplete')
    for name,wanted in hashes.items():
        if not re.fullmatch(r'acl_hct/[A-Za-z_][A-Za-z_0-9]*\.py',name):raise ValueError('invalid fixture source path')
        path=Path(__file__).parent/Path(name).name
        if hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()!=wanted:
            raise ValueError('CUDA fixture/source byte mismatch')


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix('.json.tmp')
    with temporary.open('w',encoding='utf-8',newline='\n') as stream:
        json.dump(entry.jsonable(value),stream,allow_nan=False,separators=(',',':'))
    temporary.replace(path)


def moment_checks(stats):
    checks=[]
    entry.check_max(checks,'mse_decomposition',stats['mse_decomposition_max_residual'],entry.LIMITS['mse_decomposition'])
    for key,value in [('variance',stats['variance']),('mse',stats['mse']['mean'])]:
        observation=entry.summarize(value)
        checks.append({'name':key+'_minimum','minimum':entry.LIMITS['variance_mse_minimum'],
                       'observed':observation,'status':'passed' if observation['minimum']>=entry.LIMITS['variance_mse_minimum'] else 'failed'})
    return checks


def packed_ranking(ranking):
    return {**{key:value for key,value in ranking.items() if key!='rows'},
            'row_int_columns':['parent','child','candidates'],
            'row_ids':np.array([[row[k] for k in ('parent','child','candidates')] for row in ranking['rows']],dtype=np.int64).reshape(-1,3),
            'ranks':np.array([row['rank'] for row in ranking['rows']],dtype=np.float64)}


def structure_vector(structure):
    columns=[];values=[]
    for kind,rows in structure.items():
        for group,summary in rows['groups'].items():
            for metric,value in summary['weighted_covered_child_metrics'].items():
                if value is not None:columns.append(f'{kind}/{group}/{metric}');values.append(value)
    return columns,np.asarray(values,dtype=np.float64)


def geometry_design(stream):
    rows=[]
    for group,ids in stream.groups.items():
        if len(ids):rows.append((group+'/mse','mse',ids))
        valid=ids[stream.defined[ids]]
        if len(valid):rows.append((group+'/radial_projection','projection',valid))
    return rows


def geometry_vector(offsets,directions,design):
    values={'mse':dot(offsets,offsets),'projection':dot(offsets,directions)}
    return torch.stack([values[kind][ids].mean() for _,kind,ids in design]).detach().cpu().numpy()


def condition_offsets(points,base,c,local,active,native_reference):
    promoted,promotion=promote_points(points,c);checks=[];entry.promotion_checks(checks,'sample',promotion)
    offsets=log(base,promoted,c)
    if local:
        inactive=~active
        if inactive.any():
            entry.check_max(checks,'inactive_native_difference',points[inactive]-native_reference[inactive],entry.LIMITS['unchanged_output_difference'])
        offsets[inactive]=0
        target=torch.where(active[:,None],promoted,base)
    else:target=promoted  # Never erase propagation outside A or P.
    entry.check_max(checks,'tangent_constraint',dot(base,offsets),entry.LIMITS['tangent_constraint'])
    entry.check_max(checks,'roundtrip',exp(base,offsets,c)-target,entry.LIMITS['roundtrip'])
    if not all(row['status']=='passed' for row in checks):raise ValueError('sample numerical checks failed: '+json.dumps(checks))
    return offsets,promoted,checks


@torch.no_grad()
def diagnose(model,features,view,*,directory,fanouts=FANOUTS,repetitions=128,task_repetitions=8,
             max_seconds=3300.,budget=32768,namespace='E2-multibudget-development-v1',panel_target=1000,
             candidate_chunk=4096,ranking_seconds=180.,progress=None,expected_ranking=None):
    if (not fanouts or len(set(fanouts))!=len(fanouts) or any(f not in FANOUTS for f in fanouts)
            or type(repetitions) is not int or repetitions<2 or repetitions>128 or repetitions%2
            or type(task_repetitions) is not int or not 1<=task_repetitions<=min(8,repetitions)):
        raise ValueError('registered fanouts and bounded even repetition count required')
    start=time.perf_counter();device=features.device;phase='entry';current=None;streams={};structures={};task_stream=ScalarStream()
    report={'protocol':'E2-multibudget-development-v1','original_route_section':'11',
            'research_question':'Do local and two-layer sampling errors accompany structural/task damage across budgets?',
            'conclusion':'pending development evidence review','purpose_status':'insufficient_evidence',
            'status':'running','entry_status':'not_started','pilot_status':'not_started','fanouts':list(fanouts),
            'planned_fanouts':FANOUTS,'shard_only':list(fanouts)!=FANOUTS,
            'repetitions':repetitions,'task_repetitions':task_repetitions,'budgets':{},'failures':[],
            'next_experiment_authorized':False,'confirmation_evaluated':False}
    def publish():
        report['elapsed_seconds']=time.perf_counter()-start
        if progress:progress(report)
    def deadline(label):
        if time.perf_counter()-start>=max_seconds:
            report['stopped_before']=label;raise TimeoutError('fixed internal deadline reached')
    def save_budget(status):
        nonlocal current
        if current is None or current.get('artifact') is not None:return
        geometry={};quality={}
        for name,stream in streams.items():
            if stream.n>=2:
                stats=stream.finish(numerical['layers'][CONDITIONS[name]]['empirical_numerical_floor'])
                quality[name]=moment_checks(stats);geometry[name]=stats
        if status=='complete' and (set(geometry)!=set(CONDITIONS) or any(
                r['status']!='passed' for rows in quality.values() for r in rows)):
            status='failed_moment_checks'
        current['status']=status;current['scientifically_usable']=status=='complete'
        current['quality']=quality
        current['geometry_groups']={name:stats['groups'] for name,stats in geometry.items()}
        current['structure_statistics']={name:{'columns':structure_columns[name],**stream.summary()}
                                         for name,stream in structures.items() if stream.n}
        current['task_statistics']=task_stream.summary() if task_stream.n else None
        current['task_columns']=['micro_mrr','micro_change_from_full','child_macro_mrr','macro_change_from_full']
        current['completed_method_repetitions']={name:stream.n for name,stream in streams.items()}
        current['seconds']=time.perf_counter()-budget_start
        current['artifact']=write_archive(directory,f'fanout_{current["fanout"]}',{
            'status':status,'fanout':current['fanout'],'seed_namespace':current['seed_namespace'],
            'scientifically_usable':status=='complete','geometry':geometry,
            'groups':groups,'near_bound_groups_by_layer':near_groups,'plan_hashes':current['plan_hashes'],
            'geometry_repeat_columns':current['geometry_repeat_columns'],
            'structure_repeat_columns':structure_columns,
            'per_repeat_structure_and_task':observations,'full_structure':full_structure,
            'condition_counts':current['completed_method_repetitions'],'numerical_audits':audits})
        # Keep public JSON bounded. Hash streams and group counts are enough there;
        # complete plans' hashes and raw per-repeat scalar evidence are in archive.
        current['plan_hashes_sha256']=digest(current.pop('plan_hashes'))
        publish()
    try:
        deadline('reference');forward=FrozenForward(model,features,view.neighbors,budget)
        deadline('numerical_entry');numerical,quality=entry.numerical_entry(forward)
        full_paths=forward.paired([view.neighbors,view.neighbors])
        for name,layer in CONDITIONS.items():
            entry.check_max(quality['checks'],name+'/paired_full_identity',
                full_paths[name]['points']-forward.reference['layers'][layer]['output'],entry.LIMITS['unchanged_output_difference'])
        quality['status']='passed' if all(row['status']=='passed' for row in quality['checks']) else 'failed'
        del full_paths
        report['numerical_entry']=quality
        if quality['status']!='passed':raise ValueError('entry numerical quality failed')
        phase='entry_valid';deadline(phase)
        index={node:i for i,node in enumerate(view.nodes)};valid=[(index[a],index[b]) for a,b in sorted(view.valid_edges)]
        parents=defaultdict(set)
        for a,b in valid:parents[b].add(a)
        full_rank=entry.filtered_parent_ranks(model,forward.reference['output'],valid,parents,candidate_chunk,
                                            min(ranking_seconds,max_seconds-(time.perf_counter()-start)))
        if full_rank['status']!='complete':raise TimeoutError('entry complete valid timed out')
        if expected_ranking is not None:
            report['entry_full_valid_reproduction']=entry.validation_reproduction(full_rank,expected_ranking)
            if report['entry_full_valid_reproduction']['status']!='passed':raise ValueError('entry valid reproduction failed')
        panels=entry.make_panels(view,panel_target);root=index.get(view.root);near_groups=[];directions=[]
        report['reference_layers']=[]
        for layer,row in enumerate(numerical['layers']):
            base=row['base'];floor=row['empirical_numerical_floor']
            radial=torch.acosh((base[:,0]*math.sqrt(model.c)).clamp_min(1.))
            near=(radial>=1.14).cpu();near_groups.append({'near_bound':near.nonzero().flatten().tolist(),
                                                       'not_near_bound':(~near).nonzero().flatten().tolist()})
            directions.append(outward_directions(base,base[root],model.c,floor+floor[root]) if root is not None
                              else (torch.zeros_like(base),torch.zeros(len(base),device=device,dtype=torch.bool)))
            message_radius=torch.acosh((forward.reference['layers'][layer]['messages'][:,0].double()*math.sqrt(model.c)).clamp_min(1.))
            report['reference_layers'].append({'layer':layer+1,'output_radius':entry.summarize(radial),
                'message_radius':entry.summarize(message_radius),'near_bound_threshold':1.14,
                'near_bound_output_count':len(near_groups[-1]['near_bound']),
                'near_bound_message_fraction':float((message_radius>=1.14).double().mean())})
        report['entry_artifact']=write_archive(directory,'entry',{
            'nodes':view.nodes,'panels':panels,'view':view.metadata,'full_valid':packed_ranking(full_rank),
            'numerical_layers':[{key:row[key] for key in ('base','empirical_numerical_floor','native_vs_fp64_full_distance',
                                'self_log_norm','log_exp_roundtrip_distance')} for row in numerical['layers']]})
        report['full_valid_reference']={k:v for k,v in full_rank.items() if k!='rows'}
        report['view']=view.metadata;report['panel_hash']=panels['hash']
        report['entry_status']='passed';report['pilot_status']='running';publish()
        for fanout in fanouts:
            phase=f'fanout{fanout}';deadline(phase);budget_start=time.perf_counter()
            groups=entry.diagnostic_groups(view,fanout)
            streams={};structures={};structure_columns={};scalar_design={};radial_panels={};full_structure={};observations=[];audits=[];task_stream=ScalarStream()
            for name,layer in CONDITIONS.items():
                layer_groups={**groups,**near_groups[layer]};base=numerical['layers'][layer]['base']
                streams[name]=TangentStream(base,model.c,layer_groups,*directions[layer])
                scalar_design[name]=geometry_design(streams[name])
                structures[name]=ScalarStream();radial_panels[name]=RadialPanel(view,panels['panels']['development'],base,layer_groups,
                                                                model.c,numerical['layers'][layer]['empirical_numerical_floor'])
                full_structure[name]=radial_panels[name].evaluate(base)
                structure_columns[name]=structure_vector(full_structure[name])[0]
            rng=PlanStreams(BASE_SEED,namespace+f'/fanout{fanout}')
            current={'fanout':fanout,'status':'running','requested_repetitions':repetitions,
                     'requested_task_repetitions':task_repetitions,'completed_graph_repetitions':0,
                     'completed_task_repetitions':0,'group_counts':{key:len(ids) for key,ids in groups.items()},
                     'group_hashes':{key:digest(ids) for key,ids in groups.items()},'seed_namespace':rng.metadata,
                     'geometry_repeat_columns':{name:[key for key,_,_ in design] for name,design in scalar_design.items()},
                     'plan_hashes':[],'artifact':None,'timing_seconds':{'paired_forward':0.,'geometry_structure':0.,'task':0.}}
            report['budgets'][str(fanout)]=current;publish()
            active=torch.zeros(len(features),dtype=torch.bool,device=device);active[groups['A']]=True
            for repetition in range(repetitions):
                deadline(f'{phase}/repeat{repetition}')
                synchronize(device);begin=time.perf_counter()
                plans=rng.draw(view.neighbors,fanout);current['plan_hashes'].append([digest(plan) for plan in plans])
                outputs=forward.paired(plans);synchronize(device)
                current['timing_seconds']['paired_forward']+=time.perf_counter()-begin
                begin=time.perf_counter();observation={'repetition':repetition,'status':'in_progress','structure':{},'geometry':{}}
                observations.append(observation)
                for name,layer in CONDITIONS.items():
                    row=outputs[name];base=numerical['layers'][layer]['base']
                    offsets,points,checks=condition_offsets(row['points'],base,model.c,name in ('local_L1','F/S'),active,
                                                           forward.reference['layers'][layer]['output'])
                    audits.append({'repetition':repetition,'condition':name,'checks':checks})
                    streams[name].add_offsets(offsets)
                    observation['geometry'][name]=geometry_vector(offsets,directions[layer][0],scalar_design[name])
                    structure=radial_panels[name].evaluate(points)
                    columns,values=structure_vector(structure)
                    if columns!=structure_columns[name]:raise ValueError('fixed structural coverage changed')
                    observation['structure'][name]=values
                    structures[name].add(torch.from_numpy(values))
                synchronize(device);current['timing_seconds']['geometry_structure']+=time.perf_counter()-begin
                if repetition<task_repetitions:
                    phase=f'fanout{fanout}/task{repetition}';deadline(phase);begin=time.perf_counter()
                    task=entry.filtered_parent_ranks(model,outputs['S/S']['points'],valid,parents,candidate_chunk,
                                                    min(ranking_seconds,max_seconds-(time.perf_counter()-start)))
                    observation['task']=packed_ranking(task)
                    if task['status']!='complete':
                        raise TimeoutError('S/S full valid task repetition incomplete')
                    task_stream.add(torch.tensor([task['query_micro_mrr'],task['query_micro_mrr']-full_rank['query_micro_mrr'],
                        task['child_macro_mrr'],task['child_macro_mrr']-full_rank['child_macro_mrr']],dtype=torch.float64))
                    current['completed_task_repetitions']+=1
                    current['timing_seconds']['task']+=time.perf_counter()-begin
                observation['status']='complete';current['completed_graph_repetitions']+=1
                del outputs,offsets,points
                phase=f'fanout{fanout}'
            save_budget('complete')
            if current['status']!='complete':raise ValueError('completed observations failed moment audit')
            streams.clear();structures.clear();radial_panels.clear();gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
        report.update(status='complete',pilot_status='complete',purpose_status='pending_supervisor_review',
                      conclusion='Fixed development observations complete; scientific interpretation remains separate')
    except (ValueError,RuntimeError,TimeoutError) as error:
        report['failures'].append({'phase':phase,'type':type(error).__name__,'error':str(error)})
        report['status']='incomplete_time_limit' if isinstance(error,TimeoutError) else 'failed'
        if report['entry_status']=='passed':
            if current is not None and observations and observations[-1]['status']!='complete':observations[-1]['status']='partial_stopped'
            report['pilot_status']=report['status'];save_budget('partial_not_for_inference')
        else:report['entry_status']=report['status']
        report['conclusion']='Incomplete development evidence; no unreviewed continuation or inference from partial conditions'
    publish();return report


def cuda_fixture(directory,source_commit):
    from .frozen_fixture import synthetic_fixture
    identity=source_identity(source_commit)
    if identity['source_commit'] is None:raise ValueError('fixed fixture source required')
    device=torch.device('cuda');require_cuda(device)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    model,features,view=synthetic_fixture();model.to(device)
    result=diagnose(model,features.to(device),view,directory=directory,fanouts=[4,16],repetitions=4,task_repetitions=1,
                    max_seconds=120,budget=128,candidate_chunk=11)
    result['scope']='40-node engineering CUDA fixture only; no real checkpoint or research evidence'
    result['source']=identity;result['source_sha256_normalized_lf']=dependency_hashes()
    result['source_sha256_normalized_lf']['acl_hct/e2_development.py']=hashlib.sha256(Path(__file__).read_text(encoding='utf-8').encode()).hexdigest()
    result['device']=torch.cuda.get_device_name(0);result['cuda_fixture_passed']=result['status']=='complete'
    write_json(Path(directory)/'cuda-quality.json',result)
    if not result['cuda_fixture_passed']:raise ValueError('CUDA engineering fixture failed')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--cuda-fixture',action='store_true');parser.add_argument('--fanout',type=int,choices=FANOUTS)
    parser.add_argument('--seed',type=int,choices=(11,23));parser.add_argument('--source-commit')
    for name in ('prepared','checkpoint','training-release','baseline-report','approval-record','cuda-fixture-record','output-dir'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.cuda_fixture:
        if args.execute or args.output_dir is None or not args.source_commit:raise ValueError('fixture requires new output directory and fixed source only')
        args.output_dir.mkdir(parents=True,exist_ok=False);cuda_fixture(args.output_dir,args.source_commit);return
    if args.config is None:raise ValueError('--config required')
    config=json.loads(args.config.read_text(encoding='utf-8'));config_hash=validate_config(config)
    if not args.execute:
        print(json.dumps({'status':'static_only','config_sha256':config_hash,'pilot':config['pilot']}));return
    if any(getattr(args,key) is None for key in ('seed','fanout','source_commit','prepared','checkpoint','training_release','baseline_report','approval_record','cuda_fixture_record','output_dir')):
        raise ValueError('execution requires selected checkpoint, fixed source, approval and new artifact directory')
    identity=source_identity(args.source_commit);approval=json.loads(args.approval_record.read_text(encoding='utf-8'))
    entry.verify_approval(approval,config,identity)
    if approval.get('cuda_fixture_passed') is not True or not approval.get('cuda_fixture_artifact_sha256'):
        raise ValueError('new CUDA fixture evidence must be reviewed before real development execution')
    verify_cuda_evidence(args.cuda_fixture_record,approval,identity)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    output=args.output_dir/'summary.json'
    write_json(output,{'status':'verification_started','config_sha256':config_hash})
    def runner(*positional,**keywords):
        return diagnose(*positional,directory=args.output_dir,fanouts=[args.fanout],**keywords)
    try:
        result=entry.run(config,args.seed,args.prepared,args.checkpoint,args.training_release,args.baseline_report,
                         approval,args.source_commit,device='cuda',progress=lambda row:write_json(output,row),
                         diagnostic_runner=runner,configuration_check=validate_config)
        result['source_sha256_normalized_lf']['acl_hct/e2_development.py']=hashlib.sha256(Path(__file__).read_text(encoding='utf-8').encode()).hexdigest()
        write_json(output,result)
    except (ValueError,RuntimeError,OSError,KeyError) as error:
        # Preserve previous independently archived budgets on wrapper failures.
        previous=json.loads(output.read_text(encoding='utf-8'));previous['wrapper_failure']=str(error)
        previous['status']='failed';write_json(output,previous);raise
    print(json.dumps({'status':result['status'],'completed_budgets':[k for k,v in result['budgets'].items() if v['status']=='complete']}))
    if result['status']!='complete':raise SystemExit(2)


if __name__=='__main__':main()

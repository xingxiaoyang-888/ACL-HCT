"""Bounded synthetic E2/E3 wiring check, not a real-checkpoint diagnostic CLI."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import torch
from .backbone import LorentzMeanNetwork
from .development_view import build_view, make_panels, support_groups
from .frozen_forward import FrozenForward, PlanStreams
from .frozen_stats import ScalarStream, TangentStream, outward_directions, promote_points
from .frozen_structure import radial_relations
from .mechanisms import dependency_hashes, source_identity
from .protocols import digest, observed_neighbors
from .ranking import filtered_parent_ranks


def synthetic_fixture(dtype=torch.float32):
    nodes=[f'n{i:02d}' for i in range(40)]
    train={(nodes[0],nodes[i]) for i in range(1,25)}
    train.update((nodes[a],nodes[b]) for a,b in ((1,25),(25,26),(2,26),(26,27)))
    valid={(nodes[a],nodes[b]) for a,b in ((0,28),(1,28),(25,29),(29,30),(31,32),(32,33),(34,35))}
    valid_entities=[nodes[0]]+nodes[28:]
    neighbors=observed_neighbors(nodes,train)
    view=build_view(nodes,neighbors,train,valid,valid_entities)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026091610)
        features=torch.randn(len(nodes),4,dtype=dtype)*.4
        model=LorentzMeanNetwork(4,4,head_hidden=5).to(dtype=dtype).eval()
    return model,features,view


def jsonable(value):
    if isinstance(value,torch.Tensor):return jsonable(value.detach().cpu().tolist())
    if isinstance(value,dict):return {key:jsonable(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):return [jsonable(item) for item in value]
    if isinstance(value,float) and math.isnan(value):return None  # Explicit undefined direction masks.
    if isinstance(value,float) and not math.isfinite(value):raise ValueError('infinite report value')
    return value


@torch.no_grad()
def run_fixture(repetitions=4,max_seconds=30.):
    if type(repetitions) is not int or not 2<=repetitions<=16:raise ValueError('fixture repetitions must be 2..16')
    if not math.isfinite(max_seconds) or not 0<max_seconds<=60:raise ValueError('fixture wall budget must be (0,60]')
    started=time.perf_counter();torch.set_num_threads(2)
    model,features,view=synthetic_fixture()
    panels=make_panels(view);groups=support_groups(view.neighbors,16)
    forward=FrozenForward(model,features,view.neighbors,max_padded_messages=128)
    numerical=forward.numerical_audit();root=view.nodes.index(view.root)
    streams={};structure_streams={};task_streams={};projections={};diagnostic_streams={}
    structure_columns=['score','full_score','tie','correct_to_error','error_to_correct','score_change']
    bases=[row['base'] for row in numerical['layers']]
    rng=PlanStreams(2026091611,'synthetic-development-only')
    index={node:i for i,node in enumerate(view.nodes)}
    valid=[(index[a],index[b]) for a,b in sorted(view.valid_edges)]
    parents=defaultdict(set)
    for a,b in valid:parents[b].add(a)
    full_task=filtered_parent_ranks(model,forward.reference['output'],valid,parents,candidate_chunk=11)
    plan_hashes=[];invalid=[];completed=0;structure_last={}
    for repetition in range(repetitions):
        if time.perf_counter()-started>=max_seconds:break
        plans=rng.draw(view.neighbors,16);plan_hashes.append([digest(plan) for plan in plans])
        name='paired_forward'
        try:
            conditions=forward.paired(plans)
            for name,row in conditions.items():
                layer=row['layer']-1;base=bases[layer]
                floor=numerical['layers'][layer]['empirical_numerical_floor']
                points,promotion=promote_points(row['points'],model.c)
                if name not in streams:
                    directions,defined=outward_directions(base,base[root],model.c,floor+floor[root])
                    streams[name]=TangentStream(base,model.c,groups,directions,defined)
                    projections[name]={'max_constraint_before':0.,'max_projection_displacement':0.,'max_constraint_after':0.}
                streams[name].add(points)
                diagnostic=row['diagnostics']
                diagnostic_streams.setdefault(name,ScalarStream()).add(torch.stack([
                    diagnostic['N'].double().sum(),diagnostic['k'].double().sum(),
                    diagnostic['fallback'].double().mean(),diagnostic['clipped'].double().mean(),
                    diagnostic['empty'].double().mean(),diagnostic['full'].double().mean()]))
                projections[name]['max_constraint_before']=max(projections[name]['max_constraint_before'],float(promotion['constraint_before'].max()))
                projections[name]['max_projection_displacement']=max(projections[name]['max_projection_displacement'],float(promotion['ambient_projection_displacement'].max()))
                projections[name]['max_constraint_after']=max(projections[name]['max_constraint_after'],float(promotion['constraint_after'].max()))
                structure=radial_relations(view,panels['panels']['development'],points,base,groups,model.c,floor)
                structure_last[name]=structure
                for kind,result in structure['metrics'].items():
                    for group,summary in result['groups'].items():
                        metrics=summary['weighted_covered_child_metrics']
                        if metrics['score_change'] is not None:
                            key=f'{name}/{kind}/{group}'
                            structure_streams.setdefault(key,ScalarStream()).add(torch.tensor([metrics[key] for key in structure_columns],dtype=torch.float64))
                if row['layer']==2:
                    rank=filtered_parent_ranks(model,row['points'],valid,parents,candidate_chunk=11,
                                               max_seconds=max(0.,max_seconds-(time.perf_counter()-started)))
                    if rank['status']!='complete':raise TimeoutError('incomplete full fixture valid ranking')
                    task_streams.setdefault(name,ScalarStream()).add(torch.tensor(
                        [rank['query_micro_mrr'],rank['query_micro_mrr']-full_task['query_micro_mrr']],dtype=torch.float64))
            completed+=1
        except (ValueError,TimeoutError) as error:
            invalid.append({'repetition':repetition,'condition':name if 'name' in locals() else None,
                            'error':str(error),'policy':'no redraw or clipping; incomplete streams retained with their own counts'})
            break
    stats={}
    for name,stream in streams.items():
        layer=0 if name in ('F/F_L1','local_L1') else 1
        stats[name]=stream.finish(numerical['layers'][layer]['empirical_numerical_floor']) if stream.n>=2 else {'completed_observations':stream.n,'status':'insufficient_repetitions'}
    return {'status':'complete' if completed==repetitions else 'incomplete',
            'scope':'synthetic untrained CPU fixture; no real checkpoint, no collapse or competitive task claim',
            'checkpoint_selection':'none; synthetic random initialization, not a selected checkpoint',
            'prepared_source':'synthetic_fixture function; hand-built train/valid child groups, not real protocol-B data',
            'initialization_seed':2026091610,'torch':str(torch.__version__),'python':platform.python_version(),
            'feature_sha256':hashlib.sha256(features.numpy().tobytes()).hexdigest(),
            'model_tensor_sha256':{key:hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() for key,value in model.state_dict().items()},
            'requested_repetitions':repetitions,'completed_graph_repetitions':completed,'elapsed_seconds':time.perf_counter()-started,
            'budget_check':'before each graph repetition and ranking child; reference/audit construction is one indivisible fixture phase',
            'fanout':16,'reference':'exact full neighborhoods','view':view.metadata,'panels':panels,'support_groups':groups,
            'seed_namespace':rng.metadata,'paired_plan_hashes':plan_hashes,'numerical':numerical,
            'geometry':stats,'condition_promotion_audit':projections,'structure_last_repeat':structure_last,
            'structure_metric_statistics':{key:stream.summary() for key,stream in structure_streams.items()},
            'structure_stat_columns':structure_columns,
            'aggregation_diagnostic_statistics':{key:stream.summary() for key,stream in diagnostic_streams.items()},
            'aggregation_stat_columns':['N_sum','k_sum','fallback_fraction','clipped_fraction','empty_fraction','full_fraction'],
            'full_task_reference':full_task,'full_task_repetition_statistics':{key:stream.summary() for key,stream in task_streams.items()},
            'task_stat_columns':['complete_valid_query_micro_mrr','paired_change_from_full'],
            'task_scope':'all fixture valid queries and all fixture entity candidates; layer 2 only',
            'invalid_conditions':invalid,'real_R0_R_Rtask_authorized':False,'confirmation_registration':None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit')
    parser.add_argument('--repetitions',type=int,default=4)
    parser.add_argument('--max-seconds',type=float,default=30.)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('fixture output must be a new path')
    identity=source_identity(args.source_commit)
    result=run_fixture(args.repetitions,args.max_seconds)
    result['source']=identity;result['source_sha256_normalized_lf']=dependency_hashes()
    result['source_sha256_normalized_lf']['acl_hct/frozen_fixture.py']=hashlib.sha256(Path(__file__).read_text(encoding='utf-8').encode()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(jsonable(result),stream,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'scope':result['scope'],'output':str(args.output)}))


if __name__=='__main__':main()

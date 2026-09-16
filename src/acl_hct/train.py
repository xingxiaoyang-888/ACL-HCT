"""Bounded protocol-B development runner; one configuration, no job submission."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from collections import defaultdict
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time
import numpy as np
import torch
from .backbone import LorentzMeanNetwork, make_plan, validate_neighbors
from .protocols import digest, mask_indexed_queries
from .ranking import filtered_parent_ranks
from .mechanisms import source_identity, dependency_hashes


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 11
    hidden: int = 128
    head_hidden: int = 128
    c: float = 1.
    scaled_radius: float = 1.2
    fanouts: tuple = (16,16)
    method: str = 'none'
    batch_positives: int = 128
    max_steps: int = 10
    learning_rate: float = .003
    max_seconds: float = 1800.
    evaluation: str = 'validation_probe'
    probe_queries: int = 16
    evaluation_max_seconds: float = 300.
    candidate_chunk: int = 4096
    max_padded_messages: int = 32768
    save_every: int = 5
    evaluate_every: int = 0
    threads: int = 2

    def validate(self):
        for name,lower,upper in [('hidden',1,512),('head_hidden',1,512),('batch_positives',1,1024),
                                  ('max_steps',1,2000),('probe_queries',1,2000),('candidate_chunk',1,8192),
                                  ('max_padded_messages',1,262144),('save_every',1,100),('threads',1,8)]:
            value=getattr(self,name)
            if type(value) is not int or not lower<=value<=upper: raise ValueError(f'invalid bounded {name}')
        if self.evaluation not in ('validation_probe','full_validation'): raise ValueError('invalid evaluation scope')
        if self.method not in ('none','third','jackknife'): raise ValueError('invalid method')
        if len(self.fanouts)!=2 or any(f is not None and (type(f) is not int or f<1) for f in self.fanouts):
            raise ValueError('two positive fanouts or null required')
        if not 0<self.max_seconds<=3600 or not 0<self.evaluation_max_seconds<=self.max_seconds:
            raise ValueError('wall budget must be positive and <=1h; eval budget <= total budget')
        if not math.isfinite(self.learning_rate) or self.learning_rate<=0: raise ValueError('invalid learning rate')
        if type(self.evaluate_every) is not int or self.evaluate_every<0: raise ValueError('invalid evaluation interval')


def load_prepared(root):
    """Intentionally never opens evaluator_test.json or evaluator_truth.json."""
    root=Path(root)
    manifest=json.loads((root/'input_manifest.json').read_text(encoding='utf-8'))
    graph=json.loads((root/'observed_graph.json').read_text(encoding='utf-8'))
    queries=json.loads((root/'train_queries.json').read_text(encoding='utf-8'))
    valid=json.loads((root/'evaluator_valid.json').read_text(encoding='utf-8'))
    with np.load(root/'features.npz',allow_pickle=False) as archive: features=archive['features']
    nodes=graph['nodes']; index={node:i for i,node in enumerate(nodes)}
    if manifest['protocol']!='B-child-grouped-80-10-10-v1' or manifest['split_seed']!=20260914:
        raise ValueError('runner requires frozen protocol B V1 split seed')
    if (digest(nodes)!=manifest['node_order_hash'] or digest(graph['neighbors'])!=manifest['graph_hash']
            or digest(queries)!=manifest['train_queries_hash']): raise ValueError('prepared input hash mismatch')
    if (hashlib.sha256(features.tobytes()).hexdigest()!=manifest['feature_manifest']['feature_sha256']
            or features.dtype!=np.float32 or features.shape!=(len(nodes),manifest['feature_manifest']['effective_dimension'])
            or not np.isfinite(features).all()): raise ValueError('feature contract/hash mismatch')
    validate_neighbors(graph['neighbors'],len(nodes))
    neighbor_sets=[set(row) for row in graph['neighbors']]
    if any(i not in neighbor_sets[j] for i,row in enumerate(graph['neighbors']) for j in row):
        raise ValueError('observed message graph must contain explicit reverse directions')
    group_size=queries['negatives_per_positive']+1
    if group_size!=5 or len(queries['labels'])%group_size or not queries['labels']:
        raise ValueError('runner expects nonempty groups of one positive and four negatives')
    if len(queries['queries'])!=len(queries['labels']): raise ValueError('query/label mismatch')
    ids=torch.tensor([[index[a],index[b]] for a,b in queries['queries']],dtype=torch.long).reshape(-1,group_size,2)
    labels=torch.tensor(queries['labels'],dtype=torch.float32).reshape(-1,group_size)
    if not torch.equal(labels,torch.tensor([1.,0.,0.,0.,0.]).expand_as(labels)):
        raise ValueError('query groups must start with exactly one positive')
    fit=set(manifest['text_fit_entities'])
    train_parents=defaultdict(set)
    for a,b in ids[:,0].tolist():
        if nodes[b] not in fit or b not in neighbor_sets[a] or a not in neighbor_sets[b]:
            raise ValueError('train query entity or visible edge mismatch')
        train_parents[b].add(a)
    for group in ids.tolist():
        child=group[0][1];negatives=[a for a,b in group[1:]]
        if any(b!=child for a,b in group) or len(set(negatives))!=4 or any(a==child or a in train_parents[child] for a in negatives):
            raise ValueError('invalid fixed training negatives')
    valid_ids=[(index[a],index[b]) for a,b in valid]
    if not valid_ids: raise ValueError('complete validation query set must be nonempty')
    for a,b in valid_ids:
        if nodes[b] in fit or b in neighbor_sets[a] or a in neighbor_sets[b]:
            raise ValueError('held-out validation query leaks into observed graph/text-fit entities')
    return {'manifest':manifest,'manifest_hash':digest(manifest),'nodes':nodes,'features':torch.from_numpy(features),
            'neighbors':graph['neighbors'],'query_groups':ids,'labels':labels,'valid':valid_ids,'valid_hash':digest(valid)}


def synchronize(device):
    if device.type=='cuda': torch.cuda.synchronize()


def atomic_json(path, value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');temporary.replace(path)


def run(prepared_root, output_root, config=TrainConfig(), device='cpu', resume_checkpoint=None, identity=None):
    config.validate(); device=torch.device(device)
    if device.type not in ('cpu','cuda'): raise ValueError('CPU/CUDA only')
    if device.type=='cuda':
        if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise ValueError('CUDA runner requires an explicit Slurm allocation and preserved GPU binding')
        if torch.cuda.device_count()!=1: raise ValueError('initial runner requires exactly one visible allocated GPU')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
    output=Path(output_root)
    if output.exists() and any(output.iterdir()): raise ValueError('run output must be empty; resume into a new directory')
    output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();torch.set_num_threads(config.threads);torch.manual_seed(config.seed)
    data=load_prepared(prepared_root);features=data['features'].to(device)
    model=LorentzMeanNetwork(features.shape[1],config.hidden,config.c,config.scaled_radius,config.head_hidden).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    sample_rng=torch.Generator().manual_seed(config.seed);batch_rng=torch.Generator().manual_seed(config.seed+1)
    completed=0;best_score=None
    if resume_checkpoint:
        checkpoint=torch.load(resume_checkpoint,map_location='cpu')
        if checkpoint.get('run_status')=='failed': raise ValueError('failed checkpoints require explicit recovery review')
        for name,expected in checkpoint['source_sha256_normalized_lf'].items():
            path=Path(__file__).parent/Path(name).name
            if not path.is_file() or hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()!=expected:
                raise ValueError('resume source code differs from checkpoint')
        old_config=checkpoint['config'];new_config=asdict(config)
        for key in new_config:
            if key not in ('max_steps','max_seconds') and old_config[key]!=new_config[key]:
                raise ValueError(f'resume changes scientific configuration: {key}')
        if checkpoint['manifest_hash']!=data['manifest_hash'] or checkpoint['valid_hash']!=data['valid_hash'] or checkpoint['device_type']!=device.type:
            raise ValueError('resume requires same prepared inputs and device type')
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        sample_rng.set_state(checkpoint['sampling_rng']);batch_rng.set_state(checkpoint['batch_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.type=='cuda': torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        completed=checkpoint['completed_steps'];best_score=checkpoint['best_full_valid_mrr']
        if completed>=config.max_steps: raise ValueError('resume has no remaining steps')
        if best_score is not None:
            previous_best=Path(resume_checkpoint).parent/'best.pt'
            if not previous_best.is_file(): raise ValueError('resume with selected best requires its sibling best.pt')
            best_checkpoint=torch.load(previous_best,map_location='cpu')
            if best_checkpoint['best_full_valid_mrr']!=best_score or best_checkpoint['manifest_hash']!=data['manifest_hash']:
                raise ValueError('previous best checkpoint identity mismatch')
            shutil.copyfile(previous_best,output/'best.pt')
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    source_hashes=dependency_hashes()
    source_hashes['acl_hct/train.py']=hashlib.sha256(Path(__file__).read_text(encoding='utf-8').encode()).hexdigest()
    report={'started_utc':datetime.now(timezone.utc).isoformat(),'scope':'bounded protocol-B development; validation probe never selects a checkpoint',
            'status':'running','config':asdict(config),'config_hash':digest(asdict(config)),
            'manifest_hash':data['manifest_hash'],'validation_queries_hash':data['valid_hash'],'protocol':data['manifest']['protocol'],'split_seed':20260914,
            'device':torch.cuda.get_device_name(0) if device.type=='cuda' else platform.processor(),
            'torch':torch.__version__,'python':platform.python_version(),'precision':'FP32; CUDA TF32 disabled; no mixed precision',
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'source':identity or source_identity(),'source_sha256_normalized_lf':source_hashes,
            'resumed_from_step':completed,'completed_steps':completed,'steps':[],'evaluations':[],'checkpoint_seconds':[],
            'selection_status':'inherited best from complete validation' if best_score is not None else 'none; no complete validation selected yet',
            'best_full_valid_mrr':best_score,'budget_check':'at step/child boundaries; one operation may exceed deadline'}
    # Probe identities fixed independently of training batches; still all parent candidates.
    probe_rng=torch.Generator().manual_seed(20260914)
    probe_ix=torch.randperm(len(data['valid']),generator=probe_rng)[:config.probe_queries].tolist()
    evaluation_queries=data['valid'] if config.evaluation=='full_validation' else [data['valid'][i] for i in probe_ix]
    report['evaluation_query_ids']=evaluation_queries;report['evaluation_query_hash']=digest(evaluation_queries)
    report['full_valid_query_count']=len(data['valid'])
    true_parents=defaultdict(set)
    for a,b in data['valid']:true_parents[b].add(a)

    def save(name):
        synchronize(device);begin=time.perf_counter()
        checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':asdict(config),
                    'manifest_hash':data['manifest_hash'],'valid_hash':data['valid_hash'],'device_type':device.type,'completed_steps':completed,'run_status':report['status'],
                    'sampling_rng':sample_rng.get_state(),'batch_rng':batch_rng.get_state(),
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                    'best_full_valid_mrr':best_score,'selection_status':report['selection_status'],
                    'source':report['source'],'source_sha256_normalized_lf':report['source_sha256_normalized_lf']}
        temporary=output/(name+'.tmp');torch.save(checkpoint,temporary);temporary.replace(output/name)
        report['checkpoint_seconds'].append(time.perf_counter()-begin)

    def evaluate():
        nonlocal best_score
        remaining=config.max_seconds-(time.perf_counter()-started)
        if remaining<=0:return
        model.eval();synchronize(device);begin=time.perf_counter()
        with torch.no_grad():
            points,_,_=model.encode(features,data['neighbors'],[data['neighbors']]*2,max_padded_messages=config.max_padded_messages)
        synchronize(device);encoding_seconds=time.perf_counter()-begin
        budget=min(config.evaluation_max_seconds,config.max_seconds-(time.perf_counter()-started))
        result=filtered_parent_ranks(model,points,evaluation_queries,true_parents,config.candidate_chunk,max(0.,budget))
        result.update({'step':completed,'purpose':config.evaluation,'full_graph_encoding_seconds':encoding_seconds})
        report['evaluations'].append(result)
        if config.evaluation=='full_validation' and result['status']=='complete':
            score=result['query_micro_mrr']
            if best_score is None or score>best_score:
                best_score=score;report['best_full_valid_mrr']=score
                report['selection_status']='selected by complete filtered all-candidate validation query-micro MRR'
                save('best.pt')
        model.train()

    save('last.pt')
    try:
        model.train()
        for step in range(completed+1,config.max_steps+1):
            if time.perf_counter()-started>=config.max_seconds:break
            synchronize(device);begin=time.perf_counter()
            chosen=torch.randperm(len(data['query_groups']),generator=batch_rng)[:config.batch_positives]
            groups=data['query_groups'][chosen];positive=groups[:,0].tolist()
            neighbors=mask_indexed_queries(data['neighbors'],positive)
            plans=[make_plan(neighbors,fanout,sample_rng) for fanout in config.fanouts]
            planning_seconds=time.perf_counter()-begin
            queries=groups.reshape(-1,2).to(device);labels=data['labels'][chosen].reshape(-1).to(device)
            optimizer.zero_grad();points,stats,trace=model.encode(features,neighbors,plans,config.method,
                                                              config.max_padded_messages,return_trace=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model.score(points,queries),labels)
            if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('missing/nonfinite parameter gradient')
            optimizer.step();completed=step;report['completed_steps']=completed
            layers=[]
            with torch.no_grad():
                for row,diagnostic in zip(trace,stats):
                    radii=torch.acosh((row['messages'][:,0]*math.sqrt(config.c)).clamp_min(1.))
                    summary=torch.stack((radii.mean(),radii.max(),(radii>=.95*config.scaled_radius).float().mean(),
                                         diagnostic['fallback'].float().mean(),diagnostic['clipped'].float().mean(),
                                         diagnostic['N'].sum().float(),diagnostic['k'].sum().float())).cpu().tolist()
                    layers.append(dict(zip(('mean_scaled_message_radius','max_scaled_message_radius','near_bound_fraction',
                                            'fallback_rate','clipping_rate','candidate_messages','selected_messages'),summary)))
            synchronize(device)
            report['steps'].append({'step':step,'loss':float(loss.detach()),'seconds':time.perf_counter()-begin,
                                    'sampling_and_mask_seconds':planning_seconds,'positive_queries':len(chosen),
                                    'negative_queries':4*len(chosen),'layers':layers})
            if config.evaluate_every and step%config.evaluate_every==0:evaluate()
            if step%config.save_every==0:
                save('last.pt');atomic_json(output/'run.json',report)
        if not report['evaluations'] or report['evaluations'][-1]['step']!=completed:evaluate()
        report['status']='step_limit_reached' if completed==config.max_steps else 'wall_limit_reached'
    except Exception as error:
        report['status']='failed';report['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        save('last.pt');report['completed_steps']=completed;report['elapsed_seconds']=time.perf_counter()-started
        report['peak_allocated_bytes']=torch.cuda.max_memory_allocated() if device.type=='cuda' else None
        report['peak_reserved_bytes']=torch.cuda.max_memory_reserved() if device.type=='cuda' else None
        atomic_json(output/'run.json',report)
    return report


def main():
    cli=argparse.ArgumentParser();cli.add_argument('--prepared',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True);cli.add_argument('--config',type=Path,required=True)
    cli.add_argument('--device',choices=['cpu','cuda'],default='cpu');cli.add_argument('--source-commit')
    cli.add_argument('--resume-checkpoint',type=Path);args=cli.parse_args()
    values=json.loads(args.config.read_text(encoding='utf-8'))
    if 'fanouts' in values:values['fanouts']=tuple(values['fanouts'])
    result=run(args.prepared,args.output,TrainConfig(**values),args.device,args.resume_checkpoint,source_identity(args.source_commit))
    print(json.dumps({'status':result['status'],'completed_steps':result['completed_steps'],'selection_status':result['selection_status']}))


if __name__=='__main__':main()

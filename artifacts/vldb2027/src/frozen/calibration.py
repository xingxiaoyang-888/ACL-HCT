"""Exactly one label-free seed11 cost calibration per model/dataset."""
import hashlib
import json
import statistics
import time
import numpy as np
from deployment import sampled_request
from study_io import require, save


def calibrate(job,plan,model,graph,planner,batches,output,guard):
    require(job['seed']==11 and len(batches)==8,'one seed11/eight-batch calibration required')
    points=[];dataset=plan['datasets'][job['dataset']]
    with (output/'requests.jsonl').open('x',encoding='utf-8') as stream:
        for sampler,grid in [('uniform',dataset['uniform_fanouts']),('labor',dataset['labor_grid'])]:
            for k in grid:
                guard();costs=[];started=time.perf_counter()
                for repeat,batch in [(-1,0)]+[(r,b) for r in range(3) for b in range(8)]:
                    _,cost=sampled_request(model,graph,planner,batches[batch],sampler=sampler,fanout=k,
                        seed=11,repeat=repeat,batch=batch,stage='calibration-warmup' if repeat==-1 else 'calibration',device='cuda:0')
                    cost.update(phase='warmup' if repeat==-1 else 'measurement',labels_read=False)
                    stream.write(json.dumps(cost,allow_nan=False)+'\n');stream.flush()
                    if repeat!=-1:costs.append(cost['e2e_seconds'])
                guard()
                points.append({'sampler':sampler,'fanout':k,'median_e2e_seconds':statistics.median(costs),
                    'measured_requests':24,'paid_warmups':1,'point_outer_wall_seconds':time.perf_counter()-started,
                    'outer_includes_guard_and_output_writes':True})
    tiers=[]
    for tier,k in zip(['low','middle','high'],dataset['uniform_fanouts']):
        target=next(p for p in points if p['sampler']=='uniform' and p['fanout']==k)
        eligible=[p for p in points if p['sampler']=='labor' and p['median_e2e_seconds']<=target['median_e2e_seconds']]
        selected=max(eligible,key=lambda p:p['fanout']) if eligible else next(p for p in points if p['sampler']=='labor')
        tiers.append({'tier':tier,'uniform_fanout':k,'labor_fanout':selected['fanout'],'cost_match_feasible':bool(eligible),
            'uniform_median_seconds':target['median_e2e_seconds'],'labor_median_seconds':selected['median_e2e_seconds'],
            'labor_over_uniform_time_ratio':selected['median_e2e_seconds']/target['median_e2e_seconds']})
    result={'schema':'vldb_sampling_budget_lock_v1','round':job['round'],'dataset':job['dataset'],'model':job['model'],
        'calibration_seed':11,'weights_sha256':job['checkpoint']['weights']['sha256'],'plan_sha256':job['plan']['sha256'],
        'source_manifest_sha256':job['source']['manifest']['sha256'],
        'target_ids_sha256':hashlib.sha256(np.asarray(np.concatenate(batches),dtype='<i8').tobytes()).hexdigest(),
        'tiers':tiers,'points':points,'labels_read':False,'outcome_guided_selection':False,
        'timing_scope':'independent_target_dependency_cone_requests; not equal diagnostic cost',
        'contention_certification':'discrete fixed-group checks only; actual host conditions in Ops receipt'}
    save(output/'budget-lock.json',result)
    return result

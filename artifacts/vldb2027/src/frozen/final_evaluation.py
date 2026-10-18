"""Once-only frozen 15-cell campaign member: three-tier quality/cost + mechanisms.

No fitting, checkpoint selection, evaluation-driven repair or new budget search.
Native margins/predictions and per-node fix/break events retain negative effects.
"""
import gc
import json
import statistics
import time
import numpy as np
import torch
from deployment import sampled_request, prediction_arrays, four_way_arrays, timed_full, timed_ram, memory_start, memory_end, synchronize
from exact_full import ExactFullGraph
from mechanism import four_way
from study_io import require, save


def _npz(path,arrays):
    with path.open('xb') as stream:np.savez(stream,**arrays)


def _write(stream,row):
    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()


def _metrics(arrays):
    return {'correct':int(arrays['correct'].sum()),'total':len(arrays['correct']),
            'accuracy':float(arrays['correct'].mean()),'mean_true_class_margin':float(arrays['true_class_margin'].mean())}


def evaluate(job,plan,model,graph,planner,ids,batches,labels,lock,output,guard):
    summaries=[];full_cost=[];ram_cost=[];mechanism_rows=[]
    lookup={int(v):i for i,v in enumerate(ids)}
    # Whole-graph full is compared with the COMPLETE SAME target workload.
    # Its GPU formats/caches are released before any sampled request is timed.
    guard();synchronize('cuda:0');full_setup_cost=memory_start('cuda:0');started=time.perf_counter()
    kwargs={'gat_chunk_rows':plan['chunks']['max_rows'],'gat_chunk_edges':plan['chunks']['max_edges']} if job['model']=='gat' else {}
    full=ExactFullGraph(model,graph,'cuda:0',**kwargs)
    synchronize('cuda:0');full_setup=time.perf_counter()-started
    memory_end('cuda:0',full_setup_cost)
    full_setup_cost.update(e2e_seconds=full_setup,scope='full_format_construction',
        reserved_memory_scope='process_allocator_history_observation_not_cold_isolated_minimum')
    full_formats=full.report();guard()
    cache=None
    for repeat in range(3):
        guard();cache,requested,cost=timed_full(full,ids,'cuda:0');guard()
        cost['repeat']=repeat;full_cost.append(cost)
        if repeat==0:
            arrays=prediction_arrays(requested,labels);arrays['ids']=ids;arrays['y']=labels
            _npz(output/'full-predictions.npz',arrays);full_summary=_metrics(arrays)
        del requested
    for repeat in range(3):
        guard();gathers,cost=timed_ram(cache,batches);guard()
        cost['repeat']=repeat;ram_cost.append(cost);del gathers
    cache_bytes=cache.numel()*cache.element_size()
    del cache,full;gc.collect();torch.cuda.empty_cache();guard()
    points=[(sampler,tier['tier'],tier[sampler+'_fanout']) for sampler in ['uniform','labor'] for tier in lock['tiers']]
    with (output/'deployment-costs.jsonl').open('x',encoding='utf-8') as stream:
        guard()
        for sampler,tier,k in points:
            _,cost=sampled_request(model,graph,planner,batches[0],sampler=sampler,fanout=k,seed=job['seed'],
                repeat=-1,batch=0,stage='final-warmup',device='cuda:0')
            _write(stream,dict(cost,tier=tier,phase='paid_warmup'))
        guard()
        for repeat in range(4):
            outputs={(s,t):{'prediction':[],'correct':[],'true_class_margin':[]} for s,t,k in points}
            costs={(s,t):[] for s,t,k in points}
            for start in range(0,len(batches),8):
                guard()
                for batch in range(start,min(start+8,len(batches))):
                    targets=batches[batch];indices=np.asarray([lookup[int(v)] for v in targets],dtype=np.int64)
                    rotation=(repeat+batch)%len(points);ordered=points[rotation:]+points[:rotation]
                    for sampler,tier,k in ordered:
                        scores,cost=sampled_request(model,graph,planner,targets,sampler=sampler,fanout=k,
                            seed=job['seed'],repeat=repeat,batch=batch,stage='final-test',device='cuda:0')
                        arrays=prediction_arrays(scores,labels[indices])
                        for name,value in arrays.items():outputs[(sampler,tier)][name].append(value)
                        costs[(sampler,tier)].append(cost)
                        _write(stream,dict(cost,tier=tier,phase='measurement',guard_group_batch_start=start,
                                           guard_interior_continuously_certified=False))
                guard()
            for sampler,tier,k in points:
                arrays={name:np.concatenate(values) for name,values in outputs[(sampler,tier)].items()}
                arrays.update(ids=ids,y=labels)
                filename=f'deployment-{sampler}-{tier}-repeat{repeat}.npz';_npz(output/filename,arrays)
                observations=costs[(sampler,tier)]
                summaries.append(dict(_metrics(arrays),sampler=sampler,tier=tier,fanout=k,repeat=repeat,raw=filename,
                    total_workload_e2e_seconds=sum(c['e2e_seconds'] for c in observations),
                    median_request_e2e_seconds=statistics.median(c['e2e_seconds'] for c in observations),
                    peak_allocated_bytes=max(c['peak_allocated_bytes'] for c in observations),
                    peak_reserved_bytes=max(c['peak_reserved_bytes'] for c in observations),
                    scope='deployment_dependency_cone_quality'))
    # The following paid global diagnostic never supplies deployment states.
    for sampler in ['uniform','labor']:
        k=next(t[sampler+'_fanout'] for t in lock['tiers'] if t['tier']=='middle')
        for repeat in range(4):
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0)
            started=time.perf_counter()
            result=four_way(model,graph,planner,sampler=sampler,fanout=k,seed=job['seed'],repeat=repeat,
                           device='cuda:0',max_rows=plan['chunks']['max_rows'],max_edges=plan['chunks']['max_edges'],
                           guard_callback=guard)
            arrays=four_way_arrays(result,ids,labels)
            filename=f'mechanism-{sampler}-repeat{repeat}.npz';_npz(output/filename,arrays)
            branch_metrics={branch:_metrics({'correct':arrays[branch+'__correct'],
                'true_class_margin':arrays[branch+'__true_class_margin']}) for branch in result['scores']}
            q={key:value['accuracy'] for key,value in branch_metrics.items()}
            row={k:v for k,v in result.items() if k not in ('scores','local_estimator_sq_l2_sampled_inputs',
                'local_estimator_sq_l2_full_inputs','upstream_input_sq_l2')}
            row.update(raw=filename,metrics=branch_metrics,
                accuracy_interaction=q['both_restored']-q['local_restored']-q['upstream_restored']+q['sampled'],
                fixes_and_breaks={branch:{'fixed':int(arrays[branch+'__fixed_error'].sum()),
                    'introduced':int(arrays[branch+'__introduced_error'].sum())} for branch in ['local_restored','upstream_restored','both_restored']},
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                outer_wall_including_filter_score_write_seconds=time.perf_counter()-started,
                oracle_states_are_deployable=False,baseline_not_identical_to_dependency_cone=True)
            mechanism_rows.append(row);save(output/f'mechanism-{sampler}-repeat{repeat}.json',row)
            del arrays,result;gc.collect();guard()
    full_median=statistics.median(row['e2e_seconds'] for row in full_cost)
    build_median=statistics.median(row['cache_build_seconds'] for row in full_cost)
    gather_median=statistics.median(row['target_gather_seconds'] for row in full_cost)
    ram_median=statistics.median(row['e2e_seconds'] for row in ram_cost)
    # Setup is reported separately. q counts complete repetitions of the SAME
    # target workload; cache construction pays the whole-graph forward and D2H
    # once, excluding its measured one-workload target gather from the cache base.
    amortization=[{'q':q,'full_recompute_seconds':full_setup+q*full_median,
                  'paid_full_RAM_seconds':full_setup+build_median+q*ram_median,
                  'RAM_cache_bytes':cache_bytes,'common_graph_CP_setup_excluded_and_separately_reported':True}
                  for q in [1,10,100]]
    result={'schema':'vldb_frozen_final_results_v1','dataset':job['dataset'],'model':job['model'],'seed':job['seed'],
        'checkpoint_sha256':job['checkpoint']['weights']['sha256'],'budget_lock_sha256':job['budget_lock']['sha256'],
        'deployment':summaries,'mechanisms':mechanism_rows,'full':{'quality':full_summary,'format_setup_seconds':full_setup,
            'format_setup_cost':full_setup_cost,'formats':full_formats,
            'peak_allocated_bytes':max([full_setup_cost['peak_allocated_bytes']]+[c['peak_allocated_bytes'] for c in full_cost]),
            'peak_reserved_bytes':max([full_setup_cost['peak_reserved_bytes']]+[c['peak_reserved_bytes'] for c in full_cost]),
            'memory_scope':'maximum_across_format_setup_and_full_forwards_including_resident_model',
            'reserved_memory_scope':'process_allocator_history_observation_not_cold_isolated_minimum',
            'full_e2e_median_seconds':full_median,'cache_build_median_seconds':build_median,
            'target_gather_median_seconds':gather_median,'RAM_workload_median_seconds':ram_median,
            'cost_repeats':full_cost,'RAM_cost_repeats':ram_cost,'amortization':amortization},
        'training_seed_count_in_this_job':1,'diagnostic_interaction_not_additive_contribution_fraction':True,
        'negative_effects_retained':True,'test_once_campaign':True,'new_selection_or_fitting':False}
    save(output/'results.json',result)
    return result

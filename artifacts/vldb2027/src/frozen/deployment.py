"""Label-free target-request measurement and frozen prediction summaries.

The caller owns exact job/input/CP permissions and real resource guards. Each
timer executes inside an externally guarded fixed group. No diagnostic oracle
states, plans, outputs or full caches enter sampled deployment requests.
"""
import hashlib
import time
import numpy as np
import torch
from model_builders import native_scores


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def memory_start(device):
    if torch.device(device).type != 'cuda':
        return {'baseline_allocated_bytes': None, 'baseline_reserved_bytes': None}
    torch.cuda.reset_peak_memory_stats(device)
    return {'baseline_allocated_bytes': torch.cuda.memory_allocated(device),
            'baseline_reserved_bytes': torch.cuda.memory_reserved(device)}


def memory_end(device, record):
    record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type=='cuda' else None,
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if torch.device(device).type=='cuda' else None)


@torch.no_grad()
def sampled_request(model, graph, planner, targets, *, sampler, fanout, seed,
                    repeat, batch, stage, device):
    """Paid CPU sampling, indexing, transfer, native forward and CPU return."""
    record=memory_start(device)
    synchronize(device);started=time.perf_counter()
    with torch.inference_mode(False):
        blocks=planner.plan(graph,targets,[fanout]*3,sampler=sampler,stage=stage,
                            seed=seed,repeat=repeat,batch=batch)
    planned=time.perf_counter()
    x=graph.x.index_select(0,blocks[0].src_ids).to(device)
    moved=[block.to(device) for block in blocks]
    synchronize(device);transferred=time.perf_counter()
    scores=native_scores(model,x,moved,correction_enabled=False)
    synchronize(device);forwarded=time.perf_counter()
    scores=scores.detach().cpu()
    synchronize(device);finished=time.perf_counter()
    if scores.shape != (len(targets),model._study_classes) or not bool(torch.isfinite(scores).all()):
        raise ValueError('invalid native deployment scores; no retry')
    record.update(e2e_seconds=finished-started,cpu_planning_seconds=planned-started,
        indexing_h2d_seconds=transferred-planned,synchronized_forward_seconds=forwarded-transferred,
        output_d2h_seconds=finished-forwarded,sampler=sampler,fanout=fanout,repeat=repeat,batch=batch,
        target_count=len(targets),target_ids_sha256=hashlib.sha256(np.asarray(targets,dtype='<i8').tobytes()).hexdigest(),
        layers=[block.report() for block in blocks],scope='independent_target_dependency_cone_request',
        model_output_unbiasedness_claimed=False)
    memory_end(device,record)
    return scores,record


@torch.no_grad()
def timed_full(full, targets, device):
    """Compute all nodes then return exactly requested targets; paid whole graph."""
    record=memory_start(device);synchronize(device);started=time.perf_counter()
    output=full.forward().detach().cpu()
    synchronize(device);cache_built=time.perf_counter()
    requested=output[torch.as_tensor(targets,dtype=torch.long)].clone()
    synchronize(device);finished=time.perf_counter()
    if not bool(torch.isfinite(output).all()):raise ValueError('nonfinite full output')
    record.update(e2e_seconds=finished-started,cache_build_seconds=cache_built-started,
                  target_gather_seconds=finished-cache_built,
                  scope='whole_graph_recomputation_for_identical_requested_targets',
                  cache_build_includes_full_forward_and_output_d2h=True,
                  all_nodes_computed=len(output),target_count=len(targets))
    memory_end(device,record)
    return output,requested,record


def timed_ram(cache, batches):
    """Paid CPU RAM gather/copy for the same target workload; cache built elsewhere."""
    started=time.perf_counter()
    result=[cache[torch.as_tensor(targets,dtype=torch.long)].clone() for targets in batches]
    elapsed=time.perf_counter()-started
    return result,{'e2e_seconds':elapsed,'target_count':sum(map(len,batches)),
        'scope':'RAM_output_cache_same_target_workload','cache_bytes':cache.numel()*cache.element_size(),
        'precomputation_included':False,'cache_valid_only_for_fixed_graph_and_weights':True}


def prediction_arrays(scores, labels):
    """Unreduced per-node native-label margin; no fitting or threshold choice."""
    labels=torch.as_tensor(labels,dtype=torch.long)
    rows=torch.arange(len(labels));correct_score=scores[rows,labels]
    competitors=scores.clone();competitors[rows,labels]=-torch.inf
    pred=scores.argmax(-1)
    return {'prediction':pred.numpy(),'correct':pred.eq(labels).numpy(),
            'true_class_margin':(correct_score-competitors.max(-1).values).numpy()}


def four_way_arrays(result, ids, labels):
    """Persist target rows only. Four-way deltas remain paired, never additive."""
    take=torch.as_tensor(ids,dtype=torch.long)
    arrays={'ids':np.asarray(ids,dtype=np.int64),'y':np.asarray(labels,dtype=np.int64)}
    reference=result['scores']['both_restored'][take]
    for branch,scores in result['scores'].items():
        local=scores[take]
        for name,value in prediction_arrays(local,labels).items():arrays[branch+'__'+name]=value
        arrays[branch+'__score_sq_l2_to_both']=((local-reference)**2).sum(-1).numpy()
    for name in ('local_estimator_sq_l2_sampled_inputs','local_estimator_sq_l2_full_inputs','upstream_input_sq_l2'):
        arrays[name]=result[name][take].numpy()
    for branch in ('local_restored','upstream_restored','both_restored'):
        before=arrays['sampled__correct'];after=arrays[branch+'__correct']
        arrays[branch+'__fixed_error']=(~before)&after
        arrays[branch+'__introduced_error']=before&(~after)
    return arrays

"""Evaluation-only cached exact relation scores and filtered all-parent ranks."""
from collections import defaultdict
import time
import torch
from .geometry import log, origin_like


def _versions(model):
    return tuple((id(p),p._version) for p in model.relation_head.parameters())


@torch.no_grad()
def prepare_scores(model, points):
    """Cache belongs to one eval/weight state; optimizer/load changes invalidate it."""
    if model.training: raise ValueError('score cache requires model.eval()')
    tangent=log(origin_like(points,model.c),points,model.c)[...,1:]
    first,last=model.relation_head[0],model.relation_head[2]
    wp,wc,wd=first.weight.split(tangent.shape[-1],dim=1)
    return {'parent':tangent@(wp+wd).T,'child':tangent@(wc-wd).T,
            'bias':first.bias,'last':last,'versions':_versions(model),'owner':id(model)}


@torch.no_grad()
def score_cached(model, cache, parents, children):
    if model.training or cache['owner']!=id(model) or cache['versions']!=_versions(model):
        raise ValueError('stale score cache or wrong model/evaluation state')
    hidden=cache['parent'][parents]+cache['child'][children]+cache['bias']
    return cache['last'](hidden.relu()).squeeze(-1)


@torch.no_grad()
def filtered_parent_ranks(model, points, queries, true_parents, candidate_chunk=4096, max_seconds=None):
    """All entity candidates minus self/other true parents; average exact-score ties.

    All scores for ONE child are stored (O(N)); pair activations are chunked.
    Multiple true-parent queries for the same child share this scoring pass.
    This evaluator receives labels explicitly and must not be used for training.
    Incomplete timing-limited output is never a complete-validation metric.
    """
    if type(candidate_chunk) is not int or candidate_chunk<1: raise ValueError('invalid candidate chunk')
    if not queries or len(set(map(tuple,queries)))!=len(queries): raise ValueError('nonempty unique queries required')
    n=len(points); grouped=defaultdict(list)
    for a,b in queries:
        if type(a) is not int or type(b) is not int or not 0<=a<n or not 0<=b<n or a==b:
            raise ValueError('invalid evaluation query')
        if a not in true_parents.get(b,()): raise ValueError('target missing from evaluator truth')
        grouped[b].append(a)
    for child,parents in true_parents.items():
        if any(type(p) is not int or not 0<=p<n or p==child for p in parents): raise ValueError('invalid evaluator parent set')
    started=time.perf_counter(); cache=prepare_scores(model,points); rows=[]; child_rr=[]
    for child,targets in grouped.items():
        if max_seconds is not None and time.perf_counter()-started>=max_seconds: break
        blocks=[]
        for start in range(0,n,candidate_chunk):
            parents=torch.arange(start,min(n,start+candidate_chunk),device=points.device)
            blocks.append(score_cached(model,cache,parents,child))
        scores=torch.cat(blocks)
        if not torch.isfinite(scores).all(): raise ValueError('nonfinite evaluation scores')
        eligible=torch.ones(n,dtype=torch.bool,device=points.device)
        eligible[child]=False
        eligible[list(true_parents[child])]=False
        negative_scores=scores[eligible].sort().values
        target_scores=scores[targets].contiguous()
        left=torch.searchsorted(negative_scores,target_scores,right=False)
        right=torch.searchsorted(negative_scores,target_scores,right=True)
        ranks=1+(len(negative_scores)-right).to(torch.float64)+.5*(right-left).to(torch.float64)
        values=ranks.cpu().tolist(); child_rr.append(sum(1/r for r in values)/len(values))
        rows.extend({'parent':parent,'child':child,'rank':rank,'candidates':len(negative_scores)+1}
                    for parent,rank in zip(targets,values))
    count=len(rows); complete=count==len(queries)
    return {'status':'complete' if complete else 'incomplete_time_limit','metric_scope':'filtered_all_entity_candidates',
            'expected_queries':len(queries),'completed_queries':count,'completed_children':len(child_rr),
            'query_micro_mrr':sum(1/row['rank'] for row in rows)/count if count else None,
            'child_macro_mrr':sum(child_rr)/len(child_rr) if child_rr else None,
            'hits':{str(k):sum(row['rank']<=k for row in rows)/count if count else None for k in (1,3,10)},
            'rows':rows,'elapsed_seconds':time.perf_counter()-started,
            'tie_policy':'average rank of exactly equal computed scores; target excluded from competitors'}

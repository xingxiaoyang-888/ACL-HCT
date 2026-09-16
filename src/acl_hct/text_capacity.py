"""Euclidean text-only MLP and filtered parent retrieval; no graph encoding."""
from collections import defaultdict
import time

import torch
from torch import nn


class TextMLP(nn.Module):
    def __init__(self, input_dim=128, hidden=128, head_hidden=128):
        super().__init__()
        if any(type(d) is not int or d < 1 for d in (input_dim, hidden, head_hidden)):
            raise ValueError('positive integer model dimensions required')
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU())
        self.relation_head = nn.Sequential(nn.Linear(3*hidden, head_hidden), nn.ReLU(),
                                          nn.Linear(head_hidden, 1))

    def encode(self, features):
        if features.ndim != 2 or features.shape[1] != self.encoder[0].in_features:
            raise ValueError('feature matrix dimension mismatch')
        return self.encoder(features)

    def score(self, embeddings, queries):
        if (queries.ndim != 2 or queries.shape[1] != 2 or queries.dtype != torch.long
                or queries.device != embeddings.device or ((queries < 0) | (queries >= len(embeddings))).any()):
            raise ValueError('valid device int64 [Q,2] queries required')
        parent, child = embeddings[queries[:, 0]], embeddings[queries[:, 1]]
        return self.relation_head(torch.cat((parent, child, parent-child), -1)).squeeze(-1)

    def forward(self, features, queries):
        # No cross-node operations, dropout or normalization: unused entities need not be encoded.
        if (queries.ndim != 2 or queries.shape[1] != 2 or queries.dtype != torch.long
                or queries.device != features.device or ((queries < 0) | (queries >= len(features))).any()):
            raise ValueError('valid device int64 [Q,2] queries required')
        ids, inverse = torch.unique(queries.reshape(-1), sorted=True, return_inverse=True)
        return self.score(self.encode(features[ids]), inverse.reshape(-1, 2))


def versions(model):
    return tuple((id(p), p._version) for p in model.parameters())


@torch.no_grad()
def prepare_scores(model, embeddings):
    if model.training:
        raise ValueError('score cache requires model.eval()')
    first, last = model.relation_head[0], model.relation_head[2]
    wp, wc, wd = first.weight.split(embeddings.shape[1], dim=1)
    return {'parent': embeddings@(wp+wd).T, 'child': embeddings@(wc-wd).T,
            'bias': first.bias, 'last': last, 'owner': id(model), 'versions': versions(model)}


@torch.no_grad()
def score_cached(model, cache, parents, children):
    if model.training or cache['owner'] != id(model) or cache['versions'] != versions(model):
        raise ValueError('stale score cache or wrong model/evaluation state')
    hidden = cache['parent'][parents] + cache['child'][children] + cache['bias']
    return cache['last'](hidden.relu()).squeeze(-1)


@torch.no_grad()
def filtered_parent_ranks(model, embeddings, queries, true_parents, candidate_chunk=4096, max_seconds=None):
    """Match original all-entity filtering and exact computed-score average ties.

    The original ranking cache maps Lorentz points to tangent coordinates. This
    cache directly uses Euclidean embeddings; the candidate and rank definitions
    remain the same. O(N) scores are retained per child, with bounded activations.
    """
    if type(candidate_chunk) is not int or candidate_chunk < 1:
        raise ValueError('positive candidate chunk required')
    if not queries or len(set(map(tuple, queries))) != len(queries):
        raise ValueError('nonempty unique queries required')
    n = len(embeddings); grouped = defaultdict(list)
    for parent, child in queries:
        if (type(parent) is not int or type(child) is not int or not 0 <= parent < n
                or not 0 <= child < n or parent == child or parent not in true_parents.get(child, ())):
            raise ValueError('invalid evaluation query or missing evaluator target')
        grouped[child].append(parent)
    for child, parents in true_parents.items():
        if type(child) is not int or not 0 <= child < n or any(type(p) is not int or not 0 <= p < n or p == child for p in parents):
            raise ValueError('invalid evaluator parent set')
    started = time.perf_counter(); cache = prepare_scores(model, embeddings)
    rows = []; child_rr = []
    for child, targets in grouped.items():
        if max_seconds is not None and time.perf_counter()-started >= max_seconds:
            break
        blocks = [score_cached(model, cache, torch.arange(start, min(n, start+candidate_chunk), device=embeddings.device), child)
                  for start in range(0, n, candidate_chunk)]
        scores = torch.cat(blocks)
        if not torch.isfinite(scores).all():
            raise ValueError('nonfinite evaluation scores')
        eligible = torch.ones(n, dtype=torch.bool, device=embeddings.device)
        eligible[child] = False; eligible[list(true_parents[child])] = False
        negative_scores = scores[eligible].sort().values
        target_scores = scores[targets].contiguous()
        left = torch.searchsorted(negative_scores, target_scores, right=False)
        right = torch.searchsorted(negative_scores, target_scores, right=True)
        ranks = 1 + (len(negative_scores)-right).to(torch.float64) + .5*(right-left).to(torch.float64)
        values = ranks.cpu().tolist(); child_rr.append(sum(1/r for r in values)/len(values))
        rows.extend({'parent': p, 'child': child, 'rank': rank, 'candidates': len(negative_scores)+1}
                    for p, rank in zip(targets, values))
    count = len(rows)
    return {'status': 'complete' if count == len(queries) else 'incomplete_time_limit',
            'metric_scope': 'filtered_all_entity_candidates', 'expected_queries': len(queries),
            'completed_queries': count, 'completed_children': len(child_rr),
            'query_micro_mrr': sum(1/r['rank'] for r in rows)/count if count else None,
            'child_macro_mrr': sum(child_rr)/len(child_rr) if child_rr else None,
            'hits': {str(k): sum(r['rank'] <= k for r in rows)/count if count else None for k in (1, 3, 10)},
            'rows': rows, 'elapsed_seconds': time.perf_counter()-started,
            'tie_policy': 'average rank of exactly equal computed scores; target excluded from competitors'}

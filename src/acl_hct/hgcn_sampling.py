"""Uniform nonself sampling of official row-normalized A+I with HT weights."""
from dataclasses import dataclass
import torch

from .backbone import validate_neighbors
from .protocols import digest, mask_indexed_queries


def validate_graph(neighbors):
    validate_neighbors(neighbors, len(neighbors))
    if any(row != sorted(row) or i in row for i, row in enumerate(neighbors)):
        raise ValueError('sorted nonself neighbor rows required')


def mask_graph(neighbors, positives):
    """Remove both target directions before degree, self and sampling weights."""
    validate_graph(neighbors)
    for a, b in positives:
        if (type(a) is not int or type(b) is not int or a == b
                or not 0 <= a < len(neighbors) or not 0 <= b < len(neighbors)
                or b not in neighbors[a] or a not in neighbors[b]):
            raise ValueError('mask targets must be visible train positives')
    # Duplicate supervised targets do not change graph degrees or loss weights.
    return mask_indexed_queries(neighbors, positives)


@dataclass(frozen=True)
class SamplingPlan:
    graph_hash: str
    fanout: int | None
    indices: torch.Tensor
    weights: torch.Tensor
    inclusion_probabilities: torch.Tensor
    populations: torch.Tensor
    selected: torch.Tensor

    def matrix(self, *, device='cpu', dtype=torch.float32):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('FP32 or FP64 adjacency required')
        n = len(self.populations)
        return torch.sparse_coo_tensor(self.indices.to(device), self.weights.to(device=device, dtype=dtype),
                                       (n, n), device=device, dtype=dtype).coalesce()

    def archive(self):
        return {'graph_hash': self.graph_hash, 'fanout': self.fanout, 'indices': self.indices,
                'weights_fp64': self.weights, 'inclusion_probabilities': self.inclusion_probabilities,
                'nonself_populations': self.populations, 'nonself_selected': self.selected,
                'sampling': 'uniform without replacement; independent layer plans',
                'weights': 'self 1/(d+1); nonself [1/(d+1)]/[min(f,d)/d]; d is postmask degree'}


def make_plan(neighbors, fanout, generator):
    validate_graph(neighbors)
    if fanout is not None and (type(fanout) is not int or fanout < 1):
        raise ValueError('positive integer fanout or None required')
    if not isinstance(generator, torch.Generator) or generator.device.type != 'cpu':
        raise ValueError('explicit CPU sampling generator required')
    rows, columns, weights, probabilities, populations, selected = [], [], [], [], [], []
    for i, candidates in enumerate(neighbors):
        d = len(candidates)
        m = d if fanout is None else min(d, fanout)
        if m == d:
            chosen = candidates  # No RNG consumption; exact original COO order.
        else:
            ids = torch.randperm(d, generator=generator)[:m].sort().values.tolist()
            chosen = [candidates[j] for j in ids]
        populations.append(d); selected.append(m)
        pi = m / d if d else 1.
        for j in sorted([i, *chosen]):
            self_message = j == i
            rows.append(i); columns.append(j)
            probabilities.append(1. if self_message else pi)
            weights.append(1. / (d + 1) if self_message else (1. / (d + 1)) / pi)
    return SamplingPlan(digest(neighbors), fanout, torch.tensor([rows, columns], dtype=torch.long),
                        torch.tensor(weights, dtype=torch.float64), torch.tensor(probabilities, dtype=torch.float64),
                        torch.tensor(populations, dtype=torch.long), torch.tensor(selected, dtype=torch.long))

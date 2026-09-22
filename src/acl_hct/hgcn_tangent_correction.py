"""Local relation-supervised tangent correction for frozen official HGCN layers.

This is a new method path. It does not modify the accepted HGCN validation
encoder, sampler, scoring head, or any archived F/S evidence.
"""
import math

import torch
from torch import nn


class TangentCorrection(nn.Module):
    """Three local sampled-message directions at each node's aggregate point."""

    def __init__(self, c=1., hidden=16, tau=.05, eps=1e-8, chunk_edges=4096):
        super().__init__()
        if (not math.isfinite(c) or c <= 0 or type(hidden) is not int or hidden < 1
                or not math.isfinite(tau) or tau <= 0 or not math.isfinite(eps) or eps <= 0
                or type(chunk_edges) is not int or chunk_edges < 1):
            raise ValueError('positive finite correction geometry and chunk settings required')
        self.c = float(c)
        self.tau = float(tau)
        self.eps = float(eps)
        self.chunk_edges = chunk_edges
        self.coefficients = nn.Sequential(nn.Linear(10, hidden), nn.SiLU(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.coefficients[-1].weight)
        nn.init.zeros_(self.coefficients[-1].bias)

    def _conformal(self, point):
        denominator = 1 - self.c * torch.sum(point * point, dim=-1, keepdim=True)
        if not torch.isfinite(point).all() or (denominator <= 0).any():
            raise ValueError('finite strict-interior aggregate point required')
        return 2 / denominator

    def _mobius_add(self, x, y):
        c = self.c
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denominator = 1 + 2 * c * xy + c * c * x2 * y2
        return numerator / denominator.clamp_min(self.eps)

    def log_at(self, base, point):
        """Coordinate Log with differentiable conformal factor at the base."""
        if base.shape != point.shape or base.ndim != 2:
            raise ValueError('matching [N,D] Log operands required')
        lam = self._conformal(base)
        self._conformal(point)
        # Write (-base) ⊕ point in terms of point-base. This is algebraically
        # identical to Mobius addition, but is exactly zero at coincidence
        # without discarding the derivative with respect to either operand.
        difference = point - base
        base_sq = (base * base).sum(dim=-1, keepdim=True)
        point_sq = (point * point).sum(dim=-1, keepdim=True)
        difference_sq = (difference * difference).sum(dim=-1, keepdim=True)
        numerator = ((1 - self.c * base_sq) * difference
                     - self.c * difference_sq * base)
        denominator = ((1 - self.c * base_sq) * (1 - self.c * point_sq)
                       + self.c * difference_sq)
        displacement = numerator / denominator.clamp_min(self.eps)
        radius = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True).clamp_min(self.eps)
        argument = (math.sqrt(self.c) * radius).clamp(max=1 - torch.finfo(radius.dtype).eps)
        scale = 2 * torch.atanh(argument) / (math.sqrt(self.c) * lam * radius)
        return displacement * scale

    def exp_at(self, base, tangent):
        """Coordinate Exp; caller applies the official FP32 boundary projection."""
        if base.shape != tangent.shape or base.ndim != 2:
            raise ValueError('matching [N,D] Exp operands required')
        lam = self._conformal(base)
        length = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(self.eps)
        scale = torch.tanh(math.sqrt(self.c) * lam * length / 2) / (math.sqrt(self.c) * length)
        return self._mobius_add(base, tangent * scale)

    @staticmethod
    def _plan_edges(plan, nodes):
        """Check post-mask HT weights, self rows and selected message inventory."""
        index = plan.indices
        pop = plan.populations
        selected = plan.selected
        if (index.device.type != 'cpu' or index.dtype != torch.long or index.shape[0] != 2
                or pop.device.type != 'cpu' or selected.device.type != 'cpu'
                or pop.dtype != torch.long or selected.dtype != torch.long
                or pop.shape != (nodes,) or selected.shape != (nodes,)
                or plan.weights.shape != (index.shape[1],)
                or plan.inclusion_probabilities.shape != (index.shape[1],)):
            raise ValueError('complete CPU sampling metadata required')
        row, col = index
        if ((row < 0).any() or (row >= nodes).any() or (col < 0).any() or (col >= nodes).any()
                or (pop < 0).any() or (selected < 0).any() or (selected > pop).any()):
            raise ValueError('invalid sampled edge or degree')
        key = row * nodes + col
        if len(key) > 1 and not (key[1:] > key[:-1]).all():
            raise ValueError('distinct row-major sampled edges required')
        own = row == col
        if (not torch.equal(torch.bincount(row[own], minlength=nodes), torch.ones(nodes, dtype=torch.long))
                or not torch.equal(torch.bincount(row[~own], minlength=nodes), selected)):
            raise ValueError('one self plus exactly k selected nonself messages per row required')
        if plan.fanout is not None and not torch.equal(selected, torch.minimum(pop, torch.full_like(pop, plan.fanout))):
            raise ValueError('sampling fanout differs from registered selected counts')
        n = pop[row].to(torch.float64)
        k = selected[row].to(torch.float64)
        pi = torch.where(own, torch.ones_like(n), k / n.clamp_min(1))
        weights = torch.where(own, 1 / (n + 1), 1 / (n + 1) / pi.clamp_min(1e-15))
        if (not torch.allclose(plan.inclusion_probabilities, pi, rtol=0, atol=1e-14)
                or not torch.allclose(plan.weights, weights, rtol=0, atol=1e-14)):
            raise ValueError('sampling probabilities or HT weights changed')
        return row, col

    def forward(self, aggregate, messages, plan, manifold):
        if (aggregate.shape != messages.shape or aggregate.ndim != 2
                or aggregate.dtype not in (torch.float32, torch.float64)
                or aggregate.device != messages.device or aggregate.dtype != messages.dtype):
            raise ValueError('matching native aggregate and transformed-message matrices required')
        nodes = len(aggregate)
        row, col = self._plan_edges(plan, nodes)
        eligible = (plan.selected >= 3) & (plan.selected < plan.populations)
        ids_cpu = torch.nonzero(eligible, as_tuple=False).flatten()
        if not len(ids_cpu):
            return aggregate, {'normalized_step_mean': aggregate.sum() * 0,
                               'eligible_nodes': 0, 'clipped_nodes': 0,
                               'near_zero_own': 0, 'near_zero_neighbors': 0,
                               'selected_neighbors': 0, 'projected_nodes': 0}
        edge_mask = (row != col) & eligible[row]
        global_row, global_col = row[edge_mask], col[edge_mask]
        lookup = torch.full((nodes,), -1, dtype=torch.long)
        lookup[ids_cpu] = torch.arange(len(ids_cpu))
        compact_row = lookup[global_row].to(aggregate.device)
        global_col = global_col.to(aggregate.device)
        ids = ids_cpu.to(aggregate.device)
        base = aggregate.index_select(0, ids)
        own_message = messages.index_select(0, ids)
        selected = plan.selected[ids_cpu].to(device=aggregate.device, dtype=aggregate.dtype)
        population = plan.populations[ids_cpu].to(device=aggregate.device, dtype=aggregate.dtype)
        conformal = self._conformal(base)
        own_direction = self.log_at(base, own_message)
        summed = torch.zeros_like(base)
        near_zero_neighbors = 0
        for start in range(0, len(global_col), self.chunk_edges):
            rr = compact_row[start:start + self.chunk_edges]
            cc = global_col[start:start + self.chunk_edges]
            tangent = self.log_at(base.index_select(0, rr), messages.index_select(0, cc))
            near_zero_neighbors += int((torch.linalg.vector_norm(tangent.detach(), dim=-1) <= self.eps).sum())
            summed.index_add_(0, rr, tangent)
        average = summed / selected[:, None]
        variance_sum = base.new_zeros((len(base),))
        third_sum = torch.zeros_like(base)
        covariance_sum = torch.zeros_like(base)
        for start in range(0, len(global_col), self.chunk_edges):
            rr = compact_row[start:start + self.chunk_edges]
            cc = global_col[start:start + self.chunk_edges]
            centered = self.log_at(base.index_select(0, rr), messages.index_select(0, cc)) - average.index_select(0, rr)
            lam = conformal.index_select(0, rr)
            squared = (lam * lam * centered * centered).sum(dim=-1)
            variance_sum.index_add_(0, rr, squared)
            third_sum.index_add_(0, rr, squared[:, None] * centered)
            own = own_direction.index_select(0, rr)
            covariance_sum.index_add_(0, rr, (lam * lam * centered * own).sum(dim=-1, keepdim=True) * centered)
        variance = variance_sum / (selected - 1)
        third = third_sum / selected[:, None]
        covariance_own = covariance_sum / (selected - 1)[:, None]
        own_sq = (conformal * conformal * own_direction * own_direction).sum(dim=-1)
        average_sq = (conformal * conformal * average * average).sum(dim=-1)
        third_length = conformal.squeeze(-1) * torch.linalg.vector_norm(third, dim=-1)
        own_length = torch.linalg.vector_norm(conformal * own_direction, dim=-1)
        cosine = ((conformal * conformal * own_direction * third).sum(dim=-1) /
                  (own_length * third_length + self.eps))
        own_covariance = (conformal * conformal * own_direction * covariance_own).sum(dim=-1)
        features = torch.stack((torch.log1p(population), selected / population,
                                torch.log1p(selected), torch.full_like(selected, math.log1p(self.c)),
                                torch.log1p(self.c * variance), torch.log1p(self.c * own_sq),
                                torch.log1p(self.c * average_sq),
                                third_length / (variance.pow(1.5) + self.eps),
                                cosine, own_covariance / (own_sq * variance + self.eps)), dim=-1)
        coeff = 2 * torch.tanh(self.coefficients(features))
        basis = torch.stack((self.c * variance[:, None] * own_direction,
                             self.c * third, self.c * covariance_own), dim=1)
        q = (population / (population + 1)).square() * (1 - selected / population) / selected
        raw = q[:, None] * (coeff[:, :, None] * basis).sum(dim=1)
        raw_norm = conformal.squeeze(-1) * torch.linalg.vector_norm(raw, dim=-1)
        clip_scale = (self.tau / raw_norm.clamp_min(self.eps)).clamp(max=1)
        step = raw * clip_scale[:, None]
        step_norm = conformal.squeeze(-1) * torch.linalg.vector_norm(step, dim=-1)
        unprojected = self.exp_at(base, step)
        corrected = manifold.proj(unprojected, aggregate.new_tensor(self.c))
        if not (torch.isfinite(features).all() and torch.isfinite(corrected).all()
                and torch.isfinite(step_norm).all()):
            raise ValueError('nonfinite local correction')
        result = torch.index_copy(aggregate, 0, ids, corrected)
        penalty = (step_norm.square().sum() / nodes) / (self.tau * self.tau)
        return result, {'normalized_step_mean': penalty, 'eligible_nodes': len(ids_cpu),
                        'clipped_nodes': int((raw_norm.detach() > self.tau).sum()),
                        'near_zero_own': int((torch.linalg.vector_norm(own_direction.detach(), dim=-1) <= self.eps).sum()),
                        'near_zero_neighbors': near_zero_neighbors,
                        'selected_neighbors': len(global_col),
                        'projected_nodes': int((unprojected.detach() != corrected.detach()).any(dim=-1).sum())}


def ball_distance_from_anchor(anchor, point, c=1.):
    """Differentiable geodesic distance using a stable asinh expression."""
    if anchor.shape != point.shape or anchor.ndim != 2:
        raise ValueError('matching [N,D] geodesic operands required')
    if not math.isfinite(c) or c <= 0:
        raise ValueError('positive curvature required')
    a = 1 - c * (anchor * anchor).sum(dim=-1)
    b = 1 - c * (point * point).sum(dim=-1)
    if not (torch.isfinite(anchor).all() and torch.isfinite(point).all() and (a > 0).all() and (b > 0).all()):
        raise ValueError('finite interior geodesic operands required')
    difference = torch.linalg.vector_norm(point - anchor, dim=-1)
    return 2 / math.sqrt(c) * torch.asinh(math.sqrt(c) * difference / torch.sqrt(a * b))


def training_relation_weights(positives, nodes):
    """Fixed edge weights for mean_child mean_parent over train-visible edges."""
    if positives.ndim != 2 or positives.shape[1] != 2 or positives.dtype != torch.long:
        raise ValueError('train positive [E,2] int64 relation tensor required')
    if len(positives) < 1 or (positives < 0).any() or (positives >= nodes).any():
        raise ValueError('nonempty valid train relation indices required')
    if len(torch.unique(positives, dim=0)) != len(positives):
        raise ValueError('distinct train positive relations required')
    children = positives[:, 1]
    counts = torch.bincount(children, minlength=nodes)
    child_count = int((counts > 0).sum())
    return len(positives) / (child_count * counts[children].to(torch.float64))


def relation_gap_loss(points, positives, relation_weights, root, margin, c=1.):
    if (positives.ndim != 2 or positives.shape[1] != 2 or positives.dtype != torch.long
            or positives.device != points.device or relation_weights.shape != (len(positives),)
            or relation_weights.device != points.device or not 0 <= root < len(points)
            or not math.isfinite(margin) or margin <= 0):
        raise ValueError('valid train relation batch and positive margin required')
    anchor = points[root:root + 1].detach().expand(len(positives), -1)
    parent = points.index_select(0, positives[:, 0])
    child = points.index_select(0, positives[:, 1])
    gap = ball_distance_from_anchor(anchor, child, c) - ball_distance_from_anchor(anchor, parent, c)
    hinge = torch.relu((margin - gap) / margin)
    return (relation_weights.to(points.dtype) * hinge.square()).mean()

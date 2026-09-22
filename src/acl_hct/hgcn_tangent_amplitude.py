"""Versioned amplitude correction; Stage A's hard-capped module is untouched."""
import math

import torch

from .hgcn_tangent_correction import TangentCorrection


def _distribution(values):
    """Small, detached diagnostic summary of eligible node instances."""
    v = values.detach().reshape(-1)
    if not v.numel():
        return None
    sample = v.float()
    quantiles = torch.quantile(sample, sample.new_tensor([.1, .5, .9, .99]))
    return {'min': float(v.min()), 'mean': float(v.float().mean()),
            'p10': float(quantiles[0]), 'p50': float(quantiles[1]),
            'p90': float(quantiles[2]), 'p99': float(quantiles[3]),
            'max': float(v.max())}


def calibrate_and_cap(basis, coeff, conformal, q, tau=.05, eps=1e-8, scale=4):
    """Pure local amplitude map; norms use the aggregate-point Riemann metric."""
    if (basis.ndim != 3 or basis.shape[1] != 3 or coeff.shape != basis.shape[:2]
            or conformal.shape != (len(basis), 1) or q.shape != (len(basis),)
            or scale != 4 or not math.isfinite(tau) or tau <= 0
            or not math.isfinite(eps) or eps <= 0):
        raise ValueError('registered three-direction physical amplitude operands required')
    basis_norm_sq = ((conformal[:, None] * basis).square()).sum(dim=-1)
    unit = basis * torch.rsqrt(basis_norm_sq + eps * eps)[:, :, None]
    raw = (scale * tau) * q[:, None] * (coeff[:, :, None] * unit).sum(dim=1)
    raw_ratio_sq = ((conformal * raw).square()).sum(dim=-1) / (tau * tau)
    step = raw * torch.rsqrt(1 + raw_ratio_sq)[:, None]
    return step, raw_ratio_sq, basis_norm_sq


class AmplitudeCorrection(TangentCorrection):
    """Smooth direction calibration and soft radial cap in physical tangent units."""

    def __init__(self, c=1., hidden=16, tau=.05, eps=1e-8, chunk_edges=4096, scale=4):
        if scale != 4:
            raise ValueError('registered amplitude scale s=4 required')
        super().__init__(c=c, hidden=hidden, tau=tau, eps=eps, chunk_edges=chunk_edges)
        self.scale = scale

    def forward(self, aggregate, messages, plan, manifold, *, detailed=False):
        if (aggregate.shape != messages.shape or aggregate.ndim != 2
                or aggregate.dtype not in (torch.float32, torch.float64)
                or aggregate.device != messages.device or aggregate.dtype != messages.dtype):
            raise ValueError('matching native aggregate and transformed-message matrices required')
        nodes = len(aggregate)
        row, col = self._plan_edges(plan, nodes)
        eligible = (plan.selected >= 3) & (plan.selected < plan.populations)
        ids_cpu = torch.nonzero(eligible, as_tuple=False).flatten()
        if not len(ids_cpu):
            zero = aggregate.sum() * 0
            return aggregate, {'normalized_step_mean': zero, 'post_cap_step_mean': zero,
                               'eligible_nodes': 0, 'near_zero_own': 0,
                               'near_zero_neighbors': 0, 'selected_neighbors': 0,
                               'projected_nodes': 0, 'raw_over_tau_nodes': 0,
                               'basis_epsilon_affected': 0}
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
        step, raw_ratio_sq, basis_norm_sq = calibrate_and_cap(
            basis, coeff, conformal, q, self.tau, self.eps, self.scale)
        step_ratio_sq = raw_ratio_sq / (1 + raw_ratio_sq)
        unprojected = self.exp_at(base, step)
        corrected = manifold.proj(unprojected, aggregate.new_tensor(self.c))
        if not (torch.isfinite(features).all() and torch.isfinite(corrected).all()
                and torch.isfinite(raw_ratio_sq).all() and torch.isfinite(step_ratio_sq).all()):
            raise ValueError('nonfinite local amplitude correction')
        result = torch.index_copy(aggregate, 0, ids, corrected)
        penalty = raw_ratio_sq.sum() / nodes
        post_cap = step_ratio_sq.sum() / nodes
        measured = {'normalized_step_mean': penalty, 'post_cap_step_mean': post_cap,
                    'eligible_nodes': len(ids_cpu),
                    'near_zero_own': int((torch.linalg.vector_norm(own_direction.detach(), dim=-1) <= self.eps).sum()),
                    'near_zero_neighbors': near_zero_neighbors,
                    'selected_neighbors': len(global_col),
                    'projected_nodes': int((unprojected.detach() != corrected.detach()).any(dim=-1).sum()),
                    'raw_over_tau_nodes': int((raw_ratio_sq.detach() > 1).sum()),
                    'basis_epsilon_affected': [int((basis_norm_sq.detach()[:, j] <= self.eps * self.eps).sum())
                                               for j in range(3)]}
        if detailed:
            x = torch.sqrt(raw_ratio_sq.detach())
            measured['distribution'] = {
                'basis_physical_norm': [_distribution(torch.sqrt(basis_norm_sq[:, j]))
                                        for j in range(3)],
                'q': _distribution(q),
                'coefficient_abs': _distribution(coeff.abs()),
                'raw_over_tau': _distribution(x),
                'step_over_tau': _distribution(torch.sqrt(step_ratio_sq.detach())),
                'radial_jacobian': _distribution((1 + raw_ratio_sq.detach()).pow(-1.5)),
                'q_elasticity': _distribution((1 + raw_ratio_sq.detach()).reciprocal())}
            measured['coefficient_near_two'] = int((coeff.detach().abs() >= 1.99).sum())
        return result, measured

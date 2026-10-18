"""UNTESTED draft: one new cross-model correction core, not legacy E.

No data, model adapters, training, file IO, or scientific execution is included.
Weighted dispersion and coverage/ESS gates are descriptors, not general
unbiased variance or signed-bias estimators. See README.md before integration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch import Tensor, nn


RELATIVE_CAP = 0.05
MIN_EFFECTIVE_SAMPLES = 3.0


class Geometry(Protocol):
    """All methods must be differentiable, including exp(base, zero).

    Points and tangents have ambient dimension C; coordinates have intrinsic
    dimension D. Coordinate maps must use a fixed, documented isometric frame
    (e.g. registered transport to an origin orthonormal frame), not a fitted
    frame or a sample-dependent choice made after observing effects.
    """

    intrinsic_dim: int

    def log(self, base: Tensor, point: Tensor) -> Tensor: ...
    def norm2(self, base: Tensor, tangent: Tensor) -> Tensor: ...
    def exp(self, base: Tensor, tangent: Tensor) -> Tensor: ...
    def to_orthonormal(self, base: Tensor, tangent: Tensor) -> Tensor: ...
    def from_orthonormal(self, base: Tensor, coordinates: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class EuclideanGeometry:
    intrinsic_dim: int

    def log(self, base: Tensor, point: Tensor) -> Tensor:
        return point - base

    def norm2(self, base: Tensor, tangent: Tensor) -> Tensor:
        return tangent.square().sum(dim=-1)

    def exp(self, base: Tensor, tangent: Tensor) -> Tensor:
        return base + tangent

    def to_orthonormal(self, base: Tensor, tangent: Tensor) -> Tensor:
        return tangent

    def from_orthonormal(self, base: Tensor, coordinates: Tensor) -> Tensor:
        return coordinates


def candidate_hidden_width(intrinsic_dim: int) -> int:
    """Nearest integer, half-up: H = round((3D+7)/9).

    Integer numerator / odd denominator 9 cannot equal an exact half. The
    explicit rule nevertheless avoids language-specific banker rounding.
    Parameter counts: candidate 9H+1; generic 3D+8; difference <= 4.
    """
    if type(intrinsic_dim) is not int or intrinsic_dim < 1:
        raise ValueError("positive integer intrinsic dimension required")
    return (2 * (3 * intrinsic_dim + 7) + 9) // 18


def _require(condition: Tensor | bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _finite(value: Tensor, name: str) -> None:
    _require(torch.isfinite(value).all(), f"nonfinite {name}")


def _nonnegative_norm2(geometry: Geometry, base: Tensor, value: Tensor) -> Tensor:
    squared = geometry.norm2(base, value)
    _require(squared.shape == base.shape[:-1], "geometry.norm2 shape mismatch")
    _finite(squared, "geometry squared norm")
    # The geometry provider must handle roundoff in its own valid tangent
    # norm. This core does not silently repair indefinite/non-tangent values.
    _require((squared >= 0).all(), "geometry returned negative squared norm")
    return squared


def _safe_sqrt_nonnegative(value: Tensor) -> Tensor:
    # Do not evaluate sqrt(0) in an active differentiable branch: its infinite
    # derivative can contaminate masked gradients through 0 * infinity.
    positive = value > 0
    safe = torch.where(positive, value, torch.ones_like(value))
    return torch.where(positive, safe.sqrt(), torch.zeros_like(value))


@dataclass
class CorrectionOutput:
    points: Tensor
    step: Tensor
    features: Tensor
    variance: Tensor
    scale2: Tensor
    effective_samples: Tensor
    eligible: Tensor
    step_norm2: Tensor
    # Diagnostic only: mean squared fraction of the registered relative cap.
    # This value is not a loss or a recommended regularizer.
    normalized_step_size: Tensor


class SharedCorrection(nn.Module):
    """Candidate or ordinary low-rank residual under one fixed input contract.

    The candidate is self-direction constrained. The generic residual sees
    the same seven scalars plus normalized self-direction coordinates, and
    can produce a non-collinear direction. They do not have identical
    function classes, FLOPs, or end-to-end costs merely because their
    parameter counts differ by at most four.
    """

    def __init__(self, intrinsic_dim: int, variant: Literal["candidate", "generic"],
                 *, init_seed: int):
        super().__init__()
        hidden = candidate_hidden_width(intrinsic_dim)
        if variant not in ("candidate", "generic"):
            raise ValueError("variant must be candidate or generic")
        if type(init_seed) is not int or init_seed < 0:
            raise ValueError("explicit nonnegative module initialization seed required")
        self.intrinsic_dim = intrinsic_dim
        self.variant = variant
        # CPU initialization restores the caller's CPU RNG and does not
        # initialize/query CUDA. Caller subsequently moves the module.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(init_seed)
            if variant == "candidate":
                self.network = nn.Sequential(
                    nn.Linear(7, hidden), nn.SiLU(), nn.Linear(hidden, 1))
                expected = 9 * hidden + 1
            else:
                self.network = nn.Sequential(
                    nn.Linear(intrinsic_dim + 7, 1), nn.SiLU(),
                    nn.Linear(1, intrinsic_dim))
                expected = 3 * intrinsic_dim + 8
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
        if sum(parameter.numel() for parameter in self.parameters()) != expected:
            raise ValueError("registered parameter-count mismatch")

    def forward(self, *, base: Tensor, self_message: Tensor, messages: Tensor,
                neighbor_mask: Tensor, neighbor_weights: Tensor,
                population: Tensor, neighbor_mass: Tensor,
                geometry: Geometry) -> CorrectionOutput:
        """Correct B aggregate points using padded B x K x C messages.

        N=population excludes self; k=neighbor_mask.sum(-1). Masks represent
        actual distinct sampled neighbors; upstream adapters must assert ID
        uniqueness and sampler semantics. p=neighbor_weights is zero on
        padding and normalized over neighbors, excluding self. rho is
        neighbor_mass, NOT an attention probability of an individual edge.
        No full state/labels are inputs. All tensors belong to this forward.
        """
        _require(base.ndim == 2 and base.shape[0] > 0, "base must be nonempty B x C")
        batch, ambient_dim = base.shape
        _require(self_message.shape == base.shape, "self message space mismatch")
        _require(messages.ndim == 3 and messages.shape[0] == batch
                 and messages.shape[2] == ambient_dim, "message space mismatch")
        shape = messages.shape[:2]
        _require(neighbor_mask.shape == shape and neighbor_mask.dtype == torch.bool,
                 "boolean B x K neighbor mask required")
        _require(neighbor_weights.shape == shape, "neighbor weights shape mismatch")
        _require(population.shape == (batch,) and population.dtype in
                 (torch.int32, torch.int64), "integer B-vector population required")
        _require(neighbor_mass.shape == (batch,), "neighbor mass shape mismatch")
        _require(base.dtype in (torch.float32, torch.float64), "FP32 or FP64 required")
        _require(geometry.intrinsic_dim == self.intrinsic_dim, "intrinsic dimension mismatch")
        for name, value in (("self_message", self_message), ("messages", messages),
                            ("neighbor_weights", neighbor_weights),
                            ("neighbor_mass", neighbor_mass)):
            _require(value.dtype == base.dtype and value.device == base.device,
                     f"{name} dtype/device mismatch")
            _finite(value, name)
        _finite(base, "base")
        _require(population.device == base.device and neighbor_mask.device == base.device,
                 "metadata device mismatch")
        parameter = next(self.parameters())
        _require(parameter.dtype == base.dtype and parameter.device == base.device,
                 "module must match input dtype/device")

        selected_int = neighbor_mask.sum(dim=-1)
        _require(((population >= 0) & (selected_int <= population)).all(), "invalid N/k")
        _require(((neighbor_mass >= 0) & (neighbor_mass <= 1)).all(), "rho outside [0,1]")
        _require((neighbor_mass[population == 0] == 0).all(), "isolated nodes need rho=0")
        _require((neighbor_weights >= 0).all(), "negative neighbor probability")
        _require((neighbor_weights[~neighbor_mask] == 0).all(), "nonzero padded probability")
        weight_sum = neighbor_weights.sum(dim=-1)
        tolerance = 64 * torch.finfo(base.dtype).eps
        positive_mass = (selected_int > 0) & (neighbor_mass > 0)
        _require((torch.abs(weight_sum[positive_mass] - 1) <= tolerance).all(),
                 "neighbor probabilities must sum to one")
        inactive_sum = weight_sum[~positive_mass]
        _require(((inactive_sum.abs() <= tolerance)
                  | ((inactive_sum - 1).abs() <= tolerance)).all(),
                 "zero-mass rows need zero or normalized neighbor weights")

        # Input normalization allows finite-precision roundoff, but all
        # subsequent statistics use ONE consistently renormalized p. Do not
        # let the accepted sum tolerance become a fictitious dispersion.
        safe_weight_sum = torch.where(weight_sum > 0, weight_sum,
                                      torch.ones_like(weight_sum))
        normalized_weights = neighbor_weights / safe_weight_sum[:, None]
        squared_weights = normalized_weights.square().sum(dim=-1)
        safe_squared_weights = torch.where(squared_weights > 0, squared_weights,
                                           torch.ones_like(squared_weights))
        effective = torch.where(squared_weights > 0, safe_squared_weights.reciprocal(),
                                torch.zeros_like(squared_weights))
        # Tolerance only handles roundoff at the fixed k_eff=3 boundary.
        eligible_before_dispersion = ((selected_int >= 3) & (selected_int < population)
                                     & (neighbor_mass > 0)
                                     & (effective >= MIN_EFFECTIVE_SAMPLES * (1 - tolerance)))
        ids = torch.nonzero(eligible_before_dispersion, as_tuple=False).flatten()
        direction = torch.zeros_like(base)
        average = torch.zeros_like(base)
        variance = base.new_zeros(batch)
        own_norm2 = base.new_zeros(batch)
        mean_norm2 = base.new_zeros(batch)

        if ids.numel():
            local_base = base.index_select(0, ids)
            local_mask = neighbor_mask.index_select(0, ids)
            local_messages = messages.index_select(0, ids)
            local_weights = normalized_weights.index_select(0, ids)
            local_direction = geometry.log(local_base, self_message.index_select(0, ids))
            _require(local_direction.shape == local_base.shape, "geometry.log shape mismatch")
            _finite(local_direction, "self log")
            # Never call a manifold log on invalid padded points.
            row, col = torch.nonzero(local_mask, as_tuple=True)
            edge_base = local_base.index_select(0, row)
            edge_logs = geometry.log(edge_base, local_messages[row, col])
            _require(edge_logs.shape == edge_base.shape, "edge log shape mismatch")
            _finite(edge_logs, "edge log")
            # Center relative to an actual message before reduction. If all
            # messages are identical (including nonzero ones), differences
            # are exactly zero, so V remains exactly zero despite rounding
            # in p.sum(). No epsilon threshold substitutes for that identity.
            local_counts = local_mask.sum(dim=-1)
            first_edge = local_counts.cumsum(dim=0) - local_counts
            reference_logs = edge_logs.index_select(0, first_edge)
            differences = edge_logs - reference_logs.index_select(0, row)
            mean_difference = torch.zeros_like(local_base).index_add(
                0, row, local_weights[row, col, None] * differences)
            local_average = reference_logs + mean_difference
            centered = differences - mean_difference.index_select(0, row)
            edge_norm2 = _nonnegative_norm2(geometry, edge_base, centered)
            numerator = local_base.new_zeros(len(ids)).index_add(
                0, row, local_weights[row, col] * edge_norm2)
            denominator = 1 - squared_weights.index_select(0, ids)
            _require((denominator > 0).all(), "degenerate weighted dispersion")
            local_variance = numerator / denominator
            direction = direction.index_copy(0, ids, local_direction)
            average = average.index_copy(0, ids, local_average)
            variance = variance.index_copy(0, ids, local_variance)
            own_norm2 = own_norm2.index_copy(
                0, ids, _nonnegative_norm2(geometry, local_base, local_direction))
            mean_norm2 = mean_norm2.index_copy(
                0, ids, _nonnegative_norm2(geometry, local_base, local_average))

        scale2 = variance + own_norm2 + mean_norm2
        _finite(scale2, "local squared scale")
        eligible = eligible_before_dispersion & (variance > 0) & (scale2 > 0)
        # Substitute ONE only into inactive denominators, never into measured
        # dispersion. True V=0 remains V=0 and exactly disables correction.
        safe_scale2 = torch.where(scale2 > 0, scale2, torch.ones_like(scale2))
        scale = safe_scale2.sqrt()
        selected = selected_int.to(base.dtype)
        n = population.to(base.dtype)
        safe_n = torch.where(n > 0, n, torch.ones_like(n))
        coverage = selected / safe_n
        safe_effective = torch.where(effective > 0, effective, torch.ones_like(effective))
        coverage_gate = neighbor_mass.square() * (1 - coverage) / safe_effective
        coverage_gate = torch.where(eligible, coverage_gate, torch.zeros_like(coverage_gate))
        features = torch.stack((n.log1p(), coverage, effective.log1p(), neighbor_mass,
                                variance / safe_scale2, own_norm2 / safe_scale2,
                                mean_norm2 / safe_scale2), dim=-1)
        _finite(features, "gate features")

        if self.variant == "candidate":
            coefficient = 2 * torch.tanh(self.network(features)).squeeze(-1)
            raw = (coefficient * coverage_gate * (variance / safe_scale2))[:, None] * direction
        else:
            coordinates = geometry.to_orthonormal(base, direction)
            _require(coordinates.shape == (batch, self.intrinsic_dim), "frame dimension mismatch")
            _finite(coordinates, "self direction coordinates")
            combined = torch.cat((coordinates / scale[:, None], features), dim=-1)
            output_coordinates = self.network(combined)
            raw = geometry.from_orthonormal(
                base, (coverage_gate * scale)[:, None] * output_coordinates)
            _require(raw.shape == base.shape, "inverse frame shape mismatch")
        _finite(raw, "raw tangent step")

        raw_norm2 = _nonnegative_norm2(geometry, base, raw)
        raw_norm = _safe_sqrt_nonnegative(raw_norm2)
        safe_raw_norm = torch.where(raw_norm > 0, raw_norm, torch.ones_like(raw_norm))
        factor = torch.where(
            raw_norm > 0,
            torch.minimum(torch.ones_like(scale), RELATIVE_CAP * scale / safe_raw_norm),
            torch.ones_like(scale))
        step = raw * factor[:, None]
        step = torch.where(eligible[:, None], step, torch.zeros_like(step))
        step_norm2 = _nonnegative_norm2(geometry, base, step)

        # Keep zero-initialized eligible steps in exp's autograd path; skipping
        # rows just because step==0 would prevent the module from learning.
        active_ids = torch.nonzero(eligible, as_tuple=False).flatten()
        corrected = base
        if active_ids.numel():
            changed = geometry.exp(base.index_select(0, active_ids),
                                   step.index_select(0, active_ids))
            _require(changed.shape == (len(active_ids), ambient_dim), "geometry.exp shape mismatch")
            _finite(changed, "corrected aggregate")
            corrected = base.index_copy(0, active_ids, changed)
        normalized_step_size = (step_norm2 / (RELATIVE_CAP ** 2 * safe_scale2)).mean()
        return CorrectionOutput(corrected, step, features, variance, scale2,
                                effective, eligible, step_norm2, normalized_step_size)

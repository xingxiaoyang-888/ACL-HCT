"""UNEXECUTED geometry draft for the shared correction interface.

Only Euclidean reuse and Lorentz geometry at an explicit fixed radius R are
provided. No Poincare adapter, native-model integration, data/weight loading,
training, or runtime qualification is included.

Formula references read locally (not imported or changed):
  https://github.com/chenweize1998/fully-hyperbolic-nn (pinned; gcn/manifolds/lmath.py)
  _logmap:417, _parallel_transport:562, _expmap:327;
  manifolds/lorentz.py:124 documents the (-,+,...,+) inner product.
The local native _expmap and specialized origin transports contain unit-radius
assumptions; this draft does NOT reuse those implementations for non-unit R.
It uses the radius-explicit mathematical formulas with near-zero series.

POINT AND TANGENT REPRESENTATIONS ARE DIFFERENT. Points have D+1 Lorentz
coordinates. Tangent payloads also have D+1 slots, but are packed (0,w), where
w contains the D orthonormal coordinates after parallel transport to origin.
They are NOT Minkowski ambient tangent vectors. This linear representation
keeps core.py's subtraction/centering closed without a rounded time component.

After checking the native time coordinate against its input-precision error
envelope, local geometry uses the FP64 canonical lift of the native spatial
coordinates: X0=sqrt(R^2+||X_spatial||^2). This is explicit local numerical
normalization, not a claim that the original rounded point was exact. Exp
preserves relative native time roundoff: output_time=base_time*(Y0/X0),
where X0 and Y0 are canonical lifts before and after the spatial increment.
Equivalently the output has the input ratio r=base_time/X0 relative to Y0.
At zero step the ratio is exactly one and the original base is bitwise
preserved with a live tangent derivative. The raw time derivative is r times
the canonical ambient time derivative; the metric is the canonical local
metric, not a claim of exact equality to a native clipped ambient expmap.
FP32 outputs return to their native dtype. This work MUST be costed; no
runtime-lightness or native-operation equivalence is claimed. Grossly invalid
points or non-packed vectors are rejected, never silently projected.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from core import EuclideanGeometry  # Already defined in the shared core.


def _require(condition: Tensor | bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _finite(tensor: Tensor, name: str) -> None:
    _require(torch.isfinite(tensor).all(), f"nonfinite {name}")


def _coshm1_and_sinhc_from_square(square: Tensor) -> tuple[Tensor, Tensor]:
    """cosh(sqrt(t))-1, sinh(sqrt(t))/sqrt(t), continued at t=0.

    Cubic series are used for t<=1e-4. The unused regular branch receives one
    instead of zero, avoiding masked sqrt/division infinite derivatives.
    """
    _require((square >= 0).all(), "negative squared exponential argument")
    small = square <= 1e-4
    safe = torch.where(small, torch.ones_like(square), square)
    root = safe.sqrt()
    regular_coshm1 = 2 * (root / 2).sinh().square()
    regular_sinhc = root.sinh() / root
    series_coshm1 = square * (.5 + square * (1 / 24 + square / 720))
    series_sinhc = 1 + square * (1 / 6 + square * (1 / 120 + square / 5040))
    return (torch.where(small, series_coshm1, regular_coshm1),
            torch.where(small, series_sinhc, regular_sinhc))


def _asinhc_from_square(square: Tensor) -> Tensor:
    """asinh(sqrt(t))/sqrt(t), with its differentiable value one at t=0."""
    _require((square >= 0).all(), "negative squared log argument")
    small = square <= 1e-4
    safe = torch.where(small, torch.ones_like(square), square)
    root = safe.sqrt()
    regular = root.asinh() / root
    series = 1 + square * (-1 / 6 + square * (3 / 40 - 5 * square / 112))
    return torch.where(small, series, regular)


@dataclass(frozen=True)
class LorentzGeometry:
    """Future-sheet hyperboloid <x,x>_L=-R^2, curvature -1/R^2.

    Intrinsic D is explicit. Points have D+1 ambient coordinates; tangent
    payloads are (0, origin-frame coordinates), NOT ambient tangent vectors.
    Radius is a fixed configuration value. Learnable curvature and native
    projection/clipping branches require a separately registered adapter.
    The orthonormal frame is spatial coordinates at origin (R,0,...,0),
    transported along the unique origin-to-base geodesic.
    """

    intrinsic_dim: int
    radius: float = 1.0

    def __post_init__(self) -> None:
        if type(self.intrinsic_dim) is not int or self.intrinsic_dim < 1:
            raise ValueError("positive intrinsic dimension required")
        if (not isinstance(self.radius, (int, float)) or isinstance(self.radius, bool)
                or not math.isfinite(self.radius) or self.radius <= 0
                or not math.isfinite(self.radius * self.radius)
                or self.radius * self.radius <= 0):
            raise ValueError("explicit positive finite fixed radius required")

    def _point(self, point: Tensor) -> Tensor:
        _require(point.ndim >= 1 and point.shape[-1] == self.intrinsic_dim + 1,
                 "Lorentz point needs D+1 ambient coordinates")
        _require(point.dtype in (torch.float32, torch.float64), "FP32/FP64 geometry required")
        _finite(point, "Lorentz point")
        work = point.to(torch.float64)
        _require((work[..., 0] > 0).all(), "point not on future sheet")
        spatial = work[..., 1:]
        canonical_time = (spatial.square().sum(dim=-1) + self.radius ** 2).sqrt()
        _finite(canonical_time, "canonical Lorentz time")
        tolerance = 64 * torch.finfo(point.dtype).eps * canonical_time
        _require(((work[..., 0] - canonical_time).abs() <= tolerance).all(),
                 "native time is outside the registered-radius rounding envelope")
        return torch.cat((canonical_time[..., None], spatial), dim=-1)

    def _matching(self, base: Tensor, value: Tensor, name: str) -> None:
        _require(value.shape == base.shape and value.dtype == base.dtype
                 and value.device == base.device, f"{name} shape/dtype/device mismatch")
        _finite(value, name)

    def _packed_coordinates(self, base: Tensor, tangent: Tensor) -> Tensor:
        self._matching(base, tangent, "packed Lorentz tangent")
        _require((tangent[..., 0] == 0).all(),
                 "tangent must be packed (0,w), not a Minkowski ambient vector")
        return tangent[..., 1:].to(torch.float64)

    def _pack(self, base: Tensor, coordinates: Tensor) -> Tensor:
        coordinates = coordinates.to(base.dtype)
        _finite(coordinates, "Lorentz tangent coordinates")
        return torch.cat((torch.zeros_like(coordinates[..., :1]), coordinates), dim=-1)

    def _ambient_from_coordinates(self, x: Tensor, w: Tensor) -> Tensor:
        # PT_{o -> X}(0,w); X is the canonical local lift, not a rounded point.
        dot = (x[..., 1:] * w).sum(dim=-1)
        time = dot / self.radius
        spatial = w + x[..., 1:] * (dot / (self.radius * (x[..., 0] + self.radius)))[..., None]
        return torch.cat((time[..., None], spatial), dim=-1)

    def to_orthonormal(self, base: Tensor, tangent: Tensor) -> Tensor:
        self._point(base)
        return self._packed_coordinates(base, tangent).to(base.dtype)

    def from_orthonormal(self, base: Tensor, coordinates: Tensor) -> Tensor:
        self._point(base)
        _require(coordinates.shape == base.shape[:-1] + (self.intrinsic_dim,)
                 and coordinates.dtype == base.dtype and coordinates.device == base.device,
                 "origin orthonormal coordinate shape/dtype/device mismatch")
        _finite(coordinates, "origin coordinates")
        return self._pack(base, coordinates)

    def norm2(self, base: Tensor, tangent: Tensor) -> Tensor:
        self._point(base)
        w = self._packed_coordinates(base, tangent)
        # This is the canonical local Lorentz metric expressed in an
        # orthonormal frame. Do not take a Minkowski product of the payload.
        squared = w.square().sum(dim=-1).to(base.dtype)
        _finite(squared, "Lorentz squared norm")
        return squared

    def exp(self, base: Tensor, tangent: Tensor) -> Tensor:
        x = self._point(base)
        w = self._packed_coordinates(base, tangent)
        v = self._ambient_from_coordinates(x, w)
        argument_square = w.square().sum(dim=-1) / self.radius ** 2
        coshm1, sinhc = _coshm1_and_sinhc_from_square(argument_square)
        increment = coshm1[..., None] * x + sinhc[..., None] * v
        native = base.to(torch.float64)
        spatial = (native[..., 1:] + increment[..., 1:]).to(base.dtype)
        # Lift the FINAL native spatial values, so the next _point sees the
        # same coordinates. Casts retain autograd and do not change zero bits.
        canonical_output_time = (spatial.double().square().sum(-1, keepdim=True)
                                 + self.radius ** 2).sqrt()
        # In exact arithmetic this is base_t+r*canonical_delta_t. Computing
        # the canonical output time from space avoids cancellation in that
        # time increment during contraction. At zero the ratio is exactly
        # one, retaining native bits and the derivative r*v_t without detach.
        time = native[..., :1] * (canonical_output_time / x[..., :1])
        result = torch.cat((time.to(base.dtype), spatial), dim=-1)
        _finite(result, "Lorentz exponential output")
        # Finite-precision outputs must remain in the same input envelope.
        # Boundary-rounding or extreme-cancellation failure is explicit;
        # no widened one-shot output allowance or hidden radial projection.
        self._point(result)
        # At w=0 increment is exactly zero, including native base roundoff.
        # There is no detached identity override or branch deleting gradients.
        return result

    def log(self, base: Tensor, point: Tensor) -> Tensor:
        self._matching(base, point, "Lorentz log endpoint")
        x, y = self._point(base), self._point(point)
        delta_spatial = y[..., 1:] - x[..., 1:]
        # Stable difference of canonical square roots, avoiding subtraction
        # of near-identical rounded time coordinates or an indefinite chord.
        delta_time = (delta_spatial * (y[..., 1:] + x[..., 1:])).sum(dim=-1)
        delta_time = delta_time / (y[..., 0] + x[..., 0])
        spatial_dot = (x[..., 1:] * delta_spatial).sum(dim=-1)
        boost_factor = spatial_dot / (self.radius * (x[..., 0] + self.radius))
        boost_factor = boost_factor - delta_time / self.radius
        # Spatial coordinates of Y after the inverse boost sending X to o.
        relative_spatial = delta_spatial + x[..., 1:] * boost_factor[..., None]
        argument_square = relative_spatial.square().sum(dim=-1) / self.radius ** 2
        coordinates = _asinhc_from_square(argument_square)[..., None] * relative_spatial
        return self._pack(base, coordinates)


__all__ = ["EuclideanGeometry", "LorentzGeometry"]

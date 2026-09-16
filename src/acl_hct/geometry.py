"""Future-sheet Lorentz geometry, time coordinate first, curvature -c.

Batch dimensions broadcast; last axis is coordinates. Supported domain is
FP32/FP64, sqrt(c)*distance from origin <= 3. This conservative engineering
domain prevents severe cancellation; it is NOT a global geometric claim.
"""
import math
import torch


def curvature(c):
    if not isinstance(c, (int, float)) or not math.isfinite(c) or c <= 0:
        raise ValueError("c must be a finite positive scalar")
    return c


def dot(x, y):
    return (x[..., 1:] * y[..., 1:]).sum(-1) - x[..., 0] * y[..., 0]


def check_point(x, c=1.):
    curvature(c)
    if x.dtype not in (torch.float32, torch.float64) or x.shape[-1] < 2:
        raise ValueError("geometry requires FP32/64 and >=2 coordinates")
    tol = 2e-4 if x.dtype == torch.float32 else 2e-11
    if not torch.isfinite(x).all() or (x[..., 0] <= 0).any():
        raise ValueError("nonfinite point or wrong Lorentz sheet")
    if (x[..., 0] * math.sqrt(c) > math.cosh(3.) + tol).any():
        raise ValueError("point outside supported scaled radius <=3")
    if not torch.allclose(c * dot(x, x), -torch.ones_like(x[..., 0]), atol=tol, rtol=tol):
        raise ValueError("point violates manifold constraint")


def origin_like(x, c=1.):
    curvature(c)
    p = torch.zeros_like(x)
    p[..., 0] = 1 / math.sqrt(c)
    return p


def tangent(p, v, c=1.):
    curvature(c)
    return v + c * dot(p, v).unsqueeze(-1) * p


def normalize(x, c=1.):
    curvature(c)
    s = -c * dot(x, x)
    if not torch.isfinite(x).all() or (s <= 0).any() or (x[..., 0] <= 0).any():
        raise ValueError("normalization requires finite future timelike vectors")
    p = x / s.sqrt().unsqueeze(-1)
    check_point(p, c)
    return p


def exp(p, v, c=1.):
    check_point(p, c)
    if not torch.isfinite(v).all():
        raise ValueError("nonfinite tangent vector")
    v = tangent(p, v, c)
    z2 = (c * dot(v, v)).clamp_min(0)
    # Clamp only the unused analytic branch; both branches have finite derivatives.
    z = z2.clamp_min(1e-8).sqrt()
    a = torch.where(z2 < 1e-6, 1 + z2 / 2 + z2.square() / 24, z.cosh())
    b = torch.where(z2 < 1e-6, 1 + z2 / 6 + z2.square() / 120, z.sinh() / z)
    q = a.unsqueeze(-1) * p + b.unsqueeze(-1) * v
    check_point(q, c)
    return q


def log(p, q, c=1.):
    check_point(p, c)
    check_point(q, c)
    # alpha-1 from displacement avoids cancellation near coincidence.
    d = q - p
    t = (c * dot(d, d) / 2).clamp_min(0)
    safe = t.clamp_min(1e-8)
    regular = torch.acosh(1 + safe) / (safe * (2 + safe)).sqrt()
    factor = torch.where(t < 1e-5, 1 - t / 3 + 2 * t.square() / 15, regular)
    return factor.unsqueeze(-1) * tangent(p, d, c)


def norm2(v):
    return dot(v, v).clamp_min(0)


def distance(p, q, c=1.):
    # Distance is nonsmooth at coincidence; geometry maps above remain smooth.
    return norm2(log(p, q, c)).sqrt()


def from_spatial(v, c=1.):
    w = torch.cat((torch.zeros_like(v[..., :1]), v), -1)
    return exp(origin_like(w, c), w, c)

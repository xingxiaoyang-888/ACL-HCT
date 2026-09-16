"""Equal-weight normalized mean and sample-only candidate corrections."""
import math
import torch
from .geometry import check_point, normalize, log, exp, tangent, norm2


def aggregate(x, c=1., weights=None):
    if x.ndim < 2 or x.shape[-2] == 0:
        raise ValueError("empty neighborhood: caller must use explicit self fallback")
    check_point(x, c)
    if weights is None:
        return normalize(x.mean(-2), c)
    if weights.shape != x.shape[:-1] or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("weights must be finite, positive and match points")
    w = weights / weights.sum(-1, keepdim=True)
    return normalize((x * w.unsqueeze(-1)).sum(-2), c)


def sample_indices(n, k, generator):
    """CPU generator, SRSWOR. Full sampling preserves canonical input order."""
    if type(n) is not int or type(k) is not int or n < 1 or k < 1:
        raise ValueError("n and k must be positive integers")
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        raise ValueError("use an explicit CPU generator for portable sampling")
    return torch.arange(n) if k >= n else torch.randperm(n, generator=generator)[:k]


def correct(sample, population_size, c=1., method="third", max_step=0.1):
    """Input contains ONLY sampled messages, actual visible N and curvature.

    Third moment assumes local equal-weight SRSWOR. Finite-population jackknife
    scales usual (k-1) log leave-one-out bias by (N-k)/N; this is a comparator,
    not an exact intrinsic unbiased estimator. No gain is guaranteed.
    """
    if sample.ndim < 2 or sample.shape[-2] == 0:
        raise ValueError("sample must have a nonempty neighborhood axis")
    k = sample.shape[-2]
    if method not in ("none", "third", "jackknife"):
        raise ValueError("unknown correction")
    if type(population_size) is not int or population_size < k or not math.isfinite(max_step) or max_step <= 0:
        raise ValueError("invalid N or max_step")
    p = aggregate(sample, c)
    info = {"fallback": False, "clipped": False, "step": 0., "method": method}
    if method == "none" or k == population_size or k < 3:
        info["fallback"] = method != "none"
        return p, info
    if method == "third":
        a = population_size**2 * (k-1)*(k-2) / (k*k*(population_size-1)*(population_size-2))
        u = log(p.unsqueeze(-2), sample, c)
        u = u - u.mean(-2, keepdim=True)
        step = c * (1-a)/(6*a) * (norm2(u).unsqueeze(-1)*u).mean(-2)
    else:
        loo = normalize((sample.sum(-2, keepdim=True)-sample)/(k-1), c)
        step = -(k-1) * (population_size-k)/population_size * log(p.unsqueeze(-2), loo, c).mean(-2)
    step = tangent(p, step, c)
    length = norm2(step).clamp_min(1e-24).sqrt()
    scale = (max_step / length).clamp_max(1.)
    info["clipped"] = bool((length > max_step).any().detach())
    info["step"] = float((norm2(step).detach().sqrt() * scale.detach()).max())
    return exp(p, scale.unsqueeze(-1)*step, c), info


def correct_batched(sample: torch.Tensor, population_size, mask: torch.Tensor,
                    c: float = 1., method: str = "third", max_step: float = .1,
                    self_points: torch.Tensor | None = None):
    """Correct padded equal-weight SRSWOR neighborhoods without per-row CPU reads.

    sample: [..., K, D], K>=1; mask: [..., K], bool, same device.
    N: integer scalar or integer tensor broadcastable to [...], actual visible
    population count before sampling. Valid rows require 1<=k<=N. Empty rows
    require N=0 and explicit self_points [..., D] (batch broadcasting allowed).
    Padding is ignored, even if nonfinite. Empty rows return self exactly.
    All diagnostic tensors have shape [...] and stay detached on the device.
    Validation uses a bounded number of batch-wide synchronizations, not zero.
    """
    if method not in ("none", "third", "jackknife"):
        raise ValueError("unknown correction")
    if not math.isfinite(max_step) or max_step <= 0:
        raise ValueError("max_step must be finite and positive")
    if sample.ndim < 2 or sample.shape[-2] < 1:
        raise ValueError("sample must have a nonempty padded neighborhood axis")
    if mask.shape != sample.shape[:-1] or mask.dtype != torch.bool or mask.device != sample.device:
        raise ValueError("mask must be boolean, on sample device and match neighborhoods")
    if type(population_size) is int:
        population_size = torch.tensor(population_size, device=sample.device)
    if not isinstance(population_size, torch.Tensor) or population_size.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("population_size must be integer")
    if population_size.device != sample.device:
        raise ValueError("population_size must be on sample device")
    try:
        n = torch.broadcast_to(population_size, sample.shape[:-2])
    except RuntimeError as error:
        raise ValueError("N must broadcast to batch shape") from error
    k = mask.sum(-1)
    empty = k == 0
    if ((n < k) | (n < 0) | (empty & (n != 0))).any():
        raise ValueError("require k<=N and N=0 for empty neighborhoods")
    # Replace padding BEFORE geometry, products or reductions; 0*NaN is NaN.
    from .geometry import origin_like
    safe = torch.where(mask.unsqueeze(-1), sample, origin_like(sample, c))
    check_point(safe, c)
    if self_points is None:
        if empty.any():
            raise ValueError("empty neighborhoods require explicit self_points")
        self_points = origin_like(sample[..., 0, :], c)
    else:
        if (self_points.shape[-1:] != sample.shape[-1:] or
                self_points.dtype != sample.dtype or self_points.device != sample.device):
            raise ValueError("self_points coordinates, dtype and device must match sample")
        try:
            self_points = torch.broadcast_to(self_points, sample.shape[:-2] + sample.shape[-1:])
        except RuntimeError as error:
            raise ValueError("self_points must broadcast to batch shape") from error
        check_point(self_points, c)
    total = torch.where(mask.unsqueeze(-1), safe, torch.zeros_like(safe)).sum(-2)
    mean = total / k.clamp_min(1).unsqueeze(-1)
    p = normalize(torch.where(empty.unsqueeze(-1), self_points, mean), c)
    p = torch.where(empty.unsqueeze(-1), self_points, p)
    active = (k >= 3) & (k < n) & (method != "none")
    # Inactive rows use harmless finite coefficients; final output is exact p.
    kk = k.clamp_min(3).to(sample.dtype)
    nn = n.clamp_min(3).to(sample.dtype)
    if method == "third":
        u = log(p.unsqueeze(-2), safe, c)
        u = torch.where(mask.unsqueeze(-1), u, torch.zeros_like(u))
        center = u.sum(-2) / k.clamp_min(1).unsqueeze(-1)
        u = torch.where(mask.unsqueeze(-1), u-center.unsqueeze(-2), torch.zeros_like(u))
        moment = (norm2(u).unsqueeze(-1)*u).sum(-2) / k.clamp_min(1).unsqueeze(-1)
        a = nn.square() * (kk-1)*(kk-2) / (kk.square()*(nn-1)*(nn-2))
        step = (c*(1-a)/(6*a)).unsqueeze(-1)*moment
    elif method == "jackknife":
        loo_mean = (total.unsqueeze(-2)-safe) / (k-1).clamp_min(1)[..., None, None]
        use = mask & active.unsqueeze(-1)
        loo = normalize(torch.where(use.unsqueeze(-1), loo_mean, p.unsqueeze(-2)), c)
        offsets = torch.where(use.unsqueeze(-1), log(p.unsqueeze(-2), loo, c), torch.zeros_like(safe))
        step = (-(kk-1)*(nn-kk)/nn).unsqueeze(-1) * offsets.sum(-2) / k.clamp_min(1).unsqueeze(-1)
    else:
        step = torch.zeros_like(p)
    step = tangent(p, torch.where(active.unsqueeze(-1), step, torch.zeros_like(step)), c)
    squared = norm2(step)
    length = squared.clamp_min(1e-24).sqrt()
    scale = (max_step/length).clamp_max(1.)
    out = exp(p, scale.unsqueeze(-1)*step, c)
    out = torch.where(active.unsqueeze(-1), out, p)
    diagnostics = {
        "fallback": (empty | ((method != "none") & ~active)),
        "empty": empty, "full": (~empty & (k == n)),
        "small_sample": (~empty & (k < 3)),
        "clipped": active & (length > max_step),
        "raw_step": squared.detach().sqrt(),
        "step": squared.detach().sqrt()*scale.detach(), "k": k, "N": n,
    }
    return out, {key: value.detach() for key, value in diagnostics.items()}

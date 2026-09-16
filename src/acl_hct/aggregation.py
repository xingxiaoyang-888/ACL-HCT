"""Equal-weight normalized mean and sample-only candidate corrections."""
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
    if not isinstance(n, int) or not isinstance(k, int) or n < 1 or k < 1:
        raise ValueError("n and k must be positive integers")
    if generator.device.type != "cpu":
        raise ValueError("use an explicit CPU generator for portable sampling")
    return torch.arange(n) if k >= n else torch.randperm(n, generator=generator)[:k]


def correct(sample, population_size, c=1., method="third", max_step=0.1):
    """Input contains ONLY sampled messages, actual visible N and curvature.

    Third moment assumes local equal-weight SRSWOR. Finite-population jackknife
    scales usual (k-1) log leave-one-out bias by (N-k)/N; this is a comparator,
    not an exact intrinsic unbiased estimator. No gain is guaranteed.
    """
    k = sample.shape[-2]
    if method not in ("none", "third", "jackknife"):
        raise ValueError("unknown correction")
    if not isinstance(population_size, int) or population_size < k or max_step <= 0:
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
    info["step"] = float((length * scale).max().detach())
    return exp(p, scale.unsqueeze(-1)*step, c), info

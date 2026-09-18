"""Evaluation-only FP64 native-ball geometry, without an origin-radius cap."""
import math
import numpy as np


def _curvature(c):
    if type(c) not in (int, float) or not math.isfinite(c) or c <= 0:
        raise ValueError('finite positive fixed curvature required')
    return float(c)


def _point(p, c):
    p = np.asarray(p)
    if p.dtype not in (np.dtype('float32'), np.dtype('float64')) or p.ndim < 1 or p.shape[-1] < 1:
        raise ValueError('floating point final-coordinate axis required')
    p = p.astype(np.float64, copy=False)
    if not np.isfinite(p).all():
        raise ValueError('finite native ball point required')
    square = np.sum(p * p, axis=-1)
    interior = 1 - c * square
    if np.any(interior <= 0):
        raise ValueError('native point must be interior; no clipping or reprojection')
    return p, interior


def distance_ball_fp64(p, q, c=1.):
    """Stable asinh form of geodesic distance, exact zero at coincidence."""
    c = _curvature(c); p, a = _point(p, c); q, b = _point(q, c)
    if p.shape[-1] != q.shape[-1]:
        raise ValueError('matching coordinate dimensions required')
    delta = q - p
    norm = np.hypot.reduce(delta, axis=-1)
    result = 2 / math.sqrt(c) * np.arcsinh(math.sqrt(c) * norm / np.sqrt(a * b))
    if not np.isfinite(result).all():
        raise ValueError('nonfinite FP64 distance')
    return result


def log_ball_orthonormal_fp64(p, q, c=1.):
    """Log at fixed p, expressed in its orthonormal coordinate basis.

    Its norm equals D(p,q). This is lambda_p times the chart-coordinate Log,
    not a vector at the origin and not a vector shared between different p's.
    Normalize the Mobius numerator directly; the positive denominator cancels.
    """
    c = _curvature(c); p, a = _point(p, c); q, _ = _point(q, c)
    if p.shape[-1] != q.shape[-1]:
        raise ValueError('matching coordinate dimensions required')
    delta = q - p
    delta_square = np.sum(delta * delta, axis=-1)
    direction = a[..., None] * delta - c * delta_square[..., None] * p
    norm = np.hypot.reduce(direction, axis=-1)
    coincident = np.all(delta == 0, axis=-1)
    if np.any((norm == 0) & ~coincident):
        raise ValueError('undefined noncoincident FP64 direction')
    length = distance_ball_fp64(p, q, c)
    scale = np.divide(length, norm, out=np.zeros_like(length), where=~coincident)
    result = direction * scale[..., None]
    if not np.isfinite(result).all():
        raise ValueError('nonfinite orthonormal Log')
    return result


def mean_error_statistics(errors):
    """Per-node conditional moments; finite-R mean norm alone is not bias proof.

    Input [R,N,D] uses the SAME base/basis for each node over all repeats.
    Signed bias_squared_unbiased estimates ||E e_i||^2, retaining negatives.
    Spatially distinct node vectors are never averaged into one tangent vector.
    """
    errors = np.asarray(errors)
    if errors.dtype != np.float64 or errors.ndim != 3 or errors.shape[0] < 2 or not np.isfinite(errors).all():
        raise ValueError('finite FP64 [R>=2,N,D] errors required')
    r = len(errors); mean = errors.mean(axis=0); residual = errors - mean
    centered_sum = np.sum(residual * residual, axis=(0, 2))
    mean_square = np.sum(mean * mean, axis=-1)
    return {'mean_vector_per_node': mean, 'squared_mean_norm_per_node': mean_square,
            'MSE_per_node': np.mean(np.sum(errors * errors, axis=-1), axis=0),
            'centered_variance_R_per_node': centered_sum / r,
            'sample_variance_trace_per_node': centered_sum / (r - 1),
            'bias_squared_unbiased_per_node': mean_square - centered_sum / (r * (r - 1)),
            'repeat_count': r}

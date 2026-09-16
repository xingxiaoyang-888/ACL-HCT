"""Finite-population checks, not experiments with trained GNNs.

All Lorentz calculations are float64. Exhaustive subsets remove Monte Carlo
error. Correction uses only the sampled points, N, k and curvature.
"""
from itertools import combinations
from pathlib import Path
import json
import numpy as np

BASE = Path(__file__).resolve().parent

def dot(x, y):
    return np.sum(x * y, axis=-1) - 2 * x[..., 0] * y[..., 0]

def project(v, c=1.):
    return v / np.sqrt(-c * dot(v, v))[..., None]

def tangent(p, v, c=1.):
    return v + c * dot(p, v)[..., None] * p

def exp(p, v, c=1.):
    v = tangent(p, v, c)
    norm = np.sqrt(np.maximum(dot(v, v), 0.))
    z = np.sqrt(c) * norm
    factor = np.ones_like(z)
    np.divide(np.sinh(z), z, out=factor, where=z > 1e-12)
    return np.cosh(z)[..., None] * p + factor[..., None] * v

def log(p, q, c=1.):
    a = np.maximum(-c * dot(p, q), 1.)
    w = q - a[..., None] * p
    den = np.sqrt(np.maximum(a*a-1, 0.))
    factor = np.ones_like(a)
    np.divide(np.arccosh(a), den, out=factor, where=den > 1e-10)
    return factor[..., None] * w

def from_tangent(v, c=1.):
    v = np.asarray(v, dtype=np.float64)
    p = np.zeros(v.shape[-1]+1)
    p[0] = 1/np.sqrt(c)
    return exp(p, np.concatenate([np.zeros((*v.shape[:-1], 1)), v], axis=-1), c)

def norm(v):
    return float(np.sqrt(max(float(dot(v, v)), 0.)))

def bias_prediction(x, n, k, c=1.):
    """Second order formula, using x to estimate both center and covariance.

    When x is the population this is an oracle diagnostic; when x is a
    sample of length k this is the implementable plug-in estimate.
    """
    mu = np.mean(x, axis=0)
    s2 = -c * dot(mu, mu)
    p = mu / np.sqrt(s2)
    if len(x) < 2 or n == k:
        return p, np.zeros_like(p)
    d = x - mu
    # S J p without constructing an O(d^2) covariance matrix.
    v = np.sum(d * dot(d, p)[:, None], axis=0) / (len(x)-1)
    b = c * (1/k-1/n) / s2 * tangent(p, v, c)
    return p, b

def third_moment_correction(x, n, c=1.):
    """Candidate local small-spread correction. k >= 3 only.

    A is the exact SRSWOR shrink factor of a Euclidean sample third central
    moment. Transferring it to the curved space is a local expansion, not
    an exact unbiasedness guarantee for arbitrary neighborhoods.
    """
    k = len(x)
    p = project(x.mean(0), c)
    if k < 3 or k == n:
        return p
    a = n*n*(k-1)*(k-2)/(k*k*(n-1)*(n-2))
    u = log(p, x, c)
    u = u - u.mean(axis=0)
    m3 = np.mean(dot(u, u)[:, None] * u, axis=0)
    step = c*(1-a)/(6*a) * m3
    return exp(p, step, c)

def evaluate(name, x, k, c=1.):
    n = len(x)
    full, prediction = bias_prediction(x, n, k, c)
    logs, corrected_logs, third_logs, radii, correction_norms = [], [], [], [], []
    origin = np.zeros_like(full)
    origin[0] = 1/np.sqrt(c)
    for ix in combinations(range(n), k):
        sample, b = bias_prediction(x[list(ix)], n, k, c)
        corrected = exp(sample, -b, c)
        logs.append(log(full, sample, c))
        corrected_logs.append(log(full, corrected, c))
        third_logs.append(log(full, third_moment_correction(x[list(ix)], n, c), c))
        radii.append(norm(log(origin, sample, c)))
        correction_norms.append(norm(b))
    logs, corrected_logs, third_logs = np.array(logs), np.array(corrected_logs), np.array(third_logs)
    b_actual, b_corr = logs.mean(0), corrected_logs.mean(0)
    rf = norm(log(origin, full, c))
    er = -log(full, origin, c) / rf if rf > 1e-10 else None
    return dict(name=name, n=n, k=k, subsets=len(logs), curvature=-c,
        full_radius=rf, mean_sample_radius=float(np.mean(radii)),
        intrinsic_bias_norm=norm(b_actual),
        radial_bias=None if er is None else float(dot(b_actual, er)),
        second_order_bias_norm=norm(prediction),
        second_order_error=norm(prediction-b_actual),
        corrected_bias_norm=norm(b_corr),
        mse=float(np.mean(dot(logs, logs))),
        corrected_mse=float(np.mean(dot(corrected_logs, corrected_logs))),
        third_corrected_bias=norm(third_logs.mean(0)),
        third_corrected_mse=float(np.mean(dot(third_logs, third_logs))),
        max_step=max(correction_norms))

def geodesic(rs, c=1.):
    return from_tangent(np.array(rs, dtype=float)[:, None], c)

def main():
    rows = []
    for name, rs in [('inward', [0, 0, 3]),
                     ('outward', [1, 4, 4]),
                     ('symmetric_nonorigin', [1, 2, 3]),
                     ('zero_bias_positive_radius', [-1, 1])]:
        for k in range(1, len(rs)+1):
            rows.append(evaluate(name, geodesic(rs), k))
    r8 = [0.0, .1, .2, .3, .4, .5, 1.2, 1.6]
    for scale in [.2, .5, 1., 2.]:
        for k in [2, 3, 4, 6, 7, 8]:
            rows.append(evaluate('skew8_scale_'+str(scale), geodesic(1+scale*np.array(r8)), k))
    # Genuine 2D asymmetric neighborhood, not confined to one geodesic.
    v = np.array([[.1,.1], [.2,.0], [.4,.2], [.2,-.1], [.8,.4],
                  [.9,.1], [1.2,.3], [.3,-.2]], dtype=float)
    for scale in [.25, .5, 1., 2.]:
        for k in [2, 3, 4, 6, 7, 8]:
            rows.append(evaluate('2d_scale_'+str(scale), from_tangent(v*scale), k))
    # Curvature sweep, fixed intrinsic tangent inputs.
    for c in [.0001, .01, .1, 1., 4.]:
        rows.append(evaluate('curvature_sweep', from_tangent(v, c), 4, c))
    (BASE/'numerical_results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    for r in rows:
        print(f"{r['name']:29s} k={r['k']} bias={r['intrinsic_bias_norm']:.8f} "
              f"pred={r['second_order_bias_norm']:.8f} corrected={r['corrected_bias_norm']:.8f} "
              f"third={r['third_corrected_bias']:.8f} "
              f"MSE={r['mse']:.8f}->{r['corrected_mse']:.8f}->{r['third_corrected_mse']:.8f}")

if __name__ == '__main__':
    main()

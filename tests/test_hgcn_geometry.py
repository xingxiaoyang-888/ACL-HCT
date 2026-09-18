import itertools
import math
import mpmath as mp
import numpy as np
import pytest

from acl_hct.hgcn_geometry import distance_ball_fp64, log_ball_orthonormal_fp64, mean_error_statistics


def high_precision(p, q, c):
    with mp.workdps(80):
        c = mp.mpf(c); p = list(map(mp.mpf, p.tolist())); q = list(map(mp.mpf, q.tolist()))
        delta = [y - x for x, y in zip(p, q)]
        a = 1 - c * sum(x*x for x in p); b = 1 - c * sum(y*y for y in q)
        square = sum(x*x for x in delta)
        d = 2/mp.sqrt(c) * mp.asinh(mp.sqrt(c*square)/mp.sqrt(a*b))
        v = [a*z-c*square*x for x, z in zip(p, delta)]; n = mp.sqrt(sum(x*x for x in v))
        return float(d), np.array([float(x*d/n) if n else 0. for x in v])


@pytest.mark.parametrize('c', [1., .7])
@pytest.mark.parametrize('dim', [2, 16, 128])
def test_80_digit_reference_high_radius_small_displacement_angles_and_coincidence(c, dim):
    rng = np.random.default_rng(77)
    for a, b in ((0., 0.), (0., .25), (.2, .4), (.95, .996), (.996, .996), (.99999, .99999)):
        p = rng.normal(size=dim); p *= a/(np.linalg.norm(p)*math.sqrt(c))
        q = rng.normal(size=dim); q *= b/(np.linalg.norm(q)*math.sqrt(c))
        d, t = high_precision(p, q, c)
        assert abs(distance_ball_fp64(p, q, c) - d) < 1e-10
        np.testing.assert_allclose(log_ball_orthonormal_fp64(p, q, c), t, atol=1e-10, rtol=1e-10)
    p = np.zeros(dim); p[0] = .996/math.sqrt(c); q = p.copy(); q[-1] += 1e-12
    for p, q in ((p, p), (p, q), (p, -p)):
        d, t = high_precision(p, q, c)
        actual = log_ball_orthonormal_fp64(p, q, c)
        np.testing.assert_allclose(actual, t, atol=1e-10, rtol=1e-10)
        assert abs(np.linalg.norm(actual) - d) < 1e-10
        assert abs(distance_ball_fp64(p, q, c) - distance_ball_fp64(q, p, c)) < 1e-10


def test_origin_orthonormal_factor_two_broadcast_and_no_old_radius_guard():
    points = np.array([[0., 0.], [.2, 0.], [.996, 0.]], dtype=np.float32)
    radius = 2*np.arctanh(points.astype(np.float64)[:, 0])
    np.testing.assert_allclose(distance_ball_fp64(np.zeros(2), points), radius, atol=1e-14, rtol=1e-14)
    result = log_ball_orthonormal_fp64(np.zeros(2), points)
    np.testing.assert_allclose(result[:, 0], radius, atol=1e-14, rtol=1e-14)
    assert radius[-1] > 3 and result.dtype == np.float64


@pytest.mark.parametrize('point', [np.array([1., 0.]), np.array([np.nan, 0.]), np.array([np.inf, 0.])])
def test_invalid_native_points_fail_without_repair(point):
    with pytest.raises(ValueError):
        distance_ball_fp64(np.zeros(2), point)


def test_squared_mean_is_upward_but_signed_bias_estimator_is_unbiased_by_enumeration():
    mu = np.array([.3, -.1]); noise = np.array([.8, .2]); estimates = []; raw = []
    for signs in itertools.product((-1, 1), repeat=2):
        errors = np.stack([mu + s*noise for s in signs])[:, None, :]
        moment = mean_error_statistics(errors)
        estimates.append(moment['bias_squared_unbiased_per_node'][0]); raw.append(moment['squared_mean_norm_per_node'][0])
        np.testing.assert_allclose(moment['MSE_per_node'], moment['centered_variance_R_per_node'] + moment['squared_mean_norm_per_node'], atol=1e-14, rtol=0)
    assert min(estimates) < 0  # Must not clip a negative unbiased estimate.
    assert abs(np.mean(estimates) - np.dot(mu, mu)) < 1e-14
    assert np.mean(raw) > np.dot(mu, mu)


def test_error_moments_require_multiple_repeats_and_keep_bases_per_node():
    with pytest.raises(ValueError):
        mean_error_statistics(np.zeros((1, 2, 3)))
    errors = np.array([[[1., 0.], [0., 2.]], [[1., 0.], [0., 2.]]])
    result = mean_error_statistics(errors)
    assert result['mean_vector_per_node'].shape == (2, 2)
    assert np.array_equal(result['bias_squared_unbiased_per_node'], np.array([1., 4.]))

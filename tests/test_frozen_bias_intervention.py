import builtins
import copy
import hashlib
import inspect

import pytest
import torch

from acl_hct.frozen_bias_intervention import FrozenBiasIntervention, transport_from_origin
from acl_hct.frozen_fixture import synthetic_fixture
from acl_hct.frozen_forward import FrozenForward, PlanStreams
from acl_hct.frozen_stats import promote_points
from acl_hct.geometry import check_point, dot, exp, from_spatial, log, norm2, origin_like, tangent


def fixture(c=1.):
    base = from_spatial(torch.tensor([[.2, -.1], [-.3, .15], [.1, .4], [0., 0.]], dtype=torch.float64), c)
    generator = torch.Generator().manual_seed(17)
    bias = tangent(base, torch.randn(base.shape, dtype=torch.float64, generator=generator) * .025, c)
    bias[1] = 0
    error = tangent(base, torch.randn(base.shape, dtype=torch.float64, generator=generator) * .045, c)
    return base, bias, exp(base, error, c)


def adapter(base, bias, c=1., **kwargs):
    options = dict(calibration_namespace='synthetic-calibration',
                   evaluation_namespace='synthetic-evaluation',
                   direction_namespace='synthetic-unrelated-direction', direction_seed=73, c=c)
    options.update(kwargs)
    return FrozenBiasIntervention(base, bias, **options)


def apply(intervention, sample):
    return intervention.apply(sample, evaluation_namespace='synthetic-evaluation')


@pytest.mark.parametrize('c', [.5, 1., 2.])
def test_transport_preserves_metric_and_matches_geodesic_log_formula(c):
    base, _, _ = fixture(c)
    generator = torch.Generator().manual_seed(119)
    spatial = torch.randn((len(base), 2), dtype=torch.float64, generator=generator) * .2
    another = torch.randn(spatial.shape, dtype=torch.float64, generator=generator) * .2
    moved = transport_from_origin(base, spatial, c)
    moved_another = transport_from_origin(base, another, c)
    torch.testing.assert_close(dot(base, moved), torch.zeros(len(base), dtype=torch.float64), atol=2e-16, rtol=0)
    torch.testing.assert_close(dot(moved, moved), spatial.square().sum(-1), atol=2e-16, rtol=2e-13)
    torch.testing.assert_close(dot(moved, moved_another), (spatial * another).sum(-1), atol=2e-16, rtol=2e-13)
    original = torch.cat((torch.zeros_like(spatial[:, :1]), spatial), dim=-1)
    assert torch.equal(moved[-1], original[-1])  # Origin-to-origin remains exact.
    origin = origin_like(base, c)
    backward = moved + (dot(origin, moved) / (1 / c - dot(base, origin)))[:, None] * (base + origin)
    torch.testing.assert_close(backward, original, atol=2e-16, rtol=2e-13)
    # A separate log/distance construction away from coincidence checks the formula.
    outward = log(origin[:-1], base[:-1], c)
    inward = log(base[:-1], origin[:-1], c)
    expected = original[:-1] - (dot(outward, original[:-1]) / norm2(outward))[:, None] * (outward + inward)
    torch.testing.assert_close(moved[:-1], expected, atol=3e-15, rtol=2e-13)


@pytest.mark.parametrize('change', ['dtype', 'shape', 'nonfinite'])
def test_transport_rejects_invalid_spatial_vectors(change):
    base, _, _ = fixture()
    spatial = torch.ones((len(base), 2), dtype=torch.float64)
    if change == 'dtype':
        spatial = spatial.float()
    elif change == 'shape':
        spatial = spatial[:-1]
    else:
        spatial[0, 0] = float('nan')
    with pytest.raises(ValueError):
        transport_from_origin(base, spatial)


@pytest.mark.parametrize('c', [.5, 1., 2.])
def test_maps_constraints_equal_norm_and_reconstruction(c):
    base, bias, sample = fixture(c)
    intervention = adapter(base, bias, c)
    result = apply(intervention, sample)
    torch.testing.assert_close(result['reconstructed_S'], sample, atol=3e-14, rtol=2e-13)
    torch.testing.assert_close(result['offsets']['O'], log(base, sample, c) - bias, atol=1e-15, rtol=1e-13)
    torch.testing.assert_close(norm2(intervention.control), norm2(bias), atol=2e-17, rtol=2e-13)
    torch.testing.assert_close(dot(base, intervention.control), torch.zeros(len(base), dtype=torch.float64), atol=1e-14, rtol=0)
    for name in ('S', 'O', 'Q'):
        check_point(result['points'][name], c)
        assert not result['points'][name].requires_grad
        torch.testing.assert_close(log(base, result['points'][name], c), result['offsets'][name], atol=3e-14, rtol=2e-13)
    # Zero amplitude returns this node's sample, even when its sample differs from F.
    assert not torch.equal(sample[1], base[1])
    assert torch.equal(result['points']['O'][1], sample[1])
    assert torch.equal(result['points']['Q'][1], sample[1])


def test_zero_step_is_exact_identity_for_all_nodes():
    base, bias, sample = fixture()
    bias.zero_()
    intervention = adapter(base, bias)
    result = apply(intervention, sample)
    assert intervention.control.eq(0).all()
    for name in ('S', 'O', 'Q'):
        assert torch.equal(result['points'][name], sample)
        assert torch.equal(result['offsets'][name], result['offsets']['S'])


def test_centered_residuals_and_cross_node_covariance_remain_in_fixed_tangents():
    base, bias, _ = fixture()
    intervention = adapter(base, bias)
    generator = torch.Generator().manual_seed(18)
    errors = tangent(base, torch.randn((7,) + base.shape, dtype=torch.float64, generator=generator) * .05)
    # Complete synthetic fields only; seven is a fixture count, not pilot R.
    shifts = [intervention.offsets(exp(base, row), evaluation_namespace='synthetic-evaluation') for row in errors]
    original = torch.stack([row['S'] for row in shifts])
    centered = original - original.mean(0)
    covariance = centered.flatten(1).T @ centered.flatten(1) / len(errors)
    assert covariance[1, 4].abs() > 1e-6
    for name in ('O', 'Q'):
        shifted = torch.stack([row[name] for row in shifts])
        shifted_centered = shifted - shifted.mean(0)
        torch.testing.assert_close(shifted_centered, centered, atol=2e-17, rtol=2e-13)
        flat = shifted_centered.flatten(1)
        torch.testing.assert_close(flat.T @ flat / len(errors), covariance, atol=3e-18, rtol=2e-13)
    # These are fixed-tangent algebra checks, not a manifold variance assertion.


def test_direction_reproducibility_bias_orientation_and_rng_isolation():
    base, bias, sample = fixture()
    before = torch.get_rng_state().clone()
    intervention = adapter(base, bias)
    assert torch.equal(torch.get_rng_state(), before)
    repeat = adapter(base, bias)
    reverse = adapter(base, -bias)
    doubled = adapter(base, 2 * bias)
    other = adapter(base, bias, direction_seed=74)
    assert torch.equal(intervention.control, repeat.control)
    assert torch.equal(intervention.control, reverse.control)
    torch.testing.assert_close(doubled.control, 2 * intervention.control, atol=1e-16, rtol=1e-13)
    assert not torch.equal(intervention.control, other.control)
    q = intervention.control
    for _ in range(3):
        apply(intervention, sample)
        assert torch.equal(q, intervention.control)
    assert torch.equal(torch.get_rng_state(), before)


def test_clones_isolate_inputs_outputs_and_metadata():
    base, bias, sample = fixture()
    intervention = adapter(base, bias)
    expected = apply(intervention, sample)
    base.fill_(999)
    bias.fill_(999)
    intervention.base.zero_()
    intervention.bias.zero_()
    intervention.control.zero_()
    metadata = intervention.metadata
    metadata['base']['shape'][0] = 0
    expected_copy = copy.deepcopy(expected)
    expected['points']['O'].zero_()
    current = apply(intervention, sample)
    assert torch.equal(current['points']['O'], expected_copy['points']['O'])
    assert current['metadata']['base']['shape'][0] == len(sample)
    stored = intervention.base.contiguous().numpy().tobytes()
    assert current['metadata']['base']['raw_sha256'] == hashlib.sha256(stored).hexdigest()


@pytest.mark.parametrize('change', [
    {'evaluation_namespace': 'synthetic-calibration'},
    {'direction_namespace': 'synthetic-evaluation'},
    {'direction_namespace': 'synthetic-calibration'},
    {'calibration_namespace': ''}, {'evaluation_namespace': ' x '},
    {'direction_seed': True}, {'direction_seed': -1},
])
def test_namespace_and_seed_contract_rejects_aliases(change):
    base, bias, _ = fixture()
    with pytest.raises(ValueError):
        adapter(base, bias, **change)


@pytest.mark.parametrize('change', ['base_dtype', 'bias_dtype', 'bias_shape', 'bias_nan', 'bias_not_tangent', 'base_invalid'])
def test_invalid_calibration_fields_are_rejected(change):
    base, bias, _ = fixture()
    if change == 'base_dtype':
        base = base.float()
    elif change == 'bias_dtype':
        bias = bias.float()
    elif change == 'bias_shape':
        bias = bias[:-1]
    elif change == 'bias_nan':
        bias[0, 1] = float('nan')
    elif change == 'bias_not_tangent':
        bias[0, 0] += 1
    else:
        base[0, 0] += .5
    with pytest.raises(ValueError):
        adapter(base, bias)


@pytest.mark.parametrize('change', ['dtype', 'shape', 'nan', 'off_manifold', 'namespace'])
def test_invalid_evaluation_is_rejected(change):
    base, bias, sample = fixture()
    intervention = adapter(base, bias)
    namespace = 'synthetic-evaluation'
    if change == 'dtype':
        sample = sample.float()
    elif change == 'shape':
        sample = sample[:-1]
    elif change == 'nan':
        sample[0, 1] = float('nan')
    elif change == 'off_manifold':
        sample[0, 0] += .5
    else:
        namespace = 'synthetic-calibration'
    with pytest.raises(ValueError):
        intervention.apply(sample, evaluation_namespace=namespace)


def test_out_of_domain_is_a_failure_without_clipping_or_redraw():
    base = from_spatial(torch.zeros(2, 2, dtype=torch.float64))
    bias = torch.tensor([[0., 3.2, 0.], [0., 0., 0.]], dtype=torch.float64)
    intervention = adapter(base, bias)
    before = intervention.metadata
    with pytest.raises(ValueError, match='outside supported'):
        apply(intervention, base)
    assert intervention.metadata == before
    assert torch.equal(intervention.bias, bias)


def test_no_label_or_disk_interface_and_no_inferred_provenance():
    base, bias, sample = fixture()
    original_open = builtins.open
    def forbidden(*args, **kwargs):
        raise AssertionError('adapter must not read labels or files')
    try:
        builtins.open = forbidden
        intervention = adapter(base, bias)
        result = apply(intervention, sample)
    finally:
        builtins.open = original_open
    for method in (FrozenBiasIntervention, FrozenBiasIntervention.apply, FrozenBiasIntervention.offsets):
        assert not any('label' in name or 'query' in name for name in inspect.signature(method).parameters)
    with pytest.raises(TypeError):
        intervention.apply(sample, evaluation_namespace='synthetic-evaluation', labels=torch.ones(len(base)))
    assert result['metadata']['calibration_provenance_verified_by_adapter'] is False


def test_synthetic_f4_pair_and_third_paths_share_plans_without_adapter_rng_use():
    model, features, view = synthetic_fixture()
    forward = FrozenForward(model, features, view.neighbors, 128)
    evaluation = PlanStreams(73, 'synthetic-evaluation')
    control = PlanStreams(73, 'synthetic-evaluation')
    calibration = PlanStreams(73, 'synthetic-calibration')
    assert set(calibration.seeds).isdisjoint(evaluation.seeds)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    plans = evaluation.draw(view.neighbors, 4)
    assert plans == control.draw(view.neighbors, 4)
    with torch.no_grad():
        paired = forward.paired(plans)
        sampled, _, _ = model.encode(features, view.neighbors, plans, max_padded_messages=128)
        corrected, diagnostics, _ = model.encode(features, view.neighbors, plans, method='third', max_padded_messages=128)
    assert torch.equal(sampled, paired['S/S']['points'])
    assert len(diagnostics) == 2 and torch.isfinite(corrected).all()
    base, _ = promote_points(forward.reference['output'])
    sample, _ = promote_points(sampled)
    generator = torch.Generator().manual_seed(91)
    bias = tangent(base, torch.randn(base.shape, dtype=torch.float64, generator=generator) * .002)
    intervention = adapter(base, bias)
    result = apply(intervention, sample)
    assert result['points']['O'].shape == sampled.shape
    assert evaluation.draw(view.neighbors, 4) == control.draw(view.neighbors, 4)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())

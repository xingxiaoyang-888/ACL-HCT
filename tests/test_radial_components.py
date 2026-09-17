"""FP64 projection/component and native identity checks, CPU synthetic only."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from acl_hct.geometry import distance, dot, exp, from_spatial, log, norm2, tangent
from acl_hct.radial_components import RadialBiasIntervention


def fixture():
    spatial = torch.tensor([[.5, 0.], [0., 0.], [.2, .3], [-.3, .2]], dtype=torch.float64)
    base = from_spatial(spatial)
    raw = torch.tensor([[0., .04, .01], [0., .02, -.03], [0., .03, .02], [0., -.04, .015]], dtype=torch.float64)
    bias = tangent(base, raw)
    floor = torch.full((4,), 1e-5, dtype=torch.float64)
    return base, bias, floor


def numpy_dot(a, b):
    return np.sum(a[..., 1:] * b[..., 1:], axis=-1) - a[..., 0] * b[..., 0]


def test_independent_numpy_projection_and_defined_norm_identities():
    base, bias, floor = fixture()
    adapter = RadialBiasIntervention(base, bias, floor, 0)
    x = base.numpy()[1:]
    root = base.numpy()[0]
    alpha = -numpy_dot(x, root)
    # Independent closed form: outward=(alpha*p-r)/sqrt(alpha^2-1), c=1.
    unit = (alpha[:, None] * x - root) / np.sqrt(alpha * alpha - 1)[:, None]
    radial = numpy_dot(bias.numpy()[1:], unit)[:, None] * unit
    fields = adapter.fields
    np.testing.assert_allclose(adapter.outward_unit.numpy()[1:], unit, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(fields['R'].numpy()[1:], radial, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(fields['T'].numpy()[1:], bias.numpy()[1:] - radial, atol=2e-14, rtol=2e-14)
    defined = adapter.defined
    assert defined.tolist() == [False, True, True, True]
    assert torch.allclose(norm2(adapter.outward_unit)[defined], torch.ones(3, dtype=torch.float64), atol=1e-12)
    assert dot(base, fields['R']).abs().max() < 1e-12
    assert dot(base, fields['T']).abs().max() < 1e-12
    assert dot(adapter.outward_unit, fields['T'])[defined].abs().max() < 1e-12
    assert torch.allclose(norm2(fields['R'])[defined] + norm2(fields['T'])[defined], norm2(bias)[defined], atol=1e-12)
    assert torch.equal(adapter.unapplied_bias[0], bias[0])
    assert torch.allclose(fields['R'] + fields['T'] + adapter.unapplied_bias, bias, atol=1e-12)


@pytest.mark.parametrize('component', ['radial', 'orthogonal', 'zero'])
def test_pure_component_zero_counterpart_and_exact_native_identity(component):
    from acl_hct.frozen_recovery import cast_for_head, promoted
    base = from_spatial(torch.tensor([[.5, 0.], [0., 0.]], dtype=torch.float64))
    bias = torch.zeros_like(base)
    if component == 'radial':
        bias[1, 1] = .03
    elif component == 'orthogonal':
        bias[1, 2] = .03
    floor = torch.zeros(2, dtype=torch.float64)
    adapter = RadialBiasIntervention(base, bias, floor, 0)
    native = from_spatial(torch.tensor([[.4, .1], [.1, .07]], dtype=torch.float64)).float()
    sampled = promoted(native, 1.)
    moved = adapter.apply(sampled)
    cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs/e2_frozen_recovery_pilot.json').read_bytes())
    unchanged = ('T',) if component == 'radial' else ('R',) if component == 'orthogonal' else ('R', 'T')
    for name in ('R', 'T'):
        returned, audit = cast_for_head(moved['points'][name], native, sampled,
                                         moved['removed_fields'][name], floor, 1., cfg['numeric_proposal'])
        assert torch.equal(returned[0], native[0])  # undefined original root
        if name in unchanged:
            assert torch.equal(moved['removed_fields'][name], torch.zeros_like(bias))
            assert torch.equal(moved['points'][name], sampled)
            assert torch.equal(returned, native)
        assert audit['zero_native_identity'] is True


def test_direction_resolution_near_coincidence_and_boundary_retains_unapplied_bias():
    base = from_spatial(torch.tensor([[.5, 0.], [.5 + 1e-12, 0.], [0., 0.]], dtype=torch.float64))
    bias = tangent(base, torch.tensor([[0., .02, 0.], [0., .03, 0.], [0., .04, 0.]], dtype=torch.float64))
    floor = torch.zeros(3, dtype=torch.float64)
    floor[2] = distance(base[2], base[0])
    adapter = RadialBiasIntervention(base, bias, floor, 0)
    assert not adapter.defined.any()
    sample = exp(base, tangent(base, torch.full_like(base, .005)))
    moved = adapter.apply(sample)
    assert torch.equal(adapter.unapplied_bias, bias)
    for name in ('R', 'T'):
        assert torch.equal(moved['points'][name], sample)
        assert torch.equal(moved['removed_fields'][name], torch.zeros_like(bias))


def test_fixed_sample_exp_matches_each_removed_offset_and_finite_T_can_change_radius():
    base, bias, floor = fixture()
    adapter = RadialBiasIntervention(base, bias, floor, 0)
    sample = exp(base, tangent(base, torch.tensor([[0., .01, .01]] * 4, dtype=torch.float64)))
    before = base.clone(), bias.clone(), floor.clone(), sample.clone()
    moved = adapter.apply(sample)
    defined = adapter.defined
    for name in ('R', 'T'):
        expected = exp(base[defined], (log(base, sample) - adapter.fields[name])[defined])
        assert torch.allclose(moved['points'][name][defined], expected, atol=1e-13, rtol=1e-13)
        assert torch.equal(moved['points'][name][~defined], sample[~defined])
    assert (distance(base[0], moved['points']['T']) - distance(base[0], sample))[defined].abs().max() > 1e-8
    for actual, prior in zip((base, bias, floor, sample), before):
        assert torch.equal(actual, prior)
    # Returned fields and metadata cannot mutate later application.
    fields = adapter.fields
    fields['R'].zero_()
    metadata = adapter.metadata
    metadata['defined_count'] = -1
    repeated = adapter.apply(sample)
    assert torch.equal(repeated['points']['R'], moved['points']['R'])
    assert adapter.metadata['defined_count'] == 3


@pytest.mark.parametrize('case', ['base_precision', 'bias_shape', 'bias_nonfinite', 'bias_nontangent',
                                 'floor_negative', 'floor_shape', 'root_bool', 'root_oob', 'sample_shape'])
def test_invalid_complete_geometry_contract_rejected(case):
    base, bias, floor = fixture()
    root = 0
    if case == 'base_precision':
        base = base.float()
    elif case == 'bias_shape':
        bias = bias[:-1]
    elif case == 'bias_nonfinite':
        bias[1, 1] = float('nan')
    elif case == 'bias_nontangent':
        bias[1, 0] = 1
    elif case == 'floor_negative':
        floor[1] = -1
    elif case == 'floor_shape':
        floor = floor[:-1]
    elif case == 'root_bool':
        root = False
    elif case == 'root_oob':
        root = 4
    if case == 'sample_shape':
        adapter = RadialBiasIntervention(base, bias, floor, root)
        with pytest.raises(ValueError):
            adapter.apply(base[:-1])
    else:
        with pytest.raises(ValueError):
            RadialBiasIntervention(base, bias, floor, root)

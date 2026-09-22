"""Amplitude v1 geometry, native precision, identity and gradient checks."""
import copy
import os

import pytest
import torch

from acl_hct.hgcn_sampling import make_plan
from acl_hct.hgcn_rtsc_amplitude import FrozenAmplitudeHGCN
from acl_hct.hgcn_rtsc_amplitude_protocol import MixedStreams
from acl_hct.hgcn_rtsc_stage_a import FrozenCorrectedHGCN
from acl_hct.hgcn_tangent_amplitude import (AmplitudeCorrection,
                                            calibrate_and_cap)
from acl_hct.hgcn_upstream import load_upstream
from acl_hct.mature_hgcn import MatureHGCN


def dense_neighbors(n):
    return [[j for j in range(n) if j != i] for i in range(n)]


def test_physical_direction_norm_soft_cap_q_and_raw_penalty_gradient():
    basis = torch.tensor([[[.4, 0.], [0., .2], [.1, .1]]],
                         dtype=torch.float64)
    coeff = torch.tensor([[1., 0., 0.]], dtype=torch.float64)
    lam = torch.tensor([[2.]], dtype=torch.float64)
    q = torch.tensor([.25], dtype=torch.float64, requires_grad=True)
    step, x2, norms = calibrate_and_cap(basis, coeff, lam, q)
    raw_physical = .2 * .25 * (.8 / ((.8**2 + 1e-16)**.5))
    expected_x = raw_physical / .05
    torch.testing.assert_close(torch.sqrt(norms[0, 0]), torch.tensor(.8, dtype=torch.float64))
    torch.testing.assert_close(torch.sqrt(x2[0]), torch.tensor(expected_x, dtype=torch.float64))
    torch.testing.assert_close(lam[0, 0] * step[0, 0] / .05,
                               torch.tensor(expected_x / (1 + expected_x**2)**.5,
                                            dtype=torch.float64))
    amplitude_derivative = torch.autograd.grad(lam[0, 0] * step[0, 0], q,
                                               retain_graph=True)[0]
    penalty_derivative = torch.autograd.grad(x2.sum(), q)[0]
    assert amplitude_derivative > 0 and penalty_derivative > 0
    q_high = torch.tensor([.5], dtype=torch.float64, requires_grad=True)
    high_step, high_x2, _ = calibrate_and_cap(basis, coeff, lam, q_high)
    assert lam[0, 0] * high_step[0, 0] > lam[0, 0] * step[0, 0]
    assert lam[0, 0] * high_step[0, 0] < .05
    assert torch.autograd.grad(high_x2.sum(), q_high)[0] > penalty_derivative


@pytest.mark.parametrize('physical', [0., 1e-12, 1e-10, 1e-8, 1e-7, 1e-6])
@pytest.mark.parametrize('conformal', [2., 200.])
def test_near_zero_and_near_boundary_fp32_matches_quantized_fp64(physical, conformal):
    base = torch.zeros((1, 3, 2), dtype=torch.float32)
    base[0, 0] = torch.tensor([physical / conformal, .3 * physical / conformal])
    coeff32 = torch.tensor([[.6, -.2, .1]], dtype=torch.float32)
    q32 = torch.tensor([.2], dtype=torch.float32)
    lam32 = torch.tensor([[conformal]], dtype=torch.float32)
    b32 = base.clone().requires_grad_()
    step32, x32, _ = calibrate_and_cap(b32, coeff32, lam32, q32)
    grad32 = torch.autograd.grad(step32.sum() + .001 * x32.sum(), b32)[0]
    b64 = base.double().requires_grad_()
    step64, x64, _ = calibrate_and_cap(b64, coeff32.double(), lam32.double(), q32.double())
    grad64 = torch.autograd.grad(step64.sum() + .001 * x64.sum(), b64)[0]
    assert torch.isfinite(step32).all() and torch.isfinite(grad32).all()
    torch.testing.assert_close(step32.double(), step64, atol=3e-9, rtol=3e-5)
    torch.testing.assert_close(grad32.double(), grad64, atol=4., rtol=5e-5)


def test_fp64_finite_difference_and_zero_map_derivative():
    basis = torch.tensor([[[.02, -.01], [-.03, .05], [.1, -.07]]],
                         dtype=torch.float64, requires_grad=True)
    coeff = torch.tensor([[.1, -.3, .25]], dtype=torch.float64,
                         requires_grad=True)
    lam = torch.tensor([[2.2]], dtype=torch.float64)
    q = torch.tensor([.08], dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda b, a: calibrate_and_cap(b, a, lam, q)[0],
        (basis, coeff), eps=1e-7, atol=1e-6, rtol=1e-4)
    zero_coeff = torch.zeros_like(coeff, requires_grad=True)
    zero_step, zero_x2, _ = calibrate_and_cap(basis.detach(), zero_coeff, lam, q)
    assert torch.equal(zero_step, torch.zeros_like(zero_step))
    assert float(zero_x2.sum()) == 0.
    assert torch.autograd.grad(zero_step.sum(), zero_coeff)[0].abs().sum() > 0


@pytest.fixture(scope='module')
def upstream():
    root = os.environ.get('ACL_HGCN_UPSTREAM_PATH')
    if not root:
        pytest.skip('explicit pinned HGCN checkout required')
    return load_upstream(root)


def test_frozen_two_layer_identity_matched_init_gradient_and_full_fallback(upstream):
    torch.manual_seed(37)
    base = MatureHGCN(upstream, 3, 4, 5).eval()
    stream = MixedStreams()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(stream.seed('module_init', 11))
        model = FrozenAmplitudeHGCN(base)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(stream.seed('module_init', 11))
        old = FrozenCorrectedHGCN(copy.deepcopy(base))
    for a, b in zip(model.corrections, old.corrections):
        for key, tensor in a.state_dict().items():
            assert torch.equal(tensor, b.state_dict()[key])
    torch.manual_seed(52)
    features = torch.randn(8, 3) * .1
    plans = [make_plan(dense_neighbors(8), 4, torch.Generator().manual_seed(i + 52))
             for i in range(2)]
    original = base.encode(features, [p.matrix() for p in plans])
    corrected, measured = model.encode(features, plans, detailed=True)
    assert torch.equal(corrected, original)
    assert all('basis_physical_norm' in d['distribution'] for d in measured)
    assert all(len(d['distribution']['basis_physical_norm']) == 3 for d in measured)
    assert sum(p.numel() for p in model.trainable_parameters()) == 454
    loss = model.score(corrected, torch.tensor([[0, 1], [2, 3], [4, 5]])).square().mean()
    loss.backward()
    assert all(p.grad is None for p in base.parameters())
    assert all(sum(float(p.grad.abs().sum()) for p in layer.coefficients[-1].parameters()) > 0
               for layer in model.corrections)
    full = make_plan(dense_neighbors(8), None, torch.Generator())
    original_full = base.encode(features, [full.matrix()] * 2)
    corrected_full, metadata = model.encode(features, [full] * 2, detailed=True)
    assert torch.equal(corrected_full, original_full)
    assert all(d['eligible_nodes'] == 0 for d in metadata)
    with torch.no_grad():
        for layer in model.corrections:
            layer.coefficients[-1].bias.copy_(torch.tensor([.5, -.4, .3]))
    response, metadata = model.encode(features, plans, detailed=True)
    assert (response - original).abs().max() > 1e-6
    assert all(0 <= float(d['post_cap_step_mean']) < 1 for d in metadata)
    flat = torch.tensor([[.2, 0.]] * 8)
    flat_result, metadata = model.corrections[0](flat, flat, plans[0],
                                                 base.encoder.manifold, detailed=True)
    assert torch.equal(flat_result, flat)
    assert metadata['basis_epsilon_affected'] == [8, 8, 8]

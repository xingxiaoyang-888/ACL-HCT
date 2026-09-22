"""R-TSC Stage A geometry, identity, sampling and gradient quality checks."""
import copy
import os

import numpy as np
import pytest
import torch

from acl_hct.hgcn_sampling import make_plan, mask_graph
from acl_hct.hgcn_tangent_correction import (TangentCorrection, ball_distance_from_anchor,
                                             relation_gap_loss, training_relation_weights)
from acl_hct.hgcn_upstream import load_upstream
from acl_hct.mature_hgcn import MatureHGCN
from acl_hct.hgcn_rtsc_stage_a import FrozenCorrectedHGCN


@pytest.fixture(scope='module')
def upstream():
    root = os.environ.get('ACL_HGCN_UPSTREAM_PATH')
    if not root:
        pytest.skip('explicit pinned HGCN checkout required')
    return load_upstream(root)


def dense_neighbors(n):
    return [[j for j in range(n) if j != i] for i in range(n)]


def test_differentiable_log_exp_uses_riemann_metric_and_zero_displacement(upstream):
    module = TangentCorrection().double()
    base = torch.tensor([[.2, .1], [-.15, .24], [.02, .3]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[.25, .08], [-.13, .18], [.02, .3]], dtype=torch.float64, requires_grad=True)
    log = module.log_at(base, target)
    recovered = module.exp_at(base, log)
    torch.testing.assert_close(recovered, target, atol=1e-13, rtol=0)
    physical = module._conformal(base).squeeze(-1) * torch.linalg.vector_norm(log, dim=-1)
    independent = ball_distance_from_anchor(base, target)
    torch.testing.assert_close(physical, independent, atol=1e-13, rtol=0)
    assert torch.autograd.gradcheck(lambda p, q: module.log_at(p, q), (base, target), eps=1e-6,
                                    atol=2e-5, rtol=2e-4)
    assert torch.autograd.gradcheck(lambda p, v: module.exp_at(p, v),
                                    (base, log.detach().requires_grad_()), eps=1e-6,
                                    atol=2e-5, rtol=2e-4)
    same = module.log_at(base, base)
    assert torch.equal(same, torch.zeros_like(same))
    coincident = base.detach().clone().requires_grad_()
    jacobian = torch.autograd.functional.jacobian(
        lambda q: module.log_at(coincident.detach(), q), coincident)
    for row in range(len(coincident)):
        torch.testing.assert_close(jacobian[row, :, row, :], torch.eye(2, dtype=torch.float64),
                                   atol=1e-12, rtol=0)
    derivative = torch.autograd.grad(module.exp_at(base, torch.zeros_like(base)).sum(), base)[0]
    assert torch.isfinite(derivative).all()


def test_zero_initialized_stage_a_is_exact_after_hypact_and_both_modules_learn(upstream):
    torch.manual_seed(37)
    base = MatureHGCN(upstream, 3, 4, 5).eval()
    model = FrozenCorrectedHGCN(base)
    features = torch.randn(8, 3) * .04
    rng = torch.Generator().manual_seed(52)
    plans = [make_plan(dense_neighbors(8), 4, rng) for _ in range(2)]
    original = base.encode(features, [p.matrix() for p in plans])
    corrected, measured = model.encode(features, plans)
    assert torch.equal(corrected, original)
    assert [r['eligible_nodes'] for r in measured] == [8, 8]
    assert all(float(r['normalized_step_mean']) == 0. for r in measured)
    queries = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long)
    original_loss = base.score(original, queries).square().mean()
    loss = model.score(corrected, queries).square().mean() + .001 * model.step_penalty(measured)
    assert torch.equal(loss, original_loss)
    loss.backward()
    assert all(p.grad is None for p in base.parameters())
    assert all(sum(float(p.grad.abs().sum()) for p in module.coefficients[-1].parameters()) > 0
               for module in model.corrections)
    assert sum(p.numel() for p in model.trainable_parameters()) == 454


def test_nonzero_correction_survives_activation_and_respects_step_limit(upstream):
    torch.manual_seed(37)
    base = MatureHGCN(upstream, 3, 4, 5).eval()
    model = FrozenCorrectedHGCN(base)
    features = torch.randn(8, 3) * .4
    rng = torch.Generator().manual_seed(52)
    plans = [make_plan(dense_neighbors(8), 4, rng) for _ in range(2)]
    original, _ = model.encode(features, plans)
    with torch.no_grad():
        for correction in model.corrections:
            correction.coefficients[-1].bias.copy_(torch.tensor([1., -.5, .25]))
    corrected, measured = model.encode(features, plans)
    assert (original - corrected).abs().max() > 1e-4
    assert all(torch.isfinite(corrected).flatten())
    assert (corrected.square().sum(dim=1) < 1).all()
    assert all(float(r['normalized_step_mean']) <= 1.00001 for r in measured)
    assert sum(r['clipped_nodes'] for r in measured) > 0


def test_full_low_k_and_zero_dispersion_identity_and_HT_metadata(upstream):
    torch.manual_seed(3)
    base = MatureHGCN(upstream, 3, 4, 5).eval()
    model = FrozenCorrectedHGCN(base)
    features = torch.randn(6, 3) * .1
    full = [make_plan(dense_neighbors(6), None, torch.Generator()) for _ in range(2)]
    original = base.encode(features, [p.matrix() for p in full])
    corrected, diagnostics = model.encode(features, full)
    assert torch.equal(original, corrected)
    assert [d['eligible_nodes'] for d in diagnostics] == [0, 0]
    sparse = [[1], [0, 2], [1], [], [], []]
    low = [make_plan(sparse, 1, torch.Generator()) for _ in range(2)]
    original = base.encode(features, [p.matrix() for p in low])
    corrected, diagnostics = model.encode(features, low)
    assert torch.equal(original, corrected)
    assert [d['eligible_nodes'] for d in diagnostics] == [0, 0]
    # The target positive is masked in both directions before N and k exist.
    masked = mask_graph(dense_neighbors(6), [(0, 1)])
    assert 1 not in masked[0] and 0 not in masked[1]
    plan = make_plan(masked, 4, torch.Generator().manual_seed(8))
    TangentCorrection._plan_edges(plan, 6)
    altered = copy.deepcopy(plan)
    altered.weights[0] += .01
    with pytest.raises(ValueError, match='HT weights'):
        TangentCorrection._plan_edges(altered, 6)
    flat = [[.2, 0.]] * 6
    point = torch.tensor(flat, dtype=torch.float32)
    degenerate = make_plan(dense_neighbors(6), 4, torch.Generator().manual_seed(4))
    z, metadata = model.corrections[0](point, point, degenerate, base.encoder.manifold)
    assert torch.equal(z, point) and metadata['eligible_nodes'] == 6
    assert metadata['clipped_nodes'] == 0


def test_native_near_boundary_and_near_coincident_messages_are_finite(upstream):
    torch.manual_seed(91)
    base = MatureHGCN(upstream, 2, 2, 2).eval()
    module = TangentCorrection(chunk_edges=3)
    with torch.no_grad():
        module.coefficients[-1].bias.copy_(torch.tensor([.3, -.2, .1]))
    center = torch.tensor([.995, 0.], dtype=torch.float32)
    aggregate = center.repeat(6, 1).requires_grad_()
    messages = aggregate.detach().clone()
    messages[:, 1] = torch.tensor([0., 1e-6, -1e-6, 2e-6, -2e-6, 3e-6])
    messages.requires_grad_()
    plan = make_plan(dense_neighbors(6), 4, torch.Generator().manual_seed(9))
    corrected, measured = module(aggregate, messages, plan, base.encoder.manifold)
    assert torch.isfinite(corrected).all()
    assert (corrected.square().sum(dim=-1) < 1).all()
    assert float(measured['normalized_step_mean']) <= 1.00001
    corrected.square().sum().backward()
    assert torch.isfinite(aggregate.grad).all() and torch.isfinite(messages.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_native_boundary_log_matches_fp64_and_has_identity_zero_derivative():
    fp32 = TangentCorrection()
    fp64 = TangentCorrection().double()
    anchor = torch.tensor([[.996, 0.]], dtype=torch.float32)
    for offset in (0., 1e-7, 1e-6, 1e-5):
        target = torch.tensor([[.996, offset]], dtype=torch.float32)
        native = fp32.log_at(anchor, target)
        reference = fp64.log_at(anchor.double(), target.double())
        torch.testing.assert_close(native.double(), reference, atol=1e-11, rtol=0)
    jacobian = torch.autograd.functional.jacobian(
        lambda target: fp32.log_at(anchor, target), anchor)
    torch.testing.assert_close(jacobian[0, :, 0, :], torch.eye(2), atol=2e-6, rtol=0)


def test_child_macro_relation_weights_and_root_stop_gradient():
    positives = torch.tensor([[0, 2], [1, 2], [0, 3], [1, 4], [2, 4], [3, 4]], dtype=torch.long)
    weight = training_relation_weights(positives, 5)
    counts = {2: 2, 3: 1, 4: 3}
    expected = torch.tensor([6 / (3 * counts[int(c)]) for c in positives[:, 1]], dtype=torch.float64)
    torch.testing.assert_close(weight, expected, atol=0, rtol=0)
    assert abs(float(weight.mean()) - 1) < 1e-15
    points = torch.tensor([[0., 0.], [.12, 0.], [.04, 0.], [.2, 0.], [.06, 0.]],
                          dtype=torch.float64, requires_grad=True)
    loss = relation_gap_loss(points, positives, weight, 0, .1)
    anchor = points[0:1].detach().expand(len(positives), -1)
    gap = ball_distance_from_anchor(anchor, points[positives[:, 1]]) - ball_distance_from_anchor(anchor, points[positives[:, 0]])
    per_edge = torch.relu((.1 - gap) / .1).square()
    macro = torch.stack([per_edge[positives[:, 1] == child].mean() for child in (2, 3, 4)]).mean()
    torch.testing.assert_close(loss, macro, atol=1e-15, rtol=0)
    loss.backward()
    assert torch.isfinite(points.grad).all()
    assert torch.equal(points.grad[0], torch.zeros_like(points.grad[0]))
    assert points.grad[1:].abs().sum() > 0
    collapsed = torch.zeros(3, 2, dtype=torch.float64, requires_grad=True)
    pos = torch.tensor([[0, 1], [0, 2]])
    relation_gap_loss(collapsed, pos, training_relation_weights(pos, 3), 0, .1).backward()
    assert torch.isfinite(collapsed.grad).all()

import ast
import copy
import hashlib
import inspect
import random

import numpy as np
import pytest
import torch
from torch import nn

from acl_hct.hgcn_rtsc_joint_eval_v2 import (
    COMPOSITE_BACKEND_ID, _deterministic_index_add, _sampled_correction,
    backend_descriptor, deterministic_state, encode_evaluation,
    evaluate_frozen_snapshot, evaluation_source_hashes,
    make_evaluation_binding, require_compatible_evaluation_bindings,
    require_evaluation_binding)
from acl_hct.hgcn_rtsc_joint_worker import (
    _report_only_full_bridge, _training_step)
from acl_hct import hgcn_rtsc_full_eval_v2
from acl_hct.hgcn_sampling import make_plan


def _plans(fanout):
    neighbors = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
    return [make_plan(neighbors, fanout,
                      torch.Generator().manual_seed(10 + layer))
            for layer in (0, 1)]


def _binding(plans, *, artifact='checkpoint_file'):
    return make_evaluation_binding(
        evaluation_source_commit='1' * 40,
        training_source_commit='2' * 40,
        config_sha256='3' * 64,
        replay_policy_sha256='4' * 64,
        checkpoint_sha256='5' * 64,
        model_state_sha256='6' * 64,
        plans=plans,
        plan_phase='selection_layer',
        fanout=plans[0].fanout,
        repeat=0,
        device=torch.device('cpu'),
        checkpoint_artifact_kind=artifact)


class Stateful(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(3, 3)
        self.second = nn.BatchNorm1d(3)


def _state(model):
    return {name: value.detach().clone()
            for name, value in model.state_dict().items()}


def test_snapshot_preserves_live_model_optimizer_rng_modes_and_flags():
    model = Stateful()
    model.train()
    model.second.eval()
    parameters = list(model.parameters())
    parameters[0].requires_grad_(False)
    parameters[1].grad = torch.ones_like(parameters[1])
    optimizer = torch.optim.Adam(parameters, lr=.01)
    generator = torch.Generator().manual_seed(77)
    before_state = _state(model)
    before_modes = [module.training for module in model.modules()]
    before_requires = [parameter.requires_grad for parameter in parameters]
    before_grads = [None if parameter.grad is None else parameter.grad.clone()
                    for parameter in parameters]
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.random.get_rng_state().clone()
    before_generator = generator.get_state().clone()
    before_flags = deterministic_state()

    def evaluate(snapshot):
        assert snapshot is not model
        assert not any(module.training for module in snapshot.modules())
        assert not any(parameter.requires_grad for parameter in snapshot.parameters())
        random.random()
        np.random.rand()
        torch.rand(3)
        torch.rand(3, generator=generator)
        return tuple(value.detach().clone()
                     for value in snapshot.state_dict().values())

    result = evaluate_frozen_snapshot(
        model, lambda: copy.deepcopy(model), evaluate,
        optimizer=optimizer, generators=(generator,))
    assert result
    assert all(torch.equal(before_state[name], value)
               for name, value in model.state_dict().items())
    assert [module.training for module in model.modules()] == before_modes
    assert [parameter.requires_grad for parameter in parameters] == before_requires
    assert all((before is None and parameter.grad is None)
               or (before is not None and torch.equal(before, parameter.grad))
               for before, parameter in zip(before_grads, parameters))
    assert optimizer.state_dict() == before_optimizer
    assert random.getstate() == before_python
    assert np.array_equal(np.random.get_state()[1], before_numpy[1])
    assert torch.equal(torch.random.get_rng_state(), before_torch)
    assert torch.equal(generator.get_state(), before_generator)
    assert deterministic_state() == before_flags


def test_snapshot_restores_live_mutation_and_rejects_shared_storage_on_exception():
    model = Stateful()
    before = _state(model)

    def bad_factory():
        with torch.no_grad():
            next(model.parameters()).add_(10)
        model.eval()
        return model

    with pytest.raises(RuntimeError, match='changed live training state'):
        evaluate_frozen_snapshot(model, bad_factory, lambda snapshot: None)
    assert all(torch.equal(before[name], value)
               for name, value in model.state_dict().items())
    assert model.training

    shared = copy.copy(model)
    with pytest.raises(RuntimeError, match='shares parameter or buffer storage'):
        evaluate_frozen_snapshot(model, lambda: shared,
                                 lambda snapshot: None)


def test_snapshot_restores_rng_when_evaluator_raises():
    model = Stateful()
    generator = torch.Generator().manual_seed(91)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    generator_state = generator.get_state().clone()

    def fail(snapshot):
        random.random()
        np.random.rand()
        torch.rand(1)
        torch.rand(1, generator=generator)
        raise LookupError('evaluator failure')

    with pytest.raises(LookupError, match='evaluator failure'):
        evaluate_frozen_snapshot(
            model, lambda: copy.deepcopy(model), fail,
            generators=(generator,))
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert torch.equal(generator.get_state(), generator_state)


def test_local_strict_index_add_restores_both_flags_on_success_and_failure():
    before = deterministic_state()
    observed = []

    def operation(target, dim, index, source):
        observed.append(deterministic_state())
        return target.index_add_(dim, index, source)

    with torch.inference_mode():
        result = _deterministic_index_add(
            torch.zeros(4), 0, torch.tensor([0, 2, 2]),
            torch.tensor([1., 2., 3.]), _operation=operation)
    assert torch.equal(result, torch.tensor([1., 0., 5., 0.]))
    assert observed == [(True, False)]
    assert deterministic_state() == before

    def fail(*args):
        assert deterministic_state() == (True, False)
        raise LookupError('sentinel')

    with torch.inference_mode(), pytest.raises(LookupError, match='sentinel'):
        _deterministic_index_add(
            torch.zeros(1), 0, torch.tensor([0]), torch.ones(1),
            _operation=fail)
    assert deterministic_state() == before


def test_full_sparse_bmm_matches_dense_reference_and_restores_flags():
    indices = torch.tensor([[0, 0, 1, 2], [0, 2, 1, 0]])
    values = torch.tensor([.5, .5, 1., 1.])
    adjacency = torch.sparse_coo_tensor(
        indices, values, (3, 3)).coalesce()
    dense = torch.tensor([[1., 2.], [3., 4.], [5., 6.]]).contiguous()
    before = deterministic_state()
    with torch.inference_mode():
        actual = hgcn_rtsc_full_eval_v2.deterministic_spmm(
            adjacency, dense)
    assert torch.equal(actual, adjacency.to_dense() @ dense)
    assert deterministic_state() == before

def test_sampled_correction_has_exactly_four_locally_strict_reduction_sites():
    tree = ast.parse(inspect.getsource(_sampled_correction))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == '_deterministic_index_add']
    assert len(calls) == 4
    source = inspect.getsource(_sampled_correction)
    assert 'correction.coefficients(features)' in source
    assert '_coo_alg2_flag' not in source


def test_coefficient_network_runs_under_default_flags_outside_reductions():
    class Coefficients(nn.Module):
        def __init__(self):
            super().__init__()
            self.observed = []

        def forward(self, features):
            self.observed.append(deterministic_state())
            return features.new_zeros((len(features), 3))

    class Correction:
        c, tau, eps, chunk_edges = 1., .05, 1e-8, 4096

        def __init__(self):
            self.coefficients = Coefficients()

        @staticmethod
        def _plan_edges(plan, nodes):
            return plan.indices[0], plan.indices[1]

        @staticmethod
        def _conformal(points):
            return points.new_ones((len(points), 1))

        @staticmethod
        def log_at(base, target):
            return target - base

        @staticmethod
        def exp_at(base, tangent):
            return base + tangent

    class Manifold:
        @staticmethod
        def proj(points, curvature):
            return points

    neighbors = [[j for j in range(6) if j != i] for i in range(6)]
    plan = make_plan(neighbors, 3, torch.Generator().manual_seed(5))
    correction = Correction()
    aggregate = torch.zeros((6, 2))
    messages = torch.arange(12, dtype=torch.float32).reshape(6, 2) / 100
    with torch.inference_mode():
        result, diagnostics = _sampled_correction(
            correction, aggregate, messages, plan, Manifold())
    assert result.shape == aggregate.shape
    assert diagnostics['eligible_nodes'] == 6
    assert correction.coefficients.observed == [(False, False)]
    assert deterministic_state() == (False, False)


def test_production_numerical_functions_match_qualified_ast_identities():
    expected = {
        hgcn_rtsc_full_eval_v2.deterministic_spmm:
            'e2e53897991eac0c01c2b1d3be7663defde0cda78483485509d62416274a1291',
        hgcn_rtsc_full_eval_v2._encode:
            'c126604ffb20278c7de2fa714afd9f6c69cf3cd68b7a1a5622755dd23bbbfb0e',
        _sampled_correction:
            '6bcf5f0447bffc18b4129abbd990522aed3ad1518713f08681b9bd714ed99571',
    }
    for function, qualified_ast_sha256 in expected.items():
        node = ast.parse(inspect.getsource(function)).body[0]
        actual = hashlib.sha256(
            ast.dump(node, include_attributes=False).encode()).hexdigest()
        assert actual == qualified_ast_sha256


def test_route_rejects_mixed_plans_and_selects_full_or_sampled(monkeypatch):
    observed = []
    monkeypatch.setattr(
        'acl_hct.hgcn_rtsc_joint_eval_v2.encode_full_evaluation',
        lambda *args, **kwargs: observed.append('full') or ('F', []))
    monkeypatch.setattr(
        'acl_hct.hgcn_rtsc_joint_eval_v2.encode_sampled_evaluation',
        lambda *args, **kwargs: observed.append('sampled') or ('S', []))
    assert encode_evaluation(object(), torch.empty(0), _plans(None))[0] == 'F'
    assert encode_evaluation(object(), torch.empty(0), _plans(4))[0] == 'S'
    mixed = [_plans(None)[0], _plans(4)[1]]
    with pytest.raises(ValueError, match='mixed'):
        encode_evaluation(object(), torch.empty(0), mixed)
    assert observed == ['full', 'sampled']


def test_binding_rejects_missing_legacy_mixed_and_tampered_source_identity():
    full = _plans(None)
    first = _binding(full)
    assert first['evaluation_suite']['evaluation_suite_id'] == COMPOSITE_BACKEND_ID
    assert first['checkpoint_artifact']['kind'] == 'checkpoint_file'
    assert require_evaluation_binding(first)
    fixture = _binding(full, artifact='serialized_state_fixture_or_probe')
    assert 'byte_sha256' not in fixture['checkpoint_artifact']
    assert 'serialized_state_sha256' in fixture['checkpoint_artifact']

    missing = copy.deepcopy(first)
    missing.pop('plans')
    with pytest.raises(ValueError, match='complete exact'):
        require_evaluation_binding(missing)
    legacy = copy.deepcopy(first)
    legacy['schema_version'] = 1
    with pytest.raises(ValueError, match='legacy or mixed'):
        require_evaluation_binding(legacy)
    tampered = copy.deepcopy(first)
    key = next(iter(tampered['evaluator_source']['sha256_normalized_lf']))
    tampered['evaluator_source']['sha256_normalized_lf'][key] = 'a' * 64
    with pytest.raises(ValueError, match='source identity'):
        require_evaluation_binding(tampered)
    mixed = copy.deepcopy(first)
    mixed['config_sha256'] = '9' * 64
    with pytest.raises(ValueError, match='mixed'):
        require_compatible_evaluation_bindings([first, mixed])

    benchmark = make_evaluation_binding(
        evaluation_source_commit='1' * 40,
        training_source_commit='2' * 40,
        config_sha256='3' * 64,
        replay_policy_sha256='4' * 64,
        checkpoint_sha256='5' * 64,
        model_state_sha256='6' * 64,
        plans=_plans(4), plan_phase='benchmark', fanout=4,
        plan_step=7, repeat=0, device=torch.device('cpu'))
    assert {(row['step'], row['repeat']) for row in benchmark['plans']} == {
        (7, 0)}


def test_legacy_to_v2_full_bridge_is_explicitly_report_only():
    def ranking(ranks):
        rows = [{'parent': 0, 'child': child, 'rank': rank, 'candidates': 3}
                for child, rank in zip((1, 2), ranks)]
        mrr = sum(1 / rank for rank in ranks) / 2
        return {
            'status': 'complete', 'rows': rows, 'expected_queries': 2,
            'completed_queries': 2, 'completed_children': 2,
            'query_micro_mrr': mrr, 'child_macro_mrr': mrr,
            'hits': {str(k): sum(rank <= k for rank in ranks) / 2
                     for k in (1, 3, 10)}}

    class Evaluator:
        def evaluate(self, points):
            score = float(np.asarray(points).mean())
            return {kind: {'metrics': {'score': score}}
                    for kind in ('direct', 'distant')}

    old = np.zeros((4, 2), dtype=np.float32)
    new = old.copy()
    new[1, 0] = .01
    legacy = {'native_ball_points': old, 'ranking': ranking((1, 2))}
    report = _report_only_full_bridge(
        legacy, new, ranking((1, 3)), Evaluator(), {'model': {'c': 1.}},
        [(0, 1), (0, 2)], {1: {0}, 2: {0}})
    assert report['status'] == 'report_only_cross_backend_bridge'
    assert report['hard_gate_applied'] is False
    assert report['rank_changed_queries'] == 1
    assert report['hits_signed_delta']['1'] == 0


def test_training_forward_source_remains_frozen_and_uses_original_operator():
    path = inspect.getsourcefile(__import__(
        'acl_hct.hgcn_rtsc_joint', fromlist=['encode_joint']))
    normalized = open(path, 'rb').read().replace(b'\r\n', b'\n')
    assert hashlib.sha256(normalized).hexdigest() == (
        'a0b71b2a136e525acf4d4b38dd832f0fc9b6b74a9b348c4680516453c5351ce9')
    source = inspect.getsource(_training_step)
    assert 'encode_joint(model, features, plans)' in source
    assert 'encode_evaluation' not in source


def test_backend_descriptor_binds_all_qualification_artifacts_and_sources():
    descriptor = backend_descriptor()
    assert descriptor['qualification_status'] == (
        'accepted_for_recorded_runtime_and_fixtures')
    assert len(descriptor['qualification_artifacts']) == 8
    assert descriptor['training_supported'] is False
    assert len(evaluation_source_hashes()) == 5

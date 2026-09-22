"""Registered joint arithmetic, paired inference and six-arm quality gates."""
import copy
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from acl_hct.hgcn_rtsc_joint import JointCorrectedHGCN, encode_joint
from acl_hct.hgcn_rtsc_joint_protocol import (
    ARMS, NamedStreams, analyze_final_shards, choose_checkpoint,
    final_families, validate_config)
from acl_hct.hgcn_rtsc_joint_worker import (
    _binding, _evaluate_candidate, _new_model, _optimizer,
    _require_clean_benchmark_baseline, _training_step, final_shard,
    small_quality, train_arm)
from acl_hct.hgcn_evidence import complete_ranking, state_hash
from acl_hct.diagnostic_archive import read_archive
from acl_hct.hgcn_panels import HierarchyPanel
from acl_hct.hgcn_sampling import make_plan
from acl_hct.hgcn_upstream import load_upstream
from acl_hct.mature_hgcn import MatureHGCN


ROOT = Path(__file__).parents[1]


def configs():
    read = lambda name: json.loads((ROOT / 'configs' / name).read_bytes())
    return (read('hgcn_rtsc_joint_v1.json'),
            read('mature_hgcn_validation.json'),
            read('mature_hgcn_replay_policy_v2.json'),
            read('hgcn_rtsc_stage_a.json'))


def test_config_is_exact_and_joint_streams_are_disjoint():
    joint, original, policy, old = configs()
    assert validate_config(joint, original, policy, old) == (
        'ec459bfeb4cbefd55894e899086a201632d093c8e448a287c7204d731ff6e1a2')
    changed = copy.deepcopy(joint)
    changed['training']['base_lr'] *= 10
    with pytest.raises(ValueError, match='exact'):
        validate_config(changed, original, policy, old)
    streams = NamedStreams()
    first = streams.seed('module_init', 11, layer=0)
    second = streams.seed('module_init', 11, layer=1)
    assert first != second
    assert streams.seed('train_batch', 11, step=1, fanout=4) != (
        streams.seed('final_layer', 11, fanout=4, repeat=1, layer=0))
    assert NamedStreams().seed('module_init', 11, layer=0) == first


def test_common_s0_selection_gate_is_not_candidate_minus_sample():
    rows = [{'step': step, 'micro_mrr': [mrr, mrr],
             'direct_order': [order, order]}
            for step, mrr, order in
            ((0, .2, .500), (256, .5, .4979), (512, .3, .498),
             (768, .4, .499), (1024, .1, .6))]
    selected = choose_checkpoint(rows, .5, -.002)
    assert selected['step'] == 768
    assert selected['direct_order_gate'] == pytest.approx(.498)
    assert selected['mean_micro_mrr'] == .4
    with pytest.raises(ValueError, match='five'):
        choose_checkpoint(rows[:-1], .5)
    with pytest.raises(ValueError, match='finite'):
        choose_checkpoint(rows, float('nan'))
    assert choose_checkpoint(rows, .7) is None


def synthetic_grid():
    return {
        (seed, fanout, repeat, arm): {
            'micro_mrr': .2 + .00003 * repeat + .0001 * ARMS.index(arm),
            'direct_order': .5 + .00002 * repeat + .001 * ARMS.index(arm)}
        for seed in (11, 23) for fanout in (4, 8, 16)
        for repeat in range(16) for arm in ARMS}


def test_three_registered_families_require_full_paired_grid_and_zero_variance_is_conservative():
    records = synthetic_grid()
    main, protection, relation = final_families(records)
    assert (len(main), len(protection), len(relation)) == (16, 32, 36)
    assert all(row['family_size'] == 16 for row in main)
    assert all(row['family_size'] == 32 for row in protection)
    assert all(row['family_size'] == 36 for row in relation)
    assert all(row['p'] == 1 and row['marginal_student95'] is None
               and not row['reject_holm'] for row in main + protection + relation)
    assert next(row for row in main if row['contrast'] == 'three-single'
                and row['metric'] == 'direct_order')['noninferiority_margin'] == .002
    incomplete = dict(records)
    incomplete.pop((11, 4, 0, 'single_task'))
    with pytest.raises(ValueError, match='complete'):
        final_families(incomplete)


def test_final_analysis_rejects_missing_controls_and_keeps_two_starting_models_separate():
    records = synthetic_grid()
    shards = []
    for seed in (11, 23):
        for start in (0, 4, 8, 12):
            rows = [{'seed': seed, 'fanout': fanout, 'repeat': repeat,
                     'arm': arm, **records[seed, fanout, repeat, arm]}
                    for fanout in (4, 8, 16)
                    for repeat in range(start, start + 4)
                    for arm in ARMS]
            shards.append({
                'status': 'complete', 'seed': seed, 'repeat_start': start,
                'original_F_qualification': {'accepted': True},
                'full_qualifications': {arm: {'accepted': True} for arm in ARMS},
                'module_off_full_controls': {
                    arm: {'accepted': True} for arm in ARMS
                    if not arm.startswith('plain_')},
                'selected': {arm: {'checkpoint': f'{seed}/{arm}'}
                             for arm in ARMS},
                'fixed_F': {arm: {'micro_mrr': .3, 'direct_order': .6}
                            for arm in ARMS},
                'rows': rows})
    result = analyze_final_shards(shards)
    assert result['status'] == 'complete'
    assert len(result['selected']) == 2
    assert not any(result['module_increment_by_loss'].values())
    corrupt = copy.deepcopy(shards)
    corrupt[0]['module_off_full_controls'].pop('single_task')
    with pytest.raises(ValueError, match='full-qualified'):
        analyze_final_shards(corrupt)


@pytest.mark.parametrize('loss', ('task', 'relation'))
def test_two_layer_hidden_weights_match_three_and_single_for_each_loss(loss):
    upstream_root = os.environ.get('ACL_HGCN_UPSTREAM_PATH')
    if not upstream_root:
        pytest.skip('explicit pinned HGCN checkout required')
    upstream = load_upstream(upstream_root)
    torch.manual_seed(18)
    original = MatureHGCN(upstream, 3, 4, 5)
    streams = NamedStreams()
    seeds = [streams.seed('module_init', 11, layer=layer)
             for layer in (0, 1)]
    three = JointCorrectedHGCN(copy.deepcopy(original), 'three',
                               layer_init_seeds=seeds)
    single = JointCorrectedHGCN(copy.deepcopy(original), 'single',
                                layer_init_seeds=seeds)
    for layer in (0, 1):
        assert torch.equal(three.corrections[layer].coefficients[0].weight,
                           single.corrections[layer].coefficients[0].weight)
        assert torch.equal(three.corrections[layer].coefficients[0].bias,
                           single.corrections[layer].coefficients[0].bias)
    assert sum(p.numel() for p in three.corrections.parameters()) == 454
    assert sum(p.numel() for p in single.corrections.parameters()) == 386
    with torch.no_grad():
        for three_layer, single_layer in zip(
                three.corrections, single.corrections):
            three_layer.coefficients[2].bias[0] = .2
            single_layer.coefficients[2].bias[0] = .2
    features = torch.randn(8, 3) * .1
    neighbors = [[j for j in range(8) if j != i] for i in range(8)]
    plans = [make_plan(neighbors, 4, torch.Generator().manual_seed(layer))
             for layer in (0, 1)]
    three_points, _ = encode_joint(three, features, plans)
    single_points, _ = encode_joint(single, features, plans)
    assert torch.equal(three_points, single_points)
    assert all(parameter.requires_grad for parameter in three.base.parameters())
    assert all(parameter.requires_grad for parameter in single.base.parameters())


def test_six_arm_cpu_quality_updates_encoder_head_and_each_module_layer(tmp_path, monkeypatch):
    upstream_root = os.environ.get('ACL_HGCN_UPSTREAM_PATH')
    if not upstream_root:
        pytest.skip('explicit pinned HGCN checkout required')
    upstream = load_upstream(upstream_root)
    joint, original, policy, _ = configs()
    monkeypatch.setenv('ACL_HGCN_VALIDATION_PID', str(os.getppid()))
    monkeypatch.setenv('ACL_HGCN_VALIDATION_DEADLINE', str(time.monotonic() + 300))
    result = small_quality(upstream, joint, original, policy,
                           torch.device('cpu'), tmp_path, '0' * 40)
    assert result['status'] == 'passed'
    assert result['three_single_hidden_initialization_matched']
    assert result['cuda_memory_isolation']['status'] == 'not_applicable_cpu'
    assert set(result['six_arms']) == set(ARMS)
    for arm, row in result['six_arms'].items():
        assert row['fresh_own_F_qualification']['accepted']
        assert all(update['encoder'] > 0 and update['head'] > 0
                   for update in row['group_update_norm'])
        if not arm.startswith('plain_'):
            assert all(update['module_layer0'] > 0 and update['module_layer1'] > 0
                       for update in row['group_update_norm'])


def test_discarded_probe_reloads_its_updated_own_full_reference(tmp_path, monkeypatch):
    upstream_root = os.environ.get('ACL_HGCN_UPSTREAM_PATH')
    if not upstream_root:
        pytest.skip('explicit pinned HGCN checkout required')
    upstream = load_upstream(upstream_root)
    joint, _, policy, _ = configs()
    mini = {'model': {'hidden': 4, 'c': 1.},
            'prepared': {'nodes_count': 8},
            'ranking': {'candidate_chunk': 8, 'max_seconds': 60}}
    torch.manual_seed(61)
    base = MatureHGCN(upstream, 3, 4, 5)
    base_sha = state_hash(base.state_dict())
    nodes = [str(i) for i in range(8)]
    neighbors = [[j for j in range(8) if j != i] for i in range(8)]
    positives = [(0, 1), (0, 2), (1, 3)]
    groups = torch.tensor([[[a, b], [4, b], [5, b], [6, b], [7, b]]
                           for a, b in positives], dtype=torch.long)
    data = {'nodes': nodes, 'neighbors': neighbors,
            'features': torch.randn(8, 3) * .1,
            'query_groups': groups,
            'labels': torch.tensor([[1., 0., 0., 0., 0.]] * len(groups))}
    view = SimpleNamespace(nodes=nodes, root='0', reachable=set(nodes))
    panel = {
        'pool_size': 2,
        'rows': [{'id': child, 'index': int(child),
                  'inclusion_probability': 1., 'pool_mean_weight': .5}
                 for child in ('2', '3')],
        'relations': [
            {'child': '2', 'direct_parents': ['0'],
             'positive_distant_ancestors': []},
            {'child': '3', 'direct_parents': ['1'],
             'positive_distant_ancestors': ['0']}]}
    queries = [(0, 2), (1, 3)]
    truth = {2: {0}, 3: {1}}
    full = make_plan(neighbors, None, torch.Generator())
    baseline_points = base.encode(data['features'], [full.matrix()] * 2)
    evaluator = HierarchyPanel(view, panel,
                               baseline_points.detach().cpu().numpy())
    monkeypatch.setenv('ACL_HGCN_VALIDATION_PID', str(os.getppid()))
    monkeypatch.setenv('ACL_HGCN_VALIDATION_DEADLINE', str(time.monotonic() + 300))
    monkeypatch.setattr('acl_hct.hgcn_rtsc_joint_worker.full_control',
                        lambda *args: (data['features'], queries, truth,
                                       evaluator, {'accepted': True}))
    monkeypatch.setattr('acl_hct.hgcn_rtsc_joint_worker.load_panel_design',
                        lambda *args: (view, panel))
    monkeypatch.setattr('acl_hct.hgcn_rtsc_joint_worker.hierarchy_evaluator',
                        lambda view, panel, points, original: (
                            HierarchyPanel(view, panel, points),
                            HierarchyPanel(view, panel, points).evaluate(points)))
    result = train_arm(
        data, base, {}, {'original': True}, base_sha, mini, policy, tmp_path,
        joint, 11, 'three_relation', tmp_path, torch.device('cpu'), upstream,
        '0' * 40, probe_steps=2)
    assert result['probe_weights_discarded']
    assert result['probe_terminal_fresh_replay']['fresh_full_qualification']['accepted']
    assert result['probe_terminal_fresh_replay']['weights_discarded']
    assert len(result['probe_complete_readouts']) == 2
    assert result['selected_checkpoint'] is None
    assert not list(tmp_path.glob('*.pt'))
    for row in result['history']:
        assert row['group_grad_norm_before_clip']['module_layer0'] > 0
        assert row['group_grad_norm_before_clip']['module_layer1'] > 0


def test_updated_candidate_uses_its_own_full_root_and_fresh_reload(tmp_path, monkeypatch):
    upstream_root = os.environ.get('ACL_HGCN_UPSTREAM_PATH')
    if not upstream_root:
        pytest.skip('explicit pinned HGCN checkout required')
    upstream = load_upstream(upstream_root)
    settings, _, policy, _ = configs()
    original = {'model': {'hidden': 4, 'c': 1.},
                'prepared': {'nodes_count': 8},
                'ranking': {'candidate_chunk': 8, 'max_seconds': 60}}
    torch.manual_seed(70)
    base = MatureHGCN(upstream, 3, 4, 5)
    template = copy.deepcopy(base).cpu()
    seeds = [NamedStreams().seed('module_init', 11, layer=layer)
             for layer in (0, 1)]
    model = _new_model(base, 'three', settings, seeds)
    optimizer, parameter_groups = _optimizer(model, settings)
    features = torch.randn(8, 3) * .1
    neighbors = [[j for j in range(8) if j != i] for i in range(8)]
    sampled = [[make_plan(neighbors, 4,
                          torch.Generator().manual_seed(10 * repeat + layer))
                for layer in (0, 1)] for repeat in (0, 1)]
    frozen = [{'plans': plans, 'archive': {'manifest': f's0-{i}.json'}}
              for i, plans in enumerate(sampled)]
    full = make_plan(neighbors, None, torch.Generator())
    initial_points, _ = encode_joint(model, features, [full] * 2)
    groups = torch.tensor([
        [[1, 2], [3, 2], [4, 2], [5, 2], [6, 2]],
        [[1, 3], [2, 3], [4, 3], [5, 3], [6, 3]]], dtype=torch.long)
    labels = torch.tensor([[1., 0., 0., 0., 0.],
                           [1., 0., 0., 0., 0.]])
    from acl_hct.hgcn_sampling import mask_graph
    from acl_hct.hgcn_tangent_correction import training_relation_weights
    masked = mask_graph(neighbors, groups[:, 0].tolist())
    plans = [make_plan(masked, 4, torch.Generator().manual_seed(layer + 30))
             for layer in (0, 1)]
    _training_step(model, optimizer, parameter_groups, features, groups, labels,
                   training_relation_weights(groups[:, 0], 8), 0, plans,
                   settings, 'three_task', torch.device('cpu'))
    view = SimpleNamespace(nodes=list(range(8)), root=0,
                           reachable=set(range(8)))
    panel = {'pool_size': 2,
             'rows': [{'id': child, 'index': child,
                       'inclusion_probability': 1., 'pool_mean_weight': .5}
                      for child in (2, 3)],
             'relations': [{'child': child, 'direct_parents': [1],
                            'positive_distant_ancestors': [0]}
                           for child in (2, 3)]}
    valid = [(1, 2), (1, 3)]
    truth = {2: {1}, 3: {1}}
    monkeypatch.setenv('ACL_HGCN_VALIDATION_PID', str(os.getppid()))
    monkeypatch.setenv('ACL_HGCN_VALIDATION_DEADLINE', str(time.monotonic() + 300))
    monkeypatch.setattr('acl_hct.hgcn_rtsc_joint_worker.hierarchy_evaluator',
                        lambda view, panel, points, original: (
                            HierarchyPanel(view, panel, points),
                            HierarchyPanel(view, panel, points).evaluate(points)))
    candidate, checkpoint = _evaluate_candidate(
        model, template, optimizer, 256, frozen, features, full, view, panel,
        valid, truth, original, policy, settings, 11, 'three_task', seeds,
        '0' * 40, tmp_path, torch.device('cpu'), NamedStreams())
    reference = read_archive(tmp_path / 'arrays', candidate['full_reference'])
    assert checkpoint['model_state_sha256'] == reference['model_state_sha256']
    assert candidate['fresh_full_qualification']['accepted']
    assert candidate['module_off_full_qualification']['accepted']
    assert len(candidate['repeats']) == 2
    assert not np.array_equal(
        reference['native_ball_points'][0],
        initial_points.detach().cpu().numpy()[0])
    assert np.array_equal(reference['panel_design']['fixed_full_root_point_fp64'],
                          reference['native_ball_points'][0].astype(np.float64))


def test_final_shard_pairs_72_conditions_with_six_own_full_controls(tmp_path, monkeypatch):
    upstream_root = os.environ.get('ACL_HGCN_UPSTREAM_PATH')
    if not upstream_root:
        pytest.skip('explicit pinned HGCN checkout required')
    upstream = load_upstream(upstream_root)
    settings, _, policy, _ = configs()
    original = {'model': {'hidden': 4, 'c': 1.},
                'prepared': {'nodes_count': 8},
                'ranking': {'candidate_chunk': 8, 'max_seconds': 60}}
    torch.manual_seed(81)
    base = MatureHGCN(upstream, 3, 4, 5)
    nodes = list(range(8))
    neighbors = [[j for j in nodes if j != i] for i in nodes]
    features = torch.randn(8, 3) * .1
    data = {'nodes': nodes, 'neighbors': neighbors, 'features': features}
    valid = [(1, 2), (1, 3)]
    truth = {2: {1}, 3: {1}}
    view = SimpleNamespace(nodes=nodes, root=0, reachable=set(nodes))
    panel = {'pool_size': 2,
             'rows': [{'id': child, 'index': child,
                       'inclusion_probability': 1., 'pool_mean_weight': .5}
                      for child in (2, 3)],
             'relations': [{'child': child, 'direct_parents': [1],
                            'positive_distant_ancestors': [0]}
                           for child in (2, 3)]}
    seeds = [NamedStreams().seed('module_init', 11, layer=layer)
             for layer in (0, 1)]
    full_plan = make_plan(neighbors, None, torch.Generator())
    monkeypatch.setenv('ACL_HGCN_VALIDATION_PID', str(os.getppid()))
    monkeypatch.setenv('ACL_HGCN_VALIDATION_DEADLINE', str(time.monotonic() + 300))
    models, references, evaluators, selected = {}, {}, {}, {}
    for index, arm in enumerate(ARMS):
        model = _new_model(copy.deepcopy(base),
                           settings['arms'][arm]['structure'], settings, seeds)
        with torch.no_grad():
            next(model.encoder.parameters() if arm.startswith('plain_')
                 else model.base.encoder.parameters()).add_(index * .0001)
        model.eval()
        points, _ = encode_joint(model, features, [full_plan] * 2)
        ranking = complete_ranking(
            model if arm.startswith('plain_') else model.base,
            points, valid, truth, original['ranking'])
        binding = _binding(settings, '0' * 40, policy, 11, arm, 0)
        references[arm] = {
            'binding': binding, 'model_state_sha256': state_hash(model.state_dict()),
            'native_ball_points': points.detach().cpu().numpy(),
            'ranking': ranking, 'full_plan': full_plan.archive()}
        evaluators[arm] = HierarchyPanel(
            view, panel, points.detach().cpu().numpy())
        selected[arm] = {'step': 0, 'checkpoint': {'file': arm + '.pt'},
                         'full_reference': {'manifest': arm + '-F.json'}}
        models[arm] = model
    monkeypatch.setattr('acl_hct.hgcn_rtsc_joint_worker.full_control',
                        lambda *args: (features, valid, truth,
                                       evaluators['plain_task'],
                                       {'accepted': True}))
    monkeypatch.setattr('acl_hct.hgcn_rtsc_joint_worker.load_panel_design',
                        lambda *args: (view, panel))
    monkeypatch.setattr('acl_hct.hgcn_rtsc_joint_worker._load_selected_arms',
                        lambda *args: (models, references, evaluators, selected))
    result = final_shard(
        data, base, {}, {'original': True}, state_hash(base.state_dict()),
        original, policy, tmp_path, settings, 11, 0, tmp_path,
        torch.device('cpu'), '0' * 40, {})
    assert result['status'] == 'complete'
    assert len(result['rows']) == 72
    assert len(result['full_qualifications']) == 6
    assert len(result['module_off_full_controls']) == 4
    assert len({result['fixed_F'][arm]['model_state_sha256']
                for arm in ARMS}) == 6
    same_graph = [row for row in result['rows']
                  if row['fanout'] == 4 and row['repeat'] == 0]
    plans = [read_archive(tmp_path / 'arrays', row['archive'])['plans']
             for row in same_graph]
    assert all(np.array_equal(plans[0][layer]['indices'],
                              candidate[layer]['indices'])
               for candidate in plans[1:] for layer in (0, 1))


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA baseline-isolation regression requires a GPU')
def test_benchmark_memory_gate_catches_previous_full_graph_tensor():
    device = torch.device('cuda')
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(device)
    _require_clean_benchmark_baseline(device, baseline)
    previous_full_points = torch.empty((82115, 128), device=device)
    with pytest.raises(ValueError, match='polluted'):
        _require_clean_benchmark_baseline(device, baseline)
    del previous_full_points
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    _require_clean_benchmark_baseline(device, baseline)

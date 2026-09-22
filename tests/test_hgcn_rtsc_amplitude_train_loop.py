"""Tiny actual optimizer loop proves matched Stage A inputs and frozen weights."""
import copy
import json
import os
from pathlib import Path

import pytest
import torch

from acl_hct.hgcn_evidence import save_checkpoint, state_hash
from acl_hct.hgcn_rtsc_amplitude_worker import (final_shard, train_arm as new_train_arm,
                                                _binding as new_binding,
                                                _new_model)
from acl_hct.hgcn_rtsc_amplitude_protocol import MixedStreams
from acl_hct.hgcn_rtsc_worker import (train_arm as old_train_arm,
                                       _correction as old_model,
                                       _correction_binding as old_binding)
from acl_hct.hgcn_upstream import load_upstream
from acl_hct.mature_hgcn import MatureHGCN


ROOT = Path(__file__).parents[1]


def test_two_step_probe_matches_old_batch_and_graph_streams(tmp_path, monkeypatch):
    upstream_root = os.environ.get('ACL_HGCN_UPSTREAM_PATH')
    if not upstream_root:
        pytest.skip('explicit pinned HGCN checkout required')
    upstream = load_upstream(upstream_root)
    torch.manual_seed(103)
    base = MatureHGCN(upstream, 3, 4, 5).eval()
    original_hash = state_hash(base.state_dict())
    nodes = [str(i) for i in range(8)]
    neighbors = [[j for j in range(8) if j != i] for i in range(8)]
    positives = [(0, 1), (0, 2), (1, 3)]
    groups = torch.tensor([[[a, b], [4, b], [5, b], [6, b], [7, b]]
                           for a, b in positives], dtype=torch.long)
    data = {'nodes': nodes, 'neighbors': neighbors,
            'features': torch.randn(8, 3) * .1,
            'query_groups': groups,
            'labels': torch.tensor([[1., 0., 0., 0., 0.]] * len(groups))}
    read = lambda name: json.loads((ROOT / 'configs' / name).read_text())
    old, new = read('hgcn_rtsc_stage_a.json'), read('hgcn_rtsc_amplitude_v1.json')
    policy = read('mature_hgcn_replay_policy_v2.json')

    def fake_full(data, base, reference, binding, state_sha, original,
                  policy, prepared_root, output, device):
        assert state_hash(base.state_dict()) == original_hash
        return data['features'].to(device), [(0, 1)], {1: {0}}, object(), {'accepted': True}

    def fake_metrics(base, points, evaluator, valid, truth, ranking):
        assert torch.isfinite(points).all()
        return ({'micro_mrr': .2, 'direct_order': .6},
                {'elapsed_seconds': .01}, {'direct': {'metrics': {'score': .6}}})

    for module in ('acl_hct.hgcn_rtsc_worker',
                   'acl_hct.hgcn_rtsc_amplitude_worker'):
        monkeypatch.setattr(module + '.full_control', fake_full)
        monkeypatch.setattr(module + '.deadline', lambda: None)
        monkeypatch.setattr(module + '._metrics', fake_metrics)
        monkeypatch.setattr(module + '.write_archive',
                            lambda root, name, payload: {'manifest': name + '.json'})
    (tmp_path / 'old').mkdir()
    (tmp_path / 'new').mkdir()
    original_run = old_train_arm(data, copy.deepcopy(base), {}, {'seed': 11},
                                 original_hash, {}, policy, tmp_path, old, 11,
                                 'task_relation', tmp_path / 'old', torch.device('cpu'),
                                 probe_steps=2)
    revised = new_train_arm(data, copy.deepcopy(base), {}, {'seed': 11},
                            original_hash, {}, policy, tmp_path, new, old, 11,
                            'task_relation', tmp_path / 'new', torch.device('cpu'),
                            probe_steps=2)
    assert original_run['probe_weights_discarded'] and revised['probe_weights_discarded']
    assert revised['base_state_unchanged']
    assert original_run['module_init_seed63'] == revised['module_init_seed63']
    for key in ('batch_seed63', 'batch_indices', 'layer_plan_seed63',
                'layer_plan_graph_hash'):
        assert [r[key] for r in original_run['history']] == [r[key] for r in revised['history']]
    assert all(all(g > 0 for g in row['layer_grad_norm_before_clip'])
               for row in revised['history'])
    assert all('distribution' in d for readout in revised['probe_complete_sampled_ranking']
               for d in readout['diagnostics'])


def test_final_shard_loads_four_bound_checkpoints_and_pairs_all_five_conditions(tmp_path, monkeypatch):
    upstream_root = os.environ.get('ACL_HGCN_UPSTREAM_PATH')
    if not upstream_root:
        pytest.skip('explicit pinned HGCN checkout required')
    upstream = load_upstream(upstream_root)
    torch.manual_seed(77)
    base = MatureHGCN(upstream, 3, 4, 5).eval()
    state = state_hash(base.state_dict())
    neighbors = [[j for j in range(8) if j != i] for i in range(8)]
    features = torch.randn(8, 3) * .1
    data = {'neighbors': neighbors, 'features': features}
    read = lambda name: json.loads((ROOT / 'configs' / name).read_text())
    old, new = read('hgcn_rtsc_stage_a.json'), read('hgcn_rtsc_amplitude_v1.json')
    policy = read('mature_hgcn_replay_policy_v2.json')
    binding = {'seed': 11, 'original': 'fixed'}
    streams = MixedStreams()
    init_seed = streams.seed('module_init', 11)
    arm_inputs = {}
    for version in ('old', 'new'):
        for arm in ('task_only', 'task_relation'):
            name = version + '_' + arm
            model = (old_model(copy.deepcopy(base), old, init_seed) if version == 'old'
                     else _new_model(copy.deepcopy(base), new, init_seed))
            checkpoint_binding = (old_binding(old, policy, 11, arm, 0)
                                  if version == 'old' else new_binding(new, policy, 11, arm, 0))
            directory = tmp_path / name
            directory.mkdir()
            checkpoint = save_checkpoint(directory, 'selected.pt', model, checkpoint_binding)
            arm_inputs[name] = (directory, {'status': 'complete', 'seed': 11,
                                            'arm': arm, 'completed_steps': 1024,
                                            'original_best_binding': binding,
                                            'original_best_state_sha256': state,
                                            'selected': {'step': 0},
                                            'selected_checkpoint': checkpoint,
                                            'base_state_unchanged': True})

    class Evaluator:
        def evaluate(self, points):
            return {'direct': {'metrics': {'score': .6}}}

    def fake_full(*args):
        return features, [(0, 1)], {1: {0}}, Evaluator(), {'accepted': True}

    def fake_metrics(base, points, evaluator, valid, truth, ranking):
        return ({'micro_mrr': .2 + float(points.sum()) * .0001,
                 'direct_order': .6}, {'elapsed_seconds': .01},
                {'direct': {'metrics': {'score': .6}}})

    monkeypatch.setattr('acl_hct.hgcn_rtsc_amplitude_worker.full_control', fake_full)
    monkeypatch.setattr('acl_hct.hgcn_rtsc_amplitude_worker.deadline', lambda: None)
    monkeypatch.setattr('acl_hct.hgcn_rtsc_amplitude_worker._metrics', fake_metrics)
    monkeypatch.setattr('acl_hct.hgcn_rtsc_amplitude_worker.complete_ranking',
                        lambda *args: {'elapsed_seconds': .01})
    monkeypatch.setattr('acl_hct.hgcn_rtsc_amplitude_worker.qualify_full',
                        lambda *args: {'accepted': True})
    monkeypatch.setattr('acl_hct.hgcn_rtsc_amplitude_worker.write_archive',
                        lambda root, name, payload: {'manifest': name + '.json'})
    output = tmp_path / 'final'
    output.mkdir()
    result = final_shard(data, base, {'ranking': {'query_micro_mrr': .3},
                                      'native_ball_points': features.numpy()},
                         binding, state, {}, policy, tmp_path, new, old, 11, 0,
                         output, torch.device('cpu'), arm_inputs)
    assert result['status'] == 'complete' and len(result['rows']) == 60
    assert len(result['full_arm_qualifications']) == 4
    assert {r['arm'] for r in result['rows']} == {
        'S', 'old_task_only', 'old_task_relation',
        'new_task_only', 'new_task_relation'}
    assert result['base_state_unchanged']

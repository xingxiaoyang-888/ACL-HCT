"""Tiny real update loop; only full-data I/O and all-entity ranking are replaced."""
import copy
import json
import os
from pathlib import Path

import pytest
import torch

from acl_hct.hgcn_evidence import state_hash
from acl_hct.hgcn_rtsc_worker import train_arm
from acl_hct.hgcn_upstream import load_upstream
from acl_hct.mature_hgcn import MatureHGCN


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize('steps', [2])
def test_two_arms_share_batches_and_graphs_and_only_update_modules(tmp_path, monkeypatch, steps):
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
    settings = json.loads((ROOT / 'configs/hgcn_rtsc_stage_a.json').read_text())
    policy = json.loads((ROOT / 'configs/mature_hgcn_replay_policy_v2.json').read_text())

    def fake_full_control(data, base, reference, binding, state_sha, original,
                          policy, prepared_root, output, device):
        assert state_hash(base.state_dict()) == original_hash
        return data['features'].to(device), [(0, 1)], {1: {0}}, object(), {'accepted': True}

    def fake_metrics(base, points, evaluator, valid, truth, ranking):
        assert torch.isfinite(points).all()
        return ({'micro_mrr': .2, 'direct_order': .6},
                {'elapsed_seconds': .01}, {'direct': {'metrics': {'score': .6}}})

    monkeypatch.setattr('acl_hct.hgcn_rtsc_worker.full_control', fake_full_control)
    monkeypatch.setattr('acl_hct.hgcn_rtsc_worker.deadline', lambda: None)
    monkeypatch.setattr('acl_hct.hgcn_rtsc_worker._metrics', fake_metrics)
    monkeypatch.setattr('acl_hct.hgcn_rtsc_worker.write_archive',
                        lambda root, name, payload: {'manifest': name + '.json'})
    outputs = {}
    for arm in ('task_only', 'task_relation'):
        output = tmp_path / arm
        output.mkdir()
        model = copy.deepcopy(base)
        outputs[arm] = train_arm(data, model, {}, {'seed': 11}, original_hash,
                                 {}, policy, tmp_path, settings, 11, arm, output,
                                 torch.device('cpu'), probe_steps=steps)
        assert state_hash(model.state_dict()) == original_hash
        assert outputs[arm]['base_state_unchanged'] is True
        assert outputs[arm]['completed_steps'] == steps
        assert outputs[arm]['probe_weights_discarded'] is True
        assert len(outputs[arm]['probe_complete_sampled_ranking']) == 2
        assert all(row['module_update_norm'] > 0 for row in outputs[arm]['history'])
        assert all(all(grad > 0 for grad in row['layer_grad_norm_before_clip'])
                   for row in outputs[arm]['history'])
        assert all(torch.isfinite(torch.tensor([row['total_loss'], row['module_grad_norm_before_clip']])).all()
                   for row in outputs[arm]['history'])
    task, relation = outputs['task_only']['history'], outputs['task_relation']['history']
    assert [row['batch_indices'] for row in task] == [row['batch_indices'] for row in relation]
    assert [row['layer_plan_seed63'] for row in task] == [row['layer_plan_seed63'] for row in relation]
    assert [row['layer_plan_graph_hash'] for row in task] == [row['layer_plan_graph_hash'] for row in relation]
    assert all(row['relation_loss'] == 0 for row in task)
    assert all(row['relation_loss'] >= 0 for row in relation)
    assert outputs['task_only']['module_init_seed63'] == outputs['task_relation']['module_init_seed63']

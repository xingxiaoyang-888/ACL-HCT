"""Meaningful v2 rank-sensitive qualification and mixed-lineage regression."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from acl_hct.hgcn_panels import HierarchyPanel
from acl_hct.hgcn_replay import (ORIGINAL_SOURCE, POLICY_SHA256, POLICY_V2_SHA256,
                                 qualify_full, validate_policy)
from acl_hct.hgcn_registration import canonical


def policies():
    root = Path(__file__).parents[1] / 'configs'
    config = json.loads((root / 'mature_hgcn_validation.json').read_bytes())
    old = json.loads((root / 'mature_hgcn_replay_policy.json').read_bytes())
    new = json.loads((root / 'mature_hgcn_replay_policy_v2.json').read_bytes())
    return config, old, new


def rank(value):
    return {'status': 'complete', 'expected_queries': 1, 'completed_queries': 1,
            'completed_children': 1, 'query_micro_mrr': 1 / value,
            'child_macro_mrr': 1 / value, 'hits': {'1': 0., '3': 0., '10': 0.},
            'rows': [{'parent': 1, 'child': 2, 'rank': float(value), 'candidates': 39}]}


def test_v2_reports_v1_mrr_failure_but_retains_every_other_hard_gate():
    config, old, new = policies()
    assert validate_policy(old, config) == POLICY_SHA256
    assert validate_policy(new, config) == POLICY_V2_SHA256
    points = np.zeros((40, 2), dtype=np.float32)
    points[:, 0] = np.arange(40, dtype=np.float32) / 100
    view = SimpleNamespace(nodes=[str(i) for i in range(40)], root='0', reachable={str(i) for i in range(40)})
    panel = {'pool_size': 1, 'rows': [{'id': '2', 'index': 2, 'inclusion_probability': 1., 'pool_mean_weight': 1.}],
             'relations': [{'child': '2', 'direct_parents': ['1'], 'positive_distant_ancestors': ['0']}]}
    evaluator = HierarchyPanel(view, panel, points)
    binding = {'source_commit': ORIGINAL_SOURCE, 'scope': 'artificial rank gate'}
    plan = {'indices': np.array([[0], [0]], dtype=np.int64), 'weights_fp64': np.ones(1), 'fanout': None}
    reference = {'native_ball_points': points, 'ranking': rank(30), 'binding': binding,
                 'model_state_sha256': 'a' * 64, 'full_plan': plan}
    miniature = {'prepared': {'nodes_count': 40}, 'model': {'hidden': 2, 'c': 1.},
                 'hierarchy': {'registration': {'coverage': {'direct': {'covered_pairs': 1},
                                                             'distant': {'covered_pairs': 1}}}}}
    args = (torch.from_numpy(points), rank(29), reference, evaluator, miniature)
    kwargs = ([(1, 2)], {2: {1}}, 'a' * 64, binding, plan)
    v1 = qualify_full(*args, old, *kwargs)
    v2 = qualify_full(*args, new, *kwargs)
    assert not v1['accepted'] and {v['metric'] for v in v1['violations']} == {'micro_mrr_abs_delta', 'macro_mrr_abs_delta'}
    assert v2['accepted'] and v2['violations'] == [] and not v2['v1_diagnostic']['v1_accepted']
    assert v2['v1_diagnostic']['v1_policy_sha256'] == POLICY_SHA256
    assert v2['v1_diagnostic']['signed_micro_mrr_drift'] == 1 / 29 - 1 / 30
    assert v2['v1_diagnostic']['signed_macro_mrr_drift'] == 1 / 29 - 1 / 30
    assert v2['v1_diagnostic']['changed_query_contributions'][0]['micro_mrr'] == 1 / 29 - 1 / 30
    changed = points.copy(); changed[2, 0] += np.float32(1e-4)
    rejected = qualify_full(torch.from_numpy(changed), rank(29), reference, evaluator, miniature,
                            new, *kwargs)
    assert not rejected['accepted'] and 'point_max_abs' in {v['metric'] for v in rejected['violations']}
    wrong = dict(new); wrong['limits'] = {**new['limits'], 'micro_mrr_abs_delta': 1.4e-7}
    with pytest.raises(ValueError, match='exact separately frozen'):
        validate_policy(wrong, config)


def test_retrieval_sensitivity_never_claims_non_significant_damage():
    from acl_hct.hgcn_analysis import retrieval_sensitivity_decision
    old = {'mean': -0.1, 'reject_holm': False}
    adjusted = {'mean': -0.12, 'reject_holm': False}
    result = retrieval_sensitivity_decision(old, adjusted, 0.01)
    assert result['same_direction'] and result['same_Holm_decision']
    assert not result['robust_retrieval_damage_claim_allowed']
    assert not retrieval_sensitivity_decision({**old, 'reject_holm': True}, adjusted, 0.01)[
        'robust_retrieval_damage_claim_allowed']
    assert retrieval_sensitivity_decision({**old, 'reject_holm': True},
                                          {**adjusted, 'reject_holm': True}, 0.01)[
        'robust_retrieval_damage_claim_allowed']

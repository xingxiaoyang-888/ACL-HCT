"""Stage A pre-registration, stream independence and complete inference families."""
import json
from pathlib import Path

import pytest

from acl_hct.hgcn_rtsc_protocol import (NamedStreams, analyze_final_shards,
                                        choose_checkpoint, final_families, validate_config)
from acl_hct.hgcn_rtsc_worker import _correction_binding
from acl_hct.hgcn_evidence import load_checkpoint, save_checkpoint
import torch


ROOT = Path(__file__).parents[1]


def test_exact_registered_configuration_and_named_streams():
    read = lambda name: json.loads((ROOT / 'configs' / name).read_text(encoding='utf-8'))
    config = read('hgcn_rtsc_stage_a.json')
    assert len(validate_config(config, read('mature_hgcn_validation.json'),
                               read('mature_hgcn_replay_policy_v2.json'))) == 64
    stream = NamedStreams()
    first = stream.seed('train_batch', 11, step=1, fanout=4)
    assert first == stream.seed('train_batch', 11, step=1, fanout=4)
    assert first != stream.seed('train_batch', 23, step=1, fanout=4)
    assert first != stream.seed('train_layer', 11, step=1, fanout=4, layer=0)
    assert first != stream.seed('train_layer', 11, step=1, fanout=4, layer=1)
    assert first != stream.seed('selection_layer', 11, step=1, fanout=4, repeat=0, layer=0)
    assert len(stream.manifest()) == 5
    config['training']['lr'] = .004
    with pytest.raises(ValueError, match='exact separate'):
        validate_config(config, read('mature_hgcn_validation.json'),
                        read('mature_hgcn_replay_policy_v2.json'))


def test_hierarchy_gate_precedes_mrr_and_ties_keep_first():
    rows = [{'step': step, 'direct_order_delta': delta, 'micro_mrr': mrr}
            for step, delta, mrr in zip((0, 256, 512, 768, 1024),
                                        ((0., 0.), (-.003, -.003), (-.002, -.002),
                                         (-.002, -.002), (-.004, -.003)),
                                        ((.2, .2), (.9, .9), (.3, .3), (.3, .3), (.8, .8)))]
    assert choose_checkpoint(rows)['step'] == 512
    rows[2]['direct_order_delta'] = [-.003, -.003]
    rows[3]['direct_order_delta'] = [-.003, -.003]
    assert choose_checkpoint(rows)['step'] == 0
    rows[0]['direct_order_delta'] = [-.003, -.003]
    assert choose_checkpoint(rows) is None


def test_both_final_holm_families_require_complete_paired_grid():
    records = {}
    for seed in (11, 23):
        for fanout in (4, 8, 16):
            for repeat in range(16):
                for arm, shift in (('S', 0.), ('task_only', .01), ('task_relation', .02)):
                    records[seed, fanout, repeat, arm] = {
                        'micro_mrr': .2 + shift + repeat * .00001,
                        'direct_order': .6 + shift + repeat * .00001}
    primary, secondary = final_families(records)
    assert len(primary) == 24 and len(secondary) == 12
    assert all(row['n'] == 16 for row in primary + secondary)
    records.pop((11, 4, 0, 'S'))
    with pytest.raises(ValueError, match='complete paired'):
        final_families(records)


def test_checkpoint_binding_reconstructed_from_frozen_config(tmp_path):
    read = lambda name: json.loads((ROOT / 'configs' / name).read_text(encoding='utf-8'))
    config, policy = read('hgcn_rtsc_stage_a.json'), read('mature_hgcn_replay_policy_v2.json')
    expected = _correction_binding(config, policy, 11, 'task_relation', 256)
    checkpoint = save_checkpoint(tmp_path, 'step256.pt', torch.nn.Linear(2, 2), expected)
    assert load_checkpoint(tmp_path, checkpoint, expected)['binding'] == expected
    altered = dict(checkpoint)
    altered['binding'] = {**expected, 'arm': 'task_only'}
    with pytest.raises(ValueError, match='binding'):
        load_checkpoint(tmp_path, altered, expected)


def test_final_analysis_rejects_missing_shard_and_uses_paired_repeats():
    shards = []
    for seed in (11, 23):
        for start in (0, 4, 8, 12):
            rows = []
            for fanout in (4, 8, 16):
                for repeat in range(start, start + 4):
                    for arm, delta in (('S', 0.), ('task_only', .01), ('task_relation', .02)):
                        rows.append({'seed': seed, 'fanout': fanout, 'repeat': repeat,
                                     'arm': arm, 'micro_mrr': .4 + repeat * .0001 + delta * (1 + repeat * .001),
                                     'direct_order': .6 + repeat * .0001 + delta * (1 + repeat * .001)})
            shards.append({'status': 'complete', 'seed': seed, 'repeat_start': start,
                           'base_state_unchanged': True,
                           'original_F_qualification': {'accepted': True},
                           'full_arm_qualifications': {'task_only': {'accepted': True},
                                                       'task_relation': {'accepted': True}},
                           'fixed_F': {'micro_mrr': .5, 'direct_order': .7},
                           'selected': {'task_only': {'step': 256, 'checkpoint': {'sha256': 'a' * 64}},
                                        'task_relation': {'step': 512, 'checkpoint': {'sha256': 'b' * 64}}},
                           'rows': rows})
    result = analyze_final_shards(shards)
    assert len(result['primary_holm24']) == 24
    assert len(result['relation_vs_task_holm12']) == 12
    assert result['joint_recovery_rule_met'] is True
    with pytest.raises(ValueError, match='eight'):
        analyze_final_shards(shards[:-1])
    shards[1]['selected']['task_only']['step'] = 768
    with pytest.raises(ValueError, match='selected module'):
        analyze_final_shards(shards)

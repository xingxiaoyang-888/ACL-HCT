"""Exact amplitude registration, mixed streams and complete statistical families."""
import json
from pathlib import Path
from argparse import Namespace

import pytest

from acl_hct.hgcn_rtsc_amplitude_protocol import (MixedStreams,
    analyze_final_shards, final_families, one_sided_summary, validate_config)
from acl_hct.hgcn_rtsc_protocol import NamedStreams
from acl_hct.hgcn_rtsc_amplitude_worker import load_arm_inputs


ROOT = Path(__file__).parents[1]


def read(name):
    return json.loads((ROOT / 'configs' / name).read_text(encoding='utf-8'))


def test_exact_config_and_old_train_new_final_streams():
    config = read('hgcn_rtsc_amplitude_v1.json')
    digest = validate_config(config, read('mature_hgcn_validation.json'),
                             read('mature_hgcn_replay_policy_v2.json'),
                             read('hgcn_rtsc_stage_a.json'))
    assert len(digest) == 64
    mixed, old = MixedStreams(), NamedStreams()
    for phase in ('module_init', 'train_batch', 'train_layer', 'selection_layer'):
        assert mixed.seed(phase, 11, step=4, fanout=4, repeat=1, layer=0) == old.seed(
            phase, 11, step=4, fanout=4, repeat=1, layer=0)
    assert mixed.seed('final_layer', 11, fanout=4, repeat=0, layer=0) != old.seed(
        'final_layer', 11, fanout=4, repeat=0, layer=0)
    config['module']['scale'] = 5
    with pytest.raises(ValueError, match='exact separate'):
        validate_config(config, read('mature_hgcn_validation.json'),
                        read('mature_hgcn_replay_policy_v2.json'),
                        read('hgcn_rtsc_stage_a.json'))


def test_one_sided_zero_variance_is_conservatively_nonrejecting():
    same = one_sided_summary([.01] * 16)
    assert same['p'] == 1. and same['marginal_student95'] is None
    same_margin = one_sided_summary([-.001] * 16, margin=.002)
    assert same_margin['p'] == 1. and same_margin['mean'] == -.001
    varying = one_sided_summary([.001 + i * .00001 for i in range(16)])
    assert varying['p'] < .05 and varying['mean'] > 0


def test_complete_five_condition_paired_families_and_missing_rejection():
    records = {}
    for seed in (11, 23):
        for fanout in (4, 8, 16):
            for repeat in range(16):
                jitter = repeat * .00001
                for arm, shift in [('S', 0.), ('old_task_only', .001),
                                   ('old_task_relation', .0012),
                                   ('new_task_only', .0015),
                                   ('new_task_relation', .0017)]:
                    records[seed, fanout, repeat, arm] = {
                        'micro_mrr': .2 + jitter + shift * (1 + repeat * .001),
                        'direct_order': .6 + jitter + shift * (1 + repeat * .001)}
    main, support, relation = final_families(records)
    assert (len(main), len(support), len(relation)) == (24, 24, 12)
    assert all(r['n'] == 16 and r['df'] == 15 for r in main + support + relation)
    assert {r['test'] for r in main} == {
        'one-sided hierarchy noninferiority', 'one-sided MRR superiority'}
    hierarchy = next(r for r in main if r['metric'] == 'direct_order')
    assert hierarchy['noninferiority_margin'] == .002
    assert hierarchy['mean'] < .002  # unshifted effect, not mean + margin
    records.pop((11, 4, 0, 'old_task_only'))
    with pytest.raises(ValueError, match='complete five-condition'):
        final_families(records)


def test_analysis_rejects_missing_or_unqualified_shard():
    with pytest.raises(ValueError, match='eight final'):
        analyze_final_shards([])


def test_final_cli_four_result_flags_map_to_exact_old_new_conditions(tmp_path):
    paths = {}
    for label in ('old_task', 'old_relation', 'new_task', 'new_relation'):
        path = tmp_path / (label + '.json')
        path.write_text(json.dumps({'identity': label}))
        paths[label + '_result'] = path
    selected = load_arm_inputs(Namespace(**paths))
    assert {k: v[1]['identity'] for k, v in selected.items()} == {
        'old_task_only': 'old_task', 'old_task_relation': 'old_relation',
        'new_task_only': 'new_task', 'new_task_relation': 'new_relation'}

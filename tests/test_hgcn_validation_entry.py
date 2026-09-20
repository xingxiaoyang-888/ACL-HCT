import copy
import json
import os
from pathlib import Path
import sys

import pytest

from acl_hct.hgcn_validation_entry import main, supervise
from acl_hct.hgcn_validation_registration import canonical, role, source_hashes, validate_config, verify_release


@pytest.fixture
def config():
    return json.loads((Path(__file__).parents[1] / 'configs/mature_hgcn_validation.json').read_bytes())


def test_exact_candidate_and_baseline_source19_gate(config):
    assert validate_config(config) == canonical(config)
    assert len(source_hashes()) == 31
    changed = copy.deepcopy(config); changed['training']['steps'] = 1024
    with pytest.raises(ValueError, match='exact'):
        validate_config(changed)


def test_registered_roles_do_not_permit_seed_or_repeat_extensions(config):
    assert role(config, 'train', 11)['worker_seconds'] == 2400
    assert role(config, 'eval', 23, 12)['allocation_gpu_seconds'] == 540
    assert role(config, 'official')['worker_seconds'] == 480
    assert role(config, 'replay', 23)['allocation_gpu_seconds'] == 300
    for args in (('train', 99, None), ('eval', 11, 1), ('official', 11, None), ('train', 11, 0)):
        with pytest.raises(ValueError):
            role(config, *args)


def test_no_execute_only_static_and_missing_release_fails(config, monkeypatch, capsys):
    path = Path(__file__).parents[1] / 'configs/mature_hgcn_validation.json'
    monkeypatch.setattr(sys, 'argv', ['entry', '--config', str(path)])
    assert main() == 0 and 'static_only' in capsys.readouterr().out
    monkeypatch.setattr(sys, 'argv', ['entry', '--config', str(path), '--execute', '--phase', 'train'])
    assert main() == 2


def test_release_requires_final_source_quality_ledger_and_phase_inputs(config, tmp_path, monkeypatch):
    import acl_hct.hgcn_validation_registration as module
    head = '1' * 40; job = role(config, 'eval', 11, 0)
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: head + '\n')
    record = {'protocol': config['protocol'], 'accepted': True, 'authorized_scope': 'bounded_mature_hgcn_validation',
              'phase': 'eval', 'source_commit': head, 'config_sha256': canonical(config),
              'upstream_commit': config['upstream_commit'], 'source_sha256_normalized_lf': source_hashes(), **job,
              'quality_gates': {k: {'accepted': True, 'source_commit': head,
                                    'config_sha256': canonical(config), 'report_sha256': '2' * 64}
                                for k in ('cpu_fixture', 'cuda_fixture')},
              'inputs': {'training_run_sha256': '3' * 64, 'best_checkpoint_sha256': '4' * 64},
              'accounting': {'spent_gpu_seconds': 63, 'reserved_including_this_gpu_seconds': 540,
                             'concurrent_gpu_jobs_including_this': 1}}
    record['baseline_acceptance'] = {**{k: True for k in
        ('accepted', 'complete_4096_and_scheduled_valid_reviewed', 'matching_best_reload_reviewed',
         'learning_state_reviewed', 'full_hierarchy_reviewed', 'order_above_half_necessary_not_sufficient')},
        **record['inputs'], 'review_sha256': '5' * 64}
    policy = json.loads((Path(__file__).parents[1] / 'configs/mature_hgcn_replay_policy.json').read_bytes())
    record['replay_policy_sha256'] = canonical(policy)
    for gate in record['quality_gates'].values(): gate['replay_policy_sha256'] = canonical(policy)
    record['accounting']['spent_gpu_seconds'] = 4453
    record['baseline_acceptance'].update(replay_policy_sha256=canonical(policy),
        original_training_run_sha256=policy['origins']['11']['training_run_sha256'])
    path = tmp_path / 'release.json'
    def write(value):
        path.write_text(json.dumps(value), encoding='utf-8')
    write(record)
    assert verify_release(config, 'eval', head, path, 11, 0, policy)['inputs'] == record['inputs']
    with pytest.raises(ValueError, match='policy required'):
        verify_release(config, 'eval', head, path, 11, 0)
    for mutate in (lambda r: r['quality_gates']['cuda_fixture'].update(accepted=False),
                   lambda r: r['accounting'].update(spent_gpu_seconds=10800),
                   lambda r: r['accounting'].update(spent_gpu_seconds=float('nan')),
                   lambda r: r['inputs'].pop('best_checkpoint_sha256'),
                   lambda r: r['baseline_acceptance'].update(learning_state_reviewed=False),
                   lambda r: r['baseline_acceptance'].update(original_training_run_sha256='0' * 64),
                   lambda r: r.update(replay_policy_sha256='0' * 64),
                   lambda r: r.update(selectors={'seed': 11, 'repeat_start': 4}),
                   lambda r: r['source_sha256_normalized_lf'].update({'acl_hct/hgcn_geometry.py': '0' * 64})):
        bad = copy.deepcopy(record); mutate(bad); write(bad)
        with pytest.raises(ValueError):
            verify_release(config, 'eval', head, path, 11, 0, policy)
    v2 = json.loads((Path(__file__).parents[1] / 'configs/mature_hgcn_replay_policy_v2.json').read_bytes())
    amended = copy.deepcopy(record)
    amended['replay_policy_sha256'] = canonical(v2)
    amended['inputs']['training_run_sha256'] = v2['rank_sensitive_amendment']['v1_accepted_replays']['11']
    amended['inputs']['best_checkpoint_sha256'] = v2['origins']['11']['best_checkpoint_sha256']
    for gate in amended['quality_gates'].values():
        gate['replay_policy_sha256'] = canonical(v2)
    amended['baseline_acceptance'].update(training_run_sha256=amended['inputs']['training_run_sha256'],
        best_checkpoint_sha256=amended['inputs']['best_checkpoint_sha256'],
        replay_policy_sha256=canonical(policy), amendment_policy_sha256=canonical(v2))
    write(amended)
    assert verify_release(config, 'eval', head, path, 11, 0, v2)['inputs'] == amended['inputs']
    for mutate in (lambda r: r['inputs'].update(training_run_sha256='0' * 64),
                   lambda r: r['baseline_acceptance'].update(amendment_policy_sha256='0' * 64),
                   lambda r: r['baseline_acceptance'].update(replay_policy_sha256=canonical(v2))):
        bad = copy.deepcopy(amended); mutate(bad); write(bad)
        with pytest.raises(ValueError):
            verify_release(config, 'eval', head, path, 11, 0, v2)


def test_direct_worker_timeout_and_output_nonoverwrite(tmp_path):
    output = tmp_path / 'run'
    code = 'import os,time; assert int(os.environ["ACL_HGCN_VALIDATION_PID"]) == os.getppid(); time.sleep(20)'
    assert supervise([sys.executable, '-c', code], output, 1, {'test': 'artificial'}) == 124
    result = json.loads((output / 'supervisor.json').read_bytes())
    assert result['status'] == 'timeout' and result['worker_exit_code'] != 0
    with pytest.raises(ValueError, match='fresh'):
        supervise([sys.executable, '-c', 'pass'], output, 1, {})
    with pytest.raises(ValueError, match='2400'):
        supervise([], tmp_path / 'never', 2401, {})

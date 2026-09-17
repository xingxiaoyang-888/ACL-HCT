"""Synthetic head/archive integration; no real inputs, CUDA, training or samples."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from acl_hct import radial_entry as entry
from acl_hct import radial_recovery as recovery
from acl_hct.diagnostic_archive import read_archive, write_archive
from acl_hct.encoder_matched_control import validate_ranking
from acl_hct.frozen_recovery import _metrics, promoted
from acl_hct.geometry import exp, tangent
from acl_hct.radial_analysis import merge_shards, verify_repeat
from acl_hct.radial_components import RadialBiasIntervention
from acl_hct.vector_structure import RadialPanel

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / 'configs/e2_radial_component_pilot.json').read_bytes())


def ld(a, b):
    return (a[..., 1:] * b[..., 1:]).sum(axis=-1) - a[..., 0] * b[..., 0]


def test_head_only_numpy_scores_and_no_rng_draw():
    _, _, _, native, weights, data, _, _ = recovery.synthetic_data(config(), 11, 'mixed')
    before = torch.get_rng_state().clone()
    head = recovery.ArchivedRelationHead(weights, dimension=128)
    assert torch.equal(before, torch.get_rng_state())
    assert len(list(head.parameters())) == 4 and not hasattr(head, 'encode')
    pairs = torch.cartesian_prod(torch.arange(16), torch.arange(16))
    actual = head.score(native, pairs).detach().numpy().reshape(16, 16)
    point = native.numpy()
    origin = np.zeros_like(point)
    origin[:, 0] = 1
    delta = point - origin
    z = np.maximum(ld(delta, delta) / 2, 0)
    safe = np.maximum(z, 1e-8)
    factor = np.where(z < 1e-5, 1 - z / 3 + 2 * z * z / 15,
                      np.arccosh(1 + safe) / np.sqrt(safe * (2 + safe)))
    inputs = factor[:, None] * delta[:, 1:]
    parent = np.repeat(inputs, 16, axis=0)
    child = np.tile(inputs, (16, 1))
    features = np.concatenate((parent, child, parent - child), axis=1)
    hidden = np.maximum(features @ weights['relation_head.0.weight'].numpy().T + weights['relation_head.0.bias'].numpy(), 0)
    expected = hidden @ weights['relation_head.2.weight'].numpy().T + weights['relation_head.2.bias'].numpy()
    np.testing.assert_allclose(actual, expected.reshape(16, 16), atol=1e-5, rtol=0)
    assert all(p.grad is None and not p.requires_grad for p in head.parameters())


@pytest.mark.parametrize('seed,case', [(11, 'mixed'), (11, 'zero'), (23, 'mixed'), (23, 'zero')])
def test_full_fixture_archive_and_numpy_geometry(tmp_path, seed, case):
    checks, descriptor = recovery.fixture_case(config(), seed, case, torch.device('cpu'), tmp_path, time.monotonic() + 60)
    assert all(checks.values())
    archive = read_archive(tmp_path, descriptor)
    x, sample = archive['base'], archive['sample64']
    delta = sample - x
    z = np.maximum(ld(delta, delta) / 2, 0)
    safe = np.maximum(z, 1e-8)
    factor = np.where(z < 1e-5, 1 - z / 3 + 2 * z * z / 15,
                      np.arccosh(1 + safe) / np.sqrt(safe * (2 + safe)))
    error = factor[:, None] * (delta + ld(x, delta)[:, None] * x)
    for name in ('R', 'T'):
        v = error - archive['removed_fields'][name]
        v = v + ld(x, v)[:, None] * x
        z2 = np.maximum(ld(v, v), 0)
        length = np.sqrt(np.maximum(z2, 1e-8))
        a = np.where(z2 < 1e-6, 1 + z2 / 2 + z2 * z2 / 24, np.cosh(length))
        b = np.where(z2 < 1e-6, 1 + z2 / 6 + z2 * z2 / 120, np.sinh(length) / length)
        moved = a[:, None] * x + b[:, None] * v
        zero = np.maximum(ld(archive['removed_fields'][name], archive['removed_fields'][name]), 0) == 0
        moved[zero] = sample[zero]
        np.testing.assert_allclose(archive['FP64_reference_points'][name], moved, atol=1e-10, rtol=0)
        assert np.array_equal(archive['native_points'][name][zero], archive['native_points']['S'][zero])
    if case == 'mixed':
        assert np.array_equal(archive['native_points']['T'][1], archive['native_points']['S'][1])
        assert np.array_equal(archive['native_points']['R'][4], archive['native_points']['S'][4])


@pytest.mark.parametrize('case', ['missing', 'dtype', 'shape', 'nonfinite'])
def test_head_invalid_archived_parameters_rejected(case):
    weights = recovery.synthetic_data(config(), 11, 'zero')[4]
    if case == 'missing':
        weights.pop('relation_head.2.bias')
    elif case == 'dtype':
        weights['relation_head.0.bias'] = weights['relation_head.0.bias'].double()
    elif case == 'shape':
        weights['relation_head.2.weight'] = torch.zeros(2, 128)
    else:
        weights['relation_head.2.bias'][0] = float('nan')
    with pytest.raises(ValueError):
        recovery.ArchivedRelationHead(weights, dimension=128)


def test_exact_F_rows_and_candidates_gate(tmp_path):
    _, _, _, native, weights, data, _, _ = recovery.synthetic_data(config(), 11, 'mixed')
    head = recovery.ArchivedRelationHead(weights)
    expected = recovery.rank_points(head, native, data, config(), time.monotonic() + 60)
    assert recovery.rank_points(head, native, data, config(), time.monotonic() + 60, expected=expected)['rows'] == expected['rows']
    wrong = copy.deepcopy(expected)
    wrong['rows'].reverse()  # Same MRR and coverage, different frozen ordering.
    with pytest.raises(recovery.RankingGateError, match='exactly'):
        recovery.rank_points(head, native, data, config(), time.monotonic() + 60, expected=wrong)
    wrong = copy.deepcopy(expected)
    wrong['rows'][0]['candidates'] -= 1
    with pytest.raises(recovery.RankingGateError, match='candidate'):
        recovery.rank_points(head, native, data, config(), time.monotonic() + 60, expected=wrong)


def test_pre_result_budget_predicate_and_invalid_timing():
    assert recovery.feasibility(24, time.monotonic() + 700, config())['passed'] is True
    assert recovery.feasibility(24, time.monotonic() + 500, config())['passed'] is False
    with pytest.raises(ValueError):
        recovery.feasibility(float('nan'), time.monotonic() + 700, config())


def test_finite_radial_and_angle_diagnostics_with_undefined_mask():
    from acl_hct.geometry import from_spatial
    sample = from_spatial(torch.tensor([[.5, 0.], [.5, 0.], [0., 0.]], dtype=torch.float64))
    moved = from_spatial(torch.tensor([[.6, 0.], [0., .5], [0., 0.]], dtype=torch.float64))
    anchor = sample[2]
    floor = torch.full((3,), 1e-6, dtype=torch.float64)
    result = recovery.finite_move_diagnostics(anchor, sample, moved, floor, floor[2])
    assert result['radial_change'].tolist() == pytest.approx([.1, 0., 0.], abs=1e-12)
    assert result['direction_defined'].tolist() == [True, True, False]
    assert result['direction_angle_radians'][:2].tolist() == pytest.approx([0., math.pi / 2], abs=1e-7)
    assert torch.isnan(result['direction_angle_radians'][2]) and result['undefined_count'] == 1
    unchanged = recovery.finite_move_diagnostics(anchor, sample, sample, floor, floor[2])
    assert torch.equal(unchanged['direction_angle_radians'][:2], torch.zeros(2, dtype=torch.float64))


@pytest.mark.parametrize('case', ['positive', 'native', 'cast', 'sample_identity', 'rng'])
def test_CPU_archive_reproduction_rejects_drift(tmp_path, case):
    base, bias, floor, sample, weights, data, view, spec = recovery.synthetic_data(config(), 11, 'mixed')
    adapter = RadialBiasIntervention(base, bias, floor, 0)
    panel = RadialPanel(view, spec, base, {'V': list(range(16))}, 1., floor)
    head = recovery.ArchivedRelationHead(weights)
    _, _, native, casts = recovery.cast_components(adapter, sample, floor, config())
    points = {'S': sample, **native}
    ranks = {k: recovery.rank_points(head, v, data, config(), time.monotonic() + 60) for k, v in points.items()}
    structure = {k: panel.evaluate(promoted(v, 1.)) for k, v in points.items()}
    metrics = {k: _metrics(structure[k], ranks[k]) for k in points}
    identity = recovery.tensor_identity(sample)
    observation = {'repeat': 0, 'plan_hash': 'fixture-only', 'rng': {'fixture': 11},
                   'original_S_identity': identity, 'metrics': metrics}
    saved = {'repeat': 0, 'plan_hash': 'fixture-only', 'rng': {'fixture': 11},
             'native_points': {'S': sample}, 'rankings': {'S': ranks['S']}, 'structure': {'S': structure['S']}}
    payload = {**observation, 'native_points': native, 'rankings': {k: ranks[k] for k in ('R', 'T')},
               'structure': {k: structure[k] for k in ('R', 'T')}, 'casting': casts}
    payload = read_archive(tmp_path, write_archive(tmp_path, 'repeat', payload))
    saved = read_archive(tmp_path, write_archive(tmp_path, 'old-repeat', saved))
    if case == 'native':
        payload['native_points']['R'][5, 5] += .0001
    elif case == 'cast':
        payload['casting']['T']['error'][5] += .001
    elif case == 'sample_identity':
        payload['original_S_identity'] = {}
    elif case == 'rng':
        payload['rng']['fixture'] = 23
    if case == 'positive':
        assert verify_repeat(payload, observation, saved, panel, data, adapter, floor, config()) == metrics
    else:
        with pytest.raises(ValueError):
            verify_repeat(payload, observation, saved, panel, data, adapter, floor, config())


def statistical_fixture():
    shards = []
    for seed in (11, 23):
        for start in (0, 8):
            observations = []
            for i in range(start, start + 8):
                metrics = {name: {k: .5 + scale * ((i % 5) - 2) / 1024 + seed / 2048
                                  for k in ('direct', 'distant', 'micro_mrr')}
                           for name, scale in [('S', 1), ('R', 3), ('T', -2)]}
                differences = {f'{a}-{b}': {k: metrics[a][k] - metrics[b][k] for k in metrics[a]}
                               for a, b in [('R', 'S'), ('T', 'S'), ('R', 'T')]}
                observations.append({'repeat': i, 'metrics': metrics, 'paired_differences': differences})
            shards.append({'status': 'complete', 'protocol': entry.PROTOCOL, 'config_sha256': entry.CONFIG_SHA256,
                           'seed': seed, 'repeat_range': [start, start + 8], 'identity': {'seed': seed}, 'observations': observations})
    return shards


def test_statistics_entire_18_family_and_independent_SE():
    result = merge_shards(statistical_fixture())
    assert len(result['primary_comparisons']) == result['family_size'] == 18
    assert {r['comparison'] for r in result['primary_comparisons']} == {'R-S', 'T-S', 'R-T'}
    for row in result['primary_comparisons']:
        values = row['differences']
        mean = math.fsum(values) / 16
        se = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / 15 / 16)
        assert row['mean'] == pytest.approx(mean, abs=1e-16)
        assert row['mc_se'] == pytest.approx(se, abs=1e-16)
        assert row['df'] == 15
        assert row['marginal_95_interval'] == pytest.approx([mean - 2.1314495455597757 * se, mean + 2.1314495455597757 * se])
    ordered = sorted(result['primary_comparisons'], key=lambda r: r['two_sided_p'])
    previous = 0.
    for i, row in enumerate(ordered):
        previous = max(previous, min(1., (18 - i) * row['two_sided_p']))
        assert row['holm_adjusted_p'] == previous


def test_zero_observed_variance_has_no_interval_or_significance():
    shards = statistical_fixture()
    for shard in shards:
        for row in shard['observations']:
            row['metrics'] = {name: {k: .5 for k in ('direct', 'distant', 'micro_mrr')} for name in ('S', 'R', 'T')}
            row['paired_differences'] = {key: {k: 0. for k in ('direct', 'distant', 'micro_mrr')} for key in ('R-S', 'T-S', 'R-T')}
    for row in merge_shards(shards)['primary_comparisons']:
        assert row['inferable'] is False and row['zero_observed_variance'] is True
        assert row['marginal_95_interval'] is None and row['two_sided_p'] == row['holm_adjusted_p'] == 1


@pytest.mark.parametrize('case', ['duplicate', 'missing', 'identity', 'unknown', 'paired'])
def test_statistics_invalid_inventory_or_identity_rejected(case):
    shards = statistical_fixture()
    if case == 'duplicate':
        shards[1] = copy.deepcopy(shards[0])
    elif case == 'missing':
        shards[0]['observations'].pop()
    elif case == 'identity':
        shards[1]['identity'] = {'seed': 23}
    elif case == 'unknown':
        shards[0]['observations'][0]['metrics']['R']['direct'] = None
    else:
        shards[0]['observations'][0]['paired_differences']['R-T']['direct'] = 5.
    with pytest.raises(ValueError):
        merge_shards(shards)


def test_stdlib_entry_and_phase_input_isolation():
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT / 'src')
    subprocess.run([sys.executable, '-c', "import acl_hct.radial_entry,sys; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"],
                   check=True, env=env)
    status = json.loads(subprocess.check_output([sys.executable, '-m', 'acl_hct.radial_entry'], env=env))
    assert status['new_graph_samples'] == status['GNN_forward_calls'] == 0
    assert status['science_worker_seconds'] == 840 and status['fixture_worker_seconds'] == 240
    args = entry.parser().parse_args(['--phase', 'cuda_fixture', '--config', 'c', '--protocol', 'p', '--approval', 'a',
                                     '--quality', 'q', '--output', 'o', '--source-commit', '1' * 40])
    entry.validate_arguments(args)
    args.bindings = Path('original-archive')
    with pytest.raises(ValueError, match='no original'):
        entry.validate_arguments(args)


def test_exact_B_config_rejects_threshold_family_backend_change():
    assert entry.validate_config(config()) == entry.CONFIG_SHA256
    for keys, value in [(('projection', 'undefined_distance_minimum'), 1e-8),
                        (('statistics', 'family_size'), 12), (('runtime', 'intervention_geometry_device'), 'cuda:0')]:
        changed = config()
        changed[keys[0]][keys[1]] = value
        with pytest.raises(ValueError, match='configuration'):
            entry.validate_config(changed)


def synthetic_original_archives(tmp_path):
    """Test-only original inventory; calibrated-data gate is explicitly mocked."""
    old_config = json.loads((ROOT / 'configs/e2_frozen_recovery_pilot.json').read_bytes())
    bindings = {'shards': [], 'immutable_file_inventory': {},
                'old_science_source_commit': config()['old_science_source_commit']}
    roots = []

    def archive_binding(root, descriptor):
        manifest = json.loads((root / descriptor['manifest']).read_bytes())
        return {'manifest_path': str(root / descriptor['manifest']), 'manifest_raw_sha256': descriptor['manifest_sha256'],
                'NPZ_path': str(root / descriptor['array_file']), 'NPZ_raw_sha256': descriptor['array_file_sha256'],
                'NPZ_bytes': descriptor['array_bytes'], 'arrays_count': descriptor['arrays'], 'array_metadata': manifest['arrays']}

    for seed in (11, 23):
        base, bias, floor, sample, weights, data, view, panel_spec = recovery.synthetic_data(config(), seed, 'mixed')
        head = recovery.ArchivedRelationHead(weights)
        panel = RadialPanel(view, panel_spec, base, {'V': list(range(16))}, 1., floor)
        full_ranking = recovery.rank_points(head, base.float(), data, config(), time.monotonic() + 600)
        for start in (0, 8):
            root = tmp_path / f'old-{seed}-{start}'
            root.mkdir()
            roots.append(root)
            entry_payload = {'base': base, 'bias': bias, 'floor': floor, 'q': bias,
                             'native_F': base.float(), 'weights': weights, 'nodes': data['nodes'],
                             'valid': np.asarray(data['valid']), 'panel': panel_spec, 'F_ranking': full_ranking,
                             'structure_view': {'root': view.root, 'reachable': sorted(view.reachable)}}
            descriptor = write_archive(root, 'entry', entry_payload)
            entry_spec = archive_binding(root, descriptor)
            row = {'seed': seed, 'status': 'complete', 'protocol': 'E2-frozen-recovery-v1',
                   'config_sha256': config()['old_config_canonical_sha256'], 'engineering_fixture_only': False,
                   'completed_repetitions': 8, 'weights_unchanged': True, 'input_unchanged': True, 'model_updates': 0,
                   'provenance': {'source_commit': config()['old_science_source_commit'], 'release_verified': True,
                                  'release': {'fixture_only_test_mock': True}}, 'identity': {'seed': seed},
                   'repeat_range': [start, start + 8], 'entry_archive': descriptor, 'observations': [], 'repeat_archives': []}
            spec = {'key': f'test-{seed}-{start}', 'seed': seed, 'repeat_range': row['repeat_range'],
                    'global_repeat_ids': list(range(start, start + 8)), 'remote_output': root.as_posix(),
                    'frozen_identity': row['identity'], 'archives': [entry_spec]}
            for repeat in range(start, start + 8):
                ranking = recovery.rank_points(head, sample, data, config(), time.monotonic() + 600)
                structure = panel.evaluate(promoted(sample, 1.))
                observation = {'repeat': repeat, 'plan_hash': f'test-fixed-plan-{seed}-{repeat}',
                               'rng': {'fixture_only_repeat': repeat}, 'metrics': {'S': _metrics(structure, ranking)}}
                payload = {**observation, 'native_points': {'S': sample}, 'rankings': {'S': ranking}, 'structure': {'S': structure}}
                old_descriptor = write_archive(root, f'repeat-{repeat:02}', payload)
                old_spec = archive_binding(root, old_descriptor)
                old_spec.update(global_repeat_id=repeat, plan_hash=observation['plan_hash'], sampling_rng_identity=observation['rng'])
                row['observations'].append(observation)
                row['repeat_archives'].append(old_descriptor)
                spec['archives'].append(old_spec)
            supervisor = {'status': 'complete', 'worker_exit_code': 0, 'deadline_seconds': 1080}
            progress = {'phase': 'science', 'protocol': row['protocol'], 'config_sha256': row['config_sha256'],
                        'source_commit': config()['old_science_source_commit'], 'release': row['provenance']['release'],
                        'source_lf_sha256': entry.old_source_hashes()}
            for name, value in [('run.json', row), ('supervisor.json', supervisor), ('progress.json', progress)]:
                path = root / name
                path.write_bytes(json.dumps(value).encode())
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                bindings['immutable_file_inventory'][root.as_posix() + '/' + name] = {'raw_sha256': digest}
                if name == 'run.json':
                    spec['run_raw_sha256'] = digest
            bindings['shards'].append(spec)
    return old_config, bindings, roots


def test_all_four_saved_shards_head_replay_R_T_archives_and_stats(tmp_path, monkeypatch):
    old_config, bindings, roots = synthetic_original_archives(tmp_path)

    def explicitly_mocked_calibrated_data_gate(payload, row, cfg):
        # Real entry_design is separately frozen and tested; this test uses N=16.
        view = SimpleNamespace(nodes=list(payload['nodes']), **payload['structure_view'])
        panel = RadialPanel(view, payload['panel'], torch.from_numpy(payload['base']), {'V': list(range(16))},
                            1., torch.from_numpy(payload['floor']))
        return SimpleNamespace(panel=panel), {'nodes': list(payload['nodes']), 'valid': [tuple(map(int, p)) for p in payload['valid']]}

    monkeypatch.setattr(recovery, 'entry_design', explicitly_mocked_calibrated_data_gate)
    rows = []
    for i, spec in enumerate(bindings['shards']):
        output = tmp_path / f'new-{i}'
        row = recovery.run_saved_shard(config(), old_config, bindings, roots, spec['seed'], spec['repeat_range'][0],
                                      output, time.monotonic() + 600, {'fixture_only': True}, engineering=True)
        assert row['status'] == 'complete', row.get('failure')
        assert row['F_full_rank_exact'] is True and row['completed_repetitions'] == 8
        assert row['GNN_forward_calls'] == row['new_samples'] == row['model_updates'] == 0
        assert row['RNG_unchanged'] is row['weights_unchanged'] is row['input_unchanged'] is True
        assert row['engineering_fixture_only'] is True
        assert len(row['repeat_archives']) == 8
        for observation, descriptor in zip(row['observations'], row['repeat_archives']):
            payload = read_archive(output, descriptor)
            assert set(payload['native_points']) == {'R', 'T'}
            assert payload['repeat'] == observation['repeat']
        rows.append(row)
    assert merge_shards(rows)['family_size'] == 18
    monkeypatch.setattr(recovery, 'feasibility', lambda *a: {'passed': False})
    blocked_output = tmp_path / 'budget-only-stop'
    blocked = recovery.run_saved_shard(config(), old_config, bindings, roots, 11, 0, blocked_output,
                                      time.monotonic() + 600, {'fixture_only': True}, engineering=True)
    assert blocked['status'] == 'resource_time_budget_inadequate'
    assert blocked['F_full_rank_exact'] is True and blocked['completed_repetitions'] == 0
    assert not blocked['repeat_archives'] and not list(blocked_output.glob('repeat-*.npz'))
    assert read_archive(blocked_output, blocked['F_replay_archive'])['ranking']['status'] == 'complete'


def release_fixture(tmp_path, monkeypatch):
    cfg = tmp_path / 'config.json'
    cfg.write_bytes((ROOT / 'configs/e2_radial_component_pilot.json').read_bytes())
    protocol = ROOT / 'docs/operations/E2_RADIAL_COMPONENT_PROTOCOL.md'
    sources = entry.source_hashes()
    quality = {'status': 'passed', 'phase': 'B', 'config_canonical_sha256': entry.CONFIG_SHA256,
               'config_raw_sha256': entry.file_sha256(cfg), 'new_source_lf_sha256': sources,
               'protocol_lf_sha256': entry.lf_hash(protocol)}
    quality_path = tmp_path / 'quality.json'
    quality_path.write_bytes(json.dumps(quality).encode())
    approval = {'status': 'approved_B_cuda_fixture', 'protocol': entry.PROTOCOL, 'execution_phase': 'cuda_fixture',
                'source_commit': '1' * 40, 'user_authorized': True, 'quality_review_passed': True,
                'supervisor_released': True, 'GPU_requested': 1, 'config_canonical_sha256': entry.CONFIG_SHA256,
                'new_source_lf_sha256': sources, 'quality_raw_sha256': entry.file_sha256(quality_path),
                'config_raw_sha256': entry.file_sha256(cfg), 'protocol_lf_sha256': entry.lf_hash(protocol)}
    approval_path = tmp_path / 'approval.json'
    approval_path.write_bytes(json.dumps(approval).encode())
    args = entry.parser().parse_args(['--phase', 'cuda_fixture', '--config', str(cfg), '--protocol', str(protocol),
                                     '--approval', str(approval_path), '--quality', str(quality_path),
                                     '--output', str(tmp_path / 'new'), '--source-commit', '1' * 40])
    monkeypatch.setattr(entry, 'source_identity', lambda declared: {'source_commit': declared, 'git': {'available': False}})
    return args


@pytest.mark.parametrize('case', ['positive', 'phase', 'quality', 'source', 'user_authorized'])
def test_B_release_requires_exact_bytes_phase_source_authorization(tmp_path, monkeypatch, case):
    args = release_fixture(tmp_path, monkeypatch)
    if case == 'positive':
        cfg, proof = entry.verify_release(args)
        assert proof['release_verified'] is True and cfg['statistics']['family_size'] == 18
        return
    if case == 'quality':
        with args.quality.open('ab') as stream:
            stream.write(b'\n ')
    else:
        approval = json.loads(args.approval.read_bytes())
        if case == 'phase':
            approval['execution_phase'] = 'science'
        elif case == 'source':
            approval['new_source_lf_sha256']['radial_components.py'] = '0' * 64
        else:
            approval['user_authorized'] = False
        args.approval.write_bytes(json.dumps(approval).encode())
    with pytest.raises(ValueError):
        entry.verify_release(args)


def test_science_fixture_gate_requires_full_arrays_and_independent_review(tmp_path):
    cfg = config()
    run = {'status': 'complete', 'phase': 'cuda_fixture', 'protocol': entry.PROTOCOL, 'config_sha256': entry.CONFIG_SHA256,
           'source_commit': '1' * 40, 'source_lf_sha256': entry.source_hashes(), 'original_runtime_inputs_opened': [],
           'seeds': [11, 23], 'torch': cfg['runtime']['torch'], 'device': 'NVIDIA L40', 'checks': {}, 'archives': {}}
    # Metadata here is a test fixture; these CPU arrays are never a native release.
    for seed in (11, 23):
        for case in ('mixed', 'zero'):
            checks, descriptor = recovery.fixture_case(cfg, seed, case, torch.device('cpu'), tmp_path, time.monotonic() + 60)
            run['archives'][f'{seed}/{case}'] = descriptor
            run['checks'].update(checks)
    run_path = tmp_path / 'run.json'
    run_path.write_bytes(json.dumps(run).encode())
    sup = {'status': 'complete', 'worker_exit_code': 0, 'deadline_seconds': 240,
           'context': {'protocol': entry.PROTOCOL, 'phase': 'cuda_fixture', 'config_sha256': entry.CONFIG_SHA256,
                       'source_commit': '1' * 40}}
    (tmp_path / 'supervisor.json').write_bytes(json.dumps(sup).encode())
    review = {'status': 'independent_fixture_review_passed', 'fixture_raw_sha256': entry.file_sha256(run_path),
              'source_commit': '1' * 40, 'config_sha256': entry.CONFIG_SHA256, 'source_lf_sha256': entry.source_hashes()}
    review_path = tmp_path / 'review.json'
    review_path.write_bytes(json.dumps(review).encode())
    args = SimpleNamespace(fixture_record=run_path, fixture_review=review_path, source_commit='1' * 40)
    approval = {'fixture_raw_sha256': entry.file_sha256(run_path), 'fixture_review_raw_sha256': entry.file_sha256(review_path)}
    assert entry.verify_fixture(args, cfg, approval, entry.source_hashes())['fixture_raw_sha256'] == approval['fixture_raw_sha256']
    review['status'] = 'worker_selfcheck_passed'
    review_path.write_bytes(json.dumps(review).encode())
    approval['fixture_review_raw_sha256'] = entry.file_sha256(review_path)
    with pytest.raises(ValueError, match='independent supervisor'):
        entry.verify_fixture(args, cfg, approval, entry.source_hashes())
    review['status'] = 'independent_fixture_review_passed'
    review_path.write_bytes(json.dumps(review).encode())
    approval['fixture_review_raw_sha256'] = entry.file_sha256(review_path)
    original = run['checks'].pop('archive_complete')
    run_path.write_bytes(json.dumps(run).encode())
    approval['fixture_raw_sha256'] = entry.file_sha256(run_path)
    review['fixture_raw_sha256'] = approval['fixture_raw_sha256']
    review_path.write_bytes(json.dumps(review).encode())
    approval['fixture_review_raw_sha256'] = entry.file_sha256(review_path)
    with pytest.raises(ValueError, match='same-release'):
        entry.verify_fixture(args, cfg, approval, entry.source_hashes())

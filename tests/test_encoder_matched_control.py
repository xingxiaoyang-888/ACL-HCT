"""CPU synthetic engineering checks; no real prepared tensors or GPU use."""
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from acl_hct.backbone import LorentzMeanNetwork
from acl_hct.encoder_matched import SelfMessageLorentzNetwork
from acl_hct import encoder_matched_control as control
from acl_hct import encoder_matched_entry as entry
from acl_hct import encoder_matched_registration as registration
from acl_hct.geometry import check_point, exp, log, origin_like
from acl_hct.ranking import filtered_parent_ranks, prepare_scores, score_cached

ROOT = Path(__file__).parents[1]
IDENTITY = {'source_commit': '1' * 40, 'source_commit_basis': 'synthetic_engineering_not_git',
            'git': {'available': False, 'commit': None}}


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


def config():
    return json.loads((ROOT / 'configs/e2_encoder_matched_control.json').read_text())


def settings(**changes):
    return replace(control.Settings(input_dim=4, hidden=4, head_hidden=5, batch_positives=2,
                                   max_steps=4, evaluate_every=1, save_every=1, max_seconds=60,
                                   evaluation_max_seconds=10, candidate_chunk=7), **changes)


def tiny_data():
    features = torch.randn(40, 4, generator=torch.Generator().manual_seed(9281)) * .1
    groups = torch.tensor([[[0, b]] + [[a, b] for a in (1, 2, 3, 4)] for b in range(6, 16)])
    labels = torch.tensor([1., 0., 0., 0., 0.]).repeat(10, 1)
    neighbors = [[] for _ in range(40)]
    neighbors[0] = list(range(6, 16))
    for b in range(6, 16):
        neighbors[b] = [0]
    valid = [(1, 30), (2, 30), (3, 31)]
    return {'features': features, 'query_groups': groups, 'labels': labels,
            'nodes': [f'n{i:02d}' for i in range(40)], 'neighbors': neighbors,
            'valid': valid, 'manifest_hash': '2' * 64, 'valid_hash': '3' * 64,
            'manifest': {}, 'input_files_opened': []}


def history(data, s):
    rng = torch.Generator().manual_seed(s.seed + 1)
    rows = []
    for step in range(1, s.max_steps + 1):
        chosen = torch.randperm(len(data['query_groups']), generator=rng)[:s.batch_positives]
        rows.append({'step': step, 'batch_group_ids_sha256': registration.canonical(chosen.tolist()),
                     'positive_queries': len(chosen), 'negative_queries': 4 * len(chosen)})
    return {'status': 'complete', 'seed': s.seed, 'training_settings': {'seed': s.seed, 'batch_positives': s.batch_positives},
            'steps': rows, 'fixture_only': True}


def fixture_prepared(root):
    root.mkdir()
    d = tiny_data(); nodes = d['nodes']; fit = nodes[6:16]
    f = d['features'].numpy()
    manifest = {'protocol': 'B-child-grouped-80-10-10-v1', 'split_seed': 20260914,
                'node_order_hash': registration.canonical(nodes), 'graph_hash': registration.canonical(d['neighbors']),
                'text_fit_entities': fit, 'feature_manifest': {'nodes': nodes, 'train_entities': fit,
                'effective_dimension': 4, 'feature_sha256': hashlib.sha256(f.tobytes()).hexdigest()}}
    queries = {'queries': [[nodes[a], nodes[b]] for a, b in d['query_groups'].reshape(-1, 2).tolist()],
               'labels': d['labels'].reshape(-1).tolist(), 'negatives_per_positive': 4}
    manifest['train_queries_hash'] = registration.canonical(queries)
    valid = [[nodes[a], nodes[b]] for a, b in d['valid']]
    payloads = {'input_manifest.json': manifest, 'observed_graph.json': {'nodes': nodes, 'neighbors': d['neighbors']},
                'train_queries.json': queries, 'evaluator_valid.json': valid}
    for name, value in payloads.items():
        (root / name).write_text(json.dumps(value), encoding='utf-8')
    np.savez(root / 'features.npz', features=f)
    (root / 'evaluator_test.json').write_text('FORBIDDEN')
    (root / 'evaluator_truth.json').write_text('FORBIDDEN')
    cfg = config()
    cfg['model']['input_dim'] = 4; cfg['valid_query_count'] = 3
    cfg['prepared'] = {'manifest_hash': registration.canonical(manifest), 'node_order_hash': manifest['node_order_hash'],
                       'graph_hash': manifest['graph_hash'], 'train_queries_hash': manifest['train_queries_hash'],
                       'valid_queries_hash': registration.canonical(valid), 'nodes_count': 40, 'train_groups_count': 10,
                       'input_raw_sha256': {n: registration.file_sha256(root / n)
                                           for n in ('input_manifest.json', 'observed_graph.json', 'train_queries.json')}}
    cfg['prepared'].update(features_npz_raw_sha256=registration.file_sha256(root / 'features.npz'),
                           evaluator_valid_raw_sha256=registration.file_sha256(root / 'evaluator_valid.json'))
    return cfg


@pytest.mark.parametrize('seed', [11, 23])
def test_original_constructor_parameters_and_rng(seed):
    torch.manual_seed(seed); original = LorentzMeanNetwork(128); after = torch.get_rng_state()
    torch.manual_seed(seed); new = SelfMessageLorentzNetwork(128)
    assert torch.equal(after, torch.get_rng_state())
    assert control.weights_hashes(original) == control.weights_hashes(new)
    assert list(original.state_dict()) == list(config()['references'][str(seed)]['initial_weights_sha256'])
    assert sum(p.numel() for p in new.parameters()) == 82433
    assert sum(isinstance(m, torch.nn.ReLU) for m in new.modules()) == 1
    assert not any(isinstance(m, (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.LayerNorm)) for m in new.modules())


@pytest.mark.parametrize('seed', [11, 23])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_three_paths_points_loss_and_all_gradients(seed, dtype, tmp_path):
    result = control.equivalence_fixture(config(), seed, dtype=dtype,
                atol=1e-12 if dtype == torch.float64 else None, rtol=1e-11 if dtype == torch.float64 else None,
                archive_root=tmp_path / 'archive')
    assert result['status'] == 'passed' and len(result['checks']) == 23
    assert not result['historical_initialization_checked']  # Native 2.5.1 remains a real deployment gate.
    manifest = json.loads((tmp_path / 'archive/fixture.json').read_text())
    with np.load(tmp_path / 'archive/fixture.npz', allow_pickle=False) as raw:
        assert len(raw.files) == len(manifest['arrays'])
        for key in raw.files:
            assert hashlib.sha256(raw[key].tobytes()).hexdigest() == manifest['arrays'][key]['sha256']
        assert sum(k.startswith('gradients_') for k in raw.files) == 24
        assert sum(k.startswith('weights_') for k in raw.files) == 24
        np.testing.assert_array_equal(raw['unique_ids'][raw['inverse_queries']], raw['queries'])
        np.testing.assert_allclose(raw['loss_original'], raw['loss_unique'], atol=result['checks']['empty_vs_unique_mean_bce']['atol'])


def test_repeated_unsorted_queries_feature_gradients_and_unrelated_entities():
    torch.manual_seed(37); m = SelfMessageLorentzNetwork(4, 5, 6).double(); full = copy.deepcopy(m)
    x = torch.randn(12, 4, dtype=torch.float64, requires_grad=True); y = x.detach().clone().requires_grad_()
    q = torch.tensor([[5, 3], [0, 1], [5, 3], [2, 1], [0, 5]])
    a, b = m(x, q), full(y, q, unique_entities=False)
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-11)
    assert a[0] == a[2]
    a.sum().backward(); b.sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-12, rtol=1e-11)
    assert torch.count_nonzero(x.grad[6:]) == 0
    torch.testing.assert_close(m(x, q), m(torch.cat((x, torch.randn(5, 4, dtype=torch.float64))), q), atol=1e-12, rtol=1e-11)


@pytest.mark.parametrize('c', [.5, 1., 2.])
def test_manifold_maps_and_zero_linear_gradients(c):
    m = SelfMessageLorentzNetwork(4, 5, c=c).double()
    x = torch.randn(8, 4, dtype=torch.float64) * .1
    points = m.encode(x); check_point(points, c)
    torch.testing.assert_close(exp(origin_like(points, c), log(origin_like(points, c), points, c), c), points,
                               atol=1e-12, rtol=1e-11)
    with torch.no_grad():
        for layer in m.layers:
            layer.linear.weight.zero_(); layer.linear.bias.zero_()
    q = torch.tensor([[0, 1], [2, 1]])
    m(x, q).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    check_point(m.encode(x), c)


@pytest.mark.parametrize('q', [torch.tensor([[-1, 1]]), torch.tensor([[0, 40]]), torch.tensor([[0., 1.]]), torch.empty(0, 2, dtype=torch.long)])
def test_invalid_query_indices_rejected(q):
    with pytest.raises(ValueError):
        SelfMessageLorentzNetwork(4)(tiny_data()['features'], q)


def test_full_filtered_ranks_ties_and_cache_invalidation():
    d = tiny_data(); m = control.construct(settings()).eval(); points = m.encode(d['features'])
    truth = {30: {1, 2}, 31: {3}}
    row = filtered_parent_ranks(m, points, d['valid'], truth, candidate_chunk=7)
    control.validate_ranking(row, d)
    for r in row['rows']:
        eligible = [a for a in range(40) if a != r['child'] and (a not in truth[r['child']] or a == r['parent'])]
        scores = m.score(points, torch.tensor([[a, r['child']] for a in eligible]))
        target = scores[eligible.index(r['parent'])]
        assert r['rank'] == 1 + int((scores > target).sum()) + .5 * (int((scores == target).sum()) - 1)
    cache = prepare_scores(m, points)
    with torch.no_grad():
        m.layers[0].linear.weight.add_(.001)
    with pytest.raises(ValueError, match='stale'):
        score_cached(m, cache, torch.tensor([1]), 30)
    with torch.no_grad():
        for p in m.relation_head.parameters():
            p.zero_()
    tied = filtered_parent_ranks(m, m.encode(d['features']), d['valid'], truth, candidate_chunk=7)
    assert [r['rank'] for r in tied['rows']] == [19.5, 19.5, 20.]
    control.validate_ranking(tied, d)
    bins = control.degree_readings(tied, d)
    assert bins['0']['query_count'] == 3 and bins['>16']['query_micro_mrr'] is None
    assert bins['0']['contribution_to_overall_micro_mrr'] == tied['query_micro_mrr']
    partial = filtered_parent_ranks(m, points, d['valid'], truth, max_seconds=0)
    with pytest.raises(ValueError, match='coverage'):
        control.validate_ranking(partial, d)


def test_input_read_scope_raw_labels_and_valid_contract(tmp_path, monkeypatch):
    root = tmp_path / 'prepared'; cfg = fixture_prepared(root)
    before = {p.name: registration.file_sha256(p) for p in root.iterdir()}
    read = Path.read_bytes; opened = []
    def guarded(path, *a, **kw):
        if path.parent == root:
            assert path.name not in ('evaluator_test.json', 'evaluator_truth.json')
            opened.append(path.name)
        return read(path, *a, **kw)
    monkeypatch.setattr(Path, 'read_bytes', guarded)
    data = control.load_prepared(root, cfg)
    assert data['input_files_opened'] == ['input_manifest.json', 'observed_graph.json', 'train_queries.json', 'features.npz', 'evaluator_valid.json']
    assert set(opened) == {'input_manifest.json', 'observed_graph.json', 'train_queries.json', 'evaluator_valid.json'}
    assert before == {p.name: registration.file_sha256(p) for p in root.iterdir()}
    changed = copy.deepcopy(cfg); changed['prepared']['input_raw_sha256']['observed_graph.json'] = '0' * 64
    with pytest.raises(ValueError, match='raw hash'):
        control.load_prepared(root, changed)
    for key in ('features_npz_raw_sha256', 'evaluator_valid_raw_sha256'):
        changed = copy.deepcopy(cfg); changed['prepared'][key] = '0' * 64
        with pytest.raises(ValueError, match='raw hash'):
            control.load_prepared(root, changed)
    q = json.loads((root / 'train_queries.json').read_text()); q['labels'][0] = 1. + 1e-10
    (root / 'train_queries.json').write_text(json.dumps(q))
    manifest = json.loads((root / 'input_manifest.json').read_text()); manifest['train_queries_hash'] = registration.canonical(q)
    (root / 'input_manifest.json').write_text(json.dumps(manifest))
    changed = copy.deepcopy(cfg)
    changed['prepared'].update(train_queries_hash=manifest['train_queries_hash'], manifest_hash=registration.canonical(manifest))
    for n in ('train_queries.json', 'input_manifest.json'):
        changed['prepared']['input_raw_sha256'][n] = registration.file_sha256(root / n)
    with pytest.raises(ValueError, match='exactly one positive'):
        control.load_prepared(root, changed)


def test_actual_historical_reference_schema_without_real_prepared():
    cfg = config()
    for seed in [11, 23]:
        # Historical public rank metadata only; no features/checkpoints/prepared opened.
        gnn = json.loads((ROOT / f'reports/baseline-seed{seed}-run.json').read_text())
        d = {'nodes': range(82115), 'valid': [(r['parent'], r['child']) for r in gnn['evaluations'][0]['rows']],
             'manifest_hash': cfg['prepared']['manifest_hash'], 'valid_hash': cfg['prepared']['valid_queries_hash']}
        a, b = control.verify_references(ROOT / f'reports/baseline-seed{seed}-run.json',
                                        ROOT / f'reports/e2-capacity-seed{seed}-run.json', cfg, seed, d)
        assert a['status'] == 'step_limit_reached' and 'best_step' not in a
        assert b['status'] == 'complete' and b['config']['seeds'] == [11, 23]
        assert b['seed'] == b['training_settings']['seed'] == seed


def test_full_run_initial_snapshot_four_choices_and_fresh_best_ranking(tmp_path, monkeypatch):
    d = tiny_data(); s = settings(); h = history(d, s); identities = []
    rank = control.filtered_parent_ranks
    def watched(model, *a, **kw):
        identities.append(id(model)); return rank(model, *a, **kw)
    monkeypatch.setattr(control, 'filtered_parent_ranks', watched)
    before = control.tensor_identity(d)
    row = control.run(d, tmp_path / 'run', s, h, identity=IDENTITY, engineering=True)
    assert row['status'] == 'complete' and row['completed_steps'] == 4
    assert json.loads((tmp_path / 'run/progress.json').read_text())['completed_steps'] == 4
    assert len(row['evaluations']) == 4 and len(identities) == 5
    assert len(set(identities[:4])) == 1 and identities[-1] != identities[0]
    assert row['best_reload']['fresh_complete_ranking_performed'] and row['best_reload']['additional_selection_opportunities'] == 0
    assert row['best_step'] == max(row['evaluations'], key=lambda r: r['query_micro_mrr'])['step']
    assert [r['batch_group_ids_sha256'] for r in row['steps']] == [r['batch_group_ids_sha256'] for r in h['steps']]
    assert row['input_identity_after'] == before
    initial = torch.load(tmp_path / 'run/initial.pt', weights_only=True)
    loaded = control.construct(s); loaded.load_state_dict(initial['model'], strict=True)
    assert control.weights_hashes(loaded) == row['initial_weights_sha256'] == initial['weights_sha256']
    last = torch.load(tmp_path / 'run/last.pt', weights_only=True)
    assert last['completed_steps'] == 4 and last['run_status'] == 'complete' and not last['resume_allowed']
    assert torch.equal(last['batch_rng'], control.BatchStream(10, 11, 2, h).rng.get_state()) is False
    with pytest.raises(FileExistsError):
        control.run(d, tmp_path / 'run', s, h, identity=IDENTITY, engineering=True)
    saved = torch.load(tmp_path / 'run/best.pt', weights_only=True)
    saved['manifest_hash'] = '0' * 64; torch.save(saved, tmp_path / 'bad.pt')
    with pytest.raises(ValueError, match='metadata'):
        control.checkpoint_load(tmp_path / 'bad.pt', s, d, IDENTITY, registration.CONFIG_SHA256,
                                row['best_reload']['weights_sha256'], row['best_step'], row['best_full_valid_mrr'])


def test_first_best_ties_do_not_add_selection_opportunities(tmp_path, monkeypatch):
    d = tiny_data(); s = settings(); model = control.construct(s).eval()
    with torch.no_grad():
        for p in model.relation_head.parameters(): p.zero_()
    fixed = filtered_parent_ranks(model, model.encode(d['features']), d['valid'], {30: {1, 2}, 31: {3}})
    calls = []
    def tied_rank(model, *a, **kw):
        calls.append(id(model)); return copy.deepcopy(fixed)
    monkeypatch.setattr(control, 'filtered_parent_ranks', tied_rank)
    row = control.run(d, tmp_path / 'ties', s, history(d, s), identity=IDENTITY, engineering=True)
    assert row['status'] == 'complete' and row['best_step'] == 1 and len(calls) == 5
    assert row['best_reload']['additional_selection_opportunities'] == 0


@pytest.mark.parametrize('failure', ['batch_hash', 'partial_valid', 'nonfinite', 'deadline', 'reload'])
def test_failures_stop_and_keep_evidence(tmp_path, monkeypatch, failure):
    d = tiny_data(); s = settings(); h = history(d, s); out = tmp_path / failure
    if failure == 'batch_hash':
        h['steps'][1]['batch_group_ids_sha256'] = '0' * 64
    elif failure == 'partial_valid':
        old = control.filtered_parent_ranks
        def partial(*a, **kw):
            r = old(*a, **kw); r['status'] = 'incomplete_time_limit'; return r
        monkeypatch.setattr(control, 'filtered_parent_ranks', partial)
    elif failure == 'nonfinite':
        monkeypatch.setattr(SelfMessageLorentzNetwork, 'forward', lambda self, f, q: torch.full((len(q),), float('nan')))
    elif failure == 'reload':
        monkeypatch.setattr(control, 'checkpoint_load', lambda *a: (_ for _ in ()).throw(ValueError('reload mismatch')))
    deadline = time.monotonic() - 1 if failure == 'deadline' else None
    row = control.run(d, out, s, h, identity=IDENTITY, engineering=True, deadline=deadline)
    assert row['status'] != 'complete' and row['purpose_status'] == 'insufficient_evidence'
    assert (out / 'failure.json').exists() and (out / 'last.pt').exists()
    if failure == 'batch_hash': assert row['completed_steps'] == 1
    if failure == 'partial_valid': assert row['completed_steps'] == 1 and row['best_step'] is None
    if failure in ('nonfinite', 'deadline'): assert row['completed_steps'] == 0
    if failure == 'reload': assert row['completed_steps'] == 4 and row['best_reload'] is None


def approval_quality(tmp_path):
    cfg = config(); path = tmp_path / 'quality.json'
    row = {'status': 'passed', 'config_canonical_sha256': registration.CONFIG_SHA256,
           'source_lf_sha256': registration.source_hashes()}
    path.write_text(json.dumps(row))
    approval = dict(user_authorized=True, user_message_reference='synthetic engineering only',
                    quality_review_passed=True, entry_criteria_frozen=True, supervisor_released=True,
                    scope=registration.PROTOCOL, config_sha256=registration.CONFIG_SHA256,
                    source_commit=IDENTITY['source_commit'], quality_record_sha256=registration.file_sha256(path))
    return cfg, path, row, approval


@pytest.mark.parametrize('key', ['user_authorized', 'user_message_reference', 'quality_review_passed', 'entry_criteria_frozen',
                                'supervisor_released', 'scope', 'config_sha256', 'source_commit', 'quality_record_sha256'])
def test_release_missing_fields_fail_before_labels(tmp_path, key):
    cfg, path, _, approval = approval_quality(tmp_path)
    registration.verify_release(cfg, IDENTITY, approval, path)
    approval.pop(key)
    with pytest.raises(ValueError):
        registration.verify_release(cfg, IDENTITY, approval, path)


def test_config_source_quality_runtime_and_initializer_drift(tmp_path, monkeypatch):
    cfg, path, row, approval = approval_quality(tmp_path)
    changed = copy.deepcopy(cfg); changed['training']['max_steps'] += 1
    with pytest.raises(ValueError): registration.validate_config(changed)
    row['source_lf_sha256']['encoder_matched.py'] = '0' * 64; path.write_text(json.dumps(row))
    approval['quality_record_sha256'] = registration.file_sha256(path)
    with pytest.raises(ValueError, match='exact current source'):
        registration.verify_release(cfg, IDENTITY, approval, path)
    changed = copy.deepcopy(cfg); changed['references']['11']['initial_weights_sha256']['layers.0.linear.weight'] = '0' * 64
    monkeypatch.setattr(LorentzMeanNetwork, 'encode', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('must reject before forward')))
    with pytest.raises(ValueError, match='initial tensors'):
        control.equivalence_fixture(changed, 11, historical_initialization=True)
    if str(torch.__version__) != cfg['runtime']['torch']:
        with pytest.raises(ValueError, match='original PyTorch'):
            control.require_runtime(cfg, torch.device('cpu'))


def fixture_gate(tmp_path):
    cfg = config(); seeds = []
    for seed in [11, 23]:
        r = control.equivalence_fixture(cfg, seed, archive_root=tmp_path / ('seed' + str(seed)))
        r['historical_initialization_checked'] = True  # Synthetic record used only to test binding rejection.
        seeds.append(r)
    row = {'status': 'complete', 'matched_encoder_fixture_passed': True, 'protocol': registration.PROTOCOL,
           'config_sha256': registration.CONFIG_SHA256, 'source': IDENTITY, 'source_lf_sha256': registration.source_hashes(),
           'torch': cfg['runtime']['torch'], 'dtype': 'float32', 'device_name': cfg['runtime']['hardware'],
           'seeds': seeds, 'fixture_only': True}
    path = tmp_path / 'cuda-quality.json'; path.write_text(json.dumps(row))
    (tmp_path / 'supervisor.json').write_text(json.dumps({'status': 'complete', 'worker_exit_code': 0, 'deadline_seconds': 240}))
    return cfg, path, row


@pytest.mark.parametrize('failure', ['empty_checks', 'missing_gradient', 'relaxed_tolerance', 'array_tamper', 'failed_supervisor', 'missing_raw_gradient'])
def test_fixture_gate_rejects_incomplete_checks_and_arrays(tmp_path, failure):
    cfg, path, row = fixture_gate(tmp_path)
    control.verify_gate(path, registration.file_sha256(path), 'cuda_fixture', cfg, IDENTITY)
    if failure == 'empty_checks': row['seeds'][0]['checks'] = {}
    if failure == 'missing_gradient': row['seeds'][0]['checks'].pop('empty_vs_unique_gradient/layers.0.linear.bias')
    if failure == 'relaxed_tolerance': row['seeds'][0]['checks']['empty_vs_unique_logits']['atol'] = 1e-3
    if failure == 'array_tamper':
        with (tmp_path / 'seed11/fixture.npz').open('ab') as stream: stream.write(b'tamper')
    if failure == 'failed_supervisor':
        (tmp_path / 'supervisor.json').write_text(json.dumps({'status': 'timeout', 'worker_exit_code': -9, 'deadline_seconds': 240}))
    if failure == 'missing_raw_gradient':
        manifest_path = tmp_path / 'seed11/fixture.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['arrays'].pop('gradients_unique/layers.0.linear.bias')
        manifest_path.write_text(json.dumps(manifest))
        row['seeds'][0]['archive']['manifest_sha256'] = registration.file_sha256(manifest_path)
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        control.verify_gate(path, registration.file_sha256(path), 'cuda_fixture', cfg, IDENTITY)


@pytest.mark.parametrize('failure', ['missing_seed', 'wrong_seed', 'history_count', 'forward_called', 'initial_hash', 'valid_hash', 'label_scope'])
def test_cpu_gate_rejects_incomplete_or_conflicting_provenance(tmp_path, failure):
    cfg = config(); seeds = []
    for seed in cfg['seeds']:
        a = cfg['references'][str(seed)]
        seeds.append({'seed': seed, 'batch_hashes_verified': 1024, 'initial_weights_sha256': copy.deepcopy(a['initial_weights_sha256']),
                      'gnn_report_sha256': a['gnn']['baseline_report_sha256'], 'mlp_report_sha256': a['mlp']['report_sha256'],
                      'nodes_count': 82115, 'train_groups_count': 67539, 'valid_query_count': 8456,
                      'manifest_hash': cfg['prepared']['manifest_hash'], 'valid_hash': cfg['prepared']['valid_queries_hash'],
                      'input_files_opened': ['input_manifest.json', 'observed_graph.json', 'train_queries.json', 'features.npz', 'evaluator_valid.json']})
    row = {'status': 'complete', 'cpu_preflight_passed': True, 'protocol': registration.PROTOCOL,
           'config_sha256': registration.CONFIG_SHA256, 'source': IDENTITY, 'source_lf_sha256': registration.source_hashes(),
           'torch': cfg['runtime']['torch'], 'seeds': seeds, 'model_forward_calls': 0, 'model_backward_calls': 0,
           'optimizer_created': False, 'fixture_only': True}
    path = tmp_path / 'cpu-preflight.json'
    path.write_text(json.dumps(row))
    (tmp_path / 'supervisor.json').write_text(json.dumps({'status': 'complete', 'worker_exit_code': 0, 'deadline_seconds': 240}))
    control.verify_gate(path, registration.file_sha256(path), 'cpu_preflight', cfg, IDENTITY)
    if failure == 'missing_seed': row['seeds'].pop()
    if failure == 'wrong_seed': row['seeds'][1]['seed'] = 11
    if failure == 'history_count': row['seeds'][0]['batch_hashes_verified'] = 1023
    if failure == 'forward_called': row['model_forward_calls'] = 1
    if failure == 'initial_hash': row['seeds'][0]['initial_weights_sha256']['layers.0.linear.bias'] = '0' * 64
    if failure == 'valid_hash': row['seeds'][0]['valid_hash'] = '0' * 64
    if failure == 'label_scope': row['seeds'][0]['input_files_opened'].append('evaluator_truth.json')
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        control.verify_gate(path, registration.file_sha256(path), 'cpu_preflight', cfg, IDENTITY)


def test_rejected_authorization_cannot_open_prepared_labels(tmp_path, monkeypatch):
    cfg, quality, _, approval = approval_quality(tmp_path)
    approval['execution_phase'] = 'cpu_preflight'; approval['supervisor_released'] = False
    ap = tmp_path / 'approval.json'; ap.write_text(json.dumps(approval))
    forbidden = tmp_path / 'must-not-open'; output = tmp_path / 'output'; output.mkdir()
    monkeypatch.setattr(control, 'source_identity', lambda *a: IDENTITY)
    monkeypatch.setenv('ACL_ENCODER_MATCHED_SUPERVISOR_PID', str(os.getppid()))
    monkeypatch.setenv('ACL_ENCODER_MATCHED_DEADLINE', str(time.monotonic() + 240))
    def no_labels(*a, **kw): raise AssertionError('unreviewed execution read labels')
    monkeypatch.setattr(control, 'load_prepared', no_labels)
    monkeypatch.setattr(sys, 'argv', ['control', '--config', str(ROOT / 'configs/e2_encoder_matched_control.json'),
        '--phase', 'cpu_preflight', '--source-commit', IDENTITY['source_commit'], '--approval-record', str(ap),
        '--quality-record', str(quality), '--output', str(output), '--prepared-root', str(forbidden), '--reference-root', str(forbidden)])
    assert control.main() == 2
    failure = json.loads((output / 'failure.json').read_text())
    assert 'user authorization' in failure['error'] and not forbidden.exists()


def test_process_deadline_and_retained_failure(tmp_path):
    out = tmp_path / 'timeout'
    code = "import time; print('owned fixture started',flush=True); time.sleep(10)"
    status = entry.supervise([sys.executable, '-c', code], out, seconds=.3, metadata={'fixture_only': True})
    assert status == 124
    assert json.loads((out / 'supervisor.json').read_text())['status'] == 'timeout'
    assert json.loads((out / 'supervisor-failure.json').read_text())['partial_results_are_complete'] is False
    assert 'owned fixture started' in (out / 'worker.log').read_text()
    with pytest.raises(ValueError): entry.supervise([sys.executable, '-c', 'pass'], out, seconds=1)


def test_archive_static_entry_has_no_git_or_torch_dependency(tmp_path):
    release = tmp_path / 'release'; package = release / 'src/acl_hct'; package.mkdir(parents=True)
    for name in registration.SOURCE_NAMES:
        shutil.copyfile(ROOT / 'src/acl_hct' / (name + '.py'), package / (name + '.py'))
    cfg = release / 'config.json'; shutil.copyfile(ROOT / 'configs/e2_encoder_matched_control.json', cfg)
    env = {**os.environ, 'PYTHONPATH': str(release / 'src')}
    command = [sys.executable, '-m', 'acl_hct.encoder_matched_entry', '--config', str(cfg)]
    result = subprocess.run(command, cwd=release, env=env, capture_output=True, text=True)
    assert result.returncode == 0 and json.loads(result.stdout)['status'] == 'static_only'
    assert subprocess.run(command + ['--execute'], cwd=release, env=env, capture_output=True).returncode != 0
    # Tensor libraries need not be imported until the supervised worker begins.
    code = "import sys,acl_hct.encoder_matched_entry; assert 'torch' not in sys.modules and 'numpy' not in sys.modules"
    assert subprocess.run([sys.executable, '-c', code], cwd=release, env=env, capture_output=True).returncode == 0

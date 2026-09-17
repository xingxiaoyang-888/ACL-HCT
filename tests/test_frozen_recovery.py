"""Synthetic CPU integration; never opens real prepared inputs/checkpoints."""
import copy
from collections import defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from acl_hct import frozen_recovery as runner
from acl_hct import recovery_registration as registration
from acl_hct.backbone import LorentzMeanNetwork
from acl_hct.diagnostic_archive import read_archive, write_archive
from acl_hct.encoder_matched_control import validate_ranking, weights_hashes
from acl_hct.frozen_bias_intervention import FrozenBiasIntervention
from acl_hct.frozen_forward import FrozenForward
from acl_hct.geometry import distance, log, norm2
from acl_hct.recovery_analysis import merge_shards, load_shard
from acl_hct.recovery_entry import parser, phase_seconds, supervise, validate_arguments
from acl_hct.recovery_inputs import VerifiedInputs, prepared_from_cache, checkpoint_from_cache

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = {'source_commit': '1' * 40, 'git': {'available': False, 'commit': None}}


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


def config():
    return json.loads((ROOT / 'configs/e2_frozen_recovery_pilot.json').read_text(encoding='utf-8'))


def toy():
    cfg = config()
    # Most tests need only four coordinates; the fixture gate separately checks 128.
    cfg['model'].update(input_dim=4, hidden=4, head_hidden=5)
    cfg['resources']['run_feasibility']['nonranking_reserve_seconds'] = .05
    m, d, v, p = runner.fixture_data(cfg, 11)
    f, cal = runner.fixture_calibration(m, d['features'], v, p)
    return cfg, m, d, v, f, cal


def prepared(root):
    cfg, model, data, view, _, cal = toy()
    root.mkdir()
    nodes = data['nodes']; index = {n: i for i, n in enumerate(nodes)}
    pairs = []
    for a, b in sorted(view.train_edges):
        parents = {p for p, child in view.train_edges if child == b}
        negatives = [n for n in nodes if n != b and n not in parents][:4]
        pairs.extend([[a, b]] + [[n, b] for n in negatives])
    fit = sorted({b for _, b in view.train_edges})
    features = data['features'].numpy()
    queries = {'queries': pairs, 'labels': [1., 0., 0., 0., 0.] * (len(pairs) // 5), 'negatives_per_positive': 4}
    manifest = {'protocol': 'B-child-grouped-80-10-10-v1', 'split_seed': 20260914,
                'node_order_hash': registration.canonical(nodes), 'graph_hash': registration.canonical(view.neighbors),
                'train_queries_hash': registration.canonical(queries), 'text_fit_entities': fit,
                'feature_manifest': {'nodes': nodes, 'train_entities': fit, 'effective_dimension': 4,
                                     'feature_sha256': hashlib.sha256(features.tobytes()).hexdigest()}}
    valid = [[nodes[a], nodes[b]] for a, b in data['valid']]
    for name, payload in {'input_manifest.json': manifest, 'observed_graph.json': {'nodes': nodes, 'neighbors': view.neighbors},
                          'train_queries.json': queries, 'evaluator_valid.json': valid}.items():
        (root / name).write_text(json.dumps(payload), encoding='utf-8')
    np.savez(root / 'features.npz', features=features)
    for name in ('evaluator_test.json', 'evaluator_truth.json', 'entity_split.json'):
        (root / name).write_text('FORBIDDEN')
    cfg['prepared'] = {'manifest_hash': registration.canonical(manifest), 'node_order_hash': manifest['node_order_hash'],
        'graph_hash': manifest['graph_hash'], 'train_queries_hash': manifest['train_queries_hash'],
        'valid_queries_hash': registration.canonical(valid), 'nodes_count': 40, 'train_groups_count': len(pairs) // 5,
        'input_raw_sha256': {n: registration.file_sha256(root / n) for n in ('input_manifest.json', 'observed_graph.json', 'train_queries.json')},
        'features_npz_raw_sha256': registration.file_sha256(root / 'features.npz'),
        'evaluator_valid_raw_sha256': registration.file_sha256(root / 'evaluator_valid.json')}
    cfg['valid_query_count'] = len(valid)
    return cfg, model, data, view, cal


def declare_prepared(cache, root, cfg):
    for name, digest in {**cfg['prepared']['input_raw_sha256'],
                         'features.npz': cfg['prepared']['features_npz_raw_sha256'],
                         'evaluator_valid.json': cfg['prepared']['evaluator_valid_raw_sha256']}.items():
        cache.allow(root / name, digest, 'prepared/' + name)


def calibrated(root, cal, nodes, view):
    root.mkdir(parents=True)
    panels = {'hash': 'f' * 64, 'panels': {'diagnostic_confirmation': cal['panel']}}
    entry = write_archive(root, 'entry', {'nodes': np.asarray(nodes), 'p': cal['p'], 'floor': cal['floor'], 'panels': panels})
    noise = torch.full((len(nodes),), .004, dtype=torch.float64)
    fanout = write_archive(root, 'fanout_4', {'b': cal['b'], 'variance': noise, 'half_cross': noise - .006,
                                            'noise_corrected_bias_squared': noise - .005})
    fields = {}
    for stem in ('entry', 'fanout_4'):
        record = json.loads((root / (stem + '.json')).read_text())
        for name, token in record['payload'].items():
            if name != 'panels':
                key = token['npz_array']; fields[name] = {'key': key, **record['arrays'][key]}
    return {'R': 256, 'half_counts': [128, 128], 'fields': fields, 'panels_hash': panels['hash'],
            'entry_manifest_raw_sha256': entry['manifest_sha256'], 'entry_npz_raw_sha256': entry['array_file_sha256'],
            'fanout_manifest_raw_sha256': fanout['manifest_sha256'], 'fanout_npz_raw_sha256': fanout['array_file_sha256'],
            'h_dev_hash': view.metadata['h_dev_hash'], 'reference_root': view.root,
            'identity': {k: view.metadata[k] for k in ('graph_hash', 'node_order_hash')},
            'sampling_stream': {'namespace': 'independent-toy-calibration'}}


def test_frozen_config_and_no_tensor_static_import():
    cfg = config()
    assert registration.validate_config(cfg) == registration.CONFIG_SHA256
    assert cfg['resources']['cpu_preflight'] == dict(cpus=2, memory_gib=16, gpus=0, allocation_seconds=300, worker_seconds=240)
    assert phase_seconds(cfg, 'cpu_preflight') == phase_seconds(cfg, 'cuda_fixture') == 240
    assert phase_seconds(cfg, 'science') == 1080
    cfg['pilot']['fanout'] = 8
    with pytest.raises(ValueError): registration.validate_config(cfg)
    code = "import sys; sys.modules['torch']=None; sys.modules['numpy']=None; from acl_hct.recovery_entry import main; raise SystemExit(main())"
    result = subprocess.run([sys.executable, '-c', code, '--config', str(ROOT / 'configs/e2_frozen_recovery_pilot.json')],
                            env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['status'] == 'static_only'


def test_cached_legacy_prepared_reads_and_raw_hashes_once(tmp_path, monkeypatch):
    root = tmp_path / 'prepared'; cfg, _, _, _, _ = prepared(root)
    cache = VerifiedInputs(); declare_prepared(cache, root, cfg)
    read_bytes = Path.read_bytes; sha256 = hashlib.sha256
    opened = []; raw_hash_calls = {}
    def guarded(path):
        if path.parent == root:
            assert path.name not in ('evaluator_test.json', 'evaluator_truth.json', 'entity_split.json')
            opened.append(path.name)
        return read_bytes(path)
    def tracked(raw=b''):
        if isinstance(raw, bytes): raw_hash_calls[id(raw)] = raw_hash_calls.get(id(raw), 0) + 1
        return sha256(raw)
    monkeypatch.setattr(Path, 'read_bytes', guarded)
    monkeypatch.setattr(hashlib, 'sha256', tracked)
    first = prepared_from_cache(cache, root, cfg)
    second = prepared_from_cache(cache, root, cfg)
    assert sorted(opened) == sorted(n.split('/')[-1] for n in cache.receipts)
    assert len(opened) == 5 and len(cache.receipts) == 5
    assert all(raw_hash_calls[id(raw)] == 1 for raw in cache.raw.values())
    torch.testing.assert_close(first['features'], second['features'])
    assert first['input_files_opened'] == ['input_manifest.json', 'observed_graph.json', 'train_queries.json', 'features.npz', 'evaluator_valid.json']
    assert all(r['original_path_reads'] == r['full_file_hash_checks'] == 1 for r in cache.receipts.values())


def test_injected_prepared_is_legacy_equivalent_and_globals_unchanged(tmp_path):
    from acl_hct.backbone_frozen_inputs import load_train_prepared
    from acl_hct.encoder_matched_control import load_prepared
    root = tmp_path / 'prepared'; cfg, _, _, _, _ = prepared(root)
    globals_before = [(fn, dict(fn.__globals__)) for fn in (load_train_prepared, load_prepared)]
    cache = VerifiedInputs(); declare_prepared(cache, root, cfg)
    injected = prepared_from_cache(cache, root, cfg)
    original = load_prepared(root, cfg)
    assert injected.keys() == original.keys()
    for name in original:
        if isinstance(original[name], torch.Tensor): assert torch.equal(injected[name], original[name])
        else: assert injected[name] == original[name]
    for fn, before in globals_before:
        assert fn.__globals__.keys() == before.keys()
        assert all(fn.__globals__[key] is value for key, value in before.items())


@pytest.mark.parametrize('problem', ['undeclared', 'hash', 'alias', 'nonbinary', 'valid_leak'])
def test_verified_inputs_keep_legacy_rejections(tmp_path, problem):
    root = tmp_path / 'prepared'; cfg, _, _, _, _ = prepared(root)
    cache = VerifiedInputs(); declare_prepared(cache, root, cfg)
    if problem == 'undeclared':
        with pytest.raises(ValueError, match='allowlist'): cache.bytes(root / 'evaluator_test.json')
        return
    if problem == 'hash':
        (root / 'features.npz').write_bytes(b'bad')
    elif problem == 'alias':
        with pytest.raises(ValueError, match='duplicate'): cache.allow(root / 'other.json', '0' * 64, 'prepared/features.npz')
        return
    else:
        name = 'train_queries.json' if problem == 'nonbinary' else 'evaluator_valid.json'
        obj = json.loads((root / name).read_text())
        if problem == 'nonbinary': obj['labels'][0] = 1 + 1e-8
        else: obj[0] = ['n00', 'n01']
        (root / name).write_text(json.dumps(obj))
        digest = registration.file_sha256(root / name)
        if name == 'train_queries.json':
            cfg['prepared']['input_raw_sha256'][name] = digest
            # Matching canonical metadata does not waive the binary-label check.
            manifest = json.loads((root / 'input_manifest.json').read_text())
            manifest['train_queries_hash'] = registration.canonical(obj)
            (root / 'input_manifest.json').write_text(json.dumps(manifest))
            cfg['prepared'].update(train_queries_hash=registration.canonical(obj), manifest_hash=registration.canonical(manifest))
            cfg['prepared']['input_raw_sha256']['input_manifest.json'] = registration.file_sha256(root / 'input_manifest.json')
        else: cfg['prepared'].update(evaluator_valid_raw_sha256=digest, valid_queries_hash=registration.canonical(obj))
        cache = VerifiedInputs(); declare_prepared(cache, root, cfg)
    with pytest.raises(ValueError): prepared_from_cache(cache, root, cfg)


def test_checkpoint_same_verified_bytes_and_training_source(tmp_path, monkeypatch):
    from acl_hct.evaluate_checkpoint import REQUIRED_TRAINING_SOURCES, load_verified_checkpoint
    cfg, m, _, _, _, _ = toy()
    release = tmp_path / 'release'; source_hashes = {}
    for name in REQUIRED_TRAINING_SOURCES:
        path = release / 'src' / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic historical source\n')
        source_hashes[name] = hashlib.sha256(path.read_text().encode()).hexdigest()
    cp = {'source': {'source_commit': '2' * 40}, 'source_sha256_normalized_lf': source_hashes,
          'run_status': 'complete', 'completed_steps': 768, 'config': config()['checkpoints']['11']['training_config'],
          'model': m.state_dict()}
    target = tmp_path / 'checkpoint.pt'; torch.save(cp, target)
    cache = VerifiedInputs(); cache.allow(target, registration.file_sha256(target), 'checkpoint')
    old_read = Path.read_bytes; calls = []
    def guarded(path):
        if path == target: calls.append(path)
        return old_read(path)
    monkeypatch.setattr(Path, 'read_bytes', guarded)
    original_torch_load = torch.load; streams = []
    def load(stream, **kw):
        streams.append(stream); assert kw == {'map_location': 'cpu', 'weights_only': True}
        return original_torch_load(stream, **kw)
    monkeypatch.setattr(torch, 'load', load)
    globals_before = dict(load_verified_checkpoint.__globals__)
    loaded, train_cfg = checkpoint_from_cache(cache, target, cache.allowed[target.resolve()][0], '2' * 40, release)
    assert len(calls) == len(streams) == 1 and isinstance(streams[0], io.BytesIO)
    assert train_cfg.seed == 11 and loaded['completed_steps'] == 768
    monkeypatch.setattr(torch, 'load', original_torch_load)
    original, original_cfg = load_verified_checkpoint(target, cache.sha(target), '2' * 40, release)
    assert train_cfg == original_cfg and original.keys() == loaded.keys()
    for name, value in original['model'].items(): assert torch.equal(value, loaded['model'][name])
    for key in original.keys() - {'model'}: assert original[key] == loaded[key]
    assert load_verified_checkpoint.__globals__.keys() == globals_before.keys()
    assert all(load_verified_checkpoint.__globals__[key] is value for key, value in globals_before.items())
    next((release / 'src').rglob('*.py')).write_text('changed')
    with pytest.raises(ValueError, match='source hash'): checkpoint_from_cache(cache, target, cache.sha(target), '2' * 40, release)


def original_fixture(root):
    from acl_hct.evaluate_checkpoint import REQUIRED_TRAINING_SOURCES
    prepared_root = root / 'prepared'
    cfg, _, d, v, _ = prepared(prepared_root)
    release = root / 'training'; hashes = {}
    for name in REQUIRED_TRAINING_SOURCES:
        path = release / 'src' / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic historical bytes\n')
        hashes[name] = hashlib.sha256(path.read_text().encode()).hexdigest()
    cp_root = root / 'checkpoints'; cp_root.mkdir()
    refs = root / 'references'; refs.mkdir()
    for seed in (11, 23):
        m, d, v, panel = runner.fixture_data(cfg, seed)
        f, cal = runner.fixture_calibration(m, d['features'], v, panel)
        spec = cfg['checkpoints'][str(seed)]
        spec['training_config'].update(hidden=4, head_hidden=5)
        spec['training_commit'] = '2' * 40
        truth = defaultdict(set)
        for a, b in d['valid']: truth[b].add(a)
        full = runner.filtered_parent_ranks(m, f.reference['output'], d['valid'], truth, candidate_chunk=11)
        evaluations = []
        for step in (256, 512, 768, 1024):
            row = copy.deepcopy(full)
            if step != 768:
                rr = defaultdict(list)
                for r in row['rows']:
                    r['rank'] = float(r['candidates']); rr[r['child']].append(1 / r['rank'])
                row.update(query_micro_mrr=sum(1 / r['rank'] for r in row['rows']) / len(row['rows']),
                           child_macro_mrr=sum(sum(q) / len(q) for q in rr.values()) / len(rr),
                           hits={str(k): sum(r['rank'] <= k for r in row['rows']) / len(row['rows']) for k in (1, 3, 10)})
            row.update(step=step, purpose='full_validation'); evaluations.append(row)
        baseline = {'status': 'step_limit_reached', 'completed_steps': 1024, 'config': spec['training_config'],
                    'source': {'source_commit': '2' * 40}, 'manifest_hash': cfg['prepared']['manifest_hash'],
                    'validation_queries_hash': cfg['prepared']['valid_queries_hash'], 'evaluations': evaluations,
                    'best_full_valid_mrr': full['query_micro_mrr']}
        reference = refs / f'baseline-seed{seed}-run.json'; reference.write_text(json.dumps(baseline))
        spec['baseline_report_sha256'] = registration.file_sha256(reference)
        cp = {'source': baseline['source'], 'source_sha256_normalized_lf': hashes,
              'run_status': 'step_limit_reached', 'completed_steps': 768, 'config': spec['training_config'],
              'model': m.state_dict(), 'manifest_hash': baseline['manifest_hash'], 'valid_hash': baseline['validation_queries_hash'],
              'best_full_valid_mrr': full['query_micro_mrr'],
              'selection_status': 'selected by complete filtered all-candidate validation query-micro MRR'}
        checkpoint = cp_root / f'seed{seed}-best.pt'; torch.save(cp, checkpoint)
        spec['sha256'] = registration.file_sha256(checkpoint)
        cfg['calibration'][str(seed)] = calibrated(root / 'calibration' / f'seed{seed}', cal, d['nodes'], v)
    cfg['model']['parameter_count'] = sum(p.numel() for p in m.parameters())
    return cfg, v, (prepared_root, cp_root, release, refs, root / 'calibration')


def test_two_model_loader_shared_allowlist_seventeen_reads_and_receipts(tmp_path, monkeypatch):
    cfg, v, inputs = original_fixture(tmp_path)
    monkeypatch.setattr(runner, 'grouped_split', lambda nodes, edges, seed: {'entities': {'valid': v.valid_entities}})
    cache = VerifiedInputs(); read = Path.read_bytes; opened = []
    def guarded(path):
        if path.is_relative_to(tmp_path) and 'training' not in path.parts:
            assert path.name not in ('entity_split.json', 'evaluator_test.json', 'evaluator_truth.json')
            opened.append(str(path))
        return read(path)
    monkeypatch.setattr(Path, 'read_bytes', guarded)
    models = {}
    for seed in (11, 23):
        model, data, view, cal, baseline, proof = runner.load_inputs(cfg, seed, *inputs, time.monotonic() + 60, cache=cache)
        assert proof['selection']['selected_step'] == 768 and len(proof['worker_input_receipts']) == 11
        registration.validate_receipts(proof['worker_input_receipts'], registration.expected_receipts(cfg, (seed,)))
        assert proof['calibration_fields']['b'] == {k: cfg['calibration'][str(seed)]['fields']['b'][k] for k in ('shape', 'dtype', 'data_sha256')}
        models[str(seed)] = proof
    assert len(opened) == len(set(opened)) == len(cache.receipts) == 17
    registration.validate_receipts(cache.receipts, registration.expected_receipts(cfg, (11, 23)))
    evidence = {'status': 'complete', 'phase': 'cpu_preflight', 'config_sha256': registration.CONFIG_SHA256,
                'device': 'cpu', 'declared_resources': cfg['resources']['cpu_preflight'],
                'source_commit': IDENTITY['source_commit'], 'source_lf_sha256': registration.source_hashes(),
                'torch': '2.5.1+cu124', 'seeds': [11, 23], 'models': models, 'forward_backward_optimizer_called': False,
                'worker_input_receipts': cache.receipts, 'all_bulk_inputs_verified_inside_worker': True}
    path = tmp_path / 'cpu-preflight.json'; path.write_text(json.dumps(evidence))
    supervisor = {'status': 'complete', 'worker_exit_code': 0, 'deadline_seconds': 240,
                  'context': {'protocol': registration.PROTOCOL, 'phase': 'cpu_preflight',
                              'config_sha256': registration.CONFIG_SHA256, 'source_commit': IDENTITY['source_commit']}}
    (tmp_path / 'supervisor.json').write_text(json.dumps(supervisor))
    assert registration.verify_gate(path, registration.file_sha256(path), 'cpu_preflight', IDENTITY, cfg)
    evidence['models']['11']['worker_input_receipts']['prepared/features.npz']['original_path_reads'] = 2
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match='single-read'): registration.verify_gate(path, registration.file_sha256(path), 'cpu_preflight', IDENTITY, cfg)


def test_calibration_cache_identity_uncertainty_and_signed_fields(tmp_path, monkeypatch):
    _, _, d, v, _, cal = toy(); root = tmp_path / 'cal'
    spec = calibrated(root, cal, d['nodes'], v)
    read_bytes = Path.read_bytes; calls = []
    def guarded(path):
        if path.parent == root: calls.append(path.name)
        return read_bytes(path)
    monkeypatch.setattr(Path, 'read_bytes', guarded)
    cache = VerifiedInputs()
    result = runner.load_calibration(root, spec, d['nodes'], time.monotonic() + 30, cache)
    runner.load_calibration(root, spec, d['nodes'], time.monotonic() + 30, cache)
    assert sorted(calls) == ['entry.json', 'entry.npz', 'fanout_4.json', 'fanout_4.npz']
    assert (result['half_cross'] < 0).all() and (result['noise_corrected_bias_squared'] < 0).all()
    torch.testing.assert_close(result['uncertainty']['mean_estimator_se_norm'], torch.full((40,), np.sqrt(.004 / 255), dtype=torch.float64))
    assert result['uncertainty']['half_mean_vectors_available'] is False
    with pytest.raises(ValueError, match='node order'): runner.load_calibration(root, spec, d['nodes'][::-1], time.monotonic() + 30)
    (root / 'fanout_4.npz').write_bytes(b'bad')
    with pytest.raises(ValueError, match='raw SHA'): runner.load_calibration(root, spec, d['nodes'], time.monotonic() + 30)


@pytest.mark.parametrize('repeat', [0, 7, 8, 15])
def test_repeat_stream_reproduces_across_shards_and_q_is_fixed(repeat):
    cfg, m, d, v, f, cal = toy()
    before = torch.get_rng_state()
    kwargs = dict(calibration_namespace='old-independent', evaluation_namespace='evaluation',
                  direction_namespace=cfg['pilot']['q_namespace'].format(seed=11), direction_seed=cfg['pilot']['q_seed'])
    adapter = FrozenBiasIntervention(cal['p'], cal['b'], **kwargs)
    plans, rng = runner.repeat_plans(cfg, 11, repeat, v.neighbors)
    reconstructed, _ = runner.repeat_plans(cfg, 11, repeat, v.neighbors)
    assert plans == reconstructed
    assert rng['namespace'] == f'E2-frozen-recovery-v1/seed11/repeat{repeat}'
    moved = runner.evaluate_repeat(m, d['features'], v, f, adapter, cfg, 11, repeat)
    assert moved['plans'] == plans
    expected = m.encode(d['features'], v.neighbors, plans, method='third')[0]
    assert torch.equal(moved['native']['C'], expected)
    other = FrozenBiasIntervention(cal['p'], -cal['b'], **kwargs)
    repeated = runner.evaluate_repeat(m, d['features'], v, f, other, cfg, 11, repeat)
    assert torch.equal(adapter.control, other.control)
    assert torch.equal(moved['native']['C'], repeated['native']['C'])
    assert torch.equal(before, torch.get_rng_state())


def test_native_zero_cast_scores_and_ranks_and_resolution():
    cfg, m, d, v, f, cal = toy()
    sample = m.encode(d['features'], v.neighbors, runner.repeat_plans(cfg, 11, 0, v.neighbors)[0])[0]
    sample64 = runner.promoted(sample, m.c)
    points, audit = runner.cast_for_head(sample64, sample, sample64, torch.zeros_like(cal['b']), cal['floor'], m.c, cfg['numeric_proposal'])
    assert torch.equal(points, sample) and audit['zero_native_identity']
    assert not audit['resolved'].any()
    q = torch.tensor(d['valid'])
    assert torch.equal(m.score(points, q), m.score(sample, q))
    scale = 1 + sample64.norm(dim=-1)
    torch.testing.assert_close(audit['resolution_proxy'], 8 * torch.finfo(torch.float32).eps * scale)
    limits = {**cfg['numeric_proposal'], 'cast_absolute_geodesic': -1}
    with pytest.raises(ValueError, match='cast'): runner.cast_for_head(sample64, sample, sample64, cal['b'], cal['floor'], m.c, limits)


def test_cast_relative_gate_resolved_and_unresolved_nodes_retained():
    cfg, m, d, v, f, cal = toy()
    adapter = FrozenBiasIntervention(cal['p'], cal['b'], calibration_namespace='old', evaluation_namespace='new',
                                    direction_namespace='control', direction_seed=2026091706)
    result = runner.evaluate_repeat(m, d['features'], v, f, adapter, cfg, 11, 0)
    native, audit = runner.cast_for_head(result['analytic']['O'], result['native']['S'], result['sample64'],
                                        cal['b'], cal['floor'], m.c, cfg['numeric_proposal'])
    assert len(native) == len(audit['resolved']) == 40 and audit['resolved'].sum() == 39
    assert float(audit['relative_error'][audit['resolved']].max()) > 0
    limits = {**cfg['numeric_proposal'], 'cast_relative_resolved': 0.}
    with pytest.raises(ValueError, match='relative'): runner.cast_for_head(result['analytic']['O'], result['native']['S'],
        result['sample64'], cal['b'], cal['floor'], m.c, limits)


@pytest.mark.parametrize('seed', [11, 23])
def test_synthetic_production_dimension_fixture_archive(tmp_path, seed):
    cfg = config()
    checks, desc = runner.equivalence_fixture(cfg, seed, torch.device('cpu'), tmp_path, time.monotonic() + 60)
    data = read_archive(tmp_path, desc)
    assert all(checks.values()) and len(checks) == 8
    assert data['features'].shape == (40, 128)
    assert len(data['native_points']) == 10 and sum(v.size for v in data['weights'].values()) == 82433
    assert set(data['plans']) == {'0', '8'} and len(data['rankings']['8/Q']['rows']) == 7
    assert len(data['neighbors']) == 40 and set(data['FP64_reference_points']) == {'F', '0/S', '0/C', '8/S', '8/C'}


def test_cuda_phase_gate_requires_both_complete_bound_archives(tmp_path):
    cfg = config(); archives = {}; checks = None
    for seed in (11, 23):
        checks, archives[str(seed)] = runner.equivalence_fixture(cfg, seed, torch.device('cpu'), tmp_path, time.monotonic() + 60)
    # Fabricated runtime metadata is confined to this synthetic gate test.
    evidence = {'status': 'complete', 'phase': 'cuda_fixture', 'config_sha256': registration.CONFIG_SHA256,
                'source_commit': IDENTITY['source_commit'], 'source_lf_sha256': registration.source_hashes(),
                'torch': '2.5.1+cu124', 'seeds': [11, 23], 'device': 'NVIDIA L40',
                'original_runtime_inputs_opened': [], 'checks': checks, 'archives': archives}
    path = tmp_path / 'cuda-fixture.json'; path.write_text(json.dumps(evidence))
    supervisor = {'status': 'complete', 'worker_exit_code': 0, 'deadline_seconds': 240,
                  'context': {'protocol': registration.PROTOCOL, 'phase': 'cuda_fixture',
                              'config_sha256': registration.CONFIG_SHA256, 'source_commit': IDENTITY['source_commit']}}
    (tmp_path / 'supervisor.json').write_text(json.dumps(supervisor))
    assert registration.verify_gate(path, registration.file_sha256(path), 'cuda_fixture', IDENTITY, cfg)
    evidence['checks']['cast_limits'] = False
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match='fixture checks'): registration.verify_gate(path, registration.file_sha256(path), 'cuda_fixture', IDENTITY, cfg)
    evidence['checks']['cast_limits'] = True; path.write_text(json.dumps(evidence))
    (tmp_path / archives['11']['array_file']).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='NPZ hash'): registration.verify_gate(path, registration.file_sha256(path), 'cuda_fixture', IDENTITY, cfg)


def test_eight_repeat_toy_run_complete_archive_and_frozen_state(tmp_path):
    cfg, m, d, v, f, cal = toy(); before = weights_hashes(m)
    row = runner.run_slice(m, d, v, cal, cfg, 11, [8, 16], tmp_path, time.monotonic() + 60, engineering=True)
    assert row['status'] == 'complete', row.get('failure')
    assert [r['repeat'] for r in row['observations']] == list(range(8, 16))
    assert row['completed_repetitions'] == len(row['repeat_archives']) == 8
    assert before == weights_hashes(m) and row['model_updates'] == 0 and all(p.grad is None for p in m.parameters())
    payload = read_archive(tmp_path, row['repeat_archives'][0])
    assert set(payload['native_points']) == {'S', 'O', 'C', 'Q'}
    assert payload['native_points']['Q'].dtype == np.float32
    assert set(payload['casting']['O']) >= {'resolved', 'actual_step', 'error', 'resolution_proxy'}
    assert all(len(payload['casting'][name]['resolved']) == 40 for name in ('O', 'Q'))
    with pytest.raises(ValueError, match='no resume'): runner.run_slice(m, d, v, cal, cfg, 11, [8, 16], tmp_path, time.monotonic() + 60, engineering=True)


@pytest.mark.parametrize('problem', ['deadline', 'partial_ranking', 'calibration_base', 'production_cpu'])
def test_run_stops_without_false_complete(tmp_path, monkeypatch, problem):
    cfg, m, d, v, _, cal = toy()
    deadline = time.monotonic() - 1 if problem == 'deadline' else time.monotonic() + 60
    if problem == 'partial_ranking': cfg['pilot']['ranking_seconds'] = 0
    if problem == 'calibration_base': cal['p'] = torch.roll(cal['p'], 1, 0)
    if problem == 'production_cpu':
        with pytest.raises(ValueError, match='CUDA'): runner.run_slice(m, d, v, cal, cfg, 11, [0, 8], tmp_path, deadline)
        return
    row = runner.run_slice(m, d, v, cal, cfg, 11, [0, 8], tmp_path, deadline, engineering=True)
    assert row['status'] != 'complete' and row['completed_repetitions'] == 0
    assert row['failure']['partial_results_are_complete'] is False and row['weights_unchanged']
    if problem == 'partial_ranking':
        payload = read_archive(tmp_path, row['partial_ranking_archive'])
        assert payload['ranking']['status'] == 'incomplete_time_limit'


def test_mid_shard_failure_keeps_finished_archives_and_counts(tmp_path, monkeypatch):
    cfg, m, d, v, _, cal = toy(); original = runner.evaluate_repeat
    def fail(model, features, view, forward, intervention, config, seed, repeat):
        if repeat == 3: raise ValueError('injected forward failure')
        return original(model, features, view, forward, intervention, config, seed, repeat)
    monkeypatch.setattr(runner, 'evaluate_repeat', fail)
    row = runner.run_slice(m, d, v, cal, cfg, 11, [0, 8], tmp_path, time.monotonic() + 60, engineering=True)
    assert row['status'] == 'failed' and row['completed_repetitions'] == len(row['repeat_archives']) == 3
    assert row['active_repeat'] == 3 and row['failure']['phase'] == 'paired_forward'
    for desc in row['repeat_archives']: assert read_archive(tmp_path, desc)['repeat'] in range(3)


def test_resource_feasibility_stops_before_any_evaluation_and_keeps_F(tmp_path, monkeypatch):
    cfg, m, d, v, _, cal = toy()
    cfg['resources']['run_feasibility'] = config()['resources']['run_feasibility']
    def forbidden(*args, **kwargs): raise AssertionError('no S/O/C/Q repeat may start')
    monkeypatch.setattr(runner, 'evaluate_repeat', forbidden)
    row = runner.run_slice(m, d, v, cal, cfg, 11, [0, 8], tmp_path, time.monotonic() + 60, engineering=True)
    assert row['status'] == 'resource_time_budget_inadequate' and row['completed_repetitions'] == 0
    assert not row['resource_feasibility']['passed'] and row['resource_feasibility']['remaining_complete_rankings'] == 32
    gate = row['resource_feasibility']
    assert gate['estimated_remaining_seconds'] == gate['measured_F_ranking_seconds'] * 32 + 180
    assert read_archive(tmp_path, row['entry_archive'])['F_ranking']['status'] == 'complete'
    assert row['phase_costs'] and row['peak_vram'] is None


def test_merge_archive_reader_recomputes_all_points_and_rank_summaries(tmp_path):
    cfg, m, d, v, _, cal = toy()
    cfg['prepared'].update(nodes_count=40, node_order_hash=registration.canonical(d['nodes']),
                           valid_queries_hash=registration.canonical([[d['nodes'][a], d['nodes'][b]] for a, b in d['valid']]))
    cfg['valid_query_count'] = len(d['valid'])
    cfg['calibration']['11']['fields'] = {k: {'key': k, **runner.tensor_identity(cal[v])} for k, v in [('p', 'p'), ('b', 'b'), ('floor', 'floor')]}
    provenance = {**IDENTITY, 'release_verified': True}
    row = runner.run_slice(m, d, v, cal, cfg, 11, [0, 8], tmp_path, time.monotonic() + 60, provenance=provenance, engineering=True)
    assert row['status'] == 'complete', row.get('failure')
    # Only this synthetic test fabricates a science marker for exercising the reader.
    row['engineering_fixture_only'] = False
    (tmp_path / 'run.json').write_text(json.dumps(row))
    (tmp_path / 'supervisor.json').write_text(json.dumps({'status': 'complete', 'worker_exit_code': 0, 'deadline_seconds': 1080}))
    assert load_shard(tmp_path, cfg)['completed_repetitions'] == 8
    row['observations'][0]['metrics']['S']['direct'] = .123
    (tmp_path / 'run.json').write_text(json.dumps(row))
    with pytest.raises(ValueError, match='metrics differ'): load_shard(tmp_path, cfg)


def shards():
    result = []
    for seed in (11, 23):
        identity = {'F_metrics': dict.fromkeys(('direct', 'distant', 'micro_mrr'), .6), 'p': str(seed), 'q': str(seed)}
        for start in (0, 8):
            observations = []
            for i in range(start, start + 8):
                values = {'S': .4 + i * .001, 'C': .42 + i * .002, 'O': .43 + i * .003, 'Q': .415 + i * .0015}
                observations.append({'repeat': i, 'metrics': {k: dict.fromkeys(('direct', 'distant', 'micro_mrr'), v) for k, v in values.items()}})
            result.append({'status': 'complete', 'protocol': registration.PROTOCOL, 'config_sha256': registration.CONFIG_SHA256,
                           'seed': seed, 'repeat_range': [start, start + 8], 'identity': identity, 'observations': observations})
    return result


def test_all_eighteen_paired_student_intervals_and_holm():
    from scipy.stats import t
    records = shards(); result = merge_shards(records)
    assert result['family_size'] == len(result['primary_comparisons']) == 18
    row = result['primary_comparisons'][0]
    differences = np.array([.02 + i * .001 for i in range(16)])
    expected_se = differences.std(ddof=1) / 4
    assert row['mc_se'] == pytest.approx(expected_se)
    assert row['marginal_95_interval'] == pytest.approx([differences.mean() - t.ppf(.975, 15) * expected_se, differences.mean() + t.ppf(.975, 15) * expected_se])
    expected_p = float(2 * t.sf(abs(differences.mean() / expected_se), 15))
    assert row['two_sided_p'] == pytest.approx(expected_p)
    ordered = sorted(result['primary_comparisons'], key=lambda r: r['two_sided_p'])
    previous = 0
    for i, item in enumerate(ordered):
        previous = max(previous, min(1, (18 - i) * item['two_sided_p']))
        assert item['holm_adjusted_p'] == previous
    assert len(result['sampling_damage_S_minus_F']) == 6


def test_constant_difference_is_not_inferable_or_zero_width_evidence():
    records = shards()
    for shard in records:
        for row in shard['observations']:
            for condition in row['metrics']:
                row['metrics'][condition] = dict.fromkeys(('direct', 'distant', 'micro_mrr'), {'S': .25, 'C': .5, 'O': .75, 'Q': .375}[condition])
    result = merge_shards(records)
    assert all(r['zero_observed_variance'] and not r['inferable'] and r['marginal_95_interval'] is None
               and r['two_sided_p'] == r['holm_adjusted_p'] == 1 for r in result['primary_comparisons'])


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'identity', 'condition', 'unknown', 'order'])
def test_merge_refuses_incomplete_or_selected_evidence(problem):
    records = shards()
    if problem == 'missing': records.pop()
    if problem == 'duplicate': records[1] = records[0]
    if problem == 'identity': records[1]['identity'] = {**records[1]['identity'], 'q': 'changed'}
    if problem == 'condition': del records[0]['observations'][0]['metrics']['Q']
    if problem == 'unknown': records[0]['observations'][0]['metrics']['O']['direct'] = None
    if problem == 'order': records[0]['observations'].reverse()
    with pytest.raises(ValueError): merge_shards(records)


def test_numerical_review_release_required_before_any_input(tmp_path):
    cfg = config()
    quality = {'status': 'passed', 'config_canonical_sha256': registration.CONFIG_SHA256,
               'source_lf_sha256': registration.source_hashes()}
    path = tmp_path / 'quality.json'; path.write_text(json.dumps(quality))
    approval = dict(user_authorized=True, quality_review_passed=True, entry_criteria_frozen=True,
                    supervisor_released=True, numerical_limits_reviewed=True, scope=registration.PROTOCOL,
                    config_sha256=registration.CONFIG_SHA256, source_commit=IDENTITY['source_commit'],
                    execution_phase='cpu_preflight', user_message_reference='synthetic authorized test',
                    quality_record_sha256=registration.file_sha256(path))
    assert registration.verify_release(cfg, IDENTITY, approval, path, 'cpu_preflight')
    for field in ('numerical_limits_reviewed', 'user_authorized', 'supervisor_released'):
        bad = {**approval, field: False}
        with pytest.raises(ValueError, match='release'): registration.verify_release(cfg, IDENTITY, bad, path, 'cpu_preflight')
    path.write_text('changed')
    with pytest.raises(ValueError, match='SHA'): registration.verify_release(cfg, IDENTITY, approval, path, 'cpu_preflight')


@pytest.mark.parametrize('problem', ['original_fixture', 'science_no_gates', 'preflight_shard'])
def test_exact_phase_argument_allowlist(problem):
    common = ['--config', 'dummy', '--source-commit', '1' * 40, '--approval-record', 'a', '--quality-record', 'q', '--output', 'o']
    if problem == 'original_fixture': args = parser().parse_args(common + ['--phase', 'cuda_fixture', '--prepared-root', 'forbidden'])
    else:
        original = sum(([name, 'x'] for name in ('--prepared-root', '--checkpoint-root', '--training-release', '--reference-root', '--calibration-root')), [])
        args = parser().parse_args(common + original + ['--phase', 'science' if problem == 'science_no_gates' else 'cpu_preflight', '--seed', '11'])
    with pytest.raises(ValueError): validate_arguments(args)


def test_process_deadline_covers_imports_and_preserves_partial_phase(tmp_path):
    output = tmp_path / 'timed'
    code = "import json,time,pathlib; p=pathlib.Path(" + repr(str(output)) + "); (p/'run.json').write_text(json.dumps({'completed_repetitions':3,'active_repeat':3,'active_phase':'Q_ranking'})); time.sleep(20)"
    assert supervise([sys.executable, '-c', code], output, seconds=.5) == 124
    failure = json.loads((output / 'supervisor-failure.json').read_text())
    assert failure['completed_repetitions'] == failure['active_repeat'] == 3
    assert failure['active_phase'] == 'Q_ranking' and failure['partial_results_are_complete'] is False
    with pytest.raises(ValueError, match='empty'): supervise([sys.executable, '-c', 'pass'], output, seconds=.5)

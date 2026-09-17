"""Tiny offline fixtures only: no model execution or production input files."""
import builtins
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from acl_hct.backbone_frozen_inputs import (
    immutable_input_hash, load_batch_history, load_train_prepared, replay_batches,
)
from acl_hct.protocols import digest, observed_neighbors, training_candidates


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _bind(root):
    manifest = json.loads((root / 'input_manifest.json').read_text(encoding='utf-8'))
    graph = json.loads((root / 'observed_graph.json').read_text(encoding='utf-8'))
    queries = json.loads((root / 'train_queries.json').read_text(encoding='utf-8'))
    manifest.update(node_order_hash=digest(graph['nodes']), graph_hash=digest(graph['neighbors']),
                    train_queries_hash=digest(queries))
    _write_json(root / 'input_manifest.json', manifest)
    return {'manifest_hash': digest(manifest), 'node_order_hash': manifest['node_order_hash'],
            'graph_hash': manifest['graph_hash'], 'train_queries_hash': manifest['train_queries_hash'],
            'valid_queries_hash': digest([['n0', 'n7']]), 'nodes_count': len(graph['nodes']),
            'train_groups_count': len(queries['labels']) // 5,
            'input_raw_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                 for name in ('input_manifest.json', 'observed_graph.json', 'train_queries.json')}}


@pytest.fixture
def tiny_prepared(tmp_path):
    root = tmp_path / 'tiny_prepared'
    root.mkdir()
    nodes = [f'n{i}' for i in range(8)]
    edges = [('n0', 'n1'), ('n2', 'n1')]
    features = np.arange(32, dtype=np.float32).reshape(8, 4) / 32
    fit_entities = ['n1', 'n3']
    manifest = {'protocol': 'B-child-grouped-80-10-10-v1', 'split_seed': 20260914,
                'text_fit_entities': fit_entities,
                'feature_manifest': {'nodes': nodes, 'train_entities': fit_entities,
                                     'effective_dimension': 4,
                                     'feature_sha256': hashlib.sha256(features.tobytes()).hexdigest()}}
    _write_json(root / 'input_manifest.json', manifest)
    _write_json(root / 'observed_graph.json', {'nodes': nodes, 'neighbors': observed_neighbors(nodes, edges)})
    _write_json(root / 'train_queries.json', training_candidates(nodes, fit_entities, edges))
    np.savez_compressed(root / 'features.npz', features=features)
    return root, _bind(root)


def test_loader_never_opens_evaluator_labels_and_preserves_original_groups(tiny_prepared, monkeypatch):
    root, expected = tiny_prepared
    allowed = {'input_manifest.json', 'observed_graph.json', 'train_queries.json', 'features.npz'}
    opened = set()
    path_open = Path.open
    native_open = builtins.open

    def checked_path(path, *args, **kwargs):
        if path.parent == root:
            assert path.name in allowed
            opened.add(path.name)
        return path_open(path, *args, **kwargs)

    def checked_native(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).parent == root:
            assert Path(path).name in allowed
            opened.add(Path(path).name)
        return native_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', checked_path)
    monkeypatch.setattr(builtins, 'open', checked_native)
    data = load_train_prepared(root, expected)
    assert opened == allowed
    assert 'valid' not in data and data['valid_hash'] == expected['valid_queries_hash']
    assert data['input_files_opened'] == list(expected['input_raw_sha256']) + ['features.npz']
    assert data['features'].dtype == torch.float32 and data['features'].device.type == 'cpu'
    assert data['features'].shape == (8, 4) and data['query_groups'].shape == (2, 5, 2)
    assert data['query_groups'][:, 0].tolist() == [[0, 1], [2, 1]]
    assert data['labels'].tolist() == [[1., 0., 0., 0., 0.]] * 2


@pytest.mark.parametrize('field', ['manifest_hash', 'node_order_hash', 'graph_hash', 'train_queries_hash'])
def test_registered_canonical_identity_rejects_mismatch(tiny_prepared, field):
    root, expected = tiny_prepared
    expected = copy.deepcopy(expected)
    expected[field] = '0' * 64
    with pytest.raises(ValueError, match='hash mismatch'):
        load_train_prepared(root, expected)


def test_raw_json_bytes_and_feature_bytes_are_independently_bound(tiny_prepared):
    root, expected = tiny_prepared
    path = root / 'train_queries.json'
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='raw hash mismatch'):
        load_train_prepared(root, expected)
    expected = _bind(root)
    features = np.zeros((8, 4), dtype=np.float32)
    np.savez_compressed(root / 'features.npz', features=features)
    with pytest.raises(ValueError, match='feature contract/hash mismatch'):
        load_train_prepared(root, expected)


@pytest.mark.parametrize('case, error', [
    ('reverse', 'reverse directions'), ('self', 'no self loops'),
    ('extra_edge', 'equal train positives'), ('duplicate_negative', 'fixed training negatives'),
    ('other_true_parent_negative', 'fixed training negatives'),
    ('child_mismatch', 'fixed training negatives'), ('near_binary_label', 'exactly one positive'),
    ('text_fit', 'train child'),
])
def test_train_contracts_are_verified_without_evaluator_labels(tiny_prepared, case, error):
    root, _ = tiny_prepared
    graph_path = root / 'observed_graph.json'
    query_path = root / 'train_queries.json'
    manifest_path = root / 'input_manifest.json'
    graph = json.loads(graph_path.read_text(encoding='utf-8'))
    queries = json.loads(query_path.read_text(encoding='utf-8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if case == 'reverse':
        graph['neighbors'][1].remove(0)
    elif case == 'self':
        graph['neighbors'][0] = [0, 1]
    elif case == 'extra_edge':
        graph['neighbors'][3] = [4]
        graph['neighbors'][4] = [3]
    elif case == 'duplicate_negative':
        queries['queries'][2][0] = queries['queries'][1][0]
    elif case == 'other_true_parent_negative':
        queries['queries'][1][0] = 'n2'
    elif case == 'child_mismatch':
        queries['queries'][1][1] = 'n3'
    elif case == 'near_binary_label':
        queries['labels'][0] = 1 + 1e-9
    elif case == 'text_fit':
        manifest['text_fit_entities'] = ['n3', 'n4']
        manifest['feature_manifest']['train_entities'] = ['n3', 'n4']
    _write_json(graph_path, graph)
    _write_json(query_path, queries)
    _write_json(manifest_path, manifest)
    expected = _bind(root)
    with pytest.raises(ValueError, match=error):
        load_train_prepared(root, expected)


@pytest.mark.parametrize('field', ['features', 'neighbors', 'query_groups', 'labels', 'manifest', 'valid_hash'])
def test_immutable_hash_detects_each_scientific_input_mutation(tiny_prepared, field):
    root, expected = tiny_prepared
    data = load_train_prepared(root, expected)
    before = immutable_input_hash(data)
    assert immutable_input_hash(data) == before
    if field == 'features':
        data[field][0, 0] += 1
    elif field == 'neighbors':
        data[field][0].append(3)
    elif field == 'query_groups':
        data[field][0, 1, 0] = (data[field][0, 1, 0] + 1) % len(data['nodes'])
    elif field == 'labels':
        data[field][0, 1] = 1
    elif field == 'manifest':
        data[field]['split_seed'] += 1
    else:
        data[field] = '0' * 64
    assert immutable_input_hash(data) != before


def _tiny_history(seed=11, settings_name='training_settings'):
    generator = torch.Generator(device='cpu').manual_seed(seed + 1)
    chosen = [torch.randperm(8, generator=generator)[:2] for _ in range(5)]
    return {'status': 'complete', 'seed': seed, settings_name: {'seed': seed, 'batch_positives': 2},
            'steps': [{'step': step, 'batch_group_ids_sha256': digest(ids.tolist())}
                      for step, ids in enumerate(chosen, 1)]}, chosen


@pytest.mark.parametrize('settings_name', ['training_settings', 'config'])
def test_replay_matches_original_stream_and_leaves_global_rng_untouched(tmp_path, settings_name):
    history, chosen = _tiny_history(settings_name=settings_name)
    path = tmp_path / 'tiny_history.json'
    _write_json(path, history)
    expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_batch_history(path, expected_sha, 11)
    rng_before = torch.get_rng_state().clone()
    result = replay_batches(8, 11, [4, 5], 2, loaded, expected_sha)
    assert list(result) == [4, 5]
    assert all(torch.equal(result[step], chosen[step - 1]) for step in [4, 5])
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert all(ids.device.type == 'cpu' and ids.dtype == torch.int64 for ids in result.values())


def test_replay_fails_at_first_mismatching_historical_batch():
    history, _ = _tiny_history()
    history['steps'][1]['batch_group_ids_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='hash mismatch at step 2'):
        replay_batches(8, 11, [5], 2, history)


@pytest.mark.parametrize('case, error', [
    ('wrong_seed', 'seed mismatch'), ('partial', 'completed run'),
    ('missing_steps', 'missing required steps'), ('reordered', 'step order mismatch'),
    ('unverified_sha', 'SHA was not verified'),
])
def test_history_rejects_invalid_identity_or_order(case, error):
    history, _ = _tiny_history()
    expected_sha = None
    if case == 'wrong_seed':
        history['training_settings']['seed'] = 23
    elif case == 'partial':
        history['status'] = 'running'
    elif case == 'missing_steps':
        history['steps'].pop()
    elif case == 'reordered':
        history['steps'][1]['step'] = 3
    else:
        expected_sha = '0' * 64
    with pytest.raises(ValueError, match=error):
        replay_batches(8, 11, [5], 2, history, expected_sha)


def test_history_loader_rejects_raw_byte_tampering(tmp_path):
    history, _ = _tiny_history()
    path = tmp_path / 'tiny_history.json'
    _write_json(path, history)
    expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='raw hash mismatch'):
        load_batch_history(path, expected_sha, 11)


@pytest.mark.parametrize('seed', [11, 23])
def test_actual_capacity_multi_seed_schema_uses_explicit_per_run_settings(tmp_path, seed):
    history, chosen = _tiny_history(seed)
    history['config'] = {'protocol': 'E2-model-capacity-control-v1', 'seeds': [11, 23],
                         'training': {'batch_positives': 2}}
    path = tmp_path / 'multi_seed_run.json'
    _write_json(path, history)
    raw_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_batch_history(path, raw_sha, seed)
    replayed = replay_batches(8, seed, [4, 5], 2, loaded, raw_sha)
    assert all(torch.equal(replayed[step], chosen[step - 1]) for step in (4, 5))
    with pytest.raises(ValueError, match='batch_positives mismatch'):
        replay_batches(8, seed, [5], 3, loaded, raw_sha)


@pytest.mark.parametrize('case', ['top_seed', 'settings_seed', 'config_seed', 'registration_membership',
                                  'duplicate_registration', 'boolean_registration', 'missing_top_seed',
                                  'missing_settings', 'missing_settings_seed', 'registration_batch',
                                  'config_batch', 'malformed_config'])
def test_multi_seed_schema_rejects_conflicts_in_both_loader_and_replay(tmp_path, case):
    history, _ = _tiny_history()
    history['config'] = {'seeds': [11, 23], 'training': {'batch_positives': 2}}
    if case == 'top_seed':
        history['seed'] = 23
    elif case == 'settings_seed':
        history['training_settings']['seed'] = 23
    elif case == 'config_seed':
        history['config']['seed'] = 23
    elif case == 'registration_membership':
        history['config']['seeds'] = [23]
    elif case == 'duplicate_registration':
        history['config']['seeds'] = [11, 11, 23]
    elif case == 'boolean_registration':
        history['config']['seeds'] = [True, 11, 23]
    elif case == 'missing_top_seed':
        del history['seed']
    elif case == 'missing_settings':
        del history['training_settings']
    elif case == 'missing_settings_seed':
        del history['training_settings']['seed']
    elif case == 'registration_batch':
        history['config']['training']['batch_positives'] = 3
    elif case == 'config_batch':
        history['config']['batch_positives'] = 3
    else:
        history['config'] = None
    path = tmp_path / 'conflicting_multi_seed_run.json'
    _write_json(path, history)
    raw_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        load_batch_history(path, raw_sha, 11)
    with pytest.raises(ValueError):
        replay_batches(8, 11, [5], 2, history)


@pytest.mark.parametrize('case', ['nonfinite', 'wrong_dtype', 'wrong_shape'])
def test_feature_numerical_contract_is_not_replaced_by_hash_match(tiny_prepared, case):
    root, _ = tiny_prepared
    features = np.zeros((8, 4), dtype=np.float32)
    if case == 'nonfinite':
        features[0, 0] = np.nan
    elif case == 'wrong_dtype':
        features = features.astype(np.float64)
    else:
        features = features[:, :3]
    np.savez_compressed(root / 'features.npz', features=features)
    path = root / 'input_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest['feature_manifest']['feature_sha256'] = hashlib.sha256(features.tobytes()).hexdigest()
    _write_json(path, manifest)
    expected = _bind(root)
    with pytest.raises(ValueError, match='feature contract/hash mismatch'):
        load_train_prepared(root, expected)

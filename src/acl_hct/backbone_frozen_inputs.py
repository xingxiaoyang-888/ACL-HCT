"""Train-only provenance and original batch replay for fixed-weight diagnosis.

This module does not instantiate a model or read evaluator labels. Validation
identity is supplied by the frozen registration and checked by the caller
against historical runs/checkpoints; it is never recomputed from valid labels.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .backbone import validate_neighbors
from .protocols import digest


_JSON_INPUTS = ('input_manifest.json', 'observed_graph.json', 'train_queries.json')
_IDENTITY_FIELDS = ('manifest_hash', 'node_order_hash', 'graph_hash',
                    'train_queries_hash', 'valid_queries_hash')


def _sha256(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in '0123456789abcdef' for char in value)):
        raise ValueError(f'{name} must be a lowercase SHA256')
    return value


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _registered_inputs(expected):
    if not isinstance(expected, dict):
        raise ValueError('expected prepared identity must be an object')
    for name in _IDENTITY_FIELDS:
        _sha256(expected.get(name), name)
    for name in ('nodes_count', 'train_groups_count'):
        _positive_int(expected.get(name), name)
    raw = expected.get('input_raw_sha256')
    if not isinstance(raw, dict) or set(raw) != set(_JSON_INPUTS):
        raise ValueError('raw input identity must contain exactly the three train JSON files')
    for name in _JSON_INPUTS:
        _sha256(raw[name], name)


def load_train_prepared(root, expected):
    """Load only manifest, observed graph, training groups and features.

    ``expected`` binds canonical hashes, three JSON byte hashes and node/group
    counts. Feature bytes are bound by the already verified manifest. No valid,
    test, truth or entity-split file is opened, including for existence checks.
    """
    _registered_inputs(expected)
    root = Path(root)
    payloads = {}
    for name in _JSON_INPUTS:
        raw = (root / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected['input_raw_sha256'][name]:
            raise ValueError(f'prepared raw hash mismatch: {name}')
        payloads[name] = json.loads(raw.decode('utf-8'))
    manifest = payloads['input_manifest.json']
    graph = payloads['observed_graph.json']
    queries = payloads['train_queries.json']
    if digest(manifest) != expected['manifest_hash']:
        raise ValueError('prepared manifest hash mismatch')
    if (manifest.get('protocol') != 'B-child-grouped-80-10-10-v1'
            or manifest.get('split_seed') != 20260914):
        raise ValueError('requires original frozen protocol B V1 split seed')
    nodes = graph['nodes']
    if (not isinstance(nodes, list) or len(nodes) != expected['nodes_count']
            or any(not isinstance(node, str) for node in nodes)
            or len(set(nodes)) != len(nodes)):
        raise ValueError('node identity/count contract mismatch')
    neighbors = graph['neighbors']
    actual = {'node_order_hash': digest(nodes), 'graph_hash': digest(neighbors),
              'train_queries_hash': digest(queries)}
    if any(actual[name] != expected[name] or actual[name] != manifest.get(name)
           for name in actual):
        raise ValueError('prepared canonical input hash mismatch')
    validate_neighbors(neighbors, len(nodes))
    if any(row != sorted(row) or i in row for i, row in enumerate(neighbors)):
        raise ValueError('original observed rows must be sorted and contain no self loops')
    neighbor_sets = [set(row) for row in neighbors]
    if any(i not in neighbor_sets[j] for i, row in enumerate(neighbors) for j in row):
        raise ValueError('observed graph requires explicit reverse directions')
    feature_manifest = manifest['feature_manifest']
    dimension = _positive_int(feature_manifest.get('effective_dimension'), 'effective_dimension')
    fit_entities = manifest['text_fit_entities']
    if (not isinstance(fit_entities, list) or any(not isinstance(node, str) for node in fit_entities)
            or len(set(fit_entities)) != len(fit_entities)
            or not set(fit_entities).issubset(nodes)
            or feature_manifest.get('nodes') != nodes
            or feature_manifest.get('train_entities') != fit_entities):
        raise ValueError('original text-fit entity/node identity mismatch')
    with np.load(root / 'features.npz', allow_pickle=False) as archive:
        features = archive['features']
    if (features.dtype != np.float32 or features.shape != (len(nodes), dimension)
            or not np.isfinite(features).all()
            or hashlib.sha256(features.tobytes()).hexdigest() != feature_manifest['feature_sha256']):
        raise ValueError('feature contract/hash mismatch')
    index = {node: i for i, node in enumerate(nodes)}
    if (type(queries.get('negatives_per_positive')) is not int
            or queries['negatives_per_positive'] != 4
            or not isinstance(queries.get('queries'), list)
            or not isinstance(queries.get('labels'), list)
            or len(queries['labels']) != expected['train_groups_count'] * 5
            or len(queries['queries']) != len(queries['labels'])):
        raise ValueError('requires registered groups of one positive and four negatives')
    indexed = []
    for pair in queries['queries']:
        if (not isinstance(pair, list) or len(pair) != 2
                or any(not isinstance(node, str) or node not in index for node in pair)):
            raise ValueError('training query contains invalid/unknown node IDs')
        indexed.append([index[pair[0]], index[pair[1]]])
    # Check labels before casting, so a nearby nonbinary value cannot round to 1.
    if any(type(label) not in (int, float) or label != (1 if i % 5 == 0 else 0)
           for i, label in enumerate(queries['labels'])):
        raise ValueError('query groups must start with exactly one positive')
    ids = torch.tensor(indexed, dtype=torch.long).reshape(-1, 5, 2)
    labels = torch.tensor(queries['labels'], dtype=torch.float32).reshape(-1, 5)
    fit = set(fit_entities)
    train_parents = defaultdict(set)
    for a, b in ids[:, 0].tolist():
        if a == b or nodes[b] not in fit or b not in neighbor_sets[a] or a not in neighbor_sets[b]:
            raise ValueError('train child or visible positive edge mismatch')
        train_parents[b].add(a)
    for group in ids.tolist():
        child = group[0][1]
        negatives = [a for a, _ in group[1:]]
        if (any(b != child for _, b in group) or len(set(negatives)) != 4
                or any(a == child or a in train_parents[child] for a in negatives)):
            raise ValueError('invalid fixed training negatives')
    # Without evaluator labels, visible graph equality can still be verified
    # entirely from train positives, including the required reverse edges.
    positive_graph = [set() for _ in nodes]
    for a, b in ids[:, 0].tolist():
        positive_graph[a].add(b)
        positive_graph[b].add(a)
    if positive_graph != neighbor_sets:
        raise ValueError('observed graph must equal train positives plus reverse')
    return {'manifest': manifest, 'manifest_hash': expected['manifest_hash'],
            'nodes': nodes, 'features': torch.from_numpy(features),
            'neighbors': neighbors, 'query_groups': ids, 'labels': labels,
            'valid_hash': expected['valid_queries_hash'],
            'input_raw_sha256': dict(expected['input_raw_sha256']),
            'input_files_opened': [*_JSON_INPUTS, 'features.npz']}


def _history_seed(history, seed):
    if not isinstance(history, dict) or history.get('status') != 'complete':
        raise ValueError('batch history must be a completed run')
    settings = history.get('config', history.get('training_settings'))
    if not isinstance(settings, dict) or type(settings.get('seed')) is not int or settings['seed'] != seed:
        raise ValueError('batch history configuration seed mismatch')
    if 'seed' in history and (type(history['seed']) is not int or history['seed'] != seed):
        raise ValueError('batch history top-level seed mismatch')
    if not isinstance(history.get('steps'), list):
        raise ValueError('batch history requires ordered steps')


def load_batch_history(path, expected_sha256, seed):
    """Verify historical file bytes and seed before returning its step records."""
    _sha256(expected_sha256, 'expected_history_sha256')
    if type(seed) is not int or not 0 <= seed < 2**63 - 1:
        raise ValueError('seed must permit original seed+1 CPU generator')
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('historical batch file raw hash mismatch')
    history = json.loads(raw.decode('utf-8'))
    _history_seed(history, seed)
    if '_raw_sha256' in history:
        raise ValueError('reserved batch history provenance field')
    history['_raw_sha256'] = expected_sha256
    return history


def replay_batches(group_count, seed, steps, batch_positives, history, expected_history_sha=None):
    """Replay every original randperm through max(steps), checking each hash.

    Return only requested batches as ``{one_based_step: CPU LongTensor}``.
    A completed history object or its ordered step list is accepted; the file
    SHA gate requires the object returned by ``load_batch_history``.
    """
    _positive_int(group_count, 'group_count')
    _positive_int(batch_positives, 'batch_positives')
    if batch_positives > group_count:
        raise ValueError('batch_positives exceeds the available positive groups')
    if type(seed) is not int or not 0 <= seed < 2**63 - 1:
        raise ValueError('seed must permit original seed+1 CPU generator')
    if (not isinstance(steps, (list, tuple)) or not steps
            or any(type(step) is not int or step < 1 for step in steps)
            or len(set(steps)) != len(steps)):
        raise ValueError('steps must be distinct positive one-based integers')
    if isinstance(history, dict):
        _history_seed(history, seed)
        records = history['steps']
        settings = history.get('config', history.get('training_settings'))
        if ('batch_positives' in settings and settings['batch_positives'] != batch_positives):
            raise ValueError('historical batch_positives mismatch')
    else:
        records = history
    if expected_history_sha is not None:
        _sha256(expected_history_sha, 'expected_history_sha256')
        if not isinstance(history, dict) or history.get('_raw_sha256') != expected_history_sha:
            raise ValueError('historical file SHA was not verified by load_batch_history')
    if not isinstance(records, list) or len(records) < max(steps):
        raise ValueError('historical batch stream is missing required steps')
    generator = torch.Generator(device='cpu').manual_seed(seed + 1)
    selected = {}
    wanted = set(steps)
    for step in range(1, max(steps) + 1):
        chosen = torch.randperm(group_count, generator=generator)[:batch_positives]
        record = records[step - 1]
        if not isinstance(record, dict) or type(record.get('step')) is not int or record['step'] != step:
            raise ValueError(f'historical batch step order mismatch at step {step}')
        if record.get('batch_group_ids_sha256') != digest(chosen.tolist()):
            raise ValueError(f'historical batch hash mismatch at step {step}')
        if step in wanted:
            selected[step] = chosen
    return selected


def _cpu_tensor_identity(tensor, name):
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu':
        raise ValueError(f'{name} immutable provenance requires a CPU tensor')
    value = tensor.detach().contiguous().numpy()
    return {'dtype': str(value.dtype), 'shape': list(value.shape),
            'sha256': hashlib.sha256(value.tobytes()).hexdigest()}


def immutable_input_hash(data):
    """Bind all train-only tensors, ordered graph and manifest identity."""
    return digest({'manifest': data['manifest'], 'manifest_hash': data['manifest_hash'],
                   'valid_hash': data['valid_hash'], 'nodes': data['nodes'],
                   'neighbors': data['neighbors'],
                   'features': _cpu_tensor_identity(data['features'], 'features'),
                   'query_groups': _cpu_tensor_identity(data['query_groups'], 'query_groups'),
                   'labels': _cpu_tensor_identity(data['labels'], 'labels'),
                   'input_raw_sha256': data['input_raw_sha256']})

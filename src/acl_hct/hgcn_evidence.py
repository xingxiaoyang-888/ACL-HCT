"""Checkpoint, ranking and panel identities shared by new validation drivers."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .development_view import load_development_view, make_panels
from .hgcn_panels import HierarchyPanel
from .hgcn_registration import canonical
from .hgcn_quality import atomic_json
from .text_capacity import filtered_parent_ranks


def deadline():
    if (int(os.environ.get('ACL_HGCN_VALIDATION_PID', '0')) != os.getppid()
            or time.monotonic() >= float(os.environ.get('ACL_HGCN_VALIDATION_DEADLINE', '0'))):
        raise ValueError('live bounded validation supervisor required')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def child_file(root, name):
    root = Path(root).resolve(); path = (root / name).resolve()
    if path.parent != root:
        raise ValueError('artifact path escapes owning directory')
    return path


def array_hash(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.ascontiguousarray(value)
    return canonical({'shape': list(value.shape), 'dtype': value.dtype.str,
                      'data_sha256': hashlib.sha256(value.tobytes()).hexdigest()})


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def state_hash(state):
    if not state or any(not torch.is_tensor(v) or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('finite complete tensor state required')
    return canonical({k: array_hash(v) for k, v in state.items()})


def checkpoint_binding(config, source_commit, seed, step):
    return {'protocol': config['protocol'], 'config_sha256': canonical(config),
            'source_commit': source_commit, 'upstream_commit': config['upstream_commit'],
            'manifest_hash': config['prepared']['manifest_hash'], 'seed': seed, 'step': step}


def save_checkpoint(output, name, model, binding, extra=None):
    path = child_file(output, name)
    if path.exists():
        raise ValueError('checkpoint exists; immutable step-specific name required')
    state = cpu_tree(model.state_dict())
    payload = {'schema': 1, 'binding': binding, 'model_state_sha256': state_hash(state),
               'model_state': state, 'extra': cpu_tree(extra or {})}
    with path.open('xb') as stream:
        torch.save(payload, stream)
    loaded = torch.load(path, map_location='cpu', weights_only=True)
    if loaded['binding'] != binding or state_hash(loaded['model_state']) != payload['model_state_sha256']:
        raise ValueError('serialized checkpoint roundtrip mismatch')
    return {'file': path.name, 'sha256': file_hash(path), 'model_state_sha256': payload['model_state_sha256'],
            'binding': binding}


def load_checkpoint(root, descriptor, binding):
    path = child_file(root, descriptor['file'])
    if descriptor['binding'] != binding or file_hash(path) != descriptor['sha256']:
        raise ValueError('reviewed checkpoint file/binding mismatch')
    value = torch.load(path, map_location='cpu', weights_only=True)
    if (value['schema'] != 1 or value['binding'] != binding
            or value['model_state_sha256'] != descriptor['model_state_sha256']
            or state_hash(value['model_state']) != descriptor['model_state_sha256']):
        raise ValueError('serialized checkpoint state mismatch')
    return value


def validate_ranking(ranking, queries, truth, n):
    rows = ranking['rows']; expected = set(map(tuple, queries))
    keys = [(r['parent'], r['child']) for r in rows]
    if (ranking['status'] != 'complete' or len(keys) != len(expected) or len(set(keys)) != len(keys)
            or set(keys) != expected or ranking['expected_queries'] != len(expected)
            or ranking['completed_queries'] != len(expected)):
        raise ValueError('complete unique query identity mismatch')
    children = defaultdict(list)
    for row in rows:
        p, c, rank, candidates = row['parent'], row['child'], row['rank'], row['candidates']
        if (type(p) is not int or type(c) is not int or type(candidates) is not int
                or candidates != n - len(truth[c]) or not np.isfinite(rank)
                or not 1 <= rank <= candidates or rank * 2 != round(rank * 2)):
            raise ValueError('all-candidate denominator or exact-average-tie rank mismatch')
        children[c].append(1 / rank)
    mrr = sum(1 / r['rank'] for r in rows) / len(rows)
    macro = sum(sum(v) / len(v) for v in children.values()) / len(children)
    if (ranking['completed_children'] != len(children)
            or abs(ranking['query_micro_mrr'] - mrr) > 1e-14
            or abs(ranking['child_macro_mrr'] - macro) > 1e-14
            or any(abs(ranking['hits'][str(k)] - sum(r['rank'] <= k for r in rows) / len(rows)) > 1e-14
                   for k in (1, 3, 10))):
        raise ValueError('saved retrieval aggregates do not reproduce from complete rank rows')
    return mrr


def complete_ranking(model, points, queries, truth, settings):
    deadline()
    ranking = filtered_parent_ranks(model, model.tangent(points), queries, truth,
                                    settings['candidate_chunk'], settings['max_seconds'])
    validate_ranking(ranking, queries, truth, len(points))
    return ranking


def compare_full(points, ranking, reference):
    # The upstream deterministic FP32 operation and cache are identical.
    native = points.detach().cpu().numpy() if torch.is_tensor(points) else points
    if not np.array_equal(native, reference['native_ball_points']):
        raise ValueError('full native point replay differs from matching best reference')
    actual = {(r['parent'], r['child']): (r['rank'], r['candidates']) for r in ranking['rows']}
    expected = {(r['parent'], r['child']): (r['rank'], r['candidates']) for r in reference['ranking']['rows']}
    if actual != expected:
        raise ValueError('full rank replay differs by query identity')


def load_panel_design(prepared_root, config):
    r = config['hierarchy']['registration']; root = Path(prepared_root)
    if file_hash(root / 'entity_split.json') != r['entity_split_raw_sha256']:
        raise ValueError('registered outcome-independent entity split raw identity mismatch')
    view = load_development_view(root); panels = make_panels(view, target=r['target'])
    panel = panels['panels'][r['panel_name']]
    if (panels['hash'] != r['combined_hash'] or panel['panel_hash'] != r['panel_hash']
            or panel['relation_hash'] != r['relation_hash'] or view.root != r['reference_root']
            or view.nodes.index(view.root) != r['reference_root_index']
            or view.metadata['h_dev_hash'] != r['h_dev_hash']
            or view.metadata['reachable_including_root'] != r['reachable_including_root']
            or view.metadata['entity_split_hash'] != r['entity_split_hash']):
        raise ValueError('fixed pre-effect hierarchy panel/root/relation identity mismatch')
    return view, panel


def hierarchy_evaluator(view, panel, full, config):
    evaluator = HierarchyPanel(view, panel, full, config['model']['c'], config['hierarchy']['near_zero_gap_floor'])
    baseline = evaluator.evaluate(full); registered = config['hierarchy']['registration']
    for kind in ('direct', 'distant'):
        coverage = registered['coverage'][kind]; actual = baseline[kind]
        if any(actual[k] != coverage[k] for k in ('covered_pairs', 'unknown_pairs', 'covered_children')):
            raise ValueError('fixed hierarchy coverage differs from data-only registration')
        if abs(actual['weighted_covered_child_mass'] - coverage['weighted_covered_child_mass_in_pool']) > 1e-12:
            raise ValueError('fixed hierarchy weight mass differs from data-only registration')
    if baseline['bias']['root_known_nodes'] != registered['selected_root_reachable_children']:
        raise ValueError('fixed node panel root-known count mismatch')
    return evaluator, baseline

"""Explicit numeric qualification of historical full-only HGCN replays."""
import hashlib
import json
from pathlib import Path

from .hgcn_registration import canonical

POLICY_PROTOCOL = 'MATURE-HGCN-numeric-replay-v1'
POLICY_SHA256 = 'a2eb9589b6e88fb20aa42ead081a5bbe9abfa294856e0f6c3d8bfce32c8d7d40'
POLICY_V2_PROTOCOL = 'MATURE-HGCN-numeric-replay-v2'
POLICY_V2_SHA256 = 'c07c5a6a5cb871ce896daea1e56e597f651b8da0bb8a97ae6ca6d43d182a0130'


def is_v2(policy):
    return policy.get('protocol') == POLICY_V2_PROTOCOL
ORIGINAL_SOURCE = 'c2fdfb64efaa25b9ba067baeba2e81db2e16f3ba'
ORIGINAL_CONFIG = '4a3b003a39b6bfcb74957e74a62a5a2b0329735f1dda0ae622172fbc205a3edc'
ORIGINAL_FAILURE = 'ValueError: full native point replay differs from matching best reference'


def validate_policy(policy, config):
    if ((canonical(policy), policy.get('protocol')) not in ((POLICY_SHA256, POLICY_PROTOCOL),
                                                           (POLICY_V2_SHA256, POLICY_V2_PROTOCOL))
            or canonical(config) != ORIGINAL_CONFIG or policy['original_source_commit'] != ORIGINAL_SOURCE):
        raise ValueError('exact separately frozen numerical replay policy required')
    root = Path(__file__).parent
    for name, digest in policy['immutable_math_source_sha256_normalized_lf'].items():
        actual = hashlib.sha256((root / Path(name).name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if actual != digest:
            raise ValueError('original hierarchy/geometry definition changed')
    return canonical(policy)


def load_policy(path, config):
    policy = json.loads(Path(path).read_bytes())
    validate_policy(policy, config)
    return policy


def native_array(value):
    import numpy as np
    import torch
    value = value.detach().cpu().numpy() if torch.is_tensor(value) else value
    if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError('finite unmodified native FP32 points required')
    # Strict interior check; never clip, reproject or change official epsilon.
    if np.any(np.sum(value.astype(np.float64) ** 2, axis=-1) >= 1):
        raise ValueError('native Ball point outside strict interior')
    return value


def reference_identity(reference, actual_state_sha256, expected_binding, full_plan):
    import numpy as np
    import torch
    if (reference['model_state_sha256'] != actual_state_sha256 or reference['binding'] != expected_binding
            or expected_binding['source_commit'] != ORIGINAL_SOURCE):
        raise ValueError('original serialized weight/config/seed/step identity mismatch')
    expected = reference['full_plan']
    if set(full_plan) != set(expected):
        raise ValueError('full adjacency archive inventory mismatch')
    for key, value in full_plan.items():
        value = value.detach().cpu().numpy() if torch.is_tensor(value) else value
        if isinstance(expected[key], np.ndarray):
            if (not isinstance(value, np.ndarray) or value.dtype != expected[key].dtype
                    or value.shape != expected[key].shape or not np.array_equal(value, expected[key])):
                raise ValueError('full adjacency identity mismatch: ' + key)
        elif value != expected[key]:
            raise ValueError('full adjacency identity mismatch: ' + key)


def qualify_full(points, ranking, reference, evaluator, config, policy, queries, truth,
                 actual_state_sha256, expected_binding, full_plan):
    """Return all measured qualifications; numerical failures are archived first."""
    import numpy as np
    from .hgcn_evidence import validate_ranking
    from .hgcn_geometry import distance_ball_fp64
    reference_identity(reference, actual_state_sha256, expected_binding, full_plan)
    native = native_array(points); fixed = native_array(reference['native_ball_points'])
    if native.shape != fixed.shape or fixed.shape != (config['prepared']['nodes_count'], config['model']['hidden']):
        raise ValueError('complete original node order/dimension required')
    validate_ranking(ranking, queries, truth, len(fixed))
    validate_ranking(reference['ranking'], queries, truth, len(fixed))
    for saved in (ranking, reference['ranking']):
        if any(not np.isfinite(v) for v in [saved['query_micro_mrr'], saved['child_macro_mrr'], *saved['hits'].values()]):
            raise ValueError('finite complete ranking aggregates required')
    if not np.array_equal(evaluator.full, fixed.astype(np.float64)):
        raise ValueError('hierarchy evaluator must retain the original fixed F basis')
    actual = {(r['parent'], r['child']): (r['rank'], r['candidates']) for r in ranking['rows']}
    original = {(r['parent'], r['child']): (r['rank'], r['candidates']) for r in reference['ranking']['rows']}
    if set(actual) != set(original) or any(actual[k][1] != original[k][1] for k in actual):
        raise ValueError('complete query/candidate identity mismatch')
    delta = native.astype(np.float64) - fixed.astype(np.float64)
    rank_delta = [abs(actual[k][0] - original[k][0]) for k in actual]
    base = evaluator.evaluate(fixed); replay = evaluator.evaluate(native)
    measured = {'point_max_abs': float(np.max(np.abs(delta))), 'point_RMS': float(np.sqrt(np.mean(delta * delta))),
                'max_geodesic': float(np.max(distance_ball_fp64(fixed, native, config['model']['c']))),
                'rank_changed_queries': sum(v != 0 for v in rank_delta), 'rank_max_abs_delta': max(rank_delta),
                'micro_mrr_abs_delta': abs(ranking['query_micro_mrr'] - reference['ranking']['query_micro_mrr']),
                'macro_mrr_abs_delta': abs(ranking['child_macro_mrr'] - reference['ranking']['child_macro_mrr']),
                'hits_abs_delta': max(abs(ranking['hits'][k] - reference['ranking']['hits'][k]) for k in ('1', '3', '10')),
                'hierarchy': {}}
    limits = policy['limits']; violations = []
    def bound(name, value, limit):
        if not np.isfinite(value) or value > limit:
            violations.append({'metric': name, 'observed': value, 'limit': limit})
    for key, value in measured.items():
        if key != 'hierarchy': bound(key, value, limits[key])
    for kind in ('direct', 'distant'):
        a = replay[kind]; b = base[kind]
        for key in ('covered_children', 'selected_children', 'covered_pairs', 'unknown_pairs',
                    'weighted_covered_child_mass', 'weighted_selected_child_mass'):
            if a[key] != b[key]: raise ValueError('fixed hierarchy coverage/weight identity mismatch')
        if not len(a['gap']) or a['metrics']['score'] is None:
            registered = config.get('hierarchy', {}).get('registration', {}).get('coverage', {}).get(kind, {})
            if registered.get('covered_pairs') != 0:
                raise ValueError('registered nonempty hierarchy coverage required')
            measured['hierarchy'][kind] = {'status': 'unknown_no_registered_covered_pairs', 'covered_pairs': 0}
            continue  # Tiny QA stars have no distant ancestors; never fabricate a zero score.
        values = {'pair_gap_max_abs_delta': float(np.max(np.abs(a['gap'] - b['gap']))),
                  'weighted_gap_abs_delta': abs(a['metrics']['gap'] - b['metrics']['gap']),
                  'order_abs_delta': abs(a['metrics']['score'] - b['metrics']['score']),
                  'pair_sign_changes': int(np.count_nonzero(a['score'] != b['score'])),
                  'tie_or_unresolved_abs_delta': max(abs(a['metrics'][k] - b['metrics'][k]) for k in ('tie', 'unresolved'))}
        measured['hierarchy'][kind] = values
        for name in ('pair_gap_max_abs_delta', 'weighted_gap_abs_delta'):
            bound(kind + '/' + name, values[name], limits['hierarchy'][kind][name])
        for name in ('order_abs_delta', 'pair_sign_changes', 'tie_or_unresolved_abs_delta'):
            bound(kind + '/' + name, values[name], limits['hierarchy_' + name])
    result = {'protocol': POLICY_V2_PROTOCOL if is_v2(policy) else POLICY_PROTOCOL,
              'policy_sha256': canonical(policy), 'accepted': not violations,
              'measured': measured, 'limits': limits, 'violations': violations,
              'original_bit_exact': {'native_points': bool(np.array_equal(native, fixed)), 'rank_rows': actual == original},
              'fixed_reference': 'original matching-best points and ranks; no replacement by this replay',
              'limits_are_guaranteed_error_bounds': False, 'original_failure_preserved': True}
    if is_v2(policy):
        report_only = ('micro_mrr_abs_delta', 'macro_mrr_abs_delta')
        old_violations = violations
        result['violations'] = [v for v in old_violations if v['metric'] not in report_only]
        result['accepted'] = not result['violations']
        counts = {}
        for parent, child in actual:
            counts[child] = counts.get(child, 0) + 1
        contributions = []
        for parent, child in sorted(actual):
            old_rank = original[parent, child][0]; new_rank = actual[parent, child][0]
            if new_rank != old_rank:
                reciprocal = 1 / new_rank - 1 / old_rank
                contributions.append({'parent': parent, 'child': child, 'F_rank': old_rank,
                                      'full_rank': new_rank, 'micro_mrr': reciprocal / len(actual),
                                      'macro_mrr': reciprocal / (len(counts) * counts[child])})
        result['v1_diagnostic'] = {
            'v1_policy_sha256': POLICY_SHA256, 'v1_accepted': not old_violations,
            'v1_violations': old_violations,
            'signed_micro_mrr_drift': ranking['query_micro_mrr'] - reference['ranking']['query_micro_mrr'],
            'signed_macro_mrr_drift': ranking['child_macro_mrr'] - reference['ranking']['child_macro_mrr'],
            'changed_query_contributions': contributions,
            'unchanged_queries_have_zero_contribution': True,
            'flat_MRR_limits_are_report_only_in_v2': True}
    return result


def require_qualified(qualification):
    if qualification['accepted'] is not True:
        raise ValueError('numerical full replay qualification failed: ' + ', '.join(v['metric'] for v in qualification['violations']))


def qualification_fixture(output, device, policy):
    """Contrived geometry QA; this is not a trained/checkpoint baseline."""
    from types import SimpleNamespace
    import numpy as np
    import torch
    from .diagnostic_archive import write_archive
    from .hgcn_panels import HierarchyPanel
    points = np.array([[.05, 0.], [.2, 0.], [.35, 0.], [.3, 0.]], dtype=np.float32)
    view = SimpleNamespace(nodes=['r', 'p', 'c', 'u'], root='r', reachable={'r', 'p', 'c'})
    panel = {'pool_size': 2, 'rows': [{'id': n, 'index': i, 'inclusion_probability': 1., 'pool_mean_weight': .5}
                                    for n, i in [('c', 2), ('u', 3)]],
             'relations': [{'child': n, 'direct_parents': ['p'], 'positive_distant_ancestors': ['r']} for n in ('c', 'u')]}
    evaluator = HierarchyPanel(view, panel, points); queries = [(1, 2), (0, 3)]; truth = {2: {1}, 3: {0}}
    rank = {'status': 'complete', 'expected_queries': 2, 'completed_queries': 2, 'completed_children': 2,
            'query_micro_mrr': (1/2 + 1/3)/2, 'child_macro_mrr': (1/2 + 1/3)/2,
            'hits': {'1': 0., '3': 1., '10': 1.},
            'rows': [{'parent': 1, 'child': 2, 'rank': 2., 'candidates': 3}, {'parent': 0, 'child': 3, 'rank': 3., 'candidates': 3}]}
    binding = {'source_commit': ORIGINAL_SOURCE, 'scope': 'contrived metadata only; not a checkpoint'}
    plan = {'indices': np.array([[0], [0]], dtype=np.int64), 'weights_fp64': np.ones(1, dtype=np.float64), 'fanout': None}
    reference = {'native_ball_points': points, 'ranking': rank, 'binding': binding,
                 'model_state_sha256': 'a' * 64, 'full_plan': plan}
    miniature = {'model': {'hidden': 2, 'c': 1.}, 'prepared': {'nodes_count': 4}}
    accepted_points = points.copy(); accepted_points[2, 0] += np.float32(1e-7)
    rejected_points = points.copy(); rejected_points[2, 0] += np.float32(1e-4)
    results = [qualify_full(torch.from_numpy(p).to(device), rank, reference, evaluator, miniature, policy,
                            queries, truth, 'a' * 64, binding, plan) for p in (accepted_points, rejected_points)]
    if results[0]['accepted'] is not True or results[1]['accepted'] is not False:
        raise ValueError('separate numerical qualification fixture failed')
    archive = write_archive(Path(output), 'science-numeric-replay-qualification-fixture',
                            {'scope': 'contrived native geometry only; no network/checkpoint premise',
                             'reference': reference, 'accepted_points': accepted_points, 'rejected_points': rejected_points,
                             'qualifications': results, 'policy_sha256': canonical(policy)})
    return {'status': 'passed', 'scope': 'contrived geometry qualification, no scientific baseline or sampled effect',
            'policy_sha256': canonical(policy), 'archive': archive}


def archive_descriptor(root, name):
    from .hgcn_evidence import file_hash
    path = Path(root) / 'arrays' / (name + '.json'); record = json.loads(path.read_bytes())
    return {'manifest': path.name, 'manifest_sha256': file_hash(path), 'array_file': record['array_file'],
            'array_file_sha256': record['array_file_sha256'], 'array_bytes': record['array_bytes'], 'arrays': len(record['arrays'])}


def historical_training(training_run, config, policy, seed):
    """Recover metadata from exact preserved artifacts, without resuming training."""
    import torch
    from .diagnostic_archive import read_archive
    from .hgcn_evidence import file_hash, load_checkpoint
    path = Path(training_run); root = path.parent; origin = policy['origins'][str(seed)]
    targets = {'training_run_sha256': path, 'training_progress_sha256': root / 'training-progress.json',
               'batch_history_manifest_sha256': root / 'arrays/training-history.json',
               'initial_reference_manifest_sha256': root / 'arrays/initial-full-reference.json',
               'last_checkpoint_sha256': root / 'last.pt'}
    if any(file_hash(p) != origin[k] for k, p in targets.items()):
        raise ValueError('preserved original training artifact identity mismatch')
    run = json.loads(path.read_bytes()); progress = json.loads((root / 'training-progress.json').read_bytes())
    if (run['status'] != 'failed' or run['phase'] != 'train' or run.get('error') != ORIGINAL_FAILURE
            or run['source_commit'] != ORIGINAL_SOURCE or run['config_sha256'] != canonical(config)
            or progress['seed'] != seed or progress['completed_steps'] != config['training']['steps']
            or progress['best']['step'] != origin['best_step']
            or progress['best']['checkpoint']['sha256'] != origin['best_checkpoint_sha256']
            or progress['best']['reference']['manifest_sha256'] != origin['best_reference_manifest_sha256']):
        raise ValueError('exact original fully trained failed replay lineage required')
    initial = archive_descriptor(root, 'initial-full-reference'); initial_data = read_archive(root / 'arrays', initial)
    last = torch.load(root / 'last.pt', map_location='cpu', weights_only=True)
    last_descriptor = {'file': 'last.pt', 'sha256': origin['last_checkpoint_sha256'],
                       'model_state_sha256': last['model_state_sha256'], 'binding': last['binding']}
    load_checkpoint(root, last_descriptor, {**progress['best']['checkpoint']['binding'], 'step': config['training']['steps']})
    trained = {**progress, 'initial_state_sha256': initial_data['model_state_sha256'], 'initial_reference': initial,
               'batch_history': archive_descriptor(root, 'training-history'), 'last': last_descriptor,
               'original_training': {'run_file': str(path.resolve()), **origin}, 'original_failure_preserved': True}
    return root, trained

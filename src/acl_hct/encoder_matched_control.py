"""Bounded matched-encoder E2 control; scientific execution uses the entry module."""
import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .backbone import LorentzMeanNetwork
from .backbone_frozen_inputs import load_train_prepared, load_batch_history
from .encoder_matched import SelfMessageLorentzNetwork
from .encoder_matched_registration import (PROTOCOL, CONFIG_SHA256, canonical, file_sha256,
                                           source_hashes, source_identity, validate_config, verify_release)
from .ranking import filtered_parent_ranks, prepare_scores, score_cached


@dataclass(frozen=True)
class Settings:
    seed: int = 11
    input_dim: int = 128
    hidden: int = 128
    head_hidden: int = 128
    c: float = 1.
    scaled_radius: float = 1.2
    batch_positives: int = 128
    max_steps: int = 1024
    learning_rate: float = .003
    evaluate_every: int = 256
    evaluation_max_seconds: float = 180.
    candidate_chunk: int = 4096
    save_every: int = 50
    threads: int = 2
    max_seconds: float = 840.

    def validate(self):
        if type(self.seed) is not int or self.seed not in (11, 23):
            raise ValueError('registered seed required')
        limits = dict(input_dim=128, hidden=128, head_hidden=128, batch_positives=128,
                      max_steps=1024, evaluate_every=256, candidate_chunk=4096, save_every=50, threads=2)
        for name, limit in limits.items():
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= limit:
                raise ValueError('invalid bounded ' + name)
        if self.max_steps % self.evaluate_every:
            raise ValueError('last step must be a scheduled full validation')
        for name in ('c', 'scaled_radius', 'learning_rate', 'max_seconds', 'evaluation_max_seconds'):
            v = getattr(self, name)
            if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
                raise ValueError('finite positive ' + name + ' required')
        if self.scaled_radius > 2.5 or self.max_seconds > 840 or self.evaluation_max_seconds > min(180, self.max_seconds):
            raise ValueError('bounded total/evaluation deadline required')


def settings_from_config(config, seed):
    validate_config(config)
    if type(seed) is not int or seed not in config['seeds']:
        raise ValueError('unregistered seed')
    training = config['training']
    names = ('batch_positives', 'max_steps', 'learning_rate', 'evaluate_every',
             'evaluation_max_seconds', 'candidate_chunk', 'save_every', 'threads')
    return Settings(seed=seed, **{k: config['model'][k] for k in ('input_dim', 'hidden', 'head_hidden', 'c', 'scaled_radius')},
                    **{k: training[k] for k in names}, max_seconds=config['resources']['science_worker_seconds'])


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    temporary.replace(path)


def weights_hashes(model):
    return {k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for k, v in model.state_dict().items()}


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def load_prepared(root, config):
    """Exactly train inputs plus original valid; never opens test/full truth."""
    if file_sha256(Path(root) / 'features.npz') != config['prepared']['features_npz_raw_sha256']:
        raise ValueError('original feature NPZ raw hash mismatch')
    data = load_train_prepared(root, config['prepared'])
    path = Path(root) / 'evaluator_valid.json'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != config['prepared']['evaluator_valid_raw_sha256']:
        raise ValueError('original valid raw hash mismatch')
    external = json.loads(raw.decode('utf-8'))
    if (not isinstance(external, list) or len(external) != config['valid_query_count']
            or canonical(external) != config['prepared']['valid_queries_hash']):
        raise ValueError('original complete valid canonical identity/count mismatch')
    index = {n: i for i, n in enumerate(data['nodes'])}
    valid = []
    fit = set(data['manifest']['text_fit_entities'])
    neighbors = [set(row) for row in data['neighbors']]
    for pair in external:
        if (not isinstance(pair, list) or len(pair) != 2
                or any(not isinstance(n, str) or n not in index for n in pair)):
            raise ValueError('invalid validation entity identity')
        a, b = map(index.__getitem__, pair)
        if a == b or pair[1] in fit or b in neighbors[a] or a in neighbors[b]:
            raise ValueError('validation query leaks into train graph/text fit')
        valid.append((a, b))
    if len(set(valid)) != len(valid) or data['features'].shape[1] != config['model']['input_dim']:
        raise ValueError('duplicate validation pairs or wrong original feature width')
    data.update(valid=valid, valid_raw_sha256=hashlib.sha256(raw).hexdigest())
    data['input_files_opened'].append('evaluator_valid.json')
    return data


def validate_ranking(row, data):
    truth = defaultdict(set)
    for a, b in data['valid']:
        truth[b].add(a)
    rows = row['rows']
    pairs = [(r['parent'], r['child']) for r in rows]
    if (row['status'] != 'complete' or row['metric_scope'] != 'filtered_all_entity_candidates'
            or len(rows) != len(data['valid']) or len(set(pairs)) != len(rows)
            or set(pairs) != set(data['valid']) or row['completed_queries'] != len(rows)
            or row['expected_queries'] != len(rows) or row['completed_children'] != len(truth)):
        raise ValueError('complete original full valid coverage required')
    rr = defaultdict(list)
    for r in rows:
        candidates = len(data['nodes']) - len(truth[r['child']])
        rank = r['rank']
        if (type(rank) not in (int, float) or not math.isfinite(rank)
                or not 1 <= rank <= candidates or r['candidates'] != candidates):
            raise ValueError('invalid saved rank/candidate count')
        rr[r['child']].append(1 / rank)
    expected = {'query_micro_mrr': sum(1 / r['rank'] for r in rows) / len(rows),
                'child_macro_mrr': sum(sum(v) / len(v) for v in rr.values()) / len(rr)}
    for k, v in expected.items():
        if not math.isfinite(row[k]) or abs(row[k] - v) > 1e-12:
            raise ValueError('ranking summary differs from complete saved ranks')
    for k in ('1', '3', '10'):
        expected_hit = sum(r['rank'] <= int(k) for r in rows) / len(rows)
        if not math.isfinite(row['hits'][k]) or abs(row['hits'][k] - expected_hit) > 1e-12:
            raise ValueError('Hits summary differs from saved ranks')


def degree_readings(row, data):
    """One fixed parent grouping, with original all-query denominator."""
    grouped = {k: [] for k in ('0', '1', '2-16', '>16')}
    for r in row['rows']:
        degree = len(data['neighbors'][r['parent']])
        key = '0' if degree == 0 else '1' if degree == 1 else '2-16' if degree <= 16 else '>16'
        grouped[key].append(r)
    n = len(row['rows'])
    return {k: {'query_count': len(v), 'unique_parent_count': len({r['parent'] for r in v}),
                'query_micro_mrr': sum(1 / r['rank'] for r in v) / len(v) if v else None,
                'contribution_to_overall_micro_mrr': sum(1 / r['rank'] for r in v) / n if v else None,
                'hits10': sum(r['rank'] <= 10 for r in v) / len(v) if v else None,
                'contribution_to_overall_hits10': sum(r['rank'] <= 10 for r in v) / n if v else None,
                'overall_query_denominator': n} for k, v in grouped.items()}


def verify_references(gnn_path, mlp_path, config, seed, data):
    anchor = config['references'][str(seed)]
    if file_sha256(gnn_path) != anchor['gnn']['baseline_report_sha256']:
        raise ValueError('original GNN raw report hash mismatch')
    gnn = json.loads(Path(gnn_path).read_text(encoding='utf-8'))
    mlp = load_batch_history(mlp_path, anchor['mlp']['report_sha256'], seed)
    if (gnn['status'] != 'step_limit_reached' or gnn['completed_steps'] != 1024
            or gnn['config'] != anchor['gnn']['training_config']
            or gnn['source']['source_commit'] != anchor['gnn']['training_commit']
            or mlp['completed_steps'] != 1024 or mlp['training_settings'] != anchor['mlp']['training_settings']
            or mlp['source']['source_commit'] != anchor['mlp']['source_commit']):
        raise ValueError('original reference training/source identity mismatch')
    for record in (gnn, mlp):
        if (record['manifest_hash'] != data['manifest_hash'] or record['validation_queries_hash'] != data['valid_hash']
                or [r['step'] for r in record['evaluations']] != config['training']['evaluation_steps']):
            raise ValueError('reference data identity or four validation opportunities mismatch')
        for row in record['evaluations']:
            if row['purpose'] != 'full_validation':
                raise ValueError('reference requires full validation')
            validate_ranking(row, data)
        winner = max(record['evaluations'], key=lambda r: r['query_micro_mrr'])
        if winner['query_micro_mrr'] != record['best_full_valid_mrr']:
            raise ValueError('reference best summary mismatch')
    if (max(gnn['evaluations'], key=lambda r: r['query_micro_mrr'])['step'] != anchor['gnn']['step']
            or max(mlp['evaluations'], key=lambda r: r['query_micro_mrr'])['step'] != anchor['mlp']['best_step']
            or mlp['best_step'] != anchor['mlp']['best_step']
            or len(mlp['steps']) != 1024 or [r['step'] for r in mlp['steps']] != list(range(1, 1025))):
        raise ValueError('reference first-best selection or full batch history mismatch')
    return gnn, mlp


class BatchStream:
    def __init__(self, group_count, seed, batch_positives, history):
        if (type(group_count) is not int or type(batch_positives) is not int
                or not 1 <= batch_positives <= group_count or type(seed) is not int
                or history.get('seed') != seed or history['training_settings']['seed'] != seed
                or history['training_settings']['batch_positives'] != batch_positives):
            raise ValueError('original batch stream identity/size required')
        self.group_count, self.batch_positives = group_count, batch_positives
        self.rows, self.completed = history['steps'], 0
        self.rng = torch.Generator(device='cpu').manual_seed(seed + 1)

    def next(self, step):
        if step != self.completed + 1 or step > len(self.rows):
            raise ValueError('batch stream step out of order/missing history')
        row = self.rows[step - 1]
        chosen = torch.randperm(self.group_count, generator=self.rng)[:self.batch_positives]
        if (row.get('step') != step or row.get('positive_queries') != len(chosen)
                or row.get('negative_queries') != 4 * len(chosen)
                or canonical(chosen.tolist()) != row.get('batch_group_ids_sha256')):
            raise ValueError('original batch hash/count mismatch at step ' + str(step))
        self.completed = step
        return chosen


def require_runtime(config, device):
    if str(torch.__version__) != config['runtime']['torch']:
        raise ValueError('registered original PyTorch runtime required')
    if device.type == 'cuda':
        if (device.index not in (None, 0) or not os.environ.get('SLURM_JOB_ID')
                or not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count() != 1
                or torch.cuda.get_device_name(0) != config['runtime']['hardware']):
            raise ValueError('one visible allocated L40 with preserved Slurm binding required')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')


def construct(settings):
    torch.manual_seed(settings.seed)
    return SelfMessageLorentzNetwork(settings.input_dim, settings.hidden, settings.c,
                                    settings.scaled_radius, settings.head_hidden)


def verify_initial(model, expected):
    if weights_hashes(model) != expected:
        raise ValueError('initial tensors differ from registered historical corresponding hashes')


def tensor_identity(data):
    return {name: hashlib.sha256(data[name].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name in ('features', 'query_groups', 'labels')}


def _comparison(a, b, atol, rtol):
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {'status': 'passed' if finite and torch.allclose(a, b, atol=atol, rtol=rtol) else 'failed',
            'maximum_absolute_difference': float((a - b).abs().max()) if finite else None,
            'finite': finite, 'atol': atol, 'rtol': rtol}


def equivalence_fixture(config, seed, device='cpu', *, dtype=torch.float32,
                        historical_initialization=False, atol=None, rtol=None, archive_root=None):
    """Synthetic forward/gradient evidence only; never reads real data/weights."""
    d = torch.device(device); cfg = config['model']; tol = config['equivalence']
    atol = tol['atol'] if atol is None else atol; rtol = tol['rtol'] if rtol is None else rtol
    args = (cfg['input_dim'], cfg['hidden'], cfg['c'], cfg['scaled_radius'], cfg['head_hidden'])
    models = []
    states = []
    for cls in (LorentzMeanNetwork, SelfMessageLorentzNetwork, SelfMessageLorentzNetwork):
        torch.manual_seed(seed); m = cls(*args)
        states.append(torch.get_rng_state().clone())
        if historical_initialization:
            verify_initial(m, config['references'][str(seed)]['initial_weights_sha256'])
        models.append(m.to(device=d, dtype=dtype))
    if not torch.equal(states[0], states[1]) or not torch.equal(states[0], states[2]):
        raise ValueError('constructor RNG drift')
    if any(weights_hashes(m) != weights_hashes(models[0]) for m in models[1:]):
        raise ValueError('constructor tensor/name drift')
    features = (torch.randn(40, cfg['input_dim'], generator=torch.Generator().manual_seed(2026091709)) * .1).to(d, dtype)
    queries = torch.tensor([[[0, child]] + [[p, child] for p in (1, 2, 3, 4)]
                            for child in range(6, 16)], dtype=torch.long, device=d).reshape(-1, 2)
    labels = torch.tensor([1., 0., 0., 0., 0.], device=d, dtype=dtype).repeat(10)
    empty = [[] for _ in range(40)]
    points, _, _ = models[0].encode(features, empty, [empty, empty])
    full = models[1].encode(features)
    ids = torch.unique(queries.reshape(-1), sorted=True)
    unique_points = models[2].encode(features[ids])
    logits = [models[0].score(points, queries), models[1].score(full, queries), models[2](features, queries)]
    losses = [torch.nn.functional.binary_cross_entropy_with_logits(x, labels) for x in logits]
    for loss in losses:
        loss.backward()
    checks = {'empty_vs_all_points': _comparison(points, full, atol, rtol),
              'empty_vs_unique_points': _comparison(points[ids], unique_points, atol, rtol)}
    for i, name in [(1, 'all'), (2, 'unique')]:
        checks['empty_vs_' + name + '_logits'] = _comparison(logits[0], logits[i], atol, rtol)
        checks['empty_vs_' + name + '_mean_bce'] = _comparison(losses[0], losses[i], atol, rtol)
        for (pa, a), (pb, b) in zip(models[0].named_parameters(), models[i].named_parameters()):
            if pa != pb or a.grad is None or b.grad is None:
                raise ValueError('missing/mismatched fixture parameter gradient')
            checks['empty_vs_' + name + '_gradient/' + pa] = _comparison(a.grad, b.grad, atol, rtol)
    models[1].eval()
    with torch.no_grad():
        cache = prepare_scores(models[1], full.detach())
        pairs = torch.cartesian_prod(torch.arange(40, device=d), torch.arange(40, device=d))
        direct = models[1].score(full.detach(), pairs)
        cached = score_cached(models[1], cache, pairs[:, 0], pairs[:, 1])
        checks['cached_scores'] = _comparison(direct, cached, tol['cache_absolute_tolerance'], 0.)
    models[0].eval()
    truth = {20: {0, 1}, 21: {2}}
    ranking_queries = [(0, 20), (1, 20), (2, 21)]
    ranks = filtered_parent_ranks(models[1], full.detach(), ranking_queries, truth, candidate_chunk=11)
    original_ranks = filtered_parent_ranks(models[0], points.detach(), ranking_queries, truth, candidate_chunk=11)
    if ranks['rows'] != original_ranks['rows']:
        raise ValueError('synthetic cached ranking differs from original empty path')
    passed = all(r['status'] == 'passed' for r in checks.values())
    result = {'seed': seed, 'status': 'passed' if passed else 'failed', 'checks': checks,
            'parameter_count': sum(p.numel() for p in models[0].parameters()),
            'same_constructor_rng': True, 'historical_initialization_checked': historical_initialization,
            'nodes': 40, 'query_records': 50, 'fixture_only': True}
    if archive_root is not None:
        root = Path(archive_root); root.mkdir(parents=True, exist_ok=False)
        ids, inverse = torch.unique(queries.reshape(-1), sorted=True, return_inverse=True)
        arrays = {'features': features, 'queries': queries, 'labels': labels, 'unique_ids': ids,
                  'inverse_queries': inverse.reshape(-1, 2), 'points_original': points,
                  'points_all': full, 'points_unique': unique_points, 'cache_pairs': pairs,
                  'cached_scores': cached, 'direct_scores': direct}
        for i, name in enumerate(('original', 'all', 'unique')):
            arrays['logits_' + name] = logits[i]; arrays['loss_' + name] = losses[i]
            for key, parameter in models[i].named_parameters():
                arrays['weights_' + name + '/' + key] = parameter
                arrays['gradients_' + name + '/' + key] = parameter.grad
        raw = {k: v.detach().cpu().contiguous().numpy() for k, v in arrays.items()}
        path = root / 'fixture.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, **raw)
        manifest = {'schema': 'E2-matched-encoder-fixture-arrays-v1', 'seed': seed, 'fixture_only': True,
                    'array_file': path.name, 'array_file_sha256': file_sha256(path),
                    'arrays': {k: {'dtype': str(v.dtype), 'shape': list(v.shape),
                                   'sha256': hashlib.sha256(v.tobytes()).hexdigest()} for k, v in raw.items()},
                    'ranking_queries': ranking_queries, 'ranking_truth': {str(k): sorted(v) for k, v in truth.items()},
                    'cached_ranking': ranks, 'original_cached_ranking': original_ranks, 'checks': checks}
        atomic_json(root / 'fixture.json', manifest)
        result['archive'] = {'directory': root.name, 'manifest': 'fixture.json',
                             'manifest_sha256': file_sha256(root / 'fixture.json'),
                             'array_file': path.name, 'array_file_sha256': file_sha256(path), 'arrays': len(raw)}
    return result


def cpu_preflight(config, roots, reference_paths, identity, evidence):
    require_runtime(config, torch.device('cpu')); torch.set_num_threads(config['training']['threads'])
    results = []
    for seed in config['seeds']:
        data = load_prepared(roots[seed], config)
        _, history = verify_references(*reference_paths[seed], config, seed, data)
        settings = settings_from_config(config, seed)
        model = construct(settings); verify_initial(model, config['references'][str(seed)]['initial_weights_sha256'])
        stream = BatchStream(len(data['query_groups']), seed, settings.batch_positives, history)
        for step in range(1, 1025):
            stream.next(step)
        results.append({'seed': seed, 'batch_hashes_verified': 1024, 'initial_weights_sha256': weights_hashes(model),
                        'input_identity': tensor_identity(data), 'valid_hash': data['valid_hash'],
                        'manifest_hash': data['manifest_hash'], 'nodes_count': len(data['nodes']),
                        'train_groups_count': len(data['query_groups']), 'valid_query_count': len(data['valid']),
                        'gnn_report_sha256': file_sha256(reference_paths[seed][0]),
                        'mlp_report_sha256': file_sha256(reference_paths[seed][1]),
                        'input_files_opened': data['input_files_opened']})
    return {'status': 'complete', 'cpu_preflight_passed': True, 'protocol': PROTOCOL,
            'config_sha256': CONFIG_SHA256, 'source': identity, 'source_lf_sha256': source_hashes(),
            'seeds': results, 'model_forward_calls': 0, 'model_backward_calls': 0,
            'optimizer_created': False, 'torch': str(torch.__version__), **evidence}


def verify_gate(path, expected_hash, kind, config, identity):
    if file_sha256(path) != expected_hash:
        raise ValueError(kind + ' raw SHA mismatch')
    row = json.loads(Path(path).read_text(encoding='utf-8'))
    supervisor = json.loads((Path(path).parent / 'supervisor.json').read_text(encoding='utf-8'))
    if (supervisor.get('status') != 'complete' or supervisor.get('worker_exit_code') != 0
            or supervisor.get('deadline_seconds') != config['resources']['fixture_worker_seconds']):
        raise ValueError('completed supervised gate required')
    flag = 'cpu_preflight_passed' if kind == 'cpu_preflight' else 'matched_encoder_fixture_passed'
    if (row.get('status') != 'complete' or row.get(flag) is not True or row.get('protocol') != PROTOCOL
            or row.get('config_sha256') != CONFIG_SHA256 or row.get('source', {}).get('source_commit') != identity['source_commit']
            or row.get('source_lf_sha256') != source_hashes() or row.get('torch') != config['runtime']['torch']
            or [r['seed'] for r in row.get('seeds', [])] != config['seeds']):
        raise ValueError('exact source/config/runtime ' + kind + ' evidence required')
    if kind == 'cpu_preflight':
        if (row.get('model_forward_calls') != 0 or row.get('model_backward_calls') != 0
                or row.get('optimizer_created') is not False
                or any(r.get('batch_hashes_verified') != 1024 for r in row['seeds'])):
            raise ValueError('complete no-forward CPU preflight required')
        for seed in row['seeds']:
            anchor = config['references'][str(seed['seed'])]
            if (seed.get('initial_weights_sha256') != anchor['initial_weights_sha256']
                    or seed.get('gnn_report_sha256') != anchor['gnn']['baseline_report_sha256']
                    or seed.get('mlp_report_sha256') != anchor['mlp']['report_sha256']
                    or seed.get('nodes_count') != config['prepared']['nodes_count']
                    or seed.get('train_groups_count') != config['prepared']['train_groups_count']
                    or seed.get('valid_query_count') != config['valid_query_count']
                    or seed.get('manifest_hash') != config['prepared']['manifest_hash']
                    or seed.get('valid_hash') != config['prepared']['valid_queries_hash']
                    or seed.get('input_files_opened') != ['input_manifest.json', 'observed_graph.json', 'train_queries.json', 'features.npz', 'evaluator_valid.json']):
                raise ValueError('CPU preflight input/reference/initialization evidence mismatch')
    else:
        if row.get('device_name') != config['runtime']['hardware'] or row.get('dtype') != 'float32':
            raise ValueError('original FP32 L40 fixture required')
        for seed in row['seeds']:
            if (seed.get('status') != 'passed' or seed.get('historical_initialization_checked') is not True
                    or seed.get('parameter_count') != config['model']['parameter_count']):
                raise ValueError('failed or incomplete matched encoder CUDA fixture')
            required = {'empty_vs_all_points', 'empty_vs_unique_points', 'cached_scores'}
            for name in ('all', 'unique'):
                required.update({'empty_vs_' + name + '_logits', 'empty_vs_' + name + '_mean_bce'})
                required.update('empty_vs_' + name + '_gradient/' + p for p in config['references'][str(seed['seed'])]['initial_weights_sha256'])
            if set(seed.get('checks', {})) != required or not seed.get('archive'):
                raise ValueError('complete fixture comparisons and full-array archive required')
            archive = seed['archive']
            base = Path(path).resolve().parent
            directory = base / archive['directory']
            manifest_path = directory / archive['manifest']; array_path = directory / archive['array_file']
            if any(not p.resolve().is_relative_to(base) for p in (directory, manifest_path, array_path)):
                raise ValueError('fixture archive path escapes evidence root')
            if (file_sha256(manifest_path) != archive['manifest_sha256']
                    or file_sha256(array_path) != archive['array_file_sha256']):
                raise ValueError('full-array fixture archive SHA mismatch')
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest.get('seed') != seed['seed'] or manifest.get('checks') != seed['checks']:
                raise ValueError('fixture manifest identity/comparisons mismatch')
            required_arrays = {'features', 'queries', 'labels', 'unique_ids', 'inverse_queries',
                               'points_original', 'points_all', 'points_unique', 'cache_pairs', 'cached_scores', 'direct_scores'}
            for route in ('original', 'all', 'unique'):
                required_arrays.update({'logits_' + route, 'loss_' + route})
                for parameter in config['references'][str(seed['seed'])]['initial_weights_sha256']:
                    required_arrays.update({'weights_' + route + '/' + parameter, 'gradients_' + route + '/' + parameter})
            if (set(manifest.get('arrays', {})) != required_arrays or archive.get('arrays') != len(required_arrays)
                    or manifest.get('array_file') != archive['array_file']
                    or manifest.get('array_file_sha256') != archive['array_file_sha256']):
                raise ValueError('complete raw fixture input/output/weight/gradient evidence required')
            with np.load(array_path, allow_pickle=False) as arrays:
                if set(arrays.files) != set(manifest['arrays']):
                    raise ValueError('fixture array descriptor set mismatch')
                for name in arrays.files:
                    a = arrays[name]; binding = manifest['arrays'][name]
                    if (str(a.dtype) != binding['dtype'] or list(a.shape) != binding['shape']
                            or hashlib.sha256(a.tobytes()).hexdigest() != binding['sha256']):
                        raise ValueError('fixture internal array hash/shape/dtype mismatch')
                if (arrays['features'].shape != (40, config['model']['input_dim'])
                        or arrays['queries'].shape != (50, 2) or arrays['queries'].dtype != np.int64
                        or arrays['labels'].shape != (50,)
                        or not np.array_equal(arrays['labels'], np.tile([1., 0., 0., 0., 0.], 10))
                        or not np.array_equal(arrays['unique_ids'][arrays['inverse_queries']], arrays['queries'])):
                    raise ValueError('raw fixture input/label/inverse-map contract mismatch')
                for name in required_arrays - {'queries', 'unique_ids', 'inverse_queries', 'cache_pairs'}:
                    if arrays[name].dtype != np.float32 or not np.isfinite(arrays[name]).all():
                        raise ValueError('finite FP32 raw fixture observations required')
            for name, result in seed['checks'].items():
                atol = config['equivalence']['cache_absolute_tolerance'] if name == 'cached_scores' else config['equivalence']['atol']
                rtol = 0. if name == 'cached_scores' else config['equivalence']['rtol']
                if (result.get('status') != 'passed' or result.get('finite') is not True
                        or result.get('atol') != atol or result.get('rtol') != rtol):
                    raise ValueError('fixture threshold/finite gate drift')
    return row


def checkpoint_load(path, settings, data, identity, config_hash, expected_weights, expected_step, expected_score):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved.get('protocol') != PROTOCOL or saved.get('config_sha256') != config_hash
            or saved.get('training_settings') != asdict(settings) or saved.get('source') != identity
            or saved.get('source_lf_sha256') != source_hashes()
            or saved.get('manifest_hash') != data['manifest_hash'] or saved.get('valid_hash') != data['valid_hash']
            or saved.get('completed_steps') != expected_step or saved.get('best_step') != expected_step
            or saved.get('best_full_valid_mrr') != expected_score
            or saved.get('selection_evaluation_complete') is not True
            or saved.get('weights_sha256') != expected_weights):
        raise ValueError('best checkpoint source/data/selection metadata mismatch')
    model = construct(settings)
    model.load_state_dict(saved['model'], strict=True)
    if weights_hashes(model) != expected_weights:
        raise ValueError('best checkpoint tensor hash mismatch')
    return model


def compare_references(report, gnn, mlp):
    metrics = ('query_micro_mrr', 'child_macro_mrr')
    def values(a, b):
        return {k: {'new': a[k], 'reference': b[k], 'new_minus_reference': a[k] - b[k]} for k in metrics} | {
            'hits10': {'new': a['hits']['10'], 'reference': b['hits']['10'], 'new_minus_reference': a['hits']['10'] - b['hits']['10']}}
    scheduled = []
    for row in report['evaluations']:
        if row['status'] == 'complete':
            scheduled.append({'step': row['step'], 'versus_gnn': values(row, next(v for v in gnn['evaluations'] if v['step'] == row['step'])),
                              'versus_mlp': values(row, next(v for v in mlp['evaluations'] if v['step'] == row['step']))})
    best = next((r for r in report['evaluations'] if r['step'] == report['best_step']), None)
    return {'valid_for_scientific_judgment': report['status'] == 'complete', 'difference_sign': 'new minus reference',
            'scheduled_full_valid': scheduled,
            'best': {'new_step': report['best_step'], 'versus_gnn': values(best, max(gnn['evaluations'], key=lambda r: r['query_micro_mrr'])),
                     'versus_mlp': values(best, max(mlp['evaluations'], key=lambda r: r['query_micro_mrr']))} if best else None,
            'caveat': 'Two descriptive seeds; graph path includes sampling and mixing; no pure curvature, FLOPs or convergence claim'}


def run(data, output, settings, history, *, identity, deadline=None, references=None,
        evidence=None, engineering=False, expected_initial=None, output_prepared=False):
    """CPU fixture or reviewed scientific worker; no resume or job submission."""
    settings.validate()
    device = data['features'].device
    if not engineering and (device.type != 'cuda' or expected_initial is None or references is None
                            or not (evidence or {}).get('release_verified')):
        raise ValueError('scientific run requires reviewed CUDA release/initial/reference evidence')
    features = data['features']
    if (features.dtype != torch.float32 or features.shape != (len(data['nodes']), settings.input_dim)
            or not torch.isfinite(features).all() or data['query_groups'].dtype != torch.long
            or data['query_groups'].shape[1:] != (5, 2) or data['labels'].dtype != torch.float32
            or data['labels'].shape != data['query_groups'].shape[:2]
            or not torch.equal(data['labels'].cpu(), torch.tensor([1., 0., 0., 0., 0.]).expand_as(data['labels'].cpu()))
            or not len(data['valid']) or settings.batch_positives > len(data['query_groups'])):
        raise ValueError('finite features/original labels/group contract required')
    root = Path(output)
    if not output_prepared:
        root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + settings.max_seconds if deadline is None else deadline
    if not math.isfinite(deadline) or deadline > started + settings.max_seconds:
        raise ValueError('finite deadline cannot exceed registered total worker budget')
    torch.set_num_threads(settings.threads)
    model = construct(settings)
    if expected_initial is not None:
        verify_initial(model, expected_initial)
    model.to(device)
    initial = {'protocol': PROTOCOL, 'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
               'seed': settings.seed, 'training_settings': asdict(settings), 'config_sha256': CONFIG_SHA256,
               'source': identity, 'source_lf_sha256': source_hashes(), 'weights_sha256': weights_hashes(model)}
    initial_tmp = root / 'initial.pt.tmp'; torch.save(initial, initial_tmp); initial_tmp.replace(root / 'initial.pt')
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    stream = BatchStream(len(data['query_groups']), settings.seed, settings.batch_positives, history)
    before = tensor_identity(data)
    best = best_step = None; completed = 0; best_weights = None
    report = {'protocol': PROTOCOL, 'scope': 'CPU synthetic engineering fixture only' if engineering else 'Bounded matched-encoder E2 control',
              'research_question': 'At the same radial encoder and head, does replacing neighbor aggregation with self messages change parent retrieval?',
              'purpose_status': 'insufficient_evidence', 'conclusion': 'Awaiting complete acquisition and independent review',
              'status': 'running', 'seed': settings.seed, 'config_sha256': CONFIG_SHA256,
              'training_settings': asdict(settings), 'source': identity, 'source_lf_sha256': source_hashes(),
              'manifest_hash': data['manifest_hash'], 'validation_queries_hash': data['valid_hash'],
              'input_files_opened': data.get('input_files_opened', []), 'input_identity_before': before,
              'initial_weights_sha256': weights_hashes(model), 'parameter_count': sum(p.numel() for p in model.parameters()),
              'initial_snapshot': {'file': 'initial.pt', 'sha256': file_sha256(root / 'initial.pt'),
                                   'bytes': (root / 'initial.pt').stat().st_size},
              'python': platform.python_version(), 'torch': str(torch.__version__), 'dtype': 'float32',
              'device_type': device.type, 'device_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
              'TF32': False, 'AMP': False, 'optimizer': 'Adam', 'test_truth_opened': False,
              'completed_steps': 0, 'steps': [], 'evaluations': [], 'checkpoint_seconds': [],
              'best_full_valid_mrr': None, 'best_step': None, 'best_reload': None,
              'graph_use': 'Input validation and fixed degree reporting only; no neighbors enter encoder',
              'batch_rng': 'CPU randperm each step, model_seed+1; every hash checked against original MLP',
              'selection_status': 'none; no complete full valid yet', **(evidence or {})}
    truth = defaultdict(set)
    for a, b in data['valid']:
        truth[b].add(a)
    def remaining():
        return deadline - time.monotonic()
    def phase(name, step=None):
        report['active_phase'] = name
        if step is not None:
            report['active_step'] = step
        atomic_json(root / 'progress.json', {'completed_steps': completed,
                    'active_step': report.get('active_step'), 'active_phase': name})
    def save(name):
        synchronize(device); begin = time.monotonic()
        checkpoint = {'protocol': PROTOCOL, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'training_settings': asdict(settings), 'config_sha256': CONFIG_SHA256,
                      'source': identity, 'source_lf_sha256': report['source_lf_sha256'],
                      'manifest_hash': data['manifest_hash'], 'valid_hash': data['valid_hash'],
                      'completed_steps': completed, 'best_full_valid_mrr': best, 'best_step': best_step,
                      'selection_evaluation_complete': name == 'best.pt', 'run_status': report['status'],
                      'batch_rng': stream.rng.get_state(), 'weights_sha256': weights_hashes(model),
                      'resume_allowed': False}
        tmp = root / (name + '.tmp'); torch.save(checkpoint, tmp); tmp.replace(root / name)
        report['checkpoint_seconds'].append(time.monotonic() - begin)
    def rank(which):
        if remaining() <= 0:
            raise TimeoutError('deadline before full validation')
        which.eval(); synchronize(device); begin = time.monotonic()
        with torch.no_grad():
            points = which.encode(features)
        synchronize(device)
        result = filtered_parent_ranks(which, points, data['valid'], truth, settings.candidate_chunk,
                                       max_seconds=max(0., min(settings.evaluation_max_seconds - (time.monotonic() - begin), remaining())))
        result['encoding_seconds'] = time.monotonic() - begin - result['elapsed_seconds']
        if result['status'] != 'complete' or time.monotonic() - begin > settings.evaluation_max_seconds or remaining() <= 0:
            atomic_json(root / 'partial-validation.json', result)
            raise TimeoutError('incomplete scheduled/reloaded full validation; cannot select or continue')
        validate_ranking(result, data)
        result['parent_degree_groups'] = degree_readings(result, data)
        return result
    try:
        save('last.pt'); atomic_json(root / 'run.json', report)
        model.train()
        for step in range(1, settings.max_steps + 1):
            phase('batch_hash_verification', step)
            if remaining() <= 0:
                raise TimeoutError('total deadline before update')
            synchronize(device); begin = time.monotonic(); chosen = stream.next(step)
            queries = data['query_groups'][chosen].reshape(-1, 2).to(device)
            labels = data['labels'][chosen].reshape(-1).to(device)
            phase('forward_backward_update')
            optimizer.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(model(features, queries), labels)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite mean training BCE')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('missing/nonfinite parameter gradient')
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError('nonfinite parameter after Adam update')
            completed = step; report['completed_steps'] = completed; synchronize(device)
            phase('update_completed')
            report['steps'].append({'step': step, 'loss': float(loss.detach()), 'seconds': time.monotonic() - begin,
                                    'positive_queries': len(chosen), 'negative_queries': 4 * len(chosen),
                                    'batch_group_ids_sha256': canonical(chosen.tolist())})
            if step % settings.evaluate_every == 0:
                phase('scheduled_full_validation')
                result = rank(model); result.update(step=step, purpose='full_validation')
                report['evaluations'].append(result)
                score = result['query_micro_mrr']
                if best is None or score > best:
                    best, best_step = score, step; best_weights = weights_hashes(model)
                    report.update(best_full_valid_mrr=best, best_step=best_step,
                                  selection_status='first complete full-valid micro MRR maximum; strictly greater replaces best')
                    save('best.pt')
                model.train(); atomic_json(root / 'run.json', report)
            if step % settings.save_every == 0:
                save('last.pt'); atomic_json(root / 'run.json', report)
        if completed != settings.max_steps or len(report['evaluations']) != settings.max_steps // settings.evaluate_every:
            raise ValueError('missing registered steps/evaluations')
        phase('best_reload_full_validation')
        reloaded = checkpoint_load(root / 'best.pt', settings, data, identity, CONFIG_SHA256,
                                   best_weights, best_step, best).to(device)
        reproduction = rank(reloaded)
        selected = next(r for r in report['evaluations'] if r['step'] == best_step)
        for field in ('rows', 'query_micro_mrr', 'child_macro_mrr', 'hits', 'parent_degree_groups',
                      'expected_queries', 'completed_queries', 'completed_children', 'metric_scope', 'tie_policy'):
            if reproduction[field] != selected[field]:
                raise ValueError('reloaded best full valid differs: ' + field)
        report['best_reload'] = {'status': 'passed', 'checkpoint_sha256': file_sha256(root / 'best.pt'),
                                 'weights_sha256': best_weights, 'best_step': best_step, 'ranking': reproduction,
                                 'fresh_complete_ranking_performed': True, 'additional_selection_opportunities': 0}
        if tensor_identity(data) != before or remaining() <= 0:
            raise ValueError('input tensors changed or final deadline exceeded')
        report.update(status='complete', purpose_status='pending_supervisor_review',
                      conclusion='Complete matched control acquired; scientific interpretation awaits independent review')
    except TimeoutError as error:
        report.update(status='incomplete_time_limit', error=str(error), conclusion='Incomplete control is not scientific comparison evidence')
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', conclusion='Failed control is not scientific comparison evidence')
    finally:
        phase('final_archival')
        try:
            save('last.pt')
        except Exception as error:
            report.update(status='failed', archival_error=f'{type(error).__name__}: {error}')
        report.update(elapsed_seconds=time.monotonic() - started, input_identity_after=tensor_identity(data),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.type == 'cuda' else None)
        report['checkpoints'] = {n: {'sha256': file_sha256(root / n), 'bytes': (root / n).stat().st_size}
                                 for n in ('last.pt', 'best.pt') if (root / n).is_file()}
        if report['status'] == 'complete' and remaining() <= 0:
            report.update(status='incomplete_time_limit', purpose_status='insufficient_evidence',
                          error='deadline exceeded during final archival', conclusion='Incomplete control is not scientific comparison evidence')
        if references is not None:
            report['comparison'] = compare_references(report, *references)
        atomic_json(root / 'run.json', report)
        if report['status'] != 'complete':
            atomic_json(root / 'failure.json', {'status': report['status'], 'completed_steps': completed,
                                               'active_phase': report.get('active_phase'), 'error': report.get('error', report.get('archival_error'))})
    return report


def supervision_deadline(config, phase):
    expected = config['resources']['science_worker_seconds'] if phase == 'science' else config['resources']['fixture_worker_seconds']
    if os.environ.get('ACL_ENCODER_MATCHED_SUPERVISOR_PID') != str(os.getppid()):
        raise ValueError('must use encoder_matched_entry process deadline')
    deadline = float(os.environ.get('ACL_ENCODER_MATCHED_DEADLINE', 'nan'))
    if not math.isfinite(deadline) or not 0 < deadline - time.monotonic() <= expected:
        raise ValueError('registered external worker deadline required')
    return deadline


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--config', type=Path, required=True)
    cli.add_argument('--phase', choices=('cpu_preflight', 'cuda_fixture', 'science'), required=True)
    for name in ('source-commit', 'approval-record', 'quality-record', 'output'):
        cli.add_argument('--' + name, required=True)
    cli.add_argument('--seed', type=int, choices=(11, 23))
    for name in ('prepared-root', 'reference-root', 'cpu-preflight-record', 'cuda-fixture-record'):
        cli.add_argument('--' + name, type=Path)
    args = cli.parse_args(); root = Path(args.output)
    try:
        config = json.loads(args.config.read_text(encoding='utf-8')); validate_config(config)
        deadline = supervision_deadline(config, args.phase)
        identity = source_identity(args.source_commit)
        approval = json.loads(Path(args.approval_record).read_text(encoding='utf-8'))
        if approval.get('execution_phase') != args.phase:
            raise ValueError('reviewed approval phase mismatch')
        evidence = verify_release(config, identity, approval, args.quality_record)
        require_runtime(config, torch.device('cpu' if args.phase == 'cpu_preflight' else 'cuda'))
        if args.phase == 'cuda_fixture':
            if any(v is not None for v in (args.prepared_root, args.reference_root, args.seed)):
                raise ValueError('CUDA fixture must not open real prepared/reference paths')
            torch.set_num_threads(config['training']['threads']); torch.cuda.reset_peak_memory_stats()
            results = [equivalence_fixture(config, s, 'cuda', historical_initialization=True,
                                          archive_root=root / ('seed' + str(s))) for s in config['seeds']]
            passed = all(r['status'] == 'passed' for r in results)
            row = {'status': 'complete' if passed else 'failed', 'matched_encoder_fixture_passed': passed,
                   'protocol': PROTOCOL, 'source': identity, 'source_lf_sha256': source_hashes(),
                   'config_sha256': CONFIG_SHA256, 'torch': str(torch.__version__), 'dtype': 'float32',
                   'device_name': torch.cuda.get_device_name(0), 'seeds': results, 'fixture_only': True, **evidence}
            atomic_json(root / 'cuda-quality.json', row)
        else:
            if args.prepared_root is None or args.reference_root is None:
                raise ValueError('fixed prepared/reference roots required')
            paths = {s: (args.reference_root / f'baseline-seed{s}-run.json', args.reference_root / f'e2-capacity-seed{s}-run.json') for s in config['seeds']}
            if args.phase == 'cpu_preflight':
                if args.seed is not None:
                    raise ValueError('CPU preflight must cover both registered seeds')
                row = cpu_preflight(config, {s: args.prepared_root for s in config['seeds']}, paths, identity, evidence)
                atomic_json(root / 'cpu-preflight.json', row)
            else:
                if args.seed is None or args.cpu_preflight_record is None or args.cuda_fixture_record is None:
                    raise ValueError('seed and independently reviewed preflight/CUDA evidence required')
                if approval.get('cpu_preflight_review_passed') is not True or approval.get('cuda_fixture_review_passed') is not True:
                    raise ValueError('supervisor must accept real CPU and CUDA gates before scientific training')
                verify_gate(args.cpu_preflight_record, approval.get('cpu_preflight_artifact_sha256'), 'cpu_preflight', config, identity)
                verify_gate(args.cuda_fixture_record, approval.get('cuda_fixture_artifact_sha256'), 'cuda_fixture', config, identity)
                data = load_prepared(args.prepared_root, config)
                refs = verify_references(*paths[args.seed], config, args.seed, data)
                data['features'] = data['features'].to('cuda'); torch.cuda.reset_peak_memory_stats()
                row = run(data, root, settings_from_config(config, args.seed), refs[1], identity=identity, deadline=deadline,
                          references=refs, evidence={**evidence, 'release_verified': True,
                          'cpu_preflight_artifact_sha256': file_sha256(args.cpu_preflight_record),
                          'cuda_fixture_artifact_sha256': file_sha256(args.cuda_fixture_record)},
                          expected_initial=config['references'][str(args.seed)]['initial_weights_sha256'], output_prepared=True)
        if time.monotonic() >= deadline:
            raise TimeoutError('deadline reached before final worker acceptance')
        print(json.dumps({'status': row['status'], 'phase': args.phase}))
        return 0 if row['status'] == 'complete' else 2
    except Exception as error:
        atomic_json(root / 'failure.json', {'status': 'failed', 'phase': args.phase, 'error': f'{type(error).__name__}: {error}'})
        print(f'{type(error).__name__}: {error}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

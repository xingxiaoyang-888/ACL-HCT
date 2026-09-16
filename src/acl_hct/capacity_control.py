"""Registered bounded E2 text capacity control; separate from frozen diagnostics."""
import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import torch
from .mechanisms import source_identity
from .protocols import digest
from .train import atomic_json, load_prepared, synchronize
from .text_capacity import TextMLP, filtered_parent_ranks, prepare_scores, score_cached

PROTOCOL = 'E2-model-capacity-control-v1'
CONFIG_SHA256 = 'c40ad1d32bb338047f307749bf0ef023c204ce1cf39550e003d77ce665df7634'
SOURCE_NAMES = ('__init__ aggregation backbone data development_view diagnostic_archive diagnostics '
                'e1_audit e1_closure e1_controls e2_development e2_pilot evaluate_checkpoint '
                'frozen_fixture frozen_forward frozen_stats frozen_structure geometry mechanisms '
                'model protocols ranking smoke taxonomy train vector_structure '
                'e2_confirmation confirmation_registration confirmation_analysis text_capacity capacity_control').split()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def source_hashes():
    return {'acl_hct/'+name+'.py': hashlib.sha256((Path(__file__).parent/(name+'.py')).read_text(encoding='utf-8').encode()).hexdigest()
            for name in SOURCE_NAMES}


def validate_config(config):
    if digest(config) != CONFIG_SHA256:
        raise ValueError('fixed E2 text capacity registration required')
    return CONFIG_SHA256


@dataclass(frozen=True)
class TrainSettings:
    seed: int = 11
    input_dim: int = 128
    hidden: int = 128
    head_hidden: int = 128
    batch_positives: int = 128
    max_steps: int = 1024
    learning_rate: float = .003
    max_seconds: float = 3300.
    evaluation_max_seconds: float = 180.
    candidate_chunk: int = 4096
    evaluate_every: int = 256
    save_every: int = 50
    threads: int = 2

    def validate(self):
        if type(self.seed) is not int or self.seed not in (11, 23):
            raise ValueError('registered model seeds required')
        for name, limit in [('input_dim', 128), ('hidden', 128), ('head_hidden', 128), ('batch_positives', 128),
                            ('max_steps', 1024), ('candidate_chunk', 4096), ('evaluate_every', 256),
                            ('save_every', 50), ('threads', 2)]:
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError('invalid bounded '+name)
        if self.max_steps % self.evaluate_every:
            raise ValueError('training must end at a scheduled complete-valid evaluation')
        if (not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not math.isfinite(self.max_seconds) or not 0 < self.max_seconds <= 3300
                or not math.isfinite(self.evaluation_max_seconds) or not 0 < self.evaluation_max_seconds <= min(180, self.max_seconds)):
            raise ValueError('finite bounded training and validation budgets required')


def settings_from_config(config, seed):
    validate_config(config)
    if seed not in config['seeds']:
        raise ValueError('unregistered model seed')
    training = {k: v for k, v in config['training'].items() if k not in ('optimizer', 'negatives_per_positive')}
    return TrainSettings(seed=seed, **{k: config['model'][k] for k in ('input_dim', 'hidden', 'head_hidden')}, **training)


def require_cuda(device):
    if (device.type != 'cuda' or device.index not in (None, 0) or not os.environ.get('SLURM_JOB_ID')
            or not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count() != 1):
        raise ValueError('one visible Slurm GPU with preserved binding required')


def validate_ranking(row, data):
    truth = defaultdict(set)
    for parent, child in data['valid']:
        truth[child].add(parent)
    rows = row['rows']; pairs = [(r['parent'], r['child']) for r in rows]
    if (row['status'] != 'complete' or row['metric_scope'] != 'filtered_all_entity_candidates'
            or len(rows) != len(data['valid']) or len(set(pairs)) != len(rows) or set(pairs) != set(data['valid'])
            or row['completed_queries'] != len(rows) or row['expected_queries'] != len(rows)
            or row['completed_children'] != len(truth)):
        raise ValueError('complete original valid query coverage required')
    child_rr = defaultdict(list)
    for r in rows:
        candidates = len(data['nodes'])-len(truth[r['child']])
        if r['candidates'] != candidates or not math.isfinite(r['rank']) or not 1 <= r['rank'] <= candidates:
            raise ValueError('invalid candidate count or rank')
        child_rr[r['child']].append(1/r['rank'])
    micro = sum(1/r['rank'] for r in rows)/len(rows)
    macro = sum(sum(rs)/len(rs) for rs in child_rr.values())/len(child_rr)
    if any(not math.isfinite(row[k]) or abs(row[k]-v) > 1e-12 for k, v in [('query_micro_mrr', micro), ('child_macro_mrr', macro)]):
        raise ValueError('ranking summary differs from saved ranks')


def verify_baseline(path, config, seed, data):
    validate_config(config)
    anchor = config['baseline_anchors'][str(seed)]
    if file_sha256(path) != anchor['baseline_report_sha256']:
        raise ValueError('original baseline raw hash mismatch')
    baseline = json.loads(Path(path).read_text(encoding='utf-8'))
    if (digest(baseline['config']) != digest(anchor['training_config'])
            or baseline['source']['source_commit'] != anchor['training_commit']
            or baseline['status'] != 'step_limit_reached' or baseline['completed_steps'] != 1024
            or data['manifest_hash'] != config['data']['prepared_manifest_hash']
            or data['valid_hash'] != config['data']['valid_queries_hash']
            or baseline['manifest_hash'] != data['manifest_hash']
            or baseline['validation_queries_hash'] != data['valid_hash']
            or data['features'].shape[1] != config['model']['input_dim']
            or [r['step'] for r in baseline['evaluations']] != [256, 512, 768, 1024]):
        raise ValueError('original data/config/baseline identity mismatch')
    for row in baseline['evaluations']:
        if row['purpose'] != 'full_validation':
            raise ValueError('full valid baseline evaluations required')
        validate_ranking(row, data)
    winner = max(baseline['evaluations'], key=lambda r: r['query_micro_mrr'])
    if winner['step'] != anchor['step'] or winner['query_micro_mrr'] != baseline['best_full_valid_mrr']:
        raise ValueError('original first-best selection mismatch')
    return baseline


def verify_approval(approval, config, identity):
    if (approval.get('user_authorized') is not True or not approval.get('user_message_reference')
            or approval.get('scope') != PROTOCOL or approval.get('config_sha256') != validate_config(config)
            or approval.get('source_commit') != identity['source_commit']
            or approval.get('quality_review_passed') is not True or approval.get('entry_criteria_frozen') is not True
            or approval.get('cuda_fixture_passed') is not True):
        raise ValueError('user authorization and reviewed fixed capacity code/config/CUDA evidence required')


def verify_cuda_evidence(path, approval, identity):
    if file_sha256(path) != approval.get('cuda_fixture_artifact_sha256'):
        raise ValueError('capacity CUDA fixture raw hash mismatch')
    row = json.loads(Path(path).read_text(encoding='utf-8'))
    if (row.get('capacity_fixture_passed') is not True or row.get('status') != 'complete'
            or row.get('config_sha256') != CONFIG_SHA256
            or row.get('source', {}).get('source_commit') != identity['source_commit']
            or row.get('source_sha256_normalized_lf') != source_hashes()):
        raise ValueError('new capacity CUDA fixture for exact source and config required')


def weights_hashes(model):
    return {k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k, v in model.state_dict().items()}


def compare_to_baseline(report, baseline):
    previous = {r['step']: r for r in baseline['evaluations']}
    pairs = []
    for row in report['evaluations']:
        if row['status'] == 'complete':
            pairs.append({'step': row['step'], **{k: {'text_mlp': row[k], 'gnn': previous[row['step']][k],
                'text_minus_gnn': row[k]-previous[row['step']][k]} for k in ('query_micro_mrr', 'child_macro_mrr')}})
    winner = max(baseline['evaluations'], key=lambda r: r['query_micro_mrr'])
    best = next((r for r in report['evaluations'] if r['step'] == report['best_step']), None)
    return {'difference_sign': 'text minus GNN', 'scheduled_valid': pairs,
            'best': {'text_step': report['best_step'], 'gnn_step': winner['step'],
                     **{k: {'text_mlp': best[k], 'gnn': winner[k], 'text_minus_gnn': best[k]-winner[k]}
                        for k in ('query_micro_mrr', 'child_macro_mrr')}} if best else None,
            'gnn_elapsed_seconds': baseline['elapsed_seconds'],
            'caveat': 'two fixed seeds; bounded reference, no training-randomness significance, SOTA or matched FLOPs/convergence claim'}


def run(data, output_dir, settings=TrainSettings(), *, device='cpu', identity=None,
        config_hash=CONFIG_SHA256, baseline=None, started=None, evidence=None, engineering=False):
    """Internal bounded runner; scientific CLI below enforces the pinned registration."""
    settings.validate(); device = torch.device(device)
    if device.type not in ('cpu', 'cuda') or device.index not in (None, 0):
        raise ValueError('single CPU/CUDA device required')
    if device.type == 'cuda':
        require_cuda(device)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest'); torch.cuda.reset_peak_memory_stats()
    features = data['features'].to(device)
    if (features.dtype != torch.float32 or features.ndim != 2 or features.shape != (len(data['nodes']), settings.input_dim)
            or not torch.isfinite(features).all() or data['query_groups'].shape[1:] != (5, 2)
            or not torch.equal(data['labels'], torch.tensor([1., 0., 0., 0., 0.]).expand_as(data['labels']))):
        raise ValueError('finite feature matrix and original fixed training groups required')
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter() if started is None else started
    torch.set_num_threads(settings.threads); torch.manual_seed(settings.seed)
    model = TextMLP(settings.input_dim, settings.hidden, settings.head_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    batch_rng = torch.Generator().manual_seed(settings.seed+1)
    best = None; best_step = None; completed = 0
    report = {'protocol': PROTOCOL, 'scope': 'CPU/CUDA engineering fixture only' if engineering else 'bounded original E2 model capacity reference',
              'research_question': 'Does the existing GNN outperform a bounded graph-free text MLP on original full valid?',
              'conclusion': 'pending acquired evidence and independent review', 'purpose_status': 'insufficient_evidence',
              'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'config_sha256': config_hash, 'training_settings': asdict(settings), 'seed': settings.seed,
              'source': identity or source_identity(), 'source_sha256_normalized_lf': source_hashes(),
              'manifest_hash': data['manifest_hash'], 'validation_queries_hash': data['valid_hash'],
              'graph_hash': data['manifest']['graph_hash'], 'train_queries_hash': data['manifest']['train_queries_hash'],
              'feature_sha256': data['manifest']['feature_manifest']['feature_sha256'],
              'label_files_read': ['input_manifest.json', 'observed_graph.json', 'train_queries.json', 'evaluator_valid.json', 'features.npz']
                                 if not engineering else [],
              'observed_graph_use': 'input-contract validation only; never enters encoder or scorer',
              'training_encoding': 'only unique query entities; no cross-entity encoder operations',
              'batch_rng': 'CPU randperm(number of original positive groups), first batch_positives, seed=model_seed+1',
              'parameter_count': sum(p.numel() for p in model.parameters()), 'initial_weights_sha256': weights_hashes(model),
              'precision': 'FP32; no AMP/TF32', 'device_type': device.type,
              'device_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
              'python': platform.python_version(), 'torch': str(torch.__version__),
              'completed_steps': 0, 'steps': [], 'evaluations': [], 'checkpoint_seconds': [],
              'best_full_valid_mrr': None, 'best_step': None, 'selection_status': 'none; no complete full valid yet',
              'budget_check': 'at step/child boundaries; one operation may exceed deadline',
              'full_valid_query_count': len(data['valid']), 'valid_query_pairs_hash': digest(sorted(data['valid']))}
    report.update(evidence or {})
    truth = defaultdict(set)
    for parent, child in data['valid']:
        truth[child].add(parent)
    def remaining():
        return settings.max_seconds-(time.perf_counter()-started)
    def save(name):
        synchronize(device); begin = time.perf_counter()
        checkpoint = {'protocol': PROTOCOL, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'training_settings': asdict(settings), 'config_sha256': config_hash,
                      'manifest_hash': data['manifest_hash'], 'valid_hash': data['valid_hash'],
                      'batch_rng': batch_rng.get_state(), 'torch_rng': torch.get_rng_state(),
                      'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                      'device_type': device.type, 'completed_steps': completed,
                      'best_full_valid_mrr': best, 'best_step': best_step,
                      'run_status': report['status'], 'selection_status': report['selection_status'],
                      'selection_evaluation_complete': name == 'best.pt',
                      'source': report['source'], 'source_sha256_normalized_lf': report['source_sha256_normalized_lf'],
                      'weights_sha256': weights_hashes(model)}
        temporary = output/(name+'.tmp'); torch.save(checkpoint, temporary); temporary.replace(output/name)
        report['checkpoint_seconds'].append(time.perf_counter()-begin)
    def evaluate():
        nonlocal best, best_step
        model.eval(); synchronize(device); begin = time.perf_counter()
        with torch.no_grad():
            embeddings = model.encode(features)
        synchronize(device); encoding_seconds = time.perf_counter()-begin
        result = filtered_parent_ranks(model, embeddings, data['valid'], truth, settings.candidate_chunk,
                                       max_seconds=max(0., min(settings.evaluation_max_seconds, remaining())))
        result.update(step=completed, purpose='full_validation', full_graph_encoding_seconds=encoding_seconds)
        report['evaluations'].append(result)
        atomic_json(output/'run.json', report)
        if result['status'] != 'complete':
            raise TimeoutError('scheduled complete valid timed out; partial evaluation cannot select or continue')
        validate_ranking(result, data)
        score = result['query_micro_mrr']
        if best is None or score > best:
            best, best_step = score, completed
            report.update(best_full_valid_mrr=best, best_step=best_step,
                          selection_status='first complete filtered all-candidate validation micro MRR maximum')
            save('best.pt')
        model.train()
    try:
        save('last.pt'); atomic_json(output/'run.json', report)
        model.train()
        for step in range(1, settings.max_steps+1):
            if remaining() <= 0:
                raise TimeoutError('fixed total training deadline reached')
            synchronize(device); begin = time.perf_counter()
            chosen = torch.randperm(len(data['query_groups']), generator=batch_rng)[:settings.batch_positives]
            queries = data['query_groups'][chosen].reshape(-1, 2).to(device)
            labels = data['labels'][chosen].reshape(-1).to(device)
            optimizer.zero_grad()
            logits = model(features, queries)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('missing/nonfinite model gradient')
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError('nonfinite model weight after optimizer step')
            completed = step; report['completed_steps'] = completed; synchronize(device)
            report['steps'].append({'step': step, 'loss': float(loss.detach()), 'seconds': time.perf_counter()-begin,
                                    'positive_queries': len(chosen), 'negative_queries': 4*len(chosen),
                                    'batch_group_ids_sha256': digest(chosen.tolist())})
            if step % settings.evaluate_every == 0:
                if remaining() <= 0:
                    raise TimeoutError('deadline before scheduled complete valid')
                evaluate()
            if step % settings.save_every == 0:
                save('last.pt'); atomic_json(output/'run.json', report)
        if remaining() <= 0:
            raise TimeoutError('deadline exceeded during final scheduled operation')
        report.update(status='complete', conclusion='Bounded text reference acquired; scientific capacity judgment awaits independent review',
                      purpose_status='pending_supervisor_review')
    except TimeoutError as error:
        report.update(status='incomplete_time_limit', error=str(error),
                      conclusion='Incomplete bounded text reference; not valid evidence of model competitiveness')
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}',
                      conclusion='Failed bounded text reference; not valid evidence of model competitiveness')
    finally:
        save('last.pt')
        report['elapsed_seconds'] = time.perf_counter()-started
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated() if device.type == 'cuda' else None
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved() if device.type == 'cuda' else None
        report['checkpoints'] = {name: {'sha256': file_sha256(output/name), 'bytes': (output/name).stat().st_size}
                                 for name in ('last.pt', 'best.pt') if (output/name).exists()}
        if baseline is not None:
            report['comparison'] = compare_to_baseline(report, baseline)
            report['comparison']['valid_for_capacity_judgment'] = report['status'] == 'complete'
            cfg = baseline['config']; d = settings.input_dim; h = cfg['hidden']; hh = cfg['head_hidden']
            report['comparison']['gnn_parameter_count'] = (d+1)*h+(h+1)*h+(3*h+1)*hh+hh+1
        atomic_json(output/'run.json', report)
    return report


def engineering_data():
    """40-node synthetic train+valid contract; no real prepared directory or sealed panel."""
    from .frozen_fixture import synthetic_fixture
    _, _, view = synthetic_fixture()
    features = torch.randn(40, 128, generator=torch.Generator().manual_seed(2026091703))*.1
    index = {n: i for i, n in enumerate(view.nodes)}
    valid = [(index[a], index[b]) for a, b in sorted(view.valid_edges)]
    groups = torch.tensor([[[0, child]]+[[n, child] for n in (1, 2, 3, 4)] for child in range(5, 15)], dtype=torch.long)
    manifest = {'graph_hash': digest(view.neighbors), 'train_queries_hash': digest(groups.tolist()),
                'feature_manifest': {'feature_sha256': hashlib.sha256(features.numpy().tobytes()).hexdigest()}}
    return {'features': features, 'nodes': view.nodes, 'manifest': manifest, 'manifest_hash': digest(manifest),
            'valid_hash': digest(valid), 'valid': valid, 'neighbors': view.neighbors,
            'query_groups': groups, 'labels': torch.tensor([1., 0., 0., 0., 0.]).expand(len(groups), -1).clone()}


@torch.no_grad()
def cache_check(model, features):
    model.eval(); embeddings = model.encode(features)
    queries = torch.cartesian_prod(torch.arange(len(features), device=features.device),
                                   torch.arange(len(features), device=features.device))
    cache = prepare_scores(model, embeddings)
    direct = model.score(embeddings, queries)
    cached = score_cached(model, cache, queries[:, 0], queries[:, 1])
    finite = bool(torch.isfinite(direct).all() and torch.isfinite(cached).all())
    error = float((direct-cached).abs().max()) if finite else None
    return {'pairs': len(queries), 'maximum_absolute_score_difference': error, 'absolute_tolerance': 1e-5,
            'nonfinite_scores': not finite, 'status': 'passed' if finite and error <= 1e-5 else 'failed'}


def cuda_fixture(output, identity, config):
    validate_config(config); require_cuda(torch.device('cuda'))
    begin = time.perf_counter(); data = engineering_data()
    result = run(data, output, TrainSettings(batch_positives=4, max_steps=4, evaluate_every=1,
                 save_every=1, max_seconds=120, evaluation_max_seconds=20, candidate_chunk=11),
                 device='cuda', identity=identity, engineering=True)
    if result['status'] == 'complete':
        saved = torch.load(Path(output)/'last.pt', map_location='cpu', weights_only=True)
        model = TextMLP().to('cuda'); model.load_state_dict(saved['model'], strict=True)
        result['cache_check'] = cache_check(model, data['features'].to('cuda'))
        if result['cache_check']['status'] != 'passed':
            result.update(status='failed', error='CUDA cached scores differ from direct MLP scores')
    result['capacity_fixture_passed'] = result['status'] == 'complete' and len(result['evaluations']) == 4
    result['fixture_elapsed_seconds'] = time.perf_counter()-begin
    atomic_json(Path(output)/'cuda-quality.json', result)
    if not result['capacity_fixture_passed']:
        raise ValueError('capacity CUDA engineering fixture failed')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true'); parser.add_argument('--cuda-fixture', action='store_true')
    parser.add_argument('--seed', type=int, choices=(11, 23)); parser.add_argument('--source-commit')
    for name in ('prepared', 'baseline-report', 'approval-record', 'cuda-fixture-record', 'output-dir'):
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args(); config = json.loads(args.config.read_text(encoding='utf-8'))
    config_hash = validate_config(config)
    if not args.execute and not args.cuda_fixture:
        print(json.dumps({'status': 'static_only', 'config_sha256': config_hash, 'seeds': config['seeds']})); return
    if args.source_commit is None or args.output_dir is None:
        raise ValueError('fixed source and new output directory required')
    identity = source_identity(args.source_commit)
    if args.cuda_fixture:
        if args.execute or args.prepared is not None or args.baseline_report is not None:
            raise ValueError('engineering fixture must not use real model/data evidence')
        cuda_fixture(args.output_dir, identity, config); return
    if any(getattr(args, k) is None for k in ('seed', 'prepared', 'baseline_report', 'approval_record', 'cuda_fixture_record')):
        raise ValueError('registered seed/data/baseline, approval and new capacity CUDA gate required')
    started = time.perf_counter()
    approval = json.loads(args.approval_record.read_text(encoding='utf-8'))
    verify_approval(approval, config, identity); verify_cuda_evidence(args.cuda_fixture_record, approval, identity)
    data = load_prepared(args.prepared); baseline = verify_baseline(args.baseline_report, config, args.seed, data)
    evidence = {'config': config, 'approval_record_sha256_canonical': digest(approval),
                'cuda_fixture_artifact_sha256': file_sha256(args.cuda_fixture_record),
                'baseline_report_raw_sha256': file_sha256(args.baseline_report),
                'original_best_checkpoint_sha256': config['baseline_anchors'][str(args.seed)]['sha256']}
    result = run(data, args.output_dir, settings_from_config(config, args.seed), device='cuda', identity=identity,
                 config_hash=config_hash, baseline=baseline, started=started, evidence=evidence)
    print(json.dumps({'status': result['status'], 'completed_steps': result['completed_steps'], 'best_step': result['best_step']}))
    if result['status'] != 'complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()

"""Real-runtime HGCN compatibility and cost worker; never a science entry."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .backbone_frozen_inputs import load_train_prepared
from .diagnostic_archive import write_archive
from .hgcn_fixture import run_fixture
from .hgcn_registration import canonical, source_hashes, validate_config, verify_release
from .hgcn_sampling import make_plan, mask_graph
from .hgcn_upstream import load_upstream
from .mature_hgcn import MatureHGCN
from .text_capacity import filtered_parent_ranks


def atomic_json(path, value):
    path = Path(path); temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    temporary.replace(path)


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def check_runtime(config, phase):
    if torch.__version__ != config['runtime']['torch'] or np.__version__ != config['runtime']['numpy']:
        raise ValueError('reviewed original torch/numpy runtime required')
    if torch.get_default_dtype() != torch.float32:
        raise ValueError('FP32 default dtype required')
    torch.set_num_threads(config['runtime']['threads'])
    if phase == 'cuda_quality':
        if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise ValueError('recorded Slurm allocation and preserved GPU binding required')
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != config['runtime']['hardware']:
            raise ValueError('one allocated NVIDIA L40 required')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest'); torch.cuda.reset_peak_memory_stats()
        return torch.device('cuda:0')
    return torch.device('cpu')


def check_deadline():
    if (int(os.environ.get('ACL_HGCN_SUPERVISOR_PID', '0')) != os.getppid()
            or time.monotonic() >= float(os.environ.get('ACL_HGCN_DEADLINE', '0'))):
        raise ValueError('live bounded entry supervisor required')


def load_train(root, config):
    """Only four train inputs; evaluator files are loaded separately by caller."""
    expected = config['prepared']
    feature_raw = (Path(root) / 'features.npz').read_bytes()
    if hashlib.sha256(feature_raw).hexdigest() != expected['features_npz_raw_sha256']:
        raise ValueError('original feature archive raw hash mismatch')
    data = load_train_prepared(root, expected)
    if data['features'].shape[1] != config['model']['input_dim']:
        raise ValueError('registered train-only feature width mismatch')
    return data


def load_valid(root, data, config):
    """Evaluator-only original valid positives; test/truth never opened."""
    raw = (Path(root) / 'evaluator_valid.json').read_bytes(); external = json.loads(raw)
    if (hashlib.sha256(raw).hexdigest() != config['prepared']['evaluator_valid_raw_sha256']
            or canonical(external) != config['prepared']['valid_queries_hash']
            or len(external) != config['valid_query_count']):
        raise ValueError('original complete valid identity/count mismatch')
    index = {n: i for i, n in enumerate(data['nodes'])}
    fit = set(data['manifest']['text_fit_entities']); valid = []; truth = defaultdict(set)
    for pair in external:
        if (not isinstance(pair, list) or len(pair) != 2 or any(n not in index for n in pair)
                or pair[0] == pair[1] or pair[1] in fit):
            raise ValueError('invalid heldout child-grouped positive')
        a, b = map(index.__getitem__, pair)
        if b in data['neighbors'][a] or a in data['neighbors'][b]:
            raise ValueError('valid edge leaks into visible train graph')
        valid.append((a, b)); truth[b].add(a)
    if len(set(valid)) != len(valid):
        raise ValueError('duplicate valid target')
    return valid, truth


def official_quality(upstream, settings, device, archive_dir):
    """One original Disease LP update with official data, model and metrics.

    We preserve original negative sampling, Fermi-Dirac loss and ROC/AP. The
    bounded driver preserves Slurm binding and saves/reloads the actual state;
    official train.py rewrites CUDA_VISIBLE_DEVICES and saves final weights next
    to best embeddings. Those driver behaviors are deliberately not invoked.
    """
    args = upstream.parser.parse_args([])
    for name, value in settings.items():
        if name != 'quality_steps':
            setattr(args, name, value)
    args.cuda = -1 if device.type == 'cpu' else 0; args.device = str(device)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    started = time.perf_counter()
    # Root is not published in the provenance record.
    source_data_root = Path(upstream.data.__file__).parents[1] / 'data' / args.dataset
    data = upstream.data.load_data(args, str(source_data_root))
    args.n_nodes, args.feat_dim = data['features'].shape
    args.nb_false_edges = len(data['train_edges_false']); args.nb_edges = len(data['train_edges'])
    model = upstream.LPModel(args).to(device)
    data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    losses = []
    for _ in range(settings['quality_steps']):
        check_deadline(); model.train(); optimizer.zero_grad(set_to_none=True)
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        train_metrics = model.compute_metrics(embeddings, data, 'train')
        loss = train_metrics['loss']; loss.backward()
        if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('official Disease training loss/gradients not finite')
        optimizer.step(); losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        points = model.encode(data['features'], data['adj_train_norm'])
        metrics = {split: model.compute_metrics(points, data, split) for split in ('val', 'test')}
        summary = {s: {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in row.items()}
                   for s, row in metrics.items()}
        reload = upstream.LPModel(args).to(device); reload.load_state_dict(model.state_dict(), strict=True); reload.eval()
        torch.testing.assert_close(reload.encode(data['features'], data['adj_train_norm']), points, atol=0, rtol=0)
    if any(not np.isfinite(v) for row in summary.values() for v in row.values()):
        raise ValueError('nonfinite official ROC/AP/loss')
    adj = data['adj_train_norm'].coalesce()
    archive = write_archive(archive_dir, 'official-disease-quality',
                            {'features': data['features'], 'adjacency_indices': adj.indices(), 'adjacency_values': adj.values(),
                             'model_state': model.state_dict(), 'embeddings': points,
                             'edge_sets': {k: data[k] for k in ('train_edges', 'train_edges_false', 'val_edges',
                                                              'val_edges_false', 'test_edges', 'test_edges_false')},
                             'metrics': summary})
    synchronize(device)
    return {'status': 'passed', 'scope': 'one-update original-task compatibility, not paper-result reproduction',
            'quality_steps': settings['quality_steps'], 'nodes': args.n_nodes, 'features': args.feat_dim,
            'training_loss': losses, 'metrics': summary, 'seconds': time.perf_counter() - started,
            'decoder': 'official symmetric squared distance/Fermi-Dirac', 'archive': archive}


def wordnet_quality(data, config, upstream, device, output, prepared_root):
    settings = config['wordnet_quality']; m = config['model']; torch.manual_seed(settings['seed'])
    model = MatureHGCN(upstream, m['input_dim'], m['hidden'], m['head_hidden'], m['c'], m['dropout'], device)
    features = data['features'].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    rng = torch.Generator().manual_seed(settings['seed'] + 1); rows = []
    for step in range(settings['quality_steps']):
        check_deadline(); synchronize(device); begin = time.perf_counter()
        chosen = torch.randperm(len(data['query_groups']), generator=rng)[:settings['batch_positives']]
        groups = data['query_groups'][chosen]; masked = mask_graph(data['neighbors'], groups[:, 0].tolist())
        plan = make_plan(masked, None, torch.Generator()); adjacency = plan.matrix(device=device)
        model.train(); optimizer.zero_grad(set_to_none=True)
        points = model.encode(features, [adjacency, adjacency])
        queries = groups.reshape(-1, 2).to(device); labels = data['labels'][chosen].reshape(-1).to(device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model.score(points, queries), labels)
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('finite all-parameter WordNet quality gradients required')
        optimizer.step(); synchronize(device)
        rows.append({'step': step + 1, 'loss': float(loss.detach()), 'seconds': time.perf_counter() - begin,
                     'target_mask_pairs': len(set(map(tuple, groups[:, 0].tolist()))),
                     'postmask_nonself_edges': int(plan.populations.sum())})
        atomic_json(output / 'wordnet-progress.json', {'status': 'engineering_only', 'steps': rows})
    model.eval(); check_deadline(); synchronize(device); begin = time.perf_counter()
    full_plan = make_plan(data['neighbors'], None, torch.Generator()); adjacency = full_plan.matrix(device=device)
    with torch.no_grad():
        full_points = model.encode(features, [adjacency, adjacency])
    synchronize(device); full_encoding_seconds = time.perf_counter() - begin
    # Measure forward cost only. No sampled hierarchy, ranking or effect readout
    # is inspected before the supervisor freezes the scientific registration.
    sample_cost = {}
    for fanout in (4, 8, 16):
        check_deadline(); begin = time.perf_counter(); generator = torch.Generator().manual_seed(9000 + fanout)
        plans = [make_plan(data['neighbors'], fanout, generator) for _ in range(2)]
        with torch.no_grad():
            sampled = model.encode(features, [p.matrix(device=device) for p in plans])
        if not torch.isfinite(sampled).all():
            raise ValueError('nonfinite sampled quality forward')
        synchronize(device)
        sample_cost[str(fanout)] = {'seconds': time.perf_counter() - begin,
                                   'directly_pruned_nodes': int((plans[0].populations > fanout).sum()),
                                   'nonself_selected_each_layer': [int(p.selected.sum()) for p in plans]}
    result = {'status': 'passed', 'scope': 'discarded three-step quality model; no baseline selection or mature hierarchy premise',
              'steps': rows, 'full_encoding_and_plan_seconds': full_encoding_seconds, 'sample_forward_cost': sample_cost,
              'train_files_opened': data['input_files_opened'], 'complete_valid_ranking': None}
    if device.type == 'cuda':
        check_deadline(); valid, truth = load_valid(prepared_root, data, config)
        ranking = filtered_parent_ranks(model, model.tangent(full_points), valid, truth,
                                        settings['candidate_chunk'], settings['ranking_max_seconds'])
        if ranking['status'] != 'complete':
            raise ValueError('cost gate requires complete original valid ranking; partial metric not admitted')
        if any(r['candidates'] != len(data['nodes']) - len(truth[r['child']]) for r in ranking['rows']):
            raise ValueError('complete filtered candidate denominator mismatch')
        result['complete_valid_ranking'] = {k: v for k, v in ranking.items() if k != 'rows'}
        result['ranking_archive'] = write_archive(output / 'arrays', 'wordnet-quality-ranking',
                                                  {'model_state': model.state_dict(), 'native_ball_points': full_points,
                                                   'queries': np.array(valid, dtype=np.int64), 'ranking': ranking,
                                                   'manifest_hash': data['manifest_hash'], 'valid_hash': data['valid_hash']})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--phase', choices=('cpu_quality', 'cuda_quality'), required=True)
    for name in ('source-commit', 'release-record', 'upstream-root', 'prepared-root', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--upstream-manifest')
    args = parser.parse_args(); output = Path(args.output)
    started = time.perf_counter()
    report = {'status': 'running', 'phase': args.phase, 'scope': 'engineering only', 'science_released': False}
    try:
        check_deadline(); config = json.loads(args.config.read_bytes()); validate_config(config)
        release = verify_release(config, args.phase, args.source_commit, args.release_record)
        before = source_hashes(); device = check_runtime(config, args.phase)
        upstream = load_upstream(args.upstream_root, args.upstream_manifest)
        report.update(release=release, upstream=upstream.identity, torch=torch.__version__, numpy=np.__version__,
                      python=platform.python_version(), device=str(device))
        report['fixture'] = run_fixture(upstream, device, output / 'arrays'); atomic_json(output / 'quality.json', report)
        check_deadline()
        report['official_task'] = official_quality(upstream, config['official_task'], device, output / 'arrays')
        atomic_json(output / 'quality.json', report); check_deadline()
        begin = time.perf_counter(); data = load_train(args.prepared_root, config)
        report['prepared_load_seconds'] = time.perf_counter() - begin
        report['wordnet'] = wordnet_quality(data, config, upstream, device, output, args.prepared_root)
        if source_hashes() != before:
            raise ValueError('source changed during quality worker')
        # Verify unmodified external code/data again after all official imports.
        load_upstream(args.upstream_root, args.upstream_manifest)
        report['status'] = 'passed'
    except Exception as error:
        report['status'] = 'failed'; report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None
        atomic_json(output / 'quality.json', report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

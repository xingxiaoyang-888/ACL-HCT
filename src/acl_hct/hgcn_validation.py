"""Bounded official training and fixed-weight sampling; no correction method."""
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .diagnostic_archive import read_archive, write_archive
from .hgcn_evidence import (array_hash, atomic_json, checkpoint_binding, compare_full, complete_ranking, deadline,
                           file_hash, hierarchy_evaluator, load_checkpoint, load_panel_design,
                           save_checkpoint, state_hash, validate_ranking)
from .hgcn_fixture import run_fixture
from .hgcn_quality import check_runtime, load_train, load_valid, synchronize
from .hgcn_sampling import make_plan, mask_graph
from .hgcn_upstream import load_upstream
from .hgcn_validation_entry import parser
from .hgcn_validation_registration import canonical, source_hashes, verify_release
from .mature_hgcn import MatureHGCN


def new_model(upstream, config, device):
    m = config['model']
    return MatureHGCN(upstream, m['input_dim'], m['hidden'], m['head_hidden'], m['c'], m['dropout'], device)


def binding(config, release, seed, step):
    return checkpoint_binding(config, release['source_commit'], seed, step)


def train_wordnet(data, config, upstream, device, output, prepared_root, release, seed):
    t = config['training']; np.random.seed(seed); torch.manual_seed(seed)
    model = new_model(upstream, config, device); initial_state_hash = state_hash(model.state_dict())
    features = data['features'].to(device); valid, truth = load_valid(prepared_root, data, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'])
    rng = torch.Generator().manual_seed(seed + 1); initial_rng = rng.get_state()
    full_plan = make_plan(data['neighbors'], None, torch.Generator()); full_adj = full_plan.matrix(device=device)
    history = []; batches = []; evaluations = []; best = None; best_mrr = -1.

    def evaluate(step):
        model.eval(); deadline()
        with torch.no_grad():
            points = model.encode(features, [full_adj, full_adj])
            ranks = complete_ranking(model, points, valid, truth, config['ranking'])
        rank_archive = write_archive(output / 'arrays', f'valid-step-{step}',
                                     {'step': step, 'queries': np.array(valid, dtype=np.int64), 'ranking': ranks})
        evaluations.append({'step': step, 'selection_eligible': step != 0,
                            'micro_mrr': ranks['query_micro_mrr'], 'ranking_archive': rank_archive})
        return points, ranks

    initial_points, initial_ranks = evaluate(0)
    initial_reference = write_archive(output / 'arrays', 'initial-full-reference',
                                      {'native_ball_points': initial_points, 'ranking': initial_ranks,
                                       'model_state_sha256': initial_state_hash})
    del initial_points, initial_ranks
    for step in range(1, t['steps'] + 1):
        deadline(); synchronize(device); started = time.perf_counter()
        chosen = torch.randperm(len(data['query_groups']), generator=rng)[:t['batch_positives']]
        groups = data['query_groups'][chosen]; batches.append(chosen.numpy().copy())
        masked = mask_graph(data['neighbors'], groups[:, 0].tolist())
        plan = make_plan(masked, None, torch.Generator()); adjacency = plan.matrix(device=device)
        model.train(); optimizer.zero_grad(set_to_none=True)
        points = model.encode(features, [adjacency, adjacency])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model.score(points, groups.reshape(-1, 2).to(device)), data['labels'][chosen].reshape(-1).to(device))
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('all-parameter finite training gradient gate failed')
        optimizer.step(); synchronize(device)
        history.append({'step': step, 'loss': float(loss.detach()), 'seconds': time.perf_counter() - started,
                        'postmask_nonself_messages': int(plan.populations.sum())})
        del points, adjacency, plan, masked
        if step in t['complete_valid_selection_steps']:
            points, ranks = evaluate(step)
            if ranks['query_micro_mrr'] > best_mrr:  # First strict maximum; never sampled selection.
                best_mrr = ranks['query_micro_mrr']; cp_binding = binding(config, release, seed, step)
                checkpoint = save_checkpoint(output, f'best-step-{step}.pt', model, cp_binding)
                reference = write_archive(output / 'arrays', f'best-full-step-{step}',
                                          {'binding': cp_binding, 'model_state_sha256': checkpoint['model_state_sha256'],
                                           'native_ball_points': points, 'ranking': ranks,
                                           'queries': np.array(valid, dtype=np.int64), 'full_plan': full_plan.archive()})
                best = {'step': step, 'micro_mrr': best_mrr, 'checkpoint': checkpoint, 'reference': reference}
            del points, ranks
        if step % 32 == 0 or step == t['steps']:
            atomic_json(output / 'training-progress.json', {'status': 'running', 'seed': seed,
                        'completed_steps': step, 'history': history, 'evaluations': evaluations, 'best': best})
    if best is None or [e['step'] for e in evaluations if e['selection_eligible']] != t['complete_valid_selection_steps']:
        raise ValueError('all scheduled complete validation selections required')
    last = save_checkpoint(output, 'last.pt', model, binding(config, release, seed, t['steps']),
                           {'optimizer_state': optimizer.state_dict(), 'batch_rng_state': rng.get_state(),
                            'torch_rng_state': torch.get_rng_state(),
                            'cuda_rng_state': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None})
    batch_history = write_archive(output / 'arrays', 'training-history',
                                  {'batch_indices': np.stack(batches), 'initial_batch_rng_state': initial_rng,
                                   'final_batch_rng_state': rng.get_state(), 'steps': history,
                                   'sampling': 'CPU randperm of fixed train query groups, seed=training seed+1'})
    loaded = load_checkpoint(output, best['checkpoint'], binding(config, release, seed, best['step']))
    model.load_state_dict(loaded['model_state'], strict=True); model.eval()
    reference = read_archive(output / 'arrays', best['reference'])
    with torch.no_grad():
        points = model.encode(features, [full_adj, full_adj])
        ranks = complete_ranking(model, points, valid, truth, config['ranking'])
    compare_full(points, ranks, reference)
    view, panel = load_panel_design(prepared_root, config)
    evaluator, baseline = hierarchy_evaluator(view, panel, points.detach().cpu().numpy(), config)
    hierarchy_archive = write_archive(output / 'arrays', 'best-full-hierarchy',
                                      {'native_full_points_sha256': array_hash(points),
                                       'model_state_sha256': loaded['model_state_sha256'],
                                       'panel_design': evaluator.archive_design(), 'hierarchy': baseline})
    full_order = baseline['direct']['metrics']['score']
    return {'status': 'complete', 'seed': seed, 'completed_steps': t['steps'], 'initial_state_sha256': initial_state_hash,
            'initial_reference': initial_reference, 'evaluations': evaluations, 'best': best, 'last': last,
            'batch_history': batch_history, 'matching_best_serialized_reload_full_rank': 'passed',
            'train_files_opened': data['input_files_opened'], 'selection': t['selection'],
            'valid_hash': data['valid_hash'], 'query_count': len(valid), 'sampling_effects_inspected': False,
            'full_baseline': {'direct_order': full_order, 'distant_order': baseline['distant']['metrics']['score'],
                              'micro_mrr': ranks['query_micro_mrr'], 'hierarchy_archive': hierarchy_archive,
                              'positive_order_necessary_premise': full_order is not None and full_order > .5,
                              'interpretation': '>0.5 order is necessary, not sufficient; completion and tiny significance do not establish maturity or hierarchy collapse'},
            'learning_state': {'initial_valid_micro_mrr': evaluations[0]['micro_mrr'], 'best_valid_micro_mrr': best_mrr,
                               'last_valid_micro_mrr': evaluations[-1]['micro_mrr'], 'best_step': best['step'],
                               'mean_first_32_losses': float(np.mean([r['loss'] for r in history[:32]])),
                               'mean_last_32_losses': float(np.mean([r['loss'] for r in history[-32:]])),
                               'requires_independent_review_before_sampling': True}}


def training_input(training_run, config, release, seed):
    path = Path(training_run)
    if file_hash(path) != release['inputs']['training_run_sha256']:
        raise ValueError('reviewed training report hash mismatch')
    run = json.loads(path.read_bytes())
    if (run['status'] != 'complete' or run['phase'] != 'train' or run['config_sha256'] != canonical(config)
            or run['source_commit'] != release['source_commit'] or run['result']['seed'] != seed
            or run['result']['completed_steps'] != config['training']['steps']
            or run['result']['best']['checkpoint']['sha256'] != release['inputs']['best_checkpoint_sha256']):
        raise ValueError('complete accepted matching baseline required')
    return path.parent, run['result']


def evaluate_shard(data, config, upstream, device, output, prepared_root, training_run, release, seed, start):
    root, trained = training_input(training_run, config, release, seed); best = trained['best']
    model = new_model(upstream, config, device)
    cp = load_checkpoint(root, best['checkpoint'], binding(config, release, seed, best['step']))
    model.load_state_dict(cp['model_state'], strict=True); model.eval(); before = state_hash(model.state_dict())
    reference = read_archive(root / 'arrays', best['reference'])
    if (reference['binding'] != cp['binding'] or reference['model_state_sha256'] != before):
        raise ValueError('matching best full reference/weight identity mismatch')
    valid, truth = load_valid(prepared_root, data, config); validate_ranking(reference['ranking'], valid, truth, len(data['nodes']))
    features = data['features'].to(device); full_plan = make_plan(data['neighbors'], None, torch.Generator())
    full_adj = full_plan.matrix(device=device)
    with torch.no_grad():
        full = model.encode(features, [full_adj, full_adj])
        full_ranking = complete_ranking(model, full, valid, truth, config['ranking'])
    compare_full(full, full_ranking, reference)
    view, panel = load_panel_design(prepared_root, config)
    evaluator, baseline = hierarchy_evaluator(view, panel, full.detach().cpu().numpy(), config)
    design = write_archive(output / 'arrays', 'full-control',
                           {'binding': cp['binding'], 'model_state_sha256': before, 'native_ball_points': full,
                            'ranking': full_ranking, 'panel_design': evaluator.archive_design(), 'hierarchy': baseline,
                            'queries': np.array(valid, dtype=np.int64), 'full_plan': full_plan.archive()})
    records = []
    for repeat in range(start, start + 4):
        for fanout in (4, 8, 16):
            deadline(); graph_seed = 202609190000 + seed * 10000 + repeat * 100 + fanout
            rng = torch.Generator().manual_seed(graph_seed); rng_before = rng.get_state()
            plans = [make_plan(data['neighbors'], fanout, rng) for _ in range(2)]
            with torch.no_grad():
                points = model.encode(features, [p.matrix(device=device) for p in plans])
                ranking = complete_ranking(model, points, valid, truth, config['ranking'])
            native = points.detach().cpu().numpy(); hierarchy = evaluator.evaluate(native)
            artifact = write_archive(output / 'arrays', f'sample-r{repeat}-f{fanout}',
                                     {'seed': seed, 'repeat': repeat, 'fanout': fanout, 'graph_seed': graph_seed,
                                      'native_ball_points': native, 'plans': [p.archive() for p in plans],
                                      'rng_before': rng_before, 'rng_after': rng.get_state(),
                                      'ranking': ranking, 'hierarchy': hierarchy,
                                      'checkpoint_sha256': best['checkpoint']['sha256'], 'model_state_sha256': before,
                                      'full_native_points_sha256': array_hash(full)})
            records.append({'repeat': repeat, 'fanout': fanout, 'artifact': artifact,
                            'micro_mrr': ranking['query_micro_mrr'], 'direct_order': hierarchy['direct']['metrics']['score']})
            atomic_json(output / 'sampling-progress.json', {'status': 'running', 'seed': seed,
                                                           'repeat_start': start, 'samples': records})
            del points, native, hierarchy, plans, ranking
    if len(records) != 12 or state_hash(model.state_dict()) != before:
        raise ValueError('complete shard and unchanged model state required')
    return {'status': 'complete', 'seed': seed, 'repeat_start': start, 'repeats': list(range(start, start + 4)),
            'full_control': design, 'full_replay': 'exact native points and ranks passed once at shard start',
            'best_checkpoint': best['checkpoint'], 'samples': records,
            'full_control_inference_unit': 'shared deterministic reference, no duplicate independent F observations'}


def fixture(upstream, device, output, config, release):
    # Official mathematical/gradient/state comparison remains the accepted fixture.
    official = run_fixture(upstream, device, output / 'arrays')
    from .hgcn_geometry import distance_ball_fp64, log_ball_orthonormal_fp64, mean_error_statistics
    p = np.array([[0., 0.], [.996, 0.], [.2, -.3]], dtype=np.float32)
    q = np.array([[.1, 0.], [.99, .01], [.2, -.3]], dtype=np.float32)
    distance = distance_ball_fp64(p, q); error = log_ball_orthonormal_fp64(p, q)
    np.testing.assert_allclose(np.linalg.norm(error, axis=1), distance, atol=1e-12, rtol=1e-12)
    moments = mean_error_statistics(np.stack([error, -error]).astype(np.float64))
    if not np.all(moments['bias_squared_unbiased_per_node'] <= 0):
        raise ValueError('signed finite-repeat bias estimator fixture failed')
    archive = write_archive(output / 'arrays', 'science-native-geometry-fixture',
                            {'p': p, 'q': q, 'distance': distance, 'nodewise_log': error, 'moments': moments})
    integration = integration_fixture(upstream, device, output / 'artificial-integration', config, release)
    return {'status': 'passed', 'scope': 'artificial engineering only; no prepared scientific model/effect inspected',
            'official_fixture': official, 'native_geometry_archive': archive, 'integration_fixture': integration}


def integration_fixture(upstream, device, output, config, release):
    """Exercise exact drivers on two four-update random-feature star models."""
    from .hgcn_analysis import analyze
    from .hgcn_official import train_official
    from .hgcn_synthetic import miniature_inputs
    output.mkdir(); root = output / 'inputs'; miniature = miniature_inputs(root, config)
    atomic_json(output / 'ARTIFICIAL-config.json', miniature)
    data = load_train(root, miniature)
    index = {'protocol': config['protocol'], 'config_sha256': canonical(miniature), 'training': [], 'evaluation': []}

    def save_run(directory, phase, result):
        path = directory / 'run.json'
        atomic_json(path, {'status': 'complete', 'phase': phase, 'source_commit': release['source_commit'],
                          'config_sha256': canonical(miniature), 'scope': 'ARTIFICIAL engineering fixture only', 'result': result})
        return {'run_file': str(path.resolve()), 'sha256': file_hash(path)}

    for seed in (11, 23):
        directory = output / f'train-{seed}'; directory.mkdir()
        result = train_wordnet(data, miniature, upstream, device, directory, root, release, seed)
        training_item = save_run(directory, 'train', result); index['training'].append(training_item)
        for start in (0, 4, 8, 12):
            shard_dir = output / f'eval-{seed}-{start}'; shard_dir.mkdir()
            fixture_release = {**release, 'inputs': {'training_run_sha256': training_item['sha256'],
                                                   'best_checkpoint_sha256': result['best']['checkpoint']['sha256']}}
            shard = evaluate_shard(data, miniature, upstream, device, shard_dir, root, training_item['run_file'],
                                   fixture_release, seed, start)
            index['evaluation'].append(save_run(shard_dir, 'eval', shard))
    directory = output / 'official-three-epoch'; directory.mkdir()
    official = train_official(upstream, miniature, device, directory, release)
    index['official'] = save_run(directory, 'official', official)
    index_path = output / 'artifact-index.json'; atomic_json(index_path, index)
    analysis_dir = output / 'analysis'; analysis_dir.mkdir()
    analysis = analyze(index_path, root, miniature,
                       {**release, 'inputs': {'artifact_index_sha256': file_hash(index_path)}}, analysis_dir)
    if analysis['replayed_sample_arrays'] != 96 or len(analysis['primary_family']) != 12:
        raise ValueError('complete artificial archive/statistics integration failed')
    item = save_run(analysis_dir, 'analyze', analysis)
    return {'status': 'passed', 'scope': 'random-feature forty-node graph, two four-update models, official three-epoch QA only',
            'complete_samples_replayed': 96, 'primary_comparisons_replayed': 12,
            'analysis_run_sha256': item['sha256'], 'artificial_config_sha256': canonical(miniature)}


def main():
    args = parser().parse_args(); output = Path(args.output); started = time.perf_counter()
    report = {'status': 'running', 'phase': args.phase, 'source_commit': args.source_commit}
    try:
        deadline(); config = json.loads(args.config.read_bytes())
        released = verify_release(config, args.phase, args.source_commit, args.release_record, args.seed, args.repeat_start)
        before = source_hashes(); device = check_runtime(config, 'cpu_quality' if args.phase in ('cpu_fixture', 'analyze') else 'cuda_quality')
        report.update(config_sha256=canonical(config), release=released, torch=torch.__version__,
                      numpy=np.__version__, python=platform.python_version(), device=str(device))
        if args.phase == 'analyze':
            from .hgcn_analysis import analyze
            result = analyze(args.artifact_index, args.prepared_root, config, released, output)
        else:
            upstream = load_upstream(args.upstream_root, args.upstream_manifest); report['upstream'] = upstream.identity
            if args.phase.endswith('fixture'):
                result = fixture(upstream, device, output, config, released)
            elif args.phase == 'official':
                from .hgcn_official import train_official
                result = train_official(upstream, config, device, output, released)
            else:
                data = load_train(args.prepared_root, config)
                if args.phase == 'train':
                    result = train_wordnet(data, config, upstream, device, output, args.prepared_root, released, args.seed)
                else:
                    result = evaluate_shard(data, config, upstream, device, output, args.prepared_root,
                                            args.training_run, released, args.seed, args.repeat_start)
            load_upstream(args.upstream_root, args.upstream_manifest)
        if source_hashes() != before:
            raise ValueError('source changed during worker')
        report.update(status='complete', result=result)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}'); raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None
        atomic_json(output / 'run.json', report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

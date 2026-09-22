"""Bounded Stage A frozen-HGCN correction worker; invoked only by its release entry."""
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from .development_view import build_view
from .diagnostic_archive import read_archive, write_archive
from .hgcn_evidence import (checkpoint_binding, complete_ranking, deadline,
                            hierarchy_evaluator, load_checkpoint,
                            load_panel_design, save_checkpoint, state_hash)
from .hgcn_quality import atomic_json, load_train, load_valid, synchronize
from .hgcn_registration import canonical
from .hgcn_replay import ORIGINAL_SOURCE, historical_training, qualify_full, require_qualified
from .hgcn_rtsc_protocol import NamedStreams, choose_checkpoint
from .hgcn_rtsc_stage_a import FrozenCorrectedHGCN
from .hgcn_sampling import make_plan, mask_graph
from .mature_hgcn import MatureHGCN
from .hgcn_tangent_correction import (ball_distance_from_anchor,
                                      relation_gap_loss, training_relation_weights)


def small_quality(upstream, settings, device, output):
    """Original-runtime small graph identity, gradients and nonzero response."""
    torch.manual_seed(41)
    base = MatureHGCN(upstream, 3, 4, 5, 1., 0., device).eval()
    before = state_hash(base.state_dict())
    model = FrozenCorrectedHGCN(base, hidden=16, tau=.05, eps=1e-8, chunk_edges=3)
    features = (torch.randn(8, 3, device=device) * .1)
    neighbors = [[j for j in range(8) if j != i] for i in range(8)]
    plans = [make_plan(neighbors, 4, _generator(i + 4)) for i in (0, 1)]
    original = base.encode(features, [p.matrix(device=device) for p in plans])
    points, diagnostics = model.encode(features, plans)
    if not torch.equal(original, points):
        raise ValueError('zero initialized module changed original HGCN output')
    query = torch.tensor([[0, 1], [2, 3], [4, 5]], device=device)
    loss = model.score(points, query).square().mean()
    loss.backward()
    gradients = [float(sum(p.grad.detach().abs().sum() for p in m.coefficients[-1].parameters()))
                 for m in model.corrections]
    if any(g <= 0 or not np.isfinite(g) for g in gradients):
        raise ValueError('both output coefficient layers require learnable gradients')
    with torch.no_grad():
        for correction in model.corrections:
            correction.coefficients[-1].bias.copy_(torch.tensor([.2, -.1, .1], device=device))
        changed, measured = model.encode(features, plans)
    response = float((changed - original).abs().max())
    if response <= 1e-5 or not torch.isfinite(changed).all() or state_hash(base.state_dict()) != before:
        raise ValueError('nonzero response, finite outputs and frozen backbone required')
    result = {'status': 'passed', 'device': str(device), 'zero_output_exact': True,
              'two_layer_gradient_l1': gradients, 'nonzero_response_max_abs': response,
              'diagnostics': [{k: float(v.detach()) if torch.is_tensor(v) else v
                               for k, v in d.items()} for d in measured],
              'base_state_unchanged': True}
    atomic_json(output / 'quality.json', result)
    return result


def _generator(value):
    return torch.Generator().manual_seed(value)


def _model(upstream, original, device):
    m = original['model']
    return MatureHGCN(upstream, m['input_dim'], m['hidden'], m['head_hidden'],
                      m['c'], m['dropout'], device)


def _correction(base, settings, init_seed):
    # Modules start on CPU; fork_rng prevents this initialization from changing
    # any sampling or later global RNG state. Both arms use the same seed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        m = settings['module']
        return FrozenCorrectedHGCN(base, hidden=m['hidden'], tau=m['tau'],
                                   eps=m['eps'], chunk_edges=m['chunk_edges'])


def _correction_binding(settings, policy, seed, arm, step):
    if seed not in settings['seeds'] or arm not in settings['arms'] or step not in settings['selection']['steps']:
        raise ValueError('registered correction checkpoint identity required')
    return {'protocol': settings['protocol'], 'config_sha256': canonical(settings),
            'seed': seed, 'arm': arm, 'step': step,
            'original_checkpoint_sha256': policy['origins'][str(seed)]['best_checkpoint_sha256']}


def _train_root(data):
    ids = data['nodes']
    positives = data['query_groups'][:, 0].tolist()
    edges = [(ids[a], ids[b]) for a, b in positives]
    view = build_view(ids, data['neighbors'], edges, [], [])
    if view.root is None:
        raise ValueError('train-visible directed taxonomy has no semantic root')
    return ids.index(view.root), view.root, view.metadata['h_dev_hash']


def _plans(neighbors, streams, phase, seed, *, step=None, fanout=4, repeat=None):
    return [make_plan(neighbors, fanout, _generator(streams.seed(
        phase, seed, step=step, fanout=fanout, repeat=repeat, layer=layer)))
            for layer in (0, 1)]


def _metrics(base, points, evaluator, valid, truth, ranking_settings):
    ranking = complete_ranking(base, points, valid, truth, ranking_settings)
    hierarchy = evaluator.evaluate(points.detach().cpu().numpy())
    return {'micro_mrr': ranking['query_micro_mrr'],
            'direct_order': hierarchy['direct']['metrics']['score']}, ranking, hierarchy


def _probe_readout(model, plans, features, evaluator, valid, truth,
                   settings, output, name):
    model.eval()
    with torch.no_grad():
        points, diagnostics = model.encode(features, plans)
        metrics, ranking, hierarchy = _metrics(model.base, points, evaluator,
                                               valid, truth, settings['ranking'])
    archive = write_archive(output / 'arrays', name,
                            {'points': points, 'plans': [p.archive() for p in plans],
                             'ranking': ranking, 'hierarchy': hierarchy,
                             'diagnostics': [{k: float(v.detach()) if torch.is_tensor(v) else v
                                              for k, v in d.items()} for d in diagnostics]})
    return {'metrics': metrics, 'ranking_seconds': ranking['elapsed_seconds'],
            'archive': archive}


def original_inputs(original_run, original, policy, seed, upstream, device, prepared_root):
    """Load only accepted original best and fixed F; retain exact lineage."""
    root, trained = historical_training(original_run, original, policy, seed)
    best = trained['best']
    if best['step'] != policy['origins'][str(seed)]['best_step']:
        raise ValueError('wrong original best step')
    binding = checkpoint_binding(original, ORIGINAL_SOURCE, seed, best['step'])
    payload = load_checkpoint(root, best['checkpoint'], binding)
    base = _model(upstream, original, device)
    base.load_state_dict(payload['model_state'], strict=True)
    base.eval()
    before = state_hash(base.state_dict())
    reference = read_archive(root / 'arrays', best['reference'])
    if reference['binding'] != binding or reference['model_state_sha256'] != before:
        raise ValueError('original best weights and native F reference mismatch')
    data = load_train(prepared_root, original)
    return data, base, reference, binding, before, best


def full_control(data, base, reference, binding, state_sha, original, policy,
                 prepared_root, output, device):
    """Numerically qualify the exact frozen F on this runtime before sampling."""
    valid, truth = load_valid(prepared_root, data, original)
    view, panel = load_panel_design(prepared_root, original)
    evaluator, baseline = hierarchy_evaluator(
        view, panel, reference['native_ball_points'], original)
    plan = make_plan(data['neighbors'], None, _generator(0))
    features = data['features'].to(device)
    with torch.no_grad():
        full = base.encode(features, [plan.matrix(device=device)] * 2)
        ranking = complete_ranking(base, full, valid, truth, original['ranking'])
    qualification = qualify_full(full, ranking, reference, evaluator, original,
                                 policy, valid, truth, state_sha, binding, plan.archive())
    archive = write_archive(output / 'arrays', 'original-full-control',
                            {'qualification': qualification, 'native_ball_points': full,
                             'ranking': ranking, 'hierarchy': baseline,
                             'model_state_sha256': state_sha, 'full_plan': plan.archive()})
    atomic_json(output / 'full-qualification.json', {'qualification': qualification,
                                                      'archive': archive})
    require_qualified(qualification)
    return features, valid, truth, evaluator, qualification


def _development_plans(data, base, features, evaluator, valid, truth,
                       settings, streams, seed, output, device):
    """Freeze S for two matching f4 graph plans; never use an old 96-repeat plan."""
    frozen = []
    for repeat in range(settings['selection']['repeats']):
        deadline()
        plans = _plans(data['neighbors'], streams, 'selection_layer', seed,
                       fanout=4, repeat=repeat)
        with torch.no_grad():
            points = base.encode(features, [p.matrix(device=device) for p in plans])
            metrics, ranking, hierarchy = _metrics(base, points, evaluator, valid,
                                                     truth, settings['ranking'])
        archive = write_archive(output / 'arrays', f'dev-S-r{repeat}',
                                {'points': points, 'plans': [p.archive() for p in plans],
                                 'ranking': ranking, 'hierarchy': hierarchy})
        frozen.append({'plans': plans, 'metrics': metrics, 'archive': archive})
    return frozen


def _evaluate_checkpoint(model, step, frozen, features, evaluator, valid, truth,
                         settings, output):
    model.eval()
    rows = []
    for repeat, original in enumerate(frozen):
        deadline()
        with torch.no_grad():
            points, diagnostics = model.encode(features, original['plans'])
            metrics, ranking, hierarchy = _metrics(model.base, points, evaluator,
                                                     valid, truth, settings['ranking'])
        archive = write_archive(output / 'arrays', f'dev-C-step{step}-r{repeat}',
                                {'points': points, 'ranking': ranking, 'hierarchy': hierarchy,
                                 'plans': [p.archive() for p in original['plans']],
                                 'diagnostics': [{k: float(v.detach()) if torch.is_tensor(v) else v
                                                  for k, v in d.items()} for d in diagnostics]})
        rows.append({'repeat': repeat, 'micro_mrr': metrics['micro_mrr'],
                     'direct_order': metrics['direct_order'],
                     'direct_order_delta': metrics['direct_order'] - original['metrics']['direct_order'],
                     'S_archive': original['archive'], 'C_archive': archive})
    return {'step': step, 'micro_mrr': [r['micro_mrr'] for r in rows],
            'direct_order_delta': [r['direct_order_delta'] for r in rows],
            'repeats': rows}


def train_arm(data, base, reference, binding, state_sha, original, policy,
              prepared_root, settings, seed, arm, output, device, *, probe_steps=None):
    if seed not in settings['seeds'] or arm not in settings['arms']:
        raise ValueError('registered seed and training arm required')
    if probe_steps is not None and (type(probe_steps) is not int or not 1 <= probe_steps <= 32):
        raise ValueError('bounded discarded probe must use 1..32 updates')
    streams = NamedStreams()
    features, valid, truth, evaluator, qualification = full_control(
        data, base, reference, binding, state_sha, original, policy,
        prepared_root, output, device)
    init_seed = streams.seed('module_init', seed)
    model = _correction(base, settings, init_seed)
    if not torch.equal(base.curvature, model.base.curvature):
        raise ValueError('curvature changed')
    root, root_id, train_view_hash = _train_root(data)
    positives = data['query_groups'][:, 0].to(torch.long)
    relation_weights = training_relation_weights(positives, len(data['nodes']))
    optimizer = torch.optim.Adam(model.trainable_parameters(),
                                 lr=settings['training']['lr'], weight_decay=0.)
    frozen = None if probe_steps is not None else _development_plans(
        data, base, features, evaluator, valid, truth, settings, streams, seed,
        output, device)
    probe_plans = (_plans(data['neighbors'], streams, 'selection_layer', seed,
                          fanout=4, repeat=0) if probe_steps is not None else None)
    probe_readouts = ([_probe_readout(model, probe_plans, features, evaluator,
                                     valid, truth, settings, output, 'probe-step0')]
                      if probe_plans is not None else None)
    evaluations = []
    checkpoints = {}
    if frozen is not None:
        evaluations.append(_evaluate_checkpoint(model, 0, frozen, features, evaluator,
                                                valid, truth, settings, output))
        checkpoints[0] = save_checkpoint(output, 'correction-step-0.pt', model,
                                         _correction_binding(settings, policy, seed, arm, 0))
    history = []
    steps = probe_steps if probe_steps is not None else settings['training']['steps']
    for step in range(1, steps + 1):
        deadline(); synchronize(device); start = time.perf_counter()
        batch_seed = streams.seed('train_batch', seed, step=step,
                                  fanout=settings['training']['fanout'])
        chosen = torch.randperm(len(data['query_groups']), generator=_generator(batch_seed))[:128]
        groups = data['query_groups'][chosen]
        masked = mask_graph(data['neighbors'], groups[:, 0].tolist())
        plans = _plans(masked, streams, 'train_layer', seed, step=step, fanout=4)
        model.train(); optimizer.zero_grad(set_to_none=True)
        points, diagnostics = model.encode(features, plans)
        task = torch.nn.functional.binary_cross_entropy_with_logits(
            model.score(points, groups.reshape(-1, 2).to(device)),
            data['labels'][chosen].reshape(-1).to(device))
        step_penalty = model.step_penalty(diagnostics)
        relation = (relation_gap_loss(points, groups[:, 0].to(device),
                                      relation_weights[chosen].to(device), root,
                                      settings['module']['margin'], float(base.curvature.item()))
                    if arm == 'task_relation' else points.sum() * 0)
        loss = task + settings['training']['step_penalty_weight'] * step_penalty + settings['arms'][arm] * relation
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training loss')
        loss.backward()
        params = list(model.trainable_parameters())
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
            raise ValueError('missing or nonfinite module gradient')
        layer_grads = [float(torch.linalg.vector_norm(torch.stack(
            [torch.linalg.vector_norm(p.grad) for p in module.parameters()])))
                       for module in model.corrections]
        norm = float(torch.nn.utils.clip_grad_norm_(params, settings['training']['grad_clip_norm']))
        previous = [p.detach().clone() for p in params]
        optimizer.step(); synchronize(device)
        update_norm = float(torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(p.detach() - old) for p, old in zip(params, previous)])))
        if not np.isfinite(update_norm) or any(not torch.isfinite(p).all() for p in params):
            raise ValueError('nonfinite correction parameter after optimizer update')
        if state_hash(base.state_dict()) != state_sha:
            raise ValueError('frozen original HGCN state changed')
        row = {'step': step, 'task_loss': float(task.detach()),
               'relation_loss': float(relation.detach()),
               'step_penalty': float(step_penalty.detach()), 'total_loss': float(loss.detach()),
               'module_grad_norm_before_clip': norm,
               'layer_grad_norm_before_clip': layer_grads,
               'gradient_clipped': norm > settings['training']['grad_clip_norm'],
               'module_update_norm': update_norm,
               'batch_seed63': batch_seed, 'batch_indices': chosen.tolist(),
               'layer_plan_seed63': [streams.seed('train_layer', seed, step=step, fanout=4, layer=i)
                                     for i in (0, 1)],
               'layer_plan_graph_hash': [p.graph_hash for p in plans],
               'diagnostics': [{k: float(v.detach()) if torch.is_tensor(v) else v
                                for k, v in d.items()} for d in diagnostics],
               'seconds': time.perf_counter() - start}
        if probe_steps is not None and step in (1, steps):
            with torch.no_grad():
                anchor = points[root:root + 1].detach().expand(len(groups), -1)
                batch_positive = groups[:, 0].to(device)
                parent_radius = ball_distance_from_anchor(anchor, points[batch_positive[:, 0]])
                child_radius = ball_distance_from_anchor(anchor, points[batch_positive[:, 1]])
                non_root = ((batch_positive[:, 0] != root) & (batch_positive[:, 1] != root))
                row['relation_geometry'] = {
                    'parent_near_zero': int((parent_radius <= settings['module']['eps']).sum()),
                    'child_near_zero': int((child_radius <= settings['module']['eps']).sum()),
                    'near_zero_gap': int(((child_radius - parent_radius).abs() <= settings['module']['eps']).sum()),
                    'batch_positive_count': len(groups),
                    'non_root_positive_count': int(non_root.sum()),
                    'non_root_parent_near_zero': int(((parent_radius <= settings['module']['eps']) & non_root).sum()),
                    'non_root_child_near_zero': int(((child_radius <= settings['module']['eps']) & non_root).sum()),
                    'non_root_near_zero_gap': int((((child_radius - parent_radius).abs() <= settings['module']['eps']) & non_root).sum())}
        history.append(row)
        if frozen is not None and step in settings['selection']['steps']:
            evaluations.append(_evaluate_checkpoint(model, step, frozen, features,
                                                    evaluator, valid, truth, settings, output))
            checkpoints[step] = save_checkpoint(output, f'correction-step-{step}.pt', model,
                                                _correction_binding(settings, policy, seed, arm, step))
        if step % 16 == 0 or step == steps:
            atomic_json(output / 'progress.json', {'status': 'running', 'seed': seed,
                                                    'arm': arm, 'completed_steps': step,
                                                    'history': history, 'evaluations': evaluations})
        del points, plans, masked
    selected = None if frozen is None else choose_checkpoint(evaluations,
                settings['selection']['direct_order_delta_floor'])
    result = {'status': 'probe_complete' if probe_steps is not None else 'complete',
              'seed': seed, 'arm': arm, 'completed_steps': steps,
              'original_best_binding': binding, 'original_best_state_sha256': state_sha,
              'original_F_qualification': qualification,
              'train_root_id': root_id, 'train_root_index': root,
              'train_view_hash': train_view_hash, 'root_source': 'train positives only',
              'module_init_seed63': init_seed, 'history': history,
              'evaluations': evaluations, 'selected': selected,
              'selected_checkpoint': checkpoints[selected['step']] if selected else None,
              'checkpoints': checkpoints, 'rng_manifest': streams.manifest(),
              'base_state_unchanged': state_hash(base.state_dict()) == state_sha,
              'probe_weights_discarded': probe_steps is not None,
              'valid_used_only_for_selection': probe_steps is None}
    if probe_steps is not None:
        probe_readouts.append(_probe_readout(model, probe_plans, features,
                                            evaluator, valid, truth, settings,
                                            output, f'probe-step{steps}'))
        result['probe_complete_sampled_ranking'] = probe_readouts
        result['peak_allocated_bytes'] = (torch.cuda.max_memory_allocated()
                                          if device.type == 'cuda' else None)
        result['peak_reserved_bytes'] = (torch.cuda.max_memory_reserved()
                                         if device.type == 'cuda' else None)
    atomic_json(output / 'result.json', result)
    return result


def final_shard(data, base, reference, binding, state_sha, original, policy,
                prepared_root, settings, seed, repeat_start, output, device, arm_inputs):
    """Paired final graph plans, independent of old and selection streams."""
    streams = NamedStreams()
    features, valid, truth, evaluator, qualification = full_control(
        data, base, reference, binding, state_sha, original, policy,
        prepared_root, output, device)
    models = {}
    selections = {}
    for arm in ('task_only', 'task_relation'):
        run_dir, run = arm_inputs[arm]
        if (run['status'] != 'complete' or run['seed'] != seed or run['arm'] != arm
                or run['completed_steps'] != 1024 or run['original_best_binding'] != binding
                or run['original_best_state_sha256'] != state_sha
                or run['selected'] is None or run['selected_checkpoint'] is None
                or run['base_state_unchanged'] is not True):
            raise ValueError('complete selected Stage A arm and matching frozen base required')
        chosen = run['selected_checkpoint']
        expected_binding = _correction_binding(settings, policy, seed, arm,
                                               run['selected']['step'])
        cp = load_checkpoint(run_dir, chosen, expected_binding)
        model = _correction(copy.deepcopy(base), settings, streams.seed('module_init', seed))
        model.load_state_dict(cp['model_state'], strict=True)
        model.eval()
        if state_hash(model.base.state_dict()) != state_sha:
            raise ValueError('selected correction checkpoint changed frozen HGCN')
        models[arm] = model
        selections[arm] = {'step': run['selected']['step'], 'checkpoint': chosen}
    full_plan = make_plan(data['neighbors'], None, _generator(0))
    full_qualifications = {}
    with torch.no_grad():
        for arm, model in models.items():
            corrected_full, diagnostics = model.encode(features, [full_plan] * 2)
            if any(d['eligible_nodes'] for d in diagnostics):
                raise ValueError('module must be a local identity on full adjacency')
            full_ranking = complete_ranking(base, corrected_full, valid, truth,
                                            settings['ranking'])
            qualification_arm = qualify_full(
                corrected_full, full_ranking, reference, evaluator, original,
                policy, valid, truth, state_sha, binding, full_plan.archive())
            archive = write_archive(output / 'arrays', f'full-control-{arm}',
                                    {'points': corrected_full, 'ranking': full_ranking,
                                     'qualification': qualification_arm,
                                     'diagnostics': diagnostics})
            atomic_json(output / f'full-qualification-{arm}.json',
                        {'qualification': qualification_arm, 'archive': archive})
            require_qualified(qualification_arm)
            full_qualifications[arm] = qualification_arm
    if repeat_start not in (0, 4, 8, 12):
        raise ValueError('registered final four-repeat shard required')
    rows = []
    for fanout in settings['final']['fanouts']:
        for repeat in range(repeat_start, repeat_start + settings['final']['shard_repeats']):
            deadline(); plans = _plans(data['neighbors'], streams, 'final_layer', seed,
                                       fanout=fanout, repeat=repeat)
            for arm in ('S', 'task_only', 'task_relation'):
                deadline()
                with torch.no_grad():
                    if arm == 'S':
                        points = base.encode(features, [p.matrix(device=device) for p in plans])
                        diagnostics = None
                    else:
                        points, diagnostics = models[arm].encode(features, plans)
                    metrics, ranking, hierarchy = _metrics(base, points, evaluator,
                                                             valid, truth, settings['ranking'])
                archive = write_archive(output / 'arrays', f'final-seed{seed}-f{fanout}-r{repeat}-{arm}',
                                        {'points': points, 'plans': [p.archive() for p in plans],
                                         'ranking': ranking, 'hierarchy': hierarchy,
                                         'diagnostics': None if diagnostics is None else [
                                             {k: float(v.detach()) if torch.is_tensor(v) else v
                                              for k, v in d.items()} for d in diagnostics]})
                rows.append({'seed': seed, 'fanout': fanout, 'repeat': repeat, 'arm': arm,
                             **metrics, 'archive': archive})
            atomic_json(output / 'progress.json', {'status': 'running', 'seed': seed,
                                                    'completed_conditions': len(rows), 'rows': rows})
    if len(rows) != 3 * len(settings['final']['fanouts']) * settings['final']['shard_repeats']:
        raise ValueError('all paired final conditions required')
    result = {'status': 'complete', 'seed': seed, 'repeat_start': repeat_start,
              'rows': rows, 'selected': selections,
              'fixed_F': {'micro_mrr': reference['ranking']['query_micro_mrr'],
                          'direct_order': evaluator.evaluate(reference['native_ball_points'])['direct']['metrics']['score']},
              'original_F_qualification': qualification,
              'full_module_identity': 'local k=N identity; independent full forward qualified to fixed F',
              'full_arm_qualifications': full_qualifications,
              'rng_manifest': streams.manifest(),
              'base_state_unchanged': state_hash(base.state_dict()) == state_sha}
    atomic_json(output / 'result.json', result)
    return result


def main():
    from .hgcn_quality import check_runtime
    from .hgcn_replay import load_policy
    from .hgcn_rtsc_entry import parser
    from .hgcn_rtsc_protocol import source_hashes, validate_config, verify_release
    from .hgcn_upstream import load_upstream

    args = parser().parse_args()
    output = Path(args.output)
    report = {'status': 'running', 'phase': args.phase,
              'scope': 'R-TSC/HGCN Stage A only', 'science_released': True}
    started = time.perf_counter()
    try:
        deadline()
        original = json.loads(args.original_config.read_bytes())
        policy = load_policy(args.policy, original)
        settings = json.loads(args.config.read_bytes())
        validate_config(settings, original, policy)
        release = verify_release(settings, original, policy, args.phase,
                                 args.release_record, args.source_commit,
                                 seed=args.seed, arm=args.arm, repeat_start=args.repeat_start,
                                 original_run=args.original_training_run,
                                 task_run=args.task_result,
                                 relation_run=args.relation_result)
        before = source_hashes()
        device = check_runtime(original, 'cpu_quality' if args.phase == 'cpu_quality'
                               else 'cuda_quality')
        upstream = load_upstream(args.upstream_root, args.upstream_manifest)
        report.update(release=release, device=str(device), upstream=upstream.identity)
        atomic_json(output / 'run.json', report)
        if args.phase in ('cpu_quality', 'cuda_quality'):
            result = small_quality(upstream, settings, device, output)
        else:
            data, base, reference, binding, state_sha, best = original_inputs(
                args.original_training_run, original, policy, args.seed,
                upstream, device, args.prepared_root)
            report['original_best'] = {'step': best['step'],
                                       'checkpoint_sha256': best['checkpoint']['sha256'],
                                       'state_sha256': state_sha}
            if args.phase in ('probe', 'train'):
                result = train_arm(data, base, reference, binding, state_sha,
                                   original, policy, args.prepared_root, settings,
                                   args.seed, args.arm, output, device,
                                   probe_steps=args.probe_steps if args.phase == 'probe' else None)
            else:
                arm_inputs = {}
                for arm, path in (('task_only', args.task_result),
                                  ('task_relation', args.relation_result)):
                    arm_inputs[arm] = (Path(path).parent,
                                       json.loads(Path(path).read_bytes()))
                result = final_shard(data, base, reference, binding, state_sha,
                                     original, policy, args.prepared_root, settings,
                                     args.seed, args.repeat_start, output, device, arm_inputs)
        if source_hashes() != before:
            raise ValueError('source changed during Stage A worker')
        load_upstream(args.upstream_root, args.upstream_manifest)
        report['status'] = 'passed'
        report['result_status'] = result['status']
        report['result_file'] = 'quality.json' if args.phase.endswith('quality') else 'result.json'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None
        atomic_json(output / 'run.json', report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

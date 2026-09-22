"""Release-gated frozen-HGCN amplitude worker; never mutates Stage A artifacts."""
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from .diagnostic_archive import write_archive
from .hgcn_evidence import (complete_ranking, deadline, load_checkpoint,
                            save_checkpoint, state_hash)
from .hgcn_quality import atomic_json, check_runtime, synchronize
from .hgcn_replay import load_policy, qualify_full, require_qualified
from .hgcn_rtsc_amplitude import FrozenAmplitudeHGCN
from .hgcn_rtsc_amplitude_protocol import (CONDITIONS, MixedStreams, choose_checkpoint,
                                          source_hashes, validate_config, verify_release)
from .hgcn_rtsc_worker import (_correction as old_correction,
                               _correction_binding as old_binding,
                               _development_plans, _generator, _metrics, _plans,
                               _train_root, full_control, original_inputs)
from .hgcn_sampling import make_plan, mask_graph
from .hgcn_tangent_correction import (ball_distance_from_anchor,
                                      relation_gap_loss, training_relation_weights)
from .hgcn_tangent_amplitude import AmplitudeCorrection, calibrate_and_cap
from .hgcn_upstream import load_upstream
from .mature_hgcn import MatureHGCN


def _serialize(diagnostics):
    return [{k: float(v.detach()) if torch.is_tensor(v) else v
             for k, v in row.items()} for row in diagnostics]


def _new_model(base, settings, init_seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        m = settings['module']
        return FrozenAmplitudeHGCN(base, hidden=m['hidden'], tau=m['tau'],
                                   eps=m['eps'], chunk_edges=m['chunk_edges'],
                                   scale=m['scale'])


def _binding(settings, policy, seed, arm, step):
    if seed not in settings['seeds'] or arm not in settings['arms'] or step not in settings['selection']['steps']:
        raise ValueError('registered amplitude checkpoint identity required')
    from .hgcn_registration import canonical
    return {'protocol': settings['protocol'], 'config_sha256': canonical(settings),
            'seed': seed, 'arm': arm, 'step': step,
            'original_checkpoint_sha256': policy['origins'][str(seed)]['best_checkpoint_sha256']}


def _matched_initialization(new, base, old_settings, init_seed):
    prior = old_correction(copy.deepcopy(base), old_settings, init_seed)
    for new_layer, old_layer in zip(new.corrections, prior.corrections):
        for key, tensor in new_layer.state_dict().items():
            if not torch.equal(tensor, old_layer.state_dict()[key]):
                raise ValueError('module initialization differs from Stage A matched stream')
    del prior


def _readout(model, plans, features, evaluator, valid, truth, settings, output, name):
    model.eval()
    with torch.no_grad():
        points, diagnostics = model.encode(features, plans, detailed=True)
        metrics, ranking, hierarchy = _metrics(model.base, points, evaluator, valid, truth,
                                                settings['ranking'])
    archive = write_archive(output / 'arrays', name,
                            {'points': points, 'plans': [p.archive() for p in plans],
                             'ranking': ranking, 'hierarchy': hierarchy,
                             'diagnostics': _serialize(diagnostics)})
    return {'metrics': metrics, 'ranking_seconds': ranking['elapsed_seconds'],
            'archive': archive, 'diagnostics': _serialize(diagnostics)}


def small_quality(upstream, settings, old_settings, device, output):
    torch.manual_seed(41)
    base = MatureHGCN(upstream, 3, 4, 5, 1., 0., device).eval()
    before = state_hash(base.state_dict())
    streams = MixedStreams()
    init_seed = streams.seed('module_init', 11)
    model = _new_model(base, settings, init_seed)
    _matched_initialization(model, base, old_settings, init_seed)
    features = torch.randn(8, 3, device=device) * .1
    neighbors = [[j for j in range(8) if j != i] for i in range(8)]
    plans = [make_plan(neighbors, 4, _generator(i + 4)) for i in (0, 1)]
    original = base.encode(features, [p.matrix(device=device) for p in plans])
    points, diagnostics = model.encode(features, plans, detailed=True)
    if not torch.equal(original, points):
        raise ValueError('zero-initialized amplitude module changed official HGCN output')
    query = torch.tensor([[0, 1], [2, 3], [4, 5]], device=device)
    loss = model.score(points, query).square().mean()
    loss.backward()
    gradients = [float(sum(p.grad.detach().abs().sum() for p in m.coefficients[-1].parameters()))
                 for m in model.corrections]
    if any(g <= 0 or not np.isfinite(g) for g in gradients):
        raise ValueError('both amplitude output layers require finite learnable gradients')
    with torch.no_grad():
        for correction in model.corrections:
            correction.coefficients[-1].bias.copy_(torch.tensor([.2, -.1, .1], device=device))
        changed, measured = model.encode(features, plans, detailed=True)
    response = float((changed - original).abs().max())
    if response <= 1e-5 or not torch.isfinite(changed).all() or state_hash(base.state_dict()) != before:
        raise ValueError('nonzero finite response and frozen backbone required')
    # The direction normalization is most sensitive around epsilon and near
    # the ball boundary. Compare physical inputs after FP32 quantization.
    numerical = []
    for conformal in (2., 200.):
        for physical in (0., 1e-12, 1e-10, 1e-8, 1e-7, 1e-6):
            basis32 = torch.zeros((1, 3, 2), dtype=torch.float32, device=device)
            basis32[0, 0] = torch.tensor([physical / conformal,
                                           .3 * physical / conformal],
                                          dtype=torch.float32, device=device)
            basis32.requires_grad_()
            coeff32 = torch.tensor([[.6, -.2, .1]], dtype=torch.float32, device=device)
            lam32 = torch.tensor([[conformal]], dtype=torch.float32, device=device)
            q32 = torch.tensor([.2], dtype=torch.float32, device=device)
            step32, raw32, _ = calibrate_and_cap(basis32, coeff32, lam32, q32)
            grad32 = torch.autograd.grad(step32.sum() + .001 * raw32.sum(), basis32)[0]
            basis64 = basis32.detach().cpu().double().requires_grad_()
            step64, raw64, _ = calibrate_and_cap(
                basis64, coeff32.cpu().double(), lam32.cpu().double(), q32.cpu().double())
            grad64 = torch.autograd.grad(step64.sum() + .001 * raw64.sum(), basis64)[0]
            value_error = float((step32.detach().cpu().double() - step64.detach()).abs().max())
            gradient_error = float((grad32.detach().cpu().double() - grad64.detach()).abs().max())
            relative_gradient_error = gradient_error / max(float(grad64.detach().abs().max()), 1.)
            if (not torch.isfinite(step32).all() or not torch.isfinite(grad32).all()
                    or value_error > 3e-9 + 3e-5 * float(step64.detach().abs().max())
                    or gradient_error > 4. + 5e-5 * float(grad64.detach().abs().max())):
                raise ValueError('native FP32 amplitude value/gradient disagrees with quantized FP64 reference')
            numerical.append({'conformal': conformal, 'physical_basis_component': physical,
                              'step_max_abs_error': value_error,
                              'gradient_max_abs_error': gradient_error,
                              'gradient_relative_max_error': relative_gradient_error})
    # Full and low-k plans remain exact local identities even after nonzero
    # coefficients; no diagnostic input may create an oracle correction.
    full = make_plan(neighbors, None, _generator(0))
    full_original = base.encode(features, [full.matrix(device=device)] * 2)
    full_corrected, full_meta = model.encode(features, [full] * 2)
    if not torch.equal(full_corrected, full_original) or any(r['eligible_nodes'] for r in full_meta):
        raise ValueError('complete-neighborhood amplitude identity failed')
    low_neighbors = [[1], [0, 2], [1], [], [], [], [], []]
    low = make_plan(low_neighbors, 1, _generator(1))
    low_original = base.encode(features, [low.matrix(device=device)] * 2)
    low_corrected, low_meta = model.encode(features, [low] * 2)
    if not torch.equal(low_corrected, low_original) or any(r['eligible_nodes'] for r in low_meta):
        raise ValueError('low-k amplitude identity failed')
    q = torch.tensor([.5], device=device, requires_grad=True)
    test_basis = torch.tensor([[[.4, 0.], [0., .2], [.1, .1]]], device=device)
    test_coeff = torch.tensor([[1., 0., 0.]], device=device)
    test_lam = torch.tensor([[2.]], device=device)
    test_step, test_raw, _ = calibrate_and_cap(test_basis, test_coeff, test_lam, q)
    radial_q_grad = float(torch.autograd.grad(test_step[0, 0], q, retain_graph=True)[0])
    penalty_q_grad = float(torch.autograd.grad(test_raw.sum(), q)[0])
    if radial_q_grad <= 0 or penalty_q_grad <= 0 or not np.isfinite(radial_q_grad + penalty_q_grad):
        raise ValueError('soft-cap q response or raw amplitude regularizer gradient absent')
    boundary_module = AmplitudeCorrection(chunk_edges=3).to(device)
    with torch.no_grad():
        boundary_module.coefficients[-1].bias.copy_(
            torch.tensor([.3, -.2, .1], device=device))
    boundary = torch.tensor([[.995, 0.]], device=device).repeat(6, 1).requires_grad_()
    messages = boundary.detach().clone()
    messages[:, 1] = torch.tensor([0., 1e-6, -1e-6, 2e-6, -2e-6, 3e-6], device=device)
    messages.requires_grad_()
    boundary_plan = make_plan([[j for j in range(6) if j != i] for i in range(6)],
                              4, _generator(9))
    boundary_out, boundary_diag = boundary_module(
        boundary, messages, boundary_plan, base.encoder.manifold, detailed=True)
    boundary_out.square().sum().backward()
    if (not torch.isfinite(boundary_out).all()
            or not torch.isfinite(boundary.grad).all()
            or not torch.isfinite(messages.grad).all()
            or any(p.grad is None or not torch.isfinite(p.grad).all()
                   for p in boundary_module.parameters())):
        raise ValueError('near-boundary complete amplitude gradient is nonfinite')
    if (boundary_out.square().sum(dim=-1) >= 1).any():
        raise ValueError('near-boundary correction left the Poincare ball')
    result = {'status': 'passed', 'device': str(device), 'zero_output_exact': True,
              'matched_old_initialization_exact': True,
              'two_layer_gradient_l1': gradients, 'nonzero_response_max_abs': response,
              'diagnostics': _serialize(measured), 'base_state_unchanged': True,
              'physical_precision_checks': numerical,
              'full_identity_exact': True, 'low_k_identity_exact': True,
              'soft_cap_q_gradient': radial_q_grad,
              'raw_penalty_q_gradient': penalty_q_grad,
              'near_boundary_forward_gradient_finite': True,
              'near_boundary_diagnostics': _serialize([boundary_diag])}
    atomic_json(output / 'quality.json', result)
    return result


def _evaluate_checkpoint(model, step, frozen, features, evaluator, valid, truth,
                         settings, output):
    model.eval()
    rows = []
    for repeat, original in enumerate(frozen):
        deadline()
        with torch.no_grad():
            points, diagnostics = model.encode(features, original['plans'], detailed=True)
            metrics, ranking, hierarchy = _metrics(model.base, points, evaluator, valid,
                                                    truth, settings['ranking'])
        archive = write_archive(output / 'arrays', f'dev-C-step{step}-r{repeat}',
                                {'points': points, 'ranking': ranking, 'hierarchy': hierarchy,
                                 'plans': [p.archive() for p in original['plans']],
                                 'diagnostics': _serialize(diagnostics)})
        rows.append({'repeat': repeat, 'micro_mrr': metrics['micro_mrr'],
                     'direct_order': metrics['direct_order'],
                     'direct_order_delta': metrics['direct_order'] - original['metrics']['direct_order'],
                     'S_archive': original['archive'], 'C_archive': archive,
                     'diagnostics': _serialize(diagnostics)})
    return {'step': step, 'micro_mrr': [r['micro_mrr'] for r in rows],
            'direct_order_delta': [r['direct_order_delta'] for r in rows],
            'repeats': rows}


def train_arm(data, base, reference, binding, state_sha, original, policy,
              prepared_root, settings, old_settings, seed, arm, output, device,
              *, probe_steps=None):
    if seed not in settings['seeds'] or arm not in settings['arms']:
        raise ValueError('registered fixed seed and arm required')
    if probe_steps is not None and (type(probe_steps) is not int or not 1 <= probe_steps <= 32):
        raise ValueError('discarded amplitude probe must use 1..32 steps')
    streams = MixedStreams()
    features, valid, truth, evaluator, qualification = full_control(
        data, base, reference, binding, state_sha, original, policy,
        prepared_root, output, device)
    init_seed = streams.seed('module_init', seed)
    model = _new_model(base, settings, init_seed)
    _matched_initialization(model, base, old_settings, init_seed)
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
    probe_readouts = ([_readout(model, probe_plans, features, evaluator, valid, truth,
                                settings, output, 'probe-step0')]
                      if probe_plans is not None else None)
    evaluations, checkpoints, history = [], {}, []
    if frozen is not None:
        evaluations.append(_evaluate_checkpoint(model, 0, frozen, features, evaluator,
                                                valid, truth, settings, output))
        checkpoints[0] = save_checkpoint(output, 'correction-step-0.pt', model,
                                         _binding(settings, policy, seed, arm, 0))
    steps = probe_steps if probe_steps is not None else settings['training']['steps']
    for step in range(1, steps + 1):
        deadline(); synchronize(device); started = time.perf_counter()
        batch_seed = streams.seed('train_batch', seed, step=step, fanout=4)
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
            raise ValueError('nonfinite amplitude training loss')
        loss.backward()
        params = list(model.trainable_parameters())
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
            raise ValueError('missing or nonfinite amplitude gradient')
        layer_grads = [float(torch.linalg.vector_norm(torch.stack(
            [torch.linalg.vector_norm(p.grad) for p in module.parameters()])))
                       for module in model.corrections]
        norm = float(torch.nn.utils.clip_grad_norm_(params, settings['training']['grad_clip_norm']))
        previous = [p.detach().clone() for p in params]
        optimizer.step(); synchronize(device)
        update_norm = float(torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(p.detach() - old) for p, old in zip(params, previous)])))
        if not np.isfinite(update_norm) or any(not torch.isfinite(p).all() for p in params):
            raise ValueError('nonfinite amplitude parameters after update')
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
               'diagnostics': _serialize(diagnostics),
               'seconds': time.perf_counter() - started}
        if probe_steps is not None and step in (1, steps):
            with torch.no_grad():
                anchor = points[root:root + 1].detach().expand(len(groups), -1)
                positive = groups[:, 0].to(device)
                parent_radius = ball_distance_from_anchor(anchor, points[positive[:, 0]])
                child_radius = ball_distance_from_anchor(anchor, points[positive[:, 1]])
                row['relation_geometry'] = {
                    'parent_near_zero': int((parent_radius <= settings['module']['eps']).sum()),
                    'child_near_zero': int((child_radius <= settings['module']['eps']).sum()),
                    'near_zero_gap': int(((child_radius - parent_radius).abs() <= settings['module']['eps']).sum())}
        history.append(row)
        if frozen is not None and step in settings['selection']['steps']:
            evaluations.append(_evaluate_checkpoint(model, step, frozen, features,
                                                    evaluator, valid, truth, settings, output))
            checkpoints[step] = save_checkpoint(output, f'correction-step-{step}.pt', model,
                                                _binding(settings, policy, seed, arm, step))
        if step % 16 == 0 or step == steps:
            atomic_json(output / 'progress.json', {'status': 'running', 'seed': seed,
                                                    'arm': arm, 'completed_steps': step,
                                                    'history': history, 'evaluations': evaluations})
        del points, plans, masked
    selected = None if frozen is None else choose_checkpoint(
        evaluations, settings['selection']['direct_order_delta_floor'])
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
        probe_readouts.append(_readout(model, probe_plans, features, evaluator, valid,
                                       truth, settings, output, f'probe-step{steps}'))
        result['probe_complete_sampled_ranking'] = probe_readouts
        result['peak_allocated_bytes'] = (torch.cuda.max_memory_allocated()
                                          if device.type == 'cuda' else None)
        result['peak_reserved_bytes'] = (torch.cuda.max_memory_reserved()
                                         if device.type == 'cuda' else None)
    atomic_json(output / 'result.json', result)
    return result


def final_shard(data, base, reference, binding, state_sha, original, policy,
                prepared_root, settings, old_settings, seed, repeat_start, output,
                device, arm_inputs):
    streams = MixedStreams()
    features, valid, truth, evaluator, qualification = full_control(
        data, base, reference, binding, state_sha, original, policy,
        prepared_root, output, device)
    models, selections = {}, {}
    for version in ('old', 'new'):
        for arm in ('task_only', 'task_relation'):
            name = version + '_' + arm
            run_dir, run = arm_inputs[name]
            if (run['status'] != 'complete' or run['seed'] != seed or run['arm'] != arm
                    or run['completed_steps'] != 1024 or run['original_best_binding'] != binding
                    or run['original_best_state_sha256'] != state_sha
                    or run['selected'] is None or run['selected_checkpoint'] is None
                    or run['base_state_unchanged'] is not True):
                raise ValueError('four complete selected arms and matching frozen base required')
            chosen = run['selected_checkpoint']
            if version == 'old':
                expected = old_binding(old_settings, policy, seed, arm, run['selected']['step'])
                model = old_correction(copy.deepcopy(base), old_settings,
                                       streams.seed('module_init', seed))
            else:
                expected = _binding(settings, policy, seed, arm, run['selected']['step'])
                model = _new_model(copy.deepcopy(base), settings,
                                   streams.seed('module_init', seed))
            cp = load_checkpoint(run_dir, chosen, expected)
            model.load_state_dict(cp['model_state'], strict=True)
            model.eval()
            if state_hash(model.base.state_dict()) != state_sha:
                raise ValueError('selected checkpoint changed frozen HGCN')
            models[name] = model
            selections[name] = {'step': run['selected']['step'], 'checkpoint': chosen}
    full_plan = make_plan(data['neighbors'], None, _generator(0))
    full_qualifications = {}
    with torch.no_grad():
        for name, model in models.items():
            if name.startswith('new_'):
                corrected, diagnostics = model.encode(features, [full_plan] * 2, detailed=True)
            else:
                corrected, diagnostics = model.encode(features, [full_plan] * 2)
            if any(d['eligible_nodes'] for d in diagnostics):
                raise ValueError('full adjacency must be local identity')
            ranking = complete_ranking(base, corrected, valid, truth, settings['ranking'])
            quality = qualify_full(corrected, ranking, reference, evaluator, original,
                                   policy, valid, truth, state_sha, binding, full_plan.archive())
            archive = write_archive(output / 'arrays', f'full-control-{name}',
                                    {'points': corrected, 'ranking': ranking,
                                     'qualification': quality,
                                     'diagnostics': _serialize(diagnostics)})
            atomic_json(output / f'full-qualification-{name}.json',
                        {'qualification': quality, 'archive': archive})
            require_qualified(quality)
            full_qualifications[name] = quality
    if repeat_start not in (0, 4, 8, 12):
        raise ValueError('registered final four-repeat shard required')
    rows = []
    for fanout in settings['final']['fanouts']:
        for repeat in range(repeat_start, repeat_start + 4):
            deadline(); plans = _plans(data['neighbors'], streams, 'final_layer', seed,
                                       fanout=fanout, repeat=repeat)
            for name in CONDITIONS:
                deadline()
                with torch.no_grad():
                    if name == 'S':
                        points = base.encode(features, [p.matrix(device=device) for p in plans])
                        diagnostics = None
                    elif name.startswith('new_'):
                        points, diagnostics = models[name].encode(features, plans, detailed=True)
                    else:
                        points, diagnostics = models[name].encode(features, plans)
                    metrics, ranking, hierarchy = _metrics(base, points, evaluator,
                                                             valid, truth, settings['ranking'])
                archive = write_archive(output / 'arrays', f'final-seed{seed}-f{fanout}-r{repeat}-{name}',
                                        {'points': points, 'plans': [p.archive() for p in plans],
                                         'ranking': ranking, 'hierarchy': hierarchy,
                                         'diagnostics': None if diagnostics is None else _serialize(diagnostics)})
                rows.append({'seed': seed, 'fanout': fanout, 'repeat': repeat,
                             'arm': name, **metrics, 'archive': archive})
            atomic_json(output / 'progress.json', {'status': 'running', 'seed': seed,
                                                    'completed_conditions': len(rows), 'rows': rows})
    if len(rows) != 60:
        raise ValueError('all 60 paired final conditions required')
    result = {'status': 'complete', 'seed': seed, 'repeat_start': repeat_start,
              'rows': rows, 'selected': selections,
              'fixed_F': {'micro_mrr': reference['ranking']['query_micro_mrr'],
                          'direct_order': evaluator.evaluate(reference['native_ball_points'])['direct']['metrics']['score']},
              'original_F_qualification': qualification,
              'full_arm_qualifications': full_qualifications,
              'rng_manifest': streams.manifest(),
              'base_state_unchanged': state_hash(base.state_dict()) == state_sha}
    atomic_json(output / 'result.json', result)
    return result


def load_arm_inputs(args):
    """Map public final CLI flags to the exact five-condition arm names."""
    result_flags = {'old_task_only': args.old_task_result,
                    'old_task_relation': args.old_relation_result,
                    'new_task_only': args.new_task_result,
                    'new_task_relation': args.new_relation_result}
    if any(path is None for path in result_flags.values()):
        raise ValueError('four selected arm results required')
    return {name: (Path(path).parent, json.loads(Path(path).read_bytes()))
            for name, path in result_flags.items()}


def main():
    from .hgcn_rtsc_amplitude_entry import parser

    args = parser().parse_args()
    output = Path(args.output)
    report = {'status': 'running', 'phase': args.phase,
              'scope': 'R-TSC/HGCN amplitude v1 only', 'science_released': True}
    started = time.perf_counter()
    try:
        deadline()
        original = json.loads(args.original_config.read_bytes())
        policy = load_policy(args.policy, original)
        settings = json.loads(args.config.read_bytes())
        old_settings = json.loads(args.old_config.read_bytes())
        validate_config(settings, original, policy, old_settings)
        release = verify_release(settings, original, policy, old_settings,
                                 args.phase, args.release_record, args.source_commit,
                                 seed=args.seed, arm=args.arm, repeat_start=args.repeat_start,
                                 original_run=args.original_training_run,
                                 new_task_run=args.new_task_result,
                                 new_relation_run=args.new_relation_result,
                                 old_task_run=args.old_task_result,
                                 old_relation_run=args.old_relation_result)
        before = source_hashes()
        device = check_runtime(original, 'cpu_quality' if args.phase == 'cpu_quality'
                               else 'cuda_quality')
        upstream = load_upstream(args.upstream_root, args.upstream_manifest)
        report.update(release=release, device=str(device), upstream=upstream.identity)
        atomic_json(output / 'run.json', report)
        if args.phase in ('cpu_quality', 'cuda_quality'):
            result = small_quality(upstream, settings, old_settings, device, output)
        else:
            data, base, reference, binding, state_sha, best = original_inputs(
                args.original_training_run, original, policy, args.seed,
                upstream, device, args.prepared_root)
            report['original_best'] = {'step': best['step'],
                                       'checkpoint_sha256': best['checkpoint']['sha256'],
                                       'state_sha256': state_sha}
            if args.phase in ('probe', 'train'):
                result = train_arm(data, base, reference, binding, state_sha, original,
                                   policy, args.prepared_root, settings, old_settings,
                                   args.seed, args.arm, output, device,
                                   probe_steps=args.probe_steps if args.phase == 'probe' else None)
            else:
                arm_inputs = load_arm_inputs(args)
                result = final_shard(data, base, reference, binding, state_sha,
                                     original, policy, args.prepared_root, settings,
                                     old_settings, args.seed, args.repeat_start, output,
                                     device, arm_inputs)
        if source_hashes() != before:
            raise ValueError('source changed during amplitude worker')
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
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        atomic_json(output / 'run.json', report)


if __name__ == '__main__':
    main()

"""Bounded six-arm joint HGCN worker; only the reviewed entry may invoke it."""
import copy
import gc
import hashlib
import io
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .diagnostic_archive import read_archive, write_archive
from .hgcn_evidence import (complete_ranking, deadline, hierarchy_evaluator,
                            file_hash, load_checkpoint, load_panel_design,
                            save_checkpoint, state_hash, validate_ranking)
from .hgcn_geometry import distance_ball_fp64
from .hgcn_panels import HierarchyPanel
from .hgcn_quality import atomic_json, load_train, load_valid, synchronize
from .hgcn_registration import canonical
from .hgcn_replay import (POLICY_SHA256, native_array, require_qualified)
from .hgcn_rtsc_joint import (JointCorrectedHGCN, encode_joint, joint_base)
from .hgcn_rtsc_joint_eval_v2 import (
    encode_evaluation, evaluate_frozen_snapshot, make_evaluation_binding,
    require_compatible_evaluation_bindings, require_evaluation_binding)
from .hgcn_rtsc_joint_protocol import (ARMS, NamedStreams, choose_checkpoint,
                                       final_families, source_hashes,
                                       validate_config, verify_release)
from .hgcn_rtsc_worker import (_generator, _model, _plans, _train_root,
                               original_inputs)
from .hgcn_sampling import make_plan, mask_graph
from .hgcn_tangent_correction import (ball_distance_from_anchor,
                                      relation_gap_loss, training_relation_weights)
from .mature_hgcn import MatureHGCN


def _serialize(diagnostics):
    return [{key: float(value.detach()) if torch.is_tensor(value) else value
             for key, value in row.items()} for row in diagnostics]


def _snapshot_evaluate(model, features, plans, callback, *, optimizer=None,
                       module_off=False):
    """Run one evidence-producing forward on a disjoint frozen model copy."""
    def evaluate(snapshot):
        points, diagnostics = encode_evaluation(
            snapshot, features, plans, module_off=module_off)
        return callback(snapshot, points, diagnostics)

    return evaluate_frozen_snapshot(
        model, lambda: copy.deepcopy(model), evaluate, optimizer=optimizer)


def _eval_binding(settings, policy, source_commit, training_source_commit,
                  checkpoint_sha, state_sha, plans, phase, fanout, repeat,
                  device, *, plan_step=None,
                  checkpoint_artifact_kind='checkpoint_file'):
    return make_evaluation_binding(
        evaluation_source_commit=source_commit,
        training_source_commit=training_source_commit,
        config_sha256=canonical(settings),
        replay_policy_sha256=canonical(policy),
        checkpoint_sha256=checkpoint_sha,
        model_state_sha256=state_sha,
        plans=plans, plan_phase=phase, fanout=fanout, repeat=repeat,
        device=device, plan_step=plan_step,
        checkpoint_artifact_kind=checkpoint_artifact_kind)


def _state_bytes_sha256(model):
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def _report_only_full_bridge(legacy, points, ranking, evaluator, original,
                             valid, truth):
    """Describe legacy-full to v2-full drift without applying same-suite gates."""
    old = native_array(legacy['native_ball_points']).astype(np.float64)
    new = native_array(points).astype(np.float64)
    if old.shape != new.shape:
        raise ValueError('legacy and v2 original full point shapes differ')
    validate_ranking(legacy['ranking'], valid, truth, len(old))
    validate_ranking(ranking, valid, truth, len(new))
    old_rows = {(row['parent'], row['child']): row['rank']
                for row in legacy['ranking']['rows']}
    new_rows = {(row['parent'], row['child']): row['rank']
                for row in ranking['rows']}
    if set(old_rows) != set(new_rows):
        raise ValueError('legacy and v2 full ranking queries differ')
    delta = new - old
    old_hierarchy = evaluator.evaluate(old)
    new_hierarchy = evaluator.evaluate(new)
    return {
        'status': 'report_only_cross_backend_bridge',
        'hard_gate_applied': False,
        'legacy_backend': 'historical_training_evaluator',
        'new_backend': 'rtsc_hgcn_full_eval_torch251_sparse_bmm_coo_alg2_v1',
        'point_max_abs': float(np.max(np.abs(delta))),
        'point_RMS': float(np.sqrt(np.mean(delta * delta))),
        'max_geodesic': float(np.max(distance_ball_fp64(
            old, new, original['model']['c']))),
        'rank_changed_queries': sum(old_rows[key] != new_rows[key]
                                    for key in old_rows),
        'rank_max_abs_delta': max(abs(old_rows[key] - new_rows[key])
                                  for key in old_rows),
        'micro_mrr_signed_delta': (ranking['query_micro_mrr']
                                   - legacy['ranking']['query_micro_mrr']),
        'macro_mrr_signed_delta': (ranking['child_macro_mrr']
                                   - legacy['ranking']['child_macro_mrr']),
        'hits_signed_delta': {key: ranking['hits'][key] - legacy['ranking']['hits'][key]
                              for key in ('1', '3', '10')},
        'hierarchy_score_signed_delta': {
            kind: (new_hierarchy[kind]['metrics']['score']
                   - old_hierarchy[kind]['metrics']['score'])
            if (new_hierarchy[kind]['metrics']['score'] is not None
                and old_hierarchy[kind]['metrics']['score'] is not None)
            else None
            for kind in ('direct', 'distant')},
    }


class _PreparedPlan:
    """Benchmark-only fixed GPU matrix plus the original sampled metadata."""

    def __init__(self, plan, matrix):
        self.plan = plan
        self.cached_matrix = matrix

    def __getattr__(self, name):
        return getattr(self.plan, name)

    def matrix(self, *, device='cpu', dtype=torch.float32):
        if self.cached_matrix.device != torch.device(device) or self.cached_matrix.dtype != dtype:
            raise ValueError('benchmark prebuilt matrix device/dtype differs')
        return self.cached_matrix


def _prepare_plans(plans, features):
    return [_PreparedPlan(plan, plan.matrix(
        device=features.device, dtype=features.dtype)) for plan in plans]


def _new_model(base, structure, settings, init_seeds):
    if structure == 'plain':
        return base
    if structure not in ('three', 'single'):
        raise ValueError('registered joint structure required')
    module = settings['module']
    return JointCorrectedHGCN(
        base, structure, hidden=module['hidden'], tau=module['tau'],
        eps=module['eps'], chunk_edges=module['chunk_edges'],
        layer_init_seeds=init_seeds)


def _binding(settings, source_commit, policy, seed, arm, step):
    if seed not in settings['seeds'] or arm not in ARMS or step not in settings['selection']['steps']:
        raise ValueError('registered joint checkpoint selector required')
    return {'protocol': settings['protocol'], 'config_sha256': canonical(settings),
            'source_commit': source_commit, 'seed': seed, 'arm': arm, 'step': step,
            'original_checkpoint_sha256': policy['origins'][str(seed)]['best_checkpoint_sha256']}


def _optimizer(model, settings):
    base = joint_base(model)
    encoder = list(base.encoder.parameters())
    head = list(base.relation_head.parameters())
    layers = ([list(correction.parameters()) for correction in model.corrections]
              if isinstance(model, JointCorrectedHGCN) else [[], []])
    module = layers[0] + layers[1]
    if not encoder or not head or any(not p.requires_grad for p in encoder + head + module):
        raise ValueError('all registered trainable joint parameter groups required')
    if len({id(p) for p in encoder + head + module}) != len(encoder + head + module):
        raise ValueError('overlapping joint optimizer parameter groups')
    t = settings['training']
    groups = [{'params': encoder, 'lr': t['base_lr']},
              {'params': head, 'lr': t['base_lr']}]
    if module:
        groups.append({'params': module, 'lr': t['module_lr']})
    optimizer = torch.optim.Adam(groups, betas=tuple(t['betas']), eps=t['eps'],
                                 weight_decay=t['weight_decay'])
    return optimizer, {'encoder': encoder, 'head': head,
                       'module_layer0': layers[0], 'module_layer1': layers[1]}


def _norm(values):
    if not values:
        return 0.
    return float(torch.linalg.vector_norm(torch.stack(
        [torch.linalg.vector_norm(value) for value in values])))


def _candidate_metrics(model, points, evaluator, valid, truth, settings):
    ranking = complete_ranking(joint_base(model), points, valid, truth, settings['ranking'])
    hierarchy = evaluator.evaluate(points.detach().cpu().numpy())
    return {'micro_mrr': ranking['query_micro_mrr'],
            'direct_order': hierarchy['direct']['metrics']['score']}, ranking, hierarchy


def _same_plan(actual, saved):
    if set(actual) != set(saved):
        raise ValueError('joint full adjacency inventory differs')
    for key, value in actual.items():
        value = value.detach().cpu().numpy() if torch.is_tensor(value) else value
        recorded = saved[key]
        recorded = (recorded.detach().cpu().numpy()
                    if torch.is_tensor(recorded) else recorded)
        if isinstance(recorded, np.ndarray) or isinstance(value, np.ndarray):
            if (not isinstance(value, np.ndarray)
                    or not isinstance(recorded, np.ndarray)
                    or value.dtype != recorded.dtype
                    or value.shape != recorded.shape or not np.array_equal(value, recorded)):
                raise ValueError('joint full adjacency differs: ' + key)
        elif value != recorded:
            raise ValueError('joint full adjacency differs: ' + key)


def qualify_joint_full(points, ranking, reference, evaluator, original, policy,
                       valid, truth, state_sha, binding, full_plan):
    """V2 limits applied to the same *new* checkpoint's immutable saved F."""
    if reference['binding'] != binding or reference['model_state_sha256'] != state_sha:
        raise ValueError('joint full reference binding or state differs')
    _same_plan(full_plan, reference['full_plan'])
    actual = native_array(points)
    fixed = native_array(reference['native_ball_points'])
    if actual.shape != fixed.shape or fixed.shape != (
            original['prepared']['nodes_count'], original['model']['hidden']):
        raise ValueError('complete native joint point matrix required')
    validate_ranking(ranking, valid, truth, len(fixed))
    validate_ranking(reference['ranking'], valid, truth, len(fixed))
    if not np.array_equal(evaluator.full, fixed.astype(np.float64)):
        raise ValueError('joint evaluator must use selected checkpoint own F')
    rows = {(r['parent'], r['child']): (r['rank'], r['candidates'])
            for r in ranking['rows']}
    baseline = {(r['parent'], r['child']): (r['rank'], r['candidates'])
                for r in reference['ranking']['rows']}
    if set(rows) != set(baseline) or any(rows[k][1] != baseline[k][1] for k in rows):
        raise ValueError('joint complete query and candidate identity mismatch')
    change = actual.astype(np.float64) - fixed.astype(np.float64)
    rank_changes = [abs(rows[key][0] - baseline[key][0]) for key in rows]
    old_hierarchy, new_hierarchy = evaluator.evaluate(fixed), evaluator.evaluate(actual)
    measured = {
        'point_max_abs': float(np.max(np.abs(change))),
        'point_RMS': float(np.sqrt(np.mean(change * change))),
        'max_geodesic': float(np.max(distance_ball_fp64(fixed, actual,
                                                       original['model']['c']))),
        'rank_changed_queries': sum(value != 0 for value in rank_changes),
        'rank_max_abs_delta': max(rank_changes),
        'micro_mrr_abs_delta': abs(ranking['query_micro_mrr']
                                   - reference['ranking']['query_micro_mrr']),
        'macro_mrr_abs_delta': abs(ranking['child_macro_mrr']
                                   - reference['ranking']['child_macro_mrr']),
        'hits_abs_delta': max(abs(ranking['hits'][k] - reference['ranking']['hits'][k])
                              for k in ('1', '3', '10')),
        'hierarchy': {}}
    limits = policy['limits']
    violations = []

    def bound(name, value, limit):
        if not np.isfinite(value) or value > limit:
            violations.append({'metric': name, 'observed': value, 'limit': limit})

    for name, value in measured.items():
        if name != 'hierarchy' and name not in ('micro_mrr_abs_delta', 'macro_mrr_abs_delta'):
            bound(name, value, limits[name])
    for kind in ('direct', 'distant'):
        a, b = new_hierarchy[kind], old_hierarchy[kind]
        for key in ('covered_children', 'selected_children', 'covered_pairs',
                    'unknown_pairs', 'weighted_covered_child_mass',
                    'weighted_selected_child_mass'):
            if a[key] != b[key]:
                raise ValueError('joint fixed panel coverage changed')
        if not len(a['gap']) or a['metrics']['score'] is None:
            registered = (original.get('hierarchy', {}).get('registration', {})
                          .get('coverage', {}).get(kind, {}))
            if registered.get('covered_pairs') not in (None, 0):
                raise ValueError('registered joint hierarchy coverage disappeared')
            measured['hierarchy'][kind] = {
                'status': 'unknown_no_covered_pairs', 'covered_pairs': 0}
            continue
        values = {
            'pair_gap_max_abs_delta': float(np.max(np.abs(a['gap'] - b['gap']))),
            'weighted_gap_abs_delta': abs(a['metrics']['gap'] - b['metrics']['gap']),
            'order_abs_delta': abs(a['metrics']['score'] - b['metrics']['score']),
            'pair_sign_changes': int(np.count_nonzero(a['score'] != b['score'])),
            'tie_or_unresolved_abs_delta': max(
                abs(a['metrics'][key] - b['metrics'][key])
                for key in ('tie', 'unresolved'))}
        measured['hierarchy'][kind] = values
        for name in ('pair_gap_max_abs_delta', 'weighted_gap_abs_delta'):
            bound(kind + '/' + name, values[name], limits['hierarchy'][kind][name])
        for name in ('order_abs_delta', 'pair_sign_changes',
                     'tie_or_unresolved_abs_delta'):
            bound(kind + '/' + name, values[name], limits['hierarchy_' + name])
    counts = {}
    for _, child in rows:
        counts[child] = counts.get(child, 0) + 1
    contributions = []
    for parent, child in sorted(rows):
        old_rank, new_rank = baseline[parent, child][0], rows[parent, child][0]
        if old_rank != new_rank:
            reciprocal = 1 / new_rank - 1 / old_rank
            contributions.append({
                'parent': parent, 'child': child, 'F_rank': old_rank,
                'full_rank': new_rank, 'micro_mrr': reciprocal / len(rows),
                'macro_mrr': reciprocal / (len(counts) * counts[child])})
    v1_mrr_violations = [
        {'metric': name, 'observed': measured[name], 'limit': limits[name]}
        for name in ('micro_mrr_abs_delta', 'macro_mrr_abs_delta')
        if measured[name] > limits[name]]
    return {
        'protocol': policy['protocol'], 'policy_sha256': canonical(policy),
        'accepted': not violations, 'measured': measured,
        'limits': limits, 'violations': violations,
        'own_checkpoint_reference': True,
        'original_bit_exact': {
            'native_points': bool(np.array_equal(actual, fixed)),
            'rank_rows': rows == baseline},
        'signed_micro_mrr_drift': ranking['query_micro_mrr']
        - reference['ranking']['query_micro_mrr'],
        'signed_macro_mrr_drift': ranking['child_macro_mrr']
        - reference['ranking']['child_macro_mrr'],
        'flat_MRR_limits_report_only_in_v2': True,
        'v1_diagnostic': {
            'v1_policy_sha256': POLICY_SHA256,
            'v1_accepted': not violations and not v1_mrr_violations,
            'v1_violations': violations + v1_mrr_violations,
            'changed_query_contributions': contributions,
            'unchanged_queries_have_zero_contribution': True},
        'limits_are_guaranteed_error_bounds': False}


def _fresh_reload(template, structure, settings, init_seeds, checkpoint_root,
                  descriptor, binding, device):
    torch_rng = torch.random.get_rng_state().clone()
    cuda_rng = ([state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else None)
    try:
        base = copy.deepcopy(template).to(device)
        model = _new_model(base, structure, settings, init_seeds)
        payload = load_checkpoint(checkpoint_root, descriptor, binding)
        model.load_state_dict(payload['model_state'], strict=True)
        if state_hash(model.state_dict()) != descriptor['model_state_sha256']:
            raise ValueError('fresh joint checkpoint state differs')
        model.eval()
        return model, payload
    finally:
        torch.random.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _full_reference(model, features, full_plan, valid, truth, view, panel,
                    original, policy, settings, binding, state_sha, output,
                    name, source_commit, device, *, checkpoint=None,
                    optimizer=None, phase='candidate_full'):
    plans = [full_plan] * 2

    def evaluate(snapshot, points, diagnostics):
        if any(row['eligible_nodes'] for row in diagnostics):
            raise ValueError('joint module must be identity on full adjacency')
        ranking = complete_ranking(joint_base(snapshot), points, valid, truth,
                                   settings['ranking'])
        evaluator, hierarchy = hierarchy_evaluator(
            view, panel, points.detach().cpu().numpy(), original)
        return points, diagnostics, ranking, evaluator, hierarchy

    points, diagnostics, ranking, evaluator, hierarchy = _snapshot_evaluate(
        model, features, plans, evaluate, optimizer=optimizer)
    checkpoint_sha = (checkpoint['sha256'] if checkpoint is not None
                      else _state_bytes_sha256(model))
    evaluation_binding = _eval_binding(
        settings, policy, source_commit,
        binding.get('source_commit', source_commit), checkpoint_sha, state_sha,
        plans, phase, None, None, device,
        checkpoint_artifact_kind=(
            'checkpoint_file' if checkpoint is not None
            else 'serialized_state_fixture_or_probe'))
    archive = write_archive(output / 'arrays', name,
                            {'binding': binding, 'model_state_sha256': state_sha,
                             'evaluation_binding': evaluation_binding,
                             'native_ball_points': points, 'ranking': ranking,
                             'hierarchy': hierarchy, 'full_plan': full_plan.archive(),
                             'checkpoint': checkpoint, 'diagnostics': _serialize(diagnostics),
                             'panel_design': evaluator.archive_design()})
    return archive, evaluator, {
        'micro_mrr': ranking['query_micro_mrr'],
        'direct_order': hierarchy['direct']['metrics']['score']}


def _original_v2_control(data, base, legacy_reference, original_binding,
                         original_state_sha, original, policy, prepared_root,
                         settings, seed, output, device, source_commit):
    """Build the v2 original-start F and keep the legacy bridge report-only."""
    features = data['features'].to(device)
    valid, truth = load_valid(prepared_root, data, original)
    view, panel = load_panel_design(prepared_root, original)
    full_plan = make_plan(data['neighbors'], None, _generator(0))
    plans = [full_plan] * 2

    def build_reference(snapshot, points, diagnostics):
        if any(row['eligible_nodes'] for row in diagnostics):
            raise ValueError('original full evaluation must keep correction identity')
        ranking = complete_ranking(snapshot, points, valid, truth,
                                   settings['ranking'])
        evaluator, hierarchy = hierarchy_evaluator(
            view, panel, points.detach().cpu().numpy(), original)
        return points, diagnostics, ranking, evaluator, hierarchy

    points, diagnostics, ranking, evaluator, hierarchy = _snapshot_evaluate(
        base, features, plans, build_reference)
    checkpoint_sha = policy['origins'][str(seed)]['best_checkpoint_sha256']
    evaluation_binding = _eval_binding(
        settings, policy, source_commit, original_binding['source_commit'],
        checkpoint_sha, original_state_sha, plans, 'original_full', None, None,
        device)
    reference = {
        'binding': original_binding,
        'evaluation_binding': evaluation_binding,
        'model_state_sha256': original_state_sha,
        'native_ball_points': points.detach().cpu().numpy(),
        'ranking': ranking,
        'hierarchy': hierarchy,
        'full_plan': full_plan.archive(),
        'diagnostics': _serialize(diagnostics),
        'panel_design': evaluator.archive_design(),
    }
    reference_archive = write_archive(
        output / 'arrays', 'original-v2-F', reference)

    def replay(snapshot, fresh_points, fresh_diagnostics):
        fresh_ranking = complete_ranking(
            snapshot, fresh_points, valid, truth, settings['ranking'])
        return fresh_points, fresh_diagnostics, fresh_ranking

    fresh_points, fresh_diagnostics, fresh_ranking = _snapshot_evaluate(
        base, features, plans, replay, module_off=True)
    qualification = qualify_joint_full(
        fresh_points, fresh_ranking, reference, evaluator, original, policy,
        valid, truth, original_state_sha, original_binding,
        full_plan.archive())
    require_qualified(qualification)
    _same_plan(full_plan.archive(), legacy_reference['full_plan'])
    bridge = _report_only_full_bridge(
        legacy_reference, points, ranking, evaluator, original, valid, truth)
    control_archive = write_archive(
        output / 'arrays', 'original-v2-fresh-module-off-control', {
            'points': fresh_points,
            'ranking': fresh_ranking,
            'diagnostics': _serialize(fresh_diagnostics),
            'qualification': qualification,
            'evaluation_binding': evaluation_binding,
            'reference': reference_archive,
        })
    summary = {
        'status': 'accepted_same_v2_original_start', 'accepted': True,
        'historical_replay': {
            'binding': original_binding,
            'model_state_sha256': original_state_sha,
            'checkpoint_sha256': checkpoint_sha,
            'legacy_reference_preserved': True,
        },
        'new_full_reference': reference_archive,
        'fresh_module_off_control': control_archive,
        'same_v2_qualification': qualification,
        'legacy_to_v2_bridge': bridge,
        'evaluation_binding': evaluation_binding,
    }
    atomic_json(output / 'original-v2-control.json', summary)
    return (features, valid, truth, evaluator, summary, full_plan,
            reference_archive)


def _development_plans(data, base, features, original_evaluator, valid, truth,
                       settings, policy, streams, seed, output, source_commit,
                       original_binding, original_state_sha, device,
                       optimizer=None):
    """The two original-start S0 graphs are shared by all six arm names."""
    plans_and_scores = []
    for repeat in range(2):
        deadline()
        plans = _plans(data['neighbors'], streams, 'selection_layer', seed,
                       fanout=4, repeat=repeat)
        def evaluate(snapshot, points, diagnostics):
            metrics, ranking, hierarchy = _candidate_metrics(
                snapshot, points, original_evaluator, valid, truth, settings)
            return points, diagnostics, metrics, ranking, hierarchy

        points, diagnostics, metrics, ranking, hierarchy = _snapshot_evaluate(
            base, features, plans, evaluate, optimizer=optimizer)
        evaluation_binding = _eval_binding(
            settings, policy, source_commit, original_binding['source_commit'],
            policy['origins'][str(seed)]['best_checkpoint_sha256'],
            original_state_sha, plans, 'selection_layer', 4, repeat, device)
        archive = write_archive(output / 'arrays', f'common-S0-r{repeat}',
                                {'points': points, 'plans': [p.archive() for p in plans],
                                  'ranking': ranking, 'hierarchy': hierarchy,
                                  'diagnostics': _serialize(diagnostics),
                                  'evaluation_binding': evaluation_binding,
                                  'original_F_root': original_evaluator.anchor})
        plans_and_scores.append({'plans': plans, 'metrics': metrics,
                                 'evaluation_binding': evaluation_binding,
                                 'archive': archive})
    return plans_and_scores


def _evaluate_candidate(model, template, optimizer, step, frozen, features,
                        full_plan, view, panel, valid, truth, original, policy,
                        settings, seed, arm, init_seeds, source_commit, output,
                        device, streams):
    deadline()
    binding = _binding(settings, source_commit, policy, seed, arm, step)
    descriptor = save_checkpoint(
        output, f'joint-{arm}-step-{step}.pt', model, binding,
        {'optimizer_state': optimizer.state_dict(),
         'rng_manifest': streams.manifest(), 'completed_steps': step})
    if state_hash(model.state_dict()) != descriptor['model_state_sha256']:
        raise ValueError('saved joint candidate differs from live state')
    reference_archive, evaluator, full_metrics = _full_reference(
        model, features, full_plan, valid, truth, view, panel, original, policy,
        settings, binding, descriptor['model_state_sha256'], output,
        f'candidate-{arm}-step{step}-F', source_commit, device,
        checkpoint=descriptor, optimizer=optimizer)
    reference = read_archive(output / 'arrays', reference_archive)
    fresh, payload = _fresh_reload(
        template, settings['arms'][arm]['structure'], settings, init_seeds,
        output, descriptor, binding, device)
    def replay(snapshot, points, diagnostics):
        ranking = complete_ranking(joint_base(snapshot), points, valid, truth,
                                   settings['ranking'])
        return points, diagnostics, ranking

    replay_points, replay_diagnostics, replay_ranking = _snapshot_evaluate(
        fresh, features, [full_plan] * 2, replay)
    if any(row['eligible_nodes'] for row in replay_diagnostics):
        raise ValueError('reloaded joint module changed full-neighborhood identity')
    replay_quality = qualify_joint_full(
        replay_points, replay_ranking, reference, evaluator, original, policy,
        valid, truth, payload['model_state_sha256'], binding, full_plan.archive())
    replay_archive = write_archive(output / 'arrays',
                                   f'candidate-{arm}-step{step}-fresh-full',
                                   {'points': replay_points, 'ranking': replay_ranking,
                                     'quality': replay_quality,
                                     'evaluation_binding':
                                         reference['evaluation_binding'],
                                     'diagnostics': _serialize(replay_diagnostics)})
    require_qualified(replay_quality)
    off_quality = None
    off_archive = None
    if isinstance(model, JointCorrectedHGCN):
        def module_off(snapshot, points, diagnostics):
            ranking = complete_ranking(joint_base(snapshot), points, valid,
                                       truth, settings['ranking'])
            return points, diagnostics, ranking

        off_points, off_diagnostics, off_ranking = _snapshot_evaluate(
            model, features, [full_plan] * 2, module_off,
            optimizer=optimizer, module_off=True)
        off_quality = qualify_joint_full(
            off_points, off_ranking, reference, evaluator, original, policy,
            valid, truth, descriptor['model_state_sha256'], binding,
            full_plan.archive())
        off_archive = write_archive(output / 'arrays',
                                     f'candidate-{arm}-step{step}-module-off',
                                    {'points': off_points, 'ranking': off_ranking,
                                     'quality': off_quality,
                                     'diagnostics': _serialize(off_diagnostics),
                                     'evaluation_binding':
                                         reference['evaluation_binding']})
        require_qualified(off_quality)
    del fresh
    rows = []
    for repeat, control in enumerate(frozen):
        deadline()
        def evaluate(snapshot, points, diagnostics):
            metrics, ranking, hierarchy = _candidate_metrics(
                snapshot, points, evaluator, valid, truth, settings)
            return points, diagnostics, metrics, ranking, hierarchy

        points, diagnostics, metrics, ranking, hierarchy = _snapshot_evaluate(
            model, features, control['plans'], evaluate, optimizer=optimizer)
        evaluation_binding = _eval_binding(
            settings, policy, source_commit, binding['source_commit'],
            descriptor['sha256'], descriptor['model_state_sha256'],
            control['plans'], 'selection_layer', 4, repeat, device)
        archive = write_archive(
            output / 'arrays', f'candidate-{arm}-step{step}-r{repeat}',
            {'points': points, 'plans': [p.archive() for p in control['plans']],
              'ranking': ranking, 'hierarchy': hierarchy,
              'evaluation_binding': evaluation_binding,
              'diagnostics': _serialize(diagnostics)})
        rows.append({'repeat': repeat, **metrics, 'archive': archive,
                     'evaluation_binding': evaluation_binding,
                     'common_S0_archive': control['archive']})
    return ({
        'step': step, 'micro_mrr': [row['micro_mrr'] for row in rows],
        'direct_order': [row['direct_order'] for row in rows],
        'full_metrics': full_metrics, 'full_reference': reference_archive,
        'fresh_full_archive': replay_archive,
        'fresh_full_qualification': replay_quality,
        'module_off_full_archive': off_archive,
        'module_off_full_qualification': off_quality,
        'evaluation_suite_core': require_compatible_evaluation_bindings(
            [reference['evaluation_binding']]
            + [control['evaluation_binding'] for control in frozen]
            + [row['evaluation_binding'] for row in rows]),
        'repeats': rows}, descriptor)


def _training_step(model, optimizer, groups, features, batch_groups,
                   batch_labels, positive_weights, root, plans, settings, arm,
                   device):
    t = settings['training']
    model.train()
    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    started = time.perf_counter()
    points, diagnostics = encode_joint(model, features, plans)
    task = torch.nn.functional.binary_cross_entropy_with_logits(
        joint_base(model).score(points, batch_groups.reshape(-1, 2).to(device)),
        batch_labels.reshape(-1).to(device))
    relation = (relation_gap_loss(
        points, batch_groups[:, 0].to(device), positive_weights.to(device),
        root, settings['module']['margin'], float(joint_base(model).curvature.item()))
        if settings['arms'][arm]['relation_weight'] else points.sum() * 0)
    penalty = (model.step_penalty(diagnostics)
               if isinstance(model, JointCorrectedHGCN) else points.sum() * 0)
    loss = (task + settings['arms'][arm]['relation_weight'] * relation
            + t['step_penalty_weight'] * penalty)
    if not torch.isfinite(loss):
        raise ValueError('nonfinite joint training loss')
    synchronize(device)
    forward_seconds = time.perf_counter() - started
    loss.backward()
    if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
           for values in groups.values() for parameter in values):
        raise ValueError('missing or nonfinite joint encoder/head/module gradient')
    before_clip = {name: _norm([p.grad for p in values])
                   for name, values in groups.items()}
    synchronize(device)
    backward_seconds = time.perf_counter() - started - forward_seconds
    parameters = [p for values in groups.values() for p in values]
    all_norm = float(torch.nn.utils.clip_grad_norm_(parameters,
                                                   t['grad_clip_norm']))
    previous = {name: [p.detach().clone() for p in values]
                for name, values in groups.items()}
    optimizer.step()
    synchronize(device)
    total_seconds = time.perf_counter() - started
    updates = {name: _norm([p.detach() - old for p, old in
                            zip(values, previous[name])])
               for name, values in groups.items()}
    before_clip['module'] = float(np.hypot(
        before_clip['module_layer0'], before_clip['module_layer1']))
    updates['module'] = float(np.hypot(
        updates['module_layer0'], updates['module_layer1']))
    if (not np.isfinite(all_norm) or any(not np.isfinite(value)
                                        for value in updates.values())
            or any(not torch.isfinite(p).all() for p in parameters)):
        raise ValueError('nonfinite joint parameter after update')
    row = {
        'task_loss': float(task.detach()), 'relation_loss': float(relation.detach()),
        'step_penalty': float(penalty.detach()), 'total_loss': float(loss.detach()),
        'group_grad_norm_before_clip': before_clip,
        'joint_grad_norm_before_clip': all_norm,
        'gradient_clipped': all_norm > t['grad_clip_norm'],
        'group_update_norm': updates,
        'diagnostics': _serialize(diagnostics),
        'forward_seconds': forward_seconds,
        'backward_seconds': backward_seconds,
        'optimizer_seconds': total_seconds - forward_seconds - backward_seconds,
        'core_step_seconds': total_seconds}
    row['recorded_step_seconds'] = time.perf_counter() - started
    return row


def _probe_readout(model, features, plans, full_plan, view, panel, valid,
                   truth, original, policy, settings, output, label,
                   source_commit, device, optimizer):
    state_sha = state_hash(model.state_dict())
    checkpoint_sha = _state_bytes_sha256(model)

    def full_readout(snapshot, full, full_diagnostics):
        evaluator, full_hierarchy = hierarchy_evaluator(
            view, panel, full.detach().cpu().numpy(), original)
        full_ranking = complete_ranking(joint_base(snapshot), full, valid, truth,
                                        settings['ranking'])
        return full, full_diagnostics, evaluator, full_hierarchy, full_ranking

    full, full_diagnostics, evaluator, full_hierarchy, full_ranking = (
        _snapshot_evaluate(model, features, [full_plan] * 2, full_readout,
                           optimizer=optimizer))

    def sample_readout(snapshot, sampled, diagnostics):
        metrics, sampled_ranking, hierarchy = _candidate_metrics(
            snapshot, sampled, evaluator, valid, truth, settings)
        return sampled, diagnostics, metrics, sampled_ranking, hierarchy

    sampled, diagnostics, metrics, sampled_ranking, hierarchy = (
        _snapshot_evaluate(model, features, plans, sample_readout,
                           optimizer=optimizer))
    full_binding = _eval_binding(
        settings, policy, source_commit, source_commit, checkpoint_sha,
        state_sha, [full_plan] * 2, 'probe_full', None, None, device,
        checkpoint_artifact_kind='serialized_state_fixture_or_probe')
    sample_binding = _eval_binding(
        settings, policy, source_commit, source_commit, checkpoint_sha,
        state_sha, plans, 'selection_layer', plans[0].fanout, 0, device,
        checkpoint_artifact_kind='serialized_state_fixture_or_probe')
    full_archive = write_archive(
        output / 'arrays', label + '-full',
        {'points': full, 'ranking': full_ranking, 'hierarchy': full_hierarchy,
         'evaluation_binding': full_binding,
         'panel_design': evaluator.archive_design(),
         'full_plan': full_plan.archive(), 'diagnostics': _serialize(full_diagnostics)})
    sample_archive = write_archive(
        output / 'arrays', label + '-sample',
        {'points': sampled, 'ranking': sampled_ranking, 'hierarchy': hierarchy,
         'evaluation_binding': sample_binding,
         'plans': [p.archive() for p in plans],
         'diagnostics': _serialize(diagnostics)})
    return {'full_archive': full_archive, 'sample_archive': sample_archive,
            'evaluation_bindings': [full_binding, sample_binding],
            'sample_metrics': metrics,
            'full_metrics': {'micro_mrr': full_ranking['query_micro_mrr'],
                             'direct_order': full_hierarchy['direct']['metrics']['score']}}


def _probe_terminal_replay(model, template, init_seeds, features, full_plan,
                           view, panel, valid, truth, original, policy,
                           settings, seed, arm, source_commit, output, device,
                           steps):
    """Roundtrip discarded weights in memory; archive only full readouts."""
    state_sha = state_hash(model.state_dict())
    binding = {'protocol': settings['protocol'], 'config_sha256': canonical(settings),
               'source_commit': source_commit, 'seed': seed, 'arm': arm,
               'probe_step': steps,
               'original_checkpoint_sha256':
               policy['origins'][str(seed)]['best_checkpoint_sha256'],
               'weights_discarded': True}
    reference_archive, evaluator, _ = _full_reference(
        model, features, full_plan, valid, truth, view, panel, original, policy,
        settings, binding, state_sha, output, 'probe-terminal-own-F',
        source_commit, device, phase='probe_full')
    reference = read_archive(output / 'arrays', reference_archive)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    serialized = stream.getvalue()
    serialized_sha = hashlib.sha256(serialized).hexdigest()
    stream.seek(0)
    fresh = _new_model(copy.deepcopy(template).to(device),
                       settings['arms'][arm]['structure'], settings, init_seeds)
    fresh.load_state_dict(torch.load(stream, map_location='cpu',
                                     weights_only=True), strict=True)
    if state_hash(fresh.state_dict()) != state_sha:
        raise ValueError('discarded probe serialized weight roundtrip changed state')
    def replay(snapshot, points, diagnostics):
        ranking = complete_ranking(joint_base(snapshot), points, valid, truth,
                                   settings['ranking'])
        return points, diagnostics, ranking

    points, diagnostics, ranking = _snapshot_evaluate(
        fresh, features, [full_plan] * 2, replay)
    quality = qualify_joint_full(
        points, ranking, reference, evaluator, original, policy, valid, truth,
        state_sha, binding, full_plan.archive())
    archive = write_archive(output / 'arrays', 'probe-terminal-fresh-full',
                             {'points': points, 'ranking': ranking,
                              'quality': quality,
                              'evaluation_binding':
                                  reference['evaluation_binding'],
                             'diagnostics': _serialize(diagnostics),
                             'reference': reference_archive})
    require_qualified(quality)
    del fresh, stream, serialized
    return {'reference_archive': reference_archive, 'fresh_full_archive': archive,
            'fresh_full_qualification': quality, 'state_sha256': state_sha,
            'serialized_state_sha256': serialized_sha,
            'weights_discarded': True}


def train_arm(data, base, original_reference, original_binding,
              original_state_sha, original, policy, prepared_root, settings,
              seed, arm, output, device, upstream, source_commit,
              *, probe_steps=None):
    if seed not in (11, 23) or arm not in ARMS:
        raise ValueError('registered joint starting model and arm required')
    if (probe_steps is not None
            and ((seed, arm) != (11, 'three_relation')
                 or type(probe_steps) is not int or not 1 <= probe_steps <= 32)):
        raise ValueError('only bounded discarded seed11 three-relation probe allowed')
    streams = NamedStreams()
    (features, valid, truth, original_evaluator, original_quality, full_plan,
     original_v2_reference) = _original_v2_control(
        data, base, original_reference, original_binding, original_state_sha,
        original, policy, prepared_root, settings, seed, output, device,
        source_commit)
    template = copy.deepcopy(base).cpu()
    init_seeds = [streams.seed('module_init', seed, layer=layer)
                  for layer in (0, 1)]
    structure = settings['arms'][arm]['structure']
    model = _new_model(base, structure, settings, init_seeds)
    if state_hash(joint_base(model).state_dict()) != original_state_sha:
        raise ValueError('joint arm did not start from exact original HGCN/head')
    hidden_init = (state_hash({
        f'layer{i}.{name}': tensor for i, correction in enumerate(model.corrections)
        for name, tensor in correction.coefficients[0].state_dict().items()})
        if isinstance(model, JointCorrectedHGCN) else None)
    optimizer, groups = _optimizer(model, settings)
    root, root_id, train_view_hash = _train_root(data)
    positives = data['query_groups'][:, 0].to(torch.long)
    relation_weights = training_relation_weights(positives, len(data['nodes']))
    view, panel = load_panel_design(prepared_root, original)
    frozen = (None if probe_steps is not None else _development_plans(
        data, base, features, original_evaluator, valid, truth, settings,
        policy, streams, seed, output, source_commit, original_binding,
        original_state_sha, device, optimizer=optimizer))
    common_s0 = (None if frozen is None else float(np.mean(
        [row['metrics']['direct_order'] for row in frozen])))
    probe_plans = (_plans(data['neighbors'], streams, 'selection_layer', seed,
                          fanout=4, repeat=0)
                   if probe_steps is not None else None)
    probe_readouts = ([ _probe_readout(
        model, features, probe_plans, full_plan, view, panel, valid, truth,
        original, policy, settings, output, 'probe-step0', source_commit,
        device, optimizer)]
        if probe_plans is not None else None)
    evaluations, checkpoints, history = [], {}, []
    if frozen is not None:
        candidate, cp = _evaluate_candidate(
            model, template, optimizer, 0, frozen, features, full_plan,
            view, panel, valid, truth, original, policy, settings, seed, arm,
            init_seeds, source_commit, output, device, streams)
        evaluations.append(candidate)
        checkpoints[0] = cp
    steps = probe_steps if probe_steps is not None else settings['training']['steps']
    for step in range(1, steps + 1):
        deadline()
        synchronize(device)
        pipeline_started = time.perf_counter()
        batch_seed = streams.seed('train_batch', seed, step=step, fanout=4)
        chosen = torch.randperm(
            len(data['query_groups']), generator=_generator(batch_seed))[
                :settings['training']['batch_positives']]
        batch_groups = data['query_groups'][chosen]
        masked = mask_graph(data['neighbors'], batch_groups[:, 0].tolist())
        plans = _plans(masked, streams, 'train_layer', seed, step=step, fanout=4)
        synchronize(device)
        plan_seconds = time.perf_counter() - pipeline_started
        row = _training_step(
            model, optimizer, groups, features, batch_groups,
            data['labels'][chosen], relation_weights[chosen], root,
            plans, settings, arm, device)
        row.update({
            'step': step, 'batch_seed63': batch_seed,
            'batch_indices': chosen.tolist(),
            'layer_plan_seed63': [
                streams.seed('train_layer', seed, step=step, fanout=4, layer=layer)
                for layer in (0, 1)],
            'layer_plan_graph_hash': [plan.graph_hash for plan in plans],
            'sampling_pipeline_seconds': plan_seconds,
            'whole_step_seconds': time.perf_counter() - pipeline_started})
        if probe_steps is not None and step in (1, steps):
            def relation_readout(snapshot, current, diagnostics):
                selected_positive = batch_groups[:, 0].to(device)
                anchor = current[root:root + 1].detach().expand(
                    len(selected_positive), -1)
                parent = ball_distance_from_anchor(
                    anchor, current[selected_positive[:, 0]])
                child = ball_distance_from_anchor(
                    anchor, current[selected_positive[:, 1]])
                eps = settings['module']['eps']
                geometry = {
                    'parent_near_zero': int((parent <= eps).sum()),
                    'child_near_zero': int((child <= eps).sum()),
                    'near_zero_gap': int(((child - parent).abs() <= eps).sum())}
                return geometry

            row['relation_geometry'] = _snapshot_evaluate(
                model, features, plans, relation_readout,
                optimizer=optimizer)
            row['relation_geometry_evaluation_binding'] = _eval_binding(
                settings, policy, source_commit, source_commit,
                _state_bytes_sha256(model), state_hash(model.state_dict()),
                plans, 'train_layer_probe_readout', 4, None, device,
                plan_step=step,
                checkpoint_artifact_kind='serialized_state_fixture_or_probe')
        history.append(row)
        if frozen is not None and step in settings['selection']['steps']:
            candidate, cp = _evaluate_candidate(
                model, template, optimizer, step, frozen, features, full_plan,
                view, panel, valid, truth, original, policy, settings, seed,
                arm, init_seeds, source_commit, output, device, streams)
            evaluations.append(candidate)
            checkpoints[step] = cp
        if step % 16 == 0 or step == steps:
            atomic_json(output / 'progress.json', {
                'status': 'running', 'seed': seed, 'arm': arm,
                'completed_steps': step, 'history': history,
                'evaluations': evaluations})
        del masked, plans
    if probe_steps is not None:
        probe_readouts.append(_probe_readout(
            model, features, probe_plans, full_plan, view, panel, valid,
            truth, original, policy, settings, output, f'probe-step{steps}',
            source_commit, device, optimizer))
        if (not all(sum(row['group_update_norm'][name] for row in history) > 0
                    for name in ('encoder', 'head', 'module',
                                 'module_layer0', 'module_layer1'))
                or not all(sum(row['group_grad_norm_before_clip'][name]
                                for row in history) > 0
                           for name in ('encoder', 'head', 'module',
                                        'module_layer0', 'module_layer1'))):
            raise ValueError('discarded joint probe lacks three-group learning')
        terminal_replay = _probe_terminal_replay(
            model, template, init_seeds, features, full_plan, view, panel,
            valid, truth, original, policy, settings, seed, arm, source_commit,
            output, device, steps)
    else:
        terminal_replay = None
    selected = (None if frozen is None else choose_checkpoint(
        evaluations, common_s0,
        settings['selection']['direct_order_from_common_S0_floor']))
    selected_row = (None if selected is None else next(
        row for row in evaluations if row['step'] == selected['step']))
    all_evaluation_bindings = [original_quality['evaluation_binding']]
    if frozen is not None:
        all_evaluation_bindings.extend(
            row['evaluation_binding'] for row in frozen)
        all_evaluation_bindings.extend(
            repeat['evaluation_binding'] for candidate in evaluations
            for repeat in candidate['repeats'])
    if probe_readouts is not None:
        all_evaluation_bindings.extend(
            binding for readout in probe_readouts
            for binding in readout['evaluation_bindings'])
        all_evaluation_bindings.extend(
            row['relation_geometry_evaluation_binding'] for row in history
            if 'relation_geometry_evaluation_binding' in row)
    result = {
        'status': 'probe_complete' if probe_steps is not None else 'complete',
        'phase': 'probe' if probe_steps is not None else 'train',
        'source_commit': source_commit, 'config_sha256': canonical(settings),
        'evaluation_suite_core': require_compatible_evaluation_bindings(
            all_evaluation_bindings),
        'seed': seed, 'arm': arm, 'completed_steps': steps,
        'original_best_binding': original_binding,
        'original_best_state_sha256': original_state_sha,
        'original_F_qualification': original_quality,
        'original_v2_reference': original_v2_reference,
        'initial_base_state_sha256': original_state_sha,
        'module_hidden_initialization_sha256': hidden_init,
        'module_init_layer_seeds63': init_seeds,
        'train_root_id': root_id, 'train_root_index': root,
        'train_view_hash': train_view_hash,
        'root_source': 'train positives only; current sampled root detached in relation loss',
        'common_S0': None if frozen is None else {
            'mean_direct_order': common_s0,
            'repeats': [{'metrics': row['metrics'], 'archive': row['archive'],
                         'evaluation_binding': row['evaluation_binding']}
                        for row in frozen]},
        'history': history, 'evaluations': evaluations, 'selected': selected,
        'selected_checkpoint': None if selected is None else checkpoints[selected['step']],
        'selected_full_reference': None if selected_row is None else selected_row['full_reference'],
        'checkpoints': checkpoints, 'rng_manifest': streams.manifest(),
        'base_state_changed': state_hash(joint_base(model).state_dict()) != original_state_sha,
        'valid_usage': 'development_selection' if frozen is not None else 'probe_quality_only',
        'probe_weights_discarded': probe_steps is not None}
    if not result['base_state_changed']:
        raise ValueError('joint backbone/head did not update')
    if probe_readouts is not None:
        result['probe_complete_readouts'] = probe_readouts
        result['probe_terminal_fresh_replay'] = terminal_replay
        result['peak_allocated_bytes'] = (
            torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)
        result['peak_reserved_bytes'] = (
            torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None)
    atomic_json(output / 'result.json', result)
    return result


def reevaluate_completed_plain_arm(
        data, base, original_reference, original_binding, original_state_sha,
        original, policy, prepared_root, settings, seed, arm, output, device,
        evaluation_source_commit, checkpoint_root, legacy_run,
        legacy_result_sha256, input_index):
    """Reevaluate one immutable completed plain arm once under composite v2."""
    if seed != 11 or arm not in ('plain_task', 'plain_relation'):
        raise ValueError('only the two completed seed11 plain arms are eligible')
    if (input_index.get('evaluation_suite')
            != 'rtsc_hgcn_eval_composite_full_det_sampled_det_v2'
            or input_index.get('steps') != settings['selection']['steps']):
        raise ValueError('exact reviewed plain-v2 input index required')
    entries = [row for row in input_index.get('arms', [])
               if (row.get('seed'), row.get('arm')) == (seed, arm)]
    if len(entries) != 1:
        raise ValueError('unique indexed completed plain arm required')
    indexed = entries[0]
    training_source_commit = indexed.get('training_source_commit')
    if (not isinstance(training_source_commit, str)
            or training_source_commit != legacy_run.get('source_commit')
            or legacy_result_sha256 != indexed.get('training_result_sha256')
            or legacy_run.get('status') != 'complete'
            or legacy_run.get('phase') != 'train'
            or legacy_run.get('seed') != seed
            or legacy_run.get('arm') != arm
            or legacy_run.get('completed_steps') != 1024
            or legacy_run.get('config_sha256') != canonical(settings)
            or legacy_run.get('original_best_binding') != original_binding
            or legacy_run.get('original_best_state_sha256') != original_state_sha
            or len(legacy_run.get('history', [])) != 1024):
        raise ValueError('immutable completed legacy training provenance differs')
    indexed_checkpoints = {row['step']: row
                           for row in indexed.get('checkpoints', [])}
    if set(indexed_checkpoints) != set(settings['selection']['steps']):
        raise ValueError('all five indexed checkpoint files required')
    legacy_checkpoints = legacy_run.get('checkpoints', {})
    legacy_evaluations = legacy_run.get('evaluations', [])
    if [row.get('step') for row in legacy_evaluations] != settings['selection']['steps']:
        raise ValueError('five legacy candidate records required')

    (features, valid, truth, original_evaluator, original_quality, full_plan,
     original_v2_reference) = _original_v2_control(
        data, base, original_reference, original_binding, original_state_sha,
        original, policy, prepared_root, settings, seed, output, device,
        evaluation_source_commit)
    streams = NamedStreams()
    frozen = _development_plans(
        data, base, features, original_evaluator, valid, truth, settings,
        policy, streams, seed, output, evaluation_source_commit,
        original_binding, original_state_sha, device)
    common_s0 = float(np.mean(
        [row['metrics']['direct_order'] for row in frozen]))
    view, panel = load_panel_design(prepared_root, original)
    template = copy.deepcopy(base).cpu()
    init_seeds = [streams.seed('module_init', seed, layer=layer)
                  for layer in (0, 1)]
    candidates = []
    all_bindings = [original_quality['evaluation_binding']]
    all_bindings.extend(row['evaluation_binding'] for row in frozen)
    for legacy_row in legacy_evaluations:
        step = legacy_row['step']
        descriptor = legacy_checkpoints.get(str(step),
                                            legacy_checkpoints.get(step))
        indexed_descriptor = indexed_checkpoints[step]
        if (not isinstance(descriptor, dict)
                or descriptor.get('file') != indexed_descriptor['file']
                or descriptor.get('sha256') != indexed_descriptor['sha256']
                or file_hash(Path(checkpoint_root) / descriptor['file'])
                != indexed_descriptor['sha256']):
            raise ValueError('indexed immutable checkpoint byte identity differs')
        training_binding = _binding(
            settings, training_source_commit, policy, seed, arm, step)
        model, payload = _fresh_reload(
            template, 'plain', settings, init_seeds, checkpoint_root,
            descriptor, training_binding, device)
        reference_archive, evaluator, full_metrics = _full_reference(
            model, features, full_plan, valid, truth, view, panel, original,
            policy, settings, training_binding, payload['model_state_sha256'],
            output, f'v2-candidate-{arm}-step{step}-F',
            evaluation_source_commit, device, checkpoint=descriptor,
            phase='reevaluation_candidate_full')
        reference = read_archive(output / 'arrays', reference_archive)

        def replay(snapshot, replay_points, replay_diagnostics):
            replay_ranking = complete_ranking(
                snapshot, replay_points, valid, truth, settings['ranking'])
            return replay_points, replay_diagnostics, replay_ranking

        replay_points, replay_diagnostics, replay_ranking = _snapshot_evaluate(
            model, features, [full_plan] * 2, replay, module_off=True)
        replay_quality = qualify_joint_full(
            replay_points, replay_ranking, reference, evaluator, original,
            policy, valid, truth, payload['model_state_sha256'],
            training_binding, full_plan.archive())
        require_qualified(replay_quality)
        replay_archive = write_archive(
            output / 'arrays', f'v2-candidate-{arm}-step{step}-fresh-off', {
                'points': replay_points, 'ranking': replay_ranking,
                'diagnostics': _serialize(replay_diagnostics),
                'quality': replay_quality,
                'evaluation_binding': reference['evaluation_binding'],
                'reference': reference_archive,
            })
        legacy_reference = read_archive(
            Path(checkpoint_root) / 'arrays', legacy_row['full_reference'])
        _same_plan(full_plan.archive(), legacy_reference['full_plan'])
        bridge = _report_only_full_bridge(
            legacy_reference, reference['native_ball_points'],
            reference['ranking'], evaluator, original, valid, truth)
        rows = []
        for repeat, control in enumerate(frozen):
            def evaluate(snapshot, points, diagnostics):
                metrics, ranking, hierarchy = _candidate_metrics(
                    snapshot, points, evaluator, valid, truth, settings)
                return points, diagnostics, metrics, ranking, hierarchy

            points, diagnostics, metrics, ranking, hierarchy = (
                _snapshot_evaluate(model, features, control['plans'], evaluate))
            evaluation_binding = _eval_binding(
                settings, policy, evaluation_source_commit,
                training_source_commit, descriptor['sha256'],
                payload['model_state_sha256'], control['plans'],
                'selection_layer', 4, repeat, device)
            archive = write_archive(
                output / 'arrays',
                f'v2-candidate-{arm}-step{step}-r{repeat}', {
                    'points': points,
                    'plans': [plan.archive() for plan in control['plans']],
                    'ranking': ranking, 'hierarchy': hierarchy,
                    'diagnostics': _serialize(diagnostics),
                    'evaluation_binding': evaluation_binding,
                    'own_F_reference': reference_archive,
                })
            rows.append({'repeat': repeat, **metrics, 'archive': archive,
                         'evaluation_binding': evaluation_binding,
                         'common_S0_archive': control['archive']})
            all_bindings.append(evaluation_binding)
        all_bindings.append(reference['evaluation_binding'])
        candidates.append({
            'step': step,
            'micro_mrr': [row['micro_mrr'] for row in rows],
            'direct_order': [row['direct_order'] for row in rows],
            'full_metrics': full_metrics,
            'full_reference': reference_archive,
            'fresh_module_off_archive': replay_archive,
            'fresh_module_off_qualification': replay_quality,
            'legacy_full_reference': legacy_row['full_reference'],
            'legacy_to_v2_bridge': bridge,
            'checkpoint': descriptor,
            'checkpoint_training_source_commit': training_source_commit,
            'repeats': rows,
        })
        del model
    provisional = choose_checkpoint(
        candidates, common_s0,
        settings['selection']['direct_order_from_common_S0_floor'])
    selected_row = (None if provisional is None else next(
        row for row in candidates if row['step'] == provisional['step']))
    if selected_row is not None:
        selected_descriptor = selected_row['checkpoint']
        selected_source = Path(checkpoint_root) / selected_descriptor['file']
        selected_copy = Path(output) / selected_descriptor['file']
        with selected_source.open('rb') as source, selected_copy.open('xb') as target:
            shutil.copyfileobj(source, target, length=1 << 20)
        if file_hash(selected_copy) != selected_descriptor['sha256']:
            raise ValueError('provisional selected checkpoint copy differs')
    result = {
        'status': 'provisional_selection_pending_independent_raw_audit',
        'phase': 'reevaluate_completed_plain_v2',
        'accepted_for_downstream_use': False,
        'seed': seed, 'arm': arm,
        'evaluation_source_commit': evaluation_source_commit,
        'training_source_commit': training_source_commit,
        'config_sha256': canonical(settings),
        'legacy_result_sha256': legacy_result_sha256,
        'legacy_selected': legacy_run.get('selected'),
        'legacy_selected_checkpoint': legacy_run.get('selected_checkpoint'),
        'training_history_preserved_not_rewritten': True,
        'original_F_qualification': original_quality,
        'original_v2_reference': original_v2_reference,
        'common_S0': {
            'mean_direct_order': common_s0,
            'repeats': [{'metrics': row['metrics'], 'archive': row['archive'],
                         'evaluation_binding': row['evaluation_binding']}
                        for row in frozen],
        },
        'evaluations': candidates,
        'provisional_selected': provisional,
        'provisional_selected_checkpoint': (
            None if selected_row is None else selected_row['checkpoint']),
        'provisional_selected_checkpoint_copied_byte_exact':
            selected_row is not None,
        'provisional_selected_full_reference': (
            None if selected_row is None else selected_row['full_reference']),
        'selection_rule_applied_once_after_five_candidates_and_two_S0': True,
        'requires_independent_raw_audit_before_acceptance': True,
        'evaluation_suite_core': require_compatible_evaluation_bindings(
            all_bindings),
        'rng_manifest': streams.manifest(),
    }
    atomic_json(output / 'result.json', result)
    return result


def load_training_inputs(args):
    from .hgcn_rtsc_joint_entry import RESULT_FLAGS

    paths = {arm: getattr(args, flag) for arm, flag in RESULT_FLAGS.items()}
    if set(paths) != set(ARMS) or any(path is None for path in paths.values()):
        raise ValueError('all six selected joint training result paths required')
    release = json.loads(Path(args.release_record).read_bytes())
    reviews = release.get('training_reviews', {})
    return {arm: (Path(path).parent, json.loads(Path(path).read_bytes()),
                  reviews.get(arm))
            for arm, path in paths.items()}


def _load_selected_arms(arm_inputs, template, features, full_plan, view, panel,
                        valid, truth, original, policy, settings, seed,
                        source_commit, original_binding, original_state_sha,
                        device):
    models, references, evaluators, selections = {}, {}, {}, {}
    baseline = None
    paired_history = None
    common_plans = None
    hidden_initialization = None
    init_seeds = [NamedStreams().seed('module_init', seed, layer=layer)
                  for layer in (0, 1)]
    for arm in ARMS:
        item = arm_inputs[arm]
        if len(item) == 2:
            run_dir, run = item
            review = None
        elif len(item) == 3:
            run_dir, run, review = item
        else:
            raise ValueError('joint arm input tuple differs')
        reevaluated_plain = (
            arm.startswith('plain_')
            and run.get('status')
                == 'provisional_selection_pending_independent_raw_audit'
            and run.get('phase') == 'reevaluate_completed_plain_v2'
            and run.get('accepted_for_downstream_use') is False
            and run.get('requires_independent_raw_audit_before_acceptance') is True
            and isinstance(review, dict)
            and review.get('accepted') is True
            and review.get('composite_v2_reevaluation_accepted') is True
            and review.get('evaluation_source_commit') == source_commit
            and review.get('result_sha256')
                == file_hash(run_dir / 'result.json'))
        ordinary_training = (run.get('status') == 'complete'
                             and run.get('phase') == 'train')
        if (not (ordinary_training or reevaluated_plain)
                or run.get('seed') != seed or run.get('arm') != arm
                or (ordinary_training and run.get('completed_steps') != 1024)
                or (ordinary_training and run.get('source_commit') != source_commit)
                or (reevaluated_plain
                    and run.get('evaluation_source_commit') != source_commit)
                or run.get('config_sha256') != canonical(settings)
                or (ordinary_training
                    and run.get('original_best_binding') != original_binding)
                or (ordinary_training
                    and run.get('original_best_state_sha256') != original_state_sha)
                or (ordinary_training
                    and run.get('initial_base_state_sha256') != original_state_sha)
                or (ordinary_training and run.get('base_state_changed') is not True)
                or (ordinary_training
                    and run.get('module_init_layer_seeds63') != init_seeds)
                or len(run.get('history', [])) != (1024 if ordinary_training else 0)
                or [row.get('step') for row in run.get('evaluations', [])]
                != settings['selection']['steps']):
            raise ValueError('exact complete joint training arm required')
        selected_key = ('selected' if ordinary_training
                        else 'provisional_selected')
        checkpoint_key = ('selected_checkpoint' if ordinary_training
                          else 'provisional_selected_checkpoint')
        reference_key = ('selected_full_reference' if ordinary_training
                         else 'provisional_selected_full_reference')
        if (run.get(selected_key) is None or run.get(checkpoint_key) is None
                or run.get(reference_key) is None):
            raise ValueError('accepted selected joint checkpoint required')
        fingerprint = [(
            row['batch_seed63'], row['batch_indices'],
            row['layer_plan_seed63'], row['layer_plan_graph_hash'])
            for row in run.get('history', [])]
        if (ordinary_training and paired_history is not None
                and paired_history != fingerprint):
            raise ValueError('six joint training arms lost batch/graph pairing')
        if ordinary_training:
            paired_history = fingerprint
        s0 = run['common_S0']
        if s0 is None or len(s0['repeats']) != 2:
            raise ValueError('shared two-graph initial S0 required')
        run_bindings = [row.get('evaluation_binding')
                        for row in s0['repeats']]
        run_bindings.extend(
            repeat.get('evaluation_binding')
            for candidate in run['evaluations']
            for repeat in candidate.get('repeats', []))
        run_core = require_compatible_evaluation_bindings(run_bindings)
        if run.get('evaluation_suite_core') != run_core:
            raise ValueError('joint arm mixed or omitted evaluation suite binding')
        if baseline is not None and abs(baseline - s0['mean_direct_order']) > 1e-7:
            raise ValueError('six joint arms have different common S0 gate')
        baseline = s0['mean_direct_order']
        if not arm.startswith('plain_'):
            if (hidden_initialization is not None
                    and run.get('module_hidden_initialization_sha256')
                    != hidden_initialization):
                raise ValueError('six-arm single/three two-layer hidden start differs')
            hidden_initialization = run.get('module_hidden_initialization_sha256')
            if not isinstance(hidden_initialization, str):
                raise ValueError('joint module hidden initialization hash required')
        plans = [read_archive(run_dir / 'arrays', row['archive'])['plans']
                 for row in s0['repeats']]
        if common_plans is not None:
            for first, second in zip(common_plans, plans):
                for first_plan, second_plan in zip(first, second):
                    _same_plan(first_plan, second_plan)
        common_plans = plans
        selected = run[selected_key]
        expected_selection = choose_checkpoint(
            run['evaluations'], s0['mean_direct_order'],
            settings['selection']['direct_order_from_common_S0_floor'])
        if selected != expected_selection:
            raise ValueError('joint selected checkpoint violates common S0 gate')
        if selected['step'] not in settings['selection']['steps']:
            raise ValueError('registered selected joint checkpoint step required')
        if any((row.get('fresh_full_qualification')
                if ordinary_training else
                row.get('fresh_module_off_qualification', {})).get('accepted')
               is not True
               or (not arm.startswith('plain_')
                   and row.get('module_off_full_qualification', {}).get('accepted') is not True)
               for row in run['evaluations']):
            raise ValueError('all five joint candidate own-F controls required')
        chosen_row = next(
            (row for row in run['evaluations'] if row['step'] == selected['step']), None)
        chosen_module_off = (None if chosen_row is None else
                             chosen_row.get('module_off_full_qualification'))
        if (chosen_row is None or chosen_row['full_reference']
                != run[reference_key]
                or (chosen_row.get('fresh_full_qualification')
                    if ordinary_training else
                    chosen_row.get('fresh_module_off_qualification'))['accepted']
                is not True
                or (chosen_module_off is not None
                    and chosen_module_off['accepted'] is not True)):
            raise ValueError('selected joint checkpoint lacks own-F qualification')
        training_source_commit = (run.get('source_commit') if ordinary_training
                                  else run.get('training_source_commit'))
        binding = _binding(settings, training_source_commit, policy, seed, arm,
                           selected['step'])
        model, payload = _fresh_reload(
            template, settings['arms'][arm]['structure'], settings, init_seeds,
            run_dir, run[checkpoint_key], binding, device)
        reference = read_archive(run_dir / 'arrays',
                                 run[reference_key])
        if (reference['binding'] != binding
                or reference['model_state_sha256'] != payload['model_state_sha256']
                or reference['checkpoint'] != run[checkpoint_key]):
            raise ValueError('joint selected own-F reference/checkpoint mismatch')
        require_evaluation_binding(reference.get('evaluation_binding'),
                                   expected_core=run_core)
        evaluator, _ = hierarchy_evaluator(
            view, panel, reference['native_ball_points'], original)
        models[arm] = model
        references[arm] = reference
        evaluators[arm] = evaluator
        selections[arm] = {
            'step': selected['step'],
            'checkpoint': run[checkpoint_key],
            'full_reference': run[reference_key],
            'training_source_commit': training_source_commit,
            'reevaluation_review': review if reevaluated_plain else None}
    return models, references, evaluators, selections


def _qualified_selected_full(model, reference, evaluator, full_plan, features,
                             valid, truth, original, policy, settings, output,
                             seed, arm, source_commit):
    state_sha = state_hash(model.state_dict())
    binding = _binding(settings, reference['binding']['source_commit'], policy,
                       seed, arm,
                       reference['binding']['step'])
    def evaluate(snapshot, points, diagnostics):
        if any(row['eligible_nodes'] for row in diagnostics):
            raise ValueError('joint selected module changed full-neighborhood identity')
        ranking = complete_ranking(joint_base(snapshot), points, valid, truth,
                                   settings['ranking'])
        return points, diagnostics, ranking

    points, diagnostics, ranking = _snapshot_evaluate(
        model, features, [full_plan] * 2, evaluate)
    evaluation_binding = reference.get('evaluation_binding')
    require_evaluation_binding(evaluation_binding)
    checkpoint = reference.get('checkpoint')
    if not isinstance(checkpoint, dict):
        raise ValueError('selected v2 full reference checkpoint required')
    plan_coordinate = evaluation_binding['plans'][0]
    expected_evaluation_binding = _eval_binding(
        settings, policy, source_commit, binding['source_commit'],
        checkpoint['sha256'], state_sha, [full_plan] * 2,
        plan_coordinate['phase'], None, plan_coordinate['repeat'],
        features.device, plan_step=plan_coordinate['step'])
    if evaluation_binding != expected_evaluation_binding:
        raise ValueError('selected full reference evaluation binding differs')
    quality = qualify_joint_full(
        points, ranking, reference, evaluator, original, policy,
        valid, truth, state_sha, binding, full_plan.archive())
    archive = write_archive(
        output / 'arrays', f'selected-full-control-{arm}',
        {'points': points, 'ranking': ranking, 'quality': quality,
         'evaluation_binding': evaluation_binding,
         'diagnostics': _serialize(diagnostics)})
    require_qualified(quality)
    off_quality = None
    off_archive = None
    if isinstance(model, JointCorrectedHGCN):
        def module_off(snapshot, points, diagnostics):
            ranking = complete_ranking(
                joint_base(snapshot), points, valid, truth, settings['ranking'])
            return points, diagnostics, ranking

        off_points, off_diagnostics, off_ranking = _snapshot_evaluate(
            model, features, [full_plan] * 2, module_off, module_off=True)
        off_quality = qualify_joint_full(
            off_points, off_ranking, reference, evaluator, original, policy,
            valid, truth, state_sha, binding, full_plan.archive())
        off_archive = write_archive(
            output / 'arrays', f'selected-full-module-off-{arm}',
            {'points': off_points, 'ranking': off_ranking,
             'quality': off_quality,
             'diagnostics': _serialize(off_diagnostics),
             'evaluation_binding': evaluation_binding})
        require_qualified(off_quality)
    return quality, archive, off_quality, off_archive


def final_shard(data, base, original_reference, original_binding,
                original_state_sha, original, policy, prepared_root,
                settings, seed, repeat_start, output, device, source_commit,
                arm_inputs):
    if repeat_start not in (0, 4, 8, 12):
        raise ValueError('registered joint final four-repeat shard required')
    streams = NamedStreams()
    (features, valid, truth, _, original_quality, full_plan,
     original_v2_reference) = _original_v2_control(
        data, base, original_reference, original_binding, original_state_sha,
        original, policy, prepared_root, settings, seed, output, device,
        source_commit)
    view, panel = load_panel_design(prepared_root, original)
    template = copy.deepcopy(base).cpu()
    models, references, evaluators, selected = _load_selected_arms(
        arm_inputs, template, features, full_plan, view, panel, valid, truth,
        original, policy, settings, seed, source_commit, original_binding,
        original_state_sha, device)
    full_qualifications, module_off_controls, fixed = {}, {}, {}
    for arm in ARMS:
        quality, archive, off_quality, off_archive = _qualified_selected_full(
            models[arm], references[arm], evaluators[arm], full_plan,
            features, valid, truth, original, policy, settings, output,
            seed, arm, source_commit)
        full_qualifications[arm] = {'accepted': quality['accepted'],
                                    'archive': archive,
                                    'evaluation_binding':
                                        references[arm]['evaluation_binding'],
                                    'qualification': quality}
        if off_quality is not None:
            module_off_controls[arm] = {'accepted': off_quality['accepted'],
                                        'archive': off_archive,
                                        'evaluation_binding':
                                            references[arm]['evaluation_binding'],
                                        'qualification': off_quality}
        fixed[arm] = {
            'micro_mrr': references[arm]['ranking']['query_micro_mrr'],
            'direct_order': evaluators[arm].evaluate(
                references[arm]['native_ball_points'])['direct']['metrics']['score'],
            'reference': selected[arm]['full_reference'],
            'evaluation_binding': references[arm]['evaluation_binding'],
            'model_state_sha256': references[arm]['model_state_sha256']}
    rows = []
    for fanout in settings['final']['fanouts']:
        for repeat in range(repeat_start, repeat_start + 4):
            deadline()
            plans = _plans(data['neighbors'], streams, 'final_layer', seed,
                           fanout=fanout, repeat=repeat)
            for arm in ARMS:
                deadline()
                model = models[arm]
                def evaluate(snapshot, points, diagnostics):
                    metrics, ranking, hierarchy = _candidate_metrics(
                        snapshot, points, evaluators[arm], valid, truth, settings)
                    return points, diagnostics, metrics, ranking, hierarchy

                points, diagnostics, metrics, ranking, hierarchy = (
                    _snapshot_evaluate(model, features, plans, evaluate))
                descriptor = selected[arm]['checkpoint']
                evaluation_binding = _eval_binding(
                    settings, policy, source_commit,
                    descriptor['binding']['source_commit'],
                    descriptor['sha256'], descriptor['model_state_sha256'],
                    plans, 'final_layer', fanout, repeat, device)
                archive = write_archive(
                    output / 'arrays',
                    f'joint-final-seed{seed}-f{fanout}-r{repeat}-{arm}',
                    {'points': points, 'plans': [p.archive() for p in plans],
                      'ranking': ranking, 'hierarchy': hierarchy,
                      'evaluation_binding': evaluation_binding,
                     'diagnostics': _serialize(diagnostics),
                     'own_F_reference': selected[arm]['full_reference']})
                rows.append({'seed': seed, 'fanout': fanout, 'repeat': repeat,
                             'arm': arm, **metrics, 'archive': archive,
                             'evaluation_binding': evaluation_binding})
            atomic_json(output / 'progress.json', {
                'status': 'running', 'seed': seed, 'repeat_start': repeat_start,
                'completed_conditions': len(rows), 'rows': rows})
    if len(rows) != 72 or len(full_qualifications) != 6 or len(module_off_controls) != 4:
        raise ValueError('complete 72 samples and ten own-F full controls required')
    result = {
        'status': 'complete', 'phase': 'final', 'seed': seed,
        'repeat_start': repeat_start, 'source_commit': source_commit,
        'config_sha256': canonical(settings), 'rows': rows,
        'selected': selected, 'fixed_F': fixed,
        'evaluation_suite_core': require_compatible_evaluation_bindings(
            [original_quality['evaluation_binding']]
            + [references[arm]['evaluation_binding'] for arm in ARMS]
            + [row['evaluation_binding'] for row in rows]),
        'original_F_qualification': original_quality,
        'original_v2_reference': original_v2_reference,
        'full_qualifications': full_qualifications,
        'module_off_full_controls': module_off_controls,
        'rng_manifest': streams.manifest(),
        'statistical_unit': 'one paired whole-graph repeat within a starting checkpoint and budget'}
    atomic_json(output / 'result.json', result)
    return result


def _time_summary(rows, field):
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    if values.shape != (20,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('twenty finite matched benchmark measurements required')
    return {'n': 20, 'mean_seconds': float(values.mean()),
            'median_seconds': float(np.median(values)),
            'p90_seconds': float(np.percentile(values, 90)),
            'raw_seconds': values.tolist()}


def _require_clean_benchmark_baseline(device, expected_allocated):
    if torch.cuda.memory_allocated(device) != expected_allocated:
        raise ValueError('benchmark previous arm polluted GPU memory baseline')


def benchmark_seed(data, base, original_reference, original_binding,
                   original_state_sha, original, policy, prepared_root,
                   settings, seed, output, device, source_commit, arm_inputs):
    if device.type != 'cuda':
        raise ValueError('registered matched benchmark requires one L40 GPU')
    streams = NamedStreams()
    (features, valid, truth, _, original_quality, full_plan,
     original_v2_reference) = _original_v2_control(
        data, base, original_reference, original_binding, original_state_sha,
        original, policy, prepared_root, settings, seed, output, device,
        source_commit)
    view, panel = load_panel_design(prepared_root, original)
    template = copy.deepcopy(base).cpu()
    models, references, evaluators, selected = _load_selected_arms(
        arm_inputs, template, features, full_plan, view, panel, valid, truth,
        original, policy, settings, seed, source_commit, original_binding,
        original_state_sha, device)
    controls = {}
    for arm in ARMS:
        quality, archive, off_quality, off_archive = _qualified_selected_full(
            models[arm], references[arm], evaluators[arm], full_plan,
            features, valid, truth, original, policy, settings, output,
            seed, arm, source_commit)
        controls[arm] = {'qualification': quality, 'archive': archive,
                         'module_off_qualification': off_quality,
                         'module_off_archive': off_archive}
    warmups, repetitions = (settings['benchmark'][key]
                            for key in ('warmups', 'repetitions'))
    inference = {arm: [] for arm in ARMS}
    inference_plans = []
    for round_index in range(warmups + repetitions):
        deadline()
        synchronize(device)
        started = time.perf_counter()
        plans = _plans(data['neighbors'], streams, 'benchmark', seed,
                       step=round_index, fanout=4, repeat=0)
        plans = _prepare_plans(plans, features)
        synchronize(device)
        plan_seconds = time.perf_counter() - started
        inference_plans.append({'round': round_index, 'seconds': plan_seconds,
                                'plan_graph_hash': [p.graph_hash for p in plans]})
        for offset in range(len(ARMS)):
            arm = ARMS[(round_index + offset) % len(ARMS)]
            model = models[arm]
            synchronize(device)
            total_begin = time.perf_counter()

            def benchmark_evaluate(snapshot):
                synchronize(device)
                begin = time.perf_counter()
                points, diagnostics = encode_evaluation(
                    snapshot, features, plans)
                synchronize(device)
                encoder_seconds = time.perf_counter() - begin
                ranking = complete_ranking(
                    joint_base(snapshot), points, valid, truth,
                    settings['ranking'])
                synchronize(device)
                end_to_end = time.perf_counter() - begin
                return (points, diagnostics, ranking, encoder_seconds,
                        end_to_end)

            points, diagnostics, ranking, encoder_seconds, end_to_end = (
                evaluate_frozen_snapshot(
                    model, lambda: copy.deepcopy(model), benchmark_evaluate))
            synchronize(device)
            snapshot_end_to_end = time.perf_counter() - total_begin
            snapshot_seconds = max(0., snapshot_end_to_end - end_to_end)
            if ranking['status'] != 'complete':
                raise ValueError('incomplete benchmark full ranking')
            descriptor = selected[arm]['checkpoint']
            evaluation_binding = _eval_binding(
                settings, policy, source_commit,
                descriptor['binding']['source_commit'], descriptor['sha256'],
                descriptor['model_state_sha256'], plans, 'benchmark', 4,
                0, device, plan_step=round_index)
            if round_index >= warmups:
                inference[arm].append({
                    'round': round_index - warmups,
                    'encoder_seconds': encoder_seconds,
                    'ranking_seconds': end_to_end - encoder_seconds,
                    'end_to_end_seconds': end_to_end,
                    'snapshot_construction_release_seconds': snapshot_seconds,
                    'snapshot_plus_end_to_end_seconds': snapshot_end_to_end,
                    'pipeline_plus_ranking_seconds': snapshot_end_to_end + plan_seconds,
                    'plan_seconds': plan_seconds,
                    'micro_mrr': ranking['query_micro_mrr'],
                    'evaluation_binding': evaluation_binding,
                    'module_diagnostics': _serialize(diagnostics)})
            del points, diagnostics, ranking
    training_models, optimizers, parameter_groups = {}, {}, {}
    for arm in ARMS:
        training_models[arm] = copy.deepcopy(models[arm])
        optimizer, groups = _optimizer(training_models[arm], settings)
        run_dir = arm_inputs[arm][0]
        descriptor = selected[arm]['checkpoint']
        binding = _binding(settings, selected[arm]['training_source_commit'],
                           policy, seed, arm,
                           selected[arm]['step'])
        payload = load_checkpoint(run_dir, descriptor, binding)
        if 'optimizer_state' not in payload['extra']:
            raise ValueError('joint benchmark requires selected optimizer state')
        optimizer.load_state_dict(payload['extra']['optimizer_state'])
        optimizers[arm], parameter_groups[arm] = optimizer, groups
    positives = data['query_groups'][:, 0].to(torch.long)
    relation_weights = training_relation_weights(positives, len(data['nodes']))
    root, root_id, view_hash = _train_root(data)
    training = {arm: [] for arm in ARMS}
    training_plans = []
    for round_index in range(warmups + repetitions):
        deadline()
        synchronize(device)
        started = time.perf_counter()
        batch_seed = streams.seed('benchmark', seed, step=round_index,
                                  fanout=4, repeat=1)
        chosen = torch.randperm(
            len(data['query_groups']), generator=_generator(batch_seed))[:128]
        groups = data['query_groups'][chosen]
        masked = mask_graph(data['neighbors'], groups[:, 0].tolist())
        plans = _plans(masked, streams, 'benchmark', seed, step=round_index,
                       fanout=4, repeat=2)
        plans = _prepare_plans(plans, features)
        groups_device = groups.to(device)
        labels_device = data['labels'][chosen].to(device)
        weights_device = relation_weights[chosen].to(device)
        synchronize(device)
        pipeline_seconds = time.perf_counter() - started
        training_plans.append({
            'round': round_index, 'batch_seed63': batch_seed,
            'batch_indices': chosen.tolist(),
            'plan_seed63': [
                streams.seed('benchmark', seed, step=round_index,
                             fanout=4, repeat=2, layer=layer)
                for layer in (0, 1)],
            'plan_graph_hash': [p.graph_hash for p in plans],
            'sampling_pipeline_seconds': pipeline_seconds})
        for offset in range(len(ARMS)):
            arm = ARMS[(round_index + offset) % len(ARMS)]
            row = _training_step(
                training_models[arm], optimizers[arm], parameter_groups[arm],
                features, groups_device, labels_device,
                weights_device, root, plans, settings, arm, device)
            if round_index >= warmups:
                training[arm].append({
                    'round': round_index - warmups, **row,
                    'sampling_pipeline_seconds': pipeline_seconds,
                    'whole_pipeline_seconds': row['recorded_step_seconds']
                    + pipeline_seconds})
        del masked, plans
    del training_models, optimizers, parameter_groups
    del optimizer, groups, groups_device, labels_device, weights_device
    for model in models.values():
        model.cpu()
    del model
    gc.collect()
    synchronize(device)
    torch.cuda.empty_cache()
    memory = {}
    chosen = torch.arange(128, dtype=torch.long)
    groups = data['query_groups'][chosen]
    masked = mask_graph(data['neighbors'], groups[:, 0].tolist())
    plans = _plans(masked, streams, 'benchmark', seed, step=999, fanout=4,
                   repeat=2)
    gc.collect()
    synchronize(device)
    torch.cuda.empty_cache()
    common_resident_allocated = torch.cuda.memory_allocated(device)
    for arm in ARMS:
        _require_clean_benchmark_baseline(device, common_resident_allocated)
        model = copy.deepcopy(models[arm]).to(device)
        optimizer, parameter_group = _optimizer(model, settings)
        run_dir = arm_inputs[arm][0]
        descriptor = selected[arm]['checkpoint']
        binding = _binding(settings, selected[arm]['training_source_commit'],
                           policy, seed, arm,
                           selected[arm]['step'])
        payload = load_checkpoint(run_dir, descriptor, binding)
        optimizer.load_state_dict(payload['extra']['optimizer_state'])
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)

        def memory_evaluate(snapshot):
            return encode_evaluation(snapshot, features, plans)

        points, memory_diagnostics = evaluate_frozen_snapshot(
            model, lambda: copy.deepcopy(model), memory_evaluate,
            optimizer=optimizer)
        synchronize(device)
        inference_peak_allocated = torch.cuda.max_memory_allocated(device)
        inference_peak_reserved = torch.cuda.max_memory_reserved(device)
        memory_evaluation_binding = _eval_binding(
            settings, policy, source_commit,
            descriptor['binding']['source_commit'], descriptor['sha256'],
            descriptor['model_state_sha256'], plans, 'benchmark', 4, 2,
            device, plan_step=999)
        del points, memory_diagnostics
        gc.collect()
        synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        training_baseline_allocated = torch.cuda.memory_allocated(device)
        training_baseline_reserved = torch.cuda.memory_reserved(device)
        _training_step(
            model, optimizer, parameter_group, features, groups,
            data['labels'][chosen], relation_weights[chosen], root,
            plans, settings, arm, device)
        memory[arm] = {
            'inference_allocated_baseline_bytes': baseline_allocated,
            'inference_reserved_baseline_bytes': baseline_reserved,
            'inference_peak_allocated_bytes': inference_peak_allocated,
            'inference_peak_reserved_bytes': inference_peak_reserved,
            'inference_includes_required_frozen_snapshot': True,
            'evaluation_binding': memory_evaluation_binding,
            'training_allocated_baseline_bytes': training_baseline_allocated,
            'training_reserved_baseline_bytes': training_baseline_reserved,
            'training_peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'training_peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
        del model, optimizer, parameter_group, payload
        gc.collect()
        synchronize(device)
        torch.cuda.empty_cache()
    summaries = {}
    for arm in ARMS:
        summaries[arm] = {
            'encoder_inference': _time_summary(inference[arm], 'encoder_seconds'),
            'end_to_end_ranking': _time_summary(inference[arm], 'end_to_end_seconds'),
            'snapshot_construction_release': _time_summary(
                inference[arm], 'snapshot_construction_release_seconds'),
            'snapshot_plus_end_to_end': _time_summary(
                inference[arm], 'snapshot_plus_end_to_end_seconds'),
            'inference_with_pipeline': _time_summary(
                inference[arm], 'pipeline_plus_ranking_seconds'),
            'training_forward': _time_summary(training[arm], 'forward_seconds'),
            'training_backward': _time_summary(training[arm], 'backward_seconds'),
            'training_optimizer': _time_summary(training[arm], 'optimizer_seconds'),
            'training_core_step': _time_summary(training[arm], 'core_step_seconds'),
            'training_recorded_step': _time_summary(
                training[arm], 'recorded_step_seconds'),
            'training_pipeline': _time_summary(training[arm], 'whole_pipeline_seconds')}
    result = {
        'status': 'complete', 'phase': 'benchmark', 'seed': seed,
        'source_commit': source_commit, 'config_sha256': canonical(settings),
        'device': str(device), 'hardware': torch.cuda.get_device_name(device),
        'cuda_visible_devices_bound': True, 'warmups': warmups,
        'measured_repetitions': repetitions,
        'interleaved_order': 'round-rotated six arms',
        'selected': selected, 'original_F_qualification': original_quality,
        'original_v2_reference': original_v2_reference,
        'evaluation_suite_core': require_compatible_evaluation_bindings(
            [original_quality['evaluation_binding']]
            + [references[arm]['evaluation_binding'] for arm in ARMS]
            + [row['evaluation_binding'] for values in inference.values()
               for row in values]
            + [row['evaluation_binding'] for row in memory.values()]),
        'full_controls': controls, 'inference': inference,
        'training': training, 'inference_plan_cost': inference_plans,
        'training_plan_cost': training_plans, 'memory': memory,
        'common_resident_allocated_bytes': common_resident_allocated,
        'summaries': summaries,
        'train_root_id': root_id, 'train_view_hash': view_hash,
        'benchmark_updates_saved_as_science_checkpoints': False,
        'rng_manifest': streams.manifest()}
    atomic_json(output / 'result.json', result)
    return result


def small_quality(upstream, settings, original, policy, device, output,
                  source_commit):
    """Synthetic original-runtime gate for six updates and own-F replay."""
    from types import SimpleNamespace

    torch.manual_seed(41)
    initial = MatureHGCN(upstream, 3, 4, 5, 1., 0., device).eval()
    initial_state = state_hash(initial.state_dict())
    template = copy.deepcopy(initial).cpu()
    features = torch.randn(8, 3, device=device) * .1
    neighbors = [[j for j in range(8) if j != i] for i in range(8)]
    groups = torch.tensor([
        [[1, 2], [3, 2], [4, 2], [5, 2], [6, 2]],
        [[1, 3], [2, 3], [4, 3], [5, 3], [6, 3]]], dtype=torch.long)
    labels = torch.tensor([[1., 0., 0., 0., 0.],
                           [1., 0., 0., 0., 0.]])
    positives = groups[:, 0]
    weights = training_relation_weights(positives, 8)
    masked = mask_graph(neighbors, positives.tolist())
    sampled = [make_plan(masked, 4, _generator(200 + layer))
               for layer in (0, 1)]
    full_plan = make_plan(neighbors, None, _generator(0))
    initial_sample = initial.encode(features, [
        plan.matrix(device=device) for plan in sampled])
    output_rows = {}
    hidden_states = {}
    query = [(1, 2), (1, 3)]
    truth = {2: {1}, 3: {1}}
    view = SimpleNamespace(nodes=list(range(8)), root=0,
                           reachable=set(range(8)))
    panel = {
        'pool_size': 2,
        'rows': [{'id': child, 'index': child,
                  'inclusion_probability': 1., 'pool_mean_weight': .5}
                 for child in (2, 3)],
        'relations': [
            {'child': 2, 'direct_parents': [1],
             'positive_distant_ancestors': [0]},
            {'child': 3, 'direct_parents': [1],
             'positive_distant_ancestors': [0]}]}
    mini_original = {'model': {'hidden': 4, 'c': 1.},
                     'prepared': {'nodes_count': 8}}
    init_seeds = [NamedStreams().seed('module_init', 11, layer=layer)
                  for layer in (0, 1)]
    for arm in ARMS:
        model = _new_model(copy.deepcopy(template).to(device),
                           settings['arms'][arm]['structure'], settings,
                           init_seeds)
        if state_hash(joint_base(model).state_dict()) != initial_state:
            raise ValueError('six joint quality arms lack identical original start')
        optimizer, parameter_groups = _optimizer(model, settings)
        zero_sample, _ = encode_joint(model, features, sampled)
        if not torch.equal(zero_sample, initial_sample):
            raise ValueError('joint zero-output sample differs from plain base')
        full_before, full_diagnostics = encode_joint(model, features,
                                                     [full_plan] * 2)
        plain_full = joint_base(model).encode(
            features, [full_plan.matrix(device=device)] * 2)
        if not torch.equal(full_before, plain_full) or any(
                row['eligible_nodes'] for row in full_diagnostics):
            raise ValueError('joint quality full-neighborhood identity failed')
        if isinstance(model, JointCorrectedHGCN):
            hidden_states[arm] = [
                {key: tensor.detach().cpu().clone()
                 for key, tensor in correction.coefficients[0].state_dict().items()}
                for correction in model.corrections]
            expected = 454 if arm.startswith('three_') else 386
            if sum(p.numel() for p in model.corrections.parameters()) != expected:
                raise ValueError('joint 454/386 correction parameters changed')
            if arm.startswith('single_'):
                probe_features = torch.randn(3, 10, device=device)
                coefficients = model.corrections[0].coefficients(probe_features)
                if (coefficients.shape != (3, 3)
                        or torch.count_nonzero(coefficients[:, 1:])):
                    raise ValueError('single direction unexpectedly uses b2 or b3')
                del probe_features, coefficients
        steps = []
        for _ in range(2):
            steps.append(_training_step(
                model, optimizer, parameter_groups, features, groups, labels,
                weights, 0, sampled, settings, arm, device))
        if (state_hash(joint_base(model).state_dict()) == initial_state
                or not all(sum(step['group_update_norm'][name] for step in steps) > 0
                           for name in ('encoder', 'head'))
                or isinstance(model, JointCorrectedHGCN)
                and not all(sum(step['group_update_norm'][name]
                                for step in steps) > 0
                            for name in ('module', 'module_layer0',
                                         'module_layer1'))):
            raise ValueError('joint quality backbone/head/module failed to update')
        binding = {'protocol': settings['protocol'], 'source_commit': source_commit,
                   'scope': 'synthetic six-arm quality fixture', 'arm': arm}
        checkpoint = save_checkpoint(
            output, f'joint-quality-{arm}.pt', model, binding,
            {'optimizer_state': optimizer.state_dict()})

        def own_readout(snapshot, own_full, diagnostics):
            own_ranking = complete_ranking(
                joint_base(snapshot), own_full, query, truth,
                {'candidate_chunk': 8, 'max_seconds': 60})
            return own_full, diagnostics, own_ranking

        own_full, diagnostics, own_ranking = _snapshot_evaluate(
            model, features, [full_plan] * 2, own_readout,
            optimizer=optimizer)
        evaluator = HierarchyPanel(view, panel,
                                   own_full.detach().cpu().numpy())
        evaluation_binding = _eval_binding(
            settings, policy, source_commit, source_commit,
            checkpoint['sha256'], checkpoint['model_state_sha256'],
            [full_plan] * 2, 'quality_full', None, None, device)
        reference = {
            'binding': binding,
            'evaluation_binding': evaluation_binding,
            'model_state_sha256': checkpoint['model_state_sha256'],
            'native_ball_points': own_full.detach().cpu().numpy(),
            'ranking': own_ranking, 'full_plan': full_plan.archive()}
        fresh_base = copy.deepcopy(template).to(device)
        fresh = _new_model(fresh_base, settings['arms'][arm]['structure'],
                           settings, init_seeds)
        loaded = load_checkpoint(output, checkpoint, binding)
        fresh.load_state_dict(loaded['model_state'], strict=True)
        def fresh_readout(snapshot, replay, replay_diagnostics):
            replay_ranking = complete_ranking(
                joint_base(snapshot), replay, query, truth,
                {'candidate_chunk': 8, 'max_seconds': 60})
            return replay, replay_diagnostics, replay_ranking

        replay, replay_diagnostics, replay_ranking = _snapshot_evaluate(
            fresh, features, [full_plan] * 2, fresh_readout)
        quality = qualify_joint_full(
            replay, replay_ranking, reference, evaluator, mini_original,
            policy, query, truth, checkpoint['model_state_sha256'],
            binding, full_plan.archive())
        require_qualified(quality)
        archive = write_archive(
            output / 'arrays', f'joint-quality-{arm}-full',
            {'reference': reference, 'fresh_points': replay,
             'fresh_ranking': replay_ranking, 'quality': quality,
             'evaluation_binding': evaluation_binding,
             'diagnostics': _serialize(diagnostics),
             'fresh_diagnostics': _serialize(replay_diagnostics)})
        output_rows[arm] = {
            'initial_base_state_sha256': initial_state,
            'checkpoint': checkpoint,
            'full_archive': archive,
            'fresh_own_F_qualification': quality,
            'evaluation_binding': evaluation_binding,
            'group_grad_norm_before_clip': [
                step['group_grad_norm_before_clip'] for step in steps],
            'group_update_norm': [step['group_update_norm'] for step in steps],
            'zero_output_exact': True,
            'full_identity_exact_on_synthetic_graph': True}
        del model, fresh, fresh_base, optimizer, parameter_groups
        del zero_sample, full_before, full_diagnostics, plain_full
        del own_full, diagnostics, replay, replay_diagnostics
        del loaded, reference, evaluator, own_ranking, replay_ranking
    for layer in (0, 1):
        for name in ('weight', 'bias'):
            baseline = hidden_states['three_task'][layer][name]
            if any(not torch.equal(baseline, hidden_states[arm][layer][name])
                   for arm in ('three_relation', 'single_task',
                               'single_relation')):
                raise ValueError('joint two-layer/two-loss hidden initialization differs')
    dummy = {
        (seed, fanout, repeat, arm): {
            'micro_mrr': .1 + .00001 * repeat
            + .0001 * ARMS.index(arm),
            'direct_order': .5 + .00001 * repeat
            + .0001 * ARMS.index(arm)}
        for seed in (11, 23) for fanout in (4, 8, 16)
        for repeat in range(16) for arm in ARMS}
    main, protection, relation = final_families(dummy)
    if (len(main), len(protection), len(relation)) != (16, 32, 36):
        raise ValueError('registered joint inference family sizes differ')
    memory_isolation = {'status': 'not_applicable_cpu'}
    if device.type == 'cuda':
        del initial_sample
        initial.cpu()
        gc.collect()
        synchronize(device)
        torch.cuda.empty_cache()
        common = torch.cuda.memory_allocated(device)
        _require_clean_benchmark_baseline(device, common)
        previous_full_points = torch.empty((82115, 128), device=device)
        detected = False
        try:
            _require_clean_benchmark_baseline(device, common)
        except ValueError:
            detected = True
        if not detected:
            raise ValueError('CUDA quality failed to detect prior full-point contamination')
        del previous_full_points
        gc.collect()
        synchronize(device)
        torch.cuda.empty_cache()
        _require_clean_benchmark_baseline(device, common)
        memory_isolation = {'status': 'passed',
                            'detected_residual_full_points': True,
                            'restored_common_allocated_bytes': common}
    result = {
        'status': 'passed', 'device': str(device),
        'scope': 'synthetic joint quality; no scientific model effect',
        'six_arms': output_rows,
        'evaluation_suite_core': require_compatible_evaluation_bindings(
            [row['evaluation_binding'] for row in output_rows.values()]),
        'three_single_hidden_initialization_matched': True,
        'family_sizes': [len(main), len(protection), len(relation)],
        'cuda_memory_isolation': memory_isolation,
        'encoder_head_and_module_gradient_checked': True}
    atomic_json(output / 'quality.json', result)
    return result


def main():
    from .hgcn_quality import check_runtime
    from .hgcn_replay import load_policy
    from .hgcn_rtsc_joint_entry import parser, training_result_paths
    from .hgcn_upstream import load_upstream

    args = parser().parse_args()
    output = Path(args.output)
    report = {'status': 'running', 'phase': args.phase,
              'scope': 'R-TSC/HGCN joint v1 only', 'science_released': True}
    started = time.perf_counter()
    try:
        deadline()
        original = json.loads(args.original_config.read_bytes())
        policy = load_policy(args.policy, original)
        settings = json.loads(args.config.read_bytes())
        old = json.loads(args.old_config.read_bytes())
        validate_config(settings, original, policy, old)
        release = verify_release(
            settings, original, policy, old, args.phase, args.release_record,
            args.source_commit, seed=args.seed, arm=args.arm,
            repeat_start=args.repeat_start, original_run=args.original_training_run,
            training_runs=(training_result_paths(args)
                           if args.phase in ('final', 'benchmark') else None),
            input_index=args.input_index,
            legacy_training_result=args.legacy_training_result)
        before = source_hashes()
        device = check_runtime(
            original, 'cpu_quality' if args.phase == 'cpu_quality'
            else 'cuda_quality')
        upstream = load_upstream(args.upstream_root, args.upstream_manifest)
        report.update(release=release, device=str(device),
                      upstream=upstream.identity)
        atomic_json(output / 'run.json', report)
        if args.phase in ('cpu_quality', 'cuda_quality'):
            result = small_quality(upstream, settings, original, policy,
                                   device, output, args.source_commit)
        else:
            data, base, reference, binding, state_sha, best = original_inputs(
                args.original_training_run, original, policy, args.seed,
                upstream, device, args.prepared_root)
            report['original_best'] = {
                'step': best['step'],
                'checkpoint_sha256': best['checkpoint']['sha256'],
                'state_sha256': state_sha}
            if args.phase in ('probe', 'train'):
                result = train_arm(
                    data, base, reference, binding, state_sha, original,
                    policy, args.prepared_root, settings, args.seed, args.arm,
                    output, device, upstream, args.source_commit,
                    probe_steps=args.probe_steps if args.phase == 'probe' else None)
            elif args.phase == 'reevaluate':
                legacy_result_path = Path(args.legacy_training_result)
                legacy_run = json.loads(legacy_result_path.read_bytes())
                input_index = json.loads(Path(args.input_index).read_bytes())
                result = reevaluate_completed_plain_arm(
                    data, base, reference, binding, state_sha, original,
                    policy, args.prepared_root, settings, args.seed, args.arm,
                    output, device, args.source_commit,
                    Path(args.checkpoint_root), legacy_run,
                    file_hash(legacy_result_path), input_index)
            else:
                inputs = load_training_inputs(args)
                if args.phase == 'final':
                    result = final_shard(
                        data, base, reference, binding, state_sha, original,
                        policy, args.prepared_root, settings, args.seed,
                        args.repeat_start, output, device, args.source_commit,
                        inputs)
                else:
                    result = benchmark_seed(
                        data, base, reference, binding, state_sha, original,
                        policy, args.prepared_root, settings, args.seed,
                        output, device, args.source_commit, inputs)
        if source_hashes() != before:
            raise ValueError('joint source changed during worker execution')
        load_upstream(args.upstream_root, args.upstream_manifest)
        report['status'] = 'passed'
        report['result_status'] = result['status']
        report['result_file'] = ('quality.json' if args.phase.endswith('quality')
                                 else 'result.json')
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        report['peak_allocated_bytes'] = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
        atomic_json(output / 'run.json', report)


if __name__ == '__main__':
    main()

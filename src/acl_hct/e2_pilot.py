"""Approved, bounded E2 entry checks then local fixed-input sampling only."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from .backbone import LorentzMeanNetwork, make_plan
from .development_view import DEGREE_LABELS, degree_bin, load_development_view, make_panels, support_groups
from .evaluate_checkpoint import file_sha256, load_verified_checkpoint
from .frozen_forward import FrozenForward, PlanStreams
from .frozen_stats import TangentStream, outward_directions, promote_points
from .frozen_structure import radial_relations
from .geometry import dot, exp, log, norm2
from .mechanisms import dependency_hashes, source_identity
from .protocols import digest
from .ranking import filtered_parent_ranks
from .train import load_prepared, synchronize


LIMITS = {
    'native_constraint': 2e-4, 'promotion_displacement': 1e-4,
    'fp64_constraint': 1e-10, 'tangent_constraint': 1e-9,
    'roundtrip': 1e-10, 'native_fp64_distance': 1e-4,
    'unchanged_output_difference': 1e-5, 'mse_decomposition': 1e-10,
    'variance_mse_minimum': -1e-12,
    'full_valid_reproduction': 1e-8,
}
PILOT = {'fanout': 16, 'repetitions': 16, 'sampling_seed': 2026091605,
         'namespace': 'E2-local-entry-v1', 'conditions': ['local_L1', 'F/S'],
         'panel_target': 1000, 'threads': 2, 'internal_seconds': 1500,
         'outer_seconds': 1800, 'ranking_seconds': 180, 'candidate_chunk': 4096,
         'max_reach_visits': 2000000}


def jsonable(value):
    if isinstance(value, torch.Tensor): return jsonable(value.detach().cpu().tolist())
    if isinstance(value, dict): return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(item) for item in value]
    if isinstance(value, float) and math.isnan(value): return None  # Explicit direction NA only.
    if isinstance(value, float) and not math.isfinite(value): raise ValueError('infinite report value')
    return value


def summarize(values):
    values = values.detach().double().flatten()
    if not len(values): return {'count': 0, 'minimum': None, 'mean': None, 'maximum': None}
    if not torch.isfinite(values).all(): raise ValueError('nonfinite numerical evidence')
    return {'count': len(values), 'minimum': float(values.min()),
            'mean': float(values.mean()), 'maximum': float(values.max())}


def check_max(checks, name, values, limit):
    summary = summarize(values)
    # Empty numerical observations cannot pass an entry check.
    passed = summary['count'] > 0 and max(abs(summary['minimum']), abs(summary['maximum'])) <= limit
    checks.append({'name': name, 'status': 'passed' if passed else 'failed',
                   'limit_absolute': limit, 'observed': summary})


def validate_config(config):
    if (config.get('protocol') != 'E2-entry-local-pilot-v1' or config.get('pilot') != PILOT
            or config.get('numerical_limits') != LIMITS or set(config.get('checkpoints', {})) != {'11', '23'}):
        raise ValueError('fixed E2 pilot scope and numerical limits required')
    for seed, row in config['checkpoints'].items():
        if row['step'] != 768 or row['training_config']['seed'] != int(seed):
            raise ValueError('both preselected best768 checkpoints required')
    return digest(config)


def verify_approval(record, config, identity):
    if (record.get('user_authorized') is not True or not record.get('user_message_reference')
            or record.get('scope') != config['protocol']
            or record.get('config_sha256') != digest(config)
            or record.get('source_commit') != identity['source_commit']
            or record.get('quality_review_passed') is not True
            or record.get('entry_criteria_frozen') is not True):
        raise ValueError('real user authorization and fixed-release quality review required')


def verify_selection(checkpoint, baseline, spec, data):
    """Artifact hashes are external trust anchors; do not reselect on new results."""
    if (digest(checkpoint['config']) != digest(spec['training_config']) or digest(baseline['config']) != digest(spec['training_config'])
            or checkpoint['completed_steps'] != spec['step']
            or checkpoint['manifest_hash'] != data['manifest_hash']
            or checkpoint['valid_hash'] != data['valid_hash']
            or baseline['manifest_hash'] != data['manifest_hash']
            or baseline['validation_queries_hash'] != data['valid_hash']):
        raise ValueError('checkpoint/config/prepared selection provenance mismatch')
    if (baseline['source']['source_commit'] != spec['training_commit']
            or baseline['config']['evaluation'] != 'full_validation'
            or baseline['status'] != 'step_limit_reached'
            or baseline['completed_steps'] != baseline['config']['max_steps']):
        raise ValueError('complete original bounded baseline evidence required')
    evaluations = baseline['evaluations']
    if not evaluations or any(row['status'] != 'complete' or row['purpose'] != 'full_validation'
                              or row['completed_queries'] != len(data['valid']) for row in evaluations):
        raise ValueError('every scheduled selection evaluation must be complete full valid')
    # Python max preserves the original first winner in an exact tie.
    winner = max(evaluations, key=lambda row: row['query_micro_mrr'])
    if (winner['step'] != spec['step'] or winner['query_micro_mrr'] != checkpoint['best_full_valid_mrr']
            or winner['query_micro_mrr'] != baseline['best_full_valid_mrr']
            or checkpoint['selection_status'] != 'selected by complete filtered all-candidate validation query-micro MRR'):
        raise ValueError('checkpoint is not the recorded selected best')
    return {'status': 'passed', 'selected_step': winner['step'],
            'historical_full_valid_mrr': winner['query_micro_mrr'],
            'rule': 'first maximum complete valid micro MRR in immutable bounded baseline record'}


def diagnostic_groups(view, fanout):
    groups = support_groups(view.neighbors, fanout)
    for label in DEGREE_LABELS:
        groups['degree_' + label] = [i for i, row in enumerate(view.neighbors)
                                    if DEGREE_LABELS[degree_bin(len(row))] == label]
    groups['root_reachable'] = [i for i, node in enumerate(view.nodes) if node in view.reachable]
    groups['root_unknown'] = [i for i, node in enumerate(view.nodes) if node not in view.reachable]
    return groups


def validation_reproduction(current, historical):
    def pairs(row): return [(item['parent'], item['child']) for item in row['rows']]
    actual, expected = pairs(current), pairs(historical)
    coverage = (current['status'] == historical['status'] == 'complete'
                and len(actual) == len(set(actual)) == len(expected) == len(set(expected))
                and set(actual) == set(expected)
                and current['completed_queries'] == historical['completed_queries'] == len(actual)
                and current['completed_children'] == historical['completed_children'] == len({b for a, b in actual}))
    differences = {key: current[key] - historical[key] for key in ('query_micro_mrr', 'child_macro_mrr')}
    passed = coverage and all(math.isfinite(value) and abs(value) <= LIMITS['full_valid_reproduction']
                              for value in differences.values())
    return {'status': 'passed' if passed else 'failed', 'query_coverage_identical': coverage,
            'current_query_pairs_hash': digest(sorted(actual)), 'historical_query_pairs_hash': digest(sorted(expected)),
            'current_queries': len(actual), 'historical_queries': len(expected),
            'metric_differences': differences, 'absolute_tolerance': LIMITS['full_valid_reproduction']}


def promotion_checks(checks, prefix, promotion):
    check_max(checks, prefix + '/native_constraint', promotion['constraint_before'], LIMITS['native_constraint'])
    check_max(checks, prefix + '/projection', promotion['ambient_projection_displacement'], LIMITS['promotion_displacement'])
    check_max(checks, prefix + '/fp64_constraint', promotion['constraint_after'], LIMITS['fp64_constraint'])


def numerical_entry(forward):
    numerical = forward.numerical_audit(); checks = []; layers = []
    for i, row in enumerate(numerical['layers']):
        prefix = f'L{i+1}'
        promotion_checks(checks, prefix, row['native_promotion'])
        check_max(checks, prefix + '/rerun_constraint_before', row['fp64_rerun_promotion']['constraint_before'], LIMITS['fp64_constraint'])
        check_max(checks, prefix + '/rerun_constraint_after', row['fp64_rerun_promotion']['constraint_after'], LIMITS['fp64_constraint'])
        for key, limit in [('native_vs_fp64_full_distance', 'native_fp64_distance'),
                           ('self_log_norm', 'roundtrip'), ('log_exp_roundtrip_distance', 'roundtrip'),
                           ('log_tangent_constraint_residual', 'tangent_constraint')]:
            check_max(checks, prefix + '/' + key, row[key], LIMITS[limit])
        # Reaggregation with full plans must agree before local sampling is allowed.
        points, _ = forward.model.frozen_layer(forward.reference, i, forward.neighbors,
                                               forward.neighbors, max_padded_messages=forward.budget)
        check_max(checks, prefix + '/full_plan_identity', points - forward.reference['layers'][i]['output'],
                  LIMITS['unchanged_output_difference'])
        layers.append({key: summarize(row[key]) for key in (
            'native_vs_fp64_full_distance', 'self_log_norm', 'log_exp_roundtrip_distance',
            'log_tangent_constraint_residual', 'empirical_numerical_floor')})
    return numerical, {'status': 'passed' if all(r['status'] == 'passed' for r in checks) else 'failed',
                       'checks': checks, 'layers': layers, 'scope': numerical['scope']}


def compact_statistics(stats, affected):
    """All-node scalar evidence, but high-dimensional mean vectors only for A_f."""
    out = dict(stats)
    out['mean_offset'] = stats['mean_offset'][affected]
    out['mean_offset_node_indices'] = affected
    out['mean_offset_storage'] = 'A_f only; other local fixed-input offsets are structurally zero'
    return out


@torch.no_grad()
def diagnose(model, features, view, *, repetitions=16, fanout=16, seed=2026091605,
             namespace='E2-local-entry-v1', max_seconds=1500., budget=32768,
             panel_target=1000, candidate_chunk=4096, ranking_seconds=180., progress=None, expected_ranking=None):
    """Small fixtures use this core directly; real CLI additionally verifies approval/provenance."""
    if repetitions < 2 or repetitions > 16 or repetitions % 2 or fanout != 16:
        raise ValueError('local pilot requires even 2..16 fixture repetitions and fanout16')
    started = time.perf_counter(); device = features.device
    report = {'research_question': 'Is fixed-input sampling bias resolvable in these real frozen models?',
              'conclusion': 'pending evidence review', 'purpose_status': 'insufficient_evidence',
              'status': 'incomplete', 'entry_status': 'not_started', 'pilot_status': 'not_started',
              'scope': 'local L1 and full-input local L2 (F/S); no propagation, correction, N1, E3 or retraining',
              'requested_repetitions_per_layer': repetitions, 'completed_repetitions': {},
              'failures': [], 'numerical_limits': LIMITS, 'stopped_before': None,
              'next_experiment_authorized': False}
    def publish():
        report['elapsed_seconds'] = time.perf_counter() - started
        if progress: progress(report)
    def deadline(phase):
        if time.perf_counter() - started >= max_seconds:
            report['stopped_before'] = phase
            raise TimeoutError('internal budget exhausted; no automatic continuation')
    phase = 'reference'
    try:
        deadline(phase)
        groups = diagnostic_groups(view, fanout); panels = make_panels(view, panel_target)
        report.update(view=view.metadata, panels=panels, support_groups=groups)
        forward = FrozenForward(model, features, view.neighbors, budget)
        synchronize(device)
        report['native_full_forward_seconds'] = time.perf_counter() - started
        phase = 'numerical_entry'; deadline(phase)
        numerical, quality = numerical_entry(forward)
        synchronize(device); report['numerical_entry'] = quality
        if quality['status'] != 'passed': raise ValueError('predeclared numerical entry checks failed')
        phase = 'reference_structure'; deadline(phase)
        report['reference_layers'] = []
        for i, row in enumerate(numerical['layers']):
            trace = forward.reference['layers'][i]; root = view.nodes.index(view.root) if view.root else None
            radii = torch.acosh((trace['messages'][:, 0].double() * math.sqrt(model.c)).clamp_min(1.))
            output_radii = torch.acosh((row['base'][:, 0] * math.sqrt(model.c)).clamp_min(1.))
            threshold = .95 * model.layers[i].scaled_radius
            report['reference_layers'].append({
                'layer': i + 1, 'message_scaled_radius': summarize(radii),
                'output_scaled_radius': summarize(output_radii), 'near_bound_threshold': threshold,
                'near_bound_message_fraction': float((radii >= threshold).double().mean()),
                'near_bound_output_fraction': float((output_radii >= threshold).double().mean()),
                'numerical_floor_per_node': row['empirical_numerical_floor'],
                'structure': radial_relations(view, panels['panels']['development'], row['base'], row['base'],
                                             groups, model.c, row['empirical_numerical_floor']),
                'structure_scope': 'development panel only; diagnostic confirmation pool remains unopened',
                'chance_reference': None, 'full_reference_ability_status': 'descriptive_pending_supervisor_review'})
        phase = 'complete_valid'; deadline(phase)
        index = {node: i for i, node in enumerate(view.nodes)}
        valid = [(index[a], index[b]) for a, b in sorted(view.valid_edges)]; parents = defaultdict(set)
        for a, b in valid: parents[b].add(a)
        ranking = filtered_parent_ranks(model, forward.reference['output'], valid, parents,
                                       candidate_chunk, min(ranking_seconds, max_seconds - (time.perf_counter() - started)))
        report['full_valid_reference'] = ranking
        if ranking['status'] != 'complete': raise TimeoutError('entry full-valid ranking incomplete')
        if expected_ranking is not None:
            reproduction = validation_reproduction(ranking, expected_ranking)
            report['entry_full_valid_reproduction'] = reproduction
            if reproduction['status'] != 'passed': raise ValueError('entry full-valid reproduction failed')
        report['entry_status'] = 'passed'; report['pilot_status'] = 'running'; publish()
        streams = PlanStreams(seed, namespace); report['seed_namespace'] = streams.metadata
        report['geometry'] = {}; report['sampling_audits'] = {}; report['plan_hashes'] = {}
        # Finish L1, then F/S. A failed/partial L1 prevents all dependent L2 work.
        for layer, name in enumerate(('local_L1', 'F/S')):
            phase = name; deadline(phase)
            base = numerical['layers'][layer]['base']; floor = numerical['layers'][layer]['empirical_numerical_floor']
            if view.root is None:
                directions = torch.zeros_like(base); defined = torch.zeros(len(base), dtype=torch.bool, device=device)
            else:
                root = view.nodes.index(view.root)
                directions, defined = outward_directions(base, base[root], model.c, floor + floor[root])
            stream = TangentStream(base, model.c, groups, directions, defined)
            report['completed_repetitions'][name] = 0; report['plan_hashes'][name] = []
            report['sampling_audits'][name] = []
            inactive = sorted(set(groups['V']) - set(groups['A']))
            active_mask = torch.zeros(len(base), dtype=torch.bool, device=device)
            active_mask[groups['A']] = True
            for repetition in range(repetitions):
                deadline(f'{name}/{repetition}')
                # Consume only this layer's independent registered stream.
                plan = make_plan(view.neighbors, fanout, streams.generators[layer])
                report['plan_hashes'][name].append(digest(plan))
                points, aggregation = model.frozen_layer(forward.reference, layer, view.neighbors, plan,
                                                        max_padded_messages=budget)
                promoted, promotion = promote_points(points, model.c); checks = []
                promotion_checks(checks, name, promotion)
                offsets = torch.zeros_like(base)
                if groups['A']:
                    offsets[groups['A']] = log(base[groups['A']], promoted[groups['A']], model.c)
                # Inactive local rows use complete inputs exactly. Any batch-rounding
                # difference is recorded and bounded, not interpreted as sampling bias.
                if inactive:
                    check_max(checks, 'inactive_native_output_difference',
                              points[inactive] - forward.reference['layers'][layer]['output'][inactive],
                              LIMITS['unchanged_output_difference'])
                check_max(checks, 'offset_tangent_constraint', dot(base, offsets), LIMITS['tangent_constraint'])
                check_max(checks, 'offset_log_exp_roundtrip',
                          exp(base, offsets, model.c) - torch.where(
                              active_mask[:, None], promoted, base),
                          LIMITS['roundtrip'])
                sample_audit = {'repetition': repetition, 'checks': checks,
                                'selected_messages': int(aggregation['k'].sum()),
                                'candidate_messages': int(aggregation['N'].sum()),
                                'empty_nodes': int(aggregation['empty'].sum())}
                report['sampling_audits'][name].append(sample_audit)
                if not all(row['status'] == 'passed' for row in checks):
                    raise ValueError(f'{name} repetition {repetition} numerical checks failed')
                stream.add_offsets(offsets)
                report['completed_repetitions'][name] += 1
            stats = stream.finish(floor)
            checks = []
            check_max(checks, 'mse_decomposition', stats['mse_decomposition_max_residual'], LIMITS['mse_decomposition'])
            for key, tensor in [('variance', stats['variance']), ('mse', stats['mse']['mean'])]:
                observed = summarize(tensor)
                checks.append({'name': key + '_minimum', 'status': 'passed' if observed['minimum'] >= LIMITS['variance_mse_minimum'] else 'failed',
                               'minimum': LIMITS['variance_mse_minimum'], 'observed': observed})
            report['geometry'][name] = compact_statistics(stats, groups['A'])
            report['geometry'][name]['quality_checks'] = checks
            report['geometry'][name]['direction_definition'] = 'outward from fixed H_dev full-reference root; no direction fitted to repeats'
            report['geometry'][name]['projection_unresolved_at_numerical_floor'] = (
                stats['projection']['mean'].abs() <= floor) | ~defined
            if not all(row['status'] == 'passed' for row in checks): raise ValueError('pilot moment checks failed')
            publish()
        report.update(status='complete', pilot_status='complete', purpose_status='pending_supervisor_review',
                      conclusion='Entry and fixed local pilot complete; research interpretation requires review')
    except (ValueError, TimeoutError, RuntimeError) as error:
        report['failures'].append({'phase': phase, 'type': type(error).__name__, 'error': str(error),
                                   'policy': 'stop dependent steps; no redraw, threshold change or automatic retry'})
        report['status'] = 'incomplete_time_limit' if isinstance(error, TimeoutError) else 'failed'
        if report['entry_status'] != 'passed': report['entry_status'] = report['status']
        else: report['pilot_status'] = report['status']
        report['conclusion'] = 'Insufficient completed evidence; inspect recorded stopping reason'
    publish()
    return report


def run(config, seed, prepared, checkpoint_path, training_release, baseline_path, approval, source_commit,
        *, device='cpu', progress=None, diagnostic_runner=None, configuration_check=None):
    # Development entry reuses the same provenance loader with its own fixed
    # validator and bounded runner. The public local-pilot CLI retains defaults.
    (configuration_check or validate_config)(config); identity = source_identity(source_commit)
    if identity['source_commit'] is None: raise ValueError('archive execution requires explicit source commit')
    verify_approval(approval, config, identity)
    spec = config['checkpoints'][str(seed)]; pilot = config['pilot']; started = time.perf_counter()
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda') or device.index not in (None, 0): raise ValueError('single allocated device only')
    if device.type == 'cuda':
        if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count() != 1:
            raise ValueError('single visible Slurm GPU and preserved binding required')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest'); torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(pilot['threads'])
    if file_sha256(baseline_path) != spec['baseline_report_sha256']: raise ValueError('baseline report hash mismatch')
    baseline = json.loads(Path(baseline_path).read_text(encoding='utf-8'))
    checkpoint, train_config = load_verified_checkpoint(checkpoint_path, spec['sha256'], spec['training_commit'], training_release)
    data = load_prepared(prepared); view = load_development_view(prepared, pilot['max_reach_visits'])
    selection = verify_selection(checkpoint, baseline, spec, data)
    if (view.metadata['prepared_manifest_hash'] != data['manifest_hash']
            or view.metadata['valid_queries_hash'] != data['valid_hash']): raise ValueError('development view changed during loading')
    model = LorentzMeanNetwork(data['features'].shape[1], train_config.hidden, train_config.c,
                               train_config.scaled_radius, train_config.head_hidden)
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in checkpoint['model'].values()):
        raise ValueError('finite original FP32 checkpoint weights required')
    model.load_state_dict(checkpoint['model'], strict=True); model.to(device).eval().requires_grad_(False)
    weights_hash = {name: hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() for name, value in model.state_dict().items()}
    provenance = {'config': config, 'config_sha256': digest(config), 'source': identity,
                  'checkpoint_seed': seed, 'checkpoint_sha256': spec['sha256'], 'selection': selection,
                  'approval_record_sha256_canonical': digest(approval),
                  'training_source': checkpoint['source'], 'training_source_hashes': checkpoint['source_sha256_normalized_lf'],
                  'prepared_manifest_hash': data['manifest_hash'], 'valid_queries_hash': data['valid_hash'],
                  'label_files_read': sorted(set(view.metadata['loaded_files']) | {'features.npz'}),
                  'label_scope': 'train+valid only; test entity IDs are public split membership, not test relation labels',
                  'precision': 'FP32 native, same weights/features rerun FP64; no AMP or TF32',
                  'torch': str(torch.__version__), 'device_type': device.type,
                  'device_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}
    def save_progress(report):
        report.update(provenance)
        if progress: progress(report)
    result = (diagnostic_runner or diagnose)(model, data['features'].to(device), view,
                      max_seconds=pilot['internal_seconds'] - (time.perf_counter() - started),
                      budget=train_config.max_padded_messages, namespace=pilot['namespace'] + f'/seed{seed}',
                      panel_target=pilot['panel_target'], candidate_chunk=pilot['candidate_chunk'],
                      ranking_seconds=pilot['ranking_seconds'], progress=save_progress,
                      expected_ranking=next(row for row in baseline['evaluations'] if row['step'] == spec['step']))
    if file_sha256(checkpoint_path) != spec['sha256'] or any(
            hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() != weights_hash[name]
            for name, value in model.state_dict().items()): raise ValueError('frozen checkpoint or model changed')
    result.update(provenance); result['checkpoint_and_weights_unchanged'] = True
    result['source_sha256_normalized_lf'] = dependency_hashes()
    result['source_sha256_normalized_lf']['acl_hct/e2_pilot.py'] = hashlib.sha256(Path(__file__).read_text(encoding='utf-8').encode()).hexdigest()
    result['total_seconds'] = time.perf_counter() - started
    if device.type == 'cuda':
        result['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        result['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--seed', type=int, choices=(11, 23))
    for name in ('prepared', 'checkpoint', 'training-release', 'baseline-report', 'approval-record', 'output'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--source-commit'); parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args(); config = json.loads(args.config.read_text(encoding='utf-8'))
    config_hash = validate_config(config)
    if not args.execute:
        print(json.dumps({'status': 'static_only', 'config_sha256': config_hash, 'pilot': config['pilot']})); return
    if any(getattr(args, name) is None for name in ('seed', 'prepared', 'checkpoint', 'training_release',
                                                  'baseline_report', 'approval_record', 'output', 'source_commit')):
        raise ValueError('execution requires approved config, checkpoint, provenance, explicit source and new output')
    if args.output.exists() or args.output.suffix != '.json': raise ValueError('new .json output required')
    approval = json.loads(args.approval_record.read_text(encoding='utf-8'))
    # Reserve only after the approval gate. Incremental atomic saves retain completed
    # entry/L1 evidence on an external timeout; no automatic resume path exists.
    verify_approval(approval, config, source_identity(args.source_commit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump({'status': 'verification_started', 'config_sha256': config_hash}, stream)
    def save(result):
        temporary = args.output.with_suffix('.json.tmp')
        with temporary.open('w', encoding='utf-8') as stream: json.dump(jsonable(result), stream, allow_nan=False)
        temporary.replace(args.output)
    try:
        result = run(config, args.seed, args.prepared, args.checkpoint, args.training_release, args.baseline_report,
                     approval, args.source_commit, device=args.device, progress=save)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        save({'status': 'verification_or_execution_failed', 'error': str(error),
              'config_sha256': config_hash, 'next_experiment_authorized': False})
        raise
    save(result)
    print(json.dumps({'status': result['status'], 'entry_status': result['entry_status'], 'pilot_status': result['pilot_status']}))


if __name__ == '__main__': main()

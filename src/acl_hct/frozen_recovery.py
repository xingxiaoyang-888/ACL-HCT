"""Thin frozen-pilot orchestration over existing input/metric/archive primitives."""
import argparse
from collections import defaultdict
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from .backbone import LorentzMeanNetwork
from .development_view import build_view
from .diagnostic_archive import write_archive
from .encoder_matched_control import (atomic_json, load_prepared, require_runtime,
                                     synchronize, validate_ranking, weights_hashes)
from .evaluate_checkpoint import load_verified_checkpoint
from .e2_pilot import LIMITS, promotion_checks, verify_selection
from .frozen_bias_intervention import FrozenBiasIntervention
from .frozen_forward import FrozenForward, PlanStreams
from .frozen_stats import promote_points
from .geometry import check_point, distance, dot, exp, log, norm2
from .protocols import grouped_split
from .ranking import filtered_parent_ranks, prepare_scores, score_cached
from .recovery_registration import (CONFIG_SHA256, PROTOCOL, canonical, file_sha256,
                                    source_hashes, source_identity, validate_config, verify_gate, verify_release)
from .recovery_inputs import VerifiedInputs, prepared_from_cache, checkpoint_from_cache
from .vector_structure import RadialPanel


def deadline_check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('whole-worker deadline reached')


def peak_memory(device):
    return {'allocated_peak_bytes': int(torch.cuda.max_memory_allocated(device)),
            'reserved_peak_bytes': int(torch.cuda.max_memory_reserved(device))} if device.type == 'cuda' else None


class ResourceTimeBudgetInadequate(RuntimeError):
    """Operational stop before evaluation repeats; no effect-based selection."""


def tensor_identity(value):
    raw = value.detach().cpu().contiguous().numpy()
    return {'shape': list(raw.shape), 'dtype': raw.dtype.str,
            'data_sha256': hashlib.sha256(raw.tobytes()).hexdigest()}


def repeat_plans(config, seed, repeat, neighbors):
    if type(seed) is not int or seed not in (11, 23) or type(repeat) is not int or not 0 <= repeat < 16:
        raise ValueError('registered model/global repeat required')
    pilot = config['pilot']
    streams = PlanStreams(pilot['sampling_seed'], pilot['sampling_namespace'].format(seed=seed, repeat=repeat))
    return streams.draw(neighbors, 4), streams.metadata


def _decode_selected(record, filename, trees, deadline):
    """Reuse the archive contract while reading only requested calibrated fields."""
    keys = set()
    def collect(value):
        if isinstance(value, dict):
            if set(value) == {'npz_array'}:
                keys.add(value['npz_array'])
            else:
                for item in value.values():
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
    for tree in trees.values():
        collect(tree)
    arrays = {}
    with np.load(filename, allow_pickle=False) as archive:
        if set(archive.files) != set(record['arrays']):
            raise ValueError('calibration archive array inventory mismatch')
        for key in sorted(keys):
            deadline_check(deadline)
            value = archive[key]
            meta = record['arrays'][key]
            if (value.dtype.str != meta['dtype'] or list(value.shape) != meta['shape']
                    or hashlib.sha256(value.tobytes()).hexdigest() != meta['data_sha256']):
                raise ValueError('calibration selected array identity mismatch')
            arrays[key] = value
    def decode(value):
        if isinstance(value, dict):
            if set(value) == {'npz_array'}:
                return arrays[value['npz_array']]
            return {key: decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [decode(item) for item in value]
        return value
    return {name: decode(tree) for name, tree in trees.items()}


def load_calibration(folder, spec, nodes, deadline, cache=None, *, alias='calibration'):
    folder = Path(folder)
    records = {}
    raw_hashes = {}
    if cache is None:
        cache = VerifiedInputs()
    for stem, prefix in (('entry', 'entry'), ('fanout_4', 'fanout')):
        deadline_check(deadline)
        for suffix, key in (('.json', prefix + '_manifest_raw_sha256'), ('.npz', prefix + '_npz_raw_sha256')):
            path = folder / (stem + suffix)
            cache.allow(path, spec[key], alias + '/' + stem + suffix)
            observed = cache.sha(path)
            if observed != spec[key]:
                raise ValueError('calibration original artifact raw SHA mismatch: ' + stem + suffix)
            raw_hashes[stem + suffix] = observed
        record = json.loads(cache.bytes(folder / (stem + '.json')))
        if record['array_file'] != stem + '.npz' or record['array_file_sha256'] != spec[prefix + '_npz_raw_sha256']:
            raise ValueError('calibration manifest/NPZ binding mismatch')
        records[stem] = record
    fields = spec['fields']
    for name, meta in fields.items():
        record = records['entry' if name in ('nodes', 'p', 'floor') else 'fanout_4']
        if record['arrays'][meta['key']] != {k: meta[k] for k in ('shape', 'dtype', 'data_sha256')}:
            raise ValueError('registered calibration field descriptor mismatch')
    entry = records['entry']
    entry_trees = {name: {'npz_array': fields[name]['key']} for name in ('nodes', 'p', 'floor')}
    entry_trees['panels'] = entry['payload']['panels']
    first = _decode_selected(entry, io.BytesIO(cache.bytes(folder / 'entry.npz')), entry_trees, deadline)
    second = _decode_selected(records['fanout_4'], io.BytesIO(cache.bytes(folder / 'fanout_4.npz')),
                              {name: {'npz_array': fields[name]['key']} for name in fields if name not in ('nodes', 'p', 'floor')}, deadline)
    if first['nodes'].tolist() != nodes or first['panels']['hash'] != spec['panels_hash']:
        raise ValueError('calibration node order or panel identity mismatch')
    output = {name: torch.from_numpy(value.copy()) for name, value in {**first, **second}.items()
              if name not in ('nodes', 'panels')}
    if any(value.dtype != torch.float64 or not torch.isfinite(value).all() for value in output.values()):
        raise ValueError('complete finite native FP64 calibration fields required')
    if (output['p'].shape != output['b'].shape or output['p'].shape[0] != len(nodes)
            or any(output[name].shape != (len(nodes),) for name in ('floor', 'variance', 'half_cross', 'noise_corrected_bias_squared'))
            or (output['floor'] < 0).any() or (output['variance'] < -1e-12).any()):
        raise ValueError('calibration shape or variance/floor contract mismatch')
    check_point(output['p'])
    if (dot(output['p'], output['b']).abs() > LIMITS['tangent_constraint']).any():
        raise ValueError('calibration mean is not tangent at its registered full-L2 base')
    output.update(panel=first['panels']['panels']['diagnostic_confirmation'],
                  panels_hash=spec['panels_hash'], raw_hashes=raw_hashes,
                  uncertainty={'R': spec['R'], 'half_counts': spec['half_counts'],
                               'mean_estimator_se_norm': (output['variance'].clamp_min(0) / (spec['R'] - 1)).sqrt(),
                               'half_mean_vectors_available': False,
                               'scope': 'variance/(R-1) traces mean-estimator sampling covariance; signed half-cross retained; not vector confidence intervals'})
    return output


def load_inputs(config, seed, prepared, checkpoint_root, training_release, reference_root,
                calibration_root, deadline, *, cache=None):
    """All bulk hashing/loading is inside the already running supervised worker."""
    deadline_check(deadline)
    loading_started = time.monotonic()
    stage_started = loading_started
    loading_seconds = {}
    def stage(name):
        nonlocal stage_started
        now = time.monotonic()
        loading_seconds[name] = now - stage_started
        stage_started = now
        deadline_check(deadline)
    if cache is None:
        cache = VerifiedInputs()
    for name, expected in {**config['prepared']['input_raw_sha256'],
                           'features.npz': config['prepared']['features_npz_raw_sha256'],
                           'evaluator_valid.json': config['prepared']['evaluator_valid_raw_sha256']}.items():
        cache.allow(Path(prepared) / name, expected, 'prepared/' + name)
    data = prepared_from_cache(cache, prepared, config)
    stage('prepared_verification_and_decode')
    spec = config['checkpoints'][str(seed)]
    baseline_path = Path(reference_root) / f'baseline-seed{seed}-run.json'
    cache.allow(baseline_path, spec['baseline_report_sha256'], f'seed{seed}/baseline')
    baseline = json.loads(cache.bytes(baseline_path))
    checkpoint_path = Path(checkpoint_root) / f'seed{seed}-best.pt'
    cache.allow(checkpoint_path, spec['sha256'], f'seed{seed}/checkpoint')
    checkpoint, train_config = checkpoint_from_cache(cache, checkpoint_path, spec['sha256'], spec['training_commit'], training_release)
    selection = verify_selection(checkpoint, baseline, spec, data)
    for row in baseline['evaluations']:
        validate_ranking(row, data)
    stage('checkpoint_and_baseline_verification')
    calibration = load_calibration(Path(calibration_root) / f'seed{seed}', config['calibration'][str(seed)],
                                   data['nodes'], deadline, cache, alias=f'seed{seed}/calibration')
    stage('calibration_verification_and_decode')
    entities = grouped_split(data['nodes'], [], 20260914)['entities']['valid']
    train = [(data['nodes'][a], data['nodes'][b]) for a, b in data['query_groups'][:, 0].tolist()]
    valid = [(data['nodes'][a], data['nodes'][b]) for a, b in data['valid']]
    view = build_view(data['nodes'], data['neighbors'], train, valid, entities, config['pilot']['max_reach_visits'])
    anchor = config['calibration'][str(seed)]
    if (view.metadata['h_dev_hash'] != anchor['h_dev_hash'] or view.root != anchor['reference_root']
            or view.metadata['graph_hash'] != anchor['identity']['graph_hash']
            or view.metadata['node_order_hash'] != anchor['identity']['node_order_hash']):
        raise ValueError('original structure view/calibration identity mismatch')
    if train_config.__dict__ != spec['training_config']:
        raise ValueError('original training config drift')
    stage('original_structure_view')
    with torch.random.fork_rng(devices=[]):
        model = LorentzMeanNetwork(data['features'].shape[1], train_config.hidden, train_config.c,
                                   train_config.scaled_radius, train_config.head_hidden)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval().requires_grad_(False)
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('finite original FP32 weights required')
    stage('frozen_model_loading')
    proof = {'checkpoint_raw_sha256': spec['sha256'], 'baseline_raw_sha256': spec['baseline_report_sha256'],
             'calibration_raw_hashes': calibration['raw_hashes'], 'weights': weights_hashes(model),
             'selection': selection, 'prepared_manifest_hash': data['manifest_hash'],
             'valid_hash': data['valid_hash'], 'input_files_opened': data['input_files_opened'],
             'prepared_raw_hashes': {**config['prepared']['input_raw_sha256'],
                                     'features.npz': config['prepared']['features_npz_raw_sha256'],
                                     'evaluator_valid.json': config['prepared']['evaluator_valid_raw_sha256']},
             'calibration_fields': {key: tensor_identity(calibration[key]) for key in ('p', 'b', 'floor', 'variance', 'half_cross', 'noise_corrected_bias_squared')},
             'nodes_count': len(data['nodes']), 'valid_count': len(data['valid']), 'calibration_R': anchor['R'],
             'parameter_count': sum(p.numel() for p in model.parameters()), 'native_dtype': 'float32',
             'gradients_present': any(p.grad is not None for p in model.parameters()),
             'worker_input_receipts': {name: copy.deepcopy(value) for name, value in cache.receipts.items()
                                       if name.startswith(('prepared/', f'seed{seed}/'))},
             'all_bulk_inputs_verified_inside_worker': True}
    stage('input_proof_creation')
    proof['loading_phase_seconds'] = loading_seconds
    proof['loading_total_seconds'] = time.monotonic() - loading_started
    deadline_check(deadline)
    return model, data, view, calibration, baseline, proof


def promoted(points, c):
    result, audit = promote_points(points, c)
    checks = []
    promotion_checks(checks, 'condition', audit)
    if any(row['status'] != 'passed' for row in checks):
        raise ValueError('original condition promotion gate failed')
    return result


def cast_for_head(analytic, sample_native, sample64, bias, floor, c, limits):
    native = analytic.float()
    zero = norm2(bias) == 0
    native[zero] = sample_native[zero]
    check_point(native, c)
    returned = promoted(native, c)
    error = distance(analytic, returned, c)
    step = distance(sample64, analytic, c)
    scale = 1 + analytic.norm(dim=-1)
    resolution = torch.maximum(floor, limits['cast_proxy_epsilon_multiple'] * torch.finfo(torch.float32).eps * scale)
    resolved = step >= limits['step_resolution_factor'] * resolution
    relative = torch.where(step > 0, error / step.clamp_min(1e-300), torch.zeros_like(step))
    if (not torch.isfinite(relative).all() or (error > limits['cast_absolute_geodesic']).any()
            or (relative[resolved] > limits['cast_relative_resolved']).any()
            or not torch.equal(native[zero], sample_native[zero])):
        raise ValueError('registered cast absolute/relative/zero identity gate failed')
    return native, {'error': error, 'actual_step': step, 'resolution_proxy': resolution,
                    'relative_error': relative, 'resolved': resolved,
                    'resolved_fraction': float(resolved.double().mean()),
                    'max_error': float(error.max()), 'max_relative_resolved': float(relative[resolved].max()) if resolved.any() else None,
                    'analytic_identity': tensor_identity(analytic), 'zero_native_identity': True}


@torch.no_grad()
def evaluate_repeat(model, features, view, forward, intervention, config, seed, repeat):
    plans, rng = repeat_plans(config, seed, repeat, view.neighbors)
    sampled, sample_diagnostics, _ = model.encode(features, view.neighbors, plans,
                                                 max_padded_messages=config['pilot']['max_padded_messages'])
    corrected, correct_diagnostics, _ = model.encode(features, view.neighbors, plans, method='third',
                                                     max_padded_messages=config['pilot']['max_padded_messages'])
    sample64 = promoted(sampled, model.c)
    moved = intervention.apply(sample64, evaluation_namespace=intervention.metadata['evaluation_namespace'])
    if (moved['reconstructed_S'] - sample64).abs().max() > config['numeric_proposal']['roundtrip_max_abs']:
        raise ValueError('S log-exp reconstruction failed')
    return {'plans': plans, 'rng': rng, 'native': {'S': sampled, 'C': corrected},
            'analytic': moved['points'], 'offsets': moved['offsets'],
            'diagnostics': {'S': sample_diagnostics, 'C': correct_diagnostics}, 'sample64': sample64}


def geometry_summary(points, base, floor, c):
    error = log(base, points, c)
    length2 = norm2(error)
    return {'all_nodes': len(base), 'mean_squared_error': float(length2.mean()),
            'mean_error_norm': float(length2.sqrt().mean()), 'maximum_error_norm': float(length2.sqrt().max()),
            'below_empirical_floor_fraction': float((length2.sqrt() <= floor).double().mean()),
            'tangent_max_absolute': float(dot(base, error).abs().max())}


def _metrics(structure, ranking):
    result = {kind: structure[kind]['groups']['V']['weighted_covered_child_metrics']['score'] for kind in ('direct', 'distant')}
    result.update(micro_mrr=ranking['query_micro_mrr'], macro_mrr=ranking['child_macro_mrr'], hits10=ranking['hits']['10'])
    return result


@torch.no_grad()
def run_slice(model, data, view, calibration, config, seed, repeat_range, directory, deadline,
              *, provenance=None, expected_ranking=None, engineering=False):
    if not engineering and (next(model.parameters()).device.type != 'cuda' or not provenance or not provenance.get('release_verified')):
        raise ValueError('science requires reviewed native CUDA release')
    if repeat_range not in ([0, 8], [8, 16]):
        raise ValueError('registered eight-repeat shard required')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if any((root / name).exists() for name in ('run.json', 'entry.json')):
        raise ValueError('new output; no resume or overwrite')
    device = next(model.parameters()).device
    model.eval().requires_grad_(False)
    features = data['features'].to(device)
    initial_weights = weights_hashes(model)
    input_identity = tensor_identity(features)
    row = {'status': 'verification_started', 'protocol': PROTOCOL, 'config_sha256': CONFIG_SHA256,
           'seed': seed, 'repeat_range': repeat_range, 'observations': [], 'repeat_archives': [],
           'engineering_fixture_only': engineering, 'provenance': provenance, 'active_phase': 'reference',
           'phase_costs': []}
    phase_started = time.monotonic()
    phase_repeat = None
    def progress(phase):
        nonlocal phase_started, phase_repeat
        now = time.monotonic()
        row['phase_costs'].append({'phase': row['active_phase'], 'repeat': phase_repeat,
                                  'elapsed_seconds': now - phase_started})
        row['active_phase'] = phase
        phase_started = now
        phase_repeat = row.get('active_repeat')
        row['peak_vram'] = peak_memory(device)
        row['completed_repetitions'] = len(row['observations'])
        atomic_json(root / 'run.json', row)
        deadline_check(deadline)
    try:
        progress('full_reference')
        forward = FrozenForward(model, features, view.neighbors, config['pilot']['max_padded_messages'])
        base = calibration['p'].to(device)
        bias = calibration['b'].to(device)
        floor = calibration['floor'].to(device)
        if (promoted(forward.reference['output'], model.c) - base).abs().max() > config['numeric_proposal']['calibration_base_max_abs']:
            raise ValueError('native full-L2 no longer matches the original calibration base')
        anchor = config['calibration'][str(seed)]
        intervention = FrozenBiasIntervention(base, bias, c=model.c,
            calibration_namespace=anchor['sampling_stream']['namespace'],
            evaluation_namespace=f'{PROTOCOL}/seed{seed}/evaluation',
            direction_namespace=config['pilot']['q_namespace'].format(seed=seed), direction_seed=config['pilot']['q_seed'])
        panel = RadialPanel(view, calibration['panel'], base, {'V': list(range(len(base)))}, model.c, floor)
        truth = defaultdict(set)
        for a, b in data['valid']:
            truth[b].add(a)
        def rank(points):
            deadline_check(deadline)
            result = filtered_parent_ranks(model, points, data['valid'], truth,
                config['pilot']['candidate_chunk'], min(config['pilot']['ranking_seconds'], deadline - time.monotonic()))
            if result['status'] != 'complete':
                row['partial_ranking_archive'] = write_archive(root, 'partial-ranking',
                    {'active_repeat': row.get('active_repeat'), 'active_phase': row['active_phase'], 'ranking': result})
            validate_ranking(result, data)
            return result
        progress('F_ranking')
        full = rank(forward.reference['output'])
        if expected_ranking is not None:
            expected = {(r['parent'], r['child']): r['rank'] for r in expected_ranking['rows']}
            if any(expected.get((r['parent'], r['child'])) != r['rank'] for r in full['rows']):
                raise ValueError('full valid ranks do not reproduce original F')
        progress('F_structure_and_archive')
        full_structure = panel.evaluate(promoted(forward.reference['output'], model.c))
        full_metrics = _metrics(full_structure, full)
        if any(full_metrics[k] is None for k in ('direct', 'distant')):
            raise ValueError('primary structure metric has no covered relation; unknown is not zero')
        row['identity'] = {'weights': initial_weights, 'input': input_identity,
                           'base': tensor_identity(base), 'bias': tensor_identity(bias), 'q': tensor_identity(intervention.control),
                           'panels_hash': calibration['panels_hash'], 'F_rank_hash': canonical(full['rows']),
                           'F_metrics': full_metrics,
                           'source_commit': provenance.get('source_commit') if provenance else None,
                           'valid_hash': data['valid_hash'], 'h_dev_hash': view.metadata['h_dev_hash']}
        row['entry_archive'] = write_archive(root, 'entry', {'base': base, 'bias': bias, 'q': intervention.control,
            'floor': floor, 'nodes': data['nodes'], 'weights': model.state_dict(),
            'panel': calibration['panel'], 'structure_view': {'root': view.root, 'reachable': sorted(view.reachable)},
            'valid': np.asarray(data['valid'], dtype=np.int64),
            'native_F': forward.reference['output'], 'F_ranking': full, 'F_structure': full_structure,
            'uncertainty': calibration['uncertainty'], 'half_cross': calibration['half_cross'],
            'noise_corrected_bias_squared': calibration['noise_corrected_bias_squared'], 'adapter': intervention.metadata})
        progress('resource_time_feasibility')
        rule = config['resources']['run_feasibility']
        remaining_rankings = 4 * (repeat_range[1] - repeat_range[0])
        measured = full['elapsed_seconds']
        if not np.isfinite(measured) or measured < 0 or remaining_rankings != rule['remaining_complete_rankings_per_shard']:
            raise ValueError('finite F timing and fixed remaining ranking count required')
        estimate = measured * remaining_rankings + rule['nonranking_reserve_seconds']
        remaining = deadline - time.monotonic()
        row['resource_feasibility'] = {'measured_F_ranking_seconds': measured,
            'remaining_complete_rankings': remaining_rankings, 'nonranking_reserve_seconds': rule['nonranking_reserve_seconds'],
            'estimated_remaining_seconds': estimate, 'remaining_worker_seconds': remaining,
            'passed': estimate <= remaining, 'scope': rule['scope']}
        if estimate > remaining:
            raise ResourceTimeBudgetInadequate('F timing predicts insufficient remaining worker budget; keep F; change resources only')
        for repeat in range(*repeat_range):
            row['active_repeat'] = repeat
            progress('paired_forward')
            repeat_started = time.monotonic()
            result = evaluate_repeat(model, features, view, forward, intervention, config, seed, repeat)
            synchronize(device)
            forward_seconds = time.monotonic() - repeat_started
            casting = {}
            for name in ('O', 'Q'):
                result['native'][name], casting[name] = cast_for_head(result['analytic'][name], result['native']['S'],
                    result['sample64'], bias, floor, model.c, config['numeric_proposal'])
            metrics, structures, rankings, geometry = {}, {}, {}, {}
            for name in ('S', 'O', 'C', 'Q'):
                progress(name + '_ranking')
                rankings[name] = rank(result['native'][name])
                points64 = promoted(result['native'][name], model.c)
                structures[name] = panel.evaluate(points64)
                geometry[name] = geometry_summary(points64, base, floor, model.c)
                metrics[name] = _metrics(structures[name], rankings[name])
                if any(not isinstance(metrics[name][k], (int, float)) or not np.isfinite(metrics[name][k])
                       for k in ('direct', 'distant', 'micro_mrr')):
                    raise ValueError('primary metric missing/nonfinite; unknown is not zero')
            observation = {'repeat': repeat, 'metrics': metrics, 'plan_hash': canonical(result['plans']),
                           'cost': {'paired_S_C_forward_seconds': forward_seconds,
                                    'ranking_seconds': {name: rank_row['elapsed_seconds'] for name, rank_row in rankings.items()}},
                           'geometry': geometry,
                           'rng': result['rng'], 'casting': {k: {key: value for key, value in v.items()
                               if key in ('resolved_fraction', 'max_error', 'max_relative_resolved', 'zero_native_identity')} for k, v in casting.items()},
                           'paired_differences': {comparison: {metric: metrics[a][metric] - (full_metrics[metric] if b == 'F' else metrics[b][metric])
                               for metric in ('direct', 'distant', 'micro_mrr')} for comparison, a, b in
                               [('S-F', 'S', 'F'), ('O-S', 'O', 'S'), ('C-S', 'C', 'S'), ('O-Q', 'O', 'Q')]}}
            progress('repeat_archive')
            descriptor = write_archive(root, f'repeat-{repeat:02}', {'repeat': repeat, 'native_points': result['native'],
                'rankings': rankings, 'structure': structures, 'casting': casting, 'diagnostics': result['diagnostics'],
                'geometry': geometry,
                'plan_hash': observation['plan_hash'], 'rng': result['rng']})
            row['repeat_archives'].append(descriptor)
            row['observations'].append(observation)
            progress('repeat_complete')
        if weights_hashes(model) != initial_weights or tensor_identity(features) != input_identity or any(p.grad is not None for p in model.parameters()):
            raise ValueError('frozen weights/input/gradient contract failed')
        row['status'] = 'complete'
        progress('final_archive')
    except Exception as error:
        row['status'] = ('resource_time_budget_inadequate' if isinstance(error, ResourceTimeBudgetInadequate)
                         else 'incomplete_time_limit' if isinstance(error, TimeoutError) else 'failed')
        row['failure'] = {'type': type(error).__name__, 'error': str(error), 'phase': row['active_phase'],
                          'partial_results_are_complete': False, 'policy': 'stop; no redraw, clipping change, resume or automatic retry'}
    finally:
        row['phase_costs'].append({'phase': row['active_phase'], 'repeat': phase_repeat,
                                  'elapsed_seconds': time.monotonic() - phase_started})
        row['peak_vram'] = peak_memory(device)
        row['completed_repetitions'] = len(row['observations'])
        row['weights_unchanged'] = weights_hashes(model) == initial_weights
        row['input_unchanged'] = tensor_identity(features) == input_identity
        row['model_updates'] = 0
        atomic_json(root / 'run.json', row)
    if time.monotonic() >= deadline and row['status'] == 'complete':
        row['status'] = 'incomplete_time_limit'
        atomic_json(root / 'run.json', row)
    return row


def fixture_data(config, seed):
    """Forty invented nodes, production dimensions; never opens original inputs."""
    from .frozen_fixture import synthetic_fixture
    from .development_view import make_panels
    _, _, view = synthetic_fixture()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        spec = config['model']
        model = LorentzMeanNetwork(spec['input_dim'], spec['hidden'], spec['c'],
                                   spec['scaled_radius'], spec['head_hidden']).eval().requires_grad_(False)
    features = torch.randn(40, spec['input_dim'], generator=torch.Generator().manual_seed(2026091710)) * .1
    index = {n: i for i, n in enumerate(view.nodes)}
    valid = [(index[a], index[b]) for a, b in sorted(view.valid_edges)]
    data = {'features': features, 'nodes': view.nodes, 'neighbors': view.neighbors, 'valid': valid,
            'valid_hash': canonical(valid), 'manifest_hash': canonical(view.metadata)}
    return model, data, view, make_panels(view)['panels']['diagnostic_confirmation']


def fixture_calibration(model, features, view, panel):
    from .frozen_bias_intervention import transport_from_origin
    forward = FrozenForward(model, features, view.neighbors)
    base = promoted(forward.reference['output'], model.c)
    spatial = torch.randn(base[:, 1:].shape, dtype=torch.float64, device='cpu',
                          generator=torch.Generator().manual_seed(2026091711)).to(base.device)
    spatial /= spatial.norm(dim=-1, keepdim=True)
    bias = transport_from_origin(base, spatial, model.c) * .02
    bias[0] = 0
    floor = torch.zeros(len(base), dtype=torch.float64, device=base.device)
    cal = {'p': base, 'b': bias, 'floor': floor, 'panel': panel, 'panels_hash': canonical(panel),
           'half_cross': norm2(bias), 'noise_corrected_bias_squared': norm2(bias),
           'uncertainty': {'R': 256, 'mean_estimator_se_norm': floor.clone(), 'fixture_only': True}}
    return forward, cal


def _rank_from_scores(scores, valid):
    truth = defaultdict(set)
    for a, b in valid:
        truth[b].add(a)
    result = {}
    for a, b in valid:
        eligible = [i for i in range(len(scores)) if i != b and (i not in truth[b] or i == a)]
        values = scores[eligible, b]
        target = scores[a, b]
        result[a, b] = 1 + int((values > target).sum()) + .5 * (int((values == target).sum()) - 1)
    return result


@torch.no_grad()
def equivalence_fixture(config, seed, device, directory, deadline):
    """Native fixture and independently rerunnable full-array evidence."""
    model, data, view, panel = fixture_data(config, seed)
    model.to(device)
    features = data['features'].to(device)
    forward, cal = fixture_calibration(model, features, view, panel)
    initial_weights = weights_hashes(model)
    rng_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
    intervention = FrozenBiasIntervention(cal['p'], cal['b'], c=model.c,
        calibration_namespace=f'synthetic-calibration/seed{seed}', evaluation_namespace=f'{PROTOCOL}/seed{seed}/evaluation',
        direction_namespace=config['pilot']['q_namespace'].format(seed=seed), direction_seed=config['pilot']['q_seed'])
    double_model = copy.deepcopy(model).cpu().double()
    pairs = torch.cartesian_prod(torch.arange(40, device=device), torch.arange(40, device=device))
    truth = defaultdict(set)
    for a, b in data['valid']:
        truth[b].add(a)
    checks = dict.fromkeys(('FP64_reference', 'zero_native_identity', 'cache_scores', 'complete_ranking',
                           'shared_C_plan', 'q_plan_rng_isolation', 'cast_limits', 'archive_complete'), True)
    evidence = {'features': features, 'weights': model.state_dict(), 'base': cal['p'], 'bias': cal['b'],
                'q': intervention.control, 'plans': {}, 'native_points': {}, 'FP64_reference_points': {},
                'direct_scores': {}, 'cached_scores': {}, 'rankings': {}, 'casting': {},
                'valid': np.asarray(data['valid'], dtype=np.int64), 'adapter': intervention.metadata,
                'nodes': data['nodes'], 'neighbors': view.neighbors, 'view': view.metadata, 'panel': panel, 'comparisons': {}}
    full64 = double_model.encode(data['features'].double(), view.neighbors, [view.neighbors, view.neighbors])[0]
    evidence['FP64_reference_points']['F'] = full64
    checks['FP64_reference'] &= bool((distance(cal['p'].cpu(), promoted(full64, model.c), model.c) <= 1e-4).all())
    for repeat in (0, 8):
        deadline_check(deadline)
        result = evaluate_repeat(model, features, view, forward, intervention, config, seed, repeat)
        independent_plans, _ = repeat_plans(config, seed, repeat, view.neighbors)
        checks['shared_C_plan'] &= result['plans'] == independent_plans
        native = {'F': forward.reference['output'], **result['native']}
        for condition in ('O', 'Q'):
            native[condition], audit = cast_for_head(result['analytic'][condition], native['S'], result['sample64'],
                                                   cal['b'], cal['floor'], model.c, config['numeric_proposal'])
            evidence['casting'][f'{repeat}/{condition}'] = audit
        for condition, method in (('S', 'none'), ('C', 'third')):
            reference = double_model.encode(data['features'].double(), view.neighbors, independent_plans, method=method)[0]
            evidence['FP64_reference_points'][f'{repeat}/{condition}'] = reference
            checks['FP64_reference'] &= bool((distance(promoted(native[condition], model.c).cpu(), promoted(reference, model.c), model.c) <= 1e-4).all())
            expected_native = model.encode(features, view.neighbors, independent_plans, method=method,
                                            max_padded_messages=config['pilot']['max_padded_messages'])[0]
            checks['shared_C_plan'] &= torch.equal(expected_native, native[condition])
        evidence['plans'][str(repeat)] = result['plans']
        scores_by_condition = {}
        for condition, points in native.items():
            key = f'{repeat}/{condition}'
            direct = model.score(points, pairs).reshape(40, 40)
            cache = prepare_scores(model, points)
            # Match the complete-ranker call shapes, including its final chunk.
            cached = torch.stack([torch.cat([score_cached(model, cache,
                torch.arange(start, min(40, start + 11), device=device), child)
                for start in range(0, 40, 11)]) for child in range(40)], dim=1)
            error = float((direct - cached).abs().max())
            checks['cache_scores'] &= np.isfinite(error) and error <= config['numeric_proposal']['cache_absolute']
            ranking = filtered_parent_ranks(model, points, data['valid'], truth, candidate_chunk=11,
                                            max_seconds=min(180, deadline - time.monotonic()))
            validate_ranking(ranking, data)
            # Tie decisions are made from the exact cached scoring arithmetic.
            expected = _rank_from_scores(cached, data['valid'])
            checks['complete_ranking'] &= all(r['rank'] == expected[r['parent'], r['child']] for r in ranking['rows'])
            evidence['native_points'][key] = points
            evidence['direct_scores'][key] = direct
            evidence['cached_scores'][key] = cached
            evidence['rankings'][key] = ranking
            evidence['comparisons'][key] = {'cache_max_absolute': error}
            scores_by_condition[condition] = direct
        # Zero amplitude at every node must preserve native points, logits and complete ranks.
        zero_adapter = FrozenBiasIntervention(cal['p'], torch.zeros_like(cal['b']), c=model.c,
            calibration_namespace=f'synthetic-calibration/seed{seed}', evaluation_namespace=f'{PROTOCOL}/seed{seed}/evaluation',
            direction_namespace=config['pilot']['q_namespace'].format(seed=seed), direction_seed=config['pilot']['q_seed'])
        zero = zero_adapter.apply(result['sample64'], evaluation_namespace=zero_adapter.metadata['evaluation_namespace'])
        for condition in ('O', 'Q'):
            points, _ = cast_for_head(zero['points'][condition], native['S'], result['sample64'],
                                      zero_adapter.bias, cal['floor'], model.c, config['numeric_proposal'])
            scores = model.score(points, pairs).reshape(40, 40)
            ranks = filtered_parent_ranks(model, points, data['valid'], truth, candidate_chunk=11)
            checks['zero_native_identity'] &= (torch.equal(points, native['S']) and torch.equal(scores, scores_by_condition['S'])
                                               and ranks['rows'] == evidence['rankings'][f'{repeat}/S']['rows'])
            evidence.setdefault('zero_native_points', {})[f'{repeat}/{condition}'] = points
            evidence.setdefault('zero_scores', {})[f'{repeat}/{condition}'] = scores
            evidence.setdefault('zero_rankings', {})[f'{repeat}/{condition}'] = ranks
        checks['q_plan_rng_isolation'] &= torch.equal(rng_before, torch.get_rng_state())
        if cuda_before is not None:
            checks['q_plan_rng_isolation'] &= torch.equal(cuda_before, torch.cuda.get_rng_state(device))
    checks['q_plan_rng_isolation'] &= initial_weights == weights_hashes(model) and all(p.grad is None for p in model.parameters())
    if not all(checks.values()):
        raise ValueError('native fixture failed: ' + ', '.join(k for k, v in checks.items() if not v))
    evidence['checks'] = checks
    descriptor = write_archive(directory, f'fixture-seed{seed}', evidence)
    deadline_check(deadline)
    return checks, descriptor


def worker_deadline(args, config):
    from .recovery_entry import phase_seconds
    expected = phase_seconds(config, args.phase)
    if os.environ.get('ACL_FROZEN_RECOVERY_SUPERVISOR_PID') != str(os.getppid()):
        raise ValueError('worker must be launched by the direct deadline supervisor')
    deadline = float(os.environ.get('ACL_FROZEN_RECOVERY_DEADLINE', 'nan'))
    remaining = deadline - time.monotonic()
    if not np.isfinite(deadline) or not 0 < remaining <= expected:
        raise ValueError('fixed whole-worker monotonic deadline required')
    return deadline


def main():
    from .recovery_entry import parser, validate_arguments
    args = parser().parse_args()
    root = Path(args.output) if args.output else None
    try:
        config = json.loads(args.config.read_text(encoding='utf-8'))
        validate_config(config)
        validate_arguments(args)
        deadline = worker_deadline(args, config)
        identity = source_identity(args.source_commit)
        approval = json.loads(Path(args.approval_record).read_text(encoding='utf-8'))
        release = verify_release(config, identity, approval, args.quality_record, args.phase)
        device = torch.device('cpu' if args.phase == 'cpu_preflight' else 'cuda')
        require_runtime(config, device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        torch.set_num_threads(config['pilot']['threads'])
        row = {'status': 'started', 'protocol': PROTOCOL, 'phase': args.phase, 'config_sha256': CONFIG_SHA256,
               'source_commit': identity['source_commit'], 'source_lf_sha256': source_hashes(),
               'torch': str(torch.__version__), 'python': platform.python_version(), 'release': release,
               'seeds': [11, 23], 'active_phase': 'release_checks', 'completed_repetitions': 0,
               'slurm': {'job_id': os.environ.get('SLURM_JOB_ID'), 'step_id': os.environ.get('SLURM_STEP_ID'),
                         'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                         'visible_cuda_count': torch.cuda.device_count() if device.type == 'cuda' else None,
                         'placement_scope': 'single visible GPU job or step; parent allocation is not ACL GPU-time accounting'}}
        if args.phase == 'cpu_preflight':
            cache = VerifiedInputs()
            row.update(models={}, forward_backward_optimizer_called=False,
                       declared_resources=config['resources']['cpu_preflight'], device='cpu')
            for seed in (11, 23):
                row['active_phase'] = f'seed{seed}_input_loading'
                atomic_json(root / 'progress.json', row)
                model, data, view, cal, baseline, proof = load_inputs(config, seed, args.prepared_root, args.checkpoint_root,
                    args.training_release, args.reference_root, args.calibration_root, deadline, cache=cache)
                if sum(p.numel() for p in model.parameters()) != config['model']['parameter_count']:
                    raise ValueError('original parameter inventory mismatch')
                row['models'][str(seed)] = proof
                del model, data, view, cal, baseline
            row.update(status='complete', worker_input_receipts=cache.receipts,
                       all_bulk_inputs_verified_inside_worker=True, peak_vram=None)
            deadline_check(deadline)
            atomic_json(root / 'cpu-preflight.json', row)
        elif args.phase == 'cuda_fixture':
            row.update(archives={}, checks={}, device=torch.cuda.get_device_name(0), original_runtime_inputs_opened=[])
            for seed in (11, 23):
                row['active_phase'] = f'seed{seed}_synthetic_fixture'
                atomic_json(root / 'progress.json', row)
                checks, descriptor = equivalence_fixture(config, seed, device, root, deadline)
                row['archives'][str(seed)] = descriptor
                row['checks'] = {key: row['checks'].get(key, True) and value for key, value in checks.items()}
            row['status'] = 'complete'
            row['peak_vram'] = peak_memory(device)
            deadline_check(deadline)
            atomic_json(root / 'cuda-fixture.json', row)
        else:
            verify_gate(args.cpu_preflight_record, approval.get('cpu_preflight_record_sha256'), 'cpu_preflight', identity, config)
            verify_gate(args.cuda_fixture_record, approval.get('cuda_fixture_record_sha256'), 'cuda_fixture', identity, config)
            if approval.get('cpu_preflight_review_passed') is not True or approval.get('cuda_fixture_review_passed') is not True:
                raise ValueError('both native artifacts need supervisor review before science input loading')
            row['active_phase'] = 'original_input_loading'
            atomic_json(root / 'progress.json', row)
            model, data, view, cal, baseline, proof = load_inputs(config, args.seed, args.prepared_root, args.checkpoint_root,
                args.training_release, args.reference_root, args.calibration_root, deadline)
            provenance = {**identity, 'release_verified': True, 'release': release, 'input_proof': proof,
                          'slurm': row['slurm'],
                          'cpu_preflight_record_sha256': approval['cpu_preflight_record_sha256'],
                          'cuda_fixture_record_sha256': approval['cuda_fixture_record_sha256']}
            original_ranking = next(r for r in baseline['evaluations'] if r['step'] == config['checkpoints'][str(args.seed)]['step'])
            row = run_slice(model.to(device), data, view, cal, config, args.seed,
                            [args.repeat_start, args.repeat_start + 8], root, deadline,
                            provenance=provenance, expected_ranking=original_ranking)
        deadline_check(deadline)
        return 0 if row['status'] == 'complete' else 2
    except Exception as error:
        if root is not None and root.is_dir():
            atomic_json(root / 'worker-failure.json', {'status': 'failed', 'phase': args.phase,
                        'type': type(error).__name__, 'error': str(error), 'partial_results_are_complete': False})
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

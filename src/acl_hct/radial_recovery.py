"""Head-only evaluation of R/T on immutable original native-point archives."""
from collections import defaultdict
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .backbone import LorentzMeanNetwork
from .diagnostic_archive import write_archive
from .encoder_matched_control import (atomic_json, require_runtime, synchronize,
                                     validate_ranking, weights_hashes)
from .frozen_recovery import (_metrics, cast_for_head, geometry_summary, peak_memory,
                              promoted, ResourceTimeBudgetInadequate, tensor_identity)
from .geometry import dot, exp, from_spatial, log, norm2, tangent
from .radial_components import RadialBiasIntervention
from .radial_entry import CONFIG_SHA256, PROTOCOL, require, source_hashes, validate_config
from .ranking import filtered_parent_ranks, prepare_scores, score_cached
from .recovery_registration import canonical, validate_config as validate_old_config
from .recovery_relations import bound_archive, entry_design, validated_headers
from .vector_structure import RadialPanel


def deadline_check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('whole B worker deadline reached')


class RankingGateError(ValueError):
    def __init__(self, message, ranking):
        super().__init__(message)
        self.ranking = ranking


class ArchivedRelationHead(nn.Module):
    """Only the original four head parameters; meta construction draws no RNG."""
    score = LorentzMeanNetwork.score

    def __init__(self, weights, c=1., device='cpu', dimension=None):
        super().__init__()
        selected = {k: torch.as_tensor(v) for k, v in weights.items() if k.startswith('relation_head.')}
        keys = {'relation_head.0.weight', 'relation_head.0.bias', 'relation_head.2.weight', 'relation_head.2.bias'}
        require(set(selected) == keys, 'exact original four head parameters required')
        first = selected['relation_head.0.weight']
        require(first.ndim == 2 and first.shape[1] % 3 == 0 and first.shape[0] > 0,
                'original head dimensions required')
        hidden, width = first.shape[1] // 3, first.shape[0]
        require(dimension is None or hidden == width == dimension, 'registered production head dimensions required')
        shapes = {'relation_head.0.weight': (width, 3 * hidden), 'relation_head.0.bias': (width,),
                  'relation_head.2.weight': (1, width), 'relation_head.2.bias': (1,)}
        require(all(v.dtype == torch.float32 and tuple(v.shape) == shapes[k] and torch.isfinite(v).all()
                    for k, v in selected.items()), 'finite original FP32 head parameters required')
        self.c = c
        self.relation_head = nn.Sequential(nn.Linear(3 * hidden, width, device='meta', dtype=torch.float32), nn.ReLU(),
                                           nn.Linear(width, 1, device='meta', dtype=torch.float32))
        self.to_empty(device=device)
        self.load_state_dict(selected, strict=True)
        self.eval().requires_grad_(False)


def truth_for(data):
    truth = defaultdict(set)
    for parent, child in data['valid']:
        truth[child].add(parent)
    return truth


def rank_points(model, native, data, config, deadline, *, expected=None):
    deadline_check(deadline)
    require(native.dtype == torch.float32 and torch.isfinite(native).all(), 'finite native FP32 points required')
    synchronize(native.device)
    started = time.monotonic()
    result = filtered_parent_ranks(model, native, data['valid'], truth_for(data),
                                   config['ranking']['candidate_chunk'],
                                   min(config['ranking']['single_complete_seconds'], deadline - started))
    synchronize(native.device)
    result['measured_complete_seconds'] = time.monotonic() - started
    try:
        validate_ranking(result, data)
        require(result['measured_complete_seconds'] <= config['ranking']['single_complete_seconds'],
                'complete ranking exceeded fixed single-ranking bound')
        if expected is not None:
            validate_ranking(expected, data)
            require(result['rows'] == expected['rows'], 'saved F full ranks/order/candidate counts did not reproduce exactly')
    except ValueError as error:
        raise RankingGateError(str(error), result) from error
    deadline_check(deadline)
    return result


def cast_components(adapter, sample_native, floor, config):
    require(sample_native.device.type == 'cpu', 'fixed scientific CPU geometry required')
    sample64 = promoted(sample_native, adapter.c)
    moved = adapter.apply(sample64)
    native, casting = {}, {}
    for name in ('R', 'T'):
        native[name], casting[name] = cast_for_head(moved['points'][name], sample_native, sample64,
                                                  moved['removed_fields'][name], floor, adapter.c, config['numeric'])
    return sample64, moved, native, casting


def feasibility(measured, deadline, config):
    rule = config['resources']['pre_result_feasibility']
    require(type(measured) in (int, float) and math.isfinite(measured) and measured >= 0,
            'finite nonnegative complete F timing required')
    remaining = deadline - time.monotonic()
    estimate = measured * rule['remaining_complete_rankings'] + rule['nonranking_reserve_seconds']
    return {'measured_F_ranking_seconds': measured, **rule, 'estimated_remaining_seconds': estimate,
            'remaining_worker_seconds': remaining, 'passed': estimate <= remaining,
            'policy': 'resource-only stop before R/T; preserve F; retain same original sample identities'}


def finite_move_diagnostics(anchor, sample64, moved64, floor, root_floor, c=1.):
    """Finite radial and angular movement around the fixed original root.

    Angular entries at unresolved root directions are NaN placeholders, with an
    explicit mask; they are never treated as zero changes or dropped primaries.
    """
    before, after = log(anchor, sample64, c), log(anchor, moved64, c)
    radius_before, radius_after = norm2(before).sqrt(), norm2(after).sqrt()
    threshold = torch.maximum(floor + root_floor, torch.full_like(floor, 1e-10))
    defined = (radius_before > threshold) & (radius_after > threshold)
    cosine = torch.full_like(floor, float('nan'))
    angle = cosine.clone()
    if defined.any():
        value = dot(before[defined], after[defined]) / (radius_before[defined] * radius_after[defined])
        require(torch.isfinite(value).all(), 'finite resolved directional diagnostic required')
        # Rounding guard for acos only; no intervention is clipped or changed.
        cosine[defined] = value.clamp(-1., 1.)
        cosine[defined & (sample64 == moved64).all(dim=-1)] = 1.
        angle[defined] = torch.acos(cosine[defined])
    return {'radial_change': radius_after - radius_before, 'direction_cosine': cosine,
            'direction_angle_radians': angle, 'direction_defined': defined,
            'undefined_count': int((~defined).sum()),
            'direction_threshold': threshold,
            'scope': 'finite movement around fixed p[root]; angle undefined when either radius <= max(floor_i+floor_root,1e-10); NaN plus mask, all primary nodes retained'}


def synthetic_data(config, seed, case):
    """Explicit fixture-only generator; never called by scientific execution."""
    require(seed in (11, 23) and case in ('mixed', 'zero'), 'fixed synthetic fixture case required')
    generator = torch.Generator(device='cpu').manual_seed(config['fixture']['explicit_CPU_generator_seed_base'] + seed)
    n, dim = config['fixture']['nodes'], config['fixture']['head_dimension']
    spatial = torch.randn((n, dim), generator=generator, dtype=torch.float64) * .035
    spatial[0] = 0
    spatial[0, 0] = .3
    spatial[1] = 0
    spatial[2] = spatial[0]
    spatial[2, 0] += 1e-12
    spatial[4] = 0
    base = from_spatial(spatial)
    raw = torch.randn((n, dim + 1), generator=generator, dtype=torch.float64) * .0015
    bias = tangent(base, raw)
    bias[1] = 0
    bias[1, 1] = .025  # pure radial at the origin
    bias[3] = 0
    bias[4] = 0
    bias[4, 2] = .025  # pure orthogonal at the origin
    if case == 'zero':
        bias.zero_()
    floor = torch.full((n,), 1e-6, dtype=torch.float64)
    error = tangent(base, torch.randn((n, dim + 1), generator=generator, dtype=torch.float64) * .002)
    native_s = exp(base, error).float()
    weights = {'relation_head.0.weight': torch.randn((dim, 3 * dim), generator=generator) * .05,
               'relation_head.0.bias': torch.randn((dim,), generator=generator) * .01,
               'relation_head.2.weight': torch.randn((1, dim), generator=generator) * .05,
               'relation_head.2.bias': torch.randn((1,), generator=generator) * .01}
    nodes = [f'fixture-{i}' for i in range(n)]
    valid = [(0, 5), (1, 5), (2, 6), (4, 7), (3, 8), (9, 10), (10, 11), (11, 12)]
    panel = {'rows': [{'id': nodes[i], 'pool_mean_weight': 1 / 12} for i in range(4, 16)],
             'relations': [{'child': nodes[i], 'direct_parents': [nodes[i - 1]],
                            'positive_distant_ancestors': [nodes[0]]} for i in range(4, 16)]}
    view = SimpleNamespace(nodes=nodes, root=nodes[0], reachable=set(nodes))
    return base, bias, floor, native_s, weights, {'nodes': nodes, 'valid': valid}, view, panel


@torch.no_grad()
def fixture_case(config, seed, case, device, output, deadline):
    base, bias, floor, sample, weights, data, view, panel_spec = synthetic_data(config, seed, case)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
    head = ArchivedRelationHead(weights, config['model']['c'], device, config['fixture']['head_dimension'])
    original_weights = weights_hashes(head)
    adapter = RadialBiasIntervention(base, bias, floor, 0, head.c)
    sample64, moved, native, casting = cast_components(adapter, sample, floor, config)
    native['S'] = sample
    gpu_adapter = RadialBiasIntervention(base.to(device), bias.to(device), floor.to(device), 0, head.c)
    gpu_sample64 = promoted(sample.to(device), head.c)
    gpu_moved = gpu_adapter.apply(gpu_sample64)
    cpu_panel = RadialPanel(view, panel_spec, base, {'V': list(range(len(base)))}, head.c, floor)
    pairs = torch.cartesian_prod(torch.arange(len(base), device=device), torch.arange(len(base), device=device))
    direct, cached, rankings, structure = {}, {}, {}, {}
    checks = {'FP64_reference': True, 'zero_native_identity': True, 'cache_scores': True,
              'complete_ranking': True, 'head_and_RNG_unchanged': True, 'cast_limits': True,
              'undefined_unapplied_bias': True, 'archive_complete': True}
    for name in ('S', 'R', 'T'):
        point = native[name].to(device)
        direct[name] = head.score(point, pairs).reshape(len(base), len(base))
        cache = prepare_scores(head, point)
        cached[name] = score_cached(head, cache, pairs[:, 0], pairs[:, 1]).reshape(len(base), len(base))
        checks['cache_scores'] &= float((direct[name] - cached[name]).abs().max()) <= config['fixture']['score_absolute_tolerance']
        rankings[name] = rank_points(head, point, data, config, deadline)
        structure[name] = cpu_panel.evaluate(promoted(native[name], head.c))
    for name in ('R', 'T'):
        checks['FP64_reference'] &= bool(torch.allclose(moved['points'][name], gpu_moved['points'][name].cpu(),
                                                      atol=config['fixture']['FP64_geometry_absolute_tolerance'], rtol=0))
        zero = norm2(adapter.fields[name]) == 0
        checks['zero_native_identity'] &= torch.equal(native[name][zero], sample[zero])
        zero_pairs = zero[pairs[:, 0].cpu()] & zero[pairs[:, 1].cpu()]
        checks['zero_native_identity'] &= torch.equal(direct[name].flatten()[zero_pairs.to(device)],
                                                     direct['S'].flatten()[zero_pairs.to(device)])
        if case == 'zero':
            checks['zero_native_identity'] &= (torch.equal(cached[name], cached['S'])
                                                and rankings[name]['rows'] == rankings['S']['rows'])
    checks['undefined_unapplied_bias'] = (not adapter.defined[0] and not adapter.defined[2]
                                          and torch.equal(adapter.unapplied_bias[~adapter.defined], bias[~adapter.defined]))
    checks['head_and_RNG_unchanged'] = (original_weights == weights_hashes(head)
                                       and torch.equal(cpu_rng, torch.get_rng_state())
                                       and all(p.grad is None for p in head.parameters()))
    if cuda_rng is not None:
        checks['head_and_RNG_unchanged'] &= torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    require(all(bool(v) for v in checks.values()), 'synthetic B checks failed')
    evidence = {'seed': seed, 'case': case, 'base': base, 'bias': bias, 'floor': floor, 'root_index': 0,
                'sample64': sample64, 'weights': weights, 'nodes': data['nodes'],
                'valid': np.asarray(data['valid'], dtype=np.int64), 'panel': panel_spec,
                'structure_view': {'root': view.root, 'reachable': sorted(view.reachable)},
                'outward_unit': adapter.outward_unit, 'defined': adapter.defined,
                'unapplied_bias': adapter.unapplied_bias, 'direction_resolution': adapter.direction_resolution,
                'removed_fields': adapter.fields, 'offsets': moved['offsets'],
                'FP64_reference_points': moved['points'], 'GPU_FP64_points': gpu_moved['points'],
                'GPU_removed_fields': gpu_adapter.fields, 'native_points': native, 'casting': casting,
                'direct_scores': direct, 'cached_scores': cached, 'rankings': rankings,
                'structure': structure, 'checks': {k: bool(v) for k, v in checks.items()}}
    descriptor = write_archive(output, f'fixture-seed{seed}-{case}', evidence)
    deadline_check(deadline)
    return evidence['checks'], descriptor


@torch.no_grad()
def run_saved_shard(config, old_config, bindings, roots, seed, start, output, deadline, provenance,
                    *, engineering=False):
    validate_config(config)
    validate_old_config(old_config)
    require(seed in (11, 23) and start in (0, 8), 'fixed model and original repeat shard required')
    device = torch.device('cpu' if engineering else 'cuda:0')
    require(engineering or provenance.get('release_verified') is True, 'native reviewed science release required')
    rows = validated_headers(roots, bindings, old_config)
    index = next(i for i, s in enumerate(bindings['shards']) if s['seed'] == seed and s['repeat_range'] == [start, start + 8])
    original, spec, original_root = rows[index], bindings['shards'][index], roots[index]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / 'run.json').exists(), 'new B shard output required')
    entry = bound_archive(original_root, original['entry_archive'], spec['archives'][0], deadline)
    analyzer, data = entry_design(entry, original, old_config)
    base, bias, floor = (torch.from_numpy(entry[k]) for k in ('base', 'bias', 'floor'))
    adapter = RadialBiasIntervention(base, bias, floor, analyzer.panel.root, config['model']['c'])
    head = ArchivedRelationHead(entry['weights'], config['model']['c'], device, config['model']['hidden'])
    initial_weights = weights_hashes(head)
    input_identity = {k: tensor_identity(torch.from_numpy(entry[k])) for k in ('base', 'bias', 'floor', 'native_F')}
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
    row = {'status': 'verification_started', 'protocol': PROTOCOL, 'config_sha256': CONFIG_SHA256,
           'seed': seed, 'repeat_range': [start, start + 8], 'observations': [], 'repeat_archives': [],
           'engineering_fixture_only': engineering, 'provenance': provenance, 'identity': original['identity'],
           'old_run_raw_sha256': spec['run_raw_sha256'], 'input_shard_key': spec['key'],
           'completed_repetitions': 0, 'GNN_forward_calls': 0, 'new_samples': 0, 'model_updates': 0}

    def progress(phase):
        row.update(active_phase=phase, completed_repetitions=len(row['observations']), peak_vram=peak_memory(device))
        atomic_json(output / 'run.json', row)
        deadline_check(deadline)

    try:
        progress('F_ranking_exact_replay')
        full = rank_points(head, torch.from_numpy(entry['native_F']).to(device), data, config, deadline,
                           expected=entry['F_ranking'])
        row['F_replay_archive'] = write_archive(output, 'F-replay',
                                                {'ranking': full, 'old_run_raw_sha256': spec['run_raw_sha256'],
                                                 'original_entry_descriptor': original['entry_archive']})
        row['F_full_rank_exact'] = True
        row['resource_feasibility'] = feasibility(full['measured_complete_seconds'], deadline, config)
        progress('resource_time_feasibility')
        if not row['resource_feasibility']['passed']:
            raise ResourceTimeBudgetInadequate('complete F timing predicts insufficient budget before any R/T result')
        row['component_archive'] = write_archive(output, 'components',
            {'root_index': analyzer.panel.root, 'metadata': adapter.metadata, 'removed_fields': adapter.fields,
             'outward_unit': adapter.outward_unit, 'defined': adapter.defined,
             'unapplied_bias': adapter.unapplied_bias, 'direction_resolution': adapter.direction_resolution,
             'original_entry_descriptor': original['entry_archive'], 'old_run_raw_sha256': spec['run_raw_sha256']})
        for observation, descriptor, archive_spec in zip(original['observations'], original['repeat_archives'], spec['archives'][1:]):
            repeat = observation['repeat']
            row['active_repeat'] = repeat
            progress('original_repeat_verification')
            saved = bound_archive(original_root, descriptor, archive_spec, deadline)
            require(saved['repeat'] == repeat == archive_spec['global_repeat_id']
                    and saved['plan_hash'] == observation['plan_hash'] == archive_spec['plan_hash']
                    and saved['rng'] == observation['rng'] == archive_spec['sampling_rng_identity'], 'original sample/plan/RNG identity differs')
            sample = torch.from_numpy(saved['native_points']['S'])
            sample_identity = tensor_identity(sample)
            validate_ranking(saved['rankings']['S'], data)
            sample_structure = analyzer.panel.evaluate(promoted(sample, config['model']['c']))
            require(canonical(sample_structure) == canonical(saved['structure']['S']), 'original S structure does not reproduce')
            metrics = {'S': _metrics(sample_structure, saved['rankings']['S'])}
            require(metrics['S'] == observation['metrics']['S'], 'original S metrics differ from bound observations')
            progress('component_geometry_and_cast')
            sample64, moved, native, casting = cast_components(adapter, sample, floor, config)
            rankings, structures, geometry, finite_moves = {}, {}, {}, {}
            for name in ('R', 'T'):
                progress(name + '_ranking')
                rankings[name] = rank_points(head, native[name].to(device), data, config, deadline)
                points64 = promoted(native[name], config['model']['c'])
                structures[name] = analyzer.panel.evaluate(points64)
                metrics[name] = _metrics(structures[name], rankings[name])
                geometry[name] = geometry_summary(points64, base, floor, config['model']['c'])
                finite_moves[name] = finite_move_diagnostics(base[analyzer.panel.root], sample64, points64,
                                                           floor, floor[analyzer.panel.root], adapter.c)
                require(all(type(metrics[name][k]) in (float, int) and math.isfinite(metrics[name][k])
                            for k in ('direct', 'distant', 'micro_mrr')), 'unknown/nonfinite primary is not zero')
            require(sample_identity == tensor_identity(sample), 'original native S mutated')
            new_observation = {'repeat': repeat, 'metrics': metrics, 'plan_hash': saved['plan_hash'], 'rng': saved['rng'],
                'original_repeat_descriptor': descriptor, 'original_S_identity': sample_identity,
                'paired_differences': {comparison: {metric: metrics[a][metric] - metrics[b][metric]
                    for metric in ('direct', 'distant', 'micro_mrr')} for comparison, a, b in
                    [('R-S', 'R', 'S'), ('T-S', 'T', 'S'), ('R-T', 'R', 'T')]},
                'casting': {name: {key: value for key, value in record.items() if key in
                    ('resolved_fraction', 'max_error', 'max_relative_resolved', 'zero_native_identity')}
                    for name, record in casting.items()}, 'geometry': geometry}
            progress('repeat_archive')
            row['repeat_archives'].append(write_archive(output, f'repeat-{repeat:02}',
                {'repeat': repeat, 'native_points': native, 'rankings': rankings, 'structure': structures,
                 'casting': casting, 'geometry': geometry, 'finite_move_diagnostics': finite_moves,
                 'plan_hash': saved['plan_hash'], 'rng': saved['rng'], 'original_S_identity': sample_identity,
                 'original_repeat_descriptor': descriptor}))
            row['observations'].append(new_observation)
            del saved, sample, sample64, moved, native, casting
            progress('repeat_complete')
        row['status'] = 'complete'
    except Exception as error:
        row['status'] = ('resource_time_budget_inadequate' if isinstance(error, ResourceTimeBudgetInadequate)
                         else 'incomplete_time_limit' if isinstance(error, TimeoutError) else 'failed')
        row['failure'] = {'type': type(error).__name__, 'error': str(error), 'phase': row['active_phase'],
                          'partial_results_are_complete': False}
        if isinstance(error, RankingGateError):
            row['failed_ranking_archive'] = write_archive(output, 'failed-ranking',
                {'active_repeat': row.get('active_repeat'), 'active_phase': row['active_phase'],
                 'ranking': error.ranking, 'failure': row['failure']})
    finally:
        row['weights_unchanged'] = initial_weights == weights_hashes(head)
        row['input_unchanged'] = all(tensor_identity(torch.from_numpy(entry[k])) == value for k, value in input_identity.items())
        row['RNG_unchanged'] = torch.equal(cpu_rng, torch.get_rng_state())
        if cuda_rng is not None:
            row['RNG_unchanged'] &= torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
        row['gradients_present'] = any(p.grad is not None for p in head.parameters())
        if row['status'] == 'complete' and (not row['weights_unchanged'] or not row['input_unchanged'] or not row['RNG_unchanged'] or row['gradients_present']):
            row.update(status='failed', failure={'error': 'frozen inputs/head/RNG/gradients contract failed'})
        row['completed_repetitions'] = len(row['observations'])
        if time.monotonic() >= deadline and row['status'] == 'complete':
            row['status'] = 'incomplete_time_limit'
        atomic_json(output / 'run.json', row)
    return row


def worker(args, config, provenance, deadline):
    require(str(torch.__version__) == config['runtime']['torch'] and np.__version__ == config['runtime']['numpy'],
            'exact original tensor/array runtime required')
    torch.set_num_threads(config['runtime']['scientific_CPU_threads'])
    device = torch.device('cuda:0')
    require_runtime(config, device)
    deadline_check(deadline)
    if args.phase == 'cuda_fixture':
        row = {'status': 'started', 'phase': args.phase, 'protocol': PROTOCOL, 'config_sha256': CONFIG_SHA256,
               'source_commit': args.source_commit, 'source_lf_sha256': source_hashes(),
               'torch': str(torch.__version__), 'numpy': np.__version__, 'device': torch.cuda.get_device_name(device),
               'seeds': [11, 23], 'original_runtime_inputs_opened': [], 'archives': {}, 'checks': {}, 'provenance': provenance}
        atomic_json(args.output / 'run.json', row)
        for seed in (11, 23):
            for case in ('mixed', 'zero'):
                checks, archive = fixture_case(config, seed, case, device, args.output, deadline)
                row['archives'][f'{seed}/{case}'] = archive
                for name, passed in checks.items():
                    row['checks'][name] = row['checks'].get(name, True) and passed
                atomic_json(args.output / 'run.json', row)
        deadline_check(deadline)
        row['status'] = 'complete'
        atomic_json(args.output / 'run.json', row)
        return 0
    row = run_saved_shard(config, json.loads(args.old_config.read_bytes()), json.loads(args.bindings.read_bytes()),
                          args.shards, args.seed, args.repeat_start, args.output, deadline, provenance)
    return 0 if row['status'] == 'complete' else 2

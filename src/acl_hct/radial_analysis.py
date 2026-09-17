"""CPU archive reproduction and the predeclared complete 18-test R/T family."""
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .diagnostic_archive import read_archive
from .encoder_matched_control import atomic_json, validate_ranking
from .frozen_recovery import _metrics, promoted, tensor_identity
from .radial_components import RadialBiasIntervention
from .radial_entry import CONFIG_SHA256, PROTOCOL, require, source_hashes, validate_config
from .radial_recovery import cast_components, finite_move_diagnostics, geometry_summary
from .recovery_registration import canonical, file_sha256, validate_config as validate_old_config
from .recovery_relations import bound_archive, entry_design, validated_headers

METRICS = ('direct', 'distant', 'micro_mrr')
COMPARISONS = ('R-S', 'T-S', 'R-T')


def equal_archive_values(left, right):
    if isinstance(left, (np.ndarray, torch.Tensor)) or isinstance(right, (np.ndarray, torch.Tensor)):
        if isinstance(left, torch.Tensor):
            left = left.detach().cpu().numpy()
        if isinstance(right, torch.Tensor):
            right = right.detach().cpu().numpy()
        return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and left.dtype == right.dtype and np.array_equal(left, right, equal_nan=True))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equal_archive_values(left[k], right[k]) for k in left)
    return left == right


def merge_shards(shards):
    """Pair graph repeats within each fixed model; no training-seed inference."""
    from scipy.stats import t
    require(len(shards) == 4, 'exactly four complete B shards required')
    by_seed, identities = {11: {}, 23: {}}, {}
    for shard in shards:
        require(shard['status'] == 'complete' and shard['protocol'] == PROTOCOL
                and shard['config_sha256'] == CONFIG_SHA256 and type(shard['seed']) is int
                and shard['seed'] in by_seed, 'complete registered B shard required')
        seed = shard['seed']
        fingerprint = canonical(shard['identity'])
        require(seed not in identities or identities[seed] == fingerprint, 'cross-shard fixed identity drift')
        identities[seed] = fingerprint
        interval = shard['repeat_range']
        require(interval in ([0, 8], [8, 16])
                and [r['repeat'] for r in shard['observations']] == list(range(*interval)), 'complete ordered original range required')
        for row in shard['observations']:
            repeat = row['repeat']
            require(type(repeat) is int and repeat not in by_seed[seed], 'duplicate global repeat')
            require(set(row['metrics']) == {'S', 'R', 'T'}, 'exact three paired conditions required')
            require(all(type(values[k]) in (int, float) and math.isfinite(values[k])
                        for values in row['metrics'].values() for k in METRICS), 'finite primary metrics; unknown is not zero')
            for comparison in COMPARISONS:
                a, b = comparison.split('-')
                require(all(abs(row['paired_differences'][comparison][k] - (row['metrics'][a][k] - row['metrics'][b][k])) <= 1e-12
                            for k in METRICS), 'saved paired difference mismatch')
            by_seed[seed][repeat] = row
    family = []
    for seed, rows in by_seed.items():
        require(set(rows) == set(range(16)), 'both full original 16-repeat inventories required')
        for comparison in COMPARISONS:
            a, b = comparison.split('-')
            for metric in METRICS:
                values = np.array([rows[i]['metrics'][a][metric] - rows[i]['metrics'][b][metric] for i in range(16)])
                mean, se = float(values.mean()), float(values.std(ddof=1) / 4)
                inferable = se > 0
                width = float(t.ppf(.975, 15) * se) if inferable else None
                family.append({'seed': seed, 'comparison': comparison, 'metric': metric,
                               'differences': values.tolist(), 'mean': mean, 'mc_se': se,
                               'marginal_95_interval': [mean - width, mean + width] if inferable else None,
                               'two_sided_p': float(2 * t.sf(abs(mean / se), 15)) if inferable else 1.,
                               'inferable': inferable, 'zero_observed_variance': not inferable,
                               'reused_graph_repetitions': 16, 'df': 15})
    previous = 0.
    for position, index in enumerate(sorted(range(18), key=lambda i: family[i]['two_sided_p'])):
        previous = max(previous, min(1., (18 - position) * family[index]['two_sided_p']))
        family[index]['holm_adjusted_p'] = previous
    return {'status': 'complete', 'protocol': PROTOCOL, 'config_sha256': CONFIG_SHA256,
            'family_size': 18, 'primary_comparisons': family,
            'scope': 'exploratory paired mechanism analysis on previously observed samples; fixed calibration uncertainty excluded; no independent confirmation or training-seed inference',
            'stop_boundary': 'deliver current E2 A/B, then discuss with user before more experiments/module/training/stage'}


def verify_repeat(payload, observation, saved, panel, data, adapter, floor, config):
    """Derive new native points and all cast/structure/rank summaries on CPU."""
    require(payload['repeat'] == observation['repeat'] == saved['repeat']
            and payload['plan_hash'] == observation['plan_hash'] == saved['plan_hash']
            and payload['rng'] == observation['rng'] == saved['rng'], 'original repeat/plan/RNG differs')
    sample = torch.from_numpy(saved['native_points']['S'])
    require(payload['original_S_identity'] == observation['original_S_identity'] == tensor_identity(sample), 'original native S identity differs')
    require(set(payload['native_points']) == set(payload['rankings']) == set(payload['structure']) == {'R', 'T'},
            'complete R/T point/rank/structure inventory required')
    sample64, _, native, casting = cast_components(adapter, sample, floor, config)
    metrics = {}
    for name in ('R', 'T'):
        points = torch.from_numpy(payload['native_points'][name])
        require(points.dtype == torch.float32 and torch.equal(points, native[name]), 'new native points do not reproduce fixed R/T intervention')
        require(equal_archive_values(payload['casting'][name], casting[name]),
                'saved cast arrays/summary differ from CPU reproduction')
        validate_ranking(payload['rankings'][name], data)
        points64 = promoted(points, adapter.c)
        structure = panel.evaluate(points64)
        require(canonical(structure) == canonical(payload['structure'][name]), 'new saved structure differs from native points')
        metrics[name] = _metrics(structure, payload['rankings'][name])
        if 'geometry' in payload:
            require(payload['geometry'][name] == geometry_summary(points64, panel.reference, floor, adapter.c), 'geometry auxiliary differs')
        if 'finite_move_diagnostics' in payload:
            expected = finite_move_diagnostics(panel.reference[panel.root], sample64, points64,
                                               floor, floor[panel.root], adapter.c)
            require(equal_archive_values(payload['finite_move_diagnostics'][name], expected), 'finite movement auxiliary differs')
    validate_ranking(saved['rankings']['S'], data)
    s_structure = panel.evaluate(promoted(sample, adapter.c))
    require(canonical(s_structure) == canonical(saved['structure']['S']), 'original S structure differs')
    metrics['S'] = _metrics(s_structure, saved['rankings']['S'])
    require(metrics == observation['metrics'], 'metrics differ from complete archives')
    return metrics


def load_shard(directory, config, old_config, bindings, original_roots, original_rows, deadline):
    root = Path(directory)
    row = json.loads((root / 'run.json').read_bytes())
    supervisor = json.loads((root / 'supervisor.json').read_bytes())
    require(row['status'] == 'complete' and row['protocol'] == PROTOCOL and row['config_sha256'] == CONFIG_SHA256
            and row['engineering_fixture_only'] is False and row['completed_repetitions'] == 8
            and row['weights_unchanged'] is True and row['input_unchanged'] is True and row['RNG_unchanged'] is True
            and row['gradients_present'] is False and row['model_updates'] == row['new_samples'] == row['GNN_forward_calls'] == 0
            and row['F_full_rank_exact'] is True and row['resource_feasibility']['passed'] is True
            and row['provenance']['release_verified'] is True and row['provenance']['new_source_lf_sha256'] == source_hashes()
            and supervisor['status'] == 'complete' and supervisor['worker_exit_code'] == 0 and supervisor['deadline_seconds'] == 840
            and supervisor['context']['protocol'] == PROTOCOL and supervisor['context']['phase'] == 'science'
            and supervisor['context']['config_sha256'] == CONFIG_SHA256
            and supervisor['context']['source_commit'] == row['provenance']['source_commit'], 'complete same-source supervised scientific shard required')
    matches = [i for i, spec in enumerate(bindings['shards']) if spec['seed'] == row['seed'] and spec['repeat_range'] == row['repeat_range']]
    require(len(matches) == 1 and len(row['repeat_archives']) == len(row['observations']) == 8, 'fixed original shard and complete inventory required')
    index = matches[0]
    spec, old_row, old_root = bindings['shards'][index], original_rows[index], original_roots[index]
    require(row['identity'] == old_row['identity'] and row['old_run_raw_sha256'] == spec['run_raw_sha256'], 'fixed original identity differs')
    entry = bound_archive(old_root, old_row['entry_archive'], spec['archives'][0], deadline)
    analyzer, data = entry_design(entry, old_row, old_config)
    adapter = RadialBiasIntervention(torch.from_numpy(entry['base']), torch.from_numpy(entry['bias']),
                                    torch.from_numpy(entry['floor']), analyzer.panel.root, config['model']['c'])
    components = read_archive(root, row['component_archive'])
    require(components['metadata'] == adapter.metadata and components['root_index'] == analyzer.panel.root,
            'fixed component identity differs')
    for name in ('R', 'T'):
        require(np.array_equal(components['removed_fields'][name], adapter.fields[name].numpy()), 'component arrays differ')
    require(np.array_equal(components['defined'], adapter.defined.numpy())
            and np.array_equal(components['outward_unit'], adapter.outward_unit.numpy())
            and np.array_equal(components['unapplied_bias'], adapter.unapplied_bias.numpy()), 'undefined/unit/unapplied arrays differ')
    require(equal_archive_values(components['direction_resolution'], adapter.direction_resolution)
            and components['original_entry_descriptor'] == old_row['entry_archive']
            and components['old_run_raw_sha256'] == spec['run_raw_sha256'], 'component reference/resolution differs')
    replay = read_archive(root, row['F_replay_archive'])
    validate_ranking(replay['ranking'], data)
    require(replay['ranking']['rows'] == entry['F_ranking']['rows'], 'saved F exact replay differs')
    require(replay['original_entry_descriptor'] == old_row['entry_archive']
            and replay['old_run_raw_sha256'] == spec['run_raw_sha256'], 'F replay reference differs')
    for observation, descriptor, old_observation, old_descriptor, archive_spec in zip(
            row['observations'], row['repeat_archives'], old_row['observations'], old_row['repeat_archives'], spec['archives'][1:]):
        saved = bound_archive(old_root, old_descriptor, archive_spec, deadline)
        require(observation['repeat'] == old_observation['repeat'] and observation['metrics']['S'] == old_observation['metrics']['S']
                and observation['original_repeat_descriptor'] == old_descriptor, 'original observation/reference differs')
        payload = read_archive(root, descriptor)
        require({'casting', 'geometry', 'finite_move_diagnostics'} <= set(payload), 'complete numeric/finite movement auxiliary required')
        require(payload['original_repeat_descriptor'] == old_descriptor, 'original payload/reference differs')
        verify_repeat(payload, observation, saved, analyzer.panel, data, adapter, torch.from_numpy(entry['floor']), config)
    return row


def main():
    import argparse
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'old-config', 'bindings', 'output'):
        cli.add_argument('--' + name, type=Path, required=True)
    cli.add_argument('--original-shards', nargs=4, type=Path, required=True)
    cli.add_argument('--shards', nargs=4, type=Path, required=True)
    args = cli.parse_args()
    require(not args.output.exists(), 'new CPU merge output required')
    config, old_config, bindings = (json.loads(p.read_bytes()) for p in (args.config, args.old_config, args.bindings))
    validate_config(config)
    validate_old_config(old_config)
    require(file_sha256(args.bindings) == config['original_input_catalog_raw_sha256']
            and file_sha256(args.old_config) == bindings['old_config_raw_sha256'], 'exact original input bytes required')
    require(str(torch.__version__) == config['runtime']['torch'] and np.__version__ == config['runtime']['numpy'], 'original CPU runtime required')
    torch.set_num_threads(4)
    deadline = time.monotonic() + 1680
    old_rows = validated_headers(args.original_shards, bindings, old_config)
    rows = [load_shard(p, config, old_config, bindings, args.original_shards, old_rows, deadline) for p in args.shards]
    require(len({r['provenance']['source_commit'] for r in rows}) == 1, 'cross-shard release differs')
    result = merge_shards(rows)
    result['shard_run_raw_sha256'] = [file_sha256(p / 'run.json') for p in args.shards]
    result['original_bindings_raw_sha256'] = file_sha256(args.bindings)
    atomic_json(args.output, result)


if __name__ == '__main__':
    main()

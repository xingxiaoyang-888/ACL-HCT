"""Merge four fixed shards and reproduce the entire 18-comparison family."""
import math
from .recovery_registration import CONFIG_SHA256, PROTOCOL, canonical

METRICS = ('direct', 'distant', 'micro_mrr')
COMPARISONS = ('C-S', 'O-S', 'O-Q')


def merge_shards(shards):
    """Inputs are decoded, hash-verified observation archives, not chosen rows."""
    import numpy as np
    from scipy.stats import t
    if len(shards) != 4:
        raise ValueError('exactly four complete fixed shards required')
    by_seed = {11: {}, 23: {}}
    identities = {}
    references = {}
    for shard in shards:
        if (shard['status'] != 'complete' or shard['protocol'] != PROTOCOL
                or shard['config_sha256'] != CONFIG_SHA256 or shard['seed'] not in by_seed):
            raise ValueError('complete registered shard required')
        seed = shard['seed']
        fingerprint = canonical(shard['identity'])
        if seed in identities and identities[seed] != fingerprint:
            raise ValueError('calibration/weights/panels/F identity differs across shards')
        identities[seed] = fingerprint
        references[seed] = shard['identity']['F_metrics']
        expected = list(range(*shard['repeat_range']))
        if shard['repeat_range'] not in ([0, 8], [8, 16]) or [row['repeat'] for row in shard['observations']] != expected:
            raise ValueError('complete ordered eight-repeat shard required')
        for row in shard['observations']:
            i = row['repeat']
            if i in by_seed[seed]:
                raise ValueError('duplicate global repeat')
            if set(row['metrics']) != {'S', 'O', 'C', 'Q'}:
                raise ValueError('all four paired conditions required')
            for condition in row['metrics'].values():
                if any(type(condition[k]) not in (int, float) or not math.isfinite(condition[k]) for k in METRICS):
                    raise ValueError('finite evaluable primary metrics required; unknown is not zero')
            if 'paired_differences' in row:
                for comparison in ('S-F', 'C-S', 'O-S', 'O-Q'):
                    a, b = comparison.split('-')
                    for metric in METRICS:
                        reference = shard['identity']['F_metrics'][metric] if b == 'F' else row['metrics'][b][metric]
                        if abs(row['paired_differences'][comparison][metric] - (row['metrics'][a][metric] - reference)) > 1e-12:
                            raise ValueError('saved paired difference differs from full observations')
            by_seed[seed][i] = row
    family = []
    for seed, rows in by_seed.items():
        if set(rows) != set(range(16)):
            raise ValueError('all sixteen global repeats for each model required')
        for comparison in COMPARISONS:
            a, b = comparison.split('-')
            for metric in METRICS:
                differences = np.array([rows[i]['metrics'][a][metric] - rows[i]['metrics'][b][metric] for i in range(16)])
                mean = float(differences.mean())
                se = float(differences.std(ddof=1) / 4)
                inferable = se > 0
                p = float(2 * t.sf(abs(mean / se), 15)) if inferable else 1.
                width = float(t.ppf(.975, 15) * se) if inferable else None
                family.append({'seed': seed, 'comparison': comparison, 'metric': metric,
                               'differences': differences.tolist(), 'mean': mean, 'mc_se': se,
                               'marginal_95_interval': [mean - width, mean + width] if inferable else None,
                               'two_sided_p': p, 'inferable': inferable,
                               'zero_observed_variance': not inferable,
                               'independent_graph_repetitions': 16, 'df': 15})
    ordered = sorted(range(18), key=lambda i: family[i]['two_sided_p'])
    previous = 0.
    for position, index in enumerate(ordered):
        previous = max(previous, min(1., (18 - position) * family[index]['two_sided_p']))
        family[index]['holm_adjusted_p'] = previous
    damage = [{'seed': seed, 'metric': metric,
               'differences': [rows[i]['metrics']['S'][metric] - references[seed][metric]
                               for i in range(16)]}
              for seed, rows in by_seed.items() for metric in METRICS]
    for row in damage:
        values = np.asarray(row['differences'])
        row.update(mean=float(values.mean()), mc_se=float(values.std(ddof=1) / 4), descriptive_only=True)
    return {'status': 'complete', 'protocol': PROTOCOL, 'config_sha256': CONFIG_SHA256,
            'family_size': 18, 'primary_comparisons': family,
            'sampling_damage_S_minus_F': damage,
            'scope': 'conditional paired graph-sampling pilot; fixed calibration uncertainty is not in these MC intervals; not independent test or training-seed inference'}


def load_shard(directory, config):
    """Verify every saved archive and recompute summaries from all points/ranks.

    This CPU merge does not rerun the exhaustive relation head. Independent
    fixture/head audits remain a separate supervisor gate.
    """
    import json
    import hashlib
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np
    import torch
    from .diagnostic_archive import read_archive
    from .encoder_matched_control import validate_ranking
    from .frozen_recovery import _metrics, promoted, tensor_identity
    from .vector_structure import RadialPanel
    root = Path(directory)
    row = json.loads((root / 'run.json').read_text(encoding='utf-8'))
    supervisor = json.loads((root / 'supervisor.json').read_text(encoding='utf-8'))
    if (row.get('status') != 'complete' or row.get('engineering_fixture_only') is not False
            or row.get('protocol') != PROTOCOL or row.get('config_sha256') != CONFIG_SHA256
            or supervisor.get('status') != 'complete' or supervisor.get('worker_exit_code') != 0
            or supervisor.get('deadline_seconds') != 1080 or row.get('completed_repetitions') != 8
            or row.get('weights_unchanged') is not True or row.get('input_unchanged') is not True
            or row.get('model_updates') != 0 or len(row.get('repeat_archives', [])) != 8
            or row['provenance'].get('release_verified') is not True):
        raise ValueError('complete supervised frozen scientific shard required')
    entry = read_archive(root, row['entry_archive'])
    seed = row['seed']
    n = config['prepared']['nodes_count']
    if len(entry['nodes']) != n or len(entry['valid']) != config['valid_query_count']:
        raise ValueError('original full node/valid inventory required')
    for name, key in (('base', 'p'), ('bias', 'b'), ('floor', 'floor')):
        meta = config['calibration'][str(seed)]['fields'][key]
        if tensor_identity(torch.from_numpy(entry[name])) != {k: meta[k] for k in ('shape', 'dtype', 'data_sha256')}:
            raise ValueError('archive no longer binds original calibrated arrays')
    data = {'nodes': entry['nodes'], 'valid': [tuple(map(int, p)) for p in entry['valid']]}
    if canonical(entry['nodes'].tolist() if isinstance(entry['nodes'], np.ndarray) else entry['nodes']) != config['prepared']['node_order_hash']:
        raise ValueError('original node order required')
    if canonical([[data['nodes'][a], data['nodes'][b]] for a, b in data['valid']]) != config['prepared']['valid_queries_hash']:
        raise ValueError('original valid identity required')
    validate_ranking(entry['F_ranking'], data)
    if canonical(entry['F_ranking']['rows']) != row['identity']['F_rank_hash']:
        raise ValueError('F rank identity mismatch')
    for key in ('base', 'bias', 'q'):
        if tensor_identity(torch.from_numpy(entry[key])) != row['identity'][key]:
            raise ValueError('fixed point/bias/direction identity mismatch')
    if ({key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in entry['weights'].items()}
            != row['identity']['weights']):
        raise ValueError('archived weight identity mismatch')
    view = SimpleNamespace(nodes=list(entry['nodes']), root=entry['structure_view']['root'],
                           reachable=set(entry['structure_view']['reachable']))
    base = torch.from_numpy(entry['base'])
    panel = RadialPanel(view, entry['panel'], base, {'V': list(range(n))}, config['model']['c'], torch.from_numpy(entry['floor']))
    if entry['native_F'].dtype != np.float32 or entry['native_F'].shape != base.shape:
        raise ValueError('complete original native FP32 F points required')
    expected_f = _metrics(panel.evaluate(promoted(torch.from_numpy(entry['native_F']), config['model']['c'])), entry['F_ranking'])
    if any(expected_f[k] != row['identity']['F_metrics'][k] for k in METRICS):
        raise ValueError('F primary summaries differ from full saved evidence')
    for observation, descriptor in zip(row['observations'], row['repeat_archives']):
        payload = read_archive(root, descriptor)
        if payload['repeat'] != observation['repeat'] or payload['plan_hash'] != observation['plan_hash'] or payload['rng'] != observation['rng']:
            raise ValueError('global repeat/plan identity mismatch')
        if set(payload['native_points']) != {'S', 'O', 'C', 'Q'}:
            raise ValueError('full paired point inventory required')
        for condition in ('S', 'O', 'C', 'Q'):
            points = payload['native_points'][condition]
            if points.dtype != np.float32 or points.shape != entry['native_F'].shape or not np.isfinite(points).all():
                raise ValueError('complete finite native points required')
            validate_ranking(payload['rankings'][condition], data)
            structure = panel.evaluate(promoted(torch.from_numpy(points), config['model']['c']))
            if canonical(structure) != canonical(payload['structure'][condition]):
                raise ValueError('saved structure differs from all archived native points')
            metrics = _metrics(structure, payload['rankings'][condition])
            if any(metrics[k] != observation['metrics'][condition][k] for k in metrics):
                raise ValueError('saved primary/auxiliary metrics differ from full archives')
    return row


def main():
    import argparse
    import json
    from pathlib import Path
    import torch
    from .encoder_matched_control import atomic_json
    from .recovery_registration import file_sha256, validate_config
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--config', type=Path, required=True)
    cli.add_argument('--shards', nargs=4, type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('new merge output required')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    validate_config(config)
    torch.set_num_threads(2)
    result = merge_shards([load_shard(path, config) for path in args.shards])
    result['shard_run_raw_sha256'] = [file_sha256(path / 'run.json') for path in args.shards]
    atomic_json(args.output, result)


if __name__ == '__main__':
    main()

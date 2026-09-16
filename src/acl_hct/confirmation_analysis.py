"""Independent NumPy/SciPy inference from archived E2 repeats; never loads a model."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import t
from . import confirmation_registration as reg


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain(v) for v in value]
    return value


def verify_panels(panels, binding):
    panels = plain(panels)
    if reg.digest(panels['panels']) != panels['hash'] or panels['hash'] != binding['combined_hash']:
        raise ValueError('panel contents/hash mismatch')
    for name in ('development', 'diagnostic_confirmation'):
        p = panels['panels'][name]
        base = {k: v for k, v in p.items() if k not in ('panel_hash', 'relations', 'relation_seed', 'relation_hash')}
        if (reg.digest(base) != p['panel_hash'] or reg.digest(p['relations']) != p['relation_hash']
                or any(p[k] != v for k, v in binding[name].items())):
            raise ValueError('sealed panel contents mismatch')
    if set(panels['panels']['development']['pool']) & set(panels['panels']['diagnostic_confirmation']['pool']):
        raise ValueError('overlapping panel pools')


def read_archive(directory, descriptor):
    manifest = reg.child_path(directory, descriptor['manifest'])
    if reg.file_sha256(manifest) != descriptor['manifest_sha256']:
        raise ValueError('manifest hash mismatch')
    record = json.loads(manifest.read_text(encoding='utf-8'))
    path = reg.child_path(directory, record['array_file'])
    if (record['schema_version'] != 1 or record['array_file'] != descriptor['array_file']
            or reg.file_sha256(path) != record['array_file_sha256']
            or record['array_file_sha256'] != descriptor['array_file_sha256']):
        raise ValueError('NPZ identity mismatch')
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != set(record['arrays']):
            raise ValueError('array inventory mismatch')
        arrays = {}
        for key, meta in record['arrays'].items():
            value = loaded[key]
            if (list(value.shape) != meta['shape'] or value.dtype.str != meta['dtype']
                    or hashlib.sha256(value.tobytes()).hexdigest() != meta['data_sha256']):
                raise ValueError('array identity mismatch')
            arrays[key] = value
    def decode(value):
        if isinstance(value, dict):
            if set(value) == {'npz_array'}:
                return arrays[value['npz_array']]
            return {k: decode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [decode(v) for v in value]
        return value
    return decode(record['payload'])


def estimate(values, target, reason=None):
    x = np.asarray([] if values is None else values, dtype=np.float64)
    result = dict(n=len(x), mean=None, sample_sd=None, se=None, marginal_ci95=None,
                  simultaneous_ci95=None, marginal_halfwidth=None, precision_target=target,
                  precision_target_met=None, p_raw=None, p_for_adjustment=1., inferable=False,
                  reason=reason)
    if reason or x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        result['reason'] = reason or 'missing_or_nonfinite_repeats'
        return result
    mean = float(x.mean()); sd = float(x.std(ddof=1)); se = sd / math.sqrt(len(x))
    result.update(mean=mean, sample_sd=sd, se=se)
    if se == 0:
        result['reason'] = 'zero_observed_variance'
        return result
    half = float(t.ppf(.975, len(x) - 1) * se)
    simultaneous = float(t.ppf(1 - .05 / (2 * 70), len(x) - 1) * se)
    p = float(2 * t.sf(abs(mean / se), len(x) - 1))
    result.update(marginal_ci95=[mean-half, mean+half], simultaneous_ci95=[mean-simultaneous, mean+simultaneous],
                  marginal_halfwidth=half, precision_target_met=half <= target,
                  p_raw=p, p_for_adjustment=p, inferable=True, reason=None)
    return result


def holm(rows):
    if len(rows) != 70 or len({r['id'] for r in rows}) != 70:
        raise ValueError('one fixed family of 70 distinct hypotheses required')
    previous = 0.
    for rank, row in enumerate(sorted(rows, key=lambda r: r['p_for_adjustment'])):
        p = row['p_for_adjustment']
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('invalid p value')
        previous = min(1., max(previous, (70-rank) * p))
        row.update(p_holm=previous, reject_holm_05=row['inferable'] and previous <= .05)
    return rows


def ranking_metrics(row):
    ids = np.asarray(row['row_ids']); ranks = np.asarray(row['ranks'], dtype=float)
    if (row['status'] != 'complete' or row['row_int_columns'] != ['parent', 'child', 'candidates']
            or ids.ndim != 2 or ids.shape != (len(ranks), 3) or ids.dtype.kind not in 'iu'
            or len(ranks) == 0 or not np.isfinite(ranks).all()
            or len(np.unique(ids[:, :2], axis=0)) != len(ranks)
            or np.any(ranks < 1) or np.any(ranks > ids[:, 2])
            or row['completed_queries'] != len(ranks) or row['expected_queries'] != len(ranks)):
        raise ValueError('invalid complete ranking')
    _, child_ids = np.unique(ids[:, 1], return_inverse=True)
    rr = 1 / ranks
    micro = float(rr.mean())
    macro = float((np.bincount(child_ids, weights=rr) / np.bincount(child_ids)).mean())
    if (abs(row['query_micro_mrr']-micro) > 1e-12 or abs(row['child_macro_mrr']-macro) > 1e-12):
        raise ValueError('reported ranking differs from raw ranks')
    return micro, macro


def check_numerics(checks, limits):
    mapping = {'projection': 'promotion_displacement', 'rerun_constraint_before': 'fp64_constraint',
               'rerun_constraint_after': 'fp64_constraint', 'native_vs_fp64_full_distance': 'native_fp64_distance',
               'self_log_norm': 'roundtrip', 'log_exp_roundtrip_distance': 'roundtrip',
               'log_tangent_constraint_residual': 'tangent_constraint', 'full_plan_identity': 'unchanged_output_difference',
               'paired_full_identity': 'unchanged_output_difference', 'inactive_native_difference': 'unchanged_output_difference'}
    if not checks:
        raise ValueError('missing numerical checks')
    for check in checks:
        name = check['name'].rsplit('/', 1)[-1]
        obs = check['observed']
        if check['status'] != 'passed' or obs['count'] < 1 or not all(math.isfinite(obs[k]) for k in ('minimum', 'mean', 'maximum')):
            raise ValueError('failed or nonfinite numerical check')
        if name in ('variance_minimum', 'mse_minimum'):
            if check['minimum'] != limits['variance_mse_minimum'] or obs['minimum'] < check['minimum']:
                raise ValueError('negative variance/MSE')
        else:
            limit = limits[mapping.get(name, name)]
            if check['limit_absolute'] != limit or max(abs(obs['minimum']), abs(obs['maximum'])) > limit:
                raise ValueError('numerical threshold violated')


def extract_repeats(entry, budget, spec):
    """Recompute primary samples; usable for small engineering fixtures as well."""
    n, nt = spec['repetitions'], spec['task_repetitions']
    observations = budget['per_repeat_structure_and_task']
    if ([r['repetition'] for r in observations] != list(range(n))
            or any(r['status'] != 'complete' for r in observations)
            or budget['condition_counts'] != {c: n for c in reg.CONDITIONS}
            or len(budget['plan_hashes']) != n or any(len(p) != 2 for p in budget['plan_hashes'])
            or ['task' in r for r in observations] != [i < nt for i in range(n)]):
        raise ValueError('incomplete or incorrectly placed registered repeats')
    values = {}; metadata = {}
    for category, conditions, column_key in (('geometry', reg.CONDITIONS, 'geometry_repeat_columns'),
                                            ('structure', ('S/S',), 'structure_repeat_columns')):
        for condition in conditions:
            columns = list(budget[column_key][condition])
            matrix = np.asarray([r[category][condition] for r in observations], dtype=float)
            if matrix.shape != (n, len(columns)) or len(set(columns)) != len(columns) or not np.isfinite(matrix).all():
                raise ValueError('invalid per-repeat columns')
            requested = [('geometry/'+condition, 'V/radial_projection')] if category == 'geometry' else [
                ('structure/'+kind, kind+'/V/score_change') for kind in ('direct', 'distant')]
            for key, column in requested:
                values[key] = matrix[:, columns.index(column)] if column in columns else None
    v = np.asarray(budget['groups']['V'], dtype=int)
    if sorted(v.tolist()) != list(range(len(entry['nodes']))):
        raise ValueError('primary geometry requires fixed full V')
    for condition in reg.CONDITIONS:
        stats = budget['geometry'][condition]
        defined = np.asarray(stats['direction_defined'])
        floor = np.asarray(stats['empirical_numerical_floor'])
        if defined.shape != (len(v),) or defined.dtype != np.bool_ or floor.shape != (len(v),) or not np.isfinite(floor).all() or np.any(floor < 0):
            raise ValueError('invalid direction/floor evidence')
        count = int(defined.sum())
        if count != stats['groups']['V']['projection_nodes'] or (values['geometry/'+condition] is None) != (count == 0):
            raise ValueError('projection coverage mismatch')
        metadata['geometry/'+condition] = dict(V=len(v), defined=count, NA=len(v)-count,
            NA_fraction=(len(v)-count)/len(v), mean_empirical_floor=float(floor[defined].mean()) if count else None)
    for kind in ('direct', 'distant'):
        metadata['structure/'+kind] = budget['full_structure']['S/S'][kind]['groups']['V']
        if (values['structure/'+kind] is None) != (metadata['structure/'+kind]['covered_children'] == 0):
            raise ValueError('structure coverage/NA mismatch')
    full_micro, full_macro = ranking_metrics(entry['full_valid'])
    differences = []; macro_differences = []
    for row in observations[:nt]:
        if not np.array_equal(row['task']['row_ids'], entry['full_valid']['row_ids']):
            raise ValueError('task query/candidate universe differs from full reference')
        micro, macro = ranking_metrics(row['task'])
        differences.append(micro-full_micro); macro_differences.append(macro-full_macro)
    values['task/micro_mrr_change'] = differences
    metadata['task/micro_mrr_change'] = dict(full_micro_mrr=full_micro, full_macro_mrr=full_macro,
        queries=len(entry['full_valid']['ranks']), secondary_macro_changes=macro_differences)
    return values, metadata


def validate_summary(summary, config, spec, source_commit, fixture_hash, fixture_sources):
    expected = {'protocol': reg.PROTOCOL, 'confirmation_evaluated': True,
                'evaluated_panel': 'diagnostic_confirmation', 'registration_sha256': reg.REGISTRATION_SHA256,
                'config_sha256': reg.REGISTRATION_SHA256, 'checkpoint_seed': spec['seed'],
                'checkpoint_sha256': config['checkpoints'][str(spec['seed'])]['sha256'],
                'repetitions': spec['repetitions'], 'task_repetitions': spec['task_repetitions'],
                'fanouts': [spec['fanout']], 'panel_hash': config['panels']['combined_hash'],
                'cuda_fixture_artifact_sha256': fixture_hash,
                'source_sha256_normalized_lf': fixture_sources,
                'execution_internal_seconds': spec['internal_seconds'],
                'development_audit': {'status': 'passed', 'index_raw_sha256': config['development']['index_raw_sha256'],
                                      'summaries_verified': 10}}
    if any(summary.get(k) != v for k, v in expected.items()) or summary.get('source', {}).get('source_commit') != source_commit:
        raise ValueError('confirmation provenance mismatch (development results are not confirmation)')
    reg.validate_config(summary['config'])
    if summary.get('checkpoint_and_weights_unchanged') is not True:
        raise ValueError('frozen model verification missing')


def analyze(config, index, directory, source_commit, fixture_hash, fixture_sources):
    reg.validate_config(config)
    expected_keys = {(s['seed'], s['fanout']) for s in config['shards']}
    supplied = {(s['seed'], s['fanout']): s for s in index['shards']}
    if len(supplied) != len(index['shards']) or not set(supplied) <= expected_keys:
        raise ValueError('duplicate or unregistered result shards')
    rows = []; audits = []
    metrics = [(f'geometry/{c}', 'geometry') for c in reg.CONDITIONS] + [
        ('structure/direct', 'structure'), ('structure/distant', 'structure'), ('task/micro_mrr_change', 'task')]
    for spec in config['shards']:
        seed, fanout = spec['seed'], spec['fanout']; reference = supplied.get((seed, fanout))
        values = {}; metadata = {}; reason = 'missing_shard'
        if reference is not None:
            path = reg.child_path(directory, reference['summary_file'])
            if reg.file_sha256(path) != reference['summary_sha256']:
                raise ValueError('summary hash mismatch')
            summary = json.loads(path.read_text(encoding='utf-8'))
            # A failed shard still occupies its seven planned family positions.
            reason = 'incomplete_or_failed_shard'
            if summary.get('status') == 'complete':
                validate_summary(summary, config, spec, source_commit, fixture_hash, fixture_sources)
                root = Path(directory) / reference['artifact_directory']
                entry = read_archive(root, summary['entry_artifact'])
                b = summary['budgets'][str(fanout)]; budget = read_archive(root, b['artifact'])
                verify_panels(entry['panels'], config['panels'])
                namespace = f'{reg.PROTOCOL}/seed{seed}/fanout{fanout}'
                seeds = [int.from_bytes(hashlib.sha256(f'{reg.BASE_SEED}/{namespace}/layer{i}'.encode()).digest()[:8], 'big') % (2**63) for i in (1, 2)]
                if (budget['seed_namespace'] != b['seed_namespace'] or budget['seed_namespace']['seed'] != reg.BASE_SEED
                        or budget['seed_namespace']['namespace'] != namespace or budget['seed_namespace']['layer_seeds'] != seeds
                        or reg.digest(budget['plan_hashes']) != b['plan_hashes_sha256']):
                    raise ValueError('independent plan provenance mismatch')
                try:
                    if (summary['entry_status'] != 'passed' or summary['numerical_entry']['status'] != 'passed'
                            or b['status'] != 'complete' or budget['status'] != 'complete'
                            or not b['scientifically_usable'] or not budget['scientifically_usable']):
                        raise ValueError('invalid numerical shard')
                    check_numerics(summary['numerical_entry']['checks'], config['numerical_limits'])
                    if len(summary['numerical_entry']['checks']) != 24 or set(b['quality']) != set(reg.CONDITIONS):
                        raise ValueError('incomplete numerical gate inventory')
                    for checks in b['quality'].values():
                        if {r['name'] for r in checks} != {'mse_decomposition', 'variance_minimum', 'mse_minimum'}:
                            raise ValueError('missing moment checks')
                        check_numerics(checks, config['numerical_limits'])
                    numerical_audits = budget['numerical_audits']
                    if {(r['repetition'], r['condition']) for r in numerical_audits} != {(i, c) for i in range(spec['repetitions']) for c in reg.CONDITIONS} or len(numerical_audits) != 4 * spec['repetitions']:
                        raise ValueError('missing numerical audits')
                    for audit in numerical_audits:
                        required = {'sample/native_constraint', 'sample/projection', 'sample/fp64_constraint', 'tangent_constraint', 'roundtrip'}
                        if audit['condition'] in ('local_L1', 'F/S') and len(budget['groups']['A']) < len(entry['nodes']):
                            required.add('inactive_native_difference')
                        if {r['name'] for r in audit['checks']} != required:
                            raise ValueError('missing per-repeat checks')
                        check_numerics(audit['checks'], config['numerical_limits'])
                    reproduction = summary['entry_full_valid_reproduction']
                    if (reproduction['status'] != 'passed' or not reproduction['query_coverage_identical']
                            or reproduction['current_queries'] != reproduction['historical_queries']
                            or reproduction['current_query_pairs_hash'] != reproduction['historical_query_pairs_hash']
                            or reproduction['absolute_tolerance'] != config['numerical_limits']['full_valid_reproduction']
                            or any(not math.isfinite(v) or abs(v) > reproduction['absolute_tolerance'] for v in reproduction['metric_differences'].values())):
                        raise ValueError('full valid reproduction failed')
                    values, metadata = extract_repeats(entry, budget, spec)
                    if (b['completed_graph_repetitions'] != spec['repetitions']
                            or b['completed_task_repetitions'] != spec['task_repetitions']
                            or b['completed_method_repetitions'] != budget['condition_counts']
                            or len(entry['full_valid']['ranks']) != reproduction['current_queries']
                            or reg.digest(sorted(map(tuple, entry['full_valid']['row_ids'][:, :2].tolist()))) != reproduction['current_query_pairs_hash']):
                        raise ValueError('summary and archived repetition/query counts differ')
                    reason = None
                except ValueError as error:
                    values = {}; metadata = {}; reason = 'invalid_for_inference: ' + str(error)
        audits.append({'seed': seed, 'fanout': fanout, 'status': 'passed' if reason is None else reason})
        for metric, family in metrics:
            row = {'id': f'seed{seed}/f{fanout}/{metric}', 'seed': seed, 'fanout': fanout, 'metric': metric,
                   'family': 'all_70_primary', 'metric_category': family,
                   **estimate(values.get(metric), config['inference']['halfwidth_targets'][family], reason),
                   'coverage': metadata.get(metric)}
            if metric.startswith('geometry/') and row['mean'] is not None:
                row['absolute_mean_above_empirical_floor'] = abs(row['mean']) > metadata[metric]['mean_empirical_floor']
            rows.append(row)
    return {'status': 'complete' if all(a['status'] == 'passed' for a in audits) else 'incomplete_evidence',
            'scope': config['scope'], 'registration_sha256': reg.REGISTRATION_SHA256,
            'family_size': 70, 'alpha': .05, 'correction': 'Holm two-sided, one family',
            'interval_caveat': 'Student-t MC approximation, not a finite-distribution guarantee',
            'numerical_caveat': config['inference']['numerical_resolution'],
            'source_commit': source_commit, 'cuda_fixture_sha256': fixture_hash,
            'shard_audits': audits, 'primary_hypotheses': holm(rows),
            'software': {'numpy': np.__version__, 'scipy': scipy.__version__},
            'analysis_source_sha256_normalized_lf': hashlib.sha256(Path(__file__).read_text(encoding='utf-8').encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'results-index', 'cuda-fixture-record', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('no overwrite of independent analysis')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    fixture = json.loads(args.cuda_fixture_record.read_text(encoding='utf-8'))
    if (fixture.get('confirmation_fixture_passed') is not True or fixture.get('status') != 'complete'
            or fixture.get('registration_sha256') != reg.REGISTRATION_SHA256
            or fixture.get('source', {}).get('source_commit') != args.source_commit):
        raise ValueError('reviewed confirmation CUDA fixture required')
    result = analyze(config, json.loads(args.results_index.read_text(encoding='utf-8')), args.results_index.parent,
                     args.source_commit, reg.file_sha256(args.cuda_fixture_record), fixture['source_sha256_normalized_lf'])
    result['results_index_raw_sha256'] = reg.file_sha256(args.results_index)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, allow_nan=False, indent=2)
    if result['status'] != 'complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()

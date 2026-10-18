"""Portable publication wrapper; empirical plot calculations are unchanged."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import time

ARTIFACT = Path(__file__).resolve().parents[1]
SOURCE = 'ece3665a84b76ac7801639561f0d07ab98af02d94c55f28aa8bc1dca82bb5e75'
SEEDS = (11, 23, 37)
REPEATS = (0, 1, 2, 3)
SAMPLERS = ('uniform', 'labor')
TIERS = ('low', 'middle', 'high')
BRANCHES = ('local_restored', 'upstream_restored', 'both_restored')
GROUPS = (('ogbn-arxiv', 'sage'), ('ogbn-arxiv', 'gat'), ('PubMed', 'sage'),
          ('ogbn-arxiv', 'hgcn'), ('ogbn-arxiv', 'hybonet'))
NAMES = {'sage': 'GraphSAGE', 'gat': 'GAT', 'hgcn': 'HGCN', 'hybonet': 'HyboNet'}
COLORS = {'uniform': '#2367A3', 'labor': '#D47520', 'full': '#C73534', 'ram': '#287D68'}
MARKERS = {11: 'o', 23: '^', 37: 's'}
ROLES = {
    'scalar_summary': ('summary.json', 'json'),
    'scalar_per_seed': ('per-seed.csv', 'csv'),
    'scalar_across_seeds': ('across-seeds.csv', 'csv'),
    'frozen_budgets': ('frozen-budgets.csv', 'csv'),
    'hardware_context': ('hardware-and-usage-context.json', 'json'),
    'mechanism_summary': ('summary.json', 'json'),
    'mechanism_per_seed': ('per-seed.csv', 'csv'),
    'mechanism_across_seeds': ('across-seeds.csv', 'csv'),
    'mechanism_per_repeat': ('per-repeat.csv', 'csv'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    number = float(value)
    require(math.isfinite(number), 'Missing or nonfinite scalar; no imputation')
    return number


def reject_constant(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


def exact_row(rows, filters):
    matches = [r for r in rows if all(str(r.get(k)) == str(v) for k, v in filters.items())]
    require(len(matches) == 1, 'Missing or ambiguous accepted scalar: ' + str(filters))
    row = matches[0]
    require(row['status'] == 'complete', 'Pending scalar cannot be plotted: ' + str(filters))
    return row


def value(rows, filters, unit, field='value'):
    row = exact_row(rows, filters)
    require(row['unit'] == unit, 'Scalar unit differs: ' + str(filters))
    return finite(row[field])


def close(a, b, context):
    require(math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-10),
            'Accepted scalar/formula mismatch: ' + context)


def load_inputs(manifest_path, output_dir):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8'), parse_constant=reject_constant)
    require(manifest.get('schema') == 'public_vldb_figure_inputs_v1', 'Unexpected public plot manifest')
    require(set(manifest['inputs']) == set(ROLES), 'All nine scalar roles are required')
    output = Path(output_dir).resolve()
    require(not output.exists(), 'Use a new output directory to retain existing figures')
    documents, provenance = {}, []
    for role, (_, kind) in ROLES.items():
        binding = manifest['inputs'][role]
        file = (ARTIFACT / binding['path']).resolve()
        require(file.is_relative_to(ARTIFACT), 'Input must stay within the artifact')
        raw = file.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == binding['sha256'], 'Changed scalar input: ' + role)
        text = raw.decode('utf-8-sig')
        documents[role] = list(csv.DictReader(io.StringIO(text))) if kind == 'csv' else json.loads(text, parse_constant=reject_constant)
        provenance.append({'role': role, 'path': binding['path'], 'sha256': binding['sha256']})
    require(documents['scalar_summary']['source_manifest_sha256'] == SOURCE, 'Different scientific source')
    require(all(documents[k]['status'] == 'complete' for k in ('scalar_summary','mechanism_summary')), 'Incomplete input')
    return manifest, documents, provenance, output


def group_title(group):
    dataset, model = group
    family = 'ordinary' if model in ('sage', 'gat') else 'hyperbolic'
    return NAMES[model] + ' / ' + dataset + '\n' + family + ' model'


def select(rows, group):
    return [r for r in rows if (r['dataset'], r['model']) == group]


def repeat_values(rows, filt, unit):
    matches = [r for r in rows if all(str(r.get(k)) == str(v) for k, v in filt.items())]
    require(len(matches) == 4 and {int(r['repeat']) for r in matches} == set(REPEATS),
            'Exactly four existing repeat scalars required: ' + str(filt))
    return [value(rows, dict(filt, repeat=r), unit) for r in REPEATS]


def contrast_interval(rows, filt, unit, center, mode):
    """Range of existing paired differences, or explicitly unpaired bounds.

    No node reduction or contrast recomputation is performed. The fallback
    uses min(L)-max(U), max(L)-min(U), not a paired SD, CI or observed range.
    """
    if mode == 'paired_repeat_range':
        vals = repeat_values(rows, dict(filt, sampler='labor_minus_uniform'), unit)
        close(sum(vals) / 4, center, 'paired-repeat center')
        lo, hi = min(vals), max(vals)
        evidence = {'paired_repeat_contrasts': vals}
    else:
        # This mode is selected in the reviewed manifest, never silently used
        # to fill a partial/missing paired export during execution.
        u = repeat_values(rows, dict(filt, sampler='uniform'), unit)
        l = repeat_values(rows, dict(filt, sampler='labor'), unit)
        close(sum(l) / 4 - sum(u) / 4, center, 'method-range center')
        lo, hi = min(l) - max(u), max(l) - min(u)
        evidence = {'uniform_repeat_scalars': u, 'labor_repeat_scalars': l}
    require(lo - 1e-10 <= center <= hi + 1e-10, 'Range does not contain existing mean')
    return dict(low=lo, high=hi, mode=mode, evidence=evidence)


def prepare_points(manifest, docs):
    scalar = docs['scalar_per_seed']; mechanism = docs['mechanism_per_seed']
    across = docs['mechanism_across_seeds']; budgets = docs['frozen_budgets']
    points = {'mechanism': [], 'gain_annotations': [], 'native_pair': [],
              'deployment': [], 'full': [], 'cost_coefficients': [], 'cost_crossings': []}
    for dataset, model in GROUPS:
        base = {'dataset': dataset, 'model': model}
        for sampler in SAMPLERS:
            branch_means = {}
            for branch in BRANCHES:
                filt = dict(base, sampler=sampler, section='paired_corner',
                            contrast=branch + '_minus_sampled', metric='accuracy_delta')
                mean = 100 * value(across, filt, 'fraction', 'mean')
                require(exact_row(across, filt)['n_seeds'] == '3', 'Three-seed mean required')
                branch_means[branch] = mean
                seed_values = []
                for seed in SEEDS:
                    delta = 100 * value(mechanism, dict(filt, seed=seed), 'fraction')
                    seed_values.append(delta)
                    points['mechanism'].append(dict(base, sampler=sampler, seed=seed, branch=branch, accuracy_delta_pp=delta))
                close(sum(seed_values) / 3, mean, 'published three-seed restoration mean')
            filt = dict(base, sampler=sampler, section='interaction',
                        contrast='both-local-upstream+sampled', metric='accuracy_interaction')
            points['gain_annotations'].append(dict(base, sampler=sampler, branch_means_pp=branch_means,
                maximum_branch_mean_pp=max(branch_means.values()),
                maximum_scope='max of local/upstream/joint three-seed means; not best seed',
                interaction_pp=100 * value(across, filt, 'fraction', 'mean')))
        for seed in SEEDS:
            pair = dict(base, seed=seed, sampler='labor_minus_uniform')
            error_filter = dict(base, seed=seed, section='error_mean',
                                contrast='local_estimator_sq_l2_sampled_inputs', metric='mean')
            quality_filter = dict(base, seed=seed, section='corner', contrast='sampled', metric='accuracy')
            error = value(mechanism, dict(error_filter, sampler='labor_minus_uniform'), 'native_squared_l2')
            quality = value(mechanism, dict(quality_filter, sampler='labor_minus_uniform'), 'fraction')
            erange = contrast_interval(docs['mechanism_per_repeat'], error_filter, 'native_squared_l2', error,
                                      manifest['figure2_interval_mode'])
            qrange = contrast_interval(docs['mechanism_per_repeat'], quality_filter, 'fraction', quality,
                                      manifest['figure2_interval_mode'])
            margin = value(mechanism, dict(pair, section='corner', contrast='sampled', metric='mean_true_class_margin'), 'native_score')
            points['native_pair'].append(dict(base, seed=seed,
                local_error_delta_native_squared_l2=error, sampled_accuracy_delta_pp=100 * quality,
                sampled_native_margin_delta=margin, error_range=erange,
                accuracy_range_pp=dict(qrange, low=100 * qrange['low'], high=100 * qrange['high'])))
            for sampler in SAMPLERS:
                for tier in TIERS:
                    filt = dict(base, seed=seed, section='deployment', sampler=sampler, tier=tier, branch='', q='')
                    budget = exact_row(budgets, dict(base, tier=tier))
                    require(budget['cost_match_feasible'] in ('True', 'False', 'true', 'false'), 'Explicit feasibility flag required')
                    points['deployment'].append(dict(base, seed=seed, sampler=sampler, tier=tier,
                        fanout=int(budget['uniform_fanout' if sampler == 'uniform' else 'labor_fanout']),
                        lock_feasible=sampler == 'uniform' or budget['cost_match_feasible'].lower() == 'true',
                        accuracy_pct=100 * value(scalar, dict(filt, metric='accuracy'), 'fraction'),
                        workload_seconds=value(scalar, dict(filt, metric='total_workload_e2e_seconds'), 'seconds'),
                        peak_allocated_GiB=value(scalar, dict(filt, metric='peak_allocated_bytes'), 'bytes', 'maximum_observed') / 2**30,
                        peak_reserved_GiB=value(scalar, dict(filt, metric='peak_reserved_bytes'), 'bytes', 'maximum_observed') / 2**30))
            filt = dict(base, seed=seed, sampler='', tier='', branch='', q='')
            points['full'].append(dict(base, seed=seed,
                accuracy_pct=100 * value(scalar, dict(filt, section='full_quality', metric='accuracy'), 'fraction'),
                workload_seconds=value(scalar, dict(filt, section='amortization', q=1, metric='full_recompute_seconds'), 'seconds'),
                peak_allocated_GiB=value(scalar, dict(filt, section='full_RAM', metric='peak_allocated_bytes'), 'bytes') / 2**30))

            def cost(metric):
                return value(scalar, dict(filt, section='full_RAM', metric=metric), 'seconds')
            setup = cost('format_setup_seconds')
            full = cost('full_e2e_median_seconds')
            build = cost('cache_build_median_seconds')
            gather = cost('target_gather_median_seconds')
            ram = cost('RAM_workload_median_seconds')
            require(min(setup, build, gather) >= 0 and min(full, ram) > 0, 'Invalid time coefficient')
            coeff = {'full': {'intercept_seconds': setup, 'slope_seconds_per_workload': full},
                     'ram': {'intercept_seconds': setup + build, 'slope_seconds_per_workload': ram}}
            # full_e2e is the recorded median of complete forward/D2H/gather,
            # not a sum of separately computed medians. Do not replace it.
            for sampler in SAMPLERS:
                selected = exact_row(scalar,
                    dict(base, seed=seed, section='deployment', sampler=sampler, tier='middle',
                         branch='', q='', metric='total_workload_e2e_seconds'))
                require(selected['unit'] == 'seconds', 'Sampled complete-workload seconds required')
                require(finite(selected['value']) > 0, 'Positive sampled workload cost required')
                coeff[sampler] = {'intercept_seconds': 0.0,
                                  'slope_seconds_per_workload': finite(selected['value'])}
            original_q = []
            for q in (1, 10, 100):
                original = dict(q=q)
                for route, metric in (('full', 'full_recompute_seconds'), ('ram', 'paid_full_RAM_seconds')):
                    recorded = value(scalar, dict(filt, section='amortization', q=q, metric=metric), 'seconds')
                    close(line_cost(coeff[route], q), recorded, 'registered q=' + str(q) + '/' + route)
                    original[metric] = recorded
                original_q.append(original)
            cache_bytes = value(scalar, dict(filt, section='amortization', q=1, metric='RAM_cache_bytes'), 'bytes')
            points['cost_coefficients'].append(dict(base, seed=seed, routes=coeff,
                full_timing_components=dict(format_setup_seconds=setup, full_e2e_median_seconds=full,
                    cache_build_median_seconds=build, target_gather_median_seconds=gather,
                    RAM_workload_median_seconds=ram), RAM_cache_bytes=cache_bytes, original_q=original_q,
                common_graph_checkpoint_setup_excluded=True,
                sampled_scope='q times the accepted four-repeat mean middle-tier complete-workload time; no unrecorded setup imputed',
                exact_RAM_scope='original median-based registered coefficients; analytic extension, not new measurements'))
        coeff_rows = select(points['cost_coefficients'], (dataset, model))
        means = {route: {field: sum(r['routes'][route][field] for r in coeff_rows) / 3
                        for field in ('intercept_seconds', 'slope_seconds_per_workload')}
                 for route in ('uniform', 'labor', 'full', 'ram')}
        # Means are algebraic averages of three already-published seed costs.
        # Crossings are explicitly intersections of cost formulas, not data
        # points, a fitted frontier, or a claim of accuracy equivalence.
        for other in ('uniform', 'labor', 'full'):
            crossing = analytic_crossing(means['ram'], means[other])
            if crossing is not None:
                points['cost_crossings'].append(dict(base, comparison='ram_vs_' + other,
                    seed_scope='three-seed mean formula', **crossing))
    return points


def line_cost(coefficient, q):
    return coefficient['intercept_seconds'] + q * coefficient['slope_seconds_per_workload']


def analytic_crossing(ram, other):
    denominator = other['slope_seconds_per_workload'] - ram['slope_seconds_per_workload']
    if math.isclose(denominator, 0.0, rel_tol=0, abs_tol=1e-14):
        return None
    q = (ram['intercept_seconds'] - other['intercept_seconds']) / denominator
    if not math.isfinite(q) or not 1 <= q <= 100:
        return None
    cost = line_cost(ram, q)
    require(cost > 0, 'Analytic crossover must have positive cost')
    close(cost, line_cost(other, q), 'analytic crossover')
    return {'q_star': q, 'cost_seconds': cost,
            'ram_cheaper_condition': 'q > q_star' if denominator > 0 else 'q < q_star',
            'scope': 'continuous formula intersection; integer q and unchanged graph/features/weights required'}


def save_figure(fig, output, name):
    fig.savefig(output / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(output / (name + '.png'), dpi=200, bbox_inches='tight')


def render(manifest, points, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.titlesize': 10})
    sampler_handles = [Line2D([], [], color=COLORS[s], marker='o', linestyle='None', label=s.title()) for s in SAMPLERS]
    seed_handles = [Line2D([], [], color='#555555', marker=MARKERS[s], linestyle='None', label='Seed ' + str(s)) for s in SEEDS]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.0), constrained_layout=True)
    for ax, group in zip(axes.flat, GROUPS):
        rows = select(points['mechanism'], group)
        for p in rows:
            x = BRANCHES.index(p['branch']) + (-0.16 if p['sampler'] == 'uniform' else 0.16) + (SEEDS.index(p['seed']) - 1) * 0.045
            ax.scatter(x, p['accuracy_delta_pp'], color=COLORS[p['sampler']], marker=MARKERS[p['seed']], s=32, zorder=3)
        notes = select(points['gain_annotations'], group)
        text = 'Max mean gain / interaction I_Q (pp)\n' + '\n'.join(
            ('U' if p['sampler'] == 'uniform' else 'L') + ': ' +
            f"{p['maximum_branch_mean_pp']:+.3f} / {p['interaction_pp']:+.3f}" for p in notes)
        ys = [0] + [r['accuracy_delta_pp'] for r in rows]
        span = max(max(ys) - min(ys), 0.03)
        ax.set_ylim(min(ys) - .15 * span, max(ys) + .55 * span)
        ax.text(.98, .98, text, transform=ax.transAxes, ha='right', va='top', fontsize=7.7,
                bbox=dict(facecolor='#F1F1F1', edgecolor='none', boxstyle='round,pad=.35'), zorder=4)
        ax.axhline(0, color='#666666', linestyle='--', linewidth=.8, zorder=1)
        ax.set_title(group_title(group)); ax.set_xticks(range(3), ('Local', 'Upstream', 'Joint'))
        ax.set_ylabel('Accuracy change vs sampled (pp)')
    axes.flat[-1].axis('off'); axes.flat[-1].legend(handles=sampler_handles + seed_handles, loc='upper left', frameon=False)
    axes.flat[-1].text(0, .39, 'Dashed zero = sampled baseline.\nMax = largest of three branch means,\ncomputed across all three seeds.\nI_Q is an additive accuracy contrast,\nnot an independent error share.\nPanel scales can differ.', fontsize=8.5, va='top')
    fig.suptitle('Local and upstream restoration at the frozen middle budget\nWhole-graph diagnostic prefixes; paired native computations', fontsize=11)
    save_figure(fig, output, 'final_mechanism_restoration'); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(10, 5.05), constrained_layout=True)
    interval_label = ('Range of four paired repeats' if manifest['figure2_interval_mode'] == 'paired_repeat_range'
                      else 'Bounds from separate method ranges')
    for ax, group in zip(axes.flat, GROUPS):
        for p in select(points['native_pair'], group):
            x, y = p['local_error_delta_native_squared_l2'], p['sampled_accuracy_delta_pp']
            er, qr = p['error_range'], p['accuracy_range_pp']
            ax.errorbar(x, y, xerr=[[max(0, x - er['low'])], [max(0, er['high'] - x)]],
                        yerr=[[max(0, y - qr['low'])], [max(0, qr['high'] - y)]],
                        marker=MARKERS[p['seed']], color='#6F5290', ecolor='#B3A5C6',
                        markersize=5, capsize=2.5, elinewidth=.8, linestyle='none', zorder=3)
        ax.axhline(0, color='#888888', linestyle='--', linewidth=.7)
        ax.axvline(0, color='#888888', linestyle='--', linewidth=.7)
        ax.set_title(group_title(group)); ax.set_xlabel('LABOR - Uniform local error\n(native squared L2; sampled inputs)')
        ax.set_ylabel('LABOR - Uniform accuracy (pp)'); ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
        ax.margins(x=.15, y=.15)
    axes.flat[-1].axis('off'); axes.flat[-1].legend(handles=seed_handles, loc='upper left', frameon=False)
    axes.flat[-1].text(0, .50, 'Points: per-seed means.\n' + interval_label + ' (not CIs).\nNative scales differ across models.\nCompare within each panel only.\nEach sampler keeps its own prefix.', fontsize=8.2, va='top')
    fig.suptitle('Local error and task decisions, with observed repeat variability\nFrozen middle budget only; all three seeds retained', fontsize=11)
    save_figure(fig, output, 'final_local_error_decision'); plt.close(fig)

    fig, axes = plt.subplots(5, 2, figsize=(10.0, 10.2), constrained_layout=True)
    for row_axes, group in zip(axes, GROUPS):
        sampled = select(points['deployment'], group); exact = select(points['full'], group)
        for sampler in SAMPLERS:
            for seed in SEEDS:
                seq = [next(p for p in sampled if p['sampler'] == sampler and p['seed'] == seed and p['tier'] == tier) for tier in TIERS]
                for ax, xfield in zip(row_axes, ('workload_seconds', 'peak_allocated_GiB')):
                    ax.plot([p[xfield] for p in seq], [p['accuracy_pct'] for p in seq],
                            color=COLORS[sampler], linestyle='--', linewidth=.8, alpha=.38, zorder=1)
                    for p in seq:
                        ax.scatter(p[xfield], p['accuracy_pct'], color=COLORS[sampler], marker=MARKERS[seed], s=26, zorder=3)
            # Label each budget once per method; retain all original seed points and links.
            # Means below locate text only and never add a data point or frontier.
            for tier in TIERS:
                tier_rows = [p for p in sampled if p['sampler'] == sampler and p['tier'] == tier]
                if group == ('PubMed', 'sage') and sampler == 'labor' and tier != 'middle':
                    continue
                label = {'low': 'L', 'middle': 'M', 'high': 'H'}[tier]
                if group == ('PubMed', 'sage') and sampler == 'labor':
                    label = 'L/M/H: K6'
                label += '!' if any(not p['lock_feasible'] for p in tier_rows) else ''
                for ax, xfield in zip(row_axes, ('workload_seconds', 'peak_allocated_GiB')):
                    tx = sum(p[xfield] for p in tier_rows) / len(tier_rows)
                    ty = sum(p['accuracy_pct'] for p in tier_rows) / len(tier_rows)
                    ax.annotate(label, (tx, ty), xytext=({'low': -5, 'middle': -6, 'high': 7}[tier], 10 if sampler == 'uniform' else -13),
                                textcoords='offset points', fontsize=7.8, fontweight='bold', color=COLORS[sampler], ha='center', va='center')
        for rank, p in enumerate(sorted(exact, key=lambda p: p['accuracy_pct'])):
            for column, (ax, xfield) in enumerate(zip(row_axes, ('workload_seconds', 'peak_allocated_GiB'))):
                ax.scatter(p[xfield], p['accuracy_pct'], color=COLORS['full'], marker='*', s=66, zorder=4)
                ax.annotate(str(p['seed']), (p[xfield], p['accuracy_pct']), xytext=(p[xfield] + .035 * (max(t[xfield] for t in sampled + exact) - min(t[xfield] for t in sampled + exact)), .66 + .12 * rank),
                            textcoords=('data', 'axes fraction'), fontsize=7.0, color=COLORS['full'], ha='center', va='center',
                            arrowprops=dict(arrowstyle='-', color=COLORS['full'], linewidth=.45, alpha=.65))
        fastest = all(p['workload_seconds'] < min(s['workload_seconds'] for s in sampled if s['seed'] == p['seed']) for p in exact)
        row_axes[0].set_title(group_title(group)); row_axes[1].set_title('Same targets / allocated GPU memory')
        row_axes[0].text(.98, .10, 'Exact: fastest measured' if fastest else 'Exact: measured reference',
                         transform=row_axes[0].transAxes, ha='right', va='bottom', fontsize=7.2,
                         bbox=dict(facecolor='white', edgecolor='none', alpha=.8))
        row_axes[0].set_xlabel('Complete target-workload time (s)'); row_axes[1].set_xlabel('Max observed allocated GPU memory (GiB)')
        for ax in row_axes:
            ax.set_ylabel('Accuracy (%)'); ax.grid(alpha=.14); ax.tick_params(labelsize=8); ax.margins(x=.10, y=.25)
    fig.legend(handles=sampler_handles + seed_handles + [Line2D([], [], color=COLORS['full'], marker='*', linestyle='none', label='Exact (seed labels)')],
               loc='outside lower center', ncol=6, frameon=False, fontsize=8.5)
    fig.suptitle('Quality, complete-workload time and memory\nLight L-M-H paths connect each seed; budget labels shown once per method\n! = infeasible calibration match; exact stars trade faster time for memory', fontsize=10.5)
    save_figure(fig, output, 'final_quality_workload_memory'); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.0), constrained_layout=True)
    qgrid = [math.exp(math.log(100) * i / 400) for i in range(401)]
    qgrid[0], qgrid[-1] = 1.0, 100.0
    for ax, group in zip(axes.flat, GROUPS):
        coeff_rows = select(points['cost_coefficients'], group)
        for route in ('uniform', 'labor', 'full', 'ram'):
            coefficients = [p['routes'][route] for p in coeff_rows]
            for p, c in zip(coeff_rows, coefficients):
                ys = [line_cost(c, q) for q in qgrid]
                require(min(ys) > 0, 'Nonpositive cost cannot be drawn on log axis')
                ax.plot(qgrid, ys, color=COLORS[route], linewidth=.7, alpha=.30, zorder=1)
                ax.scatter((1, 10, 100), [line_cost(c, q) for q in (1, 10, 100)],
                           color=COLORS[route], marker=MARKERS[p['seed']], s=12, alpha=.65, zorder=3)
            mean = {field: sum(c[field] for c in coefficients) / 3
                    for field in ('intercept_seconds', 'slope_seconds_per_workload')}
            ax.plot(qgrid, [line_cost(mean, q) for q in qgrid], color=COLORS[route], linewidth=1.9, zorder=2)
        for i, crossing in enumerate(select(points['cost_crossings'], group)):
            ax.scatter(crossing['q_star'], crossing['cost_seconds'], marker='x', color='#222222', s=25, zorder=5)
            ax.annotate('q*=' + f"{crossing['q_star']:.2f}", (crossing['q_star'], crossing['cost_seconds']),
                        xytext=(4, 8 + 9 * i), textcoords='offset points', fontsize=6.5)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks((1, 10, 100), ('1', '10', '100'))
        ax.set_xlabel('Identical complete-workload repetitions q'); ax.set_ylabel('Analytical cumulative cost (s)')
        ax.set_title(group_title(group)); ax.grid(alpha=.12)
    axes.flat[-1].axis('off')
    route_handles = [Line2D([], [], color=COLORS[r], linewidth=2, label=n) for r, n in
                     (('uniform', 'Uniform middle'), ('labor', 'LABOR middle'),
                      ('full', 'Exact recompute'), ('ram', 'Paid exact + RAM'))]
    axes.flat[-1].legend(handles=route_handles, loc='upper left', frameon=False)
    axes.flat[-1].text(0, .58, 'Thin lines: all three seeds.\nThick lines: three-seed mean formula.\nSeed markers: q = 1, 10, 100.\nx: analytic RAM/route intersection.\nCurves are a + bq, not fitted data.\nCommon graph/CP loading excluded.\nValid only for unchanged graph,\nfeatures, weights and workload.', fontsize=8.4, va='top')
    fig.suptitle('Paid recomputation and reuse over q = 1 to 100\nAnalytic extension of recorded coefficients; no new repeated executions', fontsize=11)
    save_figure(fig, output, 'final_full_RAM_amortization'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Regenerate Figures 1-4 from published scalar results; no GPU or model loading.')
    parser.add_argument('--manifest', type=Path, default=ARTIFACT/'results/figure-inputs.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest, docs, provenance, output = load_inputs(args.manifest, args.output)
    points = prepare_points(manifest, docs)
    published = json.loads((ARTIFACT/'results/published-figure-points.json').read_text(encoding='utf-8'))
    require(points == published['exact_plotted_scalars_and_coefficients'], 'Plotted numbers differ from the published figures')
    output.mkdir(parents=True, exist_ok=False)
    render(manifest, points, output)
    (output/'figure-provenance.json').write_text(json.dumps(dict(
        inputs=provenance, exact_plotted_scalars_and_coefficients=points,
        hardware_scope=manifest['hardware_scope'], workload_scope=manifest['workload_scope']), indent=2)+'\n', encoding='utf-8')
    print('Reproduced Figures 1-4; all plotted scalars exactly match the published record.')


if __name__ == '__main__':
    main()

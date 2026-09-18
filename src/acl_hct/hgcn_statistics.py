"""Conditional graph-repeat Student summaries and one complete Holm12 family."""
import numpy as np
from scipy.stats import t as student


def student_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('finite independent graph-repeat scalar vector required')
    n = len(values); mean = float(values.mean()); sd = float(values.std(ddof=1)); se = sd / np.sqrt(n)
    if np.all(values == values[0]):
        sd = 0.; se = 0.
        return {'n': n, 'df': n - 1, 'mean': mean, 'sample_sd': sd, 'SE': se,
                't': None, 'p': 1., 'marginal_student95': None,
                'zero_variance_policy': 'undefined inference, no rejection'}
    statistic = mean / se; critical = float(student.ppf(.975, n - 1))
    return {'n': n, 'df': n - 1, 'mean': mean, 'sample_sd': sd, 'SE': float(se),
            't': float(statistic), 'p': float(2 * student.sf(abs(statistic), n - 1)),
            'marginal_student95': [float(mean - critical * se), float(mean + critical * se)]}


def holm(rows, alpha=.05, family_size=12):
    if len(rows) != family_size or len({r['comparison'] for r in rows}) != family_size:
        raise ValueError('complete distinct registered primary family required')
    if any(not np.isfinite(r['p']) or not 0 <= r['p'] <= 1 for r in rows):
        raise ValueError('valid primary p-values required')
    order = sorted(range(len(rows)), key=lambda i: (rows[i]['p'], rows[i]['comparison']))
    adjusted = 0.; output = [dict(r) for r in rows]
    for rank, i in enumerate(order):
        adjusted = max(adjusted, min(1., (len(rows) - rank) * rows[i]['p']))
        output[i].update(p_holm=adjusted, reject_holm=adjusted <= alpha,
                         family_size=family_size, alpha=alpha, interval_scope='marginal, not simultaneous')
    return output


def primary_family(differences, seeds=(11, 23), budgets=(4, 8, 16), repeats=16):
    expected = {(s, b, metric) for s in seeds for b in budgets for metric in ('direct_order', 'micro_mrr')}
    if set(differences) != expected:
        raise ValueError('all two-model/three-budget/two-outcome comparisons required')
    rows = []
    for seed, budget, metric in sorted(expected):
        values = np.asarray(differences[seed, budget, metric], dtype=np.float64)
        if values.shape != (repeats,):
            raise ValueError('all sixteen complete graph replicates required')
        rows.append({'comparison': f'seed{seed}-f{budget}-{metric}', 'seed': seed, 'fanout': budget,
                     'metric': metric, 'contrast': 'S-F', **student_summary(values)})
    return holm(rows)

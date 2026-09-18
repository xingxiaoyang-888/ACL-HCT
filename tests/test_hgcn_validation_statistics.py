import numpy as np
import pytest
from scipy.stats import t

from acl_hct.hgcn_statistics import holm, primary_family, student_summary


def test_student_df15_against_known_symmetric_vector_and_translation():
    values = np.arange(16, dtype=np.float64) - 7.5
    symmetric = student_summary(values)
    assert symmetric['mean'] == 0 and symmetric['p'] == 1 and symmetric['df'] == 15
    shifted = student_summary(values + 2)
    se = np.sqrt(17 / 12)  # variance of 0..15 is 68/3, divided by n=16.
    assert shifted['SE'] == pytest.approx(se)
    assert shifted['t'] == pytest.approx(2 / se)
    assert shifted['p'] == pytest.approx(2 * t.sf(2 / se, 15))
    np.testing.assert_allclose(shifted['marginal_student95'], [2 - t.ppf(.975, 15) * se, 2 + t.ppf(.975, 15) * se])


def test_holm_step_down_ties_and_family_completeness():
    values = [.0001, .001, .003, .004, .01, .03, .1, .2, .3, .4, .8, 1.]
    rows = [{'comparison': str(i), 'p': p} for i, p in enumerate(values)]
    adjusted = holm(rows)
    np.testing.assert_allclose([r['p_holm'] for r in adjusted], [.0012, .011, .03, .036, .08, .21, .6, 1., 1., 1., 1., 1.])
    assert sum(r['reject_holm'] for r in adjusted) == 4
    tied = holm([{'comparison': str(i), 'p': .004} for i in range(12)])
    assert all(r['p_holm'] == .048 and r['reject_holm'] for r in tied)
    with pytest.raises(ValueError, match='complete'):
        holm(rows[:-1])
    with pytest.raises(ValueError, match='distinct'):
        holm([rows[0]] * 12)


def test_zero_variance_is_conservative_and_no_missing_primary_conditions():
    for value in (0., -.1, .1):
        row = student_summary(np.full(16, value))
        assert row['p'] == 1 and row['t'] is None and row['marginal_student95'] is None
    values = {(s, b, m): np.zeros(16) for s in (11, 23) for b in (4, 8, 16) for m in ('direct_order', 'micro_mrr')}
    assert all(not row['reject_holm'] for row in primary_family(values))
    values.pop((11, 4, 'direct_order'))
    with pytest.raises(ValueError, match='all'):
        primary_family(values)
    with pytest.raises(ValueError, match='finite'):
        student_summary([0., np.nan])

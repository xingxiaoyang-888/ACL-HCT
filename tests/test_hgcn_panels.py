from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from acl_hct.hgcn_panels import HierarchyPanel


def example():
    view = SimpleNamespace(nodes=['r', 'p', 'c', 'u'], root='r', reachable={'r', 'p', 'c'})
    panel = {'pool_size': 2, 'rows': [{'id': n, 'index': i, 'inclusion_probability': 1., 'pool_mean_weight': .5}
                                    for n, i in (('c', 2), ('u', 3))],
             'relations': [{'child': 'c', 'direct_parents': ['p'], 'positive_distant_ancestors': ['r']},
                           {'child': 'u', 'direct_parents': ['p'], 'positive_distant_ancestors': []}]}
    full = np.array([[.05, 0.], [.2, 0.], [.996, 0.], [0., .5]])
    return view, panel, full


def test_fixed_full_root_and_unknown_weight_denominator_do_not_move_with_sample():
    view, panel, full = example(); evaluation = HierarchyPanel(view, panel, full)
    actual = full.copy(); actual[0] = [.999, 0.]
    result = evaluation.evaluate(actual)
    assert result['direct']['metrics']['score'] == 1
    assert result['direct']['covered_pairs'] == 1 and result['direct']['unknown_pairs'] == 1
    assert result['direct']['weighted_covered_child_mass'] == .5
    assert result['direct']['weighted_selected_child_mass'] == 1
    assert np.array_equal(evaluation.anchor, full[0])
    assert np.array_equal(result['bias']['nodewise_error_fp64'], np.zeros((2, 2)))
    assert result['bias']['direction_known_nodes'] == 1


def test_unresolved_secondary_does_not_change_exact_sign_primary():
    view, panel, full = example(); full[2] = full[1] + np.array([1e-12, 0.])
    evaluation = HierarchyPanel(view, panel, full, gap_floor=1e-10)
    result = evaluation.evaluate(full)
    assert result['direct']['metrics']['score'] == 1
    assert result['direct']['metrics']['unresolved'] == 1
    changed = full.copy(); changed[2] = full[1]
    assert evaluation.evaluate(changed)['direct']['metrics']['score'] == .5


def test_no_root_or_relations_gives_unknown_not_fabricated_zero_score():
    view, panel, full = example(); view.root = None; view.reachable = set()
    result = HierarchyPanel(view, panel, full).evaluate(full)
    assert result['direct']['metrics']['score'] is None
    assert result['bias']['weighted_outward_projection'] is None
    assert result['direct']['unknown_pairs'] == 2


def test_invalid_weights_indices_or_noninterior_points_rejected():
    view, panel, full = example()
    bad = deepcopy(panel); bad['rows'][0]['pool_mean_weight'] = .6
    with pytest.raises(ValueError, match='weight'):
        HierarchyPanel(view, bad, full)
    bad = deepcopy(panel); bad['rows'][0]['index'] = 3
    with pytest.raises(ValueError, match='indices'):
        HierarchyPanel(view, bad, full)
    evaluation = HierarchyPanel(view, panel, full); changed = full.copy(); changed[0] = [1., 0.]
    with pytest.raises(ValueError, match='interior'):
        evaluation.evaluate(changed)

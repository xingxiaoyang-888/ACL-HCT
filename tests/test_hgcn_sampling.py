import itertools
import pytest
import torch

from acl_hct.hgcn_sampling import make_plan, mask_graph


def test_full_and_empty_preserve_self_weights_and_random_state():
    neighbors = [[], [2], [1, 3, 4], [2], [2]]
    rng = torch.Generator().manual_seed(88); before = rng.get_state().clone()
    full = make_plan(neighbors, None, rng); complete = make_plan(neighbors, 3, rng)
    assert torch.equal(before, rng.get_state())
    assert torch.equal(full.indices, complete.indices) and torch.equal(full.weights, complete.weights)
    dense = full.matrix(dtype=torch.float64).to_dense()
    assert dense[0, 0] == 1 and dense[1, 1] == .5
    torch.testing.assert_close(dense.sum(1), torch.ones(5, dtype=torch.float64), atol=0, rtol=0)


def test_sampling_probabilities_are_nonself_and_ht_not_sample_degree_normalization():
    neighbors = [[1, 2, 3, 4], [], [], [], []]
    plan = make_plan(neighbors, 2, torch.Generator().manual_seed(29))
    row = plan.indices[0] == 0; columns = plan.indices[1, row]
    assert len(columns) == 3 and len(set(columns.tolist())) == 3
    assert plan.weights[row][columns == 0].item() == .2
    assert set(plan.weights[row][columns != 0].tolist()) == {.4}
    assert set(plan.inclusion_probabilities[row][columns != 0].tolist()) == {.5}
    assert plan.populations.tolist() == [4, 0, 0, 0, 0]


def test_exact_enumeration_fixed_input_linear_tangent_mean_is_unbiased():
    # Independent enumeration of all six m=2 subsets, retaining exact self.
    features = torch.tensor([[8., -1.], [1., 2.], [2., 0.], [-1., 4.], [9., 1.]], dtype=torch.float64)
    rows = []
    for chosen in itertools.combinations(range(1, 5), 2):
        rows.append(.2 * features[0] + .4 * features[list(chosen)].sum(0))
    torch.testing.assert_close(torch.stack(rows).mean(0), features.mean(0), atol=1e-14, rtol=0)


def test_empirical_uniform_without_replacement_and_deterministic_replay():
    neighbors = [[1, 2, 3, 4], [], [], [], []]
    one = torch.Generator().manual_seed(91); two = torch.Generator().manual_seed(91)
    counts = torch.zeros(4, dtype=torch.long)
    for _ in range(1200):
        a = make_plan(neighbors, 2, one); b = make_plan(neighbors, 2, two)
        assert torch.equal(a.indices, b.indices) and torch.equal(a.weights, b.weights)
        selected = a.indices[1, (a.indices[0] == 0) & (a.indices[1] != 0)]
        counts[selected - 1] += 1
    assert (counts - 600).abs().max() < 90


def test_target_mask_precedes_degree_and_duplicates_do_not_change_graph():
    neighbors = [[1, 2], [0, 2], [0, 1]]
    masked = mask_graph(neighbors, [[0, 1], [0, 1]])
    assert masked == [[2], [2], [0, 1]] and neighbors == [[1, 2], [0, 2], [0, 1]]
    plan = make_plan(masked, 1, torch.Generator().manual_seed(8))
    assert plan.populations.tolist() == [1, 1, 2]
    dense = plan.matrix(dtype=torch.float64).to_dense()
    assert dense[0, 0] == .5 and dense[2, 2] == 1 / 3
    assert dense[0, 1] == dense[1, 0] == 0


@pytest.mark.parametrize('neighbors,fanout', [([[0]], 1), ([[1], []], 0), ([[1, 1], []], 1),
                                              ([[2, 1], [], []], 1), ([[]], True)])
def test_invalid_graph_or_budget_is_rejected(neighbors, fanout):
    with pytest.raises(ValueError):
        make_plan(neighbors, fanout, torch.Generator())


def test_missing_cpu_generator_and_unavailable_target_rejected():
    with pytest.raises(ValueError, match='CPU'):
        make_plan([[]], None, None)
    with pytest.raises(ValueError, match='visible'):
        mask_graph([[1], [0], []], [[0, 2]])

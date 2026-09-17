"""Artificial CPU-only tests; no prepared data, checkpoints, or downloads."""
import json
import math

import numpy as np
import pytest
import torch

from acl_hct.backbone_frozen_readings import (
    DEGREE_BINS, degree_readings, layer_readings, paired_comparisons,
    parameter_gradient_readings,
)


def test_local_coefficients_match_cpu_fp64_jacobian_and_directional_gradients():
    u = torch.tensor([[0., 0.], [3., 4.], [0., 2.], [100., 0.]], dtype=torch.float64)
    grad = torch.tensor([[3., 4.], [6., 8.], [3., 4.], [-2., 3.]], dtype=torch.float64)
    result = layer_readings(u, grad, 4., 1.2, [1, 1, 2], [0, 3])
    raw = result["raw"]
    np.testing.assert_allclose(raw["gradient_norm"], [5., 10., 5., math.sqrt(13)])
    np.testing.assert_allclose(raw["radial_signed"][1:], [10., 4., -2.], atol=1e-14)
    np.testing.assert_allclose(raw["perpendicular_norm"][1:], [0., 3., 3.], atol=1e-14)
    assert not raw["direction_defined"][0]
    for field in ("radial_signed", "radial_norm", "perpendicular_norm"):
        assert np.isnan(raw[field][0])
    for i in range(len(u)):
        def squash(value):
            return (.6 * value) / torch.sqrt(1 + value.square().sum())
        jac = torch.autograd.functional.jacobian(squash, u[i])
        if i == 0:
            torch.testing.assert_close(jac, .6 * torch.eye(2, dtype=torch.float64), atol=1e-14, rtol=1e-13)
        else:
            radial = u[i] / torch.linalg.vector_norm(u[i])
            tangent = torch.stack((-radial[1], radial[0]))
            np.testing.assert_allclose(float(radial @ jac @ radial), raw["jacobian_radial"][i],
                                       atol=1e-14, rtol=1e-11)
            np.testing.assert_allclose(float(tangent @ jac @ tangent), raw["jacobian_tangential"][i],
                                       atol=1e-14, rtol=1e-13)
    parent = result["summary"]["parent_occurrences"]
    assert (parent["count"], parent["unique_node_count"]) == (3, 2)
    assert parent["near_bound_fraction"] == pytest.approx(2/3)
    assert parent["radial_signed"]["mean"] == 8
    assert result["summary"]["all_unique_nodes"]["count"] == 4
    assert result["summary"]["child_occurrences"]["direction_undefined_count"] == 1
    json.dumps(result["summary"], allow_nan=False)
    snapshot = raw["u_norm"].copy()
    u.zero_()
    np.testing.assert_array_equal(raw["u_norm"], snapshot)


def test_zero_directions_and_empty_occurrences_are_null_and_counted():
    result = layer_readings(torch.zeros(2, 3), torch.ones(2, 3), 1., 1.2, [], [0, 0])
    assert result["summary"]["all_unique_nodes"]["radial_norm"]["mean"] is None
    assert result["summary"]["all_unique_nodes"]["direction_undefined_count"] == 2
    assert result["summary"]["parent_occurrences"]["near_bound_fraction"] is None
    assert result["summary"]["child_occurrences"]["gradient_norm"]["count"] == 2
    json.dumps(result["summary"], allow_nan=False)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_layer_gradients_are_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        layer_readings(torch.ones(2, 3), torch.full((2, 3), bad), 1., 1.2, [0], [1])


def test_parameter_gradients_are_actual_required_and_zero_fraction_is_observed():
    model = torch.nn.Linear(2, 1, dtype=torch.float64)
    with pytest.raises(ValueError, match="missing parameter gradient"):
        parameter_gradient_readings(model)
    model.weight.grad = torch.tensor([[3., 0.]], dtype=torch.float64)
    model.bias.grad = torch.zeros_like(model.bias)
    result = parameter_gradient_readings(model)
    assert result["weight"]["norm"] == 3
    assert result["weight"]["exact_zero_fraction"] == .5
    assert result["bias"]["exact_zero_fraction"] == 1
    assert all(row["finite"] for row in result.values())
    model.bias.grad.fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        parameter_gradient_readings(model)


def tiny_graph():
    graph = [[] for _ in range(21)]
    graph[1] = [2]
    graph[2] = [1, 3]
    graph[3] = list(range(4, 21))
    return graph


def test_degrees_preserve_occurrences_unique_nodes_and_mask_switches():
    graph = tiny_graph()
    condition = [row.copy() for row in graph]
    condition[1] = []
    condition[2] = [3]
    plans = [[row[:16] for row in condition], [row[-16:] for row in condition]]
    queries = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [1, 1]])
    result = degree_readings(graph, condition, plans, queries)
    public = result["summary"]
    assert public["layers"][0]["message_count"] == 17
    assert public["layers"][0]["selected_neighbor_message_count"] == 17
    assert public["layers"][0]["self_fallback_message_count"] == 19
    assert public["layers"][0]["effective_message_count"] == 36
    assert public["layers"][0]["empty_neighborhood_count"] == 19
    group = public["positive_parent_degree"]["1"]
    assert group["occurrence_count"] == 2
    assert group["unique_node_count"] == 1
    assert group["nonempty_to_empty_occurrence_count"] == 2
    assert group["nonempty_to_empty_unique_node_count"] == 1
    assert group["layers"][0]["self_fallback_occurrence_count"] == 2
    assert public["positive_parent_degree"][">16"]["layers"][0]["k"]["mean"] == 16
    assert result["raw"]["N"].shape == (2, 21)
    assert result["raw"]["k"][0, 3] == 16
    assert result["raw"]["N"][0, 3] == 17
    json.dumps(public, allow_nan=False)
    plans[0][2] = []
    with pytest.raises(ValueError, match="empty iff"):
        degree_readings(graph, condition, plans, queries)


def tiny_cells():
    # Negative parents deliberately disagree with each group's true parent bin.
    queries = np.zeros((3, 5, 2), dtype=np.int64)
    queries[:, :, 0] = 3
    queries[:, 0, 0] = [0, 1, 3]
    queries[:, :, 1] = np.array([1, 3, 0])[:, None]
    cells = {}
    mask = np.repeat([1., 2., 4.], 5)
    for graph in ("unmasked", "masked"):
        base = mask if graph == "masked" else np.zeros(15)
        metric_base = 10. if graph == "unmasked" else 12.
        for fanout, reps in (("full", [None]), ("f16", range(8))):
            for rep in reps:
                sample = np.zeros(15) if rep is None else np.repeat([rep, 2*rep, 3*rep], 5)
                if graph == "masked" and rep is not None:
                    sample = sample + np.repeat([-rep, 0., rep], 5)
                logits = base + sample
                cells[(graph, fanout, rep)] = {
                    "logits": logits, "bce": 20 + 2*logits,
                    "margins": logits.reshape(3, 5)[:, 0] - logits.reshape(3, 5)[:, 1:].mean(axis=1),
                    "metrics": {"gradient": metric_base + (rep or 0) *
                                (2 if graph == "masked" else 1)}}
    return cells, queries


def test_complete_paired_contrasts_use_positive_group_bins_and_all_replicates():
    cells, queries = tiny_cells()
    result = paired_comparisons(cells, queries, tiny_graph())
    np.testing.assert_array_equal(result["raw"]["mask_full"]["logits"], np.repeat([1., 2., 4.], 5))
    np.testing.assert_array_equal(result["raw"]["interaction"]["logits"],
                                  np.stack([np.repeat([-rep, 0., rep], 5) for rep in range(8)]))
    public = result["summary"]
    full = public["mask_full"]["logits"]["positive_parent_degree"]
    assert set(full) == set(DEGREE_BINS)
    assert full["0"]["record_count"] == 5
    assert full["0"]["statistics"]["mean"] == 1
    assert full[">16"]["record_count"] == 5
    assert full[">16"]["statistics"]["mean"] == 4
    assert full["2-16"]["record_count"] == 0
    assert full["2-16"]["statistics"]["mean"] is None
    assert public["mask_full"]["margins"]["positive_parent_degree"]["0"]["record_count"] == 1
    assert sum(group["contribution_to_overall_mean"] or 0 for group in full.values()) == pytest.approx(7/3)
    sampled = public["interaction"]["logits"]["positive_parent_degree"][">16"]
    assert sampled["replicate_means"] == list(range(8))
    assert sampled["replicate_mean_statistics"]["mean"] == 3.5
    assert sampled["replicate_mean_statistics"]["sample_sd"] == pytest.approx(np.arange(8).std(ddof=1))
    assert sampled["replicate_mean_directions"] == {"negative": 0, "zero": 1, "positive": 7}
    assert public["interaction"]["metrics"]["gradient"]["replicates"] == list(range(8))
    assert public["mask_full"]["metrics"]["gradient"]["difference"] == 2
    json.dumps(public, allow_nan=False)
    assert "node_ids" not in json.dumps(public)
    cells[("masked", "full", None)]["logits"].fill(0)
    np.testing.assert_array_equal(result["raw"]["mask_full"]["logits"], np.repeat([1., 2., 4.], 5))


def test_partial_nonfinite_and_inconsistent_metric_matrices_fail():
    cells, queries = tiny_cells()
    del cells[("masked", "f16", 7)]
    with pytest.raises(ValueError, match="complete fixed"):
        paired_comparisons(cells, queries, tiny_graph())
    cells, queries = tiny_cells()
    cells[("masked", "f16", 7)]["metrics"] = {}
    with pytest.raises(ValueError, match="identical keys"):
        paired_comparisons(cells, queries, tiny_graph())
    cells, queries = tiny_cells()
    cells[("masked", "f16", 7)]["metrics"]["gradient"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        paired_comparisons(cells, queries, tiny_graph())

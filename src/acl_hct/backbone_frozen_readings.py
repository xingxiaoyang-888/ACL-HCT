"""Low-dimensional observations for the registered frozen-backbone diagnosis.

These functions do not run a model or change tensors. Raw arrays are private
evidence; public summaries never contain node IDs. Directional quantities at
u=0 have explicit undefined masks rather than invented zero directions.
"""
import math

import numpy as np
import torch


DEGREE_BINS = ("0", "1", "2-16", ">16")
QUANTILES = (0., .25, .5, .75, .9, .95, .99, 1.)


def _array(value, name, floating=True):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if floating:
        if result.dtype.kind not in "fiu" or not np.isfinite(result).all():
            raise ValueError(f"{name} must contain finite numeric values")
        result = result.astype(np.float64, copy=True)
    elif not result.size:
        result = result.astype(np.int64)
    elif result.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer indices")
    return result.copy() if not floating else result


def _indices(value, n, name):
    result = _array(value, name, False)
    if result.ndim != 1 or (result < 0).any() or (result >= n).any():
        raise ValueError(f"{name} must be one-dimensional valid node indices")
    return result.astype(np.int64, copy=False)


def _positives(queries, n):
    result = _array(queries, "positive_queries", False)
    if result.ndim == 3 and result.shape[1:] == (5, 2):
        result = result[:, 0, :]
    if result.ndim != 2 or result.shape[1] != 2 or not len(result):
        raise ValueError("positive_queries must have shape [G,2] or [G,5,2]")
    _indices(result[:, 0], n, "positive parents")
    _indices(result[:, 1], n, "positive children")
    return result.astype(np.int64, copy=False)


def _stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("summary values must be finite")
    result = {"count": int(len(values)), "exact_zero_count": int((values == 0).sum())}
    if not len(values):
        return dict(result, mean=None, sample_sd=None, minimum=None, maximum=None,
                    quantiles={str(q): None for q in QUANTILES})
    summary = dict(result, mean=float(values.mean()),
                sample_sd=float(values.std(ddof=1)) if len(values) > 1 else None,
                minimum=float(values.min()), maximum=float(values.max()),
                quantiles={str(q): float(v) for q, v in zip(QUANTILES, np.quantile(values, QUANTILES))})
    if any(value is not None and not math.isfinite(value) for value in
           (summary["mean"], summary["sample_sd"], *summary["quantiles"].values())):
        raise ValueError("nonfinite summary statistic")
    return summary


def _directions(values):
    values = np.asarray(values)
    return {"negative": int((values < 0).sum()), "zero": int((values == 0).sum()),
            "positive": int((values > 0).sum())}


def _membership(degree):
    return {"0": degree == 0, "1": degree == 1,
            "2-16": (degree >= 2) & (degree <= 16), ">16": degree > 16}


def _neighbors(rows, name):
    n = len(rows)
    if not n:
        raise ValueError(f"{name} cannot be empty")
    for row in rows:
        if len(row) != len(set(row)) or any(
                not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_))
                or not 0 <= i < n for i in row):
            raise ValueError(f"{name} must have distinct valid node indices")
    return np.asarray([len(row) for row in rows], dtype=np.int64)


def layer_readings(u, grad, c, R, positive_parents, positive_children):
    """Observe a Linear output and its actual mean-loss gradient, in float64.

    Occurrence summaries retain repeated positive parents/children; the global
    summary counts each node exactly once. No derivatives of the entire network
    are inferred from the local analytical Jacobian coefficients.
    """
    if not math.isfinite(c) or c <= 0 or not math.isfinite(R) or R <= 0:
        raise ValueError("c and R must be positive finite scalars")
    u = _array(u, "u")
    grad = _array(grad, "dL/du")
    if u.ndim != 2 or not u.shape[0] or not u.shape[1] or grad.shape != u.shape:
        raise ValueError("u and dL/du must have the same nonempty [nodes,width] shape")
    parents = _indices(positive_parents, len(u), "positive parents")
    children = _indices(positive_children, len(u), "positive children")
    s = np.linalg.norm(u, axis=1)
    root = np.hypot(1., s)
    defined = s != 0
    unit = np.zeros_like(u)
    unit[defined] = u[defined] / s[defined, None]
    radial = np.full(len(u), np.nan)
    radial[defined] = (grad[defined] * unit[defined]).sum(axis=1)
    perpendicular = np.full(len(u), np.nan)
    perpendicular[defined] = np.linalg.norm(
        grad[defined] - radial[defined, None] * unit[defined], axis=1)
    raw = {"u_norm": s, "parameterized_radius": R * (s / root),
           "jacobian_radial": (R / math.sqrt(c)) / root / root / root,
           "jacobian_tangential": (R / math.sqrt(c)) / root,
           "gradient_norm": np.linalg.norm(grad, axis=1),
           "radial_signed": radial, "radial_norm": np.abs(radial),
           "perpendicular_norm": perpendicular, "direction_defined": defined}
    raw["near_bound"] = raw["parameterized_radius"] >= .95 * R
    directional = {"radial_signed", "radial_norm", "perpendicular_norm"}
    for name, values in raw.items():
        expected = defined if name in directional else np.ones(len(u), dtype=bool)
        if not np.isfinite(values[expected]).all() or (name in directional and
                                                      not np.isnan(values[~expected]).all()):
            raise ValueError(f"nonfinite layer reading: {name}")
    summary = {}
    for name, ids in (("all_unique_nodes", np.arange(len(u))),
                      ("parent_occurrences", parents), ("child_occurrences", children)):
        valid_ids = ids[defined[ids]]
        row = {"count": int(len(ids)), "unique_node_count": int(len(np.unique(ids))),
               "zero_u_norm_count": int((s[ids] == 0).sum()),
               "direction_undefined_count": int((~defined[ids]).sum()),
               "near_bound_count": int(raw["near_bound"][ids].sum()),
               "near_bound_fraction": float(raw["near_bound"][ids].mean()) if len(ids) else None}
        for field, values in raw.items():
            if field not in {"near_bound", "direction_defined"}:
                row[field] = _stats(values[valid_ids if field in directional else ids])
        summary[name] = row
    return {"raw": raw, "summary": summary}


def parameter_gradient_readings(model):
    """Require a finite gradient for every original model parameter."""
    summary = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise ValueError(f"missing parameter gradient: {name}")
        grad = _array(parameter.grad, f"parameter gradient {name}")
        norm = float(np.linalg.norm(grad.reshape(-1)))
        if not math.isfinite(norm):
            raise ValueError(f"nonfinite parameter gradient norm: {name}")
        summary[name] = {"element_count": int(grad.size), "norm": norm, "finite": True,
                         "exact_zero_count": int((grad == 0).sum()),
                         "exact_zero_fraction": float((grad == 0).mean()) if grad.size else None}
    if not summary:
        raise ValueError("model has no parameters")
    return summary


def degree_readings(unmasked, condition, plans, positive_queries):
    """Summarize fixed unmasked-degree bins and actual two-layer message plans."""
    base = _neighbors(unmasked, "unmasked graph")
    current = _neighbors(condition, "condition graph")
    if len(current) != len(base) or len(plans) != 2:
        raise ValueError("same node order and exactly two layer plans required")
    if any(not set(row).issubset(original) for original, row in zip(unmasked, condition)):
        raise ValueError("condition graph must only remove unmasked edges")
    queries = _positives(positive_queries, len(base))
    k = []
    for plan in plans:
        selected = _neighbors(plan, "layer plan")
        if len(selected) != len(base) or any(
                not set(row).issubset(candidate) or bool(row) != bool(candidate)
                for candidate, row in zip(condition, plan)):
            raise ValueError("plans must select visible candidates, empty iff candidate row empty")
        k.append(selected)
    raw = {"unmasked_degree": base, "condition_degree": current,
           "N": np.stack((current, current)), "k": np.stack(k),
           "nonempty_to_empty": (base > 0) & (current == 0)}
    summary = {"node_count": int(len(base)), "layers": [],
               "positive_parent_degree": {}, "positive_child_degree": {}}
    for selected in k:
        empty_count = int((current == 0).sum())
        summary["layers"].append({"N": _stats(current), "k": _stats(selected),
                                   "message_count": int(selected.sum()),
                                   "selected_neighbor_message_count": int(selected.sum()),
                                   "self_fallback_message_count": empty_count,
                                   "effective_message_count": int(selected.sum()) + empty_count,
                                   "empty_neighborhood_count": empty_count,
                                   "empty_neighborhood_fraction": float((current == 0).mean())})
    for column, name in ((0, "positive_parent_degree"), (1, "positive_child_degree")):
        ids = queries[:, column]
        for bin_name, membership in _membership(base[ids]).items():
            group = ids[membership]
            summary[name][bin_name] = {
                "occurrence_count": int(len(group)), "unique_node_count": int(len(np.unique(group))),
                "unmasked_degree": _stats(base[group]), "condition_degree": _stats(current[group]),
                "nonempty_to_empty_occurrence_count": int(raw["nonempty_to_empty"][group].sum()),
                "nonempty_to_empty_unique_node_count": int(len(np.unique(
                    group[raw["nonempty_to_empty"][group]]))),
                "layers": [{"N": _stats(current[group]), "k": _stats(selected[group]),
                            "message_occurrence_count": int(selected[group].sum()),
                            "message_unique_node_count": int(selected[np.unique(group)].sum()),
                            "self_fallback_occurrence_count": int((current[group] == 0).sum()),
                            "self_fallback_unique_node_count": int((current[np.unique(group)] == 0).sum())}
                           for selected in k]}
    return {"raw": raw, "summary": summary}


def _contrast_summary(values, queries, degree, group_records):
    """Summarize either one contrast vector or all eight replicate vectors."""
    sampled = values.ndim == 2
    matrix = values if sampled else values[None, :]
    total = matrix.shape[1]

    def summarize(mask):
        subset = matrix[:, mask]
        count = int(mask.sum())
        if sampled:
            means = subset.mean(axis=1) if count else np.full(8, np.nan)
            return {"record_count": count,
                    "replicate_means": means.tolist() if count else [None] * 8,
                    "replicate_contributions_to_overall_mean":
                        (subset.sum(axis=1) / total).tolist() if count else [None] * 8,
                    "replicate_mean_statistics": _stats(means if count else []),
                    "replicate_mean_directions": _directions(means if count else []),
                    "per_replicate_record_directions": [_directions(v) for v in subset]}
        row = _stats(subset[0])
        return {"record_count": count, "statistics": row,
                "contribution_to_overall_mean": float(subset.sum() / total) if count else None,
                "record_directions": _directions(subset[0])}

    summary = {"overall": summarize(np.ones(total, dtype=bool)),
               "positive_parent_degree": {}, "positive_child_degree": {}}
    for column, name in ((0, "positive_parent_degree"), (1, "positive_child_degree")):
        for bin_name, mask in _membership(degree[queries[:, column]]).items():
            record_mask = np.repeat(mask, 5) if group_records else mask
            row = summarize(record_mask)
            row["positive_group_count"] = int(mask.sum())
            row["unique_positive_node_count"] = int(len(np.unique(queries[mask, column])))
            summary[name][bin_name] = row
    return summary


def paired_comparisons(cells, positive_queries, unmasked):
    """Require all 18 fixed cells; retain all eight sampling contrasts.

    cells[(graph, fanout, replicate)] holds logits, bce, margins and optional
    metrics (identical finite scalar keys everywhere). Full replicate is None;
    f16 replicates are integers 0..7. Both parent and child bins are alternative
    decompositions of the same contrasts, never additive contributions.
    """
    degree = _neighbors(unmasked, "unmasked graph")
    queries = _positives(positive_queries, len(degree))
    expected = {(graph, "full", None) for graph in ("unmasked", "masked")}
    expected |= {(graph, "f16", rep) for graph in ("unmasked", "masked") for rep in range(8)}
    if set(cells) != expected:
        raise ValueError("complete fixed 18-cell matrix required (two full and sixteen f16)")
    normalized = {}
    metric_names = None
    for key, cell in cells.items():
        row = {}
        for name, count in (("logits", len(queries) * 5), ("bce", len(queries) * 5),
                            ("margins", len(queries))):
            values = _array(cell[name], f"{key} {name}")
            if values.shape != (count,):
                raise ValueError(f"{name} must have shape [{count}]")
            row[name] = values
        metrics = cell.get("metrics", {})
        if not isinstance(metrics, dict) or any(not isinstance(name, str) for name in metrics):
            raise ValueError("metrics must map string names to finite scalars")
        if metric_names is None:
            metric_names = set(metrics)
        if set(metrics) != metric_names:
            raise ValueError("metrics must have identical keys in all cells")
        row["metrics"] = {}
        for name, value in metrics.items():
            scalar = _array(value, f"metric {name}")
            if scalar.ndim != 0:
                raise ValueError("each metric must be a finite scalar")
            row["metrics"][name] = float(scalar)
        normalized[key] = row

    def subtract(left, right):
        result = {name: left[name] - right[name] for name in ("logits", "bce", "margins")}
        result["metrics"] = {name: left["metrics"][name] - right["metrics"][name]
                             for name in sorted(metric_names)}
        if any(not np.isfinite(result[name]).all() for name in ("logits", "bce", "margins")) or any(
                not math.isfinite(value) for value in result["metrics"].values()):
            raise ValueError("nonfinite paired difference")
        return result

    def stack(rows):
        result = {name: np.stack([row[name] for row in rows]) for name in ("logits", "bce", "margins")}
        result["metrics"] = {name: np.asarray([row["metrics"][name] for row in rows])
                             for name in sorted(metric_names)}
        return result

    raw = {"mask_full": subtract(normalized[("masked", "full", None)],
                                 normalized[("unmasked", "full", None)]),
           "sample_minus_full": {}}
    samples = {}
    for graph in ("unmasked", "masked"):
        samples[graph] = [subtract(normalized[(graph, "f16", rep)],
                                   normalized[(graph, "full", None)]) for rep in range(8)]
        raw["sample_minus_full"][graph] = stack(samples[graph])
    raw["interaction"] = stack([subtract(samples["masked"][rep], samples["unmasked"][rep])
                                 for rep in range(8)])

    def summary(row):
        result = {name: _contrast_summary(row[name], queries, degree, name != "margins")
                  for name in ("logits", "bce", "margins")}
        result["metrics"] = {}
        for name, value in row["metrics"].items():
            if np.ndim(value) == 0:
                result["metrics"][name] = {"difference": float(value),
                                           "directions": _directions([value])}
            else:
                result["metrics"][name] = {"replicates": value.tolist(),
                                           "statistics": _stats(value), "directions": _directions(value)}
        return result

    public = {"mask_full": summary(raw["mask_full"]),
              "sample_minus_full": {graph: summary(raw["sample_minus_full"][graph])
                                    for graph in ("unmasked", "masked")},
              "interaction": summary(raw["interaction"]),
              "replicate_count": 8, "grouping_note":
                  "parent and child degree summaries are alternative decompositions; do not add them"}
    return {"raw": raw, "summary": public}

"""CPU-only analysis of existing, prospectively enumerated fit records.

No model/weights/raw-state/sealed/test loading, no forward passes, no fitting.
Public I/O adaptation; original metric calculations are unchanged.
Input hashes bind the archived development records; no model forward is run.
"""
import argparse
import csv
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import statistics
import time


ARMS = ("AhatZhat", "AZhat", "AhatZ", "AZ")
ARM_MEANING = {
    "AhatZhat": "sampled_local_and_sampled_upstream",
    "AZhat": "restored_local_and_sampled_upstream",
    "AhatZ": "sampled_local_and_restored_upstream",
    "AZ": "restored_local_and_restored_upstream",
    "first_full": "first_layer_full_then_original_sampled_suffix",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def mean(xs):
    return statistics.fmean(xs)


def sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def tensor_digest(x):
    # Logical ID/label identity only; do not depend on pickle serialization.
    return sha(json.dumps(x.tolist(), separators=(",", ":")).encode("utf-8"))


def describe(x):
    x = x.double().flatten()
    return {"mean": x.mean().item(), "median": x.median().item(),
            "q10": x.quantile(0.1).item(), "q90": x.quantile(0.9).item(),
            "min": x.min().item(), "max": x.max().item()}


def task_values(torch, logits, metric, labels):
    if logits.ndim != 2 or logits.shape != metric.shape or logits.shape[0] != len(labels):
        raise ValueError("Unexpected native score shape")
    if not torch.isfinite(logits).all() or not torch.isfinite(metric).all():
        raise ValueError("Nonfinite native scores")
    pred = metric.argmax(-1)
    correct = pred.eq(labels)
    scores = logits.double()
    rivals = scores.clone()
    rivals.scatter_(1, labels[:, None], -torch.inf)
    margins = scores.gather(1, labels[:, None]).flatten() - rivals.max(-1).values
    return {"pred": pred, "correct": correct, "margin": margins}


def metrics(values):
    n = len(values["correct"])
    return {"node_count": n, "correct_count": int(values["correct"].sum()),
            "accuracy_pct": 100 * values["correct"].double().mean().item(),
            **{"true_margin_" + k: v for k, v in describe(values["margin"]).items()}}


def contrast(before, after):
    n = len(before["correct"])
    fixed = ~before["correct"] & after["correct"]
    harmed = before["correct"] & ~after["correct"]
    changed_wrong = ~before["correct"] & ~after["correct"] & before["pred"].ne(after["pred"])
    return {"node_count": n, "corrected_errors": int(fixed.sum()), "introduced_errors": int(harmed.sum()),
            "changed_wrong_prediction": int(changed_wrong.sum()),
            "prediction_changes": int(before["pred"].ne(after["pred"]).sum()),
            "accuracy_delta_pp": 100 * (int(fixed.sum()) - int(harmed.sum())) / n,
            "mean_true_margin_delta": (after["margin"] - before["margin"]).mean().item()}


def aggregate_rows(rows, keys, value_fields, count_fields=()):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for key, group in groups.items():
        out = dict(zip(keys, key))
        out["plan_count"] = len(group)
        for field in value_fields:
            values = [r[field] for r in group]
            out[field + "_mean"] = mean(values)
            out[field + "_plan_sd_descriptive"] = sd(values)
            out[field + "_plan_min"] = min(values)
            out[field + "_plan_max"] = max(values)
        for field in count_fields:
            out[field + "_node_plan_events"] = sum(r[field] for r in group)
        out["target_nodes_per_plan"] = group[0]["node_count"]
        out["node_plan_denominator"] = sum(r["node_count"] for r in group)
        result.append(out)
    return result


def run(args):
    started, cpu_started = time.perf_counter(), time.process_time()
    script = Path(__file__).resolve()
    inp = Path(args.inputs).resolve()
    output = Path(args.output).resolve()
    expected = {"script_sha256": sha(script.read_bytes()), "input_manifest_sha256": sha(inp.read_bytes())}
    raw_root = Path(args.raw_root).resolve()
    if output.exists():
        raise FileExistsError("Refusing output overwrite")
    manifest = read_json(inp)
    if manifest.get("schema") != "vldb_task_a_fit_input_manifest_v1" or len(manifest["jobs"]) != 24:
        raise ValueError("Unexpected input scope")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    loaded = []
    cp_meta, plan_rows, contrast_rows, interaction_rows, rep_rows, head_rows = [], [], [], [], [], []
    low_rows = []
    pair_source = {}
    try:
        for spec in manifest["jobs"]:
            job = spec["job"]
            ident = {k: job[k] for k in ("model", "dataset", "regime", "seed")}
            directory = (raw_root / spec["capture_directory"]).resolve()
            if not directory.is_relative_to(raw_root):
                raise ValueError("Input directory escape")
            file_map = {f["path"]: f for f in spec["files"]}
            def load(relative):
                if not relative.startswith("fit/") or any(x in relative.split("/") for x in ("..", "sealed", "evaluation", "test")):
                    raise ValueError("Disallowed path")
                entry = file_map[relative]
                path = (directory / relative).resolve()
                if not path.is_relative_to(directory):
                    raise ValueError("Input escape")
                raw = path.read_bytes()
                if len(raw) != entry["bytes"] or sha(raw) != entry["sha256"]:
                    raise ValueError("Input identity differs: " + str(path))
                value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
                loaded.append({"path": str(path).replace("\\", "/"), "sha256": entry["sha256"], "bytes": len(raw)})
                return value
            ids_hash = labels_hash = None
            central = spec["central_budget"]
            low = 4 if ident["dataset"] == "Airport" else 1
            for stream in range(spec["fit_stream_start"], spec["fit_stream_start"] + 16):
                key = f"stream{stream}-k{central}"
                target = load(f"fit/outcomes/{key}.pt")
                ids, labels = target["validation_ids"].long(), target["validation_labels"].long()
                if target.get("test_scoring") is not False or target.get("schema") != "validation_raw_targets_v1":
                    raise ValueError("Unexpected outcome scope")
                if ids.ndim != 1 or labels.shape != ids.shape or len(ids.unique()) != len(ids) or (labels < 0).any():
                    raise ValueError("Invalid target IDs/labels")
                current_hashes = tensor_digest(ids), tensor_digest(labels)
                if ids_hash is None:
                    ids_hash, labels_hash = current_hashes
                elif current_hashes != (ids_hash, labels_hash):
                    raise ValueError("Target ID/label mismatch across plans")
                data = load(f"fit/interventions/{key}-final-fourway.pt")
                conditions = dict(data["distinct"])
                for alias, dest in data["aliases"].items():
                    if alias in ARMS:
                        conditions[alias] = conditions[dest]
                if set(conditions) != set(ARMS):
                    raise ValueError("Incomplete four-way conditions")
                values = {a: task_values(torch, conditions[a]["native_logits"][ids],
                                        conditions[a]["native_metric_output"][ids], labels) for a in ARMS}
                if ident["model"] == "HyboNet":
                    first = load(f"fit/interventions/{key}-first-full-targets.pt")
                    if not torch.equal(first["validation_ids"].long(), ids) or not torch.equal(first["validation_labels"].long(), labels):
                        raise ValueError("First-restoration target mismatch")
                    values["first_full"] = task_values(torch, first["sample_native_logits"], first["sample_native_metric_output"], labels)
                else:
                    expected_aliases = {"AhatZ": "AhatZhat", "AZhat": "AZ", "first_layer_full_restoration": "AZ"}
                    if data["aliases"] != expected_aliases or spec["message_layers"] != 1:
                        raise ValueError("HGCN structural alias contract differs")
                    values["first_full"] = values["AZ"]
                common = {**ident, "budget": central, "stream": stream}
                for arm, v in values.items():
                    plan_rows.append({**common, "arm": arm, **metrics(v)})
                    if arm != "AhatZhat":
                        contrast_rows.append({**common, "before": "AhatZhat", "after": arm,
                                              **contrast(values["AhatZhat"], v)})
                # Explicit marginal conditional effects, in addition to common baseline.
                for before, after in (("AZhat", "AZ"), ("AhatZ", "AZ")):
                    contrast_rows.append({**common, "before": before, "after": after,
                                          **contrast(values[before], values[after])})
                interaction = (values["AZ"]["correct"].double() - values["AZhat"]["correct"].double()
                               - values["AhatZ"]["correct"].double() + values["AhatZhat"]["correct"].double())
                margin_interaction = values["AZ"]["margin"] - values["AZhat"]["margin"] - values["AhatZ"]["margin"] + values["AhatZhat"]["margin"]
                interaction_rows.append({**common, "node_count": len(ids), "accuracy_interaction_pp": 100 * interaction.mean().item(),
                                         "mean_true_margin_interaction": margin_interaction.mean().item()})
                # Descriptive Euclidean coordinate norms, not hyperbolic distance or causal shares.
                agg = {a: conditions[a]["aggregate_native"].double() for a in ARMS}
                components = {"local_at_full_inputs": agg["AhatZ"] - agg["AZ"],
                              "upstream_under_full_local": agg["AZhat"] - agg["AZ"],
                              "cross_difference": agg["AhatZhat"] - agg["AZhat"] - agg["AhatZ"] + agg["AZ"],
                              "total": agg["AhatZhat"] - agg["AZ"]}
                for name, delta in components.items():
                    for population, rows in (("validation", ids), ("all_nodes", torch.arange(len(delta)))):
                        norm = delta[rows].norm(dim=-1)
                        rep_rows.append({**common, "component": name, "population": population,
                                         "node_count": len(rows), "coordinate_l2_mean": norm.mean().item(),
                                         "coordinate_l2_max": norm.max().item(), "coordinate_squared_l2_mean": norm.square().mean().item()})
                if "head_oracle_channels" in data:
                    channels = data["head_oracle_channels"]
                    for name in ("sampled", "full_projection_sample_nu", "sample_projection_full_nu", "full"):
                        score = channels[name][ids]
                        head_rows.append({**common, "channel": name, "score_kind": "scaled_accounting_not_native_or_deployable",
                                          **metrics(task_values(torch, score, score, labels))})
                lo = load(f"fit/outcomes/stream{stream}-k{low}.pt")
                if not torch.equal(lo["validation_ids"].long(), ids) or not torch.equal(lo["validation_labels"].long(), labels):
                    raise ValueError("Low-budget target mismatch")
                for arm, prefix in (("sample", "sample"), ("full", "full")):
                    low_rows.append({**ident, "budget": low, "stream": stream, "arm": arm,
                                     **metrics(task_values(torch, lo[prefix + "_native_logits"], lo[prefix + "_native_metric_output"], labels))})
                pair_source[(ident["model"], ident["dataset"], ident["seed"], ident["regime"], stream)] = {
                    "ids_sha256": ids_hash, "labels_sha256": labels_hash,
                    "values": {a: {k: v[k].clone() for k in ("pred", "correct", "margin")} for a, v in values.items()}}
            cp_meta.append({**ident, "checkpoint_sha256": job["checkpoint_sha256"], "target_nodes": len(ids),
                            "validation_ids_sha256": ids_hash, "validation_labels_sha256": labels_hash,
                            "message_layers": spec["message_layers"], "fit_plan_count": 16,
                            "accepted_capture": spec["accepted_capture"], "capture_directory": spec["capture_directory"]})
        # Training regime contrasts use identical original node IDs and stream keys.
        regime_rows = []
        for model, dataset, seed, regime, stream in pair_source:
            if regime != "full":
                continue
            a = pair_source[(model, dataset, seed, "full", stream)]
            b = pair_source[(model, dataset, seed, "central_matched", stream)]
            if a["ids_sha256"] != b["ids_sha256"] or a["labels_sha256"] != b["labels_sha256"]:
                raise ValueError("Training regime pairing mismatch")
            for arm in a["values"]:
                regime_rows.append({"model": model, "dataset": dataset, "seed": seed, "stream": stream, "arm": arm,
                                    "direction": "central_matched_minus_full_trained", **contrast(a["values"][arm], b["values"][arm])})
        cp_keys = ("model", "dataset", "regime", "seed", "budget")
        metric_fields = ("accuracy_pct", "true_margin_mean", "true_margin_median", "true_margin_q10", "true_margin_q90")
        cells = aggregate_rows(plan_rows, cp_keys + ("arm",), metric_fields, ("correct_count",))
        changes = aggregate_rows(contrast_rows, cp_keys + ("before", "after"),
                                 ("accuracy_delta_pp", "mean_true_margin_delta"),
                                 ("corrected_errors", "introduced_errors", "changed_wrong_prediction", "prediction_changes"))
        interactions = aggregate_rows(interaction_rows, cp_keys, ("accuracy_interaction_pp", "mean_true_margin_interaction"))
        regime_cells = aggregate_rows(regime_rows, ("model", "dataset", "seed", "arm", "direction"),
                                      ("accuracy_delta_pp", "mean_true_margin_delta"),
                                      ("corrected_errors", "introduced_errors", "prediction_changes"))
        seed_groups = {}
        for c in cells:
            key = (c["model"], c["dataset"], c["regime"], c["budget"], c["arm"])
            seed_groups.setdefault(key, []).append(c)
        seed_summary = []
        for key, group in seed_groups.items():
            if sorted(g["seed"] for g in group) != [11, 23, 37]:
                raise ValueError("Missing fixed seed")
            seed_summary.append({**dict(zip(("model", "dataset", "regime", "budget", "arm"), key)),
                                 "seed_count": 3, "accuracy_pct_mean": mean([g["accuracy_pct_mean"] for g in group]),
                                 "accuracy_pct_between_seed_sd": sd([g["accuracy_pct_mean"] for g in group]),
                                 "per_seed_accuracy_pct": {str(g["seed"]): g["accuracy_pct_mean"] for g in group}})
        if len(loaded) != manifest["file_count"] or sum(f["bytes"] for f in loaded) != manifest["bytes"]:
            raise ValueError("Input consumption count differs")
        report = {"schema": "vldb_task_a_fit_aggregate_v1", "status": "completed_pending_independent_result_review",
                  "evidence_split": "original_fit_only_not_independent_confirmation", "arm_definitions": ARM_MEANING,
                  "CP_metadata": cp_meta, "per_CP_conditions": cells, "per_CP_paired_contrasts": changes,
                  "per_CP_nonadditivity": interactions, "paired_training_regime_changes": regime_cells,
                  "three_seed_descriptive_summary": seed_summary,
                  "limitations": ["16 plans are paired repeats, not independent training seeds.",
                                  "Counts are node-plan events; each CP preserves original validation nodes.",
                                  "True margins are raw native-head true-minus-best-rival scores; compare within model/CP, not absolute across heads.",
                                  "No recovery shares: four-way interaction Q11-Q10-Q01+Q00 reported explicitly.",
                                  "HGCN has one message layer: upstream-restored is a structural alias of sample.",
                                  "Full and central-trained weights differ; their paired contrast is a training-regime comparison, not a same-weight intervention.",
                                  "Coordinate L2 components are descriptive finite differences, not manifold distance.",
                                  "Hybo scaled head channels are diagnostic accounting only, not native/deployable interventions.",
                                  "Old seed37 Disease baseline and selection dependence remain; no checkpoints were replaced."]}
        write_json(output / "task-a-report.json", report)
        for name, rows in (("plan-conditions", plan_rows), ("plan-contrasts", contrast_rows),
                           ("plan-interactions", interaction_rows), ("plan-representation", rep_rows),
                           ("plan-head-accounting", head_rows), ("plan-low-budget", low_rows),
                           ("plan-training-regime", regime_rows), ("per-cp-conditions", cells),
                           ("per-cp-contrasts", changes), ("per-cp-interactions", interactions),
                           ("per-cp-training-regime", regime_cells), ("three-seed-summary", seed_summary)):
            write_csv(output / (name + ".csv"), rows)
        write_json(output / "consumed-inputs.json", {"input_manifest_sha256": expected["input_manifest_sha256"], "files": loaded})
        write_json(output / "completion.json", {"status": "completed_pending_result_review", "CPs": len(cp_meta),
                   "input_files": len(loaded), "input_bytes": sum(f["bytes"] for f in loaded),
                   "wall_seconds": time.perf_counter() - started, "process_CPU_seconds": time.process_time() - cpu_started,
                   "GPU_seconds": 0, "torch_version": torch.__version__, "torch_threads": 1,
                   "source_sha256": expected["script_sha256"], "input_manifest_sha256": expected["input_manifest_sha256"],
                   "public_input_manifest_sha256": sha(inp.read_bytes()), "sealed_or_test_access": False,
                   "new_forward_or_training": False})
    except Exception as error:
        write_json(output / "failure.json", {"status": "failed_no_retry", "error_type": type(error).__name__,
                   "message": str(error), "loaded_file_count": len(loaded), "wall_seconds": time.perf_counter()-started,
                   "process_CPU_seconds": time.process_time()-cpu_started, "GPU_seconds": 0})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())

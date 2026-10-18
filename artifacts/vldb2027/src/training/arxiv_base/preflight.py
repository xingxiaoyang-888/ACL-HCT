"""Label-free, bounded CUDA cost entry; source preparation is not a release.

Only run_preflight(job, guard_callback) is public. The release-verified worker
must first run the synthetic CUDA builder gate in this SAME process/allocation,
write its bound receipt, and supply the same-PID zero-argument ownership guard.
There is deliberately no unguarded CLI, allocation, retry or checkpoint path.

Task inputs are exactly public graph.npz, ids.json and preparation.json. No
training label loader is called. nodeID % 40 supplies disposable virtual targets
only to exercise the native loss/backward/optimizer work. No loss, predictions,
accuracy or weights are retained. This is not auxiliary supervision or evidence
about training stability/effects. All arms start from the same fresh seed, use
the same three random public batches and paired sampling/dropout keys, and use
the proposal's adaptation LR. Measured steps include all planning/recomputation,
feature indexing/transfer, forward/loss/backward, optimizer, finite guards and
CUDA synchronization. Outer Ops accounting still owns TOTAL CPU/GPU/wall usage.

The fixed first public fit batch (128 IDs, no labels) additionally receives
two warmup and three measured eval/no_grad forward trials per arm. These
fit_forward_proxy rows are separate from train_proxy; they estimate the paid
fit-selection forward workload, not a task score or deployment performance.
Per model: 45 training-proxy steps + 15 fit-forward steps = 60 steps total.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Callable, Mapping

import numpy as np
import torch

from model_builders import build_model, make_optimizer, native_scores, task_loss
from sampling_bridge import UniformPlanningBridge, plan_seed
from training_data import _checked_file, _graph, _ids, _json, _npz


_MODELS = ("hgcn", "hybonet", "sage", "gat")
_ARMS = ("plain", "candidate", "generic")
_SHAPE = {"nodes": 169343, "features": 128, "classes": 40}
_COUNTS = {"train": 90941, "fit": 9932, "gate": 4967, "confirmation": 14900}
_PUBLIC_NAMES = {"graph": "graph.npz", "ids": "ids.json", "preparation": "preparation.json"}
_LOCAL_MODULES = ("preflight", "model_builders", "training_data", "sampling_bridge",
                  "shared_blocks", "core", "geometry", "euclidean_adapters",
                  "hyperbolic_adapters", "test_model_builders_synthetic")
_STARTED = False


def _require(condition, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _binding(value, name: str) -> dict:
    _require(isinstance(value, Mapping) and set(value) == {"path", "sha256"},
             name + " needs an exact path/hash binding")
    path, digest = value["path"], value["sha256"]
    _require(isinstance(path, str) and Path(path).is_absolute(), name + " needs an absolute path")
    _require(isinstance(digest, str) and len(digest) == 64
             and all(char in "0123456789abcdef" for char in digest),
             name + " needs a lowercase SHA256")
    return {"path": str(Path(path).resolve()), "sha256": digest}


def _bundle(value, name: str) -> tuple[Path, set[Path]]:
    _require(isinstance(value, Mapping) and set(value) == {"root", "manifest"},
             name + " needs an exact source root/manifest binding")
    _require(isinstance(value["root"], str) and Path(value["root"]).is_absolute(),
             name + " needs an absolute source root")
    root = Path(value["root"]).resolve()
    _require(root.is_dir(), name + " source root is absent")
    manifest = _json(_binding(value["manifest"], name + " manifest"), name + " manifest")
    _require(isinstance(manifest, dict) and isinstance(manifest.get("files"), list)
             and manifest["files"], name + " source manifest has no files")
    verified = set()
    for record in manifest["files"]:
        _require(isinstance(record, dict) and isinstance(record.get("path"), str),
                 name + " source manifest needs relative file records")
        relative = Path(record["path"])
        target = (root / relative).resolve()
        _require(not relative.is_absolute() and target.is_relative_to(root)
                 and target not in verified, name + " source path escapes or repeats")
        # Source verification never becomes a route to hash/read task arrays or CPs.
        _require(target.suffix.lower() in (".py", ".pyi", ".md", ".json", ".txt",
                                           ".toml", ".cfg", ".ini", ".yaml", ".yml", ".sh")
                 or target.name in ("LICENSE", "NOTICE", ".gitignore"),
                 name + " source manifest contains a non-source artifact")
        binding = _binding({"path": str(target), "sha256": record.get("sha256")}, name + " source")
        with _checked_file(binding, name + " source"):
            pass
        verified.add(target)
    return root, verified


def _job(job: Mapping) -> dict:
    keys = {"schema", "authorized", "purpose", "model", "seed", "device", "config",
            "source", "native_sampler", "native_roots", "public", "prerequisite_gate",
            "batch_offsets", "batch_size", "fanouts", "warmup", "measure", "output_directory"}
    _require(isinstance(job, Mapping) and set(job) == keys, "exact preflight job schema required")
    _require(job["schema"] == "dasfaa_label_free_preflight_job_v1"
             and job["authorized"] is True and job["purpose"] == "label_free_cost_only",
             "this job is not an authorized label-free cost release")
    _require(job["model"] in _MODELS and type(job["seed"]) is int and job["seed"] == 11,
             "one registered model and seed11 required")
    _require(job["device"] == "cuda:0", "only the assigned process-local cuda:0 is allowed")
    for key, expected in (("batch_size", 1024), ("warmup", 2), ("measure", 3)):
        _require(type(job[key]) is int and job[key] == expected, "fixed preflight " + key + " differs")
    for key, expected in (("batch_offsets", [0, 1024, 2048]), ("fanouts", [15, 15, 15])):
        _require(isinstance(job[key], list) and all(type(v) is int for v in job[key])
                 and job[key] == expected, "fixed preflight " + key + " differs")
    _require(isinstance(job["output_directory"], str) and Path(job["output_directory"]).is_absolute(),
             "absolute exclusive output parent required")
    _require(isinstance(job["public"], Mapping) and set(job["public"]) == set(_PUBLIC_NAMES),
             "public firewall permits only graph/ids/preparation")
    public = {}
    for role, basename in _PUBLIC_NAMES.items():
        public[role] = _binding(job["public"][role], role)
        _require(Path(job["public"][role]["path"]).name == basename
                 and Path(public[role]["path"]).name == basename,
                 "public binding resolves to a different file role")
    _require(len({entry["path"] for entry in public.values()}) == 3, "distinct public files required")
    _require(isinstance(job["native_roots"], Mapping)
             and set(job["native_roots"]) == ({job["model"]} if job["model"] in ("hgcn", "hybonet") else set()),
             "only the selected model's native source root is allowed")
    result = copy.deepcopy(dict(job))
    result["public"] = public
    result["config"] = _binding(job["config"], "model config")
    result["prerequisite_gate"] = _binding(job["prerequisite_gate"], "tiny CUDA gate")
    return result


def _sources_and_gate(job: dict) -> tuple[dict, Path]:
    root, source_files = _bundle(job["source"], "new source")
    code = Path(__file__).resolve().parent
    _require(code.is_relative_to(root), "running preflight is outside the released source root")
    for name in _LOCAL_MODULES:
        expected = code / (name + ".py")
        _require(expected in source_files, "new source manifest omits " + name)
        loaded = sys.modules.get(name)
        if loaded is not None:
            _require(Path(loaded.__file__).resolve() == expected,
                     "loaded new module escaped the bound code root: " + name)
    sampler, sampler_files = _bundle(job["native_sampler"], "native sampler")
    _require(all(sampler / name in sampler_files for name in ("blocks.py", "common.py")),
             "bound native sampler must include blocks.py and common.py")
    roots = {}
    for model_key, binding in job["native_roots"].items():
        native, files = _bundle(binding, model_key + " native")
        required = ("models/encoders.py", "models/decoders.py", "layers/hyp_layers.py")
        _require(all(native / name in files for name in required), "native source manifest is incomplete")
        if model_key == "hybonet":
            _require(native / "optim/__init__.py" in files, "native manifold optimizer must be bound")
        roots[model_key] = str(native)
    gate = _json(job["prerequisite_gate"], "tiny CUDA gate")
    _require(gate.get("schema") == "dasfaa_cuda_tiny_gate_v1" and gate.get("status") == "passed"
             and gate.get("model") == job["model"] and gate.get("device") == job["device"]
             and gate.get("source_manifest_sha256") == job["source"]["manifest"]["sha256"],
             "this exact model/device/source has not passed its required CUDA tiny gate")
    native_gate = _json(_binding(gate.get("native_receipt"), "native tiny receipt"), "native tiny receipt")
    _require(native_gate.get("schema") == "dasfaa-cross-model-builder-tiny-gate-v1"
             and native_gate.get("status") == "passed" and native_gate.get("model") == job["model"]
             and native_gate.get("device") == job["device"] and native_gate.get("tests_run") == 1
             and native_gate.get("source_manifest_sha256") == job["source"]["manifest"]["sha256"]
             and native_gate.get("failure_count") == 0 and native_gate.get("error_count") == 0,
             "native tiny builder check did not pass exactly once")
    config = _json(job["config"], "model config")
    _require(config.get("schema") == "dasfaa_cross_model_model_config_proposal_v1"
             and config.get("task", {}).get("dataset") == "ogbn-arxiv"
             and config["task"].get("input_features") == 128 and config["task"].get("classes") == 40,
             "registered arxiv model proposal required")
    shared = config.get("shared_training", {})
    _require(shared.get("dtype") == "float32" and shared.get("batch_size") == 1024
             and shared.get("fanouts_forward_encoding_order") == [15, 15, 15]
             and shared.get("sampler") == "uniform_without_replacement",
             "proposal/public measurement shape or sampler differs")
    config["runtime_native_roots"] = roots
    return config, sampler


def _public_batches(public: dict, offsets: list[int], size: int):
    """Only public metadata and public arrays; never call load_training_data."""
    receipt = _json(public["preparation"], "preparation")
    _require(receipt.get("schema") == "icml_ca_preparation_v1"
             and receipt.get("dataset_version") == "ogbn-arxiv v1"
             and receipt.get("semantic_test_label_reads") == 0
             and receipt.get("confirmation_label_path_exposed_to_development") is False,
             "inherited public preparation receipt differs")
    for role in ("graph", "ids"):
        _require(receipt.get("public", {}).get(_PUBLIC_NAMES[role]) == public[role]["sha256"],
                 "public input differs from inherited preparation hash")
    meta = _json(public["ids"], "ids")
    _require(set(meta) == {"schema", "dataset", "classes", "ids", "batches", "official_test_ids_read"}
             and meta["schema"] == "icml_ca_ids_v1" and meta["dataset"] == "ogbn-arxiv"
             and type(meta["classes"]) is int and meta["classes"] == 40
             and meta["official_test_ids_read"] is False and set(meta["ids"]) == set(_COUNTS)
             and set(meta["batches"]) == {"fit", "gate", "confirmation"},
             "public IDs schema/partition differs")
    split_ids = {name: _ids(values, _SHAPE["nodes"], name + " public IDs")
                 for name, values in meta["ids"].items()}
    combined = np.concatenate(tuple(split_ids.values()))
    _require(all(len(split_ids[name]) == count for name, count in _COUNTS.items())
             and len(np.unique(combined)) == len(combined), "public split counts/overlap differ")
    # Split IDs are public; the corresponding label paths are never consulted.
    order_seed = plan_seed(stage="preflight-batch-order", training_seed=11, epoch=0,
                           repeat=0, batch=0, layer=0, path="main")
    shuffled = np.random.default_rng(order_seed).permutation(split_ids["train"])
    batches = tuple(shuffled[offset:offset + size].copy() for offset in offsets)
    _require(all(len(batch) == size for batch in batches), "public batch is incomplete")
    _require(isinstance(meta["batches"]["fit"], list) and meta["batches"]["fit"],
             "inherited public fit batches are absent")
    fit_batch = _ids(meta["batches"]["fit"][0], _SHAPE["nodes"], "first public fit batch")
    _require(len(fit_batch) == 128 and np.isin(fit_batch, split_ids["fit"]).all(),
             "first public fit batch must be 128 distinct authorized fit IDs")
    graph = _graph(_npz(public["graph"], "graph", {"x", "indptr", "indices", "values", "diagonal"}), _SHAPE)
    return graph, batches, order_seed, fit_batch


def _bridge(root: Path) -> UniformPlanningBridge:
    # Single model/process; preserve common.py's accepted .deps path for DGL.
    for name in ("common", "blocks"):
        if name in sys.modules:
            _require(Path(sys.modules[name].__file__).resolve() == root / (name + ".py"),
                     "native sampler module collision; fresh process required")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    native = importlib.import_module("blocks")
    _require(Path(native.__file__).resolve() == root / "blocks.py"
             and Path(sys.modules["common"].__file__).resolve() == root / "common.py",
             "native sampler import escaped the verified source root")
    return UniformPlanningBridge(native.uniform_edges, native.native_planning_threads)


def _target_hash(targets: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(targets, dtype="<i8").tobytes()).hexdigest()


def _guard_after(guard_callback: Callable[[], None]) -> None:
    """Called from finally: retain the first failure if the guard also fails."""
    original = sys.exc_info()[1]
    try:
        guard_callback()
    except BaseException as guard_error:
        if original is not None:
            # Keep the scientific/operation error as the raised exception and
            # preserve the additional guard failure in its displayed chain.
            raise original from guard_error
        raise


def _trial(job, config, model, optimizer, graph, bridge, targets, *, phase, repeat, batch,
           guard_callback: Callable[[], None], workload="train_proxy") -> dict:
    _require(workload in ("train_proxy", "fit_forward_proxy"), "unknown fixed proxy workload")
    training = workload == "train_proxy"
    _require(model.training is training, "proxy workload/model train/eval mode differs")
    guard_callback()
    device = torch.device(job["device"])
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    start = time.perf_counter()
    try:
        torch.cuda.synchronize(device)  # BOTH boundary sync costs belong to E2E.
        stage = ("label-free-preflight-" if training else "label-free-preflight-fit-forward-") + phase
        dropout_seed = plan_seed(stage=stage + "-dropout", training_seed=11, epoch=0,
                                 repeat=repeat, batch=batch, layer=0, path="main") & ((1 << 63) - 1)
        torch.default_generator.manual_seed(dropout_seed)
        torch.cuda.default_generators[0].manual_seed(dropout_seed)
        if training:
            optimizer.zero_grad(set_to_none=True)
        gradient_scope = torch.enable_grad() if training else torch.no_grad()
        with torch.inference_mode(False), gradient_scope:
            blocks = bridge.plan(graph, targets, job["fanouts"], stage=stage,
                                 training_seed=11, epoch=0, repeat=repeat, batch=batch, path="main")
            # Report CPU metadata: no additional GPU synchronization for counts.
            layer_counts = [block.report() for block in blocks]
            features = graph.x[blocks[0].src_ids].to(device)
            moved = [block.to(device) for block in blocks]
            scores = native_scores(model, features, moved)
            if training:
                virtual_targets = torch.from_numpy(targets % 40).to(device)
                # task_loss checks ALL scores and the scalar loss for finiteness
                # before returning. Reuse those paid checks, without duplicate GPU sync.
                loss = task_loss(job["model"], scores, virtual_targets, config)
                loss.backward()
                gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                _require(gradients and torch.stack([torch.isfinite(value).all() for value in gradients]).all(),
                         "nonfinite or absent proxy gradients; abort without retry")
                optimizer.step()
                _require(torch.stack([torch.isfinite(parameter).all() for parameter in model.parameters()]).all(),
                         "nonfinite proxy parameters; abort without retry")
            else:
                # No true or virtual labels, loss, argmax, optimizer update or
                # retained scores. This is only the registered fit-size forward.
                _require(scores.shape == (len(targets), 40) and torch.isfinite(scores).all(),
                         "invalid fit-forward proxy scores; abort without retry")
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        record = {"workload": workload, "phase": phase, "repeat": repeat, "batch": batch,
                  "target_count": len(targets), "target_ids_sha256": _target_hash(targets),
                  "sampling_stage": stage, "dropout_seed": dropout_seed,
                  "layers": layer_counts, "synchronized_e2e_seconds": elapsed,
                  "baseline_allocated_bytes": baseline_allocated,
                  "baseline_reserved_bytes": baseline_reserved,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                  "finite": True}
        return record
    finally:
        # Guard/writes are outside the stopwatch; failed operations never retry.
        _guard_after(guard_callback)


def run_preflight(job: Mapping, guard_callback: Callable[[], None]) -> dict:
    """Consume one exact label-free job in a fresh process after its tiny gate.

    The zero-argument guard must RAISE on ownership/contention/release failure.
    It is called at every batch/trial boundary outside measured E2E. The worker
    owns its immutable guard implementation and the total native cost record.
    Existing gate receipts may occupy output_directory; its cost/ child must
    not exist. Failure leaves a partial steps.jsonl and never a success summary.
    """
    global _STARTED
    _require(callable(guard_callback), "a bound same-PID CUDA guard callback is mandatory")
    _require(not _STARTED, "one preflight per fresh process; retries are forbidden")
    _STARTED = True
    guard_callback()
    wall_start = time.perf_counter()
    job = _job(job)
    config, sampler_root = _sources_and_gate(job)
    output = Path(job["output_directory"]).resolve() / "cost"
    # Do this before public IO so an existing run fails without re-reading data.
    output.mkdir(parents=True, exist_ok=False)
    graph, batches, order_seed, fit_batch = _public_batches(job["public"], job["batch_offsets"], job["batch_size"])
    bridge = _bridge(sampler_root)
    device = torch.device(job["device"])
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # The worker owns global policy; each native plan temporarily owns one thread.
    _require(torch.get_num_threads() == 7, "worker must establish the released seven host threads")
    records = []
    with (output / "steps.jsonl").open("x", encoding="utf-8") as stream, \
            torch.random.fork_rng(devices=[0]), torch.inference_mode(False), torch.enable_grad():
        for arm in _ARMS:
            guard_callback()
            gc.collect()
            torch.cuda.empty_cache()  # Also discard tiny-gate allocator cache before the first arm.
            model = build_model(job["model"], arm, 11, config, device)
            optimizer = make_optimizer(job["model"], model, config, "adapt")
            model.train()
            torch.cuda.synchronize(device)  # Finish untimed construction; outer wall accounting includes it.
            try:
                for batch_index, targets in enumerate(batches):
                    guard_callback()
                    try:
                        for phase, count in (("warmup", job["warmup"]), ("measure", job["measure"])):
                            for repeat in range(count):
                                record = _trial(job, config, model, optimizer, graph, bridge, targets,
                                                phase=phase, repeat=repeat, batch=batch_index,
                                                guard_callback=guard_callback)
                                record.update({"model": job["model"], "arm": arm})
                                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                                stream.flush()
                                records.append(record)
                    finally:
                        _guard_after(guard_callback)
                model.eval()
                # Match selection's no_grad forward with the current disposable
                # state/optimizer residency. Do not clear caches or gradients to
                # make selection appear cheaper. Counts are source-fixed, not a
                # new job field or a search over fit batches.
                for phase, count in (("warmup", 2), ("measure", 3)):
                    for repeat in range(count):
                        record = _trial(job, config, model, optimizer, graph, bridge, fit_batch,
                                        phase=phase, repeat=repeat, batch=0,
                                        guard_callback=guard_callback, workload="fit_forward_proxy")
                        record.update({"model": job["model"], "arm": arm})
                        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                        stream.flush()
                        records.append(record)
            finally:
                # Only disposable random models existed; no state is serialized.
                del optimizer, model
                gc.collect()
                torch.cuda.empty_cache()
                _guard_after(guard_callback)
    guard_callback()
    settings = config["models"][job["model"]]
    summary = {
        "schema": "dasfaa_label_free_preflight_cost_v1", "status": "complete",
        "model": job["model"], "seed": 11, "device": job["device"],
        "source_manifest_sha256": job["source"]["manifest"]["sha256"],
        "config_sha256": job["config"]["sha256"], "prerequisite_gate": job["prerequisite_gate"],
        "public_bindings": job["public"], "shape": _SHAPE, "fanouts": job["fanouts"],
        "batch_selection": {"method": "private_numpy_permutation_of_public_train_ids",
                            "seed": order_seed, "offsets": job["batch_offsets"],
                            "batches": [{"count": len(ids), "target_ids_sha256": _target_hash(ids)} for ids in batches]},
        "fit_batch_selection": {"method": "bound_public_ids_batches_fit_index_0",
                                "batch_index": 0, "count": 128,
                                "target_ids_sha256": _target_hash(fit_batch),
                                "label_paths_consulted": False},
        "arms": list(_ARMS), "warmup_steps": 24, "measured_steps": 36, "total_steps": 60,
        "workload_steps": {
            "train_proxy": {"batches_per_arm": 3, "target_count_per_batch": 1024,
                            "warmup_per_batch": 2, "measure_per_batch": 3,
                            "warmup_steps": 18, "measured_steps": 27, "total_steps": 45},
            "fit_forward_proxy": {"batches_per_arm": 1, "target_count_per_batch": 128,
                                  "warmup_per_batch": 2, "measure_per_batch": 3,
                                  "warmup_steps": 6, "measured_steps": 9, "total_steps": 15,
                                  "model_mode": "eval", "autograd": "no_grad",
                                  "loss_argmax_or_labels": False},
        },
        "statistics_grouping": "model x arm x workload x phase; never pool training and fit-forward rows",
        "virtual_targets": "train_proxy only: nodeID % 40; disposable cost-only optimizer exercise; none for fit_forward_proxy",
        "real_task_labels_read": False, "checkpoint_read_or_written": False,
        "loss_accuracy_predictions_written": False,
        "optimizer_stage": "adapt", "adapt_learning_rate": settings["stages"]["adapt"]["learning_rate"],
        "pretrain_learning_rate": settings["stages"]["pretrain"]["learning_rate"],
        "torch_host_threads": torch.get_num_threads(), "native_planning_threads": 1,
        "cuda_tf32": False, "per_step_sync_boundaries_charged": True,
        "all_recomputation_charged": True, "finite_guard_cost_included": True,
        "native_scores_finite_checked_all_workloads": True,
        "native_loss_finite_checked_train_proxy_only": True,
        "ownership_guard_outside_step_e2e_but_in_outer_wall_and_allocation": True,
        "timing_mode": "synchronized_end_to_end_only_no_component_estimates",
        "invocation_wall_seconds": time.perf_counter() - wall_start,
        "accounting_owner": "Ops outer native CPU/GPU/allocation/wall accounting, including gate and failures",
        "limits": "Three fixed random batches at fresh weights with virtual targets; not task effects, "
                  "pretraining stability or a runtime/memory guarantee. Adapt LR differs from pretrain; "
                  "its main computational work is represented, but weight trajectories and later costs are not. "
                  "The one fixed public fit batch only supplies a forward-cost proxy; full repeated "
                  "fit selection also has other batches, result reduction and checkpoint-saving work. "
                  "Warmup/cold graph setup and all failed work remain charged separately from measured steps.",
    }
    _require(len(records) == 60
             and sum(row["workload"] == "train_proxy" for row in records) == 45
             and sum(row["workload"] == "fit_forward_proxy" for row in records) == 15,
             "incomplete preflight workloads; no success summary")
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


__all__ = ["run_preflight"]

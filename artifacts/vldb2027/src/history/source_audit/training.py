# SOURCE AUDIT EXPORT: private storage names replaced; science functions retained.
raise RuntimeError("Historical source-audit export; not a public executable entry. Use the documented scalar aggregation route.")
"""18 new native training arms only. No final curves or test scoring here."""
import argparse
import json
from pathlib import Path
import time
from contextlib import contextmanager
from types import MethodType
from contract import CODE, REGISTRATION_SHA, identity, require_release, cell, streams, train_stream, write_json
from runtime import torch, np, configure, seed_all, build_model, make_optimizer, cpu_tree, state_hash, finite, layer_plans, sync
from sampler import Sampler, plan_hash
from inputs import load


def atomic_save(value, path):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def stopped(counter, epoch, args):
    return counter == args.patience and epoch > args.min_epochs


def native_update(net, optimizer, scheduler, args, data, plans):
    net.train()
    optimizer.zero_grad()
    with layer_plans(net, plans):
        h = net.encode(data["features"], data["adj_train_norm"])
        metrics = net.compute_metrics(h, data, "train")
    finite(metrics["loss"], "native training loss")
    metrics["loss"].backward()
    for name, p in net.named_parameters():
        if p.grad is not None:
            finite(p.grad, "gradient " + name)
    norm = None
    if args.grad_clip is not None:
        # HyboNet uses the original global norm clip; HGCN recipe has none.
        norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip))
    optimizer.step()
    scheduler.step()
    for name, p in net.named_parameters():
        finite(p, "updated state " + name)
    return {"loss": float(metrics["loss"].detach()), "acc": float(metrics["acc"]),
            "native_f1": float(metrics["f1"]), "preclip_global_norm": norm}


@torch.no_grad()
def select(net, data, plan_families):
    net.eval()
    rows, predictions, scores, metric_scores = [], [], [], []
    for plans in plan_families:
        with selection_outputs(net) as actual, layer_plans(net, plans):
            h = net.encode(data["features"], data["adj_train_norm"])
            metric = net.compute_metrics(h, data, "val")
        logits = actual["raw_logits"][data["idx_val"]]
        native_output = actual["metric_output"]
        finite(logits, "selection raw logits")
        finite(native_output, "selection actual native metric output")
        finite(metric["loss"], "selection native loss")
        rows.append({"loss": float(metric["loss"]), "acc": float(metric["acc"]), "f1": float(metric["f1"])})
        predictions.append(native_output.argmax(-1).cpu().to(torch.uint8).numpy())
        scores.append(logits.cpu())
        metric_scores.append(native_output.cpu())
    mean = sum(row["f1"] for row in rows) / len(rows)
    return mean, rows, np.stack(predictions), torch.stack(scores), torch.stack(metric_scores)


@contextmanager
def selection_outputs(net):
    """Passively bind both actual native outputs, with no head recomputation."""
    captured = {}
    calls = {"metric_output": 0, "raw_logits": 0}
    saved = []
    try:
        for obj, attr, key in ((net, "decode", "metric_output"), (net.decoder, "decode", "raw_logits")):
            original = getattr(obj, attr)
            previous = obj.__dict__.get(attr)
            saved.append((obj, attr, previous))
            def wrapped(self, *args, original=original, key=key, **kwargs):
                result = original(*args, **kwargs)
                calls[key] += 1
                captured[key] = result.detach()
                return result
            setattr(obj, attr, MethodType(wrapped, obj))
        yield captured
        if calls != {"metric_output": 1, "raw_logits": 1}:
            raise ValueError("Native selection decode skipped or repeated")
    finally:
        for obj, attr, previous in saved:
            if previous is None:
                delattr(obj, attr)
            else:
                setattr(obj, attr, previous)


def memory(device):
    if str(device) == "cpu":
        return {"allocated_bytes": None, "reserved_bytes": None, "peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {"allocated_bytes": torch.cuda.memory_allocated(device), "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def run(cli):
    job = identity("train", cli.model, cli.dataset, cli.regime, cli.seed)
    require_release(cli.release, job, "training-source-manifest-v2.json")
    output = Path(cli.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "job.json", {"job": job, "registration_sha256": REGISTRATION_SHA, "test_scoring": False})
    try:
        configure()
        task_started = time.perf_counter()
        data, public, input_record = load(cli.dataset, cli.device)
        config = cell(cli.model, cli.dataset)
        # Sampling starts after construction and never consumes global RNG.
        seed_all(cli.seed)
        net, args = build_model(cli.model, cli.dataset, cli.seed, cli.device, data["features"], 4 if cli.dataset == "Airport" else 2)
        initial = cpu_tree(net.state_dict())
        initial_sha = state_hash(initial)
        initial_serial_start = time.perf_counter()
        atomic_save({"model_state": initial, "state_sha256": initial_sha}, output / "initial-state.pt")
        initial_serial_seconds = time.perf_counter() - initial_serial_start
        write_json(output / "effective-args.json", vars(args))
        optimizer = make_optimizer(net, cli.model, args)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_reduce_freq, gamma=args.gamma)
        sampler = Sampler(data["adj_train_norm"])
        layers = config["native_message_layers"]
        full = data["adj_train_norm"]
        selector_prepare_start = time.perf_counter()
        if cli.regime == "full":
            selectors = [[full] * layers]
            selection_streams = []
        else:
            selection_streams = list(streams(cli.dataset, "selection"))
            selector_cpu = [sampler.plans(config["matched_training_budget"], s, layers, cli.dataset, "selection") for s in selection_streams]
            selector_prepare_seconds = time.perf_counter() - selector_prepare_start
            sync(cli.device)
            selector_transfer_start = time.perf_counter()
            selectors = [[p.to(cli.device) for p in plans] for plans in selector_cpu]
            sync(cli.device)
            selector_transfer_seconds = time.perf_counter() - selector_transfer_start
        if cli.regime == "full":
            selector_prepare_seconds = time.perf_counter() - selector_prepare_start
            selector_transfer_seconds = 0.0
        write_json(output / "selection-plans.json", {"streams": selection_streams, "plan_hashes": [[plan_hash(p) for p in ps] for ps in selectors],
                   "native_f1_average": net.f1_average, "strict_maximum_earliest_tie": True})
        best, best_epoch, counter = -1.0, None, 0
        started = time.perf_counter()
        common_residency = memory(cli.device)
        if str(cli.device) != "cpu":
            torch.cuda.reset_peak_memory_stats(cli.device)
        totals = {"plan_preparation_cpu_seconds": 0.0, "plan_transfer_synchronized_wall_seconds": 0.0,
                  "native_update_synchronized_wall_seconds": 0.0, "validation_synchronized_wall_seconds": 0.0,
                  "checkpoint_serialization_wall_seconds": 0.0, "log_serialization_wall_seconds": 0.0}
        with (output / "epochs.jsonl").open("w", encoding="utf-8") as log, (output / "selection-predictions.uint8").open("wb") as raw_pred:
            for epoch in range(args.epochs):
                stream = None if cli.regime == "full" else train_stream(cli.dataset, cli.seed, epoch)
                prepare_start = time.perf_counter()
                plans_cpu = [sampler.full] * layers if stream is None else sampler.plans(config["matched_training_budget"], stream, layers, cli.dataset, "training")
                preparation = time.perf_counter() - prepare_start
                sync(cli.device)
                transfer_start = time.perf_counter()
                plans = [full] * layers if stream is None else [p.to(cli.device) for p in plans_cpu]
                sync(cli.device)
                transfer = time.perf_counter() - transfer_start
                update_start = time.perf_counter()
                trained = native_update(net, optimizer, scheduler, args, data, plans)
                sync(cli.device)
                update_seconds = time.perf_counter() - update_start
                validation_start = time.perf_counter()
                selected, per_plan, pred, val_scores, val_metric_scores = select(net, data, selectors)
                sync(cli.device)
                validation_seconds = time.perf_counter() - validation_start
                improved = selected > best
                counter = 0 if improved else counter + 1
                serialization_seconds = 0.0
                if improved:
                    best, best_epoch = selected, epoch + 1
                    checkpoint_start = time.perf_counter()
                    atomic_save({"schema": 1, "job": job, "seed": cli.seed, "epoch": best_epoch,
                                 "model_state": cpu_tree(net.state_dict()), "optimizer_state": cpu_tree(optimizer.state_dict()),
                                 "scheduler_state": scheduler.state_dict(), "args": vars(args), "initial_state_sha256": initial_sha,
                                 "selection_mean_native_f1": best, "selection_per_plan": per_plan,
                                 "selection_logits": val_scores, "selection_actual_metric_output": val_metric_scores,
                                 "validation_ids": data["idx_val"],
                                 "registration_sha256": REGISTRATION_SHA, "input_record": input_record,
                                 "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                                         "cuda": torch.cuda.get_rng_state_all() if str(cli.device) != "cpu" else []},
                                 "test_evaluation": False}, output / "best-validation.pt")
                    serialization_seconds = time.perf_counter() - checkpoint_start
                epoch_timing = dict(zip(list(totals)[:-1], (preparation, transfer, update_seconds, validation_seconds, serialization_seconds)))
                for name, seconds in epoch_timing.items():
                    totals[name] += seconds
                row = {"epoch": epoch + 1, "zero_based_epoch": epoch, "training_stream": stream, "training": trained,
                       "training_plan_hashes": [plan_hash(p) for p in plans_cpu], "phase_timing": epoch_timing,
                       "selection_mean_native_f1": selected, "selection_per_plan": per_plan, "improved": improved, "counter": counter}
                log_start = time.perf_counter()
                raw_pred.write(pred.tobytes(order="C"))
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                totals["log_serialization_wall_seconds"] += time.perf_counter() - log_start
                if epoch == 0 or improved and (epoch + 1) % args.log_freq == 0:
                    print(json.dumps({"epoch": epoch + 1, "selection_native_f1": selected, "best_epoch": best_epoch}), flush=True)
                if stopped(counter, epoch, args):
                    break
        write_json(output / "completion.json", {"status": "completed", "job": job, "epochs_executed": epoch + 1,
                   "best_epoch": best_epoch, "best_selection_native_f1": best, "initial_state_sha256": initial_sha,
                   "training_loop_wall_seconds": time.perf_counter() - started, "total_task_wall_seconds": time.perf_counter() - task_started,
                   "phase_totals": totals, "initial_state_serialization_wall_seconds": initial_serial_seconds,
                   "selection_plan_preparation_cpu_seconds": selector_prepare_seconds,
                   "selection_plan_transfer_synchronized_wall_seconds": selector_transfer_seconds,
                   "common_residency_before_loop": common_residency, "training_memory_peak": memory(cli.device),
                   "timing_interpretation": "Phase totals are disjoint synchronized wall intervals. Sum excludes input/constructor/optimizer setup, plan hashing and audit overhead. Update includes native CPU metrics and finite checks; validation includes actual output CPU copies. Loop/total wall overlap phases and must not be added. Training costs are diagnostics, not deployment timings.",
                   "test_scoring": False,
                   "selection_prediction_shape": [epoch + 1, len(selectors), len(data["idx_val"])],
                   "stop": "native_patience" if stopped(counter, epoch, args) else "native_epoch_limit"})
    except Exception as error:
        write_json(output / "failure.json", {"status": "failed_no_retry", "type": type(error).__name__, "message": str(error), "job": job})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["HyboNet", "HGCN"], required=True)
    parser.add_argument("--dataset", choices=["Airport", "Disease-NC"], required=True)
    parser.add_argument("--regime", choices=["full", "central_matched"], required=True)
    parser.add_argument("--seed", type=int, choices=[11, 23, 37], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--release", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

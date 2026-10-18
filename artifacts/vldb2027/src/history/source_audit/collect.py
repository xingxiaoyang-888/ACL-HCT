# SOURCE AUDIT EXPORT: private storage names replaced; science functions retained.
raise RuntimeError("Historical source-audit export; not a public executable entry. Use the documented scalar aggregation route.")
"""24 fixed CP raw curves and minimal fit interventions; no predictor fitting."""
import argparse
from pathlib import Path
import time
from contract import CODE, REGISTRATION_SHA, identity, require_release, cell, streams, sha, write_json
from runtime import torch, configure, cpu_tree, state_hash, sync
from inputs import load
from sampler import Sampler, plan_hash
from checkpoints import load_checkpoint
from trace import capture
from views import public_graph, oracle_payload, sample_payload
from interventions import fourway, hybo_channels
from training import atomic_save, memory


def outcomes(full_capture,sample_capture,data):
    """Raw outcomes only. Never summarize evaluation effects before a lock."""
    ids = data["idx_val"]
    labels = data["labels"].detach().cpu()[ids]
    if bool((labels < 0).any()):
        raise ValueError("Protected labels requested")
    f,s = full_capture["native_metric_output"][ids],sample_capture["native_metric_output"][ids]
    fp,sp = f.argmax(-1),s.argmax(-1)
    def margin(scores):
        rivals = scores.clone()
        rivals.scatter_(1,labels[:,None],-torch.inf)
        rival_score,rival_id = rivals.max(-1)
        return scores.gather(1,labels[:,None]).flatten()-rival_score,rival_id
    fm,fr = margin(full_capture["native_logits"][ids])
    sm,sr = margin(sample_capture["native_logits"][ids])
    fc,sc = fp.eq(labels),sp.eq(labels)
    harm,benefit = fc & ~sc,~fc & sc
    return {"schema": "validation_raw_targets_v1","validation_ids": torch.tensor(ids),"validation_labels": labels,
            "full_native_metric_output": f,"sample_native_metric_output": s,
            "full_native_logits": full_capture["native_logits"][ids],"sample_native_logits": sample_capture["native_logits"][ids],
            "full_predictions": fp,"sample_predictions": sp,"harm": harm,"benefit": benefit,
            "event_class_H_B_neither": torch.where(harm,0,torch.where(benefit,1,2)),
            "mismatch": fp != sp,"full_true_margin": fm,"sample_true_margin": sm,"signed_true_margin_change": sm-fm,
            "full_true_rival": fr,"sample_true_rival": sr,"test_scoring": False}


def strip_first_inputs(record,full_capture):
    # Both frozen models have deterministic identical first inputs during eval.
    first = record["layers"][0]
    base = full_capture["layers"][0]
    names = ["linear_input"]
    if "affine_input_after_activation_dropout" in first:
        names.append("affine_input_after_activation_dropout")
    for name in names:
        if not torch.equal(first[name],base[name]):
            raise ValueError("Deterministic first input unexpectedly changed")
        del first[name]
    record["first_input_storage"] = "lossless alias of raw/full-state.pt layer0 inputs; not an extra sampled view"


def measured_capture(net,model,dataset,data,sampler,budget,stream,purpose,device,layers):
    prepare_start = time.perf_counter()
    cpu_plans = [sampler.full]*layers if budget == "full" else sampler.plans(budget,stream,layers,dataset,purpose)
    preparation = time.perf_counter()-prepare_start
    sync(device)
    transfer_start = time.perf_counter()
    plans = [data["adj_train_norm"]]*layers if budget == "full" else [p.to(device) for p in cpu_plans]
    sync(device)
    transfer = time.perf_counter()-transfer_start
    common = memory(device)
    if str(device) != "cpu":
        torch.cuda.reset_peak_memory_stats(device)
    record = capture(net,model,data["features"],data["adj_train_norm"],plans,device)
    record["timing"].update(plan_preparation_cpu_seconds=preparation,plan_transfer_synchronized_wall_seconds=transfer,
                            common_residency_before_capture=common,capture_peak=memory(device))
    return record,plans,cpu_plans


def run(cli):
    checkpoint_sha = sha(cli.checkpoint)
    job = identity("capture",cli.model,cli.dataset,cli.regime,cli.seed,checkpoint_sha)
    require_release(cli.release,job,"capture-source-manifest-v2.json")
    source_bindings = {"registration_sha256":REGISTRATION_SHA,"source_manifest_sha256":sha(CODE / "capture-source-manifest-v2.json"),
                       "root_release_sha256":sha(cli.release)}
    output = Path(cli.output)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output / "job.json",{"job": job,**source_bindings,"evaluation_outcomes_sealed": True})
    started = time.perf_counter()
    archive = []
    def save(value,relative):
        path = output / relative
        path.parent.mkdir(parents=True,exist_ok=True)
        begin = time.perf_counter()
        atomic_save(value,path)
        archive.append({"path": relative,"sha256": sha(path),"bytes": path.stat().st_size,
                        "serialization_wall_seconds": time.perf_counter()-begin})
    try:
        configure()
        data,public,inputs = load(cli.dataset,cli.device)
        config = cell(cli.model,cli.dataset)
        layers = config["native_message_layers"]
        net,args,weights = load_checkpoint(cli.checkpoint,job,cli.device,data["features"])
        expected_state = state_hash(net.state_dict())
        save(weights,"label-free-weights.pt")
        public = public_graph(data["adj_train_norm"],data["idx_train"],data["idx_val"])
        sampler = Sampler(data["adj_train_norm"])
        full_record,full_plans,full_cpu = measured_capture(net,cli.model,cli.dataset,data,sampler,"full",None,"full",cli.device,layers)
        save(full_record,"raw/full-state.pt")
        save(oracle_payload(full_record,weights,public),"views/O-preplan-full.pt")
        count = 0
        restorations = 0
        for purpose,budgets in (("fit",config["budgets"][:2]),("evaluation",config["budgets"][:3])):
            for stream in streams(cli.dataset,purpose):
                for budget in budgets:
                    record,plans,cpu_plans = measured_capture(net,cli.model,cli.dataset,data,sampler,budget,stream,purpose,cli.device,layers)
                    key = f"stream{stream}-k{budget}"
                    save(sample_payload(record,cpu_plans,sampler.full,weights,public),f"views/S/{purpose}/{key}.pt")
                    target_prefix = "fit/outcomes" if purpose == "fit" else "sealed/evaluation/outcomes"
                    save(outcomes(full_record,record,data),f"{target_prefix}/{key}.pt")
                    if purpose == "fit" and budget == config["matched_training_budget"]:
                        sync(cli.device)
                        continuation_start = time.perf_counter()
                        intervention = fourway(net,cli.model,full_record,record,plans,data["adj_train_norm"],cli.device)
                        sync(cli.device)
                        intervention["timing"] = {"cheap_final_continuations_synchronized_wall_seconds": time.perf_counter()-continuation_start,
                                                 "diagnostic_only_not_deployment_cost": True}
                        if cli.model == "HyboNet":
                            restored_plans = list(plans)
                            restored_plans[0] = data["adj_train_norm"]
                            if str(cli.device) != "cpu":
                                torch.cuda.reset_peak_memory_stats(cli.device)
                            restoration_residency = memory(cli.device)
                            restored = capture(net,cli.model,data["features"],data["adj_train_norm"],restored_plans,cli.device)
                            restored["timing"].update(common_residency_before_capture=restoration_residency,capture_peak=memory(cli.device))
                            strip_first_inputs(restored,full_record)
                            save(restored,f"fit/interventions/{key}-first-full-raw.pt")
                            save(outcomes(full_record,restored,data),f"fit/interventions/{key}-first-full-targets.pt")
                            intervention["first_layer_full_restoration_raw"] = f"{key}-first-full-raw.pt"
                            intervention["additional_encoder_passes"] = 1
                            oracle_start = time.perf_counter()
                            intervention["head_oracle_channels"] = hybo_channels(net,full_record,record)
                            intervention["timing"]["all_class_head_oracle_CPU_accounting_wall_seconds"] = time.perf_counter()-oracle_start
                            restorations += 1
                        save(intervention,f"fit/interventions/{key}-final-fourway.pt")
                    # Raw diagnostic tape is privileged: predictive feature
                    # construction uses only the separate O/S view interfaces.
                    record["actual_plans"] = cpu_plans
                    strip_first_inputs(record,full_record)
                    save(record,f"raw/{purpose}/{key}.pt")
                    count += 1
                print({"completed_core_captures": count,"purpose": purpose,"sealed_evaluation_metrics_printed": False},flush=True)
        if count != 80 or restorations != (16 if cli.model == "HyboNet" else 0):
            raise ValueError("Frozen capture workload mismatch")
        if state_hash(net.state_dict()) != expected_state:
            raise ValueError("Fixed model state mutated")
        write_json(output / "raw-archive-manifest.json",{"job": job,**source_bindings,"files": archive,"sealed_holdout": "sealed/evaluation/outcomes",
                   "lock_rule": "Root must hash-lock fit/scalers/coefficients and heldout predictions before reading heldout target tables",
                   "feature_inputs": "views only; raw and interventions are diagnostic privileged data"})
        write_json(output / "completion.json",{"status": "completed_raw_outcomes_sealed","job": job,"full_state_encoder_passes": 1,
                   "core_sampled_encoder_passes": count,"additional_first_restoration_encoder_passes": restorations,
                   "HGCN_alias_encoder_passes": 0,"model_state_sha256": expected_state,"input_record": inputs,
                   "wall_seconds": time.perf_counter()-started,"archive_bytes": sum(x["bytes"] for x in archive),
                   "aggregate_heldout_effects": "not computed or printed","test_scoring": False})
    except Exception as error:
        write_json(output / "failure.json",{"status": "failed_no_retry","job": job,"type": type(error).__name__,"message": str(error),
                   "completed_files": archive,"aggregate_heldout_effects": "not computed or printed"})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",choices=["HyboNet","HGCN"],required=True)
    parser.add_argument("--dataset",choices=["Airport","Disease-NC"],required=True)
    parser.add_argument("--regime",choices=["full","central_matched"],required=True)
    parser.add_argument("--seed",type=int,choices=[11,23,37],required=True)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--release",required=True)
    parser.add_argument("--output",required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

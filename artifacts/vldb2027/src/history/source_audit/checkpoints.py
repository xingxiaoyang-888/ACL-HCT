# SOURCE AUDIT EXPORT: private storage names replaced; science functions retained.
raise RuntimeError("Historical source-audit export; not a public executable entry. Use the documented scalar aggregation route.")
"""Fixed CP import and physically label-free weights export."""
from pathlib import Path
from contract import CODE, PROJECT, REGISTRATION_SHA, read_json, sha
from runtime import torch, build_model, seed_all, cpu_tree, state_hash, finite


def load_checkpoint(path, job, device, features):
    path = Path(path)
    if sha(path) != job["checkpoint_sha256"]:
        raise ValueError("Released checkpoint bytes differ")
    inherited = job["model"] == "HyboNet" and job["dataset"] == "Airport"
    if inherited:
        expected_budget = "full" if job["regime"] == "full" else 8
        records = read_json(CODE / "checkpoint-reuse-inventory.json")["records"]
        matches = [r for r in records if r.get("directly_reusable") and r["seed"] == job["seed"] and r["trained_budget"] == expected_budget
                   and r["sha256"] == job["checkpoint_sha256"]]
        if len(matches) != 1:
            raise ValueError("Checkpoint is not one of the six inherited states")
    cp = torch.load(path,map_location="cpu")
    if cp.get("test_evaluation") is not False or cp.get("seed") != job["seed"]:
        raise ValueError("Checkpoint guard/provenance differs")
    if not inherited:
        expected = {k:v for k,v in job.items() if k != "checkpoint_sha256"}
        expected["action"] = "train"
        if cp.get("job") != expected or cp.get("registration_sha256") != REGISTRATION_SHA:
            raise ValueError("New checkpoint did not come from this exact native training arm")
    seed_all(job["seed"])
    # Constructor args come from the frozen configuration, with raw feature
    # dimension reset, never from the mutated saved Lorentz feat_dim.
    net,args = build_model(job["model"],job["dataset"],job["seed"],device,features,4 if job["dataset"] == "Airport" else 2)
    net.load_state_dict(cp["model_state"],strict=True)
    for name,p in net.named_parameters():
        finite(p,"fixed checkpoint parameter "+name)
        p.requires_grad_(False)
    net.eval()
    weights = {"schema": "label_free_fixed_weights_v1", "model": job["model"], "dataset": job["dataset"],
               "regime": job["regime"], "seed": job["seed"], "checkpoint_sha256": job["checkpoint_sha256"],
               "model_state": cpu_tree(net.state_dict()),"state_sha256": state_hash(net.state_dict()),
               "registration_sha256": REGISTRATION_SHA}
    return net,args,weights


def load_weights(path,job,device,features):
    weights = torch.load(path,map_location="cpu")
    expected_keys = {"schema","model","dataset","regime","seed","checkpoint_sha256","model_state","state_sha256","registration_sha256"}
    if set(weights) != expected_keys or weights["schema"] != "label_free_fixed_weights_v1" or weights["registration_sha256"] != REGISTRATION_SHA:
        raise ValueError("Cost entry accepts only the physically label-free weights schema")
    for name in ("model","dataset","regime","seed","checkpoint_sha256"):
        if weights[name] != job[name]:
            raise ValueError("Fixed weights/job differs")
    if state_hash(weights["model_state"]) != weights["state_sha256"]:
        raise ValueError("Fixed state hash differs")
    seed_all(job["seed"])
    net,args = build_model(job["model"],job["dataset"],job["seed"],device,features,4 if job["dataset"] == "Airport" else 2)
    net.load_state_dict(weights["model_state"],strict=True)
    for p in net.parameters():
        p.requires_grad_(False)
    net.eval()
    return net,args

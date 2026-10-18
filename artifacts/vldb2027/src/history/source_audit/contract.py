# SOURCE AUDIT EXPORT: private storage names replaced; science functions retained.
raise RuntimeError("Historical source-audit export; not a public executable entry. Use the documented scalar aggregation route.")
"""Frozen protocol bindings; imports do not construct or score models."""
from pathlib import Path
import hashlib
import json

CODE = Path(__file__).resolve().parents[1]
ROUND = CODE.parent
PROJECT = ROUND.parents[1]
REGISTRATION_SHA = "565e6e2cdc4743f4721e7d3ae181f3a2e4ae02041bee1cb58416d982482b4d3e"
SEEDS = (11, 23, 37)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path, lf=False):
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n") if lf else raw).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def registration():
    path = ROUND / "registration.json"
    if sha(path) != REGISTRATION_SHA:
        raise ValueError("Frozen registration bytes differ")
    reg = read_json(path)
    for entry in reg["bound_files"]:
        if sha(PROJECT / entry["path"]) != entry["sha256"]:
            raise ValueError("Frozen registration input differs: " + entry["path"])
    return reg


def cell(model, dataset):
    matches = [x for x in registration()["matrix"] if x["model"] == model and x["dataset"] == dataset]
    if len(matches) != 1:
        raise ValueError("Unregistered combination")
    return matches[0]


def identity(action, model, dataset, regime, seed, checkpoint_sha=None):
    c = cell(model, dataset)
    if seed not in c["seeds"] or regime not in c["regimes"]:
        raise ValueError("Unregistered seed/regime")
    if action == "train" and c["inherited_checkpoints"]:
        raise ValueError("Inherited HyboNet/Airport checkpoints must not be retrained")
    if action not in ("train", "capture", "cost"):
        raise ValueError("Unregistered action")
    job = {"action": action, "model": model, "dataset": dataset, "regime": regime, "seed": seed}
    if action != "train":
        if not checkpoint_sha or len(checkpoint_sha) != 64:
            raise ValueError("Fixed checkpoint hash required")
        job["checkpoint_sha256"] = checkpoint_sha
    return job


def verify_manifest(name):
    path = CODE / name
    manifest = read_json(path)
    for entry in manifest["files"]:
        if sha(PROJECT / entry["path"], entry.get("normalization") == "LF") != entry["sha256"]:
            raise ValueError("Scientific executable/input bytes differ: " + entry["path"])
    return sha(path)


def require_release(path, job, manifest_name):
    # Even real CPU scoring needs an exact release; metadata/synthetic checks do not.
    rel = read_json(path)
    if rel.get("status") != "released" or rel.get("registration_sha256") != REGISTRATION_SHA:
        raise ValueError("No exact Root release")
    if rel.get("source_manifest_sha256") != verify_manifest(manifest_name):
        raise ValueError("Release/source binding differs")
    if job not in rel.get("jobs", []):
        raise ValueError("Job absent from exact release")
    return rel


def graph_id(dataset):
    return {"Airport": 0, "Disease-NC": 1}[dataset]


def train_stream(dataset, seed, epoch):
    return 20_000_000 + 1_000_000 * graph_id(dataset) + 10_000 * seed + epoch


def streams(dataset, purpose):
    return range({("Airport", "selection"): 83000, ("Disease-NC", "selection"): 183000,
                  ("Airport", "fit"): 910000, ("Airport", "evaluation"): 920000,
                  ("Disease-NC", "fit"): 930000, ("Disease-NC", "evaluation"): 940000,
                  ("Airport", "cost"): 950000, ("Disease-NC", "cost"): 960000}[(dataset, purpose)],
                 {("Airport", "selection"): 83004, ("Disease-NC", "selection"): 183004,
                  ("Airport", "fit"): 910016, ("Airport", "evaluation"): 920016,
                  ("Disease-NC", "fit"): 930016, ("Disease-NC", "evaluation"): 940016,
                  ("Airport", "cost"): 950023, ("Disease-NC", "cost"): 960023}[(dataset, purpose)])

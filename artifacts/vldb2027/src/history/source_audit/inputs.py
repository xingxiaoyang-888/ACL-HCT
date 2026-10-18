# SOURCE AUDIT EXPORT: private storage names replaced; science functions retained.
raise RuntimeError("Historical source-audit export; not a public executable entry. Use the documented scalar aggregation route.")
"""Only guarded development inputs; public cost artifacts contain no labels."""
from contract import CODE, PROJECT, sha, read_json, write_json
from runtime import torch, np, native_api
from sampler import plan_hash

PATHS = {"Airport": CODE / "inputs/airport-guarded-v1/shared-runtime.pt",
         "Disease-NC": CODE / "inputs/disease-guarded-v1/shared-runtime.pt"}


def tensor_sha(tensor):
    import hashlib
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def prepare():
    # One new guarded copy of the already accepted Airport input, no split search.
    airport = PATHS["Airport"]
    if airport.exists():
        raise FileExistsError("Prepared input is immutable")
    old = PROJECT / "public-inputs/history/hybonet-origin"
    raw = old / "operations/data/airport/airport.p"
    if sha(raw) != "ab16b04cea3666e87dbd8a5bff3de68da61330f4604fa21082f7a17e806e64d8":
        raise ValueError("Accepted Airport source changed")
    ids = read_json(old / "code/split-manifest.json")["ids"]
    native = native_api("HyboNet").data
    adj, features, labels = native.load_data_airport("airport", str(raw.parent), return_label=True)
    adj.sort_indices()
    norm, features = native.process(adj, features, normalize_adj=1, normalize_feats=1)
    features = native.augment(adj, features)
    if plan_hash(norm) != "a4f7762bc7fde105f63bb8a9d4f8ba01b2509451654912fd80dde7d645e44e4c":
        raise ValueError("Accepted Airport normalized operator changed")
    labels = torch.from_numpy(labels.copy()).long()
    labels[ids["test"]] = -1
    data = {"features": features, "adj_train_norm": norm.coalesce(), "labels": labels,
            "idx_train": ids["train"], "idx_val": ids["val"]}
    rows, cols = norm.coalesce().indices().numpy()
    degrees = np.bincount(rows[rows != cols], minlength=features.shape[0])
    payload = {"data": data, "public_graph": {"nonself_degree": torch.from_numpy(degrees)},
               "metadata": {"dataset": "Airport", "n_classes": 4, "frozen_split_sha256": sha(old / "code/split-manifest.json"),
                            "raw_source_sha256": sha(raw), "test_scoring": False}}
    airport.parent.mkdir(parents=True)
    torch.save(payload, airport)
    records = []
    for dataset in PATHS:
        shared = torch.load(PATHS[dataset], map_location="cpu")
        validate(shared)
        public = CODE / "inputs" / ("airport-guarded-v1" if dataset == "Airport" else "disease-guarded-v1") / "public-runtime.pt"
        if public.exists():
            raise FileExistsError("Public input already prepared")
        d = shared["data"]
        # No labels, test IDs or selection scores are serialized here.
        torch.save({"features": d["features"], "adj_train_norm": d["adj_train_norm"],
                    "idx_train": d["idx_train"], "idx_val": d["idx_val"],
                    "nonself_degree": shared["public_graph"]["nonself_degree"],
                    "n_classes": 4 if dataset == "Airport" else 2}, public)
        records.append({"dataset": dataset, "shared_path": PATHS[dataset].relative_to(PROJECT).as_posix(),
                        "shared_sha256": sha(PATHS[dataset]), "public_path": public.relative_to(PROJECT).as_posix(),
                        "public_sha256": sha(public), "feature_sha256": tensor_sha(d["features"]),
                        "full_plan_sha256": plan_hash(d["adj_train_norm"]),
                        "train_count": len(d["idx_train"]), "validation_count": len(d["idx_val"]),
                        "public_contains_labels": False, "runtime_contains_idx_test": False})
    write_json(CODE / "scientific-inputs-v1.json", {"status": "prepared_no_model_forward", "records": records})


def validate(payload):
    data = payload["data"]
    if set(data) != {"features", "adj_train_norm", "labels", "idx_train", "idx_val"}:
        raise ValueError("Unexpected guarded input field")
    train, val = set(data["idx_train"]), set(data["idx_val"])
    if train & val or len(train) != len(data["idx_train"]) or len(val) != len(data["idx_val"]):
        raise ValueError("Invalid development partition")
    allowed = torch.zeros(len(data["labels"]), dtype=torch.bool)
    allowed[list(train | val)] = True
    if not bool((data["labels"][~allowed] == -1).all()) or bool((data["labels"][allowed] < 0).any()):
        raise ValueError("Protected-label guard failed")
    if data["features"].dtype != torch.float32 or data["adj_train_norm"].dtype != torch.float32:
        raise ValueError("Wrong native input precision")


def load(dataset, device):
    records = read_json(CODE / "scientific-inputs-v1.json")["records"]
    record = next(x for x in records if x["dataset"] == dataset)
    path = PROJECT / record["shared_path"]
    if sha(path) != record["shared_sha256"]:
        raise ValueError("Guarded input changed")
    payload = torch.load(path, map_location="cpu")
    validate(payload)
    data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in payload["data"].items()}
    return data, payload["public_graph"], record


def load_public(dataset, device):
    record = next(x for x in read_json(CODE / "scientific-inputs-v1.json")["records"] if x["dataset"] == dataset)
    path = PROJECT / record["public_path"]
    if sha(path) != record["public_sha256"]:
        raise ValueError("Public input changed")
    d = torch.load(path, map_location="cpu")
    if set(d) != {"features", "adj_train_norm", "idx_train", "idx_val", "nonself_degree", "n_classes"}:
        raise ValueError("Cost input must be physically label-free")
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in d.items()}


if __name__ == "__main__":
    from contract import registration
    registration()
    prepare()

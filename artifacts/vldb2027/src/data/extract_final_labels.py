"""Once-only official-test target/label materialization after science freeze.

Pure source preparation is safe: no label files or model modules are opened on
import. Only designated Ops may call run_job under an exact new Root release.
Original quarantine/sealed/raw files remain read-only and in place. This does
not score predictions. PubMed test was historically exposed; no unseen-data
claim is made. The new SAGE protocol is prospectively fixed before extraction.
"""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import re
import time
import zipfile


ROUND = "vldb-sampling-mechanisms-v1"
HERE = Path(__file__).resolve().parent
TARGET_PREFIX = ROUND + ":official-test-target:"
ARXIV_TEST = "arxiv/split/time/test.csv.gz"
ARXIV_LABELS = "arxiv/raw/node-label.csv.gz"
EXPECTED_INPUTS = {
    "arxiv_archive", "arxiv_public_ids", "arxiv_preparation",
    "pubmed_test_index", "pubmed_ty", "pubmed_public_ids",
    "pubmed_preparation", "pubmed_source_manifest",
}


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def bound(ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "exact path/hash reference required")
    require(isinstance(ref["path"], str) and Path(ref["path"]).is_absolute() and is_sha(ref["sha256"]), "absolute input/hash required")
    path = Path(ref["path"]).resolve()
    require(file_sha(path) == ref["sha256"], "input byte identity changed: " + str(path))
    return path


def bound_json(ref):
    return json.loads(bound(ref).read_text(encoding="utf-8-sig"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def expected_cells():
    return {(dataset, model, seed)
            for dataset, models in (("ogbn-arxiv", ("hgcn", "hybonet", "sage", "gat")), ("PubMed", ("sage",)))
            for model in models for seed in (11, 23, 37)}


def check_science_freeze(freeze, source_sha):
    require(set(freeze) == {"schema", "round", "plan_sha256", "evaluation_source_manifest_sha256",
                           "label_extraction_source_manifest_sha256", "entries", "allow_test_once"}, "unexpected science freeze fields")
    require(freeze["schema"] == "vldb_sampling_science_freeze_v1" and freeze["round"] == ROUND
            and freeze["allow_test_once"] is True, "complete once-only science freeze required")
    require(freeze["label_extraction_source_manifest_sha256"] == source_sha
            and is_sha(freeze["plan_sha256"]) and is_sha(freeze["evaluation_source_manifest_sha256"]), "science/source freeze differs")
    require(isinstance(freeze["entries"], list) and len(freeze["entries"]) == 15, "all 15 checkpoints required before any label extraction")
    cells, locks = set(), {}
    for entry in freeze["entries"]:
        require(set(entry) == {"dataset", "model", "seed", "weights_sha256", "budget_lock_sha256"}, "unexpected freeze cell")
        key = (entry["dataset"], entry["model"], entry["seed"])
        require(type(entry["seed"]) is int and key not in cells and is_sha(entry["weights_sha256"])
                and is_sha(entry["budget_lock_sha256"]), "invalid or duplicate frozen cell")
        cells.add(key)
        group = key[:2]
        if group in locks:
            require(locks[group] == entry["budget_lock_sha256"], "all seeds must inherit one model/dataset budget lock")
        locks[group] = entry["budget_lock_sha256"]
    require(cells == expected_cells(), "frozen 15-cell matrix differs")


def public_ids_check(meta, dataset, ids):
    require(meta["dataset"] == dataset, "public dataset differs")
    require(len(ids) == len(set(ids)), "duplicate official test IDs")
    existing = {int(v) for key, values in meta["ids"].items() if key != "test" for v in values}
    require(not existing.intersection(ids), "official test overlaps prior train/development IDs")
    if dataset == "PubMed":
        require(set(meta["ids"]["test"]) == set(ids), "prepared public PubMed test IDs differ")


def target_manifest(dataset, original_ids, provenance):
    ids = sorted(original_ids, key=lambda node: (hashlib.sha256((TARGET_PREFIX + dataset + ":" + str(node)).encode("utf-8")).digest(), node))
    return {"schema": "vldb_sampling_test_targets_v1", "dataset": dataset, "split": "official_test",
            "ids": ids, "batches": [ids[i:i+128] for i in range(0, len(ids), 128)], "batch_size": 128,
            "source_provenance": {**provenance, "target_hash_prefix": TARGET_PREFIX + dataset + ":",
                                  "ordering_depends_on_labels": False}}


def parse_arxiv_labels(compressed, ids):
    """Consume one gzip stream; only official test rows are parsed as labels."""
    allowed = set(ids)
    labels = {}
    count = 0
    with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as f:
        for row, raw_line in enumerate(f):
            count += 1
            if row in allowed:
                value = int(raw_line)
                require(0 <= value < 40, "official test label outside class range")
                labels[row] = value
    require(count == 169343 and set(labels) == allowed, "official label row/test coverage differs")
    return [labels[n] for n in ids]


class RestrictedNumpyUnpickler(pickle.Unpickler):
    """Only globals used by legacy plain NumPy ndarray pickle are admissible."""
    def find_class(self, module, name):
        allowed = {("numpy.core.multiarray", "_reconstruct"), ("numpy", "ndarray"),
                   ("numpy", "dtype"), ("_codecs", "encode")}
        require((module, name) in allowed, "non-ndarray pickle global rejected")
        return super().find_class(module, name)


def parse_pubmed_labels(raw, ids):
    """Planetoid ty row j maps to test.index[j] after the official reorder.

    The immutable PubMed test IDs are exactly the contiguous appended indices
    18717..19716. Thus vstack(ally,ty)[sorted(test.index)] is ty itself, and the
    official assignment labels[test.index] = labels[sorted(test.index)] maps
    ty row j to test.index[j]. No ally/all-node label container is needed.
    """
    import numpy as np
    require(sorted(ids) == list(range(18717, 19717)), "PubMed appended test range differs")
    ty = RestrictedNumpyUnpickler(io.BytesIO(raw), encoding="latin1").load()
    require(type(ty) is np.ndarray and ty.shape == (1000, 3) and ty.dtype.kind in "biuf", "official PubMed ty shape/type differs")
    require(np.isfinite(ty).all() and np.all((ty == 0) | (ty == 1)) and np.all(ty.sum(axis=1) == 1), "PubMed test one-hot labels malformed")
    return ty.argmax(axis=1).astype(np.int64).tolist()


def run_job(release_path, expected_release_sha256, guard_callback):
    require(callable(guard_callback), "designated Ops guard callback required")
    guard_callback()
    started, cpu_started = time.perf_counter(), time.process_time()
    require(file_sha(release_path) == expected_release_sha256, "exact Root release hash differs")
    release = json.loads(Path(release_path).read_text(encoding="utf-8-sig"))
    require(set(release) == {"schema", "round", "authorized", "stage", "source_manifest", "science_freeze", "input_bindings", "output"}, "exact extraction release fields required")
    require(release["schema"] == "vldb_final_label_extraction_release_v1" and release["round"] == ROUND
            and release["authorized"] is True and release["stage"] == "final_label_extraction_once", "unreleased label entry")
    source = bound_json(release["source_manifest"])
    require(source["schema"] == "vldb_final_label_extraction_source_v1" and source["round"] == ROUND
            and len(source["files"]) == 1 and source["files"][0]["path"] == "extract_final_labels.py", "exact extraction source required")
    require(file_sha(HERE / "extract_final_labels.py") == source["files"][0]["sha256"], "extraction source changed")
    freeze = bound_json(release["science_freeze"])
    check_science_freeze(freeze, release["source_manifest"]["sha256"])
    bindings = bound_json(release["input_bindings"])
    require(bindings["schema"] == "vldb_final_label_input_bindings_v1" and bindings["round"] == ROUND
            and set(bindings["inputs"]) == EXPECTED_INPUTS, "exact raw/public input roles required")
    output = Path(release["output"])
    require(output.is_absolute() and not output.exists(), "new absolute output directory required")
    output = output.resolve()
    new_round = HERE.parents[1]
    require(output.is_relative_to(new_round / "operations") and "sealed-final-test" in output.name,
            "new Ops-only sealed-final-test output required")
    # Only after all 15 cells, budgets and sources are frozen may any raw label
    # payload be read. Hashes below bind originals; files are never relocated.
    paths = {role: bound(ref) for role, ref in bindings["inputs"].items()}
    require(output not in paths.values() and not any(output in p.parents for p in paths.values()), "output overlaps original input")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    try:
        arxiv_meta = json.loads(paths["arxiv_public_ids"].read_text(encoding="utf-8-sig"))
        arxiv_receipt = json.loads(paths["arxiv_preparation"].read_text(encoding="utf-8-sig"))
        require(arxiv_receipt["raw"]["raw/node-label.csv.gz"] == bindings["arxiv_members"]["labels"]["sha256"], "original arxiv label member receipt differs")
        require(arxiv_receipt["public"]["ids.json"] == bindings["inputs"]["arxiv_public_ids"]["sha256"], "arxiv public ID binding differs")
        with zipfile.ZipFile(paths["arxiv_archive"]) as archive:
            names = archive.namelist()
            require(names.count(ARXIV_TEST) == names.count(ARXIV_LABELS) == 1, "duplicate or absent exact archive member")
            compressed_ids = archive.read(ARXIV_TEST)
            spec = bindings["arxiv_members"]["test_ids"]
            require(spec["path"] == ARXIV_TEST and len(compressed_ids) == spec["bytes"] and sha_bytes(compressed_ids) == spec["sha256"], "official arxiv test split member differs")
            arxiv_ids = [int(row) for row in gzip.decompress(compressed_ids).splitlines()]
            require(len(arxiv_ids) == 48603 and all(0 <= n < 169343 for n in arxiv_ids), "official arxiv test count/range differs")
            public_ids_check(arxiv_meta, "ogbn-arxiv", arxiv_ids)
            compressed_labels = archive.read(ARXIV_LABELS)
            spec = bindings["arxiv_members"]["labels"]
            require(spec["path"] == ARXIV_LABELS and len(compressed_labels) == spec["bytes"] and sha_bytes(compressed_labels) == spec["sha256"], "official arxiv label member differs")
            arxiv_y = parse_arxiv_labels(compressed_labels, arxiv_ids)
        guard_callback()
        pubmed_meta = json.loads(paths["pubmed_public_ids"].read_text(encoding="utf-8-sig"))
        pubmed_receipt = json.loads(paths["pubmed_preparation"].read_text(encoding="utf-8-sig"))
        require(pubmed_receipt["public"]["ids.json"] == bindings["inputs"]["pubmed_public_ids"]["sha256"], "PubMed public ID binding differs")
        pubmed_source = json.loads(paths["pubmed_source_manifest"].read_text(encoding="utf-8-sig"))
        source_files = {v["name"]: v for v in pubmed_source["files"]}
        require(source_files["ind.pubmed.ty"]["sha256"] == bindings["inputs"]["pubmed_ty"]["sha256"]
                and source_files["ind.pubmed.test.index"]["sha256"] == bindings["inputs"]["pubmed_test_index"]["sha256"], "official PubMed source binding differs")
        pubmed_ids = [int(row) for row in paths["pubmed_test_index"].read_text(encoding="ascii").splitlines()]
        public_ids_check(pubmed_meta, "PubMed", pubmed_ids)
        pubmed_y = parse_pubmed_labels(paths["pubmed_ty"].read_bytes(), pubmed_ids)
        result = {"schema": "vldb_final_label_extraction_receipt_v1", "round": ROUND, "status": "completed_no_scores_computed",
                  "science_freeze": release["science_freeze"], "source_manifest": release["source_manifest"],
                  "release_sha256": expected_release_sha256, "input_bindings": release["input_bindings"],
                  "original_inputs_modified": False, "models_or_checkpoints_loaded": False, "prediction_scores_computed": False,
                  "datasets": {}, "label_contact": {"ogbn-arxiv": "Entire gzip stream decompressed; only the 48603 official test rows parsed; no all-node y built.",
                                                     "PubMed": "Original ty 1000x3 test-label array decoded now; historically exposed by prior HyboNet/LResNet experiments."}}
        for dataset, ids, labels, provenance in (
            ("ogbn-arxiv", arxiv_ids, arxiv_y, {"official_source_url": bindings["arxiv_url"], "archive_sha256": bindings["inputs"]["arxiv_archive"]["sha256"],
              "official_test_member": bindings["arxiv_members"]["test_ids"], "label_member": bindings["arxiv_members"]["labels"],
              "prior_test_semantic_access": "Historical prepare parsed train/development only; no prior official-test scoring reported in accepted round records."}),
            ("PubMed", pubmed_ids, pubmed_y, {"official_source_repository": pubmed_source["source_repository"], "official_source_commit": pubmed_source["source_commit"],
              "test_index_sha256": bindings["inputs"]["pubmed_test_index"]["sha256"], "ty_sha256": bindings["inputs"]["pubmed_ty"]["sha256"],
              "prior_test_semantic_access": "Previously used by HyboNet/LResNet experiments; not a never-observed dataset test.",
              "label_alignment": "ty row j belongs to test.index[j]; verified contiguous appended sorted range 18717..19716."})):
            folder = output / dataset
            folder.mkdir(mode=0o700)
            target = target_manifest(dataset, ids, provenance)
            save(folder / "targets.json", target)
            with (folder / "labels.npz").open("xb") as f:
                np.savez(f, ids=np.asarray(ids, dtype=np.int64), y=np.asarray(labels, dtype=np.int64))
            os.chmod(folder / "labels.npz", 0o600)
            result["datasets"][dataset] = {"count": len(ids), "classes": 40 if dataset == "ogbn-arxiv" else 3,
                  "targets": {"path": str(folder / "targets.json"), "sha256": file_sha(folder / "targets.json")},
                  "labels": {"path": str(folder / "labels.npz"), "sha256": file_sha(folder / "labels.npz")},
                  "target_order": "fixed label-free hash order; batches128", "label_array_order": "original official test ID file order"}
        guard_callback()
        result["wall_seconds"] = time.perf_counter() - started
        result["process_CPU_seconds"] = time.process_time() - cpu_started
        result["GPU_seconds"] = 0
        result["permission_scope"] = "Ops process/path isolation; chmod is advisory on Windows and not an account-security boundary"
        save(output / "receipt.json", result)
        return result
    except Exception as error:
        save(output / "failure.json", {"status": "failed_no_retry", "type": type(error).__name__, "message": str(error),
             "wall_seconds": time.perf_counter()-started, "process_CPU_seconds": time.process_time()-cpu_started,
             "GPU_seconds": 0, "label_extraction_may_have_started": True})
        raise

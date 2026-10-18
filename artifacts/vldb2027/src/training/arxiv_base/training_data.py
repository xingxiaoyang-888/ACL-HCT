"""UNEXECUTED training-only arxiv input firewall and paired random plumbing.

The caller supplies the exact Root-bound metadata manifest. Only public graph,
ID/preparation metadata, train labels and fit labels can be opened here. No raw
OGB loader, all-node label array, gate/confirmation/test label reader or writer.
Inherited files stay read-only. Real releases must bind the official dimensions
and counts; explicit small dimensions also permit isolated synthetic fixtures.

This module does not import old runners. The accepted native uniform sampler
and its thread scope enter through UniformPlanningBridge. Graph format creation,
metadata validation, feature indexing and transfer are real, chargeable costs.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from sampling_bridge import UniformPlanningBridge, plan_seed
from shared_blocks import NativeBlock


_ROLES = {"graph", "ids", "preparation", "train", "fit"}
_SPLITS = ("train", "fit", "gate", "confirmation")
_FILENAMES = {"graph": "graph.npz", "ids": "ids.json",
              "preparation": "preparation.json", "train": "train.npz", "fit": "fit.npz"}


def _integer(value, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _ids(values, n: int, name: str) -> np.ndarray:
    if isinstance(values, list):
        if any(type(value) is not int for value in values):
            raise ValueError(f"{name} requires integer IDs, not coercible values")
        result = np.asarray(values, dtype=np.int64)
    else:
        result = np.asarray(values)
    if (result.ndim != 1 or result.dtype != np.int64
            or np.any(result < 0) or np.any(result >= n)
            or len(np.unique(result)) != len(result)):
        raise ValueError(f"malformed {name}")
    return result.copy()


@contextmanager
def _checked_file(binding: Mapping, role: str):
    """Hash and parse the same read-only handle; never enumerate directories."""
    path = Path(binding["path"])
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(2 ** 20), b""):
            digest.update(chunk)
        if digest.hexdigest() != binding["sha256"]:
            raise ValueError(f"bound {role} file changed")
        stream.seek(0)
        yield stream


def _json(binding: Mapping, role: str):
    with _checked_file(binding, role) as stream:
        return json.loads(stream.read().decode("utf-8-sig"))


def _npz(binding: Mapping, role: str, keys: set[str]) -> dict[str, np.ndarray]:
    with _checked_file(binding, role) as stream, np.load(stream, allow_pickle=False) as archive:
        if len(archive.files) != len(keys) or set(archive.files) != keys:
            raise ValueError(f"unexpected {role} array fields")
        return {key: archive[key].copy() for key in keys}


def _manifest(value: Mapping) -> tuple[dict, dict, dict]:
    if (not isinstance(value, Mapping)
            or set(value) != {"schema", "dataset", "shape", "split_counts", "files"}
            or value["schema"] != "dasfaa_training_inputs_v1"
            or value["dataset"] != "ogbn-arxiv"):
        raise ValueError("exact training input metadata manifest required")
    shape, counts, files = value["shape"], value["split_counts"], value["files"]
    if not isinstance(shape, Mapping) or set(shape) != {"nodes", "features", "classes"}:
        raise ValueError("explicit graph shape/classes required")
    shape = {key: _integer(item, key, minimum=2 if key == "classes" else 1)
             for key, item in shape.items()}
    if not isinstance(counts, Mapping) or set(counts) != set(_SPLITS):
        raise ValueError("exact inherited split counts required")
    counts = {key: _integer(item, key + " count") for key, item in counts.items()}
    if counts["train"] == 0 or counts["fit"] == 0:
        raise ValueError("nonempty train and fit splits required")
    if not isinstance(files, Mapping) or set(files) != _ROLES:
        raise ValueError("only graph/ids/preparation/train/fit file roles are allowed")
    resolved = {}
    for role, entry in files.items():
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
            raise ValueError("exact path/hash file bindings required")
        if not isinstance(entry["path"], str) or not isinstance(entry["sha256"], str):
            raise ValueError("file paths and SHA256 must be strings")
        path, digest = Path(entry["path"]), entry["sha256"]
        if (not path.is_absolute() or path.name != _FILENAMES[role]
                or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("absolute canonical file-role path and lowercase SHA256 required")
        target = path.resolve()
        if target.name != _FILENAMES[role]:
            raise ValueError("file-role path resolves to a different input role")
        resolved[role] = {"path": str(target), "sha256": digest}
    if len({item["path"] for item in resolved.values()}) != len(_ROLES):
        raise ValueError("input roles must bind distinct files")
    return shape, counts, resolved


@dataclass
class PublicGraph:
    """CPU public-feature/CSR graph compatible with the accepted uniform sampler.

    kind='sage' is only its unweighted storage branch; all four models use it.
    CSR NumPy arrays are private read-only copies. Do not mutate x or use tensor
    aliases to modify metadata. DGL is imported only on an authorized plan call.
    """
    x: Tensor
    indptr: np.ndarray
    indices: np.ndarray
    degree: np.ndarray
    kind: str = field(default="sage", init=False)
    _dgl: object = field(default=None, init=False, repr=False)

    @property
    def n(self) -> int:
        return len(self.x)

    def dgl_graph(self):
        # Called inside the accepted sampler's native thread scope. Persistent
        # DGL indices must remain normal tensors even if the first call is eval.
        if self._dgl is None:
            with torch.inference_mode(False):
                import dgl
                rows = np.repeat(np.arange(self.n, dtype=np.int64), self.degree)
                native = dgl.graph((torch.tensor(self.indices), torch.from_numpy(rows)),
                                   num_nodes=self.n)
                native.create_formats_()
                self._dgl = native
        return self._dgl


def _graph(arrays: dict[str, np.ndarray], shape: dict) -> PublicGraph:
    x, indptr, indices = (arrays[key] for key in ("x", "indptr", "indices"))
    n = shape["nodes"]
    if x.dtype != np.float32 or x.shape != (n, shape["features"]) or not np.isfinite(x).all():
        raise ValueError("public feature shape/dtype/values differ")
    if (indptr.dtype != np.int64 or indptr.shape != (n + 1,)
            or indices.dtype != np.int64 or indices.ndim != 1
            or indptr[0] != 0 or indptr[-1] != len(indices)
            or np.any(indptr < 0) or np.any(indptr > len(indices))
            or np.any(np.diff(indptr) < 0)
            or np.any(indices < 0) or np.any(indices >= n)):
        raise ValueError("malformed CSR")
    degree = np.diff(indptr)
    rows = np.repeat(np.arange(n, dtype=np.int64), degree)
    if np.any(rows == indices):
        # Old symmetric_csr unique() preserves a diagonal already in the raw
        # edge list. diagonal==0 alone does NOT establish this invariant.
        raise ValueError("original CSR self edges are forbidden; no silent removal")
    if len(indices) > 1 and np.any((rows[1:] == rows[:-1]) & (indices[1:] <= indices[:-1])):
        raise ValueError("CSR rows must contain sorted distinct neighbors")
    values, diagonal = arrays["values"], arrays["diagonal"]
    if (values.dtype != np.float32 or values.shape != indices.shape or not np.all(values == 1)
            or diagonal.dtype != np.float32 or diagonal.shape != (n,) or not np.all(diagonal == 0)):
        raise ValueError("unweighted public graph with no stored diagonal required")
    for array in (indptr, indices, degree):
        array.setflags(write=False)
    with torch.inference_mode(False):
        features = torch.from_numpy(x)
    return PublicGraph(features, indptr, indices, degree)


@dataclass
class TrainingData:
    graph: PublicGraph
    train_ids: np.ndarray  # int64 in inherited order, not all-node arrays
    fit_ids: np.ndarray
    train_y: np.ndarray    # int64 labels aligned with the corresponding IDs
    fit_y: np.ndarray
    fit_batches: tuple[np.ndarray, ...]  # inherited fixed fit batches
    _lookup: dict[str, tuple[np.ndarray, np.ndarray]] = field(init=False, repr=False)

    def __post_init__(self):
        self._lookup = {}
        for split in ("train", "fit"):
            ids, labels = getattr(self, split + "_ids"), getattr(self, split + "_y")
            order = np.argsort(ids)
            sorted_ids, sorted_y = ids[order], labels[order]
            sorted_ids.setflags(write=False)
            sorted_y.setflags(write=False)
            self._lookup[split] = (sorted_ids, sorted_y)
        for array in (self.train_ids, self.fit_ids, self.train_y, self.fit_y, *self.fit_batches):
            array.setflags(write=False)

    def labels_for(self, split: str, target_ids: np.ndarray) -> np.ndarray:
        if split not in ("train", "fit"):
            raise ValueError("training label firewall permits only train and fit")
        targets = _ids(target_ids, self.graph.n, "label targets")
        sorted_ids, sorted_y = self._lookup[split]
        positions = np.searchsorted(sorted_ids, targets)
        if np.any(positions == len(sorted_ids)) or not np.array_equal(sorted_ids[positions], targets):
            raise ValueError("label target outside the authorized split")
        return sorted_y[positions].copy()

    def training_batches(self, *, seed: int, epoch: int, batch_size: int) -> Iterator[np.ndarray]:
        _integer(batch_size, "batch_size", minimum=1)
        generator = np.random.default_rng(paired_seed("train", seed, epoch=epoch, purpose="batch_order"))
        shuffled = generator.permutation(self.train_ids)
        for start in range(0, len(shuffled), batch_size):
            yield shuffled[start:start + batch_size].copy()


def load_training_data(manifest: Mapping) -> TrainingData:
    """Read exactly five inherited bindings; this function never writes files."""
    shape, counts, files = _manifest(manifest)
    receipt = _json(files["preparation"], "preparation")
    metadata = _json(files["ids"], "ids")
    if (receipt.get("schema") != "icml_ca_preparation_v1"
            or receipt.get("dataset_version") != "ogbn-arxiv v1"
            or receipt.get("semantic_test_label_reads") != 0
            or receipt.get("confirmation_label_path_exposed_to_development") is not False):
        raise ValueError("inherited preparation metadata differs")
    for role in ("graph", "ids", "train", "fit"):
        section = "public" if role in ("graph", "ids") else "development"
        if receipt.get(section, {}).get(_FILENAMES[role]) != files[role]["sha256"]:
            raise ValueError("input hash differs from inherited preparation")
    if (set(metadata) != {"schema", "dataset", "classes", "ids", "batches", "official_test_ids_read"}
            or metadata["schema"] != "icml_ca_ids_v1" or metadata["dataset"] != "ogbn-arxiv"
            or type(metadata["classes"]) is not int or metadata["classes"] != shape["classes"]
            or metadata["official_test_ids_read"] is not False
            or set(metadata["ids"]) != set(_SPLITS)):
        raise ValueError("inherited ID metadata differs")
    ids = {split: _ids(metadata["ids"][split], shape["nodes"], split + " IDs") for split in _SPLITS}
    combined = np.concatenate(tuple(ids.values()))
    if any(len(ids[split]) != counts[split] for split in _SPLITS) or len(np.unique(combined)) != len(combined):
        raise ValueError("split counts or disjointness differ")
    if set(metadata["batches"]) != {"fit", "gate", "confirmation"}:
        raise ValueError("inherited batch metadata differs")
    fit_batches = tuple(_ids(batch, shape["nodes"], "fit batch") for batch in metadata["batches"]["fit"])
    if not fit_batches or any(len(batch) == 0 for batch in fit_batches):
        raise ValueError("nonempty inherited fit batches required")
    fit_flat = np.concatenate(fit_batches)
    if len(fit_flat) != len(ids["fit"]) or not np.array_equal(np.sort(fit_flat), np.sort(ids["fit"])):
        raise ValueError("fit batches must partition the authorized fit IDs exactly once")
    graph = _graph(_npz(files["graph"], "graph", {"x", "indptr", "indices", "values", "diagonal"}), shape)
    labels = {}
    for split in ("train", "fit"):
        data = _npz(files[split], split, {"ids", "y"})
        if (data["ids"].dtype != np.int64 or not np.array_equal(data["ids"], ids[split])
                or data["y"].dtype != np.int64 or data["y"].shape != ids[split].shape
                or np.any(data["y"] < 0) or np.any(data["y"] >= shape["classes"])):
            raise ValueError("label IDs/order/dtype/range differ from authorized split")
        labels[split] = data["y"]
    # Gate/confirmation ID metadata was checked for overlap, then discarded.
    # Their labels and paths have never entered the reader.
    return TrainingData(graph, ids["train"], ids["fit"], labels["train"], labels["fit"], fit_batches)


def paired_seed(stage: str, seed: int, *, epoch: int = 0, repeat: int = 0,
                batch: int = 0, purpose: str) -> int:
    """No arm/model key. Fit selection uses one fixed stream across epochs."""
    if stage not in ("train", "fit") or purpose not in ("batch_order", "dropout"):
        raise ValueError("explicit train/fit stage and batch_order/dropout purpose required")
    if stage == "fit" and epoch != 0:
        raise ValueError("fit selection stream must use epoch=0 across checkpoint opportunities")
    return plan_seed(stage=stage + ":training-plumbing", training_seed=seed, epoch=epoch,
                     repeat=repeat, batch=batch, layer=0, path=purpose) & ((1 << 63) - 1)


@contextmanager
def paired_dropout_rng(*, stage: str, seed: int, epoch: int = 0, repeat: int = 0,
                       batch: int = 0, device: str | torch.device = "cpu"):
    """Native dropout uses scoped CPU/one-CUDA RNGs restored on exit.

    Unlike torch.manual_seed this never seeds all visible GPUs. On CPU it never
    initializes or queries CUDA. Call once around each arm's matching forward.
    """
    target = torch.device(device)
    if target.type not in ("cpu", "cuda") or (target.type == "cuda" and target.index is None):
        raise ValueError("CPU or an explicit process-local CUDA device index required")
    devices = [] if target.type == "cpu" else [target.index]
    value = paired_seed(stage, seed, epoch=epoch, repeat=repeat, batch=batch, purpose="dropout")
    with torch.random.fork_rng(devices=devices):
        torch.default_generator.manual_seed(value)
        if devices:
            torch.cuda.default_generators[target.index].manual_seed(value)
        yield


def plan_blocks(graph: PublicGraph, bridge: UniformPlanningBridge, targets: np.ndarray,
                fanouts: Sequence[int], *, stage: str, training_seed: int,
                epoch: int = 0, repeat: int = 0, batch: int = 0,
                device: str | torch.device = "cpu") -> list[NativeBlock]:
    """Normal persistent planning tensors even inside an eval inference context."""
    if stage not in ("train", "fit") or (stage == "fit" and epoch != 0):
        raise ValueError("training plans require train or fixed-epoch fit stage")
    with torch.inference_mode(False):
        blocks = bridge.plan(graph, targets, fanouts, stage=stage,
                             training_seed=training_seed, epoch=epoch, repeat=repeat, batch=batch)
        return [block.to(device) for block in blocks]

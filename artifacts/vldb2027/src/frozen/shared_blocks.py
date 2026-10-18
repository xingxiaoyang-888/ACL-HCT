"""UNEXECUTED v2 candidate: collision-free local-edge distinctness check only.

The accepted shared_blocks.py stays immutable. A future source manifest and
runner must explicitly bind this module origin; importing v2 here does not
redirect any existing adapter. Only the duplicate-pair count differs below.
All edge order, counts, slots, sampler checks and mutation guards are retained.

Explicit, checked dependency blocks; no sampler, task IO, or execution.

Only distinct non-self edges are stored. A model owns its self path and any
native coefficients. Validation/planning and all stored formats have real
costs; this interface does not equate fanout with runtime or memory.

Metadata is immutable after construction, including counts and edge slots.
Normal in-place edits (even add_(0)) are rejected using Tensor identity/version
fingerprints. Mutation through .data, external shared NumPy storage, or other
unversioned aliases is unsupported: callers must not use those escape hatches.
Plan with version-tracked tensors outside inference_mode; transfer preserves
the validated metadata and does not rerun device-side unique/sort/count work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor


def _distinct_local_edge_count(edge_src: Tensor, edge_dst: Tensor,
                               num_src: int, num_dst: int) -> int:
    """Count distinct pairs after the constructor's local-index checks.

    Preconditions: int64 matching vectors, S=num_src>0, B=num_dst>0,
    0<=src<S and 0<=dst<B. Global node IDs never enter this encoding.
    Division with remainder proves (dst,src)->dst*S+src is injective.
    Python-integer checks guarantee S fits the scalar int64 operand and
    B*S<=2**63, hence every intermediate/product and final key is between
    zero and B*S-1<=2**63-1. No signed overflow or collision can occur.
    Otherwise retain the original two-column unique without encoding.
    """
    maximum = (1 << 63) - 1
    if num_src <= maximum and num_dst <= (maximum + 1) // num_src:
        keys = edge_dst * num_src + edge_src
        return torch.unique(keys).numel()
    pairs = torch.stack((edge_dst, edge_src), dim=1)
    return torch.unique(pairs, dim=0).shape[0]


@dataclass(frozen=True)
class NativeBlock:
    src_ids: Tensor
    dst_ids: Tensor
    edge_src: Tensor
    edge_dst: Tensor
    root_index: Tensor
    nonself_degree: Tensor
    sampler: str = "uniform_wor"
    labor_fanout: int | None = None
    _counts: Tensor = field(init=False, repr=False)
    _edge_slot: Tensor = field(init=False, repr=False)
    _is_full: bool = field(init=False, repr=False)
    _versions: tuple[tuple[int, int], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        fields = (self.src_ids, self.dst_ids, self.edge_src, self.edge_dst,
                  self.root_index, self.nonself_degree)
        if any(t.ndim != 1 or t.dtype != torch.int64 for t in fields):
            raise ValueError("block IDs, indices and degrees require int64 vectors")
        if any(t.device != self.src_ids.device for t in fields):
            raise ValueError("all block metadata must share a device")
        if self.num_dst == 0 or self.num_src == 0:
            raise ValueError("nonempty source and destination ID vectors required")
        if not self.sampler or not isinstance(self.sampler, str):
            raise ValueError("explicit sampler provenance required")
        if self.sampler == 'labor0':
            if type(self.labor_fanout) is not int or self.labor_fanout < 1:
                raise ValueError('LABOR-0 requires its positive nominal fanout')
        elif self.labor_fanout is not None:
            raise ValueError('nominal LABOR fanout is only legal for LABOR-0')
        if any(bool((t < 0).any()) for t in (self.src_ids, self.dst_ids,
                                           self.nonself_degree)):
            raise ValueError("negative node ID or population degree")
        if (self.src_ids.unique().numel() != self.num_src
                or self.dst_ids.unique().numel() != self.num_dst):
            raise ValueError("source and destination IDs must each be distinct")
        if self.root_index.shape != self.dst_ids.shape:
            raise ValueError("one explicit root source index per destination required")
        if self.nonself_degree.shape != self.dst_ids.shape:
            raise ValueError("one nonself population degree per destination required")
        if self.edge_src.shape != self.edge_dst.shape:
            raise ValueError("edge source/destination length mismatch")
        if (bool(((self.root_index < 0) | (self.root_index >= self.num_src)).any())
                or not torch.equal(self.src_ids[self.root_index], self.dst_ids)):
            raise ValueError("root indices do not map destination global IDs")
        if (bool(((self.edge_src < 0) | (self.edge_src >= self.num_src)).any())
                or bool(((self.edge_dst < 0) | (self.edge_dst >= self.num_dst)).any())):
            raise ValueError("edge local index outside block")
        if bool((self.src_ids[self.edge_src] == self.dst_ids[self.edge_dst]).any()):
            raise ValueError("block edges must exclude self")
        if _distinct_local_edge_count(self.edge_src, self.edge_dst,
                                      self.num_src, self.num_dst) != self.num_edges:
            raise ValueError("sampled neighbors must be distinct within each destination")
        counts = torch.bincount(self.edge_dst, minlength=self.num_dst)
        if bool((counts > self.nonself_degree).any()):
            raise ValueError("sample count exceeds nonself population")
        is_full = torch.equal(counts, self.nonself_degree)
        if self.sampler == "full" and not is_full:
            raise ValueError("full provenance requires k=N at every destination")
        object.__setattr__(self, "_counts", counts)
        order = torch.argsort(self.edge_dst, stable=True)
        offset = counts.cumsum(0) - counts
        sorted_slots = torch.arange(self.num_edges, device=self.src_ids.device)
        sorted_slots = sorted_slots - torch.repeat_interleave(offset, counts)
        slots = torch.empty_like(self.edge_src)
        slots[order] = sorted_slots
        object.__setattr__(self, "_edge_slot", slots)
        object.__setattr__(self, "_is_full", is_full)
        object.__setattr__(self, "_versions", self._fingerprint())

    def _metadata_tensors(self) -> tuple[Tensor, ...]:
        return (self.src_ids, self.dst_ids, self.edge_src, self.edge_dst,
                self.root_index, self.nonself_degree, self._counts, self._edge_slot)

    def _fingerprint(self) -> tuple[tuple[int, int], ...]:
        try:
            return tuple((id(t), t._version) for t in self._metadata_tensors())
        except RuntimeError as error:
            raise ValueError("block metadata requires version-tracked tensors; "
                             "construct outside inference_mode") from error

    def assert_immutable(self) -> None:
        # Tensor version counters are host metadata: no device tensor scan,
        # comparison kernel, or host/device synchronization is introduced.
        if self._fingerprint() != self._versions:
            raise ValueError("block metadata was modified after validation; rebuild the block")

    @property
    def num_src(self) -> int:
        return self.src_ids.numel()

    @property
    def num_dst(self) -> int:
        return self.dst_ids.numel()

    @property
    def num_edges(self) -> int:
        return self.edge_src.numel()

    @property
    def counts(self) -> Tensor:
        self.assert_immutable()
        return self._counts

    @property
    def edge_slot(self) -> Tensor:
        """Within-row slots in original edge order, computed during planning."""
        self.assert_immutable()
        return self._edge_slot

    @property
    def is_full(self) -> bool:
        self.assert_immutable()
        return self._is_full

    def to(self, device: torch.device | str) -> NativeBlock:
        self.assert_immutable()
        # Internal trusted-copy path only: transfer all validated tensors,
        # including counts and slots, then take NEW target version counters.
        # Do not call __init__/__post_init__ or rescan tensors on the device.
        transferred = object.__new__(NativeBlock)
        names = ("src_ids", "dst_ids", "edge_src", "edge_dst", "root_index",
                 "nonself_degree", "_counts", "_edge_slot")
        # Keep transferred metadata version-tracked even when an evaluation
        # caller has inference_mode enabled. Integer metadata needs no grads.
        with torch.inference_mode(False):
            for name, tensor in zip(names, self._metadata_tensors()):
                object.__setattr__(transferred, name, tensor.to(device))
        object.__setattr__(transferred, "sampler", self.sampler)
        object.__setattr__(transferred, "labor_fanout", self.labor_fanout)
        object.__setattr__(transferred, "_is_full", self._is_full)
        object.__setattr__(transferred, "_versions", transferred._fingerprint())
        return transferred

    def with_self_edges(self) -> tuple[Tensor, Tensor]:
        """Append exactly one correctly mapped self edge per destination."""
        self.assert_immutable()
        return (torch.cat((self.edge_src, self.root_index)),
                torch.cat((self.edge_dst, torch.arange(
                    self.num_dst, device=self.edge_dst.device))))

    def report(self) -> dict[str, int | str]:
        self.assert_immutable()
        return {"unique_source": self.num_src, "unique_destination": self.num_dst,
                "sampled_nonself_edges": self.num_edges,
                "distinct_sampled_neighbors": self.num_edges,
                "population_nonself_edges_for_destinations": int(self.nonself_degree.sum()),
                "max_sampled_neighbors": int(self.counts.max()),
                "self_edges_stored": 0, "sampler": self.sampler,
                "metadata_tensor_elements": sum(t.numel() for t in (
                    self.src_ids, self.dst_ids, self.edge_src, self.edge_dst,
                    self.root_index, self.nonself_degree, self.counts, self.edge_slot))}


def validate_block_chain(blocks: Sequence[NativeBlock]) -> None:
    for block in blocks:
        block.assert_immutable()
    for left, right in zip(blocks, blocks[1:]):
        if not torch.equal(left.dst_ids, right.src_ids):
            raise ValueError("layer outputs must exactly match next source IDs and order")


def require_uniform_or_full(block: NativeBlock) -> None:
    block.assert_immutable()
    if block.sampler not in ("uniform_wor", "full", "labor0"):
        raise ValueError("this new plain-only adapter supports uniform WOR, LABOR-0 or full")


def check_features(x: Tensor, block: NativeBlock, channels: int) -> None:
    block.assert_immutable()
    if x.ndim != 2 or x.shape != (block.num_src, channels):
        raise ValueError("source features do not match explicit block/source dimension")
    if x.device != block.src_ids.device or x.dtype not in (torch.float32, torch.float64):
        raise ValueError("FP32/FP64 features and block on the same device required")


def mean_aggregate(x: Tensor, block: NativeBlock) -> Tensor:
    """Raw neighbor mean with zero isolated rows; no per-neighbor matrix map."""
    coefficients = block.counts[block.edge_dst].to(x.dtype).reciprocal()
    adjacency = torch.sparse_coo_tensor(
        torch.stack((block.edge_dst, block.edge_src)), coefficients,
        (block.num_dst, block.num_src), device=x.device).coalesce()
    return torch.sparse.mm(adjacency, x)

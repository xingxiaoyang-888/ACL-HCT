"""SOURCE-ONLY exact full reference for the four registered three-layer models.

No dataset/checkpoint/label IO, training, timing harness or autonomous execution.
Caller supplies its already authorized PublicGraph and its OWN selected model.
This implements full-neighbor arithmetic, not bitwise equivalence of different
FP32 sparse reductions. It preserves native heads, BN evaluation, fixed self
coefficients, attention, residuals and native hyperbolic clamps. Full coverage
makes the registered correction identically zero, so no correction is invoked.

Construction is chargeable setup: validate public CSR, build full formats and
transfer persistent graph indices. Every forward separately transfers the raw
CPU features and recomputes all three layers. No hidden states/output survive
inside this object. The caller charges synchronization and output CPU return,
owns the optional PAID full-output cache, and scores only authorized partitions.
All-node output availability never authorizes reading or scoring test labels.

SAGE CSR and sorted diagonal insertion adapt the read-only code-v4/full_graph.py
pattern; HGCN/Hybo call the new wrappers' original native operators. GAT uses
the exact accepted CorrectedGATConv eval formula with source projection ONCE
per layer and complete-destination-neighborhood chunks. Explicit chunk limits
must be frozen using label-free structure/memory; a too-large row fails rather
than splitting its softmax denominator or silently retrying another setting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.utils import softmax


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _bytes(tensors):
    return sum(value.numel() * value.element_size() for value in tensors)


@dataclass(frozen=True)
class _AttentionChunk:
    start: int
    end: int
    cpu_source: Tensor
    cpu_destination: Tensor
    source: Tensor
    destination: Tensor


class ExactFullGraph:
    """Chargeable, reusable graph format; no persistent features or hidden cache.

    graph.x is CPU FP32; graph.indptr/indices/degree are the already accepted
    simple sorted NONSELF CSR arrays. All four models share that representation.
    Format/private-index storage is additional to common model/CPU graph
    residency and must be reported even when reused across repeated requests.
    CPU and cuda:0 are supported; this constructor itself allocates/transfers
    on the supplied device only when separately authorized by the caller.
    """

    def __init__(self, model, graph, device, *, gat_chunk_rows=None, gat_chunk_edges=None):
        self.model, self.raw_features = model, graph.x
        self.model_key = getattr(model, "_study_model_key", None)
        self.device = torch.device(device)
        _require(self.model_key in {"sage", "gat", "hgcn", "hybonet"}, "registered builder model required")
        _require(str(self.device) in {"cpu", "cuda:0"}, "CPU or the unique cuda:0 device only")
        _require(self.raw_features.device.type == "cpu" and self.raw_features.dtype == torch.float32
                 and self.raw_features.ndim == 2 and self.raw_features.shape[1] == 128
                 and len(self.raw_features) > 0 and not self.raw_features.requires_grad,
                 "common nonempty CPU FP32 raw feature matrix required")
        _require(torch.isfinite(self.raw_features).all(), "nonfinite public features")
        self.n = len(self.raw_features)
        self._raw_signature = (id(self.raw_features), self.raw_features._version)
        ptr, columns, degree = (np.asarray(value) for value in (graph.indptr, graph.indices, graph.degree))
        _require(all(value.dtype == np.int64 and value.ndim == 1 for value in (ptr, columns, degree)),
                 "int64 CSR and degrees required")
        _require(len(ptr) == self.n + 1 and len(degree) == self.n and ptr[0] == 0
                 and ptr[-1] == len(columns) and np.all(degree >= 0)
                 and np.array_equal(np.diff(ptr), degree), "malformed public CSR offsets/degrees")
        _require(np.all((columns >= 0) & (columns < self.n)), "CSR source index out of range")
        rows = np.repeat(np.arange(self.n, dtype=np.int64), degree)
        _require(not np.any(rows == columns), "public CSR must exclude self edges")
        _require(not np.any((rows[1:] == rows[:-1]) & (columns[1:] <= columns[:-1])),
                 "CSR neighbors must be strictly sorted and distinct within each row")
        self.nonself_edges = len(columns)
        self.cpu_adjacency = self.adjacency = None
        self._chunks = ()
        self.gat_chunk_rows, self.gat_chunk_edges = gat_chunk_rows, gat_chunk_edges
        if self.model_key == "gat":
            _require(type(gat_chunk_rows) is int and gat_chunk_rows > 0
                     and type(gat_chunk_edges) is int and gat_chunk_edges > 0,
                     "GAT requires explicit positive, prospectively frozen chunk limits")
            self._chunks = self._attention_chunks(ptr, columns, rows, gat_chunk_rows, gat_chunk_edges)
        else:
            _require(gat_chunk_rows is None and gat_chunk_edges is None, "GAT chunk limits apply only to GAT")
            self.cpu_adjacency = self._csr(ptr, columns, rows, degree)
            self.adjacency = self.cpu_adjacency.to(device=self.device)
        self._check_model()

    def _csr(self, ptr, columns, rows, degree):
        if self.model_key == "sage":
            full_ptr, full_columns = ptr.copy(), columns.copy()
            reciprocal = torch.from_numpy(degree.copy()).to(torch.float32).clamp_min(1).reciprocal()
            values = torch.repeat_interleave(reciprocal, torch.from_numpy(degree.copy()))
        else:
            # Sorted O(N+E) insertion of one native self entry in every row.
            full_ptr = ptr + np.arange(self.n + 1, dtype=np.int64)
            full_columns = np.empty(len(columns) + self.n, dtype=np.int64)
            position = np.arange(len(columns), dtype=np.int64) + rows + (columns > rows)
            full_columns[position] = columns
            before = np.bincount(rows[columns < rows], minlength=self.n)
            full_columns[full_ptr[:-1] + before] = np.arange(self.n, dtype=np.int64)
            reciprocal = (torch.from_numpy(degree.copy()).to(torch.float32) + 1).reciprocal()
            values = torch.repeat_interleave(reciprocal, torch.from_numpy(degree.copy() + 1))
        return torch.sparse_csr_tensor(torch.from_numpy(full_ptr), torch.from_numpy(full_columns),
                                       values, size=(self.n, self.n), dtype=torch.float32)

    def _attention_chunks(self, ptr, columns, rows, max_rows, max_edges):
        cumulative = ptr + np.arange(self.n + 1, dtype=np.int64)
        _require(np.all(np.diff(cumulative) <= max_edges),
                 "one complete GAT neighborhood exceeds frozen chunk edge limit")
        chunks, start = [], 0
        while start < self.n:
            edge_end = int(np.searchsorted(cumulative, cumulative[start] + max_edges, side="right") - 1)
            end = min(self.n, start + max_rows, edge_end)
            _require(end > start, "complete-row chunk planning made no progress")
            first, last = int(ptr[start]), int(ptr[end])
            cpu_source = torch.from_numpy(np.concatenate((columns[first:last], np.arange(start, end, dtype=np.int64))))
            cpu_destination = torch.from_numpy(np.concatenate((rows[first:last] - start,
                                                               np.arange(end - start, dtype=np.int64))))
            chunks.append(_AttentionChunk(start, end, cpu_source, cpu_destination,
                                          cpu_source.to(self.device), cpu_destination.to(self.device)))
            start = end
        return tuple(chunks)

    def _check_model(self):
        _require(not any(module.training for module in self.model.modules()), "exact full requires all modules in eval")
        _require(all(parameter.device == self.device and parameter.dtype == torch.float32
                     for parameter in self.model.parameters()), "model must share full device and FP32 dtype")
        layers = self.model.layers if self.model_key in {"hgcn", "hybonet"} else self.model.convs
        _require(len(layers) == 3, "exactly three actual message layers required")
        if self.model_key in {"hgcn", "hybonet"}:
            for layer in layers:
                layer.contract.validate()
                layer._check_native()

    def _gat_layer(self, layer, x):
        # Complete-neighbor attention; projection and source attention are
        # shared across chunks within THIS paid forward only.
        values = layer.lin(x).view(self.n, layer.heads, layer.head_dim)
        alpha_source = (values * layer.att_src).sum(-1)
        alpha_target = (values * layer.att_dst).sum(-1)
        width = layer.heads * layer.head_dim if layer.concat else layer.head_dim
        output = values.new_empty((self.n, width))
        for chunk in self._chunks:
            logits = F.leaky_relu(alpha_source[chunk.source]
                                  + alpha_target[chunk.destination + chunk.start], layer.negative_slope)
            alpha = softmax(logits, chunk.destination, num_nodes=chunk.end - chunk.start)
            aggregate = values.new_zeros((chunk.end - chunk.start, layer.heads, layer.head_dim))
            aggregate.index_add_(0, chunk.destination, alpha[..., None] * values[chunk.source])
            local = aggregate.reshape(-1, width) if layer.concat else aggregate.mean(dim=1)
            if layer.res is not None:
                local = local + layer.res(x[chunk.start:chunk.end])
            if layer.bias is not None:
                local = local + layer.bias
            output[chunk.start:chunk.end] = local
        return output

    @torch.no_grad()
    def forward(self):
        """Return all-node native scores; caller pays CPU return and filters labels.

        No output cache, label lookup, timing synchronization or effect-based
        selection is hidden here. Eval dropout is identity. Equivalent real
        arithmetic can still differ by FP32 sparse/chunk reduction roundoff.
        """
        self._check_model()
        _require((id(self.raw_features), self.raw_features._version) == self._raw_signature,
                 "common raw features mutated after full setup")
        x = self.raw_features.to(self.device)  # Real raw-X H2D for every full call.
        if self.model_key == "sage":
            for index, layer in enumerate(self.model.convs):
                mean = torch.sparse.mm(self.adjacency, x)
                x = layer.lin_l(mean) + layer.lin_r(x)
                if index < 2:
                    x = F.relu(self.model.bns[index](x))
            scores = x.log_softmax(dim=-1)
        elif self.model_key == "gat":
            for index, layer in enumerate(self.model.convs):
                x = self._gat_layer(layer, x)
                if index < 2:
                    x = F.relu(self.model.norms[index](x))
            scores = x
        else:
            x = self.model.encode_input(x)
            for layer in self.model.layers:
                native = layer.native
                x = native.agg(native.linear(x), self.adjacency)
                if self.model_key == "hgcn":
                    x = native.hyp_act(x)
            scores = self.model.decoder.decode(x, None)
        _require(scores.shape == (self.n, 40), "native full head shape differs")
        return scores

    def resident_bytes(self):
        """Additional full formats; caller separately reports common residency.

        On CPU, device format aliases CPU format and is counted only once.
        Model parameters, inherited CPU graph/features and caller output caches
        are NOT included in this additional-format subtotal.
        """
        if self.cpu_adjacency is not None:
            cpu = (self.cpu_adjacency.crow_indices(), self.cpu_adjacency.col_indices(), self.cpu_adjacency.values())
            device = (self.adjacency.crow_indices(), self.adjacency.col_indices(), self.adjacency.values())
        else:
            cpu = tuple(tensor for chunk in self._chunks for tensor in (chunk.cpu_source, chunk.cpu_destination))
            device = tuple(tensor for chunk in self._chunks for tensor in (chunk.source, chunk.destination))
        return {"additional_cpu_full_format_bytes": _bytes(cpu),
                "additional_cuda_full_format_bytes": _bytes(device) if self.device.type == "cuda" else 0,
                "persistent_cuda_raw_feature_bytes": 0,
                "persistent_hidden_or_output_cache_bytes": 0}

    def report(self):
        with_self = self.model_key != "sage"
        edges = self.nonself_edges + (self.n if with_self else 0)
        layers = [{"forward_layer": index + 1, "unique_source_nodes": self.n,
                   "unique_destination_nodes": self.n, "distinct_nonself_edges": self.nonself_edges,
                   "native_self_edges": self.n if with_self else 0,
                   "processed_native_edges": edges, "computed_destination_states": self.n,
                   "linear_input_rows": self.n * (2 if self.model_key == "sage" else 1),
                   "source_projection_once_per_layer": self.model_key == "gat",
                   "gat_residual_rows": self.n if self.model_key == "gat" else 0,
                   "format_reused_from_setup": True, "correction_invoked": False}
                  for index in range(3)]
        return {"model": self.model_key, "whole_graph_forward": True, "layers": layers,
                "raw_feature_transfer_rows_each_forward": self.n,
                "raw_feature_transfer_bytes_each_forward": self.raw_features.numel() * self.raw_features.element_size(),
                "same_forward_projection_reuse_only": True, "hidden_cache_across_forwards": False,
                "gat_complete_row_chunks": len(self._chunks),
                "gat_chunk_rows_limit": self.gat_chunk_rows, "gat_chunk_edges_limit": self.gat_chunk_edges,
                "gat_actual_max_chunk_edges": max((chunk.source.numel() for chunk in self._chunks), default=0),
                "score_permission": "Only separately authorized target partitions; no test scoring",
                "residency": self.resident_bytes()}


__all__ = ["ExactFullGraph"]

"""UNEXECUTED native HGCN/HyboNet rectangular-block wrappers.

Native layer/encoder/decoder instances are injected; this file never imports
an upstream package, reads task data, or loads a checkpoint. It is a model
interface, not an arxiv trainer or an accepted scientific configuration.

Read-only source contracts:
* mature-hgcn-validation/upstream: layers/hyp_layers.py, models/encoders.py,
  models/decoders.py, models/base_models.py, manifolds/hyperboloid.py,
  utils/data_utils.py.
* icml-compute-allocation-v1/code-v4/native-hybonet: the corresponding files
  plus manifolds/lorentz.py. This vendored source contains an optional
  _stage1_observer hook; wrappers reject active observers.

Only fixed-curvature Hyperboloid HGCN and fixed unit-radius, non-attention
Lorentz HyboNet are supported. Native linear maps, aggregation normalization,
clamps, activation and task decoders are called unchanged. The correction
requires separate qualification of radius-explicit geometry; it does not
claim equivalence to every clipped native log/exp routine or repair native
clamps. CorrectionOutput.step is the geometry's packed tangent payload,
not an ambient Lorentz vector; only geometry methods may interpret it.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn

from core import CorrectionOutput, SharedCorrection
from geometry import LorentzGeometry
from shared_blocks import (NativeBlock, check_features, require_uniform_or_full,
                           validate_block_chain)


def _require(condition: Tensor | bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _fixed_scalar(value: Tensor | float, name: str) -> float:
    if isinstance(value, Tensor):
        _require(value.numel() == 1 and not value.requires_grad,
                 f"{name} requires a fixed scalar; learnable curvature unsupported")
        result = float(value.detach())
    else:
        result = float(value)
    _require(math.isfinite(result) and result > 0, f"invalid {name}")
    return result


def _native_class(module: object, expected: str) -> None:
    _require(type(module).__name__ == expected,
             f"injected native {expected} required; alternative operator needs review")


def _hyperboloid(manifold: object) -> None:
    _require(getattr(manifold, "name", None) == "Hyperboloid",
             "HGCN requires native Hyperboloid, not converted Poincare coordinates")


def _unit_lorentz(manifold: object) -> None:
    _native_class(manifold, "Lorentz")
    _require(_fixed_scalar(manifold.k, "Lorentz k=R^2") == 1.0,
             "native LorentzLinear/Agg hard-code unit radius; require k=1")


@dataclass(frozen=True)
class FixedSelfMeanContract:
    """Explicit declaration supplied by the graph preparer, never inferred.

The original graph must be binary, unweighted, simple and without diagonal
edges. Upstream process(..., normalize_adj=True) then computes D^-1(A+I).
An original self edge, weighted graph, symmetric GCN normalization or sampled
row re-normalization changes the model and is rejected here. Block metadata
alone cannot prove the source graph conditions; its preparer must audit them.
"""

    normalization: str
    original_graph_is_simple_unweighted: bool
    original_graph_has_no_self_edges: bool

    def validate(self) -> None:
        _require(self.normalization == "row_normalized_A_plus_I",
                 "only native row-normalized A+I is supported")
        _require(self.original_graph_is_simple_unweighted is True,
                 "weighted or multigraph normalization needs a different contract")
        _require(self.original_graph_has_no_self_edges is True,
                 "original self edges would change the fixed self coefficient")


def fixed_self_adjacency(block: NativeBlock, like: Tensor,
                         contract: FixedSelfMeanContract) -> Tensor:
    """Native rectangular support: self=1/(N+1), each sampled edge=N/k/(N+1).

For k=N this is exactly the registered native row mean. For uniform WOR
sampling, only the *pre-nonlinearity linear support* is HT-unbiased. Neither
the native nonlinear aggregate nor the proposed correction is called unbiased.
"""
    contract.validate()
    require_uniform_or_full(block)
    _require(like.device == block.src_ids.device, "block/feature device mismatch")
    _require(not ((block.nonself_degree > 0) & (block.counts == 0)).any(),
             "positive population needs at least one sampled neighbor")
    n = block.nonself_degree.to(like.dtype)
    self_weight = (n + 1).reciprocal()
    row = block.edge_dst
    neighbor_weight = (n[row] / block.counts[row].to(like.dtype)) * self_weight[row]
    src, dst = block.with_self_edges()
    return torch.sparse_coo_tensor(
        torch.stack((dst, src)), torch.cat((neighbor_weight, self_weight)),
        (block.num_dst, block.num_src), dtype=like.dtype,
        device=like.device).coalesce()


@dataclass
class HyperbolicLayerOutput:
    points: Tensor
    # Diagnostics cover only potentially eligible partial rows. None means
    # statistics were never constructed (off/full/all k<3). Never a loss.
    correction: CorrectionOutput | None
    correction_rows: Tensor


def _correct(*, base: Tensor, messages: Tensor, block: NativeBlock,
             correction: SharedCorrection | None, enabled: bool,
             geometry: LorentzGeometry) -> HyperbolicLayerOutput:
    empty = block.dst_ids.new_empty(0)
    if correction is None or not enabled:
        return HyperbolicLayerOutput(base, None, empty)
    rows = torch.nonzero((block.counts >= 3)
                         & (block.counts < block.nonself_degree), as_tuple=False).flatten()
    if rows.numel() == 0:
        # Exact native identity, with no B x full-degree x D allocation.
        return HyperbolicLayerOutput(base, None, rows)
    local_row = torch.full((block.num_dst,), -1, dtype=torch.long, device=base.device)
    local_row[rows] = torch.arange(rows.numel(), device=base.device)
    selected_edges = local_row[block.edge_dst] >= 0
    dst = local_row[block.edge_dst[selected_edges]]
    slot = block.edge_slot[selected_edges]
    width = int(block.counts[rows].max())
    padded = messages.new_zeros((rows.numel(), width, messages.shape[-1]))
    padded[dst, slot] = messages[block.edge_src[selected_edges]]
    mask = torch.zeros((rows.numel(), width), dtype=torch.bool, device=base.device)
    mask[dst, slot] = True
    weights = mask.to(base.dtype) / block.counts[rows, None].to(base.dtype)
    population = block.nonself_degree[rows]
    n = population.to(base.dtype)
    output = correction(
        base=base[rows], self_message=messages[block.root_index[rows]],
        messages=padded, neighbor_mask=mask, neighbor_weights=weights,
        population=population, neighbor_mass=n / (n + 1), geometry=geometry)
    return HyperbolicLayerOutput(base.index_copy(0, rows, output.points), output, rows)


def _check_correction(correction: SharedCorrection | None, ambient_dim: int) -> None:
    if correction is not None:
        _require(correction.intrinsic_dim == ambient_dim - 1,
                 "hyperbolic correction dimension is ambient channels minus one")


class HGCNRectangularLayer(nn.Module):
    """native HypLinear -> native HypAgg -> correction(c_in) -> native HypAct."""

    def __init__(self, native_layer: nn.Module, *, contract: FixedSelfMeanContract,
                 correction: SharedCorrection | None = None):
        super().__init__()
        _native_class(native_layer, "HyperbolicGraphConvolution")
        _native_class(native_layer.linear, "HypLinear")
        _native_class(native_layer.agg, "HypAgg")
        _native_class(native_layer.hyp_act, "HypAct")
        contract.validate()
        self.native = native_layer
        self.contract = contract
        self.correction = correction
        self._check_native()
        _check_correction(correction, native_layer.linear.out_features)

    def _check_native(self) -> float:
        layer = self.native
        _require(not layer.agg.use_att and not layer.agg.local_agg,
                 "only native non-attention origin aggregation is supported")
        for module in (layer.linear, layer.agg, layer.hyp_act):
            _hyperboloid(module.manifold)
        curvature = _fixed_scalar(layer.agg.c, "HGCN aggregate c_in")
        _require(curvature == _fixed_scalar(layer.linear.c, "HGCN linear c_in")
                 == _fixed_scalar(layer.hyp_act.c_in, "HGCN activation c_in"),
                 "HGCN linear/aggregate/activation input curvatures differ")
        _fixed_scalar(layer.hyp_act.c_out, "HGCN activation c_out")
        return curvature

    def _apply(self, fn):
        # Upstream fixed c tensors are ordinary attrs, not registered buffers.
        # Move them with the wrapper without changing mathematical values or
        # introducing trainable curvature. The injected modules remain native.
        super()._apply(fn)
        for module, attrs in ((self.native.linear, ("c",)),
                              (self.native.agg, ("c",)),
                              (self.native.hyp_act, ("c_in", "c_out"))):
            for name in attrs:
                value = getattr(module, name)
                if isinstance(value, Tensor):
                    setattr(module, name, fn(value))
        return self

    def forward(self, x: Tensor, block: NativeBlock, *,
                correction_enabled: bool = True) -> HyperbolicLayerOutput:
        curvature = self._check_native()
        check_features(x, block, self.native.linear.in_features)
        adjacency = fixed_self_adjacency(block, x, self.contract)
        messages = self.native.linear(x)
        base = self.native.agg(messages, adjacency)
        geometry = LorentzGeometry(messages.shape[-1] - 1, radius=1 / math.sqrt(curvature))
        result = _correct(base=base, messages=messages, block=block,
                          correction=self.correction, enabled=correction_enabled,
                          geometry=geometry)
        # In particular, c_out is never used for the correction before HypAct.
        return HyperbolicLayerOutput(self.native.hyp_act(result.points),
                                     result.correction, result.correction_rows)


class HyboNetRectangularLayer(nn.Module):
    """native LorentzLinear -> native LorentzAgg (including abs/clamp) -> core."""

    def __init__(self, native_layer: nn.Module, *, contract: FixedSelfMeanContract,
                 correction: SharedCorrection | None = None):
        super().__init__()
        _native_class(native_layer, "LorentzGraphConvolution")
        _native_class(native_layer.linear, "LorentzLinear")
        _native_class(native_layer.agg, "LorentzAgg")
        contract.validate()
        self.native = native_layer
        self.contract = contract
        self.correction = correction
        self._check_native()
        _check_correction(correction, native_layer.linear.out_features)

    def _check_native(self) -> None:
        layer = self.native
        _require(not layer.agg.use_att and not layer.agg.local_agg,
                 "only native non-attention Lorentz aggregation is supported")
        _unit_lorentz(layer.linear.manifold)
        _unit_lorentz(layer.agg.manifold)
        _require(getattr(layer.agg, "_stage1_observer", None) is None,
                 "historical observer/caching hook must be absent")

    def forward(self, x: Tensor, block: NativeBlock, *,
                correction_enabled: bool = True) -> HyperbolicLayerOutput:
        self._check_native()
        check_features(x, block, self.native.linear.in_features)
        adjacency = fixed_self_adjacency(block, x, self.contract)
        messages = self.native.linear(x)
        base = self.native.agg(messages, adjacency)
        # Invalid native points from a clamp branch fail geometry validation;
        # no projection/re-normalization is inserted to conceal that failure.
        return _correct(base=base, messages=messages, block=block,
                        correction=self.correction, enabled=correction_enabled,
                        geometry=LorentzGeometry(messages.shape[-1] - 1, radius=1.0))


@dataclass
class HyperbolicModelOutput:
    scores: Tensor
    embeddings: Tensor
    # Empty unless explicitly requested; inference need not retain all layer
    # embeddings/statistics after their downstream use.
    layers: tuple[HyperbolicLayerOutput, ...]


class NativeHyperbolicBlockModel(nn.Module):
    """Trainable native task head and a checked number of actual message layers.

Call with raw Euclidean features for blocks[0].src_ids, excluding time. The
input mapping follows native BaseModel.encode plus encoder preprocessing.
Scores are native decoder outputs, not softmax/log probabilities. Use the
frozen native task loss in a separately reviewed trainer: HGCN NLL/log-softmax
(equivalently cross entropy); HyboNet native Lorentz margin loss. The latter
head contains ManifoldParameter prototypes and needs a manifold-aware native
optimizer. An ordinary Adam trainer is not silently supplied here.

For NC the official encoders take args.num_layers=actual_message_layers+1.
Use actual_message_layers=3 explicitly in the study. No hidden representations
are retained across forwards; returned diagnostics belong to this forward.
"""

    def __init__(self, native_encoder: nn.Module, native_decoder: nn.Module, *,
                 model_kind: str, actual_message_layers: int,
                 contract: FixedSelfMeanContract,
                 corrections: Sequence[SharedCorrection | None]):
        super().__init__()
        _require(model_kind in ("hgcn", "hybonet"), "unknown hyperbolic model")
        _require(type(actual_message_layers) is int and actual_message_layers > 0,
                 "explicit positive actual message layer count required")
        _native_class(native_encoder, "HGCN" if model_kind == "hgcn" else "HyboNet")
        _native_class(native_decoder, "LinearDecoder" if model_kind == "hgcn" else "LorentzDecoder")
        _require(native_encoder.encode_graph is True, "graph encoder required")
        _require(native_decoder.decode_adj is False, "native node-local task head required")
        _fixed_scalar(native_encoder.c, "encoder curvature configuration")
        _fixed_scalar(native_decoder.c, "decoder curvature configuration")
        layers = list(native_encoder.layers)
        _require(len(layers) == actual_message_layers == len(corrections),
                 "actual native message layer count or correction count mismatch")
        self.model_kind = model_kind
        self.actual_message_layers = actual_message_layers
        self.decoder = native_decoder
        self.input_manifold = native_encoder.manifold
        wrapper = HGCNRectangularLayer if model_kind == "hgcn" else HyboNetRectangularLayer
        self.layers = nn.ModuleList([
            wrapper(layer, contract=contract, correction=correction)
            for layer, correction in zip(layers, corrections)])
        for left, right in zip(layers, layers[1:]):
            _require(left.linear.out_features == right.linear.in_features,
                     "native layer channel dimensions do not compose")
        _require(layers[-1].linear.out_features == native_decoder.input_dim,
                 "native task head dimension must match final ambient dimension")
        self.raw_feature_dim = layers[0].linear.in_features - 1
        _require(self.raw_feature_dim > 0, "raw feature dimension must exclude one time coordinate")
        if model_kind == "hgcn":
            _hyperboloid(self.input_manifold)
            _hyperboloid(self.decoder.manifold)
            for left, right in zip(layers, layers[1:]):
                _require(_fixed_scalar(left.hyp_act.c_out, "previous c_out")
                         == _fixed_scalar(right.linear.c, "next c_in"),
                         "adjacent HGCN layer curvatures differ")
            _require(_fixed_scalar(layers[-1].hyp_act.c_out, "final c_out")
                     == _fixed_scalar(native_decoder.c, "decoder curvature"),
                     "HGCN decoder curvature must match final HypAct")
            _require(_fixed_scalar(native_encoder.curvatures[0], "encoder input curvature")
                     == _fixed_scalar(layers[0].linear.c, "first c_in"),
                     "HGCN input mapping curvature mismatch")
        else:
            _unit_lorentz(self.input_manifold)
            _unit_lorentz(self.decoder.manifold)
            _require(native_decoder.use_bias is True and hasattr(native_decoder, "bias"),
                     "native LorentzDecoder.decode unconditionally adds bias; require bias=True")

    def _apply(self, fn):
        super()._apply(fn)
        # Native HGCN decoder c is also an ordinary fixed tensor attr.
        if self.model_kind == "hgcn" and isinstance(self.decoder.c, Tensor):
            self.decoder.c = fn(self.decoder.c)
        return self

    def encode_input(self, raw_features: Tensor) -> Tensor:
        _require(raw_features.ndim == 2 and raw_features.shape[1] == self.raw_feature_dim,
                 "raw input features must exclude the Lorentz time coordinate")
        x = torch.cat((raw_features.new_zeros((len(raw_features), 1)), raw_features), dim=-1)
        if self.model_kind == "hybonet":
            return self.input_manifold.expmap0(x)
        c = self.layers[0].native.linear.c
        x = self.input_manifold.proj_tan0(x, c)
        return self.input_manifold.proj(self.input_manifold.expmap0(x, c=c), c=c)

    def forward(self, raw_features: Tensor, blocks: Sequence[NativeBlock], *,
                correction_enabled: bool = True,
                return_diagnostics: bool = False) -> HyperbolicModelOutput:
        _require(len(blocks) == self.actual_message_layers,
                 "one dependency block per actual message layer required")
        validate_block_chain(blocks)
        check_features(raw_features, blocks[0], self.raw_feature_dim)
        x = self.encode_input(raw_features)
        diagnostics = []
        for layer, block in zip(self.layers, blocks):
            result = layer(x, block, correction_enabled=correction_enabled)
            x = result.points
            if return_diagnostics:
                diagnostics.append(result)
        # Both accepted native decoders ignore adjacency. No replacement head.
        scores = self.decoder.decode(x, None)
        return HyperbolicModelOutput(scores, x, tuple(diagnostics))


__all__ = ["FixedSelfMeanContract", "fixed_self_adjacency",
           "HyperbolicLayerOutput", "HGCNRectangularLayer", "HyboNetRectangularLayer",
           "HyperbolicModelOutput", "NativeHyperbolicBlockModel"]

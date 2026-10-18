"""UNQUALIFIED source: native PyG 2.6.1 SAGE/GAT correction adapters.

Read ADAPTERS_EUCLIDEAN.md. No task data, checkpoint IO, sampler or trainer.
SAGE parameters/BN names match historical code-v4 SAGE. GAT follows the local
official GATConv source, with explicit global-ID self mapping for blocks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.utils import softmax

from core import CorrectionOutput, EuclideanGeometry, SharedCorrection
from shared_blocks import (NativeBlock, check_features, mean_aggregate,
                           require_uniform_or_full, validate_block_chain)

Variant = Literal["plain", "candidate", "generic"]


@dataclass
class AdapterTrace:
    corrected_destination_rows: Tensor
    correction: CorrectionOutput | None
    costs: dict[str, int]
    # GAT native edge order: all block nonself edges, then one self per row.
    attention_pre_dropout: Tensor | None = None
    attention_used: Tensor | None = None


def _module(dim: int, variant: Variant, init_seed: int) -> SharedCorrection | None:
    if variant not in ("plain", "candidate", "generic"):
        raise ValueError("unknown correction arm")
    return None if variant == "plain" else SharedCorrection(dim, variant, init_seed=init_seed)


def _correct(*, base: Tensor, values: Tensor, anchor: Tensor,
             block: NativeBlock, module: SharedCorrection | None,
             enabled: bool, neighbor_probability: Tensor | None = None,
             neighbor_mass: Tensor | None = None) -> tuple[Tensor, AdapterTrace]:
    """Pack only potentially active partial rows, with one shared head module.

    base/anchor=B,H,D; values=S,H,D. Corrections on full rows or k<3 are
    exactly zero, so those rows allocate no padded message statistics. This
    does not skip a zero-initialized eligible gate's differentiable path.
    """
    batch, heads, dimension = base.shape
    empty_rows = block.edge_dst.new_empty(0)
    costs = {"correction_rows": 0, "head_rows": 0, "padded_neighbor_slots": 0,
             "padded_value_elements": 0, "packed_nonself_edges": 0}
    trace = AdapterTrace(empty_rows, None, costs)
    if module is None or not enabled or block.is_full:
        return base, trace
    rows = torch.nonzero((block.counts >= 3)
                         & (block.counts < block.nonself_degree), as_tuple=False).flatten()
    if rows.numel() == 0:
        return base, trace
    selected_count = rows.numel()
    max_count = int(block.counts[rows].max())
    row_map = block.edge_dst.new_full((batch,), -1)
    row_map[rows] = torch.arange(selected_count, device=base.device)
    local_row = row_map[block.edge_dst]
    selected_edges = torch.nonzero(local_row >= 0, as_tuple=False).flatten()
    local_row = local_row[selected_edges]
    slot = block.edge_slot[selected_edges]
    message = values.new_zeros((selected_count, max_count, heads, dimension))
    message[local_row, slot] = values[block.edge_src[selected_edges]]
    mask = torch.zeros((selected_count, max_count), dtype=torch.bool, device=base.device)
    mask[local_row, slot] = True
    probability = base.new_zeros((selected_count, max_count, heads))
    if neighbor_probability is None:
        edge_probability = block.counts[block.edge_dst[selected_edges]].to(base.dtype).reciprocal()
        probability[local_row, slot] = edge_probability[:, None].expand(-1, heads)
    else:
        probability[local_row, slot] = neighbor_probability[selected_edges]
    if neighbor_mass is None:
        mass = base.new_ones((selected_count, heads))
    else:
        mass = neighbor_mass[rows]
    result = module(
        base=base[rows].reshape(-1, dimension),
        self_message=anchor[rows].reshape(-1, dimension),
        messages=message.permute(0, 2, 1, 3).reshape(-1, max_count, dimension),
        neighbor_mask=mask[:, None, :].expand(-1, heads, -1).reshape(-1, max_count),
        neighbor_weights=probability.permute(0, 2, 1).reshape(-1, max_count),
        population=block.nonself_degree[rows, None].expand(-1, heads).reshape(-1),
        neighbor_mass=mass.reshape(-1), geometry=EuclideanGeometry(dimension))
    corrected = base.index_copy(0, rows, result.points.reshape(-1, heads, dimension))
    costs = {"correction_rows": selected_count, "head_rows": selected_count * heads,
             "padded_neighbor_slots": selected_count * heads * max_count,
             "padded_value_elements": selected_count * heads * max_count * dimension,
             "packed_nonself_edges": selected_edges.numel()}
    return corrected, AdapterTrace(rows, result, costs)


class CorrectedSAGEConv(nn.Module):
    """Default PyG mean SAGEConv, with correction before lin_l only."""

    def __init__(self, inputs: int, outputs: int, *, variant: Variant = "plain",
                 correction_seed: int = 0):
        super().__init__()
        native = SAGEConv(inputs, outputs, aggr="mean", normalize=False,
                          root_weight=True, project=False, bias=True)
        self.inputs, self.outputs = inputs, outputs
        self.lin_l, self.lin_r = native.lin_l, native.lin_r
        self.correction = _module(inputs, variant, correction_seed)

    def forward(self, x: Tensor, block: NativeBlock, *, correction_enabled: bool = True,
                return_trace: bool = False):
        require_uniform_or_full(block)
        check_features(x, block, self.inputs)
        mean = mean_aggregate(x, block)
        root = x[block.root_index]
        corrected, trace = _correct(
            base=mean[:, None, :], values=x[:, None, :], anchor=root[:, None, :],
            block=block, module=self.correction, enabled=correction_enabled)
        output = self.lin_l(corrected[:, 0, :]) + self.lin_r(root)
        return (output, trace) if return_trace else output


class CorrectedSAGE(nn.Module):
    """Three layers, exactly historical BN -> ReLU -> dropout after layers 0/1.

    BN training updates moments on the destination set of that layer; eval
    uses stored moments. Dropout is applied once to those outputs in train.
    These minibatch BN semantics are not claimed equal to full-graph train BN.
    """

    def __init__(self, inputs: int = 128, hidden: int = 256, outputs: int = 40,
                 dropout: float = .5, *, variant: Variant = "plain",
                 correction_seed: int = 0):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.convs = nn.ModuleList([
            CorrectedSAGEConv(a, b, variant=variant, correction_seed=correction_seed+i)
            for i, (a, b) in enumerate(((inputs, hidden), (hidden, hidden), (hidden, outputs)))])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden), nn.BatchNorm1d(hidden)])
        self.dropout = dropout

    def load_backbone_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Map already-loaded old model_state; no file IO, no optimizer reuse.

        Require every old conv/BN key and reject extras; only this model's
        correction keys may remain missing. Callers own CP loading authority.
        """
        expected = {key for key in self.state_dict() if ".correction." not in key}
        if set(state) != expected:
            raise ValueError(f"backbone state keys differ: missing={sorted(expected-set(state))}, "
                             f"extra={sorted(set(state)-expected)}")
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(".correction." not in key for key in result.missing_keys):
            raise ValueError("backbone state mapping failed")

    def forward(self, x: Tensor, blocks: Sequence[NativeBlock], *,
                correction_enabled: bool = True, return_traces: bool = False):
        if len(blocks) != 3:
            raise ValueError("SAGE requires exactly three message blocks")
        validate_block_chain(blocks)
        traces = []
        for index, (conv, block) in enumerate(zip(self.convs, blocks)):
            if return_traces:
                x, trace = conv(x, block, correction_enabled=correction_enabled, return_trace=True)
                traces.append(trace)
            else:
                x = conv(x, block, correction_enabled=correction_enabled)
            if index < 2:
                x = F.dropout(F.relu(self.bns[index](x)), p=self.dropout, training=self.training)
        output = x.log_softmax(dim=-1)
        return (output, traces) if return_traces else output


class CorrectedGATConv(nn.Module):
    """PyG homogeneous GAT: no edge features; explicit one-self-edge policy.

    Native feature projection is once per unique source. PyG GATConv has no
    internal feature dropout; a model may drop its input once before this call.
    Statistics reuse the pre-attention-dropout softmax of this same forward.
    The true aggregate uses attention dropout, including the self coefficient.
    Correction acts on each head aggregate before concat/mean, residual, bias.
    """

    def __init__(self, inputs: int, head_dim: int, *, heads: int = 1,
                 concat: bool = True, negative_slope: float = .2,
                 attention_dropout: float = 0., bias: bool = True,
                 residual: bool = False, variant: Variant = "plain",
                 correction_seed: int = 0):
        super().__init__()
        if inputs < 1 or head_dim < 1 or heads < 1 or not 0 <= attention_dropout < 1:
            raise ValueError("invalid GAT dimension/dropout")
        native = GATConv(inputs, head_dim, heads=heads, concat=concat,
                         negative_slope=negative_slope, dropout=attention_dropout,
                         add_self_loops=False, edge_dim=None, bias=bias, residual=residual)
        self.inputs, self.head_dim, self.heads = inputs, head_dim, heads
        self.concat, self.negative_slope = concat, negative_slope
        self.attention_dropout = attention_dropout
        self.lin, self.att_src, self.att_dst = native.lin, native.att_src, native.att_dst
        self.res, self.bias = native.res, native.bias
        self.correction = _module(head_dim, variant, correction_seed)

    def forward(self, x: Tensor, block: NativeBlock, *, correction_enabled: bool = True,
                return_trace: bool = False):
        require_uniform_or_full(block)
        check_features(x, block, self.inputs)
        source, destination = block.with_self_edges()
        values = self.lin(x).view(-1, self.heads, self.head_dim)
        root_values = values[block.root_index]
        alpha_source = (values * self.att_src).sum(-1)
        alpha_target = (root_values * self.att_dst).sum(-1)
        logits = F.leaky_relu(alpha_source[source] + alpha_target[destination],
                              self.negative_slope)
        alpha = softmax(logits, destination, num_nodes=block.num_dst)
        used_alpha = F.dropout(alpha, p=self.attention_dropout, training=self.training)
        aggregate = values.new_zeros((block.num_dst, self.heads, self.head_dim))
        aggregate = aggregate.index_add(0, destination, used_alpha[..., None] * values[source])
        probability = mass = None
        if self.correction is not None and correction_enabled and not block.is_full:
            nonself_alpha = alpha[:block.num_edges]
            # Sum is numerically safer than 1-alpha_self at near-unit self
            # attention. It is exactly the native nonself softmax mass in real
            # arithmetic, uses no second softmax, and preserves zero support.
            mass = alpha.new_zeros((block.num_dst, self.heads)).index_add(
                0, block.edge_dst, nonself_alpha)
            tolerance = 64 * torch.finfo(alpha.dtype).eps
            if bool(((mass < 0) | (mass > 1+tolerance)).any()):
                raise ValueError("invalid native nonself attention mass")
            # Limit only probability-sum roundoff; never replace zero mass.
            mass = mass.clamp(max=1.)
            safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
            probability = nonself_alpha / safe_mass[block.edge_dst]
        corrected, trace = _correct(
            base=aggregate, values=values, anchor=root_values, block=block,
            module=self.correction, enabled=correction_enabled,
            neighbor_probability=probability, neighbor_mass=mass)
        output = (corrected.reshape(-1, self.heads*self.head_dim)
                  if self.concat else corrected.mean(dim=1))
        if self.res is not None:
            output = output + self.res(x[block.root_index])
        if self.bias is not None:
            output = output + self.bias
        if return_trace:
            trace.attention_pre_dropout, trace.attention_used = alpha, used_alpha
            trace.costs["projected_unique_sources"] = block.num_src
            trace.costs["native_attention_edges_with_self"] = source.numel()
            return output, trace
        return output


class CorrectedGAT(nn.Module):
    """Configurable three-message-layer PyG GAT scaffold, no trainer claim.

    Hidden heads concat then ELU; output heads average; input feature dropout
    once before each layer. Exact dimensions/dropout/residual are required
    configuration arguments, to be frozen before scientific execution.
    """

    def __init__(self, *, inputs: int, hidden_head_dim: int, outputs: int,
                 hidden_heads: int, output_heads: int, feature_dropout: float,
                 attention_dropout: float, residual: bool,
                 variant: Variant = "plain", correction_seed: int = 0):
        super().__init__()
        if not 0 <= feature_dropout < 1:
            raise ValueError("feature dropout must be in [0,1)")
        width = hidden_head_dim * hidden_heads
        specs = ((inputs, hidden_head_dim, hidden_heads, True),
                 (width, hidden_head_dim, hidden_heads, True),
                 (width, outputs, output_heads, False))
        self.convs = nn.ModuleList([
            CorrectedGATConv(a, b, heads=h, concat=concat,
                             attention_dropout=attention_dropout, residual=residual,
                             variant=variant, correction_seed=correction_seed+i)
            for i, (a, b, h, concat) in enumerate(specs)])
        self.feature_dropout = feature_dropout

    def forward(self, x: Tensor, blocks: Sequence[NativeBlock], *,
                correction_enabled: bool = True, return_traces: bool = False):
        if len(blocks) != 3:
            raise ValueError("GAT requires exactly three message blocks")
        validate_block_chain(blocks)
        traces = []
        for index, (conv, block) in enumerate(zip(self.convs, blocks)):
            x = F.dropout(x, p=self.feature_dropout, training=self.training)
            if return_traces:
                x, trace = conv(x, block, correction_enabled=correction_enabled, return_trace=True)
                traces.append(trace)
            else:
                x = conv(x, block, correction_enabled=correction_enabled)
            if index < 2:
                x = F.elu(x)
        output = x.log_softmax(dim=-1)
        return (output, traces) if return_traces else output

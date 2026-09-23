"""Qualified deterministic full-graph evaluator for R-TSC/HGCN joint v2.

This module replaces only the two HGCN aggregation products during frozen
inference.  It never mutates a model, patches ``torch.spmm``, or supplies a
training/gradient implementation.
"""

from __future__ import annotations

import hashlib
import threading
from contextlib import contextmanager

import torch


BACKEND_ID = "rtsc_hgcn_full_eval_torch251_sparse_bmm_coo_alg2_v1"
_FLAG_LOCK = threading.RLock()


def deterministic_state() -> tuple[bool, bool]:
    """Return both process-global deterministic-algorithm settings."""
    return (
        bool(torch.are_deterministic_algorithms_enabled()),
        bool(torch.is_deterministic_algorithms_warn_only_enabled()),
    )


@contextmanager
def _coo_alg2_flag():
    """Select the PyTorch COO_ALG2 path and restore both state bits exactly."""
    with _FLAG_LOCK:
        before = deterministic_state()
        torch.use_deterministic_algorithms(True, warn_only=False)
        try:
            if deterministic_state() != (True, False):
                raise RuntimeError("failed to enable strict deterministic algorithms")
            yield before
        finally:
            torch.use_deterministic_algorithms(before[0], warn_only=before[1])
            if deterministic_state() != before:
                raise RuntimeError("deterministic-algorithm state was not restored")


def _require_inference_only(*tensors: torch.Tensor) -> None:
    if not torch.is_inference_mode_enabled():
        raise RuntimeError("deterministic evaluation adapter requires torch.inference_mode")
    if any(value.requires_grad for value in tensors):
        raise RuntimeError("deterministic evaluation adapter forbids gradient inputs")


def _validate_spmm_inputs(adjacency: torch.Tensor, dense: torch.Tensor) -> None:
    _require_inference_only(adjacency, dense)
    if adjacency.layout != torch.sparse_coo or adjacency.ndim != 2:
        raise ValueError("a two-dimensional sparse COO adjacency is required")
    if adjacency.shape[0] != adjacency.shape[1] or adjacency.shape[0] != dense.shape[0]:
        raise ValueError("square adjacency and matching dense rows are required")
    if not adjacency.is_coalesced():
        raise ValueError("coalesced duplicate-free COO adjacency is required")
    if dense.ndim != 2 or not dense.is_contiguous():
        raise ValueError("contiguous two-dimensional dense input is required")
    if adjacency.dtype != torch.float32 or dense.dtype != torch.float32:
        raise ValueError("the registered backend is float32 only")
    if adjacency.device != dense.device:
        raise ValueError("adjacency and dense input must use one device")
    indices = adjacency.indices()
    values = adjacency.values()
    int32_max = torch.iinfo(torch.int32).max
    if (
        adjacency.shape[0] > int32_max
        or adjacency.shape[1] > int32_max
        or dense.shape[1] > int32_max
        or adjacency._nnz() > int32_max
    ):
        raise ValueError("PyTorch 2.5.1 sparse BMM int32 dimensions exceeded")
    if indices.dtype != torch.int64 or indices.shape != (2, adjacency._nnz()):
        raise ValueError("canonical int64 COO indices are required")
    if not indices.is_contiguous() or not values.is_contiguous():
        raise ValueError("canonical contiguous COO storage is required")
    if not torch.isfinite(values).all() or not torch.isfinite(dense).all():
        raise ValueError("finite sparse values and dense input are required")
    if indices.numel() and ((indices < 0).any() or (indices >= adjacency.shape[0]).any()):
        raise ValueError("COO indices must be in the explicit [0,N) range")
    if adjacency._nnz() > 1:
        key = indices[0] * adjacency.shape[1] + indices[1]
        if not (key[1:] > key[:-1]).all():
            raise ValueError("strict row-major COO order is required")


def deterministic_spmm(
    adjacency: torch.Tensor,
    dense: torch.Tensor,
    *,
    _bmm=torch.bmm,
) -> torch.Tensor:
    """Candidate ``adjacency @ dense`` through 3D sparse BMM / COO_ALG2.

    PyTorch v2.5.1's CUDA sparse BMM implementation calls cuSPARSE with
    ``opA=NON_TRANSPOSE`` and selects ``CUSPARSE_SPMM_COO_ALG2`` while strict
    deterministic algorithms are enabled.  The zero batch coordinate preserves
    the original row/column order and values.
    """
    _validate_spmm_inputs(adjacency, dense)
    indices = adjacency.indices()
    batch = torch.zeros((1, adjacency._nnz()), dtype=torch.int64, device=indices.device)
    indices3 = torch.cat((batch, indices), dim=0)
    adjacency3 = torch.sparse_coo_tensor(
        indices3,
        adjacency.values(),
        (1, adjacency.shape[0], adjacency.shape[1]),
        dtype=adjacency.dtype,
        device=adjacency.device,
    ).coalesce()
    dense3 = dense.unsqueeze(0)
    with _coo_alg2_flag():
        output3 = _bmm(adjacency3, dense3)
    if output3.shape != (1, adjacency.shape[0], dense.shape[1]):
        raise RuntimeError("unexpected sparse BMM output shape")
    output = output3[0]
    if output.dtype != dense.dtype or output.device != dense.device or not torch.isfinite(output).all():
        raise RuntimeError("invalid sparse BMM output")
    return output


def _model_parts(model):
    corrections = getattr(model, "corrections", None)
    base = getattr(model, "base", model) if corrections is not None else model
    if not hasattr(base, "encoder") or len(base.encoder.layers) != 2:
        raise ValueError("registered two-layer HGCN model required")
    if any(layer.agg.use_att or layer.agg.local_agg for layer in base.encoder.layers):
        raise ValueError("registered nonattention origin aggregation required")
    if corrections is not None and len(corrections) != 2:
        raise ValueError("two correction modules required")
    return base, corrections


def _validate_frozen_model(model, features: torch.Tensor) -> None:
    _require_inference_only(features)
    if model.training:
        raise RuntimeError("evaluation adapter rejects a model in training mode")
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("evaluation adapter requires frozen parameters with no gradients")


def _validate_plan(plan, nodes: int, expected_fanout: int | None) -> None:
    if not hasattr(plan, "matrix") or not hasattr(plan, "fanout"):
        raise ValueError("registered sampling plan required")
    if plan.fanout != expected_fanout:
        label = "full" if expected_fanout is None else f"fanout-{expected_fanout}"
        raise ValueError(f"{label} plan required")
    if expected_fanout is None:
        if not torch.equal(plan.selected, plan.populations):
            raise ValueError("full plan must select every visible nonself message")
    if plan.indices.shape[0] != 2 or plan.populations.shape != (nodes,):
        raise ValueError("plan dimensions differ from model nodes")


def _encode(model, features, plans, *, module_off: bool, expected_fanout: int | None):
    _validate_frozen_model(model, features)
    if len(plans) != 2:
        raise ValueError("two matched layer plans required")
    base, corrections = _model_parts(model)
    if (
        features.ndim != 2
        or features.shape[1] != base.encoder.layers[0].linear.in_features
        or features.device != base.curvature.device
        or features.dtype != base.curvature.dtype
        or features.dtype != torch.float32
    ):
        raise ValueError("registered float32 feature matrix required")
    for plan in plans:
        _validate_plan(plan, len(features), expected_fanout)

    manifold = base.encoder.manifold
    curvature = base.curvature
    current = manifold.proj(
        manifold.expmap0(manifold.proj_tan0(features, curvature), c=curvature),
        c=curvature,
    )
    diagnostics = []
    for layer_number, (layer, plan) in enumerate(zip(base.encoder.layers, plans, strict=True)):
        adjacency = plan.matrix(device=features.device, dtype=features.dtype)
        messages = layer.linear.forward(current)
        tangent = layer.agg.manifold.logmap0(messages, c=layer.agg.c)
        support = deterministic_spmm(adjacency, tangent)
        aggregate = layer.agg.manifold.proj(
            layer.agg.manifold.expmap0(support, c=layer.agg.c), c=layer.agg.c
        )
        if corrections is not None and not module_off:
            adjusted, measured = corrections[layer_number](aggregate, messages, plan, manifold)
            if expected_fanout is None and int(measured["eligible_nodes"]) != 0:
                raise RuntimeError("full-graph correction must remain identity")
            diagnostics.append(measured)
        else:
            adjusted = aggregate
            diagnostics.append(
                {
                    "module_off": True,
                    "eligible_nodes": 0 if expected_fanout is None else None,
                }
            )
        current = layer.hyp_act.forward(adjusted)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("evaluation adapter created parameter gradients")
    return current, diagnostics


def encode_full_evaluation(model, features, plans, *, module_off: bool = False):
    """Accepted full-graph adapter.  Caller must freeze/eval the model first."""
    return _encode(model, features, plans, module_off=module_off, expected_fanout=None)


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def backend_descriptor(adjacency: torch.Tensor, dense: torch.Tensor) -> dict:
    """Record the exact registered operator identity and fixed inputs."""
    _validate_spmm_inputs(adjacency, dense)
    return {
        "backend_id": BACKEND_ID,
        "sparse_layout": str(adjacency.layout),
        "shape": list(adjacency.shape),
        "nnz": adjacency._nnz(),
        "dtype": str(adjacency.dtype),
        "coalesced": adjacency.is_coalesced(),
        "indices_sha256": tensor_sha256(adjacency.indices()),
        "values_sha256": tensor_sha256(adjacency.values()),
        "dense_shape": list(dense.shape),
        "dense_stride": list(dense.stride()),
        "dense_sha256": tensor_sha256(dense),
        "expected_cusparse_algorithm": "CUSPARSE_SPMM_COO_ALG2",
        "opA": "CUSPARSE_OPERATION_NON_TRANSPOSE",
        "scope": "full_graph_evaluation_only",
        "qualification_status": "accepted_for_recorded_runtime_and_fixtures",
    }

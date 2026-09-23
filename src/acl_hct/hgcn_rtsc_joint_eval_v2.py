"""Qualified composite-v2 evaluator for frozen R-TSC/HGCN inference.

The sampled lane retains the frozen correction algebra and changes only the
four ``index_add_`` calls.  Training continues to use ``encode_joint`` and is
deliberately unsupported here.
"""

from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
from importlib import metadata
import math
import os
from pathlib import Path
import platform
import random
import re
import threading

import numpy as np
import torch

from . import hgcn_rtsc_full_eval_v2 as _FULL


COMPOSITE_BACKEND_ID = "rtsc_hgcn_eval_composite_full_det_sampled_det_v2"
FULL_BACKEND_ID = "rtsc_hgcn_full_eval_torch251_sparse_bmm_coo_alg2_v1"
QUALIFIED_PRIVATE_FULL_ADAPTER_SHA256 = (
    "847cb64dbd5485f3c4a6779b3fd81e74ddb66b8feda02292c8e87c23e80595b1"
)
QUALIFIED_PRIVATE_COMPOSITE_ADAPTER_SHA256 = (
    "ee6740386a00ecefb01991069b2bffd934d56d9d5eafedeb9b32be74a105f145"
)
SAMPLED_BACKEND_ID = (
    "rtsc_hgcn_sampled_eval_torch251_coo_alg2_deterministic_index_add_v1"
)
QUALIFICATION_ARTIFACTS = {
    "deterministic_full_eval_gpu_qualification_release":
        "ed86e640464ac29dee7cd1b08a275b09790bd8ebf881c7a968c384574a223fbd",
    "supervisor_full_eval_qualification_review":
        "c23e110aea0be8b13c0bd99238e4655511f7c0d2ddd6b7ef88a34304cb786eaa",
    "supervisor_full_eval_qualification_provenance":
        "ed6bc6290a4051fabc6687ca3acf3a011f81c6cca41040f19b7bd710c44a331a",
    "deterministic_full_eval_independent_numerical_review_v2":
        "5db6e3594d3c2512af73b3a05effa742403bb6118a890519b3dd5887bfd04234",
    "composite_sampled_eval_gpu_qualification_release":
        "df021af0a4a7b42757b179a1ba29e44ec1d4d5e380751e5bc5a8f607b2fd2598",
    "supervisor_sampled_eval_qualification_review":
        "e55209772a4020e9b09c56204794823d69bd7c35123a80d07eafba11ec5ba280",
    "supervisor_sampled_eval_qualification_provenance":
        "44e97446357683bbee281de81932027d546099ea7a3aa849107740da4f54c873",
    "composite_sampled_eval_independent_numerical_review":
        "044526531c1abafbcfd62845b92c70288b273ab8a3631ae2d26a2948b14ce33c",
}
EVALUATION_AMENDMENT_SHA256 = (
    "551a4465aee7f51f6f4eb90535d4cdcc2fbc81c839eb1199eb95353810837c9e"
)
REGISTERED_FANOUTS = frozenset((4, 8, 16))
EVALUATION_SOURCE_MODULES = (
    "hgcn_rtsc_full_eval_v2", "hgcn_rtsc_joint_eval_v2",
    "hgcn_rtsc_joint_worker", "hgcn_rtsc_joint_protocol",
    "hgcn_rtsc_joint_entry",
)
_EVALUATION_LOCK = threading.RLock()
_FLAG_LOCK = threading.RLock()

if _FULL.BACKEND_ID != FULL_BACKEND_ID:
    raise RuntimeError("production full evaluator backend identity mismatch")


def deterministic_state() -> tuple[bool, bool]:
    return (
        bool(torch.are_deterministic_algorithms_enabled()),
        bool(torch.is_deterministic_algorithms_warn_only_enabled()),
    )


def _index_add(target, dim, index, source):
    return target.index_add_(dim, index, source)


def _deterministic_index_add(target, dim, index, source, *, _operation=None):
    """Run exactly one index_add under strict mode and restore both state bits."""
    if not torch.is_inference_mode_enabled():
        raise RuntimeError("deterministic sampled reduction requires inference_mode")
    if any(value.requires_grad for value in (target, source)):
        raise RuntimeError("deterministic sampled reduction forbids gradient tensors")
    operation = _index_add if _operation is None else _operation
    with _FLAG_LOCK:
        before = deterministic_state()
        torch.use_deterministic_algorithms(True, warn_only=False)
        try:
            if deterministic_state() != (True, False):
                raise RuntimeError("failed to enable strict deterministic algorithms")
            result = operation(target, dim, index, source)
        finally:
            torch.use_deterministic_algorithms(before[0], warn_only=before[1])
            if deterministic_state() != before:
                raise RuntimeError("deterministic-algorithm state was not restored")
    return result


def _sampled_correction(correction, aggregate, messages, plan, manifold):
    """Frozen TangentCorrection.forward algebra with four locally strict adds."""
    if (aggregate.shape != messages.shape or aggregate.ndim != 2
            or aggregate.dtype not in (torch.float32, torch.float64)
            or aggregate.device != messages.device or aggregate.dtype != messages.dtype):
        raise ValueError('matching native aggregate and transformed-message matrices required')
    nodes = len(aggregate)
    row, col = correction._plan_edges(plan, nodes)
    eligible = (plan.selected >= 3) & (plan.selected < plan.populations)
    ids_cpu = torch.nonzero(eligible, as_tuple=False).flatten()
    if not len(ids_cpu):
        return aggregate, {'normalized_step_mean': aggregate.sum() * 0,
                           'eligible_nodes': 0, 'clipped_nodes': 0,
                           'near_zero_own': 0, 'near_zero_neighbors': 0,
                           'selected_neighbors': 0, 'projected_nodes': 0}
    edge_mask = (row != col) & eligible[row]
    global_row, global_col = row[edge_mask], col[edge_mask]
    lookup = torch.full((nodes,), -1, dtype=torch.long)
    lookup[ids_cpu] = torch.arange(len(ids_cpu))
    compact_row = lookup[global_row].to(aggregate.device)
    global_col = global_col.to(aggregate.device)
    ids = ids_cpu.to(aggregate.device)
    base = aggregate.index_select(0, ids)
    own_message = messages.index_select(0, ids)
    selected = plan.selected[ids_cpu].to(device=aggregate.device, dtype=aggregate.dtype)
    population = plan.populations[ids_cpu].to(device=aggregate.device, dtype=aggregate.dtype)
    conformal = correction._conformal(base)
    own_direction = correction.log_at(base, own_message)
    summed = torch.zeros_like(base)
    near_zero_neighbors = 0
    for start in range(0, len(global_col), correction.chunk_edges):
        rr = compact_row[start:start + correction.chunk_edges]
        cc = global_col[start:start + correction.chunk_edges]
        tangent = correction.log_at(base.index_select(0, rr), messages.index_select(0, cc))
        near_zero_neighbors += int((torch.linalg.vector_norm(tangent.detach(), dim=-1) <= correction.eps).sum())
        _deterministic_index_add(summed, 0, rr, tangent)
    average = summed / selected[:, None]
    variance_sum = base.new_zeros((len(base),))
    third_sum = torch.zeros_like(base)
    covariance_sum = torch.zeros_like(base)
    for start in range(0, len(global_col), correction.chunk_edges):
        rr = compact_row[start:start + correction.chunk_edges]
        cc = global_col[start:start + correction.chunk_edges]
        centered = correction.log_at(base.index_select(0, rr), messages.index_select(0, cc)) - average.index_select(0, rr)
        lam = conformal.index_select(0, rr)
        squared = (lam * lam * centered * centered).sum(dim=-1)
        _deterministic_index_add(variance_sum, 0, rr, squared)
        _deterministic_index_add(third_sum, 0, rr, squared[:, None] * centered)
        own = own_direction.index_select(0, rr)
        _deterministic_index_add(covariance_sum, 0, rr, (lam * lam * centered * own).sum(dim=-1, keepdim=True) * centered)
    variance = variance_sum / (selected - 1)
    third = third_sum / selected[:, None]
    covariance_own = covariance_sum / (selected - 1)[:, None]
    own_sq = (conformal * conformal * own_direction * own_direction).sum(dim=-1)
    average_sq = (conformal * conformal * average * average).sum(dim=-1)
    third_length = conformal.squeeze(-1) * torch.linalg.vector_norm(third, dim=-1)
    own_length = torch.linalg.vector_norm(conformal * own_direction, dim=-1)
    cosine = ((conformal * conformal * own_direction * third).sum(dim=-1) /
              (own_length * third_length + correction.eps))
    own_covariance = (conformal * conformal * own_direction * covariance_own).sum(dim=-1)
    features = torch.stack((torch.log1p(population), selected / population,
                            torch.log1p(selected), torch.full_like(selected, math.log1p(correction.c)),
                            torch.log1p(correction.c * variance), torch.log1p(correction.c * own_sq),
                            torch.log1p(correction.c * average_sq),
                            third_length / (variance.pow(1.5) + correction.eps),
                            cosine, own_covariance / (own_sq * variance + correction.eps)), dim=-1)
    coeff = 2 * torch.tanh(correction.coefficients(features))
    basis = torch.stack((correction.c * variance[:, None] * own_direction,
                         correction.c * third, correction.c * covariance_own), dim=1)
    q = (population / (population + 1)).square() * (1 - selected / population) / selected
    raw = q[:, None] * (coeff[:, :, None] * basis).sum(dim=1)
    raw_norm = conformal.squeeze(-1) * torch.linalg.vector_norm(raw, dim=-1)
    clip_scale = (correction.tau / raw_norm.clamp_min(correction.eps)).clamp(max=1)
    step = raw * clip_scale[:, None]
    step_norm = conformal.squeeze(-1) * torch.linalg.vector_norm(step, dim=-1)
    unprojected = correction.exp_at(base, step)
    corrected = manifold.proj(unprojected, aggregate.new_tensor(correction.c))
    if not (torch.isfinite(features).all() and torch.isfinite(corrected).all()
            and torch.isfinite(step_norm).all()):
        raise ValueError('nonfinite local correction')
    result = torch.index_copy(aggregate, 0, ids, corrected)
    penalty = (step_norm.square().sum() / nodes) / (correction.tau * correction.tau)
    return result, {'normalized_step_mean': penalty, 'eligible_nodes': len(ids_cpu),
                    'clipped_nodes': int((raw_norm.detach() > correction.tau).sum()),
                    'near_zero_own': int((torch.linalg.vector_norm(own_direction.detach(), dim=-1) <= correction.eps).sum()),
                    'near_zero_neighbors': near_zero_neighbors,
                    'selected_neighbors': len(global_col),
                    'projected_nodes': int((unprojected.detach() != corrected.detach()).any(dim=-1).sum())}


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


def _validate_frozen_model(model, features):
    if not torch.is_inference_mode_enabled():
        raise RuntimeError("composite evaluation requires inference_mode")
    if features.requires_grad:
        raise RuntimeError("composite evaluation forbids gradient inputs")
    if model.training:
        raise RuntimeError("composite evaluation rejects a model in training mode")
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("composite evaluation requires frozen parameters with no gradients")


def _validate_sampled_plans(plans, nodes):
    if len(plans) != 2:
        raise ValueError("two matched sampled plans required")
    fanouts = {plan.fanout for plan in plans}
    if len(fanouts) != 1 or next(iter(fanouts)) not in REGISTERED_FANOUTS:
        raise ValueError("matching registered fanout-4/8/16 plans required")
    for plan in plans:
        if plan.indices.shape[0] != 2 or plan.populations.shape != (nodes,):
            raise ValueError("sampled plan dimensions differ from model nodes")
    return next(iter(fanouts))


def encode_full_evaluation(model, features, plans, *, module_off=False):
    """Delegate the accepted full lane to the immutable qualified adapter."""
    with _EVALUATION_LOCK:
        return _FULL.encode_full_evaluation(
            model, features, plans, module_off=module_off
        )


def encode_sampled_evaluation(model, features, plans, *, module_off=False):
    """Candidate sampled lane; not eligible for science before qualification."""
    with _EVALUATION_LOCK:
        _validate_frozen_model(model, features)
        fanout = _validate_sampled_plans(plans, len(features))
        base, corrections = _model_parts(model)
        if corrections is None and not module_off:
            # Plain arms have no correction reductions; the aggregation lane is still bound.
            module_off = True
        if (
            features.ndim != 2
            or features.shape[1] != base.encoder.layers[0].linear.in_features
            or features.device != base.curvature.device
            or features.dtype != base.curvature.dtype
            or features.dtype != torch.float32
        ):
            raise ValueError("registered float32 feature matrix required")
        manifold = base.encoder.manifold
        curvature = base.curvature
        current = manifold.proj(
            manifold.expmap0(manifold.proj_tan0(features, curvature), c=curvature),
            c=curvature,
        )
        diagnostics = []
        for layer_number, (layer, plan) in enumerate(
            zip(base.encoder.layers, plans, strict=True)
        ):
            adjacency = plan.matrix(device=features.device, dtype=features.dtype)
            messages = layer.linear.forward(current)
            tangent = layer.agg.manifold.logmap0(messages, c=layer.agg.c)
            support = _FULL.deterministic_spmm(adjacency, tangent)
            aggregate = layer.agg.manifold.proj(
                layer.agg.manifold.expmap0(support, c=layer.agg.c), c=layer.agg.c
            )
            if corrections is not None and not module_off:
                adjusted, measured = _sampled_correction(
                    corrections[layer_number], aggregate, messages, plan, manifold
                )
                diagnostics.append(measured)
            else:
                adjusted = aggregate
                diagnostics.append(
                    {"module_off": True, "eligible_nodes": None, "fanout": fanout}
                )
            current = layer.hyp_act.forward(adjusted)
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("composite evaluation created parameter gradients")
        return current, diagnostics


def backend_descriptor():
    return {
        "evaluation_suite_id": COMPOSITE_BACKEND_ID,
        "full_backend_id": FULL_BACKEND_ID,
        "sampled_backend_id": SAMPLED_BACKEND_ID,
        "qualification_status": "accepted_for_recorded_runtime_and_fixtures",
        "qualified_private_full_adapter_sha256":
            QUALIFIED_PRIVATE_FULL_ADAPTER_SHA256,
        "qualified_private_composite_adapter_sha256":
            QUALIFIED_PRIVATE_COMPOSITE_ADAPTER_SHA256,
        "qualification_artifacts": dict(QUALIFICATION_ARTIFACTS),
        "evaluation_amendment_sha256": EVALUATION_AMENDMENT_SHA256,
        "registered_fanouts": sorted(REGISTERED_FANOUTS),
        "sampled_strict_scope": "only four frozen correction index_add_ calls",
        "training_supported": False,
    }


def encode_evaluation(model, features, plans, *, module_off=False):
    """Route exactly one frozen forward to the qualified full or sampled lane."""
    if len(plans) != 2:
        raise ValueError("two matched evaluation plans required")
    fanouts = {plan.fanout for plan in plans}
    if len(fanouts) != 1:
        raise ValueError("mixed full/sampled or mixed-fanout plans are forbidden")
    fanout = next(iter(fanouts))
    if fanout is None:
        return encode_full_evaluation(model, features, plans, module_off=module_off)
    if fanout in REGISTERED_FANOUTS:
        return encode_sampled_evaluation(model, features, plans,
                                         module_off=module_off)
    raise ValueError("unregistered evaluation fanout")


def evaluation_source_hashes():
    """Return normalized-LF hashes of every production evaluation route file."""
    root = Path(__file__).parent
    hashes = {}
    for name in EVALUATION_SOURCE_MODULES:
        path = root / (name + ".py")
        hashes["acl_hct/" + name + ".py"] = hashlib.sha256(
            path.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
    return hashes


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _static_runtime_descriptor(device_type, device_index):
    target = torch.device(device_type, device_index)
    result = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": {
            "version": torch.version.cuda,
            "source": "torch.version.cuda; build metadata, not cuSPARSE version",
        },
        "device_type": target.type,
        "device_index": target.index,
    }
    if target.type == "cuda":
        index = torch.cuda.current_device() if target.index is None else target.index
        properties = torch.cuda.get_device_properties(index)
        try:
            package_version = metadata.version("nvidia-cusparse-cu12")
        except metadata.PackageNotFoundError:
            package_version = None
        libraries = {}
        maps_path = Path('/proc/self/maps')
        if maps_path.is_file():
            for line in maps_path.read_text(
                    encoding='utf-8', errors='replace').splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) != 6:
                    continue
                library = Path(fields[5].removesuffix(' (deleted)'))
                if (not library.is_absolute()
                        or 'libcusparse.so' not in library.name):
                    continue
                try:
                    resolved = library.resolve(strict=True)
                    size = resolved.stat().st_size
                except OSError:
                    continue
                libraries[str(resolved)] = {
                    "mapped_path": str(library),
                    "resolved_path": str(resolved),
                    "resolved_basename": resolved.name,
                    "bytes": size,
                    "sha256": _sha256_file(resolved),
                }
        result["device_index"] = index
        result["device_name"] = properties.name
        result["compute_capability"] = [properties.major, properties.minor]
        result["total_memory_bytes"] = properties.total_memory
        result["cusparse"] = {
            "package_metadata": {
                "distribution": "nvidia-cusparse-cu12",
                "version": package_version,
                "source": "importlib.metadata.version",
            },
            "loaded_libraries": [libraries[key] for key in sorted(libraries)],
            "loaded_library_source":
                "/proc/self/maps after deterministic sparse BMM",
        }
        result["qualified_recorded_runtime"] = bool(
            torch.__version__ == "2.5.1+cu124"
            and torch.version.cuda == "12.4"
            and properties.name == "NVIDIA L40"
            and (properties.major, properties.minor) == (8, 9)
            and package_version == "12.3.1.170"
            and len(libraries) == 1
            and next(iter(libraries.values()), {}).get("resolved_basename")
                == "libcusparse.so.12"
            and next(iter(libraries.values()), {}).get("bytes") == 281313984
        )
    else:
        result["device_name"] = None
        result["compute_capability"] = None
        result["total_memory_bytes"] = None
        result["cusparse"] = None
        result["qualified_recorded_runtime"] = False
    return result


def runtime_descriptor(device=None):
    target = torch.device('cpu') if device is None else torch.device(device)
    result = copy.deepcopy(_static_runtime_descriptor(
        target.type, target.index))
    result["cublas_workspace_config"] = os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG")
    result["deterministic_algorithms"] = {
        "enabled": deterministic_state()[0],
        "warn_only": deterministic_state()[1],
    }
    if target.type == 'cuda':
        if (result["cublas_workspace_config"] not in (None, '')
                or result["deterministic_algorithms"]
                != {"enabled": False, "warn_only": False}
                or not result["qualified_recorded_runtime"]):
            raise RuntimeError(
                "scientific evaluation requires the qualified recorded runtime")
    return result


def _tensor_sha256(value):
    if torch.is_tensor(value):
        array = value.detach().cpu().contiguous().numpy()
    else:
        array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def plan_identities(plans, *, phase, fanout, step=None, repeat=None):
    """Bind the actual two plan tensors and their registered coordinates."""
    if (not isinstance(phase, str) or not phase
            or step is not None and type(step) is not int
            or repeat is not None and type(repeat) is not int
            or len(plans) != 2 or fanout not in (None, *sorted(REGISTERED_FANOUTS))):
        raise ValueError("exact phase/step/repeat and two registered plans required")
    rows = []
    for layer, plan in enumerate(plans):
        if plan.fanout != fanout:
            raise ValueError("plan fanout differs from registered evidence coordinate")
        rows.append({
            "phase": phase,
            "fanout": fanout,
            "step": step,
            "repeat": repeat,
            "layer": layer,
            "graph_hash": plan.graph_hash,
            "indices_sha256": _tensor_sha256(plan.indices),
            "weights_fp64_sha256": _tensor_sha256(plan.weights),
            "inclusion_probabilities_sha256":
                _tensor_sha256(plan.inclusion_probabilities),
            "nonself_populations_sha256": _tensor_sha256(plan.populations),
            "nonself_selected_sha256": _tensor_sha256(plan.selected),
        })
    return rows


_HASH40 = re.compile(r"[0-9a-f]{40}")
_HASH64 = re.compile(r"[0-9a-f]{64}")


def make_evaluation_binding(*, evaluation_source_commit,
                            training_source_commit, config_sha256,
                            replay_policy_sha256, checkpoint_sha256,
                            model_state_sha256, plans, plan_phase, fanout,
                            repeat, device, plan_step=None,
                            checkpoint_artifact_kind="checkpoint_file"):
    """Build the required evaluator/checkpoint/plan provenance envelope."""
    if (not _HASH40.fullmatch(str(evaluation_source_commit))
            or not _HASH40.fullmatch(str(training_source_commit))):
        raise ValueError("separate 40-hex evaluation and training commits required")
    for value in (config_sha256, replay_policy_sha256, checkpoint_sha256,
                  model_state_sha256):
        if not _HASH64.fullmatch(str(value)):
            raise ValueError("64-hex config, policy, checkpoint and state hashes required")
    if checkpoint_artifact_kind == "checkpoint_file":
        checkpoint_artifact = {
            "kind": checkpoint_artifact_kind,
            "byte_sha256": checkpoint_sha256,
            "model_state_sha256": model_state_sha256,
        }
    elif checkpoint_artifact_kind == "serialized_state_fixture_or_probe":
        checkpoint_artifact = {
            "kind": checkpoint_artifact_kind,
            "serialized_state_sha256": checkpoint_sha256,
            "model_state_sha256": model_state_sha256,
        }
    else:
        raise ValueError("registered checkpoint artifact kind required")
    result = {
        "schema_version": 2,
        "evaluation_suite": backend_descriptor(),
        "evaluator_source": {
            "commit": evaluation_source_commit,
            "sha256_normalized_lf": evaluation_source_hashes(),
        },
        "training_source_commit": training_source_commit,
        "checkpoint_artifact": checkpoint_artifact,
        "config_sha256": config_sha256,
        "replay_policy_sha256": replay_policy_sha256,
        "runtime": runtime_descriptor(device),
        "plans": plan_identities(plans, phase=plan_phase, fanout=fanout,
                                  step=plan_step, repeat=repeat),
    }
    require_evaluation_binding(result)
    return result


def _suite_core(binding):
    return {key: binding[key] for key in (
        "schema_version", "evaluation_suite", "evaluator_source",
        "config_sha256", "replay_policy_sha256", "runtime")}


def require_evaluation_binding(binding, *, expected_core=None):
    """Reject incomplete, legacy, or mixed composite-v2 evidence bindings."""
    required = {
        "schema_version", "evaluation_suite", "evaluator_source",
        "training_source_commit", "checkpoint_artifact", "config_sha256",
        "replay_policy_sha256", "runtime", "plans",
    }
    if not isinstance(binding, dict) or set(binding) != required:
        raise ValueError("complete exact composite-v2 evaluation binding required")
    if (binding["schema_version"] != 2
            or binding["evaluation_suite"] != backend_descriptor()):
        raise ValueError("legacy or mixed evaluation suite identity")
    source = binding["evaluator_source"]
    expected_source_hashes = evaluation_source_hashes()
    if (not isinstance(source, dict) or set(source) != {"commit", "sha256_normalized_lf"}
            or not _HASH40.fullmatch(str(source["commit"]))
            or source["sha256_normalized_lf"] != expected_source_hashes):
        raise ValueError("exact evaluator source identity required")
    checkpoint = binding["checkpoint_artifact"]
    if (not _HASH40.fullmatch(str(binding["training_source_commit"]))
            or not isinstance(checkpoint, dict)
            or checkpoint.get("kind") not in (
                "checkpoint_file", "serialized_state_fixture_or_probe")
            or set(checkpoint) != ({"kind", "byte_sha256", "model_state_sha256"}
                if checkpoint.get("kind") == "checkpoint_file" else
                {"kind", "serialized_state_sha256", "model_state_sha256"})
            or any(not _HASH64.fullmatch(str(value))
                   for key, value in checkpoint.items() if key != "kind")
            or not _HASH64.fullmatch(str(binding["config_sha256"]))
            or not _HASH64.fullmatch(str(binding["replay_policy_sha256"]))):
        raise ValueError("separate training/checkpoint/config provenance required")
    runtime = binding["runtime"]
    if (not isinstance(runtime, dict)
            or runtime.get("device_type") not in ("cpu", "cuda")
            or runtime.get("deterministic_algorithms")
            != {"enabled": False, "warn_only": False}
            or runtime.get("cublas_workspace_config") not in (None, '')
            or (runtime.get("device_type") == "cuda"
                and runtime.get("qualified_recorded_runtime") is not True)):
        raise ValueError("unqualified or incomplete recorded runtime binding")
    plans = binding["plans"]
    if (not isinstance(plans, list) or len(plans) != 2
            or [row.get("layer") for row in plans] != [0, 1]
            or any(set(row) != {"phase", "fanout", "step", "repeat", "layer",
                               "graph_hash", "indices_sha256",
                               "weights_fp64_sha256",
                               "inclusion_probabilities_sha256",
                               "nonself_populations_sha256",
                               "nonself_selected_sha256"} for row in plans)
            or any(not _HASH64.fullmatch(str(row[key]))
                   for row in plans for key in (
                       "indices_sha256", "weights_fp64_sha256",
                       "inclusion_probabilities_sha256",
                       "nonself_populations_sha256",
                       "nonself_selected_sha256"))
            or any(row["phase"] != plans[0]["phase"]
                   or row["fanout"] != plans[0]["fanout"]
                   or row["step"] != plans[0]["step"]
                   or row["repeat"] != plans[0]["repeat"] for row in plans)):
        raise ValueError("exact two-layer plan identity required")
    if expected_core is not None and _suite_core(binding) != expected_core:
        raise ValueError("mixed composite-v2 evaluator identity")
    return _suite_core(binding)


def require_compatible_evaluation_bindings(bindings):
    if not bindings:
        raise ValueError("at least one composite-v2 evaluation binding required")
    core = None
    for binding in bindings:
        core = require_evaluation_binding(binding, expected_core=core)
    return core


def _clone_tree(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return copy.deepcopy(value)


def _same_tree(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_same_tree(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_same_tree(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def _storage_identity(value):
    if value.device.type == "meta" or value.numel() == 0:
        return None
    storage = value.untyped_storage()
    return (value.device.type, value.device.index, storage.data_ptr())


def _reject_shared_snapshot(live_model, snapshot):
    if snapshot is live_model:
        raise RuntimeError("snapshot factory returned the live training model")
    live_storage = {
        identity
        for value in list(live_model.parameters()) + list(live_model.buffers())
        if (identity := _storage_identity(value)) is not None
    }
    shared = [
        name
        for name, value in list(snapshot.named_parameters()) + list(snapshot.named_buffers())
        if _storage_identity(value) in live_storage
        and _storage_identity(value) is not None
    ]
    if shared:
        raise RuntimeError(
            "snapshot shares parameter or buffer storage with live model: "
            + ", ".join(shared)
        )


def evaluate_frozen_snapshot(
    live_model,
    model_factory,
    evaluator,
    *,
    optimizer=None,
    generators=(),
):
    """Evaluate a fresh frozen copy and preserve live training and RNG state."""
    if (not callable(model_factory) or not callable(evaluator)
            or any(not isinstance(generator, torch.Generator)
                   for generator in generators)):
        raise TypeError("snapshot factory, evaluator and torch generators required")
    with _EVALUATION_LOCK:
        live_state = _clone_tree(live_model.state_dict())
        live_module_training = [module.training for module in live_model.modules()]
        live_requires_grad = [parameter.requires_grad
                              for parameter in live_model.parameters()]
        live_gradients = [
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in live_model.parameters()
        ]
        optimizer_state = (None if optimizer is None
                           else _clone_tree(optimizer.state_dict()))
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        deterministic_before = deterministic_state()
        cuda_states = (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else None
        )
        generator_states = [generator.get_state().clone()
                            for generator in generators]
        result = None
        failure = None
        snapshot = None
        try:
            snapshot = model_factory()
            _reject_shared_snapshot(live_model, snapshot)
            snapshot.load_state_dict(live_state, strict=True)
            snapshot.requires_grad_(False).eval()
            for parameter in snapshot.parameters():
                parameter.grad = None
            if any(module.training for module in snapshot.modules()):
                raise RuntimeError(
                    "evaluation snapshot contains a training-mode submodule")
            with torch.inference_mode():
                result = evaluator(snapshot)
        except BaseException as error:
            failure = error
        finally:
            del snapshot
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
            for generator, state in zip(generators, generator_states,
                                        strict=True):
                generator.set_state(state)
            deterministic_changed = deterministic_state() != deterministic_before
            if deterministic_changed:
                torch.use_deterministic_algorithms(
                    deterministic_before[0], warn_only=deterministic_before[1]
                )

        state_changed = (
            [module.training for module in live_model.modules()]
            != live_module_training
            or [parameter.requires_grad for parameter in live_model.parameters()]
            != live_requires_grad
            or not _same_tree(live_model.state_dict(), live_state)
            or any(
                (before is None) != (parameter.grad is None)
                or (before is not None and not torch.equal(before, parameter.grad))
                for before, parameter in zip(
                    live_gradients, live_model.parameters(), strict=True)
            )
            or (optimizer is not None
                and not _same_tree(optimizer.state_dict(), optimizer_state))
            or deterministic_changed
        )
        if state_changed:
            # Restore first so even a faulty factory/callback cannot leak mutation.
            live_model.load_state_dict(live_state, strict=True)
            for module, mode in zip(live_model.modules(), live_module_training,
                                    strict=True):
                module.training = mode
            for parameter, requires_grad, gradient in zip(
                    live_model.parameters(), live_requires_grad, live_gradients,
                    strict=True):
                parameter.requires_grad_(requires_grad)
                parameter.grad = (None if gradient is None else gradient.clone())
            if optimizer is not None:
                optimizer.load_state_dict(optimizer_state)
            raise RuntimeError(
                "frozen evaluation snapshot changed live training state")
        if failure is not None:
            raise failure
        return result

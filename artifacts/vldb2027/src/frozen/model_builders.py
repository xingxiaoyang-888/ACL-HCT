"""SOURCE-ONLY four-model construction proposal; not a task-ready trainer.

No task loader, checkpoint IO, label split lookup, training loop or CLI lives
here. Native source roots are supplied explicitly at runtime and must be bound
by a separate release. Each process may import at most one hyperbolic tree.
Model selection is by model_key, never by the graph's dataset string.

Read-only references: code-v4/sage.py and train_sage.py; native HGCN/HyboNet
encoders, decoders, losses and optimizer; official DGL ogbn-arxiv models.py
and gat.py (links in model-config-proposal-v1.json). GAT's BN/residual scaffold
and logCE follow that reference, using the already prepared PyG block operator.
This is an explicit sampling adaptation, not reproduction of official scores.
All builders start on CPU under a private initialization RNG, then move once.
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from core import SharedCorrection
from euclidean_adapters import CorrectedGATConv, CorrectedSAGE
from hyperbolic_adapters import FixedSelfMeanContract, NativeHyperbolicBlockModel
from shared_blocks import NativeBlock, validate_block_chain


MODEL_KEYS = ("hgcn", "hybonet", "sage", "gat")
VARIANTS = ("plain", "candidate", "generic")
SEEDS = (11, 23, 37)
GAT_LOGCE_EPSILON = 1.0 - math.log(2.0)  # Official arxiv/gat.py constant.
_NATIVE_CACHE: dict[str, SimpleNamespace] = {}


def _require(condition, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _settings(model_key: str, config: Mapping) -> Mapping:
    _require(model_key in MODEL_KEYS, "one of the four registered model keys required")
    _require(config["task"]["dataset"] == "ogbn-arxiv", "only the registered common task")
    _require(config["task"]["input_features"] == 128 and config["task"]["classes"] == 40,
             "arxiv dimensions must be explicit metadata, not inferred from labels")
    settings = config["models"][model_key]
    _require(settings["actual_message_layers"] == 3, "exactly three actual message layers")
    return settings


def _load_native(model_key: str, config: Mapping) -> SimpleNamespace:
    """Import only a released, explicitly supplied tree; no dataset modules."""
    _require(model_key in ("hgcn", "hybonet"), "native import is hyperbolic only")
    supplied = config.get("runtime_native_roots", {}).get(model_key)
    _require(isinstance(supplied, str) and bool(supplied),
             "runtime_native_roots must bind the exact native source tree")
    root = Path(supplied).resolve()
    _require(root.is_dir(), "native source root is not an existing directory")
    if _NATIVE_CACHE:
        _require(model_key in _NATIVE_CACHE, "fresh process required for another native tree")
        api = _NATIVE_CACHE[model_key]
        _require(api.root == root, "native source root cannot change in one process")
        return api
    packages = ("models", "layers", "manifolds", "utils", "optim", "optimizers")
    _require(not any(name in sys.modules for name in packages),
             "native top-level package collision: use a fresh process")
    original_path = sys.path[:]
    sys.dont_write_bytecode = True  # The bound upstream trees stay read-only.
    sys.path.insert(0, str(root))
    try:
        encoders = importlib.import_module("models.encoders")
        decoders = importlib.import_module("models.decoders")
        optimizer = importlib.import_module("optim") if model_key == "hybonet" else None
        for module in (encoders, decoders, optimizer):
            if module is not None:
                _require(root in Path(module.__file__).resolve().parents,
                         "imported native module escaped its bound source root")
    finally:
        sys.path[:] = original_path
    api = SimpleNamespace(root=root, encoders=encoders, decoders=decoders,
                          optimizer=optimizer)
    _NATIVE_CACHE[model_key] = api
    return api


class ArxivGAT(nn.Module):
    """Three-layer BN/residual GAT, one explicit sampling adaptation.

    Three hidden heads of 250 channels, final one head of 40. Input dropout
    once; two hidden BN -> ReLU -> dropout stages. Native self-loop attention
    and learned residuals remain in CorrectedGATConv. Hidden bias is absent;
    final bias is present, matching the reference's post-head class bias.
    BN uses each block's actual destination rows during training, unlike the
    reference's full-graph BN. This difference requires baseline qualification.
    """

    def __init__(self, settings: Mapping, *, variant: str, correction_seed: int):
        super().__init__()
        head_dim, heads = settings["hidden_head_dim"], settings["hidden_heads"]
        _require((head_dim, heads, settings["output_heads"]) == (250, 3, 1),
                 "only the single registered GAT width/head configuration")
        _require(settings["residual"] is True, "registered GAT residual is required")
        width = head_dim * heads
        shapes = ((128, head_dim, heads, True),
                  (width, head_dim, heads, True), (width, 40, 1, False))
        self.convs = nn.ModuleList([
            CorrectedGATConv(inputs, outputs, heads=count, concat=concat,
                             negative_slope=.2, attention_dropout=settings["attention_dropout"],
                             bias=index == 2, residual=True, variant=variant,
                             correction_seed=correction_seed + index)
            for index, (inputs, outputs, count, concat) in enumerate(shapes)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(width), nn.BatchNorm1d(width)])
        self.input_dropout = settings["input_dropout"]
        self.hidden_dropout = settings["hidden_dropout"]

    def forward(self, x: Tensor, blocks: Sequence[NativeBlock], *,
                correction_enabled: bool = True) -> Tensor:
        _require(len(blocks) == 3, "GAT requires three dependency blocks")
        validate_block_chain(blocks)
        x = F.dropout(x, self.input_dropout, self.training)
        for index, (layer, block) in enumerate(zip(self.convs, blocks)):
            x = layer(x, block, correction_enabled=correction_enabled)
            if index < 2:
                x = F.dropout(F.relu(self.norms[index](x)),
                              self.hidden_dropout, self.training)
        return x  # Native logits, not a probability or true-label correction.


def build_model(model_key: str, variant: str, seed: int,
                config: Mapping, device) -> nn.Module:
    """Construct without IO or loading any state; caller owns runtime release.

    For adaptation, immediately call load_backbone_state on the SAME accepted
    plain-state mapping for all three arms, then make a fresh optimizer. The
    private constructor RNG does not seed training/dropout or sampling streams.
    """
    settings = _settings(model_key, config)
    _require(variant == 'plain' and seed in SEEDS, "VLDB frozen model entry is plain-only")
    _require(torch.get_default_dtype() == torch.float32,
             "native constructors require the registered FP32 default dtype")
    correction_seed = 62000000 + 100 * seed + 10 * MODEL_KEYS.index(model_key)
    api = _load_native(model_key, config) if model_key in ("hgcn", "hybonet") else None
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        if model_key == "sage":
            _require((settings["hidden"], settings["dropout"]) == (256, .5),
                     "historical SAGE checkpoint architecture is immutable")
            model = CorrectedSAGE(128, 256, 40, .5, variant=variant,
                                  correction_seed=correction_seed)
        elif model_key == "gat":
            model = ArxivGAT(settings, variant=variant, correction_seed=correction_seed)
        else:
            _require(settings["curvature"] == 1.0 and settings["ambient_hidden"] == 128,
                     "fixed native curvature and single registered ambient width")
            args = SimpleNamespace(task="nc", num_layers=4, feat_dim=129, dim=128,
                                   act=settings["activation"], c=1., cuda=-1, device="cpu",
                                   manifold="Hyperboloid" if model_key == "hgcn" else "Lorentz",
                                   bias=True, dropout=settings["dropout"], use_att=False,
                                   local_agg=False, n_classes=40)
            curvature = torch.tensor([1.], dtype=torch.float32)
            encoder_class = api.encoders.HGCN if model_key == "hgcn" else api.encoders.HyboNet
            decoder_class = api.decoders.LinearDecoder if model_key == "hgcn" else api.decoders.LorentzDecoder
            encoder, decoder = encoder_class(curvature, args), decoder_class(curvature, args)
            corrections = [None if variant == "plain" else
                           SharedCorrection(127, variant, init_seed=correction_seed + index)
                           for index in range(3)]
            contract = FixedSelfMeanContract("row_normalized_A_plus_I", True, True)
            model = NativeHyperbolicBlockModel(
                encoder, decoder, model_kind=model_key, actual_message_layers=3,
                contract=contract, corrections=corrections)
            if model_key == "hybonet":
                model._native_optimizer_class = api.optimizer.RiemannianAdam
    model._study_model_key = model_key
    model._study_variant = variant
    model._study_seed = seed
    model._study_input_features = 128
    model._study_classes = 40
    return model.to(device=device, dtype=torch.float32)


def build_pubmed_sage(seed: int, device):
    """Only the new fixed 500/256/3 plain SAGE configuration; no training."""
    _require(seed in SEEDS, "registered PubMed seed required")
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        model = CorrectedSAGE(500, 256, 3, .5, variant="plain")
    model._study_model_key, model._study_variant = "sage", "plain"
    model._study_seed = seed
    model._study_input_features, model._study_classes = 500, 3
    return model.to(device=device, dtype=torch.float32)


def native_scores(model: nn.Module, x: Tensor, blocks: Sequence[NativeBlock], *,
                  correction_enabled: bool = True) -> Tensor:
    """SAGE returns log probabilities; the other models return native logits."""
    _require(getattr(model, "_study_model_key", None) in MODEL_KEYS, "builder model required")
    output = model(x, blocks, correction_enabled=correction_enabled)
    return output.scores if model._study_model_key in ("hgcn", "hybonet") else output


def task_loss(model_key: str, scores: Tensor, labels: Tensor, config: Mapping) -> Tensor:
    """ONLY the registered native task objective, no correction regularizer.

    The caller must supply already-authorized train rows. No function here
    chooses or opens labels. SAGE uses historical log-softmax/NLL; HGCN's
    native log-softmax+NLL equals CE on its raw decoder scores. Hybo's mean
    includes the correct-class constant term, exactly as native NCModel.
    """
    settings = _settings(model_key, config)
    _require(scores.ndim == 2 and scores.shape[1] == 40 and scores.shape[0] > 0,
             "expected nonempty B x 40 native scores")
    _require(labels.shape == (scores.shape[0],) and labels.dtype == torch.long
             and labels.device == scores.device, "label shape/dtype/device mismatch")
    _require(((labels >= 0) & (labels < 40)).all(), "label outside registered class range")
    _require(torch.isfinite(scores).all(), "nonfinite native task scores")
    if model_key == "sage":
        loss = F.nll_loss(scores, labels)
    elif model_key == "hgcn":
        loss = F.cross_entropy(scores, labels)
    elif model_key == "hybonet":
        correct = scores.gather(1, labels[:, None])
        loss = F.relu(settings["margin"] - correct + scores).mean()
    else:
        ce = F.cross_entropy(scores, labels, reduction="none")
        loss = (torch.log(GAT_LOGCE_EPSILON + ce) - math.log(GAT_LOGCE_EPSILON)).mean()
    _require(torch.isfinite(loss), "nonfinite task loss; no automatic retry")
    return loss


def make_optimizer(model_key: str, model: nn.Module, config: Mapping, stage: str):
    """Always return a NEW optimizer. No old moments/scheduler state are loaded.

    Target LR is returned; trainer applies the declared epoch-based warmup.
    Pretrain is plain-only and unavailable for SAGE, whose accepted CP is reused.
    All adaptation arms receive identical backbone LR and parameter opportunity.
    """
    settings = _settings(model_key, config)
    _require(getattr(model, "_study_model_key", None) == model_key, "optimizer/model mismatch")
    _require(stage in ("pretrain", "adapt"), "unknown stage")
    stage_settings = settings["stages"][stage]
    _require(stage_settings["enabled"], "stage is disabled in this model")
    if stage == "pretrain":
        _require(model._study_variant == "plain", "only a plain warm start may be pretrained")
    learning_rate, decay = stage_settings["learning_rate"], settings["weight_decay"]
    trainable = [p for p in model.parameters() if p.requires_grad]
    _require(len(trainable) > 0, "no trainable parameters")
    if model_key == "hybonet":
        from geoopt import ManifoldParameter
        no_decay, regular = [], []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                (no_decay if isinstance(parameter, ManifoldParameter)
                 or "bias" in name or "scale" in name else regular).append(parameter)
        groups = [{"params": regular, "weight_decay": decay},
                  {"params": no_decay, "weight_decay": 0.}]
        return model._native_optimizer_class(groups, lr=learning_rate,
                                              stabilize=settings["stabilize_every"])
    if model_key == "gat":
        return torch.optim.RMSprop(trainable, lr=learning_rate, weight_decay=decay,
                                    alpha=.99, eps=1e-8, momentum=0., centered=False)
    return torch.optim.Adam(trainable, lr=learning_rate, weight_decay=decay,
                            betas=(.9, .999), eps=1e-8, amsgrad=False)


def load_backbone_state(model: nn.Module, state: Mapping[str, Tensor]) -> None:
    """Exact-key in-memory warm start, never checkpoint IO or permissive remapping.

    Strip ONLY the three known per-layer correction namespaces. Everything
    else, including BN running moments and task heads, must match. This does
    not authorize a source checkpoint: its identity and common use by all arms
    belong in the release. Historical SAGE model_state keys need no renaming.
    """
    key = getattr(model, "_study_model_key", None)
    _require(key in MODEL_KEYS, "builder model required")
    stem = "layers" if key in ("hgcn", "hybonet") else "convs"
    prefixes = tuple(f"{stem}.{index}.correction." for index in range(3))
    is_correction = lambda name: name.startswith(prefixes)
    expected = {name: tensor for name, tensor in model.state_dict().items()
                if not is_correction(name)}
    supplied = {name: tensor for name, tensor in state.items() if not is_correction(name)}
    _require(set(expected) == set(supplied),
             "warm start backbone keys must exactly match; only correction keys may be removed")
    for name, target in expected.items():
        source = supplied[name]
        _require(isinstance(source, Tensor) and source.shape == target.shape
                 and source.dtype == target.dtype, f"warm start shape/dtype differs at {name}")
        _require(torch.isfinite(source).all(), f"nonfinite warm start at {name}")
    result = model.load_state_dict(supplied, strict=False)
    _require(not result.unexpected_keys and all(is_correction(name) for name in result.missing_keys),
             "unexpected state mapping result")


__all__ = ["build_model", "native_scores", "task_loss", "make_optimizer",
           "load_backbone_state", "ArxivGAT", "GAT_LOGCE_EPSILON"]

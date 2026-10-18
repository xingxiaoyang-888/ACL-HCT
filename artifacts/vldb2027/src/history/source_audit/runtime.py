# SOURCE AUDIT EXPORT: private storage names replaced; science functions retained.
raise RuntimeError("Historical source-audit export; not a public executable entry. Use the documented scalar aggregation route.")
"""Single-worker native NC runtime, with isolated upstream absolute imports."""
import os
import sys
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.dont_write_bytecode = True
from contextlib import contextmanager
import importlib
import importlib.util
from types import SimpleNamespace, MethodType
import random
from contract import CODE, PROJECT, read_json, sha, cell

# Reuse the accepted local dependency installation when geoopt is not installed.
if importlib.util.find_spec("geoopt") is None:
    dep = PROJECT / "public-inputs/history/legacy-dependencies/code/.deps"
    if dep.is_dir():
        sys.path.insert(0, str(dep))
import numpy as np
import torch

PACKAGES = ("layers", "manifolds", "models", "utils", "optim", "optimizers", "config")
CACHE = {}


def configure():
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(False, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def belongs(name):
    return any(name == p or name.startswith(p + ".") for p in PACKAGES)


@contextmanager
def namespace(root, modules=None):
    saved = {k: v for k, v in sys.modules.items() if belongs(k)}
    for name in saved:
        del sys.modules[name]
    if modules:
        sys.modules.update(modules)
    old_path = sys.path[:]
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        for name in list(sys.modules):
            if belongs(name):
                del sys.modules[name]
        sys.modules.update(saved)
        sys.path[:] = old_path


def native_api(model):
    if model in CACHE:
        return CACHE[model]
    if model == "HyboNet":
        root = PROJECT / "public-inputs/history/hybonet-origin/code/hybonet"
        pinned = PROJECT / "public-inputs/history/hybonet-origin/code/source-manifest.json"
        if sha(pinned) != "1e60edf8ced0941d1354e675704814398dba980bb7cb3a276c7ec5fd70e649fc":
            raise ValueError("Inherited native source manifest differs")
        records = {k[8:]: {"sha256": v} for k, v in read_json(pinned)["python_sha256"].items() if k.startswith("hybonet/")}
    elif model == "HGCN":
        root = PROJECT / "public-inputs/native/hgcn"
        pinned = PROJECT / "configs/hgcn_upstream.json"
        if sha(pinned) != "7b8e2a18d40d91147f31cc09d03739b18c9335f727aad45c0e3e8d9c0954e6e8":
            raise ValueError("HGCN native manifest differs")
        records = read_json(pinned)["files"]
    else:
        raise ValueError("Unregistered model")
    with namespace(root):
        base = importlib.import_module("models.base_models")
        data = importlib.import_module("utils.data_utils")
        optimizer = importlib.import_module("optim").RiemannianAdam if model == "HyboNet" else torch.optim.Adam
        modules = {k: v for k, v in sys.modules.items() if belongs(k)}
        for name, module in modules.items():
            path = getattr(module, "__file__", None)
            if not path:
                continue
            path = __import__("pathlib").Path(path).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Native namespace shadowed: " + name)
            relative = path.relative_to(root).as_posix()
            rec = records.get(relative)
            if not rec or sha(path, rec.get("normalization") == "LF") != rec["sha256"]:
                raise ValueError("Native imported source differs: " + relative)
        api = SimpleNamespace(NCModel=base.NCModel, optimizer=optimizer, data=data, modules=modules, root=root)
    CACHE[model] = api
    return api


def effective_args(model, dataset, seed, device, features, n_classes):
    name = {("HyboNet", "Airport"): "HyboNet_Airport_inherited",
            ("HyboNet", "Disease-NC"): "HyboNet_Disease",
            ("HGCN", "Airport"): "HGCN_Airport_fixed_Disease_recipe_transfer",
            ("HGCN", "Disease-NC"): "HGCN_Disease"}[(model, dataset)]
    values = dict(read_json(CODE / "effective-config-addendum.json")["effective_model_args"][name])
    values.update(seed=seed, device=str(device), cuda=-1 if str(device) == "cpu" else int(str(device).split(":")[1]),
                  n_nodes=features.shape[0], feat_dim=features.shape[1], n_classes=n_classes,
                  pretrained_embeddings=None, pos_weight=0)
    return SimpleNamespace(**values)


def build_model(model, dataset, seed, device, features, n_classes):
    api = native_api(model)
    args = effective_args(model, dataset, seed, device, features, n_classes)
    with namespace(api.root, api.modules):
        net = api.NCModel(args).to(device)
    if len(net.encoder.layers) != cell(model, dataset)["native_message_layers"]:
        raise ValueError("Wrong native message layer count")
    return net, args


def make_optimizer(net, model, args):
    api = native_api(model)
    if model == "HGCN":
        return torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    from geoopt import ManifoldParameter
    groups = [{"params": [p for n, p in net.named_parameters() if p.requires_grad and
                            not any(s in n for s in ("bias", "scale")) and not isinstance(p, ManifoldParameter)],
               "weight_decay": args.weight_decay},
              {"params": [p for n, p in net.named_parameters() if
                            (p.requires_grad and any(s in n for s in ("bias", "scale"))) or isinstance(p, ManifoldParameter)],
               "weight_decay": 0.0}]
    return api.optimizer(params=groups, lr=args.lr, stabilize=10)


@contextmanager
def layer_plans(net, plans):
    if len(plans) != len(net.encoder.layers):
        raise ValueError("One sparse plan per native layer required")
    saved = []
    counts = [0] * len(plans)
    try:
        for layer, plan in zip(net.encoder.layers, plans):
            agg = layer.agg
            index = len(saved)
            original = agg.forward
            previous = agg.__dict__.get("forward")
            saved.append((agg, previous))
            def forward(self, x, ignored, original=original, plan=plan, index=index):
                counts[index] += 1
                return original(x, plan)
            agg.forward = MethodType(forward, agg)
        yield counts
        if counts != [1] * len(plans):
            raise ValueError("Native aggregation was skipped or repeated")
    finally:
        for agg, previous in saved:
            if previous is None:
                del agg.forward
            else:
                agg.forward = previous


def encode_scores(net, features, full, plans):
    with layer_plans(net, plans):
        h = net.encode(features, full)
    return net.decoder.decode(h, full)


def finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Nonfinite " + name)


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def state_hash(state):
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode() + b"\0" + str(tensor.dtype).encode() + b"\0")
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sync(device):
    if str(device) != "cpu":
        torch.cuda.synchronize(device)

"""Stage-separated input readers and synchronized request-to-output timing."""
from pathlib import Path
import time
import gc
import numpy as np
from common import torch, sha, read_json, load_npz, sync
from blocks import Graph, plan
from sage import SAGE


def load_public(public):
    public = Path(public)
    ids, receipt = read_json(public/'ids.json'), read_json(public/'preparation.json')
    for name in ('ids.json', 'graph.npz'):
        if receipt['public'][name] != sha(public/name):
            raise ValueError('Public prepared input changed')
    d = load_npz(public/'graph.npz', ('x', 'indptr', 'indices', 'values', 'diagonal'))
    kind = 'sage' if ids['dataset'] == 'ogbn-arxiv' else 'hybo'
    g = Graph(torch.from_numpy(d['x']), d['indptr'], d['indices'], d['values'], d['diagonal'], kind)
    return g, ids


def labels(path, expected_ids):
    d = load_npz(path, ('ids', 'y'))
    if d['ids'].tolist() != expected_ids or len(d['y']) != len(expected_ids) or d['y'].dtype != np.int64:
        raise ValueError('Labels/authorized ID list differ')
    return {int(n): int(y) for n, y in zip(d['ids'], d['y'])}


def load_sage(path, device):
    w = torch.load(path, map_location='cpu', weights_only=True)
    if set(w) != {'schema', 'seed', 'epoch', 'model_state', 'bindings'} or w['schema'] != 'icml_ca_sage_weights_v1':
        raise ValueError('Only label-free SAGE weight exports accepted')
    with torch.random.fork_rng(devices=[]):
        model = SAGE()
    model.load_state_dict(w['model_state'], strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def fanouts(kind, family, k):
    layers, b = (3, 15) if kind == 'sage' else (5, 8)
    if family == 'S':
        return [b]*layers
    if family in ('U', 'B', 'full', 'E'):
        return [-1 if family == 'full' else k]*layers
    if family == 'P':
        return [k]*((layers+1)//2) + [b]*(layers//2)
    if family == 'L':
        return [b]*(layers-1)+[k]
    raise ValueError('Unknown frozen operation')


def path_forward(model, graph, ids, config, stage, seed, repeat, batch, device, path='main', observer=None):
    blocks = plan(graph, ids, config['fanouts'], stage, seed, repeat, batch, path,
                  'labor' if config['family'] == 'B' else 'uniform')
    x = graph.x[torch.from_numpy(blocks[0].src)].to(device)
    output = model(x, blocks, observer)
    return output, blocks


def operation(model, graph, ids, config, stage, seed, repeat, batch, device, observer=None, paid_s=None):
    if config['family'] == 'E':
        if paid_s is None:
            s, sb = path_forward(model, graph, ids, {'family':'S', 'fanouts':fanouts(graph.kind, 'S', 0)},
                                stage, seed, repeat, batch, device, observer=observer)
        else:
            s, sb = paid_s
        u, ub = path_forward(model, graph, ids, {'family':'U', 'fanouts':config['fanouts']},
                            stage, seed, repeat, batch, device, path='E-independent')
        return (s+u)*.5, sb+ub
    return path_forward(model, graph, ids, config, stage, seed, repeat, batch, device, observer=observer)


def timed_request(function, device):
    # IDs already received by the callable. Planning, indexing, transfers,
    # forwards and native output construction all occur INSIDE this boundary.
    sync(device)
    if torch.device(device).type == 'cuda':
        # Caller has already dropped prior request outputs/temporary tensors.
        # Cleanup is outside timing, with the same original graph/model resident.
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        resident = {'allocated':torch.cuda.memory_allocated(device), 'reserved':torch.cuda.memory_reserved(device)}
    else:
        resident = {'allocated':0, 'reserved':0}
    start = time.perf_counter()
    def to_cpu(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value,tuple):
            return tuple(to_cpu(v) for v in value)
        if isinstance(value,list):
            return [to_cpu(v) for v in value]
        if isinstance(value,dict):
            return {k:to_cpu(v) for k,v in value.items()}
        return value
    result = to_cpu(function())
    sync(device)
    elapsed = time.perf_counter()-start
    memory = {'allocated':torch.cuda.max_memory_allocated(device), 'reserved':torch.cuda.max_memory_reserved(device)} if torch.device(device).type == 'cuda' else resident
    return result, {'e2e_seconds':elapsed, 'peak':memory, 'resident':resident}

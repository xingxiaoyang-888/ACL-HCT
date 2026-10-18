"""Real dependency cones, stateless WOR uniform, and official DGL LABOR-0."""
from dataclasses import dataclass
from contextlib import contextmanager
from threading import local
import numpy as np
from common import torch, domain


_planning_threads = local()


@contextmanager
def native_planning_threads():
    """One whole native plan/format scope; restore possibly coupled APIs.

    The owned runner is single-worker. Nested graph/sampler calls do not
    switch threads again. Both API settings are observed, since on Linux
    DGL and Torch can share OpenMP; their independence is not assumed.
    """
    depth = getattr(_planning_threads, 'depth', 0)
    if depth:
        _planning_threads.depth = depth + 1
        try:
            yield
        finally:
            _planning_threads.depth = depth
        return
    import dgl
    prior_torch, prior_dgl = torch.get_num_threads(), dgl.utils.get_num_threads()
    _planning_threads.depth = 1
    try:
        torch.set_num_threads(1)
        dgl.utils.set_num_threads(1)
        if torch.get_num_threads() != 1 or dgl.utils.get_num_threads() != 1:
            raise RuntimeError('Native planning thread policy was not applied')
        yield
    finally:
        try:
            try:
                dgl.utils.set_num_threads(prior_dgl)
            finally:
                torch.set_num_threads(prior_torch)
        finally:
            _planning_threads.depth = 0
        if torch.get_num_threads() != prior_torch or dgl.utils.get_num_threads() != prior_dgl:
            raise RuntimeError('Native planning did not restore host thread settings')


@dataclass
class Graph:
    x: torch.Tensor
    indptr: np.ndarray
    indices: np.ndarray
    values: np.ndarray  # native Hybo nonself coefficients; SAGE ignored
    diagonal: np.ndarray
    kind: str

    def __post_init__(self):
        self.n = len(self.x)
        self.degree = np.diff(self.indptr)
        if len(self.indptr) != self.n+1 or self.indptr[0] != 0 or self.indptr[-1] != len(self.indices):
            raise ValueError('CSR boundary mismatch')
        if len(self.values) != len(self.indices) or len(self.diagonal) != self.n or self.kind not in ('sage', 'hybo'):
            raise ValueError('Graph coefficient/schema mismatch')
        self._dgl = None

    @native_planning_threads()
    def dgl_graph(self):
        # Original CSR/DGL are common resident graph formats, not sampled plans.
        import dgl
        # Same deterministic native CPU planning policy for uniform and LABOR.
        # Format preparation is also scoped. Nested calls within a complete
        # plan retain its single thread without per-layer thread switches.
        if self._dgl is None:
            rows = np.repeat(np.arange(self.n, dtype=np.int64), self.degree)
            self._dgl = dgl.graph((torch.from_numpy(self.indices), torch.from_numpy(rows)), num_nodes=self.n)
            self._dgl.create_formats_()
        return self._dgl


@dataclass
class Block:
    src: np.ndarray
    dst: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    coefficients: np.ndarray
    degree: np.ndarray
    counts: np.ndarray
    backend: str

    def tensors(self, device):
        return (torch.as_tensor(self.edge_src, device=device), torch.as_tensor(self.edge_dst, device=device),
                torch.as_tensor(self.coefficients, device=device), torch.as_tensor(np.searchsorted(self.src, self.dst), device=device))

    def report(self):
        return {'unique_source': len(self.src), 'unique_destination': len(self.dst), 'deduplicated_edges': len(self.edge_src),
                'calls': 1, 'hidden_state_reuse': 0, 'sampler': self.backend}


def priorities(eids, rows, salt):
    # SplitMix64 counter stream, no model/global RNG draws. Edge+row identity
    # ensures changing dependency-cone row order cannot change a row's plan.
    with np.errstate(over='ignore'):
        z = np.asarray(eids, dtype=np.uint64) ^ (np.asarray(rows, dtype=np.uint64)*np.uint64(0xD2B74407B1CE6E93)) ^ np.uint64(salt)
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def uniform_edges_reference(g, dst, k, salt):
    deg = g.degree[dst]
    row = np.repeat(np.arange(len(dst), dtype=np.int64), deg)
    offsets = np.repeat(np.cumsum(deg)-deg, deg)
    eids = np.repeat(g.indptr[dst], deg) + np.arange(len(row))-offsets
    if k != -1 and len(row):
        if k < 1:
            raise ValueError('Positive fanout or full -1 required')
        # Sorted independent priorities yield a legal WOR subset and budget prefix.
        order = np.lexsort((eids, priorities(eids, dst[row], salt), row))
        ranked_rows = row[order]
        rank = np.arange(len(row))-np.repeat(np.cumsum(deg)-deg, deg)
        choose = order[rank < k]
        row, eids = row[choose], eids[choose]
    counts = np.bincount(row, minlength=len(dst))
    if g.kind == 'sage':
        coef = np.divide(1., counts[row], dtype=np.float64).astype(np.float32)
    else:
        coef = g.values[eids] * np.divide(deg[row], counts[row], dtype=np.float64).astype(np.float32)
    return g.indices[eids], row, coef, counts


@native_planning_threads()
def uniform_edges(g, dst, k, salt):
    import dgl
    dst = np.asarray(dst, dtype=np.int64)
    # DGL's process-local sampler RNG is owned by this single-worker runner,
    # explicitly reseeded per domain/layer. It never consumes torch/dropout or
    # numpy RNG. sample_neighbors has no local Generator argument in 1.1.x.
    native = g.dgl_graph()  # scope applied BEFORE resetting native RNG
    dgl.seed(int(salt % (2**31-1)))
    frontier = dgl.sampling.sample_neighbors(native, torch.as_tensor(dst), k,
        edge_dir='in', replace=False, copy_ndata=False, copy_edata=True)
    source, destinations = (t.numpy() for t in frontier.edges(order='eid'))
    eid = frontier.edata[dgl.EID].numpy()
    positions = np.full(g.n, -1, dtype=np.int64)
    positions[dst] = np.arange(len(dst), dtype=np.int64)
    row = positions[destinations]
    counts = np.bincount(row, minlength=len(dst))
    if g.kind == 'sage':
        coef = 1 / counts[row]
    else:
        coef = g.values[eid] * g.degree[dst][row] / counts[row]
    return source, row, coef.astype(np.float64), counts


@native_planning_threads()
def labor_edges(g, dst, k, salt):
    import dgl
    dst = np.asarray(dst, dtype=np.int64)
    native = g.dgl_graph()
    # dgl NDArray is accepted across the pinned 1.1.x native API. No default RNG.
    native_seed = dgl.ndarray.array(np.array([salt], dtype=np.int64))
    frontier, returned = dgl.sampling.sample_labors(native, torch.as_tensor(dst), k, edge_dir='in', prob=None,
        importance_sampling=0, random_seed=native_seed, copy_ndata=False, copy_edata=True)
    s, d = (t.numpy() for t in frontier.edges(order='eid'))
    eid = frontier.edata[dgl.EID].numpy()
    positions = np.full(g.n, -1, dtype=np.int64)
    positions[dst] = np.arange(len(dst), dtype=np.int64)
    row = positions[d]
    counts = np.bincount(row, minlength=len(dst))
    # Uniform LABOR-0 returns an EMPTY importance tensor (official unweighted
    # code path); official SAGEConv uses sample mean. For nonempty returned
    # importances official DGL normalizes inverse probabilities so a *mean*
    # reducer with these multipliers is the intended estimator. Do not invert
    # or discard returned weights. Assert alignment and retain it in coefficients.
    weight = returned[0].numpy()
    if weight.size not in (0, len(eid)):
        raise ValueError('Native LABOR importance/EID alignment failed')
    multiplier = weight if weight.size else np.ones(len(eid), dtype=np.float32)
    coef = multiplier / np.maximum(counts[row], 1)
    if g.kind == 'hybo':
        # Native uniform LABOR uses Bernoulli inclusion min(1,k/degree),
        # NOT k fixed edges. For Hybo's weighted *sum*, the HT coefficient
        # is A_ij / p_ij. A sample-count renormalization would change it.
        if weight.size:
            raise ValueError('Unexpected importance weights in pinned uniform LABOR-0')
        inclusion = np.ones(len(eid)) if k == -1 else np.minimum(1., k/g.degree[dst][row])
        coef = g.values[eid] / inclusion
    return s, row, coef.astype(np.float64), counts


def make_block(g, dst, k, salt, sampler='uniform'):
    dst = np.asarray(dst, dtype=np.int64)
    if len(np.unique(dst)) != len(dst) or np.any(dst < 0) or np.any(dst >= g.n):
        raise ValueError('Distinct in-range targets required')
    source, row, coef, counts = (labor_edges if sampler == 'labor' else uniform_edges)(g, dst, k, salt)
    if g.kind == 'hybo':
        source = np.concatenate((source, dst))
        row = np.concatenate((row, np.arange(len(dst))))
        coef = np.concatenate((coef, g.diagonal[dst]))
    src = np.unique(np.concatenate((source, dst)))
    return Block(src, dst.copy(), np.searchsorted(src, source), row, coef.astype(np.float64),
                 g.degree[dst].copy(), counts, sampler)


@native_planning_threads()
def plan(g, targets, fanouts, stage, seed, repeat=0, batch=0, path='main', sampler='uniform'):
    blocks, dst = [], np.asarray(targets, dtype=np.int64)
    for layer in reversed(range(len(fanouts))):
        block = make_block(g, dst, fanouts[layer], domain(stage, seed, repeat, batch, layer+1, path), sampler)
        blocks.insert(0, block)
        dst = block.src
    for a, b in zip(blocks, blocks[1:]):
        if not np.array_equal(a.dst, b.src):
            raise ValueError('Broken layer dependency chain')
    return blocks


def aggregate(x, block):
    s, d, coef, own = block.tensors(x.device)
    # Sparse mm avoids a materialized edge-by-channel tensor on arxiv. Coalesce
    # contains no new sampling/normalization and preserves full native weights.
    a = torch.sparse_coo_tensor(torch.stack((d, s)), coef.to(x.dtype),
                               (len(block.dst), len(block.src)), device=x.device).coalesce()
    return torch.sparse.mm(a, x), own

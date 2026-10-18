# SOURCE AUDIT EXPORT: private storage names replaced; science functions retained.
raise RuntimeError("Historical source-audit export; not a public executable entry. Use the documented scalar aggregation route.")
"""Native coefficients, deterministic self, private nested row-wise WOR."""
from runtime import np, torch
import hashlib


def plan_hash(adj):
    a = adj.coalesce().cpu()
    return hashlib.sha256(a.indices().numpy().astype("<i8").tobytes() +
                          a.values().numpy().astype("<f4").tobytes()).hexdigest()


class Sampler:
    def __init__(self, full):
        self.full = full.detach().coalesce().cpu()
        if self.full.dtype != torch.float32:
            raise ValueError("Native FP32 adjacency required")
        self.row, self.col = self.full.indices().numpy()
        self.value = self.full.values().numpy()
        self.bounds = np.searchsorted(self.row, np.arange(full.shape[0] + 1))
        self.degree = np.empty(full.shape[0], dtype=np.int64)
        for node in range(full.shape[0]):
            cs = self.col[self.bounds[node]:self.bounds[node + 1]]
            if (cs == node).sum() != 1 or (len(cs) > 1 and np.any(np.diff(cs) <= 0)):
                raise ValueError("Expected sorted native row with exactly one self")
            self.degree[node] = len(cs) - 1

    def plan(self, budget, stream, layer, *, legacy_selection=False):
        if budget == "full":
            return self.full
        if not isinstance(budget, int) or budget < 1:
            raise ValueError("Positive registered budget required")
        capacity = int(np.minimum(self.degree, budget).sum() + len(self.degree))
        rows = np.empty(capacity, dtype=np.int64)
        cols = np.empty(capacity, dtype=np.int64)
        vals = np.empty(capacity, dtype=np.float32)
        legacy = np.random.RandomState(np.random.SeedSequence([int(stream), int(layer)]).generate_state(1)[0]) if legacy_selection else None
        offset = 0
        for node, count in enumerate(self.degree):
            start, end = self.bounds[node:node + 2]
            cs, vs = self.col[start:end], self.value[start:end]
            own = cs == node
            rows[offset], cols[offset], vals[offset] = node, node, vs[own][0]
            offset += 1
            keep = min(budget, int(count))
            if keep:
                ns, coefficients = cs[~own], vs[~own]
                if keep == count:
                    chosen = np.arange(count)
                elif legacy_selection:
                    chosen = np.sort(legacy.choice(count, keep, replace=False))
                else:
                    private = np.random.RandomState(np.random.SeedSequence([int(stream), int(layer), int(node)]).generate_state(1)[0])
                    chosen = np.sort(private.permutation(count)[:keep])
                rows[offset:offset + keep] = node
                cols[offset:offset + keep] = ns[chosen]
                # Match native FP32 multiply exactly; no sampled renormalization.
                vals[offset:offset + keep] = coefficients[chosen] * (count / keep)
                offset += keep
        return torch.sparse_coo_tensor(torch.from_numpy(np.stack([rows, cols])), torch.from_numpy(vals), self.full.shape).coalesce()

    def plans(self, budget, stream, layers, dataset, purpose):
        legacy = dataset == "Airport" and purpose == "selection"
        if legacy and (budget != 8 or stream not in range(83000, 83004)):
            raise ValueError("Legacy exception applies only to Airport fixed K8 selectors")
        return [self.plan(budget, stream, layer, legacy_selection=legacy) for layer in range(layers)]

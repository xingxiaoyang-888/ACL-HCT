"""Actual official HGCN encoder with a separately identified directed task head."""
from types import SimpleNamespace
import math
import torch
from torch import nn


def encoder_args(input_dim, hidden, c, dropout, device):
    return SimpleNamespace(task='lp', model='HGCN', manifold='PoincareBall', feat_dim=input_dim,
                           dim=hidden, c=c, num_layers=2, act='relu', bias=1,
                           use_att=0, local_agg=0, dropout=dropout,
                           cuda=-1 if torch.device(device).type == 'cpu' else 0, device=str(device))


class MatureHGCN(nn.Module):
    """PoincareBall, fixed curvature, two official graph-convolution layers.

    The only encoder adaptation is supplying an independent sampled adjacency
    to each layer. The official input map, linear/bias maps, aggregation,
    projections and ReLU remain executed by the external upstream classes.
    """
    def __init__(self, upstream, input_dim=128, hidden=128, head_hidden=128, c=1., dropout=0., device='cpu'):
        super().__init__()
        if any(type(x) is not int or x < 1 for x in (input_dim, hidden, head_hidden)):
            raise ValueError('positive model dimensions required')
        if type(c) not in (int, float) or not math.isfinite(c) or c <= 0:
            raise ValueError('fixed positive curvature required')
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0,1)')
        self.register_buffer('curvature', torch.tensor([c], dtype=torch.float32))
        self.upstream_identity = upstream.identity
        self.encoder = upstream.HGCN(self.curvature, encoder_args(input_dim, hidden, c, dropout, 'cpu'))
        self.relation_head = nn.Sequential(nn.Linear(3 * hidden, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))
        self.to(device)

    def _apply(self, fn):
        result = super()._apply(fn)
        # Upstream fixed curvatures are plain tensors, not registered buffers.
        # Rebind every official use to the single moved/cast fixed buffer.
        if hasattr(self, 'encoder'):
            self.encoder.c = self.curvature
            self.encoder.curvatures = [self.curvature] * 3
            for layer in self.encoder.layers:
                layer.linear.c = self.curvature; layer.agg.c = self.curvature
                layer.hyp_act.c_in = self.curvature; layer.hyp_act.c_out = self.curvature
        return result

    def encode(self, features, adjacencies):
        if (features.ndim != 2 or features.shape[1] != self.encoder.layers[0].linear.in_features
                or features.dtype != self.curvature.dtype or features.device != self.curvature.device
                or not torch.isfinite(features).all()):
            raise ValueError('finite features matching encoder device/dtype/dimension required')
        if len(adjacencies) != 2:
            raise ValueError('two layer adjacencies required')
        for adj in adjacencies:
            if (adj.layout != torch.sparse_coo or adj.shape != (len(features), len(features))
                    or adj.device != features.device or adj.dtype != features.dtype
                    or not torch.isfinite(adj.coalesce().values()).all()):
                raise ValueError('matching finite sparse COO layer adjacency required')
        handles = []
        try:
            for layer, adj in zip(self.encoder.layers, adjacencies):
                handles.append(layer.register_forward_pre_hook(lambda module, inputs, a=adj: ((inputs[0][0], a),)))
            return self.encoder.encode(features, adjacencies[0])
        finally:
            for handle in handles:
                handle.remove()

    def tangent(self, points):
        """Origin orthonormal tangent features: Poincare's origin metric is 4I.

        Factor two aligns their norm with geodesic radius and the existing
        Lorentz spatial tangent head convention. This is a task-head adaptation.
        """
        return 2 * self.encoder.manifold.logmap0(points, self.curvature)

    def score(self, points, queries):
        if (queries.ndim != 2 or queries.shape[1] != 2 or queries.dtype != torch.long
                or queries.device != points.device or ((queries < 0) | (queries >= len(points))).any()):
            raise ValueError('valid device int64 [Q,2] queries required')
        coordinates = self.tangent(points)
        parent, child = coordinates[queries[:, 0]], coordinates[queries[:, 1]]
        return self.relation_head(torch.cat((parent, child, parent - child), dim=-1)).squeeze(-1)


def to_lorentz_fp64(points, c=1.):
    """Evaluation-only exact ball/hyperboloid isometry after an explicit cast.

    No reprojection, learned transform or clipping; the native ball points are
    saved too, so CPU audits can reproduce the conversion independently.
    """
    if type(c) not in (int, float) or not math.isfinite(c) or c <= 0:
        raise ValueError('positive curvature required')
    p = points.detach().to(dtype=torch.float64)
    square = (p * p).sum(dim=-1, keepdim=True)
    denominator = 1 - c * square
    if p.ndim != 2 or not torch.isfinite(p).all() or (denominator <= 0).any():
        raise ValueError('finite interior ball matrix required')
    return torch.cat(((1 + c * square) / (math.sqrt(c) * denominator), 2 * p / denominator), dim=-1)

"""Frozen official HGCN with the versioned amplitude correction modules."""
import torch
from torch import nn

from .hgcn_rtsc_stage_a import FrozenCorrectedHGCN
from .hgcn_tangent_amplitude import AmplitudeCorrection


class FrozenAmplitudeHGCN(FrozenCorrectedHGCN):
    def __init__(self, base, *, hidden=16, tau=.05, eps=1e-8, chunk_edges=4096, scale=4):
        # Do not create/discard Stage A modules: doing so would shift the
        # matched module-initialization random stream.
        nn.Module.__init__(self)
        if len(base.encoder.layers) != 2 or any(layer.agg.use_att or layer.agg.local_agg
                                                for layer in base.encoder.layers):
            raise ValueError('registered two-layer origin nonattention HGCN required')
        self.base = base
        self.base.requires_grad_(False)
        self.base.eval()
        c = float(base.curvature.item())
        self.corrections = nn.ModuleList([
            AmplitudeCorrection(c, hidden, tau, eps, chunk_edges, scale),
            AmplitudeCorrection(c, hidden, tau, eps, chunk_edges, scale)])
        self.corrections.to(device=base.curvature.device, dtype=base.curvature.dtype)
        if sum(p.numel() for p in self.corrections.parameters()) != 454:
            raise ValueError('registered two-layer 454-parameter controller required')

    def encode(self, features, plans, *, detailed=False):
        if not detailed:
            return super().encode(features, plans)
        if (len(plans) != 2 or features.ndim != 2
                or features.shape[1] != self.base.encoder.layers[0].linear.in_features
                or features.device != self.base.curvature.device or features.dtype != self.base.curvature.dtype
                or not torch.isfinite(features).all()):
            raise ValueError('registered feature matrix and two sampling plans required')
        manifold = self.base.encoder.manifold
        c = self.base.curvature
        current = manifold.proj(manifold.expmap0(manifold.proj_tan0(features, c), c=c), c=c)
        diagnostics = []
        for layer, correction, plan in zip(self.base.encoder.layers, self.corrections, plans):
            adjacency = plan.matrix(device=features.device, dtype=features.dtype)
            messages = layer.linear.forward(current)
            aggregate = layer.agg.forward(messages, adjacency)
            adjusted, measured = correction(aggregate, messages, plan, manifold, detailed=True)
            current = layer.hyp_act.forward(adjusted)
            diagnostics.append(measured)
        return current, diagnostics

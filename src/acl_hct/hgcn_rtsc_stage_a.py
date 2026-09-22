"""Separate frozen-HGCN wrapper for the R-TSC Stage A method path."""
import torch
from torch import nn

from .hgcn_tangent_correction import TangentCorrection


class FrozenCorrectedHGCN(nn.Module):
    """Official linear/aggregate/activate sequence with two inserted modules.

    Upstream layers invoke child `.forward` directly, so ordinary module hooks
    cannot insert a correction between HypAgg and HypAct. This wrapper calls the
    same official child methods in the same order and leaves the accepted base
    implementation untouched.
    """

    def __init__(self, base, *, hidden=16, tau=.05, eps=1e-8, chunk_edges=4096):
        super().__init__()
        if len(base.encoder.layers) != 2 or base.encoder.layers[0].agg.use_att or base.encoder.layers[1].agg.use_att:
            raise ValueError('registered two-layer nonattention HGCN required')
        if any(layer.agg.local_agg for layer in base.encoder.layers):
            raise ValueError('registered origin aggregation required')
        self.base = base
        self.base.requires_grad_(False)
        self.base.eval()
        c = float(base.curvature.item())
        self.corrections = nn.ModuleList([
            TangentCorrection(c, hidden, tau, eps, chunk_edges),
            TangentCorrection(c, hidden, tau, eps, chunk_edges),
        ])
        self.corrections.to(device=base.curvature.device, dtype=base.curvature.dtype)
        if sum(p.numel() for p in self.corrections.parameters()) != 454:
            raise ValueError('registered two-layer 454-parameter controller required')

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def trainable_parameters(self):
        return self.corrections.parameters()

    def encode(self, features, plans):
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
            adjusted, measured = correction(aggregate, messages, plan, manifold)
            current = layer.hyp_act.forward(adjusted)
            diagnostics.append(measured)
        return current, diagnostics

    def score(self, points, queries):
        return self.base.score(points, queries)

    @staticmethod
    def step_penalty(diagnostics):
        if len(diagnostics) != 2:
            raise ValueError('two correction diagnostics required')
        return torch.stack([row['normalized_step_mean'] for row in diagnostics]).mean()

"""Trainable HGCN and old-hard-cap correction paths for the joint comparison."""
import math

import torch
from torch import nn

from .hgcn_rtsc_stage_a import FrozenCorrectedHGCN
from .hgcn_tangent_correction import TangentCorrection


class _OnlyFirstDirection(nn.Module):
    """Expose one learned coefficient in the old three-direction algebra."""

    def forward(self, coefficient):
        return torch.cat((coefficient, torch.zeros_like(coefficient),
                          torch.zeros_like(coefficient)), dim=-1)


class SingleDirectionCorrection(TangentCorrection):
    """Old b1 geometry and ten features, with only one learnable output."""

    def __init__(self, c=1., hidden=16, tau=.05, eps=1e-8, chunk_edges=4096):
        if (not math.isfinite(c) or c <= 0 or type(hidden) is not int or hidden < 1
                or not math.isfinite(tau) or tau <= 0 or not math.isfinite(eps) or eps <= 0
                or type(chunk_edges) is not int or chunk_edges < 1):
            raise ValueError('positive finite correction geometry and chunk settings required')
        nn.Module.__init__(self)
        self.c, self.tau, self.eps, self.chunk_edges = float(c), float(tau), float(eps), chunk_edges
        self.coefficients = nn.Sequential(nn.Linear(10, hidden), nn.SiLU(),
                                          nn.Linear(hidden, 1), _OnlyFirstDirection())
        nn.init.zeros_(self.coefficients[2].weight)
        nn.init.zeros_(self.coefficients[2].bias)
        if sum(p.numel() for p in self.parameters()) != 193:
            raise ValueError('registered 193-parameter single-direction layer required')


class JointCorrectedHGCN(FrozenCorrectedHGCN):
    """Keep the accepted Stage A forward algebra; train the base and module."""

    def __init__(self, base, direction, *, hidden=16, tau=.05, eps=1e-8,
                 chunk_edges=4096, layer_init_seeds=None):
        if direction not in ('three', 'single'):
            raise ValueError('three or single registered correction required')
        if (not isinstance(layer_init_seeds, (tuple, list))
                or len(layer_init_seeds) != 2
                or any(type(seed) is not int or not 0 <= seed < (1 << 63)
                       for seed in layer_init_seeds)
                or layer_init_seeds[0] == layer_init_seeds[1]):
            raise ValueError('two distinct registered module-layer seeds required')
        nn.Module.__init__(self)
        if (len(base.encoder.layers) != 2
                or any(layer.agg.use_att or layer.agg.local_agg
                       for layer in base.encoder.layers)):
            raise ValueError('registered two-layer origin nonattention HGCN required')
        self.base = base
        c = float(base.curvature.item())
        correction_type = (TangentCorrection if direction == 'three'
                           else SingleDirectionCorrection)
        modules = []
        for layer_seed in layer_init_seeds:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(layer_seed)
                modules.append(correction_type(c, hidden, tau, eps, chunk_edges))
        self.corrections = nn.ModuleList(modules)
        self.corrections.to(device=base.curvature.device, dtype=base.curvature.dtype)
        self.layer_init_seeds = tuple(layer_init_seeds)
        self.direction = direction
        self.base.requires_grad_(True)
        expected = 454 if direction == 'three' else 386
        if sum(p.numel() for p in self.corrections.parameters()) != expected:
            raise ValueError('registered joint correction parameter count required')

    def train(self, mode=True):
        nn.Module.train(self, mode)
        return self

    def trainable_parameters(self):
        return self.parameters()


def joint_base(model):
    return model.base if isinstance(model, JointCorrectedHGCN) else model


def encode_joint(model, features, plans):
    if isinstance(model, JointCorrectedHGCN):
        return model.encode(features, plans)
    if len(plans) != 2:
        raise ValueError('two matched layer plans required')
    return model.encode(features, [plan.matrix(device=features.device,
                                                dtype=features.dtype)
                                   for plan in plans]), []

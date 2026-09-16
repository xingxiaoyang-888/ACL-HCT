"""Small self-built Lorentz mean GNN; not a reproduction of a named model."""
import torch
from torch import nn
from .geometry import from_spatial, log, origin_like
from .aggregation import sample_indices, correct


class TinyGNN(nn.Module):
    def __init__(self, features=4, hidden=4, c=1.):
        super().__init__()
        self.linear=nn.Linear(features,hidden)
        self.head=nn.Linear(hidden,1)
        self.c=c

    def forward(self, features, neighbors, generator, fanout=3, method="none"):
        # Bounded tangent parameterization stays inside documented numeric domain.
        h=from_spatial(.3*torch.tanh(self.linear(features)), self.c)
        values=[]; stats=[]
        for i, candidates in enumerate(neighbors):
            # Empty neighborhood is explicitly the node itself; deduplicate IDs.
            ids=sorted(set(candidates)) or [i]
            ix=sample_indices(len(ids),fanout,generator)
            chosen=torch.tensor(ids,device=h.device)[ix.to(h.device)]
            v, s=correct(h[chosen],len(ids),self.c,method)
            values.append(v); stats.append(s)
        out=torch.stack(values)
        tangent=log(origin_like(out,self.c),out,self.c)[...,1:]
        return self.head(tangent).squeeze(-1), out, stats

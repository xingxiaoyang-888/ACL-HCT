"""Small self-built Lorentz mean GNN; not a reproduction of a named model."""
import torch
from torch import nn
from .geometry import from_spatial, log, origin_like
from .aggregation import sample_indices, correct, correct_batched


class TinyGNN(nn.Module):
    def __init__(self, features=4, hidden=4, c=1.):
        super().__init__()
        self.linear=nn.Linear(features,hidden)
        self.head=nn.Linear(hidden,1)
        self.c=c

    def forward(self, features, neighbors, generator, fanout=3, method="none", batched=False):
        """Return predictions, points, and diagnostics.

        batched=False returns legacy per-row dictionaries; True returns a dict
        of device tensors. Both consume the same CPU sampling stream, including
        full fanout and empty rows (which consume no random numbers).
        """
        # Bounded tangent parameterization stays inside documented numeric domain.
        h=from_spatial(.3*torch.tanh(self.linear(features)), self.c)
        if features.ndim != 2 or len(neighbors) != features.shape[0] or not neighbors:
            raise ValueError("one neighborhood per feature row is required")
        values=[]; stats=[]; selections=[]; populations=[]
        for i, candidates in enumerate(neighbors):
            # Empty neighborhood is explicitly the node itself; deduplicate IDs.
            if any(type(j) is not int or j < 0 or j >= len(neighbors) for j in candidates):
                raise ValueError("neighbor IDs must be valid integer node indices")
            ids=sorted(set(candidates)) or [i]
            ix=sample_indices(len(ids),fanout,generator)
            chosen=torch.tensor(ids,device=h.device)[ix.to(h.device)]
            if batched:
                selections.append(chosen if candidates else chosen[:0])
                populations.append(len(ids) if candidates else 0)
                continue
            v, s=correct(h[chosen],len(ids),self.c,method)
            values.append(v); stats.append(s)
        if batched:
            width=max(1, max(len(chosen) for chosen in selections))
            indices=torch.zeros((len(neighbors), width), dtype=torch.long, device=h.device)
            mask=torch.zeros_like(indices, dtype=torch.bool)
            for i, chosen in enumerate(selections):
                indices[i, :len(chosen)]=chosen
                mask[i, :len(chosen)]=True
            out, stats=correct_batched(h[indices], torch.tensor(populations, device=h.device),
                                      mask, self.c, method, self_points=h)
        else:
            out=torch.stack(values)
        tangent=log(origin_like(out,self.c),out,self.c)[...,1:]
        return self.head(tangent).squeeze(-1), out, stats

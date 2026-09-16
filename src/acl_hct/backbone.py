"""Self-built two-layer equal-weight Lorentz-Mean backbone and shared relation head.

This is an explicit baseline definition, not a named-paper reproduction. No
hierarchy labels, full-graph oracle states, or unsampled messages enter correction.
"""
import math
import torch
from torch import nn
from .geometry import curvature, from_spatial, log, origin_like
from .aggregation import sample_indices, correct_batched


def validate_neighbors(neighbors, n):
    if len(neighbors)!=n or n<1: raise ValueError('one neighborhood per node required')
    for row in neighbors:
        if len(row)!=len(set(row)) or any(type(i) is not int or not 0<=i<n for i in row):
            raise ValueError('neighbors must be distinct valid integer indices')


def make_plan(neighbors, fanout, generator):
    """Return reusable CPU IDs for one layer, with identical sampling per method."""
    validate_neighbors(neighbors,len(neighbors))
    if fanout is not None and (type(fanout) is not int or fanout<1): raise ValueError('fanout must be positive or None')
    plan=[]
    for row in neighbors:
        if not row: plan.append([]); continue
        indices=sample_indices(len(row),len(row) if fanout is None else fanout,generator)
        plan.append([row[i] for i in indices.tolist()])
    return plan


def aggregate_plan(points, neighbors, plan, c=1., method='none', max_padded_messages=32768):
    """Chunk by padded message budget; every diagnostic remains on device.

    N comes from query-masked candidates. Plan cannot introduce other IDs or
    duplicates; nonempty candidates require at least one selected message.
    Full and sampled paths share this implementation and parameter values.
    """
    validate_neighbors(neighbors,len(points))
    if len(plan)!=len(points) or type(max_padded_messages) is not int or max_padded_messages<1:
        raise ValueError('invalid plan length or padded-message budget')
    for candidates,chosen in zip(neighbors,plan):
        if (any(type(i) is not int for i in chosen) or len(chosen)!=len(set(chosen)) or not set(chosen).issubset(candidates)
                or bool(candidates)!=bool(chosen)):
            raise ValueError('plan must sample distinct visible candidates; empty iff candidate set empty')
        if len(chosen)>max_padded_messages: raise ValueError('one row exceeds padded message budget')
    values=[]; metadata={}; start=0
    while start<len(points):
        end=start; width=1
        while end<len(points):
            candidate_width=max(width,len(plan[end]))
            if (end-start+1)*candidate_width>max_padded_messages: break
            width=candidate_width; end+=1
        ids=torch.zeros((end-start,width),dtype=torch.long,device=points.device)
        mask=torch.zeros_like(ids,dtype=torch.bool)
        for row,chosen in enumerate(plan[start:end]):
            if chosen:
                ids[row,:len(chosen)]=torch.tensor(chosen,device=points.device)
                mask[row,:len(chosen)]=True
        population=torch.tensor([len(row) for row in neighbors[start:end]],device=points.device)
        output,stats=correct_batched(points[ids],population,mask,c,method,self_points=points[start:end])
        values.append(output)
        for key,value in stats.items(): metadata.setdefault(key,[]).append(value)
        start=end
    return torch.cat(values),{key:torch.cat(value) for key,value in metadata.items()}


class LorentzMeanLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, c=1., scaled_radius=1.2):
        super().__init__(); curvature(c)
        if not math.isfinite(scaled_radius) or not 0<scaled_radius<=2.5:
            raise ValueError('layer scaled radius must be in (0,2.5]')
        self.linear=nn.Linear(input_dim,hidden_dim); self.c=c; self.scaled_radius=scaled_radius

    def messages(self, inputs):
        u=self.linear(inputs)
        # Smooth radial squash: norm is strictly bounded independent of width.
        v=(self.scaled_radius/math.sqrt(self.c))*u/(1+u.square().sum(-1,keepdim=True)).sqrt()
        return from_spatial(v,self.c)


class LorentzMeanNetwork(nn.Module):
    def __init__(self, features, hidden=128, c=1., scaled_radius=1.2, head_hidden=128):
        super().__init__(); self.c=c
        self.layers=nn.ModuleList([LorentzMeanLayer(features,hidden,c,scaled_radius),
                                  LorentzMeanLayer(hidden,hidden,c,scaled_radius)])
        self.relation_head=nn.Sequential(nn.Linear(3*hidden,head_hidden),nn.ReLU(),nn.Linear(head_hidden,1))

    def encode(self, features, neighbors, plans, method='none', max_padded_messages=32768, return_trace=False):
        if len(plans)!=2: raise ValueError('exactly two layer plans required')
        inputs=features; trace=[]; diagnostics=[]
        for layer,plan in zip(self.layers,plans):
            messages=layer.messages(inputs)
            output,stats=aggregate_plan(messages,neighbors,plan,self.c,method,max_padded_messages)
            if return_trace: trace.append({'inputs':inputs,'messages':messages,'output':output})
            diagnostics.append(stats)
            inputs=log(origin_like(output,self.c),output,self.c)[...,1:]
        return output,diagnostics,trace

    def score(self, points, queries):
        if queries.ndim!=2 or queries.shape[-1]!=2 or queries.dtype!=torch.long or queries.device!=points.device:
            raise ValueError('queries must be device int64 [Q,2]')
        if ((queries<0)|(queries>=len(points))).any(): raise ValueError('invalid query node index')
        inputs=log(origin_like(points,self.c),points,self.c)[...,1:]
        parent,child=inputs[queries[:,0]],inputs[queries[:,1]]
        return self.relation_head(torch.cat((parent,child,parent-child),-1)).squeeze(-1)

    @torch.no_grad()
    def full_reference(self, features, neighbors, max_padded_messages=32768):
        """Evaluation-only cache for single-layer fixed-input frozen diagnostics."""
        plans=[make_plan(neighbors,None,torch.Generator()) for _ in range(2)]
        output,stats,trace=self.encode(features,neighbors,plans,max_padded_messages=max_padded_messages,return_trace=True)
        return {'output':output.detach(), 'layers':[{key:value.detach() for key,value in row.items()} for row in trace],
                'diagnostics':stats}

    @torch.no_grad()
    def frozen_layer(self, reference, index, neighbors, plan, method='none', max_padded_messages=32768):
        if index not in (0,1): raise ValueError('layer index must be 0 or 1')
        # This is an evaluator: earlier layer inputs are fixed from the cache.
        return aggregate_plan(reference['layers'][index]['messages'],neighbors,plan,self.c,method,max_padded_messages)

"""Frozen last-message-layer four-way intervention, no labels or data IO.

Every replicate pre-samples a full-graph plan for every layer. The sampled
prefix computes all node states on that plan, including states needed when
restoring local neighbors. Four outcomes share these incoming representations
and the last-layer plan. This diagnostic baseline is not the independently
planned deployment request baseline. Its oracle caches and costs are paid and
reported separately; no oracle states can be supplied to deployment inference.
"""
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.utils import softmax
from shared_blocks import mean_aggregate
from hyperbolic_adapters import fixed_self_adjacency
from sampling_bridge import chunk_blocks


def _check(model):
    if model._study_variant!='plain' or any(m.training for m in model.modules()):
        raise ValueError('frozen plain model in eval mode required')
    if model._study_model_key not in ('sage','gat','hgcn','hybonet'):
        raise ValueError('unsupported native model')


def _layer(model,index,x,block):
    """Actual output plus local estimator coordinates before native post-map.

    SAGE coordinates: raw neighbor mean. GAT: flattened per-head weighted
    aggregate (sampled softmax, including self). HGCN: origin tangent weighted
    sum. HyboNet: ambient weighted sum before Lorentz normalization. Norms in
    these coordinates have no cross-model common scale or curvature meaning.
    """
    key=model._study_model_key
    if key=='sage':
        layer=model.convs[index];support=mean_aggregate(x,block)
        output=layer.lin_l(support)+layer.lin_r(x[block.root_index])
        if index<2:output=F.relu(model.bns[index](output))
    elif key=='gat':
        layer=model.convs[index];src,dst=block.with_self_edges()
        values=layer.lin(x).view(-1,layer.heads,layer.head_dim)
        source_alpha=(values*layer.att_src).sum(-1)
        target_alpha=(values[block.root_index]*layer.att_dst).sum(-1)
        logits=F.leaky_relu(source_alpha[src]+target_alpha[dst],layer.negative_slope)
        alpha=softmax(logits,dst,num_nodes=block.num_dst)
        aggregate=values.new_zeros((block.num_dst,layer.heads,layer.head_dim))
        aggregate.index_add_(0,dst,alpha[...,None]*values[src])
        support=aggregate.flatten(1)
        output=support if layer.concat else aggregate.mean(1)
        if layer.res is not None:output=output+layer.res(x[block.root_index])
        if layer.bias is not None:output=output+layer.bias
        if index<2:output=F.relu(model.norms[index](output))
    else:
        wrapper=model.layers[index];native=wrapper.native
        messages=native.linear(x);adj=fixed_self_adjacency(block,x,wrapper.contract)
        if key=='hgcn':
            support=torch.sparse.mm(adj,native.agg.manifold.logmap0(messages,c=native.agg.c))
        else:support=torch.sparse.mm(adj,messages)
        # Call the native finite nonlinear map unchanged. The separate support
        # trace repeats aggregation/logmap work and is explicitly diagnostic.
        output=native.agg(messages,adj)
        if key=='hgcn':output=native.hyp_act(output)
    if not bool(torch.isfinite(output).all()) or not bool(torch.isfinite(support).all()):
        raise ValueError('nonfinite frozen diagnostic operator; no retry')
    return output,support


def _global_layer(model,index,x,plan,*,retain_support,max_rows,max_edges):
    output=support=None
    for block in chunk_blocks(plan,max_rows,max_edges):
        current=block.to(x.device)
        y,z=_layer(model,index,x[current.src_ids],current)
        if output is None:
            output=y.new_empty((plan.num_dst,y.shape[1]))
            if retain_support:support=z.new_empty((plan.num_dst,z.shape[1]))
        # Global full-graph diagnostic IDs are 0..N-1 in canonical order.
        output[current.dst_ids]=y
        if retain_support:support[current.dst_ids]=z
    return output,support


def _scores(model,hidden):
    if model._study_model_key=='sage':return hidden.log_softmax(-1)
    if model._study_model_key=='gat':return hidden
    return model.decoder.decode(hidden,None)


@torch.no_grad()
def four_way(model,graph,planner,*,sampler,fanout,seed,repeat,device,
             max_rows=4096,max_edges=262144,guard_callback):
    """One method/seed/replicate diagnostic, returning arrays for allowed IDs.

    Raw arrays cover graph nodes but contain no labels. Caller must apply its
    already released target IDs before writing or scoring. All four output
    tensors and local support errors share the exact same sampled last plan.
    """
    _check(model)
    if not callable(guard_callback):raise ValueError('actual resource guard required')
    guard_callback();start=time.perf_counter()
    all_ids=np.arange(graph.n,dtype=np.int64)
    sampled=[planner.layer(graph,all_ids,fanout,sampler=sampler,stage='mechanism-global',seed=seed,repeat=repeat,batch=0,layer=i) for i in range(3)]
    complete=planner.layer(graph,all_ids,-1,sampler='uniform',stage='mechanism-full',seed=seed,repeat=0,batch=0,layer=0)
    # No ghost neighbor state: both prefixes explicitly cover all graph nodes.
    xs=graph.x.to(device)
    if model._study_model_key in ('hgcn','hybonet'):xs=model.encode_input(xs)
    xf=xs
    for i in range(2):
        xs,_=_global_layer(model,i,xs,sampled[i],retain_support=False,max_rows=max_rows,max_edges=max_edges)
        xf,_=_global_layer(model,i,xf,complete,retain_support=False,max_rows=max_rows,max_edges=max_edges)
    guard_callback()
    cache_bytes=sum(v.numel()*v.element_size() for v in (xs,xf))
    a,sa=_global_layer(model,2,xs,sampled[2],retain_support=True,max_rows=max_rows,max_edges=max_edges)
    b,sb=_global_layer(model,2,xs,complete,retain_support=True,max_rows=max_rows,max_edges=max_edges)
    c,sc=_global_layer(model,2,xf,sampled[2],retain_support=True,max_rows=max_rows,max_edges=max_edges)
    d,sd=_global_layer(model,2,xf,complete,retain_support=True,max_rows=max_rows,max_edges=max_edges)
    result={'scores':{'sampled':_scores(model,a).cpu(),'local_restored':_scores(model,b).cpu(),
        'upstream_restored':_scores(model,c).cpu(),'both_restored':_scores(model,d).cpu()},
        'local_estimator_sq_l2_sampled_inputs':((sa-sb)**2).sum(-1).cpu(),
        'local_estimator_sq_l2_full_inputs':((sc-sd)**2).sum(-1).cpu(),
        'upstream_input_sq_l2':((xs-xf)**2).sum(-1).cpu(),
        'diagnostic_cache_bytes':cache_bytes,'sampling_counts':[p.report() for p in sampled],
        'scope':'global_sampled_prefix_diagnostic_not_deployment_request',
        'sampler':sampler,'fanout':fanout,'seed':seed,'repeat':repeat,
        'last_layer_number_forward_order':3,'diagnostic_support_double_work_included':True}
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)
    result['diagnostic_wall_seconds']=time.perf_counter()-start
    guard_callback()
    return result

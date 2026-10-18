"""Two frozen samplers. Native LABOR-0 is not reimplemented or reweighted.

Only prob=None / importance_sampling=0 is supported. A nonempty importance
tensor fails, rather than being silently discarded by the attention operator.
Uniform uses the accepted native DGL WOR path. Each call owns its private
stage/seed/repeat/batch/layer domain and the inherited planning thread scope.
"""
import hashlib
import json
import numpy as np
import torch
from shared_blocks import NativeBlock, validate_block_chain

ROUND = 'vldb-sampling-mechanisms-v1'


def plan_seed(stage, seed, repeat, batch, layer):
    value=json.dumps([ROUND,stage,seed,repeat,batch,layer],separators=(',',':'))
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8],'little') & ((1<<63)-1)


class PlanningBridge:
    def __init__(self,native_uniform_edges,native_thread_scope):
        self.uniform=native_uniform_edges;self.scope=native_thread_scope

    def layer(self,graph,targets,k,*,sampler,stage,seed,repeat,batch,layer):
        if graph.kind!='sage' or sampler not in ('uniform','labor') or type(k) is not int or (k!=-1 and k<1):
            raise ValueError('public unweighted graph, fixed sampler and fanout required')
        dst=np.asarray(targets,dtype=np.int64)
        if dst.ndim!=1 or not len(dst) or len(np.unique(dst))!=len(dst) or np.any(dst<0) or np.any(dst>=graph.n):
            raise ValueError('distinct in-range targets required')
        salt=plan_seed(stage,seed,repeat,batch,layer)
        with self.scope():
            if sampler=='uniform' or k==-1:
                source,rows,_coefficients,counts=self.uniform(graph,dst,k,salt)
                expected=graph.degree[dst] if k==-1 else np.minimum(graph.degree[dst],k)
                if not np.array_equal(counts,expected):raise ValueError('WOR counts differ')
                provenance='full' if k==-1 else 'uniform_wor';labor_k=None
            else:
                import dgl
                native=graph.dgl_graph()
                frontier,importance=dgl.sampling.sample_labors(native,torch.from_numpy(dst),k,
                    edge_dir='in',prob=None,importance_sampling=0,
                    random_seed=dgl.ndarray.array(np.array([salt],dtype=np.int64)),
                    copy_ndata=False,copy_edata=True)
                if len(importance)!=1 or importance[0].numel()!=0:
                    raise ValueError('pinned uniform LABOR-0 must return empty importance weights')
                source,dest=(a.numpy() for a in frontier.edges(order='eid'))
                positions=np.full(graph.n,-1,dtype=np.int64);positions[dst]=np.arange(len(dst))
                rows=positions[dest]
                if np.any(rows<0):raise ValueError('LABOR edge destination escaped requested rows')
                counts=np.bincount(rows,minlength=len(dst));provenance='labor0';labor_k=k
            source=np.asarray(source,dtype=np.int64);rows=np.asarray(rows,dtype=np.int64)
            src=np.unique(np.r_[source,dst])
            block=NativeBlock(torch.from_numpy(src.copy()),torch.from_numpy(dst.copy()),
                torch.from_numpy(np.searchsorted(src,source)),torch.from_numpy(rows.copy()),
                torch.from_numpy(np.searchsorted(src,dst)),torch.from_numpy(graph.degree[dst].copy()),
                sampler=provenance,labor_fanout=labor_k)
            if not np.array_equal(block.counts.numpy(),counts):raise ValueError('counts/edge order differ')
            return block

    def plan(self,graph,targets,fanouts,*,sampler,stage,seed,repeat,batch):
        blocks=[];dst=np.asarray(targets,dtype=np.int64)
        with self.scope():
            for i in reversed(range(len(fanouts))):
                block=self.layer(graph,dst,fanouts[i],sampler=sampler,stage=stage,seed=seed,repeat=repeat,batch=batch,layer=i)
                blocks.insert(0,block);dst=block.src_ids.numpy()
        validate_block_chain(blocks)
        return blocks


def chunk_blocks(block,max_rows=4096,max_edges=262144):
    """Split complete destination rows only; retain the one pre-sampled plan.

    No new random draws, missing-neighbor repair, or attention-denominator split.
    Returned blocks use local source indices into their exact global ID list.
    This CPU preparation is charged to diagnostic execution, not deployment.
    """
    block.assert_immutable()
    if block.src_ids.device.type!='cpu' or max_rows<1 or max_edges<1:
        raise ValueError('CPU plan and positive frozen limits required')
    count=block.counts.numpy();ptr=np.r_[0,np.cumsum(count)]
    if np.any(count+1>max_edges):raise ValueError('complete row exceeds fixed diagnostic chunk bound')
    order=np.argsort(block.edge_dst.numpy(),kind='stable')
    global_source=block.src_ids.numpy()[block.edge_src.numpy()[order]]
    all_dest=block.dst_ids.numpy();start=0
    while start<block.num_dst:
        end=min(block.num_dst,start+max_rows)
        while end>start and ptr[end]-ptr[start]+end-start>max_edges:end-=1
        if end==start:raise ValueError('no complete-row chunk fits')
        dst=all_dest[start:end].copy();source=global_source[ptr[start]:ptr[end]].copy()
        row=np.repeat(np.arange(end-start,dtype=np.int64),count[start:end])
        src=np.unique(np.r_[source,dst])
        yield NativeBlock(torch.from_numpy(src),torch.from_numpy(dst),torch.from_numpy(np.searchsorted(src,source)),
            torch.from_numpy(row),torch.from_numpy(np.searchsorted(src,dst)),block.nonself_degree[start:end].clone(),
            sampler=block.sampler,labor_fanout=block.labor_fanout)
        start=end

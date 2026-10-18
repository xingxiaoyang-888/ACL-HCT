"""New global-layer implementation versus accepted native forward, toy only.

Call one model per fresh released process. Native config/tree is separately
bound by the caller. This does not repeat gradient checks or open task CP/data.
"""
from contextlib import nullcontext
from types import SimpleNamespace
import numpy as np
import torch
from model_builders import build_model, native_scores
from mechanism import four_way
from sampling_bridge import PlanningBridge


def run_native_correspondence(model_key, config, device, guard_callback):
    if model_key not in ('sage','gat','hgcn','hybonet') or not callable(guard_callback):
        raise ValueError('exact model and actual release guard required')
    guard_callback()
    neighbors={0:[1,2,5],1:[0,2],2:[0,1,3],3:[2,4],4:[3,5],5:[0,4]}
    def uniform(g,dst,k,salt):
        if k!=-1:raise ValueError('correspondence fixture full topology only')
        source=[];rows=[]
        for row,target in enumerate(dst):
            source.extend(neighbors[int(target)]);rows.extend([row]*len(neighbors[int(target)]))
        rows=np.asarray(rows,dtype=np.int64)
        return np.asarray(source,dtype=np.int64),rows,np.ones(len(rows)),np.bincount(rows,minlength=len(dst))
    graph=SimpleNamespace(kind='sage',n=6,degree=np.array([3,2,3,2,2,2],dtype=np.int64),
                          x=torch.linspace(-.05,.05,6*128,dtype=torch.float32).reshape(6,128))
    planner=PlanningBridge(uniform,nullcontext)
    model=build_model(model_key,'plain',11,config,device).eval()
    with torch.no_grad():
        plan=planner.plan(graph,np.arange(6),[-1]*3,sampler='uniform',stage='new-native-correspondence',seed=11,repeat=0,batch=0)
        expected=native_scores(model,graph.x.to(device),[block.to(device) for block in plan],correction_enabled=False).cpu()
        result=four_way(model,graph,planner,sampler='uniform',fanout=-1,seed=11,repeat=0,device=device,
                        max_rows=2,max_edges=16,guard_callback=guard_callback)
    differences={}
    for branch,scores in result['scores'].items():
        torch.testing.assert_close(scores,expected,atol=3e-4,rtol=2e-4)
        differences[branch]=float((scores-expected).abs().max())
    guard_callback()
    return {'schema':'vldb_native_global_correspondence_v1','model':model_key,'device':str(device),
            'synthetic_nodes':6,'native_full_forward_compared':True,'max_abs_errors':differences,
            'atol':3e-4,'rtol':2e-4,'gradients_checked':False,'task_data_or_checkpoint_opened':False}

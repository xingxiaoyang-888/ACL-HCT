"""Independent retained pre-optimization paths; CUDA coverage delegated to a job."""
import os
import pytest
import torch
from acl_hct.aggregation import sample_indices, correct_batched
from acl_hct.backbone import aggregate_plan, make_plan
from acl_hct.geometry import from_spatial

GRAPH=[[],[0],[0,1],[0,1,2],[0,1,2,3,5,6,7,8],[1,2,3,4,6,7],[2,3],[0,2,4,6],[1,2,3,4,5,6,7]]


def old_plan(neighbors,fanout,generator):
    result=[]
    for row in neighbors:
        if not row:result.append([]);continue
        ids=sample_indices(len(row),len(row) if fanout is None else fanout,generator)
        result.append([row[i] for i in ids.tolist()])
    return result


def old_aggregate(points,neighbors,plan,c,method,budget):
    values=[];metadata={};start=0
    while start<len(points):
        end=start;width=1
        while end<len(points):
            candidate_width=max(width,len(plan[end]))
            if (end-start+1)*candidate_width>budget:break
            width=candidate_width;end+=1
        ids=torch.zeros((end-start,width),dtype=torch.long,device=points.device)
        mask=torch.zeros_like(ids,dtype=torch.bool)
        for row,chosen in enumerate(plan[start:end]):
            if chosen:
                ids[row,:len(chosen)]=torch.tensor(chosen,device=points.device)
                mask[row,:len(chosen)]=True
        population=torch.tensor([len(row) for row in neighbors[start:end]],device=points.device)
        output,stats=correct_batched(points[ids],population,mask,c,method,self_points=points[start:end])
        values.append(output)
        for key,value in stats.items():metadata.setdefault(key,[]).append(value)
        start=end
    return torch.cat(values),{key:torch.cat(value) for key,value in metadata.items()}


@pytest.mark.parametrize('fanout',[1,2,3,9,None])
def test_plan_and_rng_stream_unchanged(fanout):
    left=torch.Generator().manual_seed(113);right=torch.Generator().manual_seed(113)
    for _ in range(8):
        assert make_plan(GRAPH,fanout,left)==old_plan(GRAPH,fanout,right)
        assert torch.equal(left.get_state(),right.get_state())


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('method',['none','third','jackknife'])
def test_chunk_pack_output_gradient_and_diagnostics(device,dtype,method):
    if device=='cuda' and (not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available()):
        pytest.skip('CUDA requires a separately allocated test run')
    generator=torch.Generator().manual_seed(41)
    spatial=(torch.randn(9,4,dtype=dtype,generator=generator)*.16).to(device).requires_grad_()
    points=from_spatial(spatial,c=4.)
    plan=old_plan(GRAPH,3,torch.Generator().manual_seed(57))
    new,stats=aggregate_plan(points,GRAPH,plan,c=4.,method=method,max_padded_messages=10)
    old,old_stats=old_aggregate(points,GRAPH,plan,4.,method,10)
    assert torch.equal(new,old)
    for key in stats:assert torch.equal(stats[key],old_stats[key])
    assert stats['empty'][0] and stats['k'][0]==0 and stats['N'][0]==0
    assert stats['full'][1:4].all()
    if method=='none':assert stats['fallback'].tolist()==[True]+[False]*8
    if method!='none':assert stats['fallback'][:4].all()
    weight=torch.randn(new.shape,dtype=dtype,generator=generator).to(device)
    a=torch.autograd.grad((new*weight).sum(),spatial,retain_graph=True)[0]
    b=torch.autograd.grad((old*weight).sum(),spatial)[0]
    torch.testing.assert_close(a,b,atol=3e-7 if dtype==torch.float32 else 1e-14,
                               rtol=3e-6 if dtype==torch.float32 else 1e-12)
    assert torch.isfinite(a).all()


def test_full_generator_contract_and_all_empty():
    with pytest.raises(ValueError):make_plan([[0]],None,None)
    assert make_plan([[],[]],3,None)==[[],[]]  # Original all-empty behavior retained.

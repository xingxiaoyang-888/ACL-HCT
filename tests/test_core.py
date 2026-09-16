import importlib.util
from itertools import combinations
from pathlib import Path
import numpy as np
import pytest
import torch
from acl_hct.geometry import from_spatial, dot, exp, log, norm2, origin_like, check_point
from acl_hct.aggregation import aggregate, sample_indices, correct
from acl_hct.diagnostics import frozen_case


@pytest.mark.parametrize("c", [.1,1.,4.])
def test_geometry_fp64(c):
    g=torch.Generator().manual_seed(71)
    x=from_spatial(torch.randn(7,3,dtype=torch.float64,generator=g)*.12,c)
    y=from_spatial(torch.randn(7,3,dtype=torch.float64,generator=g)*.15,c)
    v=log(x,y,c)
    torch.testing.assert_close(c*dot(x,x),-torch.ones(7,dtype=x.dtype),atol=2e-13,rtol=0)
    torch.testing.assert_close(dot(x,v),torch.zeros(7,dtype=x.dtype),atol=2e-13,rtol=0)
    torch.testing.assert_close(exp(x,v,c),y,atol=2e-13,rtol=1e-12)
    assert torch.equal(log(x,x,c),torch.zeros_like(x))
    torch.testing.assert_close(exp(x,torch.zeros_like(x),c),x)


@pytest.mark.parametrize("scale", [0.,1e-9,.2])
def test_gradients_and_gradcheck(scale):
    u=torch.full((2,2),scale,dtype=torch.float64,requires_grad=True)
    def f(a):
        x=from_spatial(a)
        return log(origin_like(x),exp(x,log(x,x))) + x
    assert torch.autograd.gradcheck(f,(u,),eps=1e-6,atol=1e-6,rtol=1e-4)
    f(u).sum().backward(); assert torch.isfinite(u.grad).all()


def test_fp32_reference_and_domain():
    u=torch.tensor([[.0,0.],[.01,.02],[1.,.4],[2.5,.1]],dtype=torch.float64)
    reference=from_spatial(u)
    actual=from_spatial(u.float()).double()
    torch.testing.assert_close(actual,reference,atol=3e-6,rtol=3e-6)
    recovered=log(origin_like(actual.float()),actual.float()).double()[...,1:]
    torch.testing.assert_close(recovered,u,atol=3e-5,rtol=3e-5)
    for c in (0.,-1.,float("nan")):
        with pytest.raises(ValueError): from_spatial(u,c)
    with pytest.raises(ValueError): from_spatial(u*3)
    with pytest.raises(ValueError): check_point(torch.ones(3))


def test_aggregation_invariants():
    x=from_spatial(torch.tensor([[.1,.2],[-.1,.1],[.3,0]],dtype=torch.float64))
    w=torch.tensor([1.,2.,4.],dtype=torch.float64)
    torch.testing.assert_close(aggregate(x),aggregate(x[[2,0,1]]))
    torch.testing.assert_close(aggregate(x,weights=w),aggregate(x,weights=w*10))
    torch.testing.assert_close(aggregate(x[:1].repeat(4,1)),x[0])
    with pytest.raises(ValueError): aggregate(x[:0])
    with pytest.raises(ValueError): aggregate(x,weights=-w)


def test_sampling_uniformity_and_full():
    a=torch.Generator().manual_seed(11); b=torch.Generator().manual_seed(11)
    counts={ix:0 for ix in combinations(range(4),2)}
    for _ in range(6000):
        ix=sample_indices(4,2,a)
        assert torch.equal(ix,sample_indices(4,2,b))
        assert len(set(ix.tolist()))==2
        counts[tuple(sorted(ix.tolist()))]+=1
    assert all(abs(v-1000)<120 for v in counts.values())
    assert torch.equal(sample_indices(4,9,a),torch.arange(4))


@pytest.mark.parametrize("k", [1,2,5])
def test_fallback_exact(k):
    x=from_spatial(torch.arange(10,dtype=torch.float64).reshape(5,2)*.02)[:k]
    for method in ("third","jackknife"):
        y,s=correct(x,5,method=method)
        assert torch.equal(y,aggregate(x)); assert s["fallback"]


def test_candidate_numpy_enumeration():
    path=Path(__file__).parents[1]/"research/legacy/numerical_check.py"
    spec=importlib.util.spec_from_file_location("reference",path)
    ref=importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
    x=from_spatial(torch.tensor([[-.1,0.],[-.04,.02],[0.,0.],[.02,.01],[.15,-.02]],dtype=torch.float64))
    for ix in combinations(range(5),3):
        y,s=correct(x[list(ix)],5)
        expected=ref.third_moment_correction(x[list(ix)].numpy(),5)
        np.testing.assert_allclose(y.numpy(),expected,atol=1e-13,rtol=1e-12)
    symmetric=frozen_case(symmetric=True)
    for m in symmetric["methods"].values():
        assert m["bias_norm"]<1e-12
        assert abs(m["mse"]-m["variance"]-m["bias_norm"]**2)<1e-12


def test_candidate_gradient_and_clip():
    u=torch.tensor([[-.3,.1],[.01,-.1],[.8,.1]],dtype=torch.float64,requires_grad=True)
    for method in ("third","jackknife"):
        y,s=correct(from_spatial(u),20,method=method,max_step=1e-5)
        assert s["clipped"] and s["step"] <= 1.000001e-5
        grad=torch.autograd.grad(y.sum(),u)[0]; assert torch.isfinite(grad).all()

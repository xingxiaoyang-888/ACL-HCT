"""E0 fixed-release device regression, owned by the experiment task.

No downloads, formula changes or training sweeps. Same CPU-generated inputs
are copied to each device. FP64 CPU scalar path is the cross-device reference.
Tolerances are fixed before the remote run; all absolute errors are reported.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import torch
import acl_hct
from acl_hct.aggregation import correct, correct_batched
from acl_hct.geometry import from_spatial, log, exp, dot, origin_like
from acl_hct.model import TinyGNN
from acl_hct.smoke import run as smoke

SOURCE_COMMIT = "9e610cbc16acfeff0d0790c63a0a11d28c4b9cb6"


def synchronized(device):
    if device == "cuda":
        torch.cuda.synchronize()


def compare(actual, expected, dtype, gradient=False):
    # CPU-reference tolerances, declared independently of observed errors.
    atol = (2e-5 if gradient else 3e-6) if dtype == torch.float32 else 2e-11
    rtol = 2e-4 if dtype == torch.float32 else 2e-9
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    return {"max_abs_error": float((a-b).abs().max()), "atol": atol, "rtol": rtol}


def ragged_inputs():
    g = torch.Generator().manual_seed(1701)
    spatial = torch.randn(8, 5, 3, dtype=torch.float64, generator=g)*.12
    self_spatial = torch.randn(8, 3, dtype=torch.float64, generator=g)*.1
    lengths = torch.tensor([0, 1, 2, 3, 4, 5, 3, 4])
    populations = torch.tensor([0, 7, 7, 3, 8, 9, 6, 4])
    mask = torch.arange(5)[None, :] < lengths[:, None]
    mask[4] = torch.tensor([True, False, True, True, True])
    weights = torch.randn(8, 4, dtype=torch.float64, generator=g)
    return spatial, self_spatial, populations, mask, weights


def evaluate_ragged(device, dtype, method, c, max_step, batched):
    u, own, n, mask, weights = ragged_inputs()
    u = u.to(device=device, dtype=dtype).requires_grad_()
    own = own.to(device=device, dtype=dtype).requires_grad_()
    n, mask, weights = n.to(device), mask.to(device), weights.to(device=device, dtype=dtype)
    x, self_points = from_spatial(u, c), from_spatial(own, c)
    if batched:
        padded = torch.where(mask[..., None], x, torch.full_like(x, float("nan")))
        y, stats = correct_batched(padded, n, mask, c, method, max_step, self_points)
        base, _ = correct_batched(padded, n, mask, c, "none", max_step, self_points)
        assert y.device.type == device
        assert all(v.device.type == device and not v.requires_grad for v in stats.values())
        inactive = stats["empty"] | stats["full"] | stats["small_sample"]
        assert torch.equal(y[inactive], base[inactive])
        assert torch.equal(y[0], self_points[0])
        assert stats["k"].tolist() == [0, 1, 2, 3, 4, 5, 3, 4]
        assert stats["N"].tolist() == [0, 7, 7, 3, 8, 9, 6, 4]
        expected_fallback = [True, False, False, False, False, False, False, False] if method == "none" else [True, True, True, True, False, False, False, True]
        assert stats["fallback"].tolist() == expected_fallback
        assert (stats["step"] <= max_step*1.00001).all()
    else:
        y = torch.stack([self_points[i] if not mask[i].any() else
                         correct(x[i, mask[i]], int(n[i]), c, method, max_step)[0]
                         for i in range(8)])
        stats = {}
    grads = torch.autograd.grad((y*weights).sum(), (u, own))
    assert all(torch.isfinite(g).all() for g in grads)
    assert torch.equal(grads[0][~mask], torch.zeros_like(grads[0][~mask]))
    return y, grads, stats


def ragged_case(device, dtype, method, c, max_step):
    y, grad, stats = evaluate_ragged(device, dtype, method, c, max_step, True)
    row, rowgrad, _ = evaluate_ragged(device, dtype, method, c, max_step, False)
    oracle, oraclegrad, _ = evaluate_ragged("cpu", torch.float64, method, c, max_step, False)
    return {"kind": "ragged", "device": y.device.type, "dtype": str(dtype),
            "method": method, "c": c, "max_step": max_step,
            "same_device_scalar_output": compare(y, row, dtype),
            "same_device_scalar_gradients": [compare(a,b,dtype,True) for a,b in zip(grad,rowgrad)],
            "fp64_cpu_output": compare(y, oracle, dtype),
            "fp64_cpu_gradients": [compare(a,b,dtype,True) for a,b in zip(grad,oraclegrad)],
            "clipped_rows": int(stats["clipped"].sum()),
            "fallback_rows": int(stats["fallback"].sum()),
            "zero_padding_gradient": True, "exact_full_low_empty_fallback": True}


def coincidence_case(device, dtype, method, delta, c=1.):
    source = torch.tensor([[[.7,-.3]]*4], dtype=torch.float64)
    source[0,1,0] += delta
    outputs, gradients = [], []
    for target, precision in [(device,dtype),("cpu",torch.float64)]:
        u = source.to(device=target,dtype=precision).requires_grad_()
        x = from_spatial(u,c)
        mask = torch.ones(1,4,dtype=torch.bool,device=target)
        y, _ = correct_batched(x,9,mask,c,method)
        # Include smooth Log/Exp composition at a nonorigin point, plus exact
        # coincidence and tangent/manifold constraints, in the backward graph.
        v = log(x[:,0],x[:,1],c)
        recovered = exp(x[:,0],v,c)
        assert torch.equal(log(x,x,c),torch.zeros_like(x))
        constraint = c*dot(y,y)+1
        tol = 2e-5 if precision == torch.float32 else 2e-11
        assert float(constraint.abs().max()) < tol
        assert float(dot(x[:,0],v).abs().max()) < tol
        torch.testing.assert_close(recovered,x[:,1],atol=tol,rtol=tol)
        grad = torch.autograd.grad(y.sum()+recovered.sum(),u)[0]
        assert torch.isfinite(grad).all()
        outputs.append(y); gradients.append(grad)
    return {"kind":"nonorigin_coincidence", "device":device,"dtype":str(dtype),
            "method":method,"delta":delta,"c":c,
            "fp64_cpu_output":compare(outputs[0],outputs[1],dtype),
            "fp64_cpu_gradients":compare(gradients[0],gradients[1],dtype,True)}


def model_case(device, dtype, method):
    torch.manual_seed(31)
    model = TinyGNN().to(device=device,dtype=dtype)
    generator=torch.Generator().manual_seed(31)
    features=torch.randn(6,4,dtype=torch.float64,generator=generator).to(device=device,dtype=dtype)
    neighbors=[[],[0],[0,1],[0,1,2,4,5],[0,1,2,3,5],[1,1,2,3,4]]
    a=model(features,neighbors,torch.Generator().manual_seed(7),method=method)
    b=model(features,neighbors,torch.Generator().manual_seed(7),method=method,batched=True)
    ga=torch.autograd.grad(a[0].square().sum(),tuple(model.parameters()))
    gb=torch.autograd.grad(b[0].square().sum(),tuple(model.parameters()))
    assert a[0].device.type==device and b[0].device.type==device
    return {"kind":"model_parameter_gradients","device":device,"dtype":str(dtype),
            "method":method,"output":compare(a[0],b[0],dtype),
            "gradients":[compare(x,y,dtype,True) for x,y in zip(ga,gb)]}


def run_checks(device):
    if device not in ("cpu","cuda"):
        raise ValueError("device must be cpu or cuda")
    if device=="cuda" and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):
        raise RuntimeError("CUDA regression requires exactly one scheduler-visible GPU; no CPU fallback")
    results=[]
    for dtype in (torch.float32,torch.float64):
        for method in ("none","third","jackknife"):
            for c in (.1,1.,4.):
                for cap in (.1,1e-5):
                    results.append(ragged_case(device,dtype,method,c,cap))
                for delta in (0.,1e-7):
                    results.append(coincidence_case(device,dtype,method,delta,c))
            results.append(model_case(device,dtype,method))
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device",choices=("cpu","cuda"),required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    imported=Path(acl_hct.__file__).resolve()
    assert imported == (args.source_root/"src/acl_hct/__init__.py").resolve(), "wrong imported source release"
    result={"source_commit":SOURCE_COMMIT,"harness_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_files_sha256":{str(f.relative_to(args.source_root)):hashlib.sha256(f.read_bytes()).hexdigest()
                                   for f in sorted((args.source_root/"src").rglob("*.py"))},
            "python":platform.python_version(),"torch":torch.__version__,"cuda_runtime":torch.version.cuda,
            "requested_device":args.device,"status":"running",
            "scope":"E0 engineering regression, no real-data training or method-benefit claim"}
    started=time.perf_counter()
    try:
        if args.device=="cuda":
            assert torch.cuda.is_available() and torch.cuda.device_count()==1
            props=torch.cuda.get_device_properties(0)
            result["hardware"]={"name":props.name,"visible_device_count":torch.cuda.device_count(),
                                "total_memory_bytes":props.total_memory,"capability":list(torch.cuda.get_device_capability(0))}
            torch.cuda.reset_peak_memory_stats()
        synchronized(args.device); checked=time.perf_counter()
        result["checks"]=run_checks(args.device)
        synchronized(args.device)
        result["checks_seconds"]=time.perf_counter()-checked
        result["checks_count"]=len(result["checks"])
        if args.device=="cuda":
            result["check_peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
            result["check_peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
        result["batched_smoke"]=smoke(args.device,steps=8,batched=True)
        result["status"]="passed";result["exit_code"]=0
    except Exception as error:
        result["status"]="failed";result["exit_code"]=1
        result["error"]={"type":type(error).__name__,"message":str(error)}
        raise
    finally:
        synchronized(args.device)
        result["total_seconds"]=time.perf_counter()-started
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"device":args.device,"checks":result["checks_count"]}))


if __name__=="__main__":
    main()

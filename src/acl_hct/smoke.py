"""Bounded synthetic train/evaluation; losses are training diagnostics only."""
import argparse
import json
import platform
import time
from pathlib import Path
import torch
from .data import synthetic_tree
from .model import TinyGNN
from .geometry import distance, origin_like
from .diagnostics import frozen_case


def run(device="cpu", seed=11, steps=8, batched=False):
    if steps < 1 or steps > 30: raise ValueError("smoke steps must be 1..30")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    edges, depths, branches=synthetic_tree()
    n=len(depths)
    features=torch.randn(n,4,device=device)
    target=torch.tensor(depths,dtype=torch.float32,device=device)/3
    neighbors=[[] for _ in depths]
    for a,b in edges:
        neighbors[a].append(b); neighbors[b].append(a)
    initial=TinyGNN().to(device).state_dict()
    results={}
    for method in ("none","third","jackknife"):
        model=TinyGNN().to(device); model.load_state_dict(initial)
        optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        generator=torch.Generator().manual_seed(seed)
        if device=="cuda": torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        started=time.perf_counter(); losses=[]; fallback=clip=total=0
        for _ in range(steps):
            optimizer.zero_grad()
            pred,points,stats=model(features,neighbors,generator,3,method,batched=batched)
            loss=(pred-target).square().mean(); loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("nonfinite gradient")
            optimizer.step(); losses.append(float(loss.detach()))
            if batched:
                # Transfer only two aggregate counters once per optimization step.
                counts=torch.stack((stats["fallback"].sum(),stats["clipped"].sum())).cpu().tolist()
                fallback+=counts[0]; clip+=counts[1]; total+=len(neighbors)
            else:
                fallback+=sum(s["fallback"] for s in stats); clip+=sum(s["clipped"] for s in stats); total+=len(stats)
        with torch.no_grad():
            _,points,_=model(features,neighbors,torch.Generator().manual_seed(seed+1),3,method,batched=batched)
            radius=distance(origin_like(points),points)
            order=float(torch.stack([(radius[b]>radius[a]).float() for a,b in edges]).mean())
            # Explicit semantic branch labels; pairwise distances of synthetic nodes only.
            pairs=[(a,b) for a in range(1,n) for b in range(a+1,n) if depths[a]==depths[b]]
            same=[distance(points[a],points[b]) for a,b in pairs if branches[a]==branches[b]]
            cross=[distance(points[a],points[b]) for a,b in pairs if branches[a]!=branches[b]]
            gap=float(torch.stack(cross).mean()-torch.stack(same).mean())
        if device=="cuda": torch.cuda.synchronize()
        results[method]={"train_losses":losses,"elapsed_seconds":time.perf_counter()-started,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated() if device=="cuda" else None,
            "peak_reserved_bytes":torch.cuda.max_memory_reserved() if device=="cuda" else None,
            "synthetic_parent_child_radius_order_accuracy":order,
            "synthetic_cross_minus_same_branch_distance":gap,
            "fallback_rate":fallback/total,"clipping_rate":clip/total}
    return {"scope":"synthetic engineering smoke; no held-out task-performance claim",
        "seed":seed,"steps":steps,"batched":batched,"torch":torch.__version__,"python":platform.python_version(),
        "device":torch.cuda.get_device_name(0) if device=="cuda" else platform.processor(),
        "cuda_runtime":torch.version.cuda,"training":results,
        "frozen":[frozen_case(spread,k,symmetric=symmetric) for spread in (.15,.7) for k in (1,2,3,5) for symmetric in (False,True)]}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--output",type=Path,required=True); parser.add_argument("--steps",type=int,default=8)
    parser.add_argument("--batched",action="store_true")
    args=parser.parse_args(); result=run(args.device,steps=args.steps,batched=args.batched)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(args.output),"device":result["device"],"methods":list(result["training"])}))

if __name__=="__main__": main()

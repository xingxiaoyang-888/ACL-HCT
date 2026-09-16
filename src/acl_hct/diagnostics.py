"""Exhaustive fixed-population diagnostics; toy evidence, not real hierarchy loss."""
from itertools import combinations
import torch
from .geometry import from_spatial, log, norm2, dot, distance, origin_like
from .aggregation import aggregate, correct


def frozen_case(spread=.15, k=3, c=1., symmetric=False):
    values=[[-1.,0.],[-.4,.2],[-.1,-.1],[.2,.1],[1.3,-.2]]
    if symmetric: values=[[-1.,0.],[-.3,.2],[0.,0.],[.3,-.2],[1.,0.]]
    x=from_spatial(torch.tensor(values,dtype=torch.float64)*spread,c)
    p=aggregate(x,c); result={}
    for method in ("none","third","jackknife"):
        offsets=[]; radii=[]; stats=[]
        for subset in combinations(range(len(x)),k):
            y,info=correct(x[list(subset)],len(x),c,method)
            offsets.append(log(p,y,c)); radii.append(distance(origin_like(y,c),y,c)); stats.append(info)
        z=torch.stack(offsets); b=z.mean(0)
        bias2=float(norm2(b)); mse=float(norm2(z).mean())
        r=distance(origin_like(p,c),p,c)
        radial=-log(p,origin_like(p,c),c)/r if r>1e-12 else torch.zeros_like(p)
        result[method]={"bias_vector":b.tolist(),"bias_norm":bias2**.5,
            "variance":float(norm2(z-b).mean()),"mse":mse,
            "mean_radius_difference":float(torch.stack(radii).mean()-r),
            "radial_bias":float(dot(radial,b)),
            "fallback_rate":sum(s["fallback"] for s in stats)/len(stats),
            "clipping_rate":sum(s["clipped"] for s in stats)/len(stats),
            "max_step":max(s["step"] for s in stats)}
    return {"N":len(x),"k":k,"c":c,"spread":spread,"symmetric":symmetric,"methods":result}

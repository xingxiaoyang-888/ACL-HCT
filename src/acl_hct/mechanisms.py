"""Bounded paired frozen-neighborhood diagnostics, not trained hierarchy evidence."""
from dataclasses import asdict, dataclass
from itertools import combinations, islice
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import torch
from .aggregation import aggregate, correct_batched
from .geometry import from_spatial, log, origin_like, norm2, dot, tangent, distance


@dataclass(frozen=True)
class Case:
    name: str
    N: int = 16
    k: int = 3
    d: int = 2
    c: float = 1.
    spread: float = .15
    family: str = "asymmetric"
    seed: int = 11
    origin_shift: float = 0.

    def validate(self):
        if type(self.N) is not int or not 3 <= self.N <= 128 or self.N % 2:
            raise ValueError("generated populations require even N in 4..128")
        if type(self.k) is not int or not 1 <= self.k <= self.N:
            raise ValueError("require 1<=k<=N")
        if type(self.d) is not int or not 2 <= self.d <= 64:
            raise ValueError("require spatial dimension 2..64")
        if not math.isfinite(self.c) or self.c <= 0 or not math.isfinite(self.spread) or self.spread <= 0:
            raise ValueError("require finite positive curvature/scale")
        if self.family not in ("symmetric", "asymmetric", "reverse") or not math.isfinite(self.origin_shift):
            raise ValueError("invalid population family or origin shift")


def population(case: Case, device="cpu"):
    """Deterministic genuinely multidimensional construction; reversal is x->-x.

    Symmetric points are paired around tangent origin. Asymmetric points use a
    fixed unequal positive-side scale, then subtract Euclidean mean. A boost
    in the first spatial direction moves the origin without changing distances.
    No points are resampled or clipped to satisfy the geometry domain.
    """
    case.validate()
    g = torch.Generator().manual_seed(case.seed)
    half = torch.randn(case.N//2, case.d, generator=g, dtype=torch.float64)
    half = half / half.norm(dim=-1, keepdim=True)
    half *= torch.linspace(.35, 1., case.N//2, dtype=torch.float64)[:, None]
    v = torch.cat((half, -half))
    if case.family != "symmetric":
        v[:case.N//2] *= torch.linspace(.6, 1.8, case.N//2, dtype=torch.float64)[:, None]
        v -= v.mean(0)
    if case.family == "reverse":
        v = -v
    x = from_spatial((v*case.spread).to(device), case.c)
    if case.origin_shift:
        # Rapidity is sqrt(c) * physical translation distance.
        z = math.sqrt(case.c)*case.origin_shift
        t = math.cosh(z)*x[:, 0]+math.sinh(z)*x[:, 1]
        s = math.sinh(z)*x[:, 0]+math.cosh(z)*x[:, 1]
        x = torch.cat((t[:, None], s[:, None], x[:, 2:]), -1)
        from .geometry import check_point
        check_point(x, case.c)
    return x


def predictions(x, k, c):
    """Population-only oracle diagnostics; never passed to correction methods."""
    n = len(x); mu = x.mean(0); p = aggregate(x, c)
    d = x-mu
    covariance_action = (d*dot(d, p)[:, None]).sum(0)/(n-1)
    second = c*(1/k-1/n)/(-c*dot(mu, mu))*tangent(p, covariance_action, c)
    u = log(p, x, c); u -= u.mean(0)
    moment = (norm2(u)[:, None]*u).mean(0)
    a = n*n*(k-1)*(k-2)/(k*k*(n-1)*(n-2))
    third = -c*(1-a)/6*moment
    return p, {"second_order_oracle": second, "local_third_order_oracle": third}


def index_chunks(n, k, seed, mode, draws, chunk_size):
    if mode == "exact":
        source = combinations(range(n), k)
    else:
        g = torch.Generator().manual_seed(seed)
        source = (torch.randperm(n, generator=g)[:k].tolist() for _ in range(draws))
    while True:
        chunk = list(islice(source, chunk_size))
        if not chunk:
            break
        yield torch.tensor(chunk, dtype=torch.long)


class Moments:
    """CPU FP64 streaming moments at one common tangent base point."""
    def __init__(self, dim):
        self.n = 0; self.sum = torch.zeros(dim, dtype=torch.float64)
        self.cross = torch.zeros(dim, dim, dtype=torch.float64)
        self.scalar = torch.zeros(6, dtype=torch.float64)
        self.scalar2 = torch.zeros(6, dtype=torch.float64)
        self.fallback = self.clipped = 0
        self.max_step = self.max_raw_step = 0.

    def add(self, offsets, radius, projection, paired_mse, paired_radius, stats):
        z = offsets.detach().cpu(); mse = norm2(z)
        scalar = torch.stack((mse, radius.cpu(), projection.cpu(), paired_mse.cpu(), paired_radius.cpu(),
                              stats['step'].cpu()), -1)
        self.n += len(z); self.sum += z.sum(0); self.cross += z.T@z
        self.scalar += scalar.sum(0); self.scalar2 += scalar.square().sum(0)
        self.fallback += int(stats['fallback'].sum()); self.clipped += int(stats['clipped'].sum())
        self.max_step = max(self.max_step, float(stats['step'].max()))
        self.max_raw_step = max(self.max_raw_step, float(stats['raw_step'].max()))

    def finish(self, exact, prediction):
        n = self.n; mean = self.sum/n; bias2 = float(norm2(mean))
        covariance = (self.cross-n*mean[:, None]*mean[None, :]) / max(1, n-1)
        scalar_mean = self.scalar/n
        scalar_variance = ((self.scalar2-n*scalar_mean.square())/max(1, n-1)).clamp_min(0)
        se = torch.zeros_like(scalar_mean) if exact else (scalar_variance/n).sqrt()
        vector_covariance_mean = torch.zeros_like(covariance) if exact else covariance/n
        variance = float(scalar_mean[0])-bias2
        pred = prediction.detach().cpu(); pred_norm = float(norm2(pred).sqrt())
        mean_norm = math.sqrt(bias2)
        cosine = float(dot(mean,pred))/(mean_norm*pred_norm) if min(mean_norm,pred_norm)>1e-14 else None
        return {"samples":n, "mean_offset":mean.tolist(), "mean_offset_norm":mean_norm,
                "bias_interpretation":"exact expectation" if exact else "noisy MC mean norm, upward biased; not exact bias",
                "mean_offset_covariance_mc":vector_covariance_mean.tolist(),
                "bias_squared_noise_corrected":bias2 if exact else bias2-variance/max(1,n-1),
                "variance_population_moment":variance, "mse":float(scalar_mean[0]),
                "mse_mc_se":float(se[0]), "radius_change":float(scalar_mean[1]), "radius_mc_se":float(se[1]),
                "predicted_direction_projection":float(scalar_mean[2]), "projection_mc_se":float(se[2]),
                "prediction_cosine":cosine, "prediction_error_norm":float(norm2(mean-pred).sqrt()),
                "paired_mse_delta_vs_none":float(scalar_mean[3]), "paired_mse_delta_mc_se":float(se[3]),
                "paired_radius_delta_vs_none":float(scalar_mean[4]), "paired_radius_delta_mc_se":float(se[4]),
                "fallback_rate":self.fallback/n, "clipping_rate":self.clipped/n,
                "mean_step":float(scalar_mean[5]), "max_step":self.max_step, "max_raw_step":self.max_raw_step}


def evaluate_points(x, k, *, c=1., seed=11, enumeration_threshold=2048,
                    mc_draws=4096, chunk_size=256):
    """Fixed threshold before observation; independent MC draws, paired methods.

    Unclipped diagnostic uses a 1e100 guard, asserts zero clipping, and records
    domain failure instead of replacing or selectively dropping samples.
    """
    if x.dtype != torch.float64 or x.ndim != 2 or len(x)<3 or not 1<=k<=len(x):
        raise ValueError("E1 requires an FP64 population N>=3 and 1<=k<=N")
    if not 1<=enumeration_threshold<=20000 or not 2<=mc_draws<=100000 or not 1<=chunk_size<=4096:
        raise ValueError("invalid bounded enumeration/MC settings")
    p, pred = predictions(x,k,c); count = math.comb(len(x),k)
    mode = 'exact' if count<=enumeration_threshold else 'mc'
    draws = count if mode=='exact' else mc_draws
    predicted = pred['second_order_oracle']; pred_norm = norm2(predicted).sqrt()
    direction = predicted/pred_norm if pred_norm>1e-14 else torch.zeros_like(p)
    base_radius = distance(origin_like(p,c),p,c)
    names = {'none':('none',.1), 'third_unclipped':('third',1e100),
             'third_protected':('third',.1), 'jackknife_protected':('jackknife',.1)}
    accumulators = {name:Moments(x.shape[-1]) for name in names}; failures = {}
    stream_hash = hashlib.sha256()
    with torch.no_grad():
        for indices in index_chunks(len(x),k,seed,mode,draws,chunk_size):
            stream_hash.update(indices.numpy().astype('<i8').tobytes())
            sample = x[indices.to(x.device)]
            mask = torch.ones(sample.shape[:-1],dtype=torch.bool,device=x.device)
            baseline_mse = baseline_radius = None
            for name,(method,cap) in names.items():
                if name in failures: continue
                try:
                    y, stats = correct_batched(sample,len(x),mask,c,method,cap)
                    if name=='third_unclipped' and stats['clipped'].any():
                        raise ValueError('unclipped diagnostic exceeded finite guard')
                    offsets = log(p,y,c)
                    radius = distance(origin_like(y,c),y,c)-base_radius
                    mse = norm2(offsets)
                    if name=='none': baseline_mse,baseline_radius=mse,radius
                    accumulators[name].add(offsets,radius,dot(offsets,direction),
                                            mse-baseline_mse,radius-baseline_radius,stats)
                except ValueError as error:
                    failures[name] = {'status':'domain_failure', 'error':str(error),
                                      'completed_samples_before_failure':accumulators[name].n,
                                      'failed_chunk_size':len(indices), 'metrics':None}
                    if name=='none':
                        return {'status':'domain_failure', 'error':str(error), 'methods':failures}
    return {'status':'ok' if not failures else 'partial_method_failure', 'N':len(x),'k':k,'c':c,
            'mode':mode,'possible_subsets':count,'draws':draws,'seed':seed,'chunk_size':chunk_size,
            'sample_stream_sha256':stream_hash.hexdigest(), 'full_point':p.tolist(),
            'full_radius':float(base_radius), 'full_reference':{'mean_offset':[0.]*len(p),'mse':0.,'radius_change':0.},
            'direction_defined':bool(pred_norm>1e-14),
            'predictions':{name:value.tolist() for name,value in pred.items()},
            'methods':{name: failures.get(name) or {'status':'ok',**acc.finish(mode=='exact',predicted)}
                       for name,acc in accumulators.items()}}


def run(config, device='cpu'):
    required={'enumeration_threshold','mc_draws','chunk_size','cases'}
    if set(config)!=required or not 1<=len(config['cases'])<=128:
        raise ValueError('configuration requires fixed settings and 1..128 cases')
    cases = [Case(**value) for value in config['cases']]
    for case in cases: case.validate()
    settings = {key:config[key] for key in required-{'cases'}}
    workload=sum(config['mc_draws']
                 if math.comb(case.N,case.k)>config['enumeration_threshold'] else math.comb(case.N,case.k)
                 for case in cases)
    if workload>2000000: raise ValueError('configuration exceeds 2M neighborhood budget')
    rows=[]
    for case in cases:
        try:
            result=evaluate_points(population(case,device),case.k,c=case.c,seed=case.seed,**settings)
        except ValueError as error:
            result={'status':'input_or_domain_failure','error':str(error)}
        rows.append({'case':asdict(case),'result':result})
    return {'schema_version':1,'config':config,'geometry_domain':'FP64; origin scaled radius <=3; no resampling on failure',
            'scope':'E1 frozen local mechanisms; no learned task or semantic hierarchy claim',
            'uncertainty':'MC SEs are across independent subset draws; exact enumeration has zero sampling SE. Mean-vector norm is noisy. Noise-corrected squared bias may be negative. Covariance is in ambient coordinates at full_point. No CI for norm is implied.',
            'missing_controls':['simple scaling comparator','matched zero-mean noise'],
            'torch':torch.__version__,'python':platform.python_version(),'device':device,'cases':rows}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args(); torch.set_num_threads(2)
    config=json.loads(args.config.read_text(encoding='utf-8'))
    result=run(config,args.device)
    result['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    result['source_sha256_normalized_lf']=hashlib.sha256(Path(__file__).read_text().encode()).hexdigest()
    result['working_tree_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(args.output),'cases':len(result['cases']),
                      'statuses':[r['result']['status'] for r in result['cases']]}))


if __name__=='__main__': main()

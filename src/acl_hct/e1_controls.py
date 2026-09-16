"""Exact finite-population E1 controls, separate from deployable corrections.

This module supplies reviewable code. The proposed scientific condition list
requires explicit user approval before execution; engineering unit tests do not
grant that approval. Historical E1 artifacts and correction formulas are unchanged.
"""
import hashlib
import math
import torch
from .aggregation import aggregate, correct_batched
from .geometry import check_point, distance, dot, exp, log, norm2, origin_like
from .mechanisms import Moments, index_chunks, predictions


SCALE_LAMBDAS={'scale_contract':-1,'scale_identity':0,'scale_expand':1}
ORIGINAL_METHODS={'third_unclipped':('third',1e100),
                  'third_protected':('third',.1),'jackknife_protected':('jackknife',.1)}


def radial_scale(points,N,k,coefficient,c=1.):
    """Fixed alpha=1+lambda*(1/k-1/N), lambda=-1/0/+1, coordinate-origin anchor."""
    if type(N) is not int or type(k) is not int or not 1<=k<=N:
        raise ValueError('integer 1<=k<=N required')
    if type(coefficient) is not int or coefficient not in (-1,0,1):
        raise ValueError('fixed lambda -1/0/+1 only; no fitted scaling')
    check_point(points,c)
    if coefficient==0 or k==N:return points
    origin=origin_like(points,c);alpha=1+coefficient*(1/k-1/N)
    return exp(origin,alpha*log(origin,points,c),c)


def lorentz_boost(values,physical_shift,c=1.):
    """Linear Lorentz map, valid for both points and ambient tangent vectors."""
    rapidity=math.sqrt(c)*physical_shift
    t=math.cosh(rapidity)*values[...,0]+math.sinh(rapidity)*values[...,1]
    x=math.sinh(rapidity)*values[...,0]+math.cosh(rapidity)*values[...,1]
    return torch.cat((t[...,None],x[...,None],values[...,2:]),-1)


class ExactMoments(Moments):
    def __init__(self,p,c,radial_direction):
        super().__init__(len(p));self.p=p;self.c=c;self.radial_direction=radial_direction
        self.max_tangent_residual=0.;self.max_roundtrip_ambient=0.;self.max_constraint=0.

    def observe(self,points,baseline,predicted_direction,stats):
        z=log(self.p,points,self.c);baseline_z=log(self.p,baseline,self.c)
        radius=distance(origin_like(points,self.c),points,self.c)
        base_radius=distance(origin_like(self.p,self.c),self.p,self.c)
        baseline_radius=distance(origin_like(baseline,self.c),baseline,self.c)
        self.add(z,radius-base_radius,dot(z,predicted_direction),
                 norm2(z)-norm2(baseline_z),radius-baseline_radius,stats)
        reconstructed=log(self.p,exp(self.p,z,self.c),self.c)
        self.max_tangent_residual=max(self.max_tangent_residual,float(dot(self.p,z).abs().max()))
        self.max_roundtrip_ambient=max(self.max_roundtrip_ambient,float((reconstructed-z).norm(dim=-1).max()))
        self.max_constraint=max(self.max_constraint,float((self.c*dot(points,points)+1).abs().max()))

    def result(self,prediction):
        result=self.finish(True,prediction)
        mean=self.sum/self.n;covariance=self.cross/self.n-mean[:,None]*mean[None,:]
        variance=float(covariance.diagonal()[1:].sum()-covariance[0,0])
        radial=None
        if self.radial_direction is not None:
            covector=self.radial_direction.detach().cpu().clone();covector[0]*=-1
            radial=float(covector@covariance@covector)
        result.update({'covariance_population_ambient':covariance.tolist(),
                       'covariance_denominator':self.n,'lorentz_covariance_trace':variance,
                       'radial_variance':radial,'transverse_variance':variance-radial if radial is not None else None,
                       'radial_direction_defined':radial is not None,
                       'max_tangent_constraint_residual':self.max_tangent_residual,
                       'max_manifold_constraint_residual':self.max_constraint,
                       'max_log_exp_roundtrip_ambient':self.max_roundtrip_ambient,
                       'numerical_floor_scope':'recorded FP64 constraint and roundtrip residuals, not a certified bias error bound'})
        return result


def _stats(points,baseline,c):
    step=distance(points,baseline,c)
    return {'step':step,'raw_step':step,'clipped':torch.zeros_like(step,dtype=torch.bool),
            'fallback':torch.zeros_like(step,dtype=torch.bool)}


@torch.no_grad()
def evaluate_exact_controls(x,k,*,c=1.,chunk_size=256,enumeration_threshold=20000,include_original=False,isometry_reference=None):
    """Two deterministic subset passes; no MC, resampling, fitted lambda or clipping.

    Pass one fixes the exact baseline mean. Pass two reconstructs the same
    residual stream and enumerates both signs around that frozen mean. A failed
    method reports no selectively retained scientific averages.
    """
    if x.dtype!=torch.float64 or x.ndim!=2 or len(x)<3 or type(k) is not int or not 1<=k<=len(x):
        raise ValueError('FP64 population and integer 1<=k<=N required')
    if type(chunk_size) is not int or not 1<=chunk_size<=4096 or type(enumeration_threshold) is not int or not 1<=enumeration_threshold<=20000:
        raise ValueError('invalid bounded exact settings')
    count=math.comb(len(x),k)
    if count>enumeration_threshold:raise ValueError('proposal is exact only; subset count exceeds threshold')
    check_point(x,c);p,pred=predictions(x,k,c)
    prediction=pred['second_order_oracle'];length=norm2(prediction).sqrt()
    direction=prediction/length if length>1e-14 else torch.zeros_like(p)
    radial=-log(p,origin_like(p,c),c);radial_length=norm2(radial).sqrt()
    radial=radial/radial_length if radial_length>1e-14 else None
    names=['none',*SCALE_LAMBDAS,'oracle_centered_pm']+(list(ORIGINAL_METHODS) if include_original else [])
    accumulators={name:ExactMoments(p,c,radial) for name in names};failures={}
    isometry={'status':'not_requested'}
    if isometry_reference is not None:
        unshifted,shift=isometry_reference
        check_point(unshifted,c)
        if unshifted.shape!=x.shape or not include_original:raise ValueError('isometry needs same-size source population and all original methods')
        if not torch.equal(lorentz_boost(unshifted,shift,c),x):raise ValueError('shifted population differs from declared boost')
        original_p=aggregate(unshifted,c)
        isometry={'status':'computed_correspondence','physical_shift':shift,'rapidity':math.sqrt(c)*shift,
                  'full_point_max_abs_difference':float((p-lorentz_boost(original_p,shift,c)).abs().max()),
                  'methods':{name:{'output_max_abs_difference':0.,'offset_max_abs_difference':0.} for name in ('none',*ORIGINAL_METHODS)},
                  'excluded_methods':list(SCALE_LAMBDAS),'exclusion_reason':'coordinate-origin scaling is not isometry equivariant'}
    def check_isometry(name,points,ids,method,cap,mask):
        if isometry_reference is None:return
        other,_=correct_batched(unshifted[ids.to(x.device)],len(x),mask,c,method,cap)
        row=isometry['methods'][name]
        row['output_max_abs_difference']=max(row['output_max_abs_difference'],float((points-lorentz_boost(other,shift,c)).abs().max()))
        row['offset_max_abs_difference']=max(row['offset_max_abs_difference'],float((log(p,points,c)-lorentz_boost(log(original_p,other,c),shift,c)).abs().max()))
    first_hash=hashlib.sha256();second_hash=hashlib.sha256();second_sum=torch.zeros_like(p)
    def indices():return index_chunks(len(x),k,0,'exact',count,chunk_size)
    for ids in indices():
        first_hash.update(ids.numpy().astype('<i8').tobytes())
        sample=x[ids.to(x.device)];mask=torch.ones(sample.shape[:-1],device=x.device,dtype=torch.bool)
        baseline,stats=correct_batched(sample,len(x),mask,c,'none')
        accumulators['none'].observe(baseline,baseline,direction,stats)
        check_isometry('none',baseline,ids,'none',.1,mask)
        if include_original:
            for name,(method,cap) in ORIGINAL_METHODS.items():
                if name in failures:continue
                try:
                    points,stats=correct_batched(sample,len(x),mask,c,method,cap)
                    if name=='third_unclipped' and stats['clipped'].any():raise ValueError('unclipped finite guard exceeded')
                    accumulators[name].observe(points,baseline,direction,stats)
                    check_isometry(name,points,ids,method,cap,mask)
                except ValueError as error:
                    failures[name]={'status':'domain_failure','error':str(error),'metrics':None,
                                    'completed_samples_before_failure':accumulators[name].n}
    mean=(accumulators['none'].sum/count).to(x.device)
    target_cov=(accumulators['none'].cross/count-mean.cpu()[:,None]*mean.cpu()[None,:])
    centered_sum=torch.zeros_like(p);centered_cross=torch.zeros((len(p),len(p)),dtype=torch.float64,device=x.device)
    for ids in indices():
        second_hash.update(ids.numpy().astype('<i8').tobytes())
        sample=x[ids.to(x.device)];mask=torch.ones(sample.shape[:-1],device=x.device,dtype=torch.bool)
        baseline,_=correct_batched(sample,len(x),mask,c,'none')
        z=log(p,baseline,c);second_sum+=z.sum(0);centered=z-mean
        centered_sum+=centered.sum(0);centered_cross+=centered.T@centered
        for name,coefficient in SCALE_LAMBDAS.items():
            if name in failures:continue
            try:
                points=radial_scale(baseline,len(x),k,coefficient,c)
                accumulators[name].observe(points,baseline,direction,_stats(points,baseline,c))
            except ValueError as error:
                failures[name]={'status':'domain_failure','error':str(error),'metrics':None,
                                'completed_samples_before_failure':accumulators[name].n}
        name='oracle_centered_pm'
        if name not in failures:
            try:
                both=torch.stack((centered,-centered),1).reshape(-1,len(p))
                points=exp(p,both,c);paired_baseline=baseline.repeat_interleave(2,dim=0)
                accumulators[name].observe(points,paired_baseline,direction,_stats(points,paired_baseline,c))
            except ValueError as error:
                failures[name]={'status':'domain_failure','error':str(error),'metrics':None,
                                'completed_samples_before_failure':accumulators[name].n}
    if first_hash.hexdigest()!=second_hash.hexdigest():raise ValueError('two-pass subset streams differ')
    methods={name:failures.get(name) or {'status':'ok',**acc.result(prediction)} for name,acc in accumulators.items()}
    covariance_difference=None;radial_difference=None;transverse_difference=None;mse_identity_difference=None
    if methods['oracle_centered_pm']['status']=='ok':
        noise=methods['oracle_centered_pm'];none=methods['none']
        covariance_difference=float((torch.tensor(noise['covariance_population_ambient'],dtype=torch.float64)-target_cov).abs().max())
        radial_difference=noise['radial_variance']-none['radial_variance'] if radial is not None else None
        transverse_difference=noise['transverse_variance']-none['transverse_variance'] if radial is not None else None
        mse_identity_difference=noise['mse']-none['variance_population_moment']
    return {'status':'partial_method_failure' if failures else 'ok','mode':'exact','N':len(x),'k':k,'c':c,
            'possible_subsets':count,'draws':count,'chunk_size':chunk_size,'full_point':p.tolist(),
            'sample_stream_sha256':first_hash.hexdigest(),'second_pass_stream_sha256':second_hash.hexdigest(),
            'two_pass_baseline_sum_max_difference':float((second_sum.cpu()-accumulators['none'].sum).abs().max()),
            'full_reference':{'mse':0.,'mean_offset_norm':0.},
            'predictions':{name:value.tolist() for name,value in pred.items()},'direction_defined':bool(length>1e-14),
            'scale_definition':{'anchor':'coordinate origin','lambda':SCALE_LAMBDAS,'alpha':'1 + lambda * (1/k - 1/N)',
                                'fitted':False,'isometry_equivariant':False,'low_k_fallback':False},
            'noise_identity':{'kind':'exact population oracle, not a deployable method or independent calibrated N1',
                              'subset_pairs':count,'signed_outputs':2*count,'independent_mc_samples':0,
                              'covariance_denominator':count,'centered_mean':(centered_sum/count).tolist(),
                              'centered_covariance_max_difference':float((centered_cross.cpu()/count-target_cov).abs().max()),
                              'recovered_noise_covariance_max_difference':covariance_difference,
                              'radial_variance_difference':radial_difference,'transverse_variance_difference':transverse_difference,
                              'mse_minus_original_variance':mse_identity_difference,
                              'interpretation':'zero tangent mean and MSE=original variance are construction identities, not research discoveries'},
            'isometry':isometry,
            'methods':methods,'local_geometry':{'max_distance_from_full':float(distance(p,x,c).max()),
                       'max_scaled_distance_from_full':float(math.sqrt(c)*distance(p,x,c).max()),
                       'tangent_matrix_rank':int(torch.linalg.matrix_rank(log(p,x,c)))}}

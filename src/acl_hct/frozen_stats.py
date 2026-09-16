"""FP64 common-base streaming moments. No cross-node vector averaging."""
import torch
from .geometry import check_point, dot, log, normalize, norm2


def promote_points(points,c=1.):
    """Uniform Lorentz radial normalization after promotion, for EVERY condition.

    Native point contract is checked first. Ambient displacement is recorded
    because the unprojected cast is not necessarily an FP64 manifold point.
    Model/task outputs are not replaced by this diagnostic-only representation.
    """
    check_point(points,c)
    raw=points.detach().double();before=(c*dot(raw,raw)+1).abs()
    projected=normalize(raw,c)
    delta=(projected-raw).norm(dim=-1)
    return projected,{'rule':'FP64 future-timelike Lorentz normalization, all references and conditions',
                      'native_dtype':str(points.dtype),'constraint_before':before,'constraint_after':(c*dot(projected,projected)+1).abs(),
                      'ambient_projection_displacement':delta}


def outward_directions(base,anchor,c=1.,numerical_floor=None):
    vectors=-log(base,anchor,c);length=norm2(vectors).sqrt()
    floor=torch.zeros_like(length) if numerical_floor is None else numerical_floor
    if floor.shape!=length.shape or not torch.isfinite(floor).all() or (floor<0).any():
        raise ValueError('finite nonnegative direction resolution floor required')
    defined=length>floor
    safe=torch.where(defined,length,torch.ones_like(length))
    return vectors/safe[:,None],defined


class ScalarStream:
    def __init__(self):self.n=0;self.mean=None;self.m2=None
    def add(self,value):
        value=value.detach().double()
        if not torch.isfinite(value).all() or (self.mean is not None and value.shape!=self.mean.shape):
            raise ValueError('finite scalar-stream observations with fixed shape required')
        self.n+=1
        if self.mean is None:self.mean=torch.zeros_like(value);self.m2=torch.zeros_like(value)
        delta=value-self.mean;self.mean+=delta/self.n;self.m2+=delta*(value-self.mean)
    def summary(self):
        return {'mean':self.mean,'mc_se':(self.m2/(self.n*(self.n-1))).clamp_min(0).sqrt() if self.n>1 else None,
                'independent_graph_repetitions':self.n}


class TangentStream:
    def __init__(self,base,c=1.,groups=None,directions=None,direction_defined=None):
        if base.dtype!=torch.float64 or base.ndim!=2:raise ValueError('FP64 [nodes,coordinates] base required')
        check_point(base,c);self.base=base.detach().clone();self.c=c;self.n=0
        self.mean=torch.zeros_like(base);self.m2=torch.zeros_like(base[:,0]);self.halves=[torch.zeros_like(base),torch.zeros_like(base)]
        self.half_counts=[0,0];self.node_mse=ScalarStream();self.node_projection=ScalarStream()
        self.directions=directions;self.defined=direction_defined
        if directions is not None:
            if directions.shape!=base.shape or direction_defined is None or direction_defined.shape!=base.shape[:1]:
                raise ValueError('direction shape/definition mask required')
            if direction_defined.dtype!=torch.bool:raise ValueError('boolean direction mask required')
            if not torch.isfinite(directions).all() or not torch.allclose(dot(base,directions),torch.zeros_like(base[:,0]),atol=1e-10,rtol=0):
                raise ValueError('directions must be finite and tangent at the fixed bases')
        self.groups={key:torch.tensor(ids,dtype=torch.long,device=base.device) for key,ids in (groups or {'V':list(range(len(base)))}).items()}
        for ids in self.groups.values():
            if len(ids)!=len(set(ids.tolist())) or ((ids<0)|(ids>=len(base))).any():raise ValueError('invalid group indices')
        self.group_mse={key:ScalarStream() for key in self.groups};self.group_projection={key:ScalarStream() for key in self.groups}
        self.max_tangent_residual=0.

    def add(self,points):
        if points.shape!=self.base.shape or points.dtype!=torch.float64:raise ValueError('points must match fixed FP64 base')
        self.add_offsets(log(self.base,points,self.c))

    def add_offsets(self,z):
        if z.shape!=self.base.shape or z.dtype!=torch.float64 or not torch.isfinite(z).all():raise ValueError('invalid tangent observations')
        z=z.detach()
        residual=dot(self.base,z).abs()
        if (residual>1e-9).any():raise ValueError('observations not in the common base tangent spaces')
        self.max_tangent_residual=max(self.max_tangent_residual,float(residual.max()))
        half=self.n%2;self.halves[half]+=z;self.half_counts[half]+=1;self.n+=1
        delta=z-self.mean;self.mean+=delta/self.n;self.m2+=dot(delta,z-self.mean)
        squared=dot(z,z);self.node_mse.add(squared)
        projection=dot(z,self.directions) if self.directions is not None else None
        if projection is not None:self.node_projection.add(projection)
        for name,indices in self.groups.items():
            if len(indices):self.group_mse[name].add(squared[indices].mean())
            if projection is not None:
                valid=indices[self.defined[indices]]
                if len(valid):self.group_projection[name].add(projection[valid].mean())

    def finish(self,numerical_floor=None):
        if self.n<2:raise ValueError('at least two independent repetitions required')
        bias2=dot(self.mean,self.mean);variance=self.m2/self.n
        cross=dot(self.halves[0]/self.half_counts[0],self.halves[1]/self.half_counts[1])
        groups={}
        for name,indices in self.groups.items():
            groups[name]={'nodes':len(indices),'mse':self.group_mse[name].summary() if len(indices) else None,
                          'mean_node_variance':variance[indices].mean() if len(indices) else None,
                          'mean_node_squared_mean_norm':bias2[indices].mean() if len(indices) else None,
                          'mean_node_noise_corrected_bias_squared':(bias2-variance/(self.n-1))[indices].mean() if len(indices) else None,
                          'mean_node_independent_half_cross':cross[indices].mean() if len(indices) else None,
                          'projection_nodes':int(self.defined[indices].sum()) if self.defined is not None else 0,
                          'projection':self.group_projection[name].summary() if self.group_projection[name].n else None}
        result={'independent_graph_repetitions':self.n,'mean_offset':self.mean,'mean_offset_norm':bias2.clamp_min(0).sqrt(),
                'variance':variance,'mse':self.node_mse.summary(),'bias_squared_noise_corrected':bias2-variance/(self.n-1),
                'independent_half_cross_inner_product':cross,'half_counts':self.half_counts,'groups':groups,
                'max_tangent_constraint_residual':self.max_tangent_residual,
                'mse_decomposition_max_residual':(self.node_mse.mean-bias2-variance).abs().max(),
                'interpretation':'conditional MC statistics; mean norm is noisy; signed squared-bias estimates are not clipped'}
        if self.directions is not None:
            result['projection']=self.node_projection.summary();result['direction_defined']=self.defined
            # Undefined direction values are masked explicitly, not counted as zero evidence.
            result['projection']['mean']=torch.where(self.defined,result['projection']['mean'],torch.nan)
            result['projection']['mc_se']=torch.where(self.defined,result['projection']['mc_se'],torch.nan)
        if numerical_floor is not None:
            if numerical_floor.shape!=bias2.shape or (numerical_floor<0).any() or not torch.isfinite(numerical_floor).all():
                raise ValueError('finite nonnegative per-node numerical floor required')
            result['empirical_numerical_floor']=numerical_floor
            result['bias_unresolved_at_numerical_floor']=result['mean_offset_norm']<=numerical_floor
        return result

"""Frozen two-layer paired sampling, with independent layer RNG streams."""
import copy
import hashlib
import torch
from .backbone import aggregate_plan, make_plan
from .frozen_stats import promote_points
from .geometry import dot, exp, log, norm2, origin_like


class PlanStreams:
    def __init__(self, seed, namespace):
        if type(seed) is not int or seed<0 or not isinstance(namespace,str) or not namespace:
            raise ValueError('nonnegative seed and nonempty namespace required')
        self.seeds=[int.from_bytes(hashlib.sha256(f'{seed}/{namespace}/layer{i}'.encode()).digest()[:8],'big')%(2**63)
                    for i in (1,2)]
        self.generators=[torch.Generator().manual_seed(s) for s in self.seeds]
        self.metadata={'seed':seed,'namespace':namespace,'layer_seeds':self.seeds,
                       'derivation':'SHA256(seed/namespace/layer1 or layer2), first 8 bytes big endian modulo 2^63'}

    def draw(self,neighbors,fanout):
        return [make_plan(neighbors,fanout,generator) for generator in self.generators]


class FrozenForward:
    """References are evaluators only; never pass them into a deployable corrector.

    Full graph states are retained. No mask zeroes errors outside a sampled row
    or a selected panel. F/S uses full first-layer inputs, whereas S/S uses the
    same sampled first-layer state as S/F and the same layer-2 plan as F/S.
    """
    @torch.no_grad()
    def __init__(self,model,features,neighbors,max_padded_messages=32768):
        if model.training:raise ValueError('frozen diagnostics require model.eval()')
        self.model=model;self.features=features.detach().clone()
        self.neighbors=[list(row) for row in neighbors];self.budget=max_padded_messages
        self.versions=tuple((id(p),p._version) for p in model.parameters())
        self.reference=model.full_reference(self.features,self.neighbors,max_padded_messages)

    def check(self):
        if self.model.training or self.versions!=tuple((id(p),p._version) for p in self.model.parameters()):
            raise ValueError('frozen model changed; rebuild complete references')

    @torch.no_grad()
    def second_layer(self,first,plan):
        self.check()
        inputs=log(origin_like(first,self.model.c),first,self.model.c)[...,1:]
        messages=self.model.layers[1].messages(inputs)
        return aggregate_plan(messages,self.neighbors,plan,self.model.c,max_padded_messages=self.budget)

    @torch.no_grad()
    def paired(self,plans):
        self.check()
        if len(plans)!=2:raise ValueError('two independent layer plans required')
        first,first_stats=self.model.frozen_layer(self.reference,0,self.neighbors,plans[0],max_padded_messages=self.budget)
        fs,fs_stats=self.model.frozen_layer(self.reference,1,self.neighbors,plans[1],max_padded_messages=self.budget)
        # Compute the same layer-2 message field once for both propagation paths.
        inputs=log(origin_like(first,self.model.c),first,self.model.c)[...,1:]
        messages=self.model.layers[1].messages(inputs)
        sf,sf_stats=aggregate_plan(messages,self.neighbors,self.neighbors,self.model.c,max_padded_messages=self.budget)
        ss,ss_stats=aggregate_plan(messages,self.neighbors,plans[1],self.model.c,max_padded_messages=self.budget)
        def row(layer,points,stats):return {'layer':layer,'points':points,'diagnostics':stats}
        return {
            'F/F_L1':row(1,self.reference['layers'][0]['output'],self.reference['diagnostics'][0]),
            'F/F':row(2,self.reference['output'],self.reference['diagnostics'][1]),
            'local_L1':row(1,first,first_stats), 'F/S':row(2,fs,fs_stats),
            'S/F':row(2,sf,sf_stats), 'S/S':row(2,ss,ss_stats),
        }

    @torch.no_grad()
    def numerical_audit(self):
        """Empirical contrasts, not guaranteed roundoff bounds or native task scores."""
        self.check()
        double_model=copy.deepcopy(self.model).double().eval()
        double_reference=double_model.full_reference(self.features.double(),self.neighbors,self.budget)
        rows=[]
        for native_row,double_row in zip(self.reference['layers'],double_reference['layers']):
            base,promotion=promote_points(native_row['output'],self.model.c)
            rerun,rerun_promotion=promote_points(double_row['output'],self.model.c)
            contrast=log(base,rerun,self.model.c)
            self_log=norm2(log(base,base,self.model.c)).sqrt()
            roundtrip=norm2(log(rerun,exp(base,contrast,self.model.c),self.model.c)).sqrt()
            floor=torch.maximum(norm2(contrast).sqrt(),torch.maximum(self_log,roundtrip))
            rows.append({'base':base,'full_fp64_rerun':rerun,'native_promotion':promotion,
                         'fp64_rerun_promotion':rerun_promotion,
                         'native_vs_fp64_full_distance':norm2(contrast).sqrt(),
                         'self_log_norm':self_log,'log_exp_roundtrip_distance':roundtrip,
                         'log_tangent_constraint_residual':dot(base,contrast).abs(),
                         'empirical_numerical_floor':floor})
        return {'layers':rows,'scope':'same weights/features promoted to FP64 and full model rerun; empirical resolution proxy, not a certified error bound'}

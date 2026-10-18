# SOURCE AUDIT EXPORT: private storage names replaced; science functions retained.
raise RuntimeError("Historical source-audit export; not a public executable entry. Use the documented scalar aggregation route.")
"""Passive native operator tape. No Jacobians, Hessians or covariances."""
from contextlib import contextmanager
from types import MethodType
from runtime import torch, cpu_tree, finite, sync
from sampler import plan_hash
from training import selection_outputs
import time


class Patches:
    def __init__(self):
        self.saved = []

    def put(self, obj, name, value):
        self.saved.append((obj,name,name in obj.__dict__,obj.__dict__.get(name)))
        setattr(obj,name,value)

    def restore(self):
        for obj,name,existed,old in reversed(self.saved):
            if existed:
                setattr(obj,name,old)
            else:
                delattr(obj,name)


@contextmanager
def tape(net, model, plans):
    """Exactly one native aggregation per layer; native operation order retained."""
    records = [{} for _ in plans]
    patch = Patches()
    handles = []
    head = {}
    counts = [0] * len(plans)
    try:
        for index,(layer,plan,record) in enumerate(zip(net.encoder.layers,plans,records)):
            if layer.agg.use_att or layer.agg.local_agg:
                raise ValueError("Only frozen no-attention operators supported")
            if model == "HyboNet":
                linear = layer.linear
                original = linear.forward
                def linear_forward(self,x,original=original,record=record):
                    record["linear_input"] = x.detach()
                    result = original(x)
                    record["linear_output"] = result.detach()
                    return result
                patch.put(linear,"forward",MethodType(linear_forward,linear))
                def affine_hook(module,inputs,result,record=record):
                    record["affine_input_after_activation_dropout"] = inputs[0].detach()
                    record["affine_output"] = result.detach()
                handles.append(linear.weight.register_forward_hook(affine_hook))
                def observer(module,z,adj,m,nu,result,record=record,index=index):
                    counts[index] += 1
                    record.update(messages=z.detach(),aggregate_native=m.detach(),normalizer_native=nu.detach(),output_native=result.detach())
                    # The original observer exposes the actual denominator. The
                    # pre-abs inner product below is explicitly a reconstruction.
                    record["q_fp32_recomputed_from_native_aggregate"] = (-module.manifold.inner(None,m,keepdim=True)).detach()
                patch.put(layer.agg,"_stage1_observer",observer)
                original_agg = layer.agg.forward
                def agg_forward(self,x,ignored,original_agg=original_agg,plan=plan):
                    return original_agg(x,plan)
                patch.put(layer.agg,"forward",MethodType(agg_forward,layer.agg))
            else:
                linear = layer.linear
                def linear_forward(self,x,record=record):
                    # Literal native no-attention HypLinear operation order.
                    drop_weight = torch.nn.functional.dropout(self.weight,self.dropout,training=self.training)
                    mv = self.manifold.mobius_matvec(drop_weight,x,self.c)
                    res = self.manifold.proj(mv,self.c)
                    record.update(linear_input=x.detach(),linear_mobius_output=mv.detach(),linear_first_projection=res.detach())
                    if self.use_bias:
                        bias = self.manifold.proj_tan0(self.bias.view(1,-1),self.c)
                        hyp_bias_raw = self.manifold.expmap0(bias,self.c)
                        hyp_bias = self.manifold.proj(hyp_bias_raw,self.c)
                        added = self.manifold.mobius_add(res,hyp_bias,c=self.c)
                        res = self.manifold.proj(added,self.c)
                        record.update(linear_bias_tangent=bias.detach(),linear_bias_exp_raw=hyp_bias_raw.detach(),
                                      linear_bias_projection=hyp_bias.detach(),linear_bias_add_raw=added.detach())
                    record["linear_output"] = res.detach()
                    return res
                patch.put(linear,"forward",MethodType(linear_forward,linear))
                def agg_forward(self,x,ignored,record=record,plan=plan,index=index):
                    z = self.manifold.logmap0(x,c=self.c)
                    m = torch.spmm(plan,z)
                    raw = self.manifold.expmap0(m,c=self.c)
                    result = self.manifold.proj(raw,c=self.c)
                    counts[index] += 1
                    record.update(messages=z.detach(),aggregate_native=m.detach(),agg_exp_raw=raw.detach(),agg_projection=result.detach())
                    return result
                patch.put(layer.agg,"forward",MethodType(agg_forward,layer.agg))
                act = layer.hyp_act
                def act_forward(self,x,record=record):
                    before = self.manifold.logmap0(x,c=self.c_in)
                    activated = self.act(before)
                    tangent = self.manifold.proj_tan0(activated,c=self.c_out)
                    raw = self.manifold.expmap0(tangent,c=self.c_out)
                    result = self.manifold.proj(raw,c=self.c_out)
                    record.update(activation_pre_relu_native=before.detach(),activation_relu_native=activated.detach(),
                                  activation_tangent_projection=tangent.detach(),activation_exp_raw=raw.detach(),output_native=result.detach())
                    return result
                patch.put(act,"forward",MethodType(act_forward,act))
        if model == "HGCN":
            cls = net.decoder.cls
            original = cls.forward
            def head_forward(self,x,original=original):
                head["head_tangent_native"] = x.detach()
                return original(x)
            patch.put(cls,"forward",MethodType(head_forward,cls))
        yield records,head
        if counts != [1]*len(plans):
            raise ValueError("Captured native layer skipped or encoded twice")
    finally:
        for handle in handles:
            handle.remove()
        patch.restore()


def add_arithmetic_fields(record,plan,full):
    """Approximate CPU FP64 HT reconstruction and residual estimates.

    The true represented-message real HT center uses rational N/k and exact
    real summation. These fields do not themselves bound reconstruction error
    or certify native FP32. The isolated extractor records outward intervals.
    """
    native = full.coalesce().cpu()
    p = plan.coalesce().cpu()
    rows,cols = p.indices()
    full_keys = native.indices()[0]*native.shape[0]+native.indices()[1]
    positions = torch.searchsorted(full_keys,rows*native.shape[0]+cols)
    original = native.values()[positions]
    degree = torch.bincount(native.indices()[0][native.indices()[0] != native.indices()[1]],minlength=native.shape[0])
    k = torch.bincount(rows[rows != cols],minlength=native.shape[0])
    factors = torch.ones(len(rows),dtype=torch.float64)
    nonself = rows != cols
    factors[nonself] = degree[rows[nonself]].double()/k[rows[nonself]].double()
    z = record["messages"].double()
    ideal = torch.zeros((native.shape[0],z.shape[1]),dtype=torch.float64)
    stored = torch.zeros_like(ideal)
    ideal.index_add_(0,rows,original.double()[:,None]*factors[:,None]*z[cols])
    stored.index_add_(0,rows,p.values().double()[:,None]*z[cols])
    record.update(aggregate_ht_real_fp64=ideal,aggregate_stored_plan_fp64=stored,
                  coefficient_rounding_residual_fp64=stored-ideal,
                  sparse_summation_residual_fp64=record["aggregate_native"].double()-stored)


@torch.no_grad()
def capture(net,model,features,full,plans,device):
    if net.training:
        raise ValueError("Capture must use eval with fixed weights")
    sync(device)
    start = time.perf_counter()
    with tape(net,model,plans) as (records,head), selection_outputs(net) as actual:
        embedding = net.encode(features,full)
        metric_output = net.decode(embedding,full,list(range(features.shape[0])))
        logits = actual["raw_logits"]
    sync(device)
    native_seconds = time.perf_counter()-start
    cpu_start = time.perf_counter()
    records = cpu_tree(records)
    head = cpu_tree(head)
    logits = logits.detach().cpu()
    metric_output = metric_output.detach().cpu()
    finite(logits,"captured native logits")
    for record,plan in zip(records,plans):
        for name,value in record.items():
            finite(value,"captured "+name)
        add_arithmetic_fields(record,plan,full)
    if model == "HGCN":
        t = records[-1]["aggregate_native"]
        inverse = torch.relu(t)
        linear = net.decoder.cls.linear
        reconstructed = torch.nn.functional.linear(inverse,linear.weight.detach().cpu(),None if linear.bias is None else linear.bias.detach().cpu())
        head["inverse_head_tangent_residual"] = head["head_tangent_native"]-inverse
        head["inverse_tangent_relu_linear_logits"] = reconstructed
        head["native_vs_inverse_expression_residual"] = logits-reconstructed
    return {"layers": records,"head": head,"native_logits": logits,"native_metric_output": metric_output,
            "native_predictions": metric_output.argmax(-1),"plan_hashes": [plan_hash(p) for p in plans],
            "timing": {"encoder_head_with_passive_tape_synchronized_wall_seconds": native_seconds,
                       "cpu_raw_conversion_and_FP64_reconstruction_wall_seconds": time.perf_counter()-cpu_start,
                       "diagnostic_only_not_deployment_cost": True}}

# SOURCE AUDIT EXPORT: private storage names replaced; science functions retained.
raise RuntimeError("Historical source-audit export; not a public executable entry. Use the documented scalar aggregation route.")
"""Fit central minimal interventions; aliases recorded without duplicate encodes."""
from runtime import torch, cpu_tree


@torch.no_grad()
def final_continuation(net,model,m,full):
    last = net.encoder.layers[-1]
    if model == "HyboNet":
        q = -last.agg.manifold.inner(None,m,keepdim=True)
        nu = q.abs().clamp_min(1e-8).sqrt()
        h = m/nu
        state = {"aggregate_native": m,"q_before_abs_clamp_native": q,"normalizer_native": nu,"output_native": h}
    else:
        raw = last.agg.manifold.expmap0(m,c=last.agg.c)
        mapped = last.agg.manifold.proj(raw,c=last.agg.c)
        h = last.hyp_act.forward(mapped)
        state = {"aggregate_native": m,"agg_exp_raw": raw,"agg_projection": mapped,"output_native": h}
    logits = net.decoder.decode(h,full)
    metric = torch.nn.functional.log_softmax(logits,dim=1) if model == "HGCN" else logits
    state.update(native_logits=logits,native_metric_output=metric)
    return cpu_tree(state)


@torch.no_grad()
def fourway(net,model,full_capture,sample_capture,plans,full,device):
    f = full_capture["layers"][-1]
    s = sample_capture["layers"][-1]
    base_sample = {"aggregate_native": s["aggregate_native"],"native_logits": sample_capture["native_logits"],
                   "native_metric_output": sample_capture["native_metric_output"]}
    base_full = {"aggregate_native": f["aggregate_native"],"native_logits": full_capture["native_logits"],
                 "native_metric_output": full_capture["native_metric_output"]}
    if model == "HGCN":
        if not torch.equal(f["messages"],s["messages"]):
            raise ValueError("HGCN one-layer incoming messages unexpectedly differ")
        return {"distinct": {"AhatZhat": base_sample,"AZ": base_full},
                "aliases": {"AhatZ": "AhatZhat","AZhat": "AZ","first_layer_full_restoration": "AZ"},
                "additional_encoder_passes": 0,"distinct_final_conditions": 2}
    sample_z,full_z = s["messages"].to(device),f["messages"].to(device)
    azhat = final_continuation(net,model,torch.spmm(full,sample_z),full)
    ahatz = final_continuation(net,model,torch.spmm(plans[-1],full_z),full)
    return {"distinct": {"AhatZhat": base_sample,"AZhat": azhat,"AhatZ": ahatz,"AZ": base_full},
            "aliases": {},"additional_encoder_passes": 0,"distinct_final_conditions": 4}


def hybo_channels(net,full_capture,sample_capture):
    """All-class scaled decision accounting, not deployable/manifold scores."""
    f = full_capture["layers"][-1]
    s = sample_capture["layers"][-1]
    prototypes = net.decoder.cls.detach().cpu()
    bias = net.decoder.bias.detach().cpu()
    j = torch.ones(prototypes.shape[1]); j[0] = -1
    sf = 2*(s["aggregate_native"]*j)@prototypes.T
    ff = 2*(f["aggregate_native"]*j)@prototypes.T
    sn,fn = s["normalizer_native"],f["normalizer_native"]
    return {"schema": "scaled_head_oracle_accounting_all_classes",
            "sampled": sf+sn*bias,"full_projection_sample_nu": ff+sn*bias,
            "sample_projection_full_nu": sf+fn*bias,"full": ff+fn*bias,
            "head_prototypes": prototypes,"head_bias": bias,
            "common_class_constant_2nu_omitted": True,"not_manifold_interventions": True,
            "not_deployable_native_scores": True}

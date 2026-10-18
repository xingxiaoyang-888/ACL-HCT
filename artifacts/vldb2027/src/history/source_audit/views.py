# SOURCE AUDIT EXPORT: private storage names replaced; science functions retained.
raise RuntimeError("Historical source-audit export; not a public executable entry. Use the documented scalar aggregation route.")
"""Explicit O/S feature information APIs, separate from raw/outcome archives.

The S accessor is row-local: another row's deterministic self message may not
be used as this row's omitted neighbor. This algorithmic restriction is part
of the prospective S definition, not encryption or an access-control system.
"""
from runtime import torch, cpu_tree

STATE_FIELDS = {"aggregate_native","aggregate_ht_real_fp64","aggregate_stored_plan_fp64",
                "coefficient_rounding_residual_fp64","sparse_summation_residual_fp64",
                "normalizer_native","q_fp32_recomputed_from_native_aggregate","output_native",
                "agg_exp_raw","agg_projection","activation_pre_relu_native","activation_relu_native",
                "activation_tangent_projection","activation_exp_raw"}


def public_graph(full,train_ids,validation_ids):
    full = full.coalesce().cpu()
    rows,cols = full.indices()
    return {"native_adjacency": full,"nonself_degree": torch.bincount(rows[rows != cols],minlength=full.shape[0]),
            "training_ids_without_labels": list(train_ids),"validation_ids_without_labels": list(validation_ids)}


def oracle_payload(full_capture,weights,public):
    return {"schema": "O_preplan_full_v1","weights": weights["model_state"],"public_graph": public,
            "full_layers": full_capture["layers"],"head_state": full_capture["head"],
            "native_logits": full_capture["native_logits"],"native_metric_output": full_capture["native_metric_output"]}


def sample_payload(sample_capture,plans,full,weights,public):
    native = full.coalesce().cpu()
    native_keys = native.indices()[0]*native.shape[0]+native.indices()[1]
    layers = []
    for record,plan in zip(sample_capture["layers"],plans):
        plan = plan.coalesce().cpu()
        rows,cols = plan.indices()
        positions = torch.searchsorted(native_keys,rows*native.shape[0]+cols)
        original = native.values()[positions]
        # All selected rows are serialized, but predictor access is by target
        # row only. No all-node incoming/population message matrix is exposed.
        selected = {"row_ptr": torch.searchsorted(rows,torch.arange(native.shape[0]+1)),
                    "selected_ids": cols,"original_native_coefficients": original,"HT_native_coefficients": plan.values(),
                    "selected_incoming_messages": record["messages"][cols],
                    "node_state": {k:v for k,v in record.items() if k in STATE_FIELDS}}
        layers.append(selected)
    return {"schema": "S_one_path_row_local_v1","weights": weights["model_state"],"public_graph": public,
            "sample_layers": layers,"head_state": {k:v for k,v in sample_capture["head"].items() if k == "head_tangent_native"},
            "native_logits": sample_capture["native_logits"],"native_metric_output": sample_capture["native_metric_output"]}


class OracleView:
    def __init__(self,payload,budget):
        if set(payload) != {"schema","weights","public_graph","full_layers","head_state","native_logits","native_metric_output"} or payload["schema"] != "O_preplan_full_v1":
            raise ValueError("Oracle view must contain only pre-plan full-state information")
        self._payload = payload
        self.budget = budget

    def node(self,node):
        p = self._payload
        a = p["public_graph"]["native_adjacency"].coalesce()
        selected = a.indices()[0] == node
        ids,coeff = a.indices()[1][selected],a.values()[selected]
        return {"requested_budget": self.budget,"neighbor_ids": ids,"native_coefficients": coeff,
                "layers": [{"population_incoming_messages": r["messages"][ids],
                            "state": {k:v[node] for k,v in r.items() if k in STATE_FIELDS}} for r in p["full_layers"]],
                "native_logits": p["native_logits"][node],"native_metric_output": p["native_metric_output"][node],
                "head_state": {k:v[node] for k,v in p["head_state"].items()},"weights": p["weights"]}

    def unlabeled_training_scores(self):
        return self._payload["native_logits"][self._payload["public_graph"]["training_ids_without_labels"]]

    def public_graph(self):
        return self._payload["public_graph"]


class SampleView:
    def __init__(self,payload,budget):
        if set(payload) != {"schema","weights","public_graph","sample_layers","head_state","native_logits","native_metric_output"} or payload["schema"] != "S_one_path_row_local_v1":
            raise ValueError("Sample view must contain exactly one observed path and no outcome/full links")
        self._payload = payload
        self.budget = budget

    def node(self,node):
        p = self._payload
        layers = []
        for row in p["sample_layers"]:
            start,end = row["row_ptr"][node:node+2].tolist()
            ids = row["selected_ids"][start:end]
            count = int((ids != node).sum())
            population = int(p["public_graph"]["nonself_degree"][node])
            layers.append({"selected_ids": ids,"original_native_coefficients": row["original_native_coefficients"][start:end],
                           "HT_native_coefficients": row["HT_native_coefficients"][start:end],
                           "selected_incoming_messages": row["selected_incoming_messages"][start:end],
                           "state": {k:v[node] for k,v in row["node_state"].items()},
                           "covariance_status": "exact_zero" if count == population or population <= 1 else ("unavailable_k1" if count == 1 else "sample_plugin"),
                           "omitted_message_envelope_available": count == population})
        return {"requested_budget": self.budget,"layers": layers,"native_logits": p["native_logits"][node],
                "native_metric_output": p["native_metric_output"][node],
                "head_state": {k:v[node] for k,v in p["head_state"].items()},"weights": p["weights"]}

    def unlabeled_training_scores(self):
        return self._payload["native_logits"][self._payload["public_graph"]["training_ids_without_labels"]]

    def public_graph(self):
        return self._payload["public_graph"]

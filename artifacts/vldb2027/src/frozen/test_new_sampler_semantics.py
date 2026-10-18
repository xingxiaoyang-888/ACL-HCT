"""New synthetic-only semantics; no task data or CP IO, no old gradient replay."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from shared_blocks import NativeBlock
from sampling_bridge import PlanningBridge, chunk_blocks, plan_seed
from hyperbolic_adapters import FixedSelfMeanContract, fixed_self_adjacency
from euclidean_adapters import CorrectedGATConv, CorrectedSAGE
from mechanism import four_way


def block(sampler='labor0',fanout=1):
    return NativeBlock(torch.tensor([0,1,2,3]),torch.tensor([0,1]),
        torch.tensor([2]),torch.tensor([0]),torch.tensor([0,1]),torch.tensor([3,2]),
        sampler=sampler,labor_fanout=fanout if sampler=='labor0' else None)


class NewSamplerSemantics(unittest.TestCase):
    def test_labor_bridge_matches_official_topology_and_rejects_weights(self):
        import dgl
        from unittest.mock import patch
        native=dgl.graph((torch.tensor([1,2,3,0,2,0,1,3,2]),torch.tensor([0,0,0,1,1,2,2,2,3])),num_nodes=4)
        graph=SimpleNamespace(kind='sage',n=4,degree=np.array([3,2,3,1]),dgl_graph=lambda:native)
        planner=PlanningBridge(None,nullcontext)
        dst=np.arange(4,dtype=np.int64)
        result=planner.layer(graph,dst,1,sampler='labor',stage='bridge-fixture',seed=11,repeat=0,batch=0,layer=2)
        salt=plan_seed('bridge-fixture',11,0,0,2)
        raw,weights=dgl.sampling.sample_labors(native,torch.from_numpy(dst),1,edge_dir='in',prob=None,importance_sampling=0,
            random_seed=dgl.ndarray.array(np.array([salt],dtype=np.int64)),copy_ndata=False,copy_edata=True)
        self.assertEqual(weights[0].numel(),0)
        s,d=raw.edges(order='eid')
        self.assertEqual(set(zip(result.src_ids[result.edge_src].tolist(),result.dst_ids[result.edge_dst].tolist())),set(zip(s.tolist(),d.tolist())))
        with patch('dgl.sampling.sample_labors',return_value=(raw,[torch.ones(max(1,raw.num_edges()))])):
            with self.assertRaisesRegex(ValueError,'empty importance'):
                planner.layer(graph,dst,1,sampler='labor',stage='bridge-fixture',seed=11,repeat=0,batch=0,layer=2)

    def test_labor_native_fixed_support_and_empty_row(self):
        b=block();x=torch.zeros(4,2)
        a=fixed_self_adjacency(b,x,FixedSelfMeanContract('row_normalized_A_plus_I',True,True)).to_dense()
        # First row p=1/3: neighbor 3/4, own 1/4. Empty second row owns 1/3.
        torch.testing.assert_close(a,torch.tensor([[.25,0.,.75,0.],[0.,1/3,0.,0.]]))
        copied=b.to('cpu');self.assertEqual(copied.labor_fanout,1)
        torch.testing.assert_close(copied.counts,torch.tensor([1,0]))

    def test_gat_labor_is_same_native_sampled_softmax_not_HT_attention(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1907);conv=CorrectedGATConv(2,3,heads=2,residual=True,variant='plain').eval()
            x=torch.randn(4,2)
        labor=block();uniform=block('uniform_wor')
        left=conv(x,labor,correction_enabled=False);right=conv(x,uniform,correction_enabled=False)
        torch.testing.assert_close(left,right,rtol=0,atol=0)
        # Random zero-neighbor LABOR row has exactly its deterministic self edge.
        with torch.no_grad():
            expected=conv.lin(x[1:2]).view(1,2,3).reshape(1,6)+conv.res(x[1:2])+conv.bias
        torch.testing.assert_close(left[1:2],expected)

    def test_chunks_preserve_global_sampled_edges_and_complete_rows(self):
        b=block();pieces=list(chunk_blocks(b,max_rows=1,max_edges=4))
        self.assertEqual(len(pieces),2)
        reconstructed=[]
        for p in pieces:
            reconstructed.extend(zip(p.src_ids[p.edge_src].tolist(),p.dst_ids[p.edge_dst].tolist()))
            self.assertEqual(p.labor_fanout,1)
        self.assertEqual(reconstructed,[(2,0)])
        self.assertEqual(pieces[1].counts.tolist(),[0])

    def test_global_four_way_full_has_identity_with_no_hidden_missing_rows(self):
        neighbors={0:[1,2],1:[0,2],2:[0,1,3],3:[2]}
        def uniform(g,dst,k,salt):
            src=[];rows=[]
            for i,v in enumerate(dst):
                ns=neighbors[int(v)] if k==-1 else neighbors[int(v)][:k]
                src.extend(ns);rows.extend([i]*len(ns))
            rows=np.array(rows,dtype=np.int64)
            counts=np.bincount(rows,minlength=len(dst))
            return np.array(src,dtype=np.int64),rows,np.ones(len(rows)),counts
        graph=SimpleNamespace(kind='sage',n=4,degree=np.array([2,2,3,1]),x=torch.tensor([[1.,2.],[3.,4.],[.5,1.],[2.,-1.]]))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1801);model=CorrectedSAGE(2,4,2,variant='plain').eval()
        model._study_model_key='sage';model._study_variant='plain'
        calls=[]
        r=four_way(model,graph,PlanningBridge(uniform,nullcontext),sampler='uniform',fanout=-1,
            seed=11,repeat=0,device='cpu',max_rows=2,max_edges=8,guard_callback=lambda:calls.append(1))
        for output in r['scores'].values():torch.testing.assert_close(output,r['scores']['both_restored'],rtol=0,atol=0)
        torch.testing.assert_close(r['local_estimator_sq_l2_sampled_inputs'],torch.zeros(4))
        self.assertEqual(len(calls),3)


if __name__=='__main__':
    unittest.main()

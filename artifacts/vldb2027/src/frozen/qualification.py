"""New semantics once inside each seed11 calibration allocation, never replayed.

Five bridge fixtures plus arxiv SAGE correspondence share the first arxiv SAGE
job. Other arxiv models each run their one native correspondence. PubMed has
one new-dimension SAGE correspondence. No gradients or task-label checks.
"""
from contextlib import nullcontext
import io
import time
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from study_io import save, require


def _pubmed_correspondence(device,guard):
    from model_builders import build_pubmed_sage,native_scores
    from sampling_bridge import PlanningBridge
    from mechanism import four_way
    from exact_full import ExactFullGraph
    neighbors=[[1,2],[0,2],[0,1,3],[2]]
    def uniform(graph,targets,k,salt):
        require(k==-1,'new dimension fixture full-only')
        source=[];rows=[]
        for row,node in enumerate(targets):
            source.extend(neighbors[int(node)]);rows.extend([row]*len(neighbors[int(node)]))
        rows=np.asarray(rows,dtype=np.int64)
        return np.asarray(source,dtype=np.int64),rows,np.ones(len(rows)),np.bincount(rows,minlength=len(targets))
    graph=SimpleNamespace(n=4,kind='sage',degree=np.asarray([2,2,3,1],dtype=np.int64),
                          x=torch.linspace(-.1,.1,4*500,dtype=torch.float32).reshape(4,500),
                          indptr=np.asarray([0,2,4,7,8],dtype=np.int64),
                          indices=np.asarray([1,2,0,2,0,1,3,2],dtype=np.int64))
    planner=PlanningBridge(uniform,nullcontext);model=build_pubmed_sage(11,device).eval()
    with torch.no_grad():
        blocks=planner.plan(graph,np.arange(4),[-1]*3,sampler='uniform',stage='new-pubmed-dimensions',seed=11,repeat=0,batch=0)
        expected=native_scores(model,graph.x.to(device),[b.to(device) for b in blocks],correction_enabled=False).cpu()
        actual=four_way(model,graph,planner,sampler='uniform',fanout=-1,seed=11,repeat=0,device=device,
                        max_rows=2,max_edges=8,guard_callback=guard)
        exact=ExactFullGraph(model,graph,device).forward().cpu()
    for value in actual['scores'].values():torch.testing.assert_close(value,expected,atol=3e-4,rtol=2e-4)
    torch.testing.assert_close(exact,expected,atol=3e-4,rtol=2e-4)
    return {'model':'sage','dataset':'PubMed','new_features':500,'new_classes':3,
            'max_abs_error':max(float((value-expected).abs().max()) for value in actual['scores'].values()),
            'native_correspondence_passed':True,'new_dimension_exact_full_passed':True,'gradients_replayed':False}


def qualify(job,config,output,guard):
    require(job['stage']=='calibrate' and job['seed']==11,'new checks only first calibration')
    guard();started=time.perf_counter();cpu=time.process_time();results=[]
    if job['dataset']=='ogbn-arxiv' and job['model']=='sage':
        import test_new_sampler_semantics
        report=io.StringIO()
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(test_new_sampler_semantics.NewSamplerSemantics)
        result=unittest.TextTestRunner(stream=report,verbosity=2).run(suite)
        (output/'new-bridge-checks.txt').write_text(report.getvalue(),encoding='utf-8')
        require(result.testsRun==5 and result.wasSuccessful(),'one or more new bridge fixtures failed; no retry')
        results.append({'new_bridge_tests':5,'passed':True,'task_data_read':False})
    if job['dataset']=='PubMed':results.append(_pubmed_correspondence('cuda:0',guard))
    else:
        from test_native_correspondence import run_native_correspondence
        results.append(run_native_correspondence(job['model'],config,'cuda:0',guard))
    torch.cuda.synchronize(0);guard()
    result={'schema':'vldb_calibration_new_semantics_v1','results':results,
            'wall_seconds':time.perf_counter()-started,'process_cpu_seconds':time.process_time()-cpu,
            'included_in_allocation_totals':True,'task_CP_or_graph_opened_before_checks':False}
    save(output/'new-semantics.json',result)
    return result

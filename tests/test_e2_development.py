import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest
import torch
from acl_hct import e2_development as dev
from acl_hct.development_view import build_view,support_groups
from acl_hct.diagnostic_archive import write_archive,read_archive
from acl_hct.frozen_fixture import synthetic_fixture
from acl_hct.frozen_forward import FrozenForward,PlanStreams
from acl_hct.frozen_stats import promote_points
from acl_hct.frozen_structure import radial_relations
from acl_hct.geometry import from_spatial,log,dot
from acl_hct.protocols import digest
from acl_hct.vector_structure import RadialPanel


def test_vector_structure_matches_independent_weighted_reference():
    nodes=list('abcdef')
    view=build_view(nodes,[[] for _ in nodes],[('a','b')],[('a','c'),('b','c'),('a','d'),('e','f')],['c','d','f'])
    panel={'rows':[{'id':n,'pool_mean_weight':w} for n,w in zip(('c','d','f'),(.2,.3,.5))],
           'relations':[{'child':'c','direct_parents':['a','b'],'positive_distant_ancestors':[]},
                        {'child':'d','direct_parents':['a'],'positive_distant_ancestors':[]},
                        {'child':'f','direct_parents':['e'],'positive_distant_ancestors':[]}]}
    reference=from_spatial(torch.tensor([[0.],[.1],[.2],[0.],[.1],[.3]],dtype=torch.float64))
    groups={'V':list(range(6)),'single':[2],'unknown':[5],'empty':[]}
    floor=torch.full((6,),.01,dtype=torch.float64)
    vector=RadialPanel(view,panel,reference,groups,numerical_floor=floor)
    for points in (reference,from_spatial(torch.tensor([[.3],[.1],[.05],[0.],[.1],[.3]],dtype=torch.float64))):
        expected=radial_relations(view,panel,points,reference,groups,numerical_floor=floor)
        actual=vector.evaluate(points)
        for kind,row in expected['metrics'].items():
            for group,wanted in row['groups'].items():
                got=actual[kind]['groups'][group]
                for key,value in wanted.items():
                    if isinstance(value,dict):
                        for metric,x in value.items():assert got[key][metric]==pytest.approx(x) if x is not None else got[key][metric] is None
                    else:assert got[key]==pytest.approx(value)
    # Root absence leaves every relation unknown without inventing a chance score.
    view.root=None;view.reachable=set()
    assert RadialPanel(view,panel,reference,groups).evaluate(reference)['direct']['groups']['V']['covered_children']==0


def test_lossless_archive_and_tamper_rejection(tmp_path):
    value={'negative':torch.tensor([-.01,0.,.03],dtype=torch.float64),
           'mask':np.array([False,True]),'directions':np.array([np.nan,.2]),
           'mean':torch.tensor(1.,dtype=torch.float64),'ids':list(range(100))}
    descriptor=write_archive(tmp_path,'test',value);loaded=read_archive(tmp_path,descriptor)
    np.testing.assert_array_equal(loaded['negative'],value['negative'].numpy())
    np.testing.assert_array_equal(loaded['mask'],value['mask'])
    assert np.isnan(loaded['directions'][0]) and loaded['mean']==1.
    np.testing.assert_array_equal(loaded['ids'],value['ids'])
    with pytest.raises(ValueError,match='overwrite'):write_archive(tmp_path,'test',value)
    with (tmp_path/descriptor['array_file']).open('ab') as stream:stream.write(b'x')
    with pytest.raises(ValueError,match='hash'):read_archive(tmp_path,descriptor)


def test_multibudget_paired_statistics_structure_and_full_task(tmp_path):
    torch.set_num_threads(2);model,features,view=synthetic_fixture()
    result=dev.diagnose(model,features,view,directory=tmp_path,fanouts=[4,16],repetitions=4,task_repetitions=2,budget=128,candidate_chunk=11)
    assert result['status']=='complete',result['failures']
    assert result['confirmation_evaluated'] is False
    forward=FrozenForward(model,features,view.neighbors,128)
    groups4=support_groups(view.neighbors,4);groups16=support_groups(view.neighbors,16)
    for fanout in (4,16):
        budget=result['budgets'][str(fanout)];archive=read_archive(tmp_path,budget['artifact'])
        assert budget['completed_graph_repetitions']==4 and budget['completed_task_repetitions']==2
        assert set(budget['completed_method_repetitions'])==set(dev.CONDITIONS)
        assert budget['group_counts']['V']==40
        rng=PlanStreams(dev.BASE_SEED,'E2-multibudget-development-v1'+f'/fanout{fanout}')
        raw={name:[] for name in dev.CONDITIONS}
        for repetition in range(4):
            plans=rng.draw(view.neighbors,fanout)
            assert [digest(p) for p in plans]==archive['plan_hashes'][repetition]
            outputs=forward.paired(plans)
            for name,layer in dev.CONDITIONS.items():
                base,_=promote_points(forward.reference['layers'][layer]['output']);points,_=promote_points(outputs[name]['points'])
                raw[name].append(log(base,points))
        for name,observed in raw.items():
            z=torch.stack(observed);stats=archive['geometry'][name]
            np.testing.assert_allclose(stats['mean_offset'],z.mean(0).numpy(),atol=1e-13,rtol=1e-8)
            np.testing.assert_allclose(stats['groups']['V']['mse']['mc_se'],dot(z,z).mean(1).std().item()/2,atol=1e-15,rtol=1e-8)
            columns=archive['geometry_repeat_columns'][name]
            scalar_rows=np.stack([row['geometry'][name] for row in archive['per_repeat_structure_and_task']])
            mse=scalar_rows[:,columns.index('V/mse')]
            np.testing.assert_allclose(mse.mean(),stats['groups']['V']['mse']['mean'],atol=1e-15)
            np.testing.assert_allclose(mse.std(ddof=1)/2,stats['groups']['V']['mse']['mc_se'],atol=1e-15)
        a=archive['geometry_repeat_columns']['S/S'].index('V/mse')
        b=archive['geometry_repeat_columns']['F/S'].index('V/mse')
        paired=np.array([r['geometry']['S/S'][a]-r['geometry']['F/S'][b] for r in archive['per_repeat_structure_and_task']])
        direct=dot(torch.stack(raw['S/S']),torch.stack(raw['S/S'])).mean(1)-dot(torch.stack(raw['F/S']),torch.stack(raw['F/S'])).mean(1)
        np.testing.assert_allclose(paired,direct.numpy(),atol=1e-15,rtol=1e-8)
        outside=archive['groups']['P_minus_A']
        assert np.any(archive['geometry']['S/F']['mse']['mean'][outside]>0)
        assert np.all(archive['geometry']['local_L1']['mse']['mean'][outside]==0)
        for row in archive['per_repeat_structure_and_task'][:2]:
            task=row['task'];assert task['status']=='complete' and task['completed_queries']==7
            assert task['row_ids'].shape==(7,3)
        columns=budget['structure_statistics']['S/S']['columns']
        values=np.stack([row['structure']['S/S'] for row in archive['per_repeat_structure_and_task']])
        np.testing.assert_allclose(budget['structure_statistics']['S/S']['mean'],values.mean(0),atol=1e-14)
        np.testing.assert_allclose(budget['structure_statistics']['S/S']['mc_se'],values.std(0,ddof=1)/2,atol=1e-14)
        assert any('score_change' in key for key in columns)
    assert result['budgets']['4']['group_counts']['A']==len(groups4['A'])
    assert result['budgets']['16']['group_counts']['A']==len(groups16['A'])
    assert result['budgets']['4']['group_hashes']['V']==result['budgets']['16']['group_hashes']['V']
    public=json.dumps(dev.entry.jsonable(result),allow_nan=False)
    assert len(public)<200000


def test_late_failure_keeps_partial_counts_and_stops_later_budget(tmp_path,monkeypatch):
    model,features,view=synthetic_fixture();original=dev.condition_offsets;calls=[]
    def fail(*a,**kw):
        calls.append(1)
        if len(calls)==10:raise ValueError('injected numerical failure after two complete graph repeats')
        return original(*a,**kw)
    monkeypatch.setattr(dev,'condition_offsets',fail)
    result=dev.diagnose(model,features,view,directory=tmp_path,fanouts=[4,16],repetitions=4,task_repetitions=1,budget=128,candidate_chunk=11)
    assert result['status']=='failed' and set(result['budgets'])=={'4'}
    row=result['budgets']['4'];assert row['status']=='partial_not_for_inference' and row['scientifically_usable'] is False
    assert row['completed_graph_repetitions']==2
    assert row['completed_method_repetitions']=={'local_L1':3,'F/S':2,'S/F':2,'S/S':2}
    archived=read_archive(tmp_path,row['artifact'])
    assert archived['scientifically_usable'] is False
    assert archived['per_repeat_structure_and_task'][-1]['status']=='partial_stopped'
    assert set(archived['per_repeat_structure_and_task'][-1]['geometry'])=={'local_L1'}


def test_entry_expiration_does_not_start_budget(tmp_path):
    model,features,view=synthetic_fixture()
    result=dev.diagnose(model,features,view,directory=tmp_path,max_seconds=-1)
    assert result['status']=='incomplete_time_limit' and result['budgets']=={}


def test_partial_task_stops_budget_without_reporting_complete_task_mean(tmp_path,monkeypatch):
    model,features,view=synthetic_fixture();original=dev.entry.filtered_parent_ranks;calls=[]
    def partial(*args,**kwargs):
        calls.append(1);row=original(*args,**kwargs)
        if len(calls)==2:row['status']='incomplete_time_limit'
        return row
    monkeypatch.setattr(dev.entry,'filtered_parent_ranks',partial)
    result=dev.diagnose(model,features,view,directory=tmp_path,fanouts=[4,16],repetitions=2,task_repetitions=1,budget=128,candidate_chunk=11)
    assert result['status']=='incomplete_time_limit' and set(result['budgets'])=={'4'}
    row=result['budgets']['4']
    assert row['completed_task_repetitions']==0 and row['task_statistics'] is None
    assert row['completed_graph_repetitions']==0 and row['scientifically_usable'] is False


def test_fixed_config_and_no_git_static_cli(tmp_path):
    root=Path(__file__).parents[1];config=json.loads((root/'configs/e2_multibudget_development.json').read_text())
    assert dev.validate_config(config)==digest(config)
    changed=copy.deepcopy(config);changed['pilot']['task_repetitions']=16
    with pytest.raises(ValueError):dev.validate_config(changed)
    release=tmp_path/'release';package=release/'src/acl_hct';package.mkdir(parents=True)
    for source in (root/'src/acl_hct').glob('*.py'):
        if source.name!='frozen_noise.py':shutil.copyfile(source,package/source.name)
    (release/'configs').mkdir()
    for name in ('e2_multibudget_development.json','e2_entry_local_pilot.json'):
        shutil.copyfile(root/'configs'/name,release/'configs'/name)
    assert subprocess.run(['git','rev-parse','--show-toplevel'],cwd=release,capture_output=True).returncode!=0
    command=[sys.executable,'-m','acl_hct.e2_development','--config',str(release/'configs/e2_multibudget_development.json')]
    env={**os.environ,'PYTHONPATH':str(release/'src')}
    result=subprocess.run(command,cwd=release,env=env,capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['status']=='static_only'
    failed=subprocess.run(command+['--execute'],cwd=release,env=env,capture_output=True,text=True)
    assert failed.returncode!=0


def test_cuda_evidence_is_bound_to_result_and_current_source(tmp_path):
    names=['e2_development.py','e2_pilot.py','vector_structure.py','diagnostic_archive.py']
    hashes={'acl_hct/'+name:hashlib.sha256((Path(dev.__file__).parent/name).read_text(encoding='utf-8').encode()).hexdigest() for name in names}
    row={'cuda_fixture_passed':True,'status':'complete','source':{'source_commit':'a'*40},'source_sha256_normalized_lf':hashes}
    path=tmp_path/'synthetic-cuda-evidence.json';path.write_text(json.dumps(row))
    approval={'cuda_fixture_artifact_sha256':dev.entry.file_sha256(path)}
    dev.verify_cuda_evidence(path,approval,{'source_commit':'a'*40})
    with pytest.raises(ValueError):dev.verify_cuda_evidence(path,approval,{'source_commit':'b'*40})
    row['source_sha256_normalized_lf']['acl_hct/vector_structure.py']='0'*64
    path.write_text(json.dumps(row));approval['cuda_fixture_artifact_sha256']=dev.entry.file_sha256(path)
    with pytest.raises(ValueError,match='byte mismatch'):dev.verify_cuda_evidence(path,approval,{'source_commit':'a'*40})


@pytest.mark.parametrize('status',['complete','failed','incomplete_time_limit'])
def test_cli_preserves_result_and_maps_noncompletion_to_nonzero(tmp_path,monkeypatch,status):
    root=Path(__file__).parents[1];config_path=root/'configs/e2_multibudget_development.json'
    config=json.loads(config_path.read_text());identity=dev.source_identity()['source_commit'] or '3'*40
    approval={'scope':config['protocol'],'user_authorized':True,'user_message_reference':'engineering fixture only',
              'quality_review_passed':True,'entry_criteria_frozen':True,'config_sha256':digest(config),
              'source_commit':identity,'cuda_fixture_passed':True,'cuda_fixture_artifact_sha256':'1'*64}
    path=tmp_path/'approval.json';path.write_text(json.dumps(approval));output=tmp_path/'output'
    argv=['e2_development','--config',str(config_path),'--execute','--seed','11','--fanout','4',
          '--source-commit',identity,'--approval-record',str(path),'--output-dir',str(output)]
    for name in ('prepared','checkpoint','training-release','baseline-report','cuda-fixture-record'):argv.extend(['--'+name,str(tmp_path/'unused')])
    monkeypatch.setattr(sys,'argv',argv)
    monkeypatch.setattr(dev,'verify_cuda_evidence',lambda *args:None)
    monkeypatch.setattr(dev.entry,'run',lambda *a,**kw:{'status':status,'budgets':{'4':{'status':status}},'source_sha256_normalized_lf':{}})
    if status=='complete':dev.main()
    else:
        with pytest.raises(SystemExit) as stop:dev.main()
        assert stop.value.code==2
    assert json.loads((output/'summary.json').read_text())['status']==status

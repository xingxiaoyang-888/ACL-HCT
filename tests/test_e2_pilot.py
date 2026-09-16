import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import os

import pytest
import torch
import acl_hct.e2_pilot as pilot
from acl_hct.frozen_fixture import synthetic_fixture
from acl_hct.geometry import dot, log
from acl_hct.frozen_stats import promote_points
from acl_hct.frozen_forward import FrozenForward, PlanStreams
from acl_hct.backbone import make_plan
from acl_hct.protocols import digest
from acl_hct.train import run as train_fixture
from test_training_runner import small_config
from test_checkpoint_evaluation import prepared


def config():
    return json.loads((Path(__file__).parents[1]/'configs/e2_entry_local_pilot.json').read_text())


def test_fixed_config_and_explicit_approval_gate():
    cfg=config();assert pilot.validate_config(cfg)==digest(cfg)
    identity={'source_commit':'a'*40}
    record={'user_authorized':True,'user_message_reference':'synthetic fixture, not real authorization',
            'scope':cfg['protocol'],'config_sha256':digest(cfg),'source_commit':'a'*40,
            'quality_review_passed':True,'entry_criteria_frozen':True}
    pilot.verify_approval(record,cfg,identity)
    for key in record:
        changed=dict(record);changed.pop(key)
        with pytest.raises(ValueError):pilot.verify_approval(changed,cfg,identity)
    for section,key in [('pilot','repetitions'),('pilot','fanout'),('numerical_limits','roundtrip')]:
        changed=copy.deepcopy(cfg);changed[section][key]*=2
        with pytest.raises(ValueError):pilot.validate_config(changed)


@pytest.mark.parametrize('limit',[v for v in pilot.LIMITS.values() if v>0])
def test_inclusive_numerical_limit_and_adjacent_float(limit):
    inside=torch.tensor([limit],dtype=torch.float64);checks=[]
    pilot.check_max(checks,'boundary',inside,limit)
    pilot.check_max(checks,'outside',torch.nextafter(inside,torch.tensor([float('inf')],dtype=torch.float64)),limit)
    assert [r['status'] for r in checks]==['passed','failed']


def test_local_pilot_matches_direct_batch_statistics_and_is_frozen(monkeypatch):
    torch.set_num_threads(2)
    model,features,view=synthetic_fixture();before={k:v.clone() for k,v in model.state_dict().items()}
    def forbidden(*args,**kwargs):raise AssertionError('forbidden propagation or checkpoint mutation')
    monkeypatch.setattr(FrozenForward,'paired',forbidden)
    monkeypatch.setattr(torch,'save',forbidden)
    result=pilot.diagnose(model,features,view,repetitions=4,budget=128,candidate_chunk=11)
    assert result['status']=='complete',result['failures']
    assert result['entry_status']=='passed' and result['completed_repetitions']=={'local_L1':4,'F/S':4}
    assert set(result['geometry'])=={'local_L1','F/S'}
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    forward=FrozenForward(model,features,view.neighbors,128);rng=PlanStreams(2026091605,'E2-local-entry-v1')
    for layer,name in enumerate(('local_L1','F/S')):
        base,_=promote_points(forward.reference['layers'][layer]['output']);raw=[]
        for repetition in range(4):
            plan=make_plan(view.neighbors,16,rng.generators[layer]);assert digest(plan)==result['plan_hashes'][name][repetition]
            points,_=model.frozen_layer(forward.reference,layer,view.neighbors,plan,max_padded_messages=128)
            points,_=promote_points(points);raw.append(log(base,points))
        raw=torch.stack(raw);active=result['support_groups']['A'];stats=result['geometry'][name]
        torch.testing.assert_close(stats['mean_offset'],raw[:,active].mean(0),atol=1e-14,rtol=1e-10)
        squared=dot(raw,raw)
        torch.testing.assert_close(stats['groups']['V']['mse']['mean'],squared.mean(),atol=1e-16,rtol=1e-10)
        torch.testing.assert_close(stats['groups']['V']['mse']['mc_se'],squared.mean(1).std()/2,atol=1e-16,rtol=1e-10)
        assert stats['half_counts']==[2,2]
        assert stats['groups']['P_minus_A']['mse']['mean']==0
    json.dumps(pilot.jsonable(result),allow_nan=False)


def test_entry_failure_stops_before_any_sampling(monkeypatch):
    model,features,view=synthetic_fixture()
    original=pilot.numerical_entry
    def failed(forward):
        numerical,quality=original(forward);quality['status']='failed';return numerical,quality
    monkeypatch.setattr(pilot,'numerical_entry',failed)
    monkeypatch.setattr(pilot,'make_plan',lambda *a,**kw:pytest.fail('sampling after failed entry'))
    result=pilot.diagnose(model,features,view,budget=128)
    assert result['entry_status']=='failed' and result['pilot_status']=='not_started'


def test_production_width_near_bound_reference_is_measured_at_selected_weights():
    from acl_hct.backbone import LorentzMeanNetwork
    _,_,view=synthetic_fixture()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(81)
        model=LorentzMeanNetwork(128,128,head_hidden=128).eval()
        features=torch.randn(40,128)
        with torch.no_grad():
            for layer in model.layers:layer.linear.bias.fill_(5.)
    result=pilot.diagnose(model,features,view,repetitions=2,budget=128,candidate_chunk=11)
    assert result['status']=='complete',result['failures']
    assert all(row['near_bound_message_fraction']==1 for row in result['reference_layers'])
    assert all(row['near_bound_threshold']==pytest.approx(1.14) for row in result['reference_layers'])
    assert len(result['geometry']['local_L1']['mean_offset'][0])==129


def test_failed_first_layer_never_executes_second_layer(monkeypatch):
    model,features,view=synthetic_fixture();original=model.frozen_layer;sampled=[]
    def invalid(reference,index,neighbors,plan,**kwargs):
        if plan!=neighbors:
            sampled.append(index)
            raise ValueError('synthetic domain failure')
        return original(reference,index,neighbors,plan,**kwargs)
    monkeypatch.setattr(model,'frozen_layer',invalid)
    result=pilot.diagnose(model,features,view,budget=128)
    assert result['entry_status']=='passed' and result['pilot_status']=='failed'
    assert sampled==[0] and result['completed_repetitions']=={'local_L1':0}


def test_expired_budget_and_partial_valid_stop(monkeypatch):
    model,features,view=synthetic_fixture()
    result=pilot.diagnose(model,features,view,max_seconds=-1)
    assert result['status']=='incomplete_time_limit' and result['stopped_before']=='reference'
    original=pilot.filtered_parent_ranks
    def partial(*a,**kw):
        value=original(*a,**kw);value['status']='incomplete_time_limit';return value
    monkeypatch.setattr(pilot,'filtered_parent_ranks',partial)
    result=pilot.diagnose(model,features,view,budget=128)
    assert result['entry_status']=='incomplete_time_limit' and result['completed_repetitions']=={}


@pytest.fixture
def provenance_inputs(prepared,tmp_path):
    identity={'source_commit':'1'*40,'source_commit_basis':'synthetic_fixture','git':{'available':False}}
    train_fixture(prepared,tmp_path/'train',small_config(max_steps=1,evaluation='full_validation'),identity=identity)
    checkpoint=tmp_path/'train/best.pt';saved=torch.load(checkpoint,weights_only=True,map_location='cpu')
    baseline=tmp_path/'train/run.json';history=json.loads(baseline.read_text())
    cfg=config();cfg['checkpoints']['11']={'step':1,'sha256':pilot.file_sha256(checkpoint),
        'training_commit':'1'*40,'baseline_report_sha256':pilot.file_sha256(baseline),'training_config':saved['config']}
    release=tmp_path/'training-release';(release/'src/acl_hct').mkdir(parents=True)
    for name in saved['source_sha256_normalized_lf']:
        shutil.copyfile(Path(pilot.__file__).parent/Path(name).name,release/'src'/name)
    return cfg,checkpoint,baseline,release,prepared,saved,history


def test_selection_provenance_and_strict_label_loading(provenance_inputs,monkeypatch):
    cfg,checkpoint,baseline,release,prepared,saved,history=provenance_inputs
    original_read=Path.read_text
    def allowlist(path,*a,**kw):
        if path.parent==prepared:
            assert path.name in {'input_manifest.json','observed_graph.json','train_queries.json','evaluator_valid.json','entity_split.json'}
        return original_read(path,*a,**kw)
    monkeypatch.setattr(Path,'read_text',allowlist)
    data=pilot.load_prepared(prepared);pilot.load_development_view(prepared)
    assert pilot.verify_selection(saved,history,cfg['checkpoints']['11'],data)['selected_step']==1
    for key in ('manifest_hash','valid_hash','selection_status','completed_steps','best_full_valid_mrr'):
        changed=copy.deepcopy(saved);changed[key]=None
        with pytest.raises(ValueError):pilot.verify_selection(changed,history,cfg['checkpoints']['11'],data)
    monkeypatch.setattr(pilot,'validate_config',lambda cfg:digest(cfg))
    source=pilot.source_identity()['source_commit'] or '2'*40
    approval={'user_authorized':True,'user_message_reference':'fixture only','scope':cfg['protocol'],
              'config_sha256':digest(cfg),'source_commit':source,'quality_review_passed':True,'entry_criteria_frozen':True}
    raw=checkpoint.read_bytes()
    result=pilot.run(cfg,11,prepared,checkpoint,release,baseline,approval,source)
    assert result['status']=='complete',result['failures']
    assert result['checkpoint_and_weights_unchanged'] and checkpoint.read_bytes()==raw
    assert result['completed_repetitions']=={'local_L1':16,'F/S':16}
    assert 'evaluator_truth.json' not in result['label_files_read']
    assert result['entry_full_valid_reproduction']['status']=='passed'
    json.dumps(pilot.jsonable(result),allow_nan=False)


@pytest.mark.parametrize('failure',['none','micro','macro','query_pairs','count'])
def test_current_valid_reproduction_gate_before_sampling(monkeypatch,failure):
    from collections import defaultdict
    model,features,view=synthetic_fixture();reference=model.full_reference(features,view.neighbors,128)
    index={node:i for i,node in enumerate(view.nodes)}
    valid=[(index[a],index[b]) for a,b in sorted(view.valid_edges)];parents=defaultdict(set)
    for a,b in valid:parents[b].add(a)
    expected=pilot.filtered_parent_ranks(model,reference['output'],valid,parents,11)
    current=copy.deepcopy(expected)
    if failure in ('micro','macro'):
        key='query_micro_mrr' if failure=='micro' else 'child_macro_mrr'
        current[key]+=2*pilot.LIMITS['full_valid_reproduction']
    elif failure=='query_pairs':current['rows'][0]['parent']=999
    elif failure=='count':current['completed_queries']-=1
    monkeypatch.setattr(pilot,'filtered_parent_ranks',lambda *a,**kw:current)
    if failure!='none':monkeypatch.setattr(pilot,'make_plan',lambda *a,**kw:pytest.fail('sampling after failed reproduction'))
    result=pilot.diagnose(model,features,view,repetitions=2,budget=128,expected_ranking=expected)
    assert result['entry_full_valid_reproduction']['status']==('passed' if failure=='none' else 'failed')
    assert result['entry_status']==('passed' if failure=='none' else 'failed')


def test_archive_cli_static_and_missing_approval(tmp_path):
    release=tmp_path/'release';package=release/'src/acl_hct';package.mkdir(parents=True)
    for source in Path(pilot.__file__).parent.glob('*.py'):
        if source.name!='frozen_noise.py':shutil.copyfile(source,package/source.name)
    assert not (release/'.git').exists()
    assert subprocess.run(['git','rev-parse','--show-toplevel'],cwd=release,capture_output=True).returncode!=0
    path=release/'config.json';path.write_text(json.dumps(config()))
    command=[sys.executable,'-m','acl_hct.e2_pilot','--config',str(path)]
    env={**os.environ,'PYTHONPATH':str(release/'src')}
    result=subprocess.run(command,cwd=release,env=env,capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['status']=='static_only'
    rejected=subprocess.run(command+['--execute'],cwd=release,env=env,capture_output=True,text=True)
    assert rejected.returncode!=0 and 'execution requires' in rejected.stderr

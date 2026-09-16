from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from acl_hct.e1_closure import HISTORICAL_COMMIT, case_key, inventory, load_historical, run_cases, verify_approval
from acl_hct.e1_controls import evaluate_exact_controls, lorentz_boost
from acl_hct.mechanisms import Case, evaluate_points, population
from acl_hct.protocols import digest


def fixture_config(rows):
    return {'status':'engineering_fixture_not_scientific_run','protocol':'E1-minimum-closure-v1',
            'scale_lambdas':[-1,0,1],'noise':'exact_centered_plus_minus_oracle',
            'historical_source_commit':HISTORICAL_COMMIT,'enumeration_threshold':20000,'chunk_size':7,'cases':rows}


def test_small_executor_reuses_supplied_metrics_and_never_mutates_them():
    # Six-point synthetic unit fixture, not any of the 102 proposed scientific cases.
    old=Case('old-unit-fixture',N=6,k=3)
    new=Case('new-unit-fixture',N=6,k=3,origin_shift=.4)
    saved={'case':asdict(old),'result':evaluate_points(population(old),3,chunk_size=7)}
    serialized=json.dumps(saved,sort_keys=True)
    config=fixture_config([{'block':'historical_controls_only','case':asdict(old)},
                           {'block':'isometry','case':asdict(new)}])
    result=run_cases(config,{case_key(old):saved},30.)
    assert result['status']=='completed_planned_cases' and len(result['cases'])==2
    assert json.dumps(saved,sort_keys=True)==serialized
    before=result['cases'][0]['result'];moved=result['cases'][1]['result']
    assert before['historical_baseline_comparison']['sample_stream_hash_matches']
    assert before['methods']['third_protected']['mse']==saved['result']['methods']['third_protected']['mse']
    assert moved['isometry']['status']=='computed_correspondence'
    for row in moved['isometry']['methods'].values():
        assert row['output_max_abs_difference']<1e-13 and row['offset_max_abs_difference']<1e-13
    assert run_cases(config,{case_key(old):saved},0.)['cases']==[]


def test_recorded_approval_must_include_scope_source_metrics_and_budget():
    config=fixture_config([{'block':'unit','case':asdict(Case('unit',N=6,k=3))}])
    identity={'source_commit':'4'*40}
    with pytest.raises(ValueError,match='explicit user'):verify_approval({},config,identity,30.)
    synthetic={'status':'explicit_user_approval','scope':'E1-minimum-closure-v1',
               'user_message_reference':'SYNTHETIC UNIT TEST RECORD ONLY, NOT REAL AUTHORIZATION',
               'source_commit':'4'*40,'config_sha256':digest(config),'quality_review_status':'passed',
               'entry_metrics_status':'passed','predeclared_acceptance_criteria':{'unit_fixture':'algebraic invariants only'},
               'max_seconds':30.,'device':'cpu','threads':2}
    verify_approval(synthetic,config,identity,30.)
    for changes in ({'config_sha256':'0'*64},{'source_commit':'5'*40},{'entry_metrics_status':'pending'},
                    {'quality_review_status':'pending'},{'predeclared_acceptance_criteria':{}},{'max_seconds':20.}):
        with pytest.raises(ValueError):verify_approval({**synthetic,**changes},config,identity,30.)


def test_approved_history_integrity_read_only_and_tamper_rejection(tmp_path):
    root=Path(__file__).parents[1];path=root/'reports/e1-s1.json'
    before=path.read_bytes();rows,metadata=load_historical(path)
    assert len(rows)==54 and metadata['source_commit']==HISTORICAL_COMMIT
    config=json.loads((root/'configs/e1_minimum_closure_proposal.json').read_text())
    expected={case_key(Case(**row['case'])) for row in config['cases'] if row['block']=='historical_controls_only'}
    assert expected==set(rows)  # Case identity check only, no scientific recomputation.
    assert path.read_bytes()==before
    altered=tmp_path/'altered.json';altered.write_bytes(before+b' ')
    with pytest.raises(ValueError,match='hash mismatch'):load_historical(altered)


def test_inventory_and_execution_refusal_in_actual_cli():
    root=Path(__file__).parents[1]
    config=root/'configs/e1_minimum_closure_proposal.json'
    env={**os.environ,'PYTHONPATH':str(root/'src')}
    command=[sys.executable,'-m','acl_hct.e1_closure','--config',str(config)]
    read_only=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,check=True)
    plan=json.loads(read_only.stdout)
    assert plan['cases']==102 and plan['exact_subsets']==435561 and not plan['user_approval_inferred']
    rejected=subprocess.run(command+['--execute'],cwd=root,env=env,capture_output=True,text=True)
    assert rejected.returncode!=0 and 'requires --approval-record; no scientific computation started' in rejected.stderr


def test_plot_displays_all_controls_from_a_small_engineering_fixture(tmp_path,monkeypatch):
    pytest.importorskip('matplotlib')
    root=Path(__file__).parents[1]
    spec=importlib.util.spec_from_file_location('closure_plot_fixture',root/'scripts/plot_e1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    case=Case('plot-unit-only',N=6,k=3)
    result=evaluate_exact_controls(population(case),3,include_original=True)
    path=tmp_path/'unit.json';path.write_text(json.dumps({'cases':[{'case':asdict(case),'result':result}]}))
    labels=set()
    def capture(figure,*args,**kwargs):
        for axis in figure.axes:
            labels.update(line.get_label() for line in axis.get_lines())
    monkeypatch.setattr(module.plt.Figure,'savefig',capture)
    module.plot(path,tmp_path/'plots')
    assert set(result['methods']).issubset(labels)

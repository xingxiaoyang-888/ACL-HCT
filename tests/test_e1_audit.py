"""Six-point engineering fixtures and synthetic report mutations only."""
import copy
from dataclasses import asdict
import json
import math
import pytest
from acl_hct.e1_audit import CONTROLS, METHODS, audit_results
from acl_hct.e1_closure import case_key, run_cases
from acl_hct.mechanisms import Case, evaluate_points, population
from test_e1_closure import fixture_config


@pytest.fixture(scope='module')
def record():
    cases=[Case('historical-unit',N=6,k=3),Case('boost-unit',N=6,k=3,origin_shift=.4),
           Case('full-unit',N=6,k=6),Case('symmetric-unit',N=6,k=3,family='symmetric')]
    blocks=['historical_quality_completion','isometry','unit_full','unit_symmetric']
    config=fixture_config([{'block':block,'case':asdict(case)} for case,block in zip(cases,blocks)])
    prior={'case':asdict(cases[0]),'result':evaluate_points(population(cases[0]),3,chunk_size=7)}
    report=run_cases(config,{case_key(cases[0]):prior},30.)
    assert report['quality_audit']['numerical_status']=='passed'
    return config,report


def set_path(result,path,value):
    for key in path[:-1]:result=result[key]
    result[path[-1]]=value


def test_complete_is_numerical_only_read_only_and_never_E1_acceptance(record):
    config,report=record;before=json.dumps(report,sort_keys=True)
    audit=audit_results(config,report['cases'])
    assert audit['completion_passed'] and audit['planned_processed']==4
    assert audit['counts']['method_reported_ok']==32
    assert audit['counts']['method_numerically_usable']==32
    assert audit['all_planned_method_outcomes_recorded'] and audit['all_controls_numerically_available']
    assert audit['numerical_status']=='passed' and audit['overall_quality_status']=='pending_external_provenance_review'
    assert audit['scientific_effect_status']=='pending' and audit['E1_passed'] is False
    assert audit['next_experiment_authorized'] is False and audit['missing_controls']==[]
    assert audit['radial_definition_counts']['explicitly_undefined']>=8
    assert json.dumps(report,sort_keys=True)==before


BOUNDARIES=[
    (0,('two_pass_baseline_sum_max_difference',),1e-10,False),
    *[(0,('methods','scale_contract',field),1e-10,False) for field in
      ('max_manifold_constraint_residual','max_tangent_constraint_residual','max_log_exp_roundtrip_ambient')],
    *[(0,('methods','scale_contract',field),-1e-12,True) for field in ('mse','variance_population_moment')],
    *[(0,('historical_method_comparison','methods',method,field),1e-12,False)
      for method in ('none','third_protected') for field in ('mean_offset_max_abs_difference','mse_difference')],
    (0,('methods','oracle_centered_pm','mean_offset_norm'),1e-10,False),
    *[(0,('noise_identity',field),1e-10,False) for field in
      ('centered_covariance_max_difference','recovered_noise_covariance_max_difference',
       'radial_variance_difference','transverse_variance_difference','mse_minus_original_variance')],
    (1,('isometry','full_point_max_abs_difference'),1e-9,False),
    *[(1,('isometry','methods',method,field),1e-9,False)
      for method in ('none','third_protected') for field in ('output_max_abs_difference','offset_max_abs_difference')],
]


@pytest.mark.parametrize('index,path,boundary,lower',BOUNDARIES)
def test_exact_threshold_and_next_representable_failure(record,index,path,boundary,lower):
    config,report=record;rows=copy.deepcopy(report['cases'])
    set_path(rows[index]['result'],path,boundary)
    if path==('methods','oracle_centered_pm','mean_offset_norm'):
        # The synthetic nonzero norm also requires a defined cosine; otherwise
        # we would trigger an unrelated consistency check instead of this bound.
        rows[index]['result']['methods']['oracle_centered_pm']['prediction_cosine']=0.
    assert audit_results(config,rows)['numerical_status']=='passed'
    outside=math.nextafter(boundary,-math.inf if lower else math.inf)
    set_path(rows[index]['result'],path,outside)
    audit=audit_results(config,rows)
    assert audit['numerical_status']=='failed' and not audit['E1_passed']
    # Audit never clips even a tiny negative variance/MSE back to zero.
    value=rows[index]['result']
    for key in path:value=value[key]
    assert value==outside


def test_identity_requires_measured_full_count_exact_output_and_bitwise_flag(record):
    config,report=record
    for index,name in [(0,'scale_identity'),(2,'scale_contract'),(2,'scale_expand')]:
        for field,value in [('bitwise_equal',False),('max_abs_output_difference',1e-30),('compared_samples',0)]:
            rows=copy.deepcopy(report['cases'])
            rows[index]['result']['scale_output_identity'][name][field]=value
            audit=audit_results(config,rows)
            assert audit['numerical_status']=='failed'
            assert any(row['case']==rows[index]['case']['name'] and row['method']==name for row in audit['missing_controls'])


def test_missing_measurements_are_insufficient_not_a_pass(record):
    config,report=record
    for path in [('methods','scale_contract','max_manifold_constraint_residual'),
                 ('noise_identity','recovered_noise_covariance_max_difference'),
                 ('scale_output_identity','scale_identity','max_abs_output_difference')]:
        rows=copy.deepcopy(report['cases']);parent=rows[0]['result']
        for key in path[:-1]:parent=parent[key]
        del parent[path[-1]]
        audit=audit_results(config,rows)
        assert audit['numerical_status']=='insufficient_evidence'
        assert audit['missing_controls'] and not audit['all_controls_numerically_available']
    for key in ('noise_identity','scale_output_identity'):
        rows=copy.deepcopy(report['cases']);rows[0]['result'][key]=None
        assert audit_results(config,rows)['numerical_status']=='insufficient_evidence'


def test_failures_and_unexecuted_cases_list_every_unavailable_control(record):
    config,report=record;rows=copy.deepcopy(report['cases'])
    rows[0]['result']['status']='partial_method_failure'
    rows[0]['result']['methods']['scale_expand']={'status':'domain_failure','error':'synthetic failure','metrics':None}
    partial=audit_results(config,rows)
    assert partial['completion_passed'] and partial['numerical_status']=='failed'
    assert {'case':'historical-unit','block':'historical_quality_completion','method':'scale_expand','reason':'method_failure'} in partial['missing_controls']
    rows[1]['result']={'status':'input_or_domain_failure','error':'synthetic input failure'}
    invalid=audit_results(config,rows)
    assert invalid['planned_processed']==4 and invalid['counts']['method_case_failure']==len(METHODS)
    assert len([r for r in invalid['missing_controls'] if r['case']=='boost-unit'])==len(CONTROLS)
    short=audit_results(config,rows[:2])
    assert not short['completion_passed'] and short['not_executed']==2
    assert short['counts']['method_not_executed']==2*len(METHODS)
    empty=audit_results(config,[])
    assert empty['planned_processed']==0 and len(empty['missing_controls'])==4*len(CONTROLS)


def test_missing_duplicate_unexpected_and_substituted_cases_never_complete(record):
    config,report=record
    duplicate=audit_results(config,[*report['cases'],report['cases'][0]])
    assert not duplicate['completion_passed'] and duplicate['duplicate_case_records']
    rows=copy.deepcopy(report['cases']);rows[0]['case']['seed']=999
    unexpected=audit_results(config,rows)
    assert not unexpected['completion_passed'] and unexpected['not_executed']==1 and unexpected['unexpected_case_records']
    rows=copy.deepcopy(report['cases']);del rows[0]['result']
    missing=audit_results(config,rows)
    assert missing['missing_case_results']==1 and not missing['completion_passed']


def test_exact_counts_hashes_finiteness_and_NA_marking(record):
    config,report=record
    mutations=[('first_pass_subsets',19),('second_pass_subsets',21),('mode','mc'),('second_pass_stream_sha256','0'*64)]
    for field,value in mutations:
        rows=copy.deepcopy(report['cases']);rows[0]['result'][field]=value
        audit=audit_results(config,rows)
        assert audit['numerical_status']=='failed' and not audit['all_controls_numerically_available']
    rows=copy.deepcopy(report['cases']);rows[0]['result']['methods']['scale_contract']['mse']=float('nan')
    audit=audit_results(config,rows)
    assert audit['numerical_status']=='failed'
    json.dumps(audit,allow_nan=False)  # Nonfinite findings remain serializable, never silently cleared in raw data.
    rows=copy.deepcopy(report['cases']);rows[3]['result']['methods']['scale_contract']['radial_variance']=0.
    assert audit_results(config,rows)['numerical_status']=='failed'  # Undefined is NA, not zero evidence.
    rows=copy.deepcopy(report['cases']);rows[0]['result']['methods']['scale_contract']['radial_direction_defined']=False
    assert audit_results(config,rows)['numerical_status']=='failed'


def test_historical_clip_fallback_same_denominator_and_no_default_pass(record):
    config,report=record
    for path,value in [(('methods','third_protected','historical_samples'),19),
                       (('methods','third_protected','fallback_rate','historical'),.5),
                       (('methods','jackknife_protected','clipping_rate','difference'),1e-30)]:
        rows=copy.deepcopy(report['cases'])
        set_path(rows[0]['result']['historical_method_comparison'],path,value)
        assert audit_results(config,rows)['numerical_status']=='failed'
    rows=copy.deepcopy(report['cases'])
    del rows[0]['result']['historical_method_comparison']['methods']['third_protected']['mse_difference']
    assert audit_results(config,rows)['numerical_status']=='insufficient_evidence'


def test_runner_empty_budget_reports_not_executed_controls(record):
    config,_=record
    result=run_cases(config,{},0.)
    assert result['status']=='incomplete_time_limit' and result['quality_audit']['planned_processed']==0
    assert len(result['missing_controls'])==len(config['cases'])*len(CONTROLS)
    assert result['scientific_effect_status']=='pending' and result['E1_passed'] is False

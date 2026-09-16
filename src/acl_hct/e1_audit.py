"""Read-only coverage and recorded FP64 evidence audit; never executes a case.

Missing measurements are insufficient evidence. Numerical checks cannot grant
scientific acceptance, independently verify a source archive, or approve E2.
"""
from collections import Counter
from dataclasses import asdict
import math
import re
from .e1_controls import ORIGINAL_METHODS, SCALE_LAMBDAS
from .mechanisms import Case
from .protocols import digest

CONTROLS=(*SCALE_LAMBDAS,'oracle_centered_pm')
METHODS=('none',*CONTROLS,*ORIGINAL_METHODS)
LIMITS={'absolute':1e-10,'historical_absolute':1e-12,
        'variance_mse_minimum':-1e-12,'isometry_absolute':1e-9,'scale_identity_absolute':0.}
MISSING=object()


def _mapping(value):
    return value if isinstance(value,dict) else {}


def _status(statuses):
    statuses=list(statuses)
    if 'failed' in statuses:return 'failed'
    if any(status!='passed' for status in statuses):return 'insufficient_evidence'
    return 'passed'


def _nonfinite(value,path=''):
    if isinstance(value,float) and not math.isfinite(value):return [path]
    if isinstance(value,dict):return [p for key,item in value.items() for p in _nonfinite(item,f'{path}.{key}')]
    if isinstance(value,(list,tuple)):return [p for i,item in enumerate(value) for p in _nonfinite(item,f'{path}[{i}]')]
    return []


class Checks:
    def __init__(self):self.rows=[]
    def add(self,name,condition,actual=MISSING):
        missing=actual is MISSING
        if isinstance(actual,float) and not math.isfinite(actual):actual=repr(actual)
        self.rows.append({'check':name,'status':'insufficient_evidence' if missing else 'passed' if condition else 'failed',
                          'observed':None if missing else actual,'missing':missing})
    def equal(self,name,value,expected):
        if value is MISSING or expected is MISSING:
            self.add(name,False);return
        self.add(name,value is not MISSING and type(value)==type(expected) and value==expected,value)
    def number(self,name,value,limit=None,minimum=None):
        ok=type(value) in (int,float) and math.isfinite(value)
        self.add(name,ok and (limit is None or abs(value)<=limit) and (minimum is None or value>=minimum),value)
    def vector(self,name,value,size):
        ok=isinstance(value,list) and len(value)==size and all(type(v) in (int,float) and math.isfinite(v) for v in value)
        self.add(name,ok,MISSING if value is MISSING else {'length':len(value) if isinstance(value,list) else None,'finite':ok})
    @property
    def status(self):return _status(row['status'] for row in self.rows)


def _method_checks(method,name,result,case):
    checks=Checks();expected=math.comb(case.N,case.k)*(2 if name=='oracle_centered_pm' else 1)
    checks.equal('samples',method.get('samples',MISSING),expected)
    checks.add('all_recorded_numeric_values_finite',not _nonfinite(method),_nonfinite(method))
    checks.vector('mean_offset',method.get('mean_offset',MISSING),case.d+1)
    for key in ('mean_offset_norm','radius_change','predicted_direction_projection','prediction_error_norm',
                'bias_squared_noise_corrected','paired_mse_delta_vs_none','paired_mse_delta_mc_se',
                'paired_radius_delta_vs_none','paired_radius_delta_mc_se','mean_step','max_step','max_raw_step'):
        checks.number(key,method.get(key,MISSING))
    for key in ('fallback_rate','clipping_rate'):
        checks.number(key,method.get(key,MISSING),limit=1.,minimum=0.)
    for key,rate in (('fallback_count','fallback_rate'),('clipped_count','clipping_rate')):
        value=method.get(key,MISSING)
        checks.add(key+'_range',type(value) is int and 0<=value<=expected,value)
        fraction=method.get(rate,MISSING)
        checks.add(key+'_rate_consistent',type(value) is int and fraction==value/expected,
                   MISSING if value is MISSING or fraction is MISSING else fraction)
    for key in ('mse','variance_population_moment'):
        checks.number(key,method.get(key,MISSING),minimum=LIMITS['variance_mse_minimum'])
    for key in ('max_manifold_constraint_residual','max_tangent_constraint_residual','max_log_exp_roundtrip_ambient'):
        checks.number(key,method.get(key,MISSING),limit=LIMITS['absolute'])
    for key in ('mse_mc_se','radius_mc_se','projection_mc_se'):
        checks.number(key,method.get(key,MISSING),limit=0.)
    covariance=method.get('covariance_population_ambient',MISSING)
    shape_ok=(isinstance(covariance,list) and len(covariance)==case.d+1
              and all(isinstance(row,list) and len(row)==case.d+1
                      and all(type(v) in (int,float) and math.isfinite(v) for v in row) for row in covariance))
    checks.add('population_covariance_shape',shape_ok,MISSING if covariance is MISSING else {'square_dimension':case.d+1,'matches':shape_ok})
    checks.equal('covariance_denominator',method.get('covariance_denominator',MISSING),expected)
    radial=method.get('radial_direction_defined',MISSING)
    checks.add('radial_definition_recorded',type(radial) is bool,radial)
    checks.equal('radial_definition_matches_common_base',radial,_mapping(_mapping(result.get('methods')).get('none')).get('radial_direction_defined',MISSING))
    for key in ('radial_variance','transverse_variance'):
        if radial is False:checks.equal(key+'_explicit_NA',method.get(key,MISSING),None)
        else:checks.number(key,method.get(key,MISSING),minimum=LIMITS['variance_mse_minimum'])
    cosine=method.get('prediction_cosine',MISSING)
    if cosine is None:
        norm=method.get('mean_offset_norm')
        checks.add('prediction_cosine_NA_has_undefined_direction',result.get('direction_defined') is False or (type(norm) in (int,float) and norm<=1e-14),cosine)
    else:checks.number('prediction_cosine_finite',cosine)
    if name in SCALE_LAMBDAS:
        identity=_mapping(_mapping(result.get('scale_output_identity')).get(name))
        applicable=SCALE_LAMBDAS[name]==0 or case.k==case.N
        checks.equal('identity_applicability',identity.get('applicable',MISSING),applicable)
        if applicable:
            checks.equal('identity_compared_samples',identity.get('compared_samples',MISSING),expected)
            checks.equal('identity_bitwise_equal',identity.get('bitwise_equal',MISSING),True)
            checks.number('identity_output_difference',identity.get('max_abs_output_difference',MISSING),limit=0.)
    if name=='oracle_centered_pm':
        checks.number('oracle_mean_norm',method.get('mean_offset_norm',MISSING),limit=LIMITS['absolute'])
        noise=_mapping(result.get('noise_identity'))
        for key in ('centered_covariance_max_difference','recovered_noise_covariance_max_difference','mse_minus_original_variance'):
            checks.number(key,noise.get(key,MISSING),limit=LIMITS['absolute'])
        for key in ('radial_variance_difference','transverse_variance_difference'):
            if radial is False:checks.equal(key+'_explicit_NA',noise.get(key,MISSING),None)
            else:checks.number(key,noise.get(key,MISSING),limit=LIMITS['absolute'])
        checks.vector('centered_mean',noise.get('centered_mean',MISSING),case.d+1)
        for key,value in (('subset_pairs',expected//2),('signed_outputs',expected),('covariance_denominator',expected//2),('independent_mc_samples',0)):
            checks.equal('noise_'+key,noise.get(key,MISSING),value)
    return checks


def _historical_checks(checks,method,name,result,case):
    comparison=_mapping(_mapping(_mapping(result.get('historical_method_comparison')).get('methods')).get(name))
    checks.equal('historical_method_compared',comparison.get('status',MISSING),'compared')
    count=math.comb(case.N,case.k)
    for key in ('current_samples','historical_samples'):
        checks.equal(key,comparison.get(key,MISSING),count)
    for key in ('mean_offset_max_abs_difference','mse_difference'):
        checks.number('historical_'+key,comparison.get(key,MISSING),limit=LIMITS['historical_absolute'])
    for rate,count_key in (('fallback_rate','fallback_count'),('clipping_rate','clipped_count')):
        rates=_mapping(comparison.get(rate));counts=_mapping(comparison.get(count_key))
        current=rates.get('current',MISSING);previous=rates.get('historical',MISSING)
        checks.number('historical_'+rate,previous,limit=1.,minimum=0.)
        checks.add('historical_'+rate+'_matches',current==previous,MISSING if current is MISSING or previous is MISSING else current==previous)
        checks.add('comparison_'+rate+'_uses_current_measurement',current==method.get(rate),current)
        checks.number('historical_'+rate+'_difference',rates.get('difference',MISSING),limit=0.)
        checks.equal('comparison_'+count_key+'_uses_current_measurement',counts.get('current',MISSING),method.get(count_key,MISSING))
        basis=counts.get('comparison_basis',MISSING)
        checks.add('historical_'+count_key+'_basis',basis in ('count','ratio_with_identical_denominator'),basis)
        if basis=='count':checks.equal('historical_'+count_key+'_matches',counts.get('historical',MISSING),method.get(count_key,MISSING))
        elif basis=='ratio_with_identical_denominator':
            checks.equal('historical_'+count_key+'_not_recorded',counts.get('historical',MISSING),None)


def _shared_checks(result,case,block):
    checks=Checks();count=math.comb(case.N,case.k)
    checks.equal('mode',result.get('mode',MISSING),'exact')
    for key in ('N','k','c'):checks.equal('case_'+key,result.get(key,MISSING),getattr(case,key))
    checks.vector('full_point',result.get('full_point',MISSING),case.d+1)
    defined=result.get('direction_defined',MISSING)
    checks.add('prediction_direction_definition_recorded',type(defined) is bool,defined)
    for key in ('possible_subsets','draws','first_pass_subsets','second_pass_subsets'):
        checks.equal(key,result.get(key,MISSING),count)
    left=result.get('sample_stream_sha256',MISSING);right=result.get('second_pass_stream_sha256',MISSING)
    for key,value in (('first_stream_hash_format',left),('second_stream_hash_format',right)):
        checks.add(key,isinstance(value,str) and bool(re.fullmatch('[0-9a-f]{64}',value)),value)
    checks.add('two_pass_hash_equality',left==right,MISSING if left is MISSING or right is MISSING else left==right)
    checks.number('two_pass_sum_difference',result.get('two_pass_baseline_sum_max_difference',MISSING),limit=LIMITS['absolute'])
    if block=='historical_quality_completion':
        comparison=_mapping(result.get('historical_method_comparison'))
        checks.equal('historical_stream_matches',comparison.get('sample_stream_hash_matches',MISSING),True)
    if block=='isometry':
        isometry=_mapping(result.get('isometry'))
        checks.equal('isometry_recorded',isometry.get('status',MISSING),'computed_correspondence')
        checks.number('boost_full_point_difference',isometry.get('full_point_max_abs_difference',MISSING),limit=LIMITS['isometry_absolute'])
    return checks


def audit_results(config,rows):
    """Audit recorded results only. Returns a new object; inputs remain untouched."""
    if not config['cases']:raise ValueError('nonempty planned case list required for audit')
    expected={digest(asdict(Case(**item['case']))):item for item in config['cases']}
    planned_duplicates=len(config['cases'])-len(expected)
    observed={};unexpected=[];duplicates=[];malformed=[];duplicate_keys=set()
    for position,row in enumerate(rows):
        try:key=digest(asdict(Case(**row['case'])))
        except (KeyError,TypeError,ValueError):malformed.append(position);continue
        if key not in expected:unexpected.append({'position':position,'case':row['case'].get('name')});continue
        if key in observed:
            duplicates.append({'position':position,'case':row['case'].get('name')});duplicate_keys.add(key);continue
        observed[key]=row
    cases=[];missing_controls=[];unavailable_methods=[]
    totals=Counter({key:0 for key in ('planned_processed','not_executed','missing_result','method_numerically_usable',
        'method_reported_ok','method_not_executed','method_missing_result','method_case_failure','method_method_failure',
        'method_numeric_passed','method_numeric_failed','method_numeric_insufficient_evidence','method_numeric_not_evaluated')})
    radial_undefined=Counter({'explicitly_undefined':0,'defined':0,'definition_not_recorded':0})
    for key,item in expected.items():
        case=Case(**item['case']);row=observed.get(key);shared=Checks();methods={}
        if row is None:
            processing='not_executed';result={};shared.add('case_result_present',False)
        else:
            result=row.get('result',{});result=result if isinstance(result,dict) else {}
            status=result.get('status')
            processing='planned_processed' if status in ('ok','partial_method_failure','input_or_domain_failure') else 'missing_result'
            shared.equal('block_identity',row.get('block',MISSING),item['block'])
            shared.add('unique_case_record',key not in duplicate_keys,key not in duplicate_keys)
            if status in ('ok','partial_method_failure'):
                shared.rows.extend(_shared_checks(result,case,item['block']).rows)
            elif status=='input_or_domain_failure':shared.add('case_input_valid',False,result.get('error','input/domain failure'))
            else:shared.add('recognized_case_result',False)
        totals[processing]+=1
        records=result.get('methods',{});records=records if isinstance(records,dict) else {}
        extras=sorted(set(records)-set(METHODS))
        shared.add('no_unplanned_methods',not extras,extras)
        for name in METHODS:
            record=records.get(name)
            if row is None:execution='not_executed'
            elif result.get('status')=='input_or_domain_failure':execution='case_failure'
            elif not isinstance(record,dict):execution='missing_result'
            elif record.get('status')!='ok':execution='method_failure'
            else:execution='reported_ok'
            check=_method_checks(record,name,result,case) if execution=='reported_ok' else Checks()
            if execution=='reported_ok' and item['block']=='historical_quality_completion' and name in ('none',*ORIGINAL_METHODS):
                _historical_checks(check,record,name,result,case)
            if execution=='reported_ok' and item['block']=='isometry' and name in ('none',*ORIGINAL_METHODS):
                iso=_mapping(_mapping(_mapping(result.get('isometry')).get('methods')).get(name))
                for field in ('output_max_abs_difference','offset_max_abs_difference'):
                    check.number('boost_'+field,iso.get(field,MISSING),limit=LIMITS['isometry_absolute'])
            numerical=check.status if execution=='reported_ok' else 'not_evaluated'
            usable=execution=='reported_ok' and numerical=='passed' and shared.status=='passed'
            methods[name]={'execution_status':execution,'numerical_status':numerical,'numerically_usable':usable,
                           'historical_reused':isinstance(record,dict) and record.get('metric_source')=='unchanged verified historical artifact',
                           'checks':check.rows}
            totals['method_'+execution]+=1;totals['method_numeric_'+numerical]+=1
            if execution=='reported_ok':
                flag=record.get('radial_direction_defined',MISSING)
                radial_undefined['explicitly_undefined' if flag is False else 'defined' if flag is True else 'definition_not_recorded']+=1
            if not usable:
                reason=execution if execution!='reported_ok' else 'case_checks_'+shared.status if shared.status!='passed' else 'numerical_'+numerical
                unavailable={'case':case.name,'block':item['block'],'method':name,'reason':reason}
                unavailable_methods.append(unavailable)
                if name in CONTROLS:missing_controls.append(dict(unavailable))
        # A control cannot be usable when its common baseline failed its audit.
        if not methods['none']['numerically_usable']:
            for name in CONTROLS:
                if methods[name]['numerically_usable']:
                    methods[name]['numerically_usable']=False
                    unavailable={'case':case.name,'block':item['block'],'method':name,'reason':'baseline_not_numerically_usable'}
                    unavailable_methods.append(unavailable);missing_controls.append(dict(unavailable))
        totals['method_numerically_usable']+=sum(m['numerically_usable'] for m in methods.values())
        numerical=_status([shared.status,*[m['numerical_status'] if m['execution_status']=='reported_ok' else 'failed' if m['execution_status'] in ('case_failure','method_failure') else 'insufficient_evidence' for m in methods.values()]])
        cases.append({'case':case.name,'block':item['block'],'processing_status':processing,
                      'shared_numerical_status':shared.status,'numerical_status':numerical,'shared_checks':shared.rows,'methods':methods})
    complete=totals['planned_processed']==len(expected) and not unexpected and not duplicates and not malformed and not planned_duplicates
    coverage_status='all_planned_cases_processed' if complete else 'incomplete_or_invalid_case_coverage'
    numerical=_status(case['numerical_status'] for case in cases)
    return {'criteria':'E1_CLOSURE_ACCEPTANCE_PROPOSAL-v1','limits':dict(LIMITS),
            'scope':'read-only coverage and recorded numerical evidence; source/approval/artifact hashes require independent provenance review',
            'planned_cases':len(expected),'planned_processed':totals['planned_processed'],
            'not_executed':totals['not_executed'],'missing_case_results':totals['missing_result'],
            'completion_status':coverage_status,'completion_passed':complete,'counts':dict(totals),
            'duplicate_planned_cases':planned_duplicates,
            'all_planned_method_outcomes_recorded':complete and not totals['method_not_executed'] and not totals['method_missing_result'],
            'unexpected_case_records':unexpected,'duplicate_case_records':duplicates,'malformed_case_positions':malformed,
            'cases':cases,'missing_controls':missing_controls,'unavailable_methods':unavailable_methods,
            'all_controls_numerically_available':complete and not missing_controls,
            'radial_definition_counts':dict(radial_undefined),'numerical_status':numerical,
            'overall_quality_status':'pending_external_provenance_review' if complete and numerical=='passed' else 'not_passed',
            'provenance_review':'pending_independent_source_approval_and_historical_hash_verification',
            'scientific_effect_status':'pending','E1_passed':False,'next_experiment_authorized':False}

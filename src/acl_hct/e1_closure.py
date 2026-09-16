"""Inventory by default; E1 scientific execution requires a recorded user approval.

The approval file is an audit record, not cryptographic proof or a substitute
for verifying the user's message. Code readiness never authorizes a run.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from .e1_controls import evaluate_exact_controls
from .mechanisms import Case, dependency_hashes, population, source_identity
from .protocols import digest


HISTORICAL_COMMIT='6cd0caaa6a970b6f472ff45e58a1b3e65ea57e06'
HISTORICAL_RESULT_SHA256='9c3b26297c1667c16fa76ac5dcf306c5a055a2f2fbdbb8d24355e86ad1fd15e1'


def inventory(config):
    if (config.get('protocol')!='E1-minimum-closure-v1' or config.get('scale_lambdas')!=[-1,0,1]
            or config.get('noise')!='exact_centered_plus_minus_oracle'
            or config.get('historical_source_commit')!=HISTORICAL_COMMIT):
        raise ValueError('unexpected closure protocol/control definition')
    if config['enumeration_threshold']!=20000 or not 1<=config['chunk_size']<=4096:
        raise ValueError('registered exact threshold and bounded chunk required')
    if not 1<=len(config['cases'])<=128:raise ValueError('bounded explicit case list required')
    blocks={};seen=set()
    for row in config['cases']:
        case=Case(**row['case']);case.validate()
        key=case_key(case)
        if key in seen:raise ValueError('duplicate scientific condition')
        seen.add(key);count=math.comb(case.N,case.k)
        if count>config['enumeration_threshold']:raise ValueError('all proposed conditions must be exact')
        block=blocks.setdefault(row['block'],{'cases':0,'exact_subsets':0})
        block['cases']+=1;block['exact_subsets']+=count
    total=sum(row['exact_subsets'] for row in blocks.values())
    if total>2000000:raise ValueError('closure exceeds bounded two-million-subset inventory')
    return {'scope':'static arithmetic only; no populations evaluated','config_sha256':digest(config),
            'blocks':blocks,'cases':len(config['cases']),'exact_subsets':total,
            'baseline_subset_passes':2,'baseline_aggregation_evaluations':2*total,
            'oracle_signed_outputs':2*total,'user_approval_inferred':False}


def case_key(case):
    return digest({key:value for key,value in asdict(case).items() if key!='name'})


def verify_approval(record,config,identity,max_seconds):
    if (record.get('status')!='explicit_user_approval' or record.get('scope')!='E1-minimum-closure-v1'
            or not isinstance(record.get('user_message_reference'),str) or not record['user_message_reference'].strip()):
        raise ValueError('recorded explicit user approval required; supervisor readiness is insufficient')
    if record.get('config_sha256')!=digest(config) or record.get('source_commit')!=identity['source_commit']:
        raise ValueError('approval must identify this exact config and execution release')
    if (record.get('quality_review_status')!='passed' or record.get('entry_metrics_status')!='passed'
            or not isinstance(record.get('predeclared_acceptance_criteria'),dict) or not record['predeclared_acceptance_criteria']):
        raise ValueError('approval record requires completed quality/entry-metric review and predeclared criteria')
    if (type(record.get('max_seconds')) not in (int,float) or not math.isfinite(record['max_seconds'])
            or not 0<max_seconds<=record['max_seconds']<=1800 or record.get('device')!='cpu'
            or record.get('threads')!=2):
        raise ValueError('approved CPU two-thread wall-time boundary required')


def load_historical(path):
    raw=Path(path).read_bytes();normalized=raw.replace(b'\r\n',b'\n')
    if hashlib.sha256(normalized).hexdigest()!=HISTORICAL_RESULT_SHA256:
        raise ValueError('historical E1 result hash mismatch')
    result=json.loads(normalized)
    if result['source_commit']!=HISTORICAL_COMMIT:raise ValueError('historical source mismatch')
    return {case_key(Case(**row['case'])):row for row in result['cases']}, {
        'source_commit':HISTORICAL_COMMIT,'raw_sha256':hashlib.sha256(raw).hexdigest(),
        'normalized_lf_sha256':HISTORICAL_RESULT_SHA256,'reuse':'original correction metrics for old 54; none recomputed only for new controls'}


def finite_scale_slopes(rows):
    """Descriptive adjacent log slopes; report unresolved values rather than fit."""
    groups={}
    for row in rows:
        case=row['case'];result=row['result']
        if (case['seed']!=23 or case['family'] not in ('asymmetric','reverse')
                or case['spread'] not in (.025,.05,.1,.15) or result.get('status') not in ('ok','partial_method_failure')):continue
        for name,method in result['methods'].items():
            if method['status']!='ok':continue
            groups.setdefault((case['family'],case['k'],name),[]).append((case['spread'],method))
    out=[]
    for (family,k,method),values in groups.items():
        values.sort(key=lambda pair:pair[0])
        for (s0,a),(s1,b) in zip(values,values[1:]):
            e0,e1=a['mean_offset_norm'],b['mean_offset_norm']
            floor0=a.get('max_log_exp_roundtrip_ambient');floor1=b.get('max_log_exp_roundtrip_ambient')
            resolved=(floor0 is not None and floor1 is not None and e0>floor0 and e1>floor1 and e0>0 and e1>0)
            out.append({'family':family,'k':k,'method':method,'scales':[s0,s1],'absolute_errors':[e0,e1],
                        'recorded_roundtrip_proxies':[floor0,floor1],
                        'finite_scale_log_slope':math.log(e1/e0)/math.log(s1/s0) if resolved else None,
                        'scope':'descriptive finite scales; unresolved at recorded numerical proxy yields NA; not an asymptotic proof'})
    return out


@torch.no_grad()
def run_cases(config,historical,max_seconds):
    """Internal executor. CLI verifies approval before calling; never auto schedules."""
    plan=inventory(config);started=time.perf_counter();rows=[]
    for item in config['cases']:
        if time.perf_counter()-started>=max_seconds:break
        case=Case(**item['case']);old=item['block']=='historical_controls_only'
        if old and case_key(case) not in historical:raise ValueError('historical control case missing from verified artifact')
        begin=time.perf_counter()
        try:
            points=population(case)
            unshifted=(population(replace(case,origin_shift=0.)),case.origin_shift) if item['block']=='isometry' else None
            result=evaluate_exact_controls(points,case.k,c=case.c,chunk_size=config['chunk_size'],
                     enumeration_threshold=config['enumeration_threshold'],include_original=not old,isometry_reference=unshifted)
            if old:
                previous=historical[case_key(case)]['result'];current=result['methods']['none'];prior=previous['methods']['none']
                result['historical_baseline_comparison']={
                    'mean_offset_max_abs_difference':max(abs(a-b) for a,b in zip(current['mean_offset'],prior['mean_offset'])),
                    'mse_difference':current['mse']-prior['mse'],
                    'sample_stream_hash_matches':result['sample_stream_sha256']==previous['sample_stream_sha256']}
                for name in ('third_unclipped','third_protected','jackknife_protected'):
                    result['methods'][name]={**previous['methods'][name],'metric_source':'unchanged verified historical artifact',
                                             'source_commit':HISTORICAL_COMMIT}
        except ValueError as error:
            result={'status':'input_or_domain_failure','error':str(error),'policy':'no redraw, no replacement configuration'}
        rows.append({'case':asdict(case),'block':item['block'],'seconds':time.perf_counter()-begin,'result':result})
    return {'schema_version':2,'scope':'E1 exact controlled geometry; oracle noise identities are not research findings',
            'status':'completed_planned_cases' if len(rows)==len(config['cases']) else 'incomplete_time_limit',
            'quality_and_scientific_acceptance':'requires supervisor review and user agreement before any next key experiment',
            'config':config,'inventory':plan,'cases':rows,'elapsed_seconds':time.perf_counter()-started,
            'deadline_policy':'checked between cases; a case may overrun; external allocation supplies hard cap',
            'finite_scale_slopes':finite_scale_slopes(rows),'missing_controls':[],
            'remaining_scope_limits':['exact oracle is not calibrated real-graph N1','no semantic hierarchy or task-effect conclusion']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--execute',action='store_true',help='Requires reviewed, user-approved exact release/config record')
    parser.add_argument('--approval-record',type=Path)
    parser.add_argument('--historical-result',type=Path)
    parser.add_argument('--source-commit')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-seconds',type=float,default=1700.)
    args=parser.parse_args();config=json.loads(args.config.read_text(encoding='utf-8'))
    plan=inventory(config)
    if not args.execute:
        print(json.dumps(plan,indent=2));return
    if args.approval_record is None:raise ValueError('--execute requires --approval-record; no scientific computation started')
    identity=source_identity(args.source_commit)
    if not identity['source_commit']:raise ValueError('execution release commit required')
    approval=json.loads(args.approval_record.read_text(encoding='utf-8'))
    verify_approval(approval,config,identity,args.max_seconds)
    if args.output is None or args.output.exists() or args.output.suffix.lower()!='.json':raise ValueError('new JSON output required')
    if args.historical_result is None:raise ValueError('verified historical result required')
    historical,provenance=load_historical(args.historical_result)
    torch.set_num_threads(2)
    result=run_cases(config,historical,args.max_seconds)
    result.update({'source':identity,'historical_result':provenance,'approval_record':approval,
                   'config_sha256':digest(config),'source_sha256_normalized_lf':dependency_hashes(),
                   'torch':str(torch.__version__),'device':'cpu','threads':2})
    result['source_sha256_normalized_lf']['acl_hct/e1_closure.py']=hashlib.sha256(Path(__file__).read_text(encoding='utf-8').encode()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'cases':len(result['cases']),'output':str(args.output)}))


if __name__=='__main__':main()

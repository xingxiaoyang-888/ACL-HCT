"""Root-released scalar-only final export. No science imports, payloads or forward calls.

Reads only exact manifest-bound JSON/JSONL artifacts. Four repeats are averaged
within each training seed; only the three training seeds enter sample std.
Missing artifacts/values remain pending. No re-selection or statistical fitting.
"""
import argparse
import csv
import hashlib
import json
import gzip
import math
from pathlib import Path
import statistics
import time

ROUND='vldb-sampling-mechanisms-v1'
SOURCE_SHA='ece3665a84b76ac7801639561f0d07ab98af02d94c55f28aa8bc1dca82bb5e75'
SEEDS=(11,23,37)
ROLES={'results':'results.json','science_completion':'completion.json','worker_completion':'completion.json',
       'hardware':'hardware-identity.json','budget':'budget-lock.json','budget_hardware':'hardware-cost-binding.json',
       'deployment_costs':'deployment-costs.jsonl.gz'}
GROUP_KEYS=('dataset','model','section','sampler','tier','branch','q','metric','unit','within_seed_rule')


def require(ok,message):
    if not ok:raise ValueError(message)


def number(value):
    return type(value) in (int,float) and math.isfinite(value)


def get(value,*keys):
    for key in keys:
        if not isinstance(value,dict):return None
        value=value.get(key)
    return value


def reject_constant(value):
    raise ValueError('nonfinite JSON constant: '+value)


def write_json(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def write_csv(path,rows,fields):
    with path.open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def run_job(manifest_path,expected_sha256,output_dir):
    started=time.perf_counter();cpu=time.process_time()
    manifest_path=Path(manifest_path).resolve();raw=manifest_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==expected_sha256,'exact Root input manifest required')
    manifest=json.loads(raw,parse_constant=reject_constant)
    require(manifest['schema']=='vldb_public_scalar_inputs_v1' and manifest['source_manifest_sha256']==SOURCE_SHA, 'Public scalar source differs')
    root=Path(__file__).resolve().parents[1];output=Path(output_dir).resolve()
    require(not output.exists(),'Fresh scalar output required')
    expected={('ogbn-arxiv',model,seed) for model in ('hgcn','hybonet','sage','gat') for seed in SEEDS}
    expected|={('PubMed','sage',seed) for seed in SEEDS}
    cells=[(entry['dataset'],entry['model'],entry['seed']) for entry in manifest['entries']]
    require(len(cells)==15 and set(cells)==expected,'fixed fifteen training-seed cells required')
    pending=[];provenance=[];cache={};repeat_rows=[];seed_rows=[];context=[];budget_rows=[]

    def load(entry,role):
        binding=entry['files'].get(role);identity=[entry[k] for k in ('dataset','model','seed')]
        if not isinstance(binding,dict) or binding.get('sha256') is None:
            pending.append({'cell':identity,'artifact':role,'reason':'unbound SHA or absent binding'});return None
        require(set(binding)=={'path','sha256'},'exact scalar artifact path/hash required')
        path=(root/binding['path']).resolve();digest=binding['sha256']
        require(path.is_relative_to(root) and path.name==ROLES[role]
                and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),'scalar artifact escaped scope')
        key=(str(path),digest)
        if key in cache:return cache[key]
        if not path.is_file():
            pending.append({'cell':identity,'artifact':role,'reason':'bound file not present'});return None
        content=path.read_bytes()
        require(hashlib.sha256(content).hexdigest()==digest,'bound scalar artifact changed: '+str(path))
        if role=='deployment_costs':
            content=gzip.decompress(content)
            value=[json.loads(line,parse_constant=reject_constant) for line in content.decode('utf-8-sig').splitlines() if line.strip()]
        else:value=json.loads(content.decode('utf-8-sig'),parse_constant=reject_constant)
        provenance.append({'path':str(path),'sha256':digest,'bytes':len(content),'role':role})
        cache[key]=value;return value

    def metric(entry,section,name,unit,values,rule='mean_of_four_paired_repeats',sampler='',tier='',branch='',q=''):
        base={'dataset':entry['dataset'],'model':entry['model'],'seed':entry['seed'],'section':section,
              'sampler':sampler,'tier':tier,'branch':branch,'q':q,'metric':name,'unit':unit,'within_seed_rule':rule}
        valid=[value for value in values if number(value)];complete=len(valid)==len(values) and len(values)>0
        for repeat,value in enumerate(values):
            repeat_rows.append(dict(base,repeat=repeat if len(values)>1 else '',value=value if number(value) else None,
                                    status='complete' if number(value) else 'pending'))
        seed_rows.append(dict(base,n_observations_expected=len(values),n_observations=len(valid),
                              value=statistics.mean(valid) if complete else None,
                              maximum_observed=max(valid) if valid else None,status='complete' if complete else 'pending'))

    for entry in manifest['entries']:
        require(set(entry['files'])==set(ROLES),'exact scalar artifact roles required')
        docs={role:load(entry,role) for role in ROLES}
        result=docs['results'];science=docs['science_completion'];worker=docs['worker_completion']
        if result is not None:
            require(result.get('schema')=='vldb_frozen_final_results_v1'
                    and all(result.get(k)==entry[k] for k in ('dataset','model','seed'))
                    and result.get('checkpoint_sha256')==entry['weights_sha256'],'result cell/CP identity differs')
            budget_sha=get(entry['files']['budget'],'sha256')
            if budget_sha is not None:require(result.get('budget_lock_sha256')==budget_sha,'result budget identity differs')
        eligible=result is not None and get(science,'status')=='completed' and get(worker,'status')=='passed'
        if not eligible:
            pending.append({'cell':[entry[k] for k in ('dataset','model','seed')],'artifact':'final_cell',
                            'reason':'complete results plus successful science/worker completions required'})
        deployment=get(result,'deployment') if eligible else None
        mechanisms=get(result,'mechanisms') if eligible else None
        deployment=deployment if isinstance(deployment,list) else []
        mechanisms=mechanisms if isinstance(mechanisms,list) else []

        def four(rows,filters,keys):
            values=[]
            for repeat in range(4):
                matches=[row for row in rows if row.get('repeat')==repeat and all(row.get(k)==v for k,v in filters.items())]
                values.append(get(matches[0],*keys) if len(matches)==1 else None)
            return values

        for sampler in ('uniform','labor'):
            for tier in ('low','middle','high'):
                filters={'sampler':sampler,'tier':tier}
                for name,unit in (('accuracy','fraction'),('mean_true_class_margin','native_score'),
                                  ('total_workload_e2e_seconds','seconds'),('median_request_e2e_seconds','seconds'),
                                  ('peak_allocated_bytes','bytes'),('peak_reserved_bytes','bytes')):
                    metric(entry,'deployment',name,unit,four(deployment,filters,(name,)),sampler=sampler,tier=tier)
                costs=docs['deployment_costs'] if eligible and isinstance(docs['deployment_costs'],list) else []
                expected_batches=380 if entry['dataset']=='ogbn-arxiv' else 8
                for name in ('cpu_planning_seconds','indexing_h2d_seconds','synchronized_forward_seconds','output_d2h_seconds'):
                    values=[]
                    for repeat in range(4):
                        rows=[row for row in costs if row.get('phase')=='measurement' and row.get('repeat')==repeat
                              and row.get('sampler')==sampler and row.get('tier')==tier]
                        complete=len(rows)==expected_batches and {row.get('batch') for row in rows}==set(range(expected_batches))
                        values.append(sum(row[name] for row in rows) if complete and all(number(row.get(name)) for row in rows) else None)
                    metric(entry,'deployment_workload_components',name,'seconds',values,sampler=sampler,tier=tier)
                warmups=[row for row in costs if row.get('phase')=='paid_warmup' and row.get('sampler')==sampler and row.get('tier')==tier]
                for name,unit in (('e2e_seconds','seconds'),('peak_allocated_bytes','bytes'),('peak_reserved_bytes','bytes')):
                    value=warmups[0].get(name) if len(warmups)==1 else None
                    metric(entry,'paid_warmup',name,unit,[value],rule='one_recorded_paid_warmup',sampler=sampler,tier=tier)
            for branch in ('sampled','local_restored','upstream_restored','both_restored'):
                for name,unit in (('accuracy','fraction'),('mean_true_class_margin','native_score')):
                    metric(entry,'mechanism',name,unit,four(mechanisms,{'sampler':sampler},('metrics',branch,name)),sampler=sampler,tier='middle',branch=branch)
                if branch!='sampled':
                    for name in ('fixed','introduced'):
                        metric(entry,'mechanism',name,'nodes',four(mechanisms,{'sampler':sampler},('fixes_and_breaks',branch,name)),sampler=sampler,tier='middle',branch=branch)
            for name,unit in (('accuracy_interaction','fraction'),('diagnostic_wall_seconds','seconds'),
                              ('outer_wall_including_filter_score_write_seconds','seconds'),
                              ('peak_allocated_bytes','bytes'),('peak_reserved_bytes','bytes')):
                metric(entry,'mechanism',name,unit,four(mechanisms,{'sampler':sampler},(name,)),sampler=sampler,tier='middle')
        full=get(result,'full') if eligible else None
        for name,unit in (('accuracy','fraction'),('mean_true_class_margin','native_score')):
            metric(entry,'full_quality',name,unit,[get(full,'quality',name)],rule='recorded_single_deterministic_full_quality')
        for name,unit in (('format_setup_seconds','seconds'),('full_e2e_median_seconds','seconds'),
                          ('cache_build_median_seconds','seconds'),('target_gather_median_seconds','seconds'),
                          ('RAM_workload_median_seconds','seconds'),('peak_allocated_bytes','bytes'),('peak_reserved_bytes','bytes')):
            rule='recorded_median_of_three_cost_repeats' if 'median' in name else 'recorded_setup_or_peak'
            metric(entry,'full_RAM',name,unit,[get(full,name)],rule=rule)
        amortization=get(full,'amortization');amortization=amortization if isinstance(amortization,list) else []
        for q in (1,10,100):
            matches=[row for row in amortization if row.get('q')==q]
            for name,unit in (('full_recompute_seconds','seconds'),('paid_full_RAM_seconds','seconds'),('RAM_cache_bytes','bytes')):
                metric(entry,'amortization',name,unit,[matches[0].get(name) if len(matches)==1 else None],rule='recorded_frozen_full_RAM_formula',q=q)
        for role,document,names in (('science_completion',science,('worker_wall_seconds','worker_process_cpu_seconds','CP_model_setup_seconds','public_graph_setup_seconds')),
                                    ('worker_completion',worker,('worker_wall_seconds','worker_CPU_seconds','maximum_RSS_KiB'))):
            for name in names:
                metric(entry,role,name,'KiB' if name.endswith('KiB') else 'seconds',[get(document,name)],rule='recorded_nested_usage_do_not_sum_overlapping_scopes')
        context.append({'dataset':entry['dataset'],'model':entry['model'],'seed':entry['seed'],
                        'hardware':docs['hardware'],'budget_hardware':docs['budget_hardware'],
                        'science_status':get(science,'status'),'worker_status':get(worker,'status'),
                        'contention_limitation':get(worker,'contention_limitation'),
                        'CPU_scope':get(worker,'CPU_source'),'legacy_budget_source':get(science,'budget_source_compatibility')})
        if entry['seed']==11:
            budget=docs['budget'];tiers=get(budget,'tiers')
            for tier in ('low','middle','high'):
                matches=[row for row in tiers if row.get('tier')==tier] if isinstance(tiers,list) else []
                row=matches[0] if len(matches)==1 else {}
                budget_rows.append(dict(dataset=entry['dataset'],model=entry['model'],tier=tier,
                    status='complete' if len(matches)==1 else 'pending',
                    **{name:row.get(name) for name in ('uniform_fanout','labor_fanout','cost_match_feasible','uniform_median_seconds','labor_median_seconds','labor_over_uniform_time_ratio')}))

    groups={}
    for row in seed_rows:groups.setdefault(tuple(row[k] for k in GROUP_KEYS),[]).append(row)
    group_rows=[]
    for key,rows in groups.items():
        complete=len(rows)==3 and {row['seed'] for row in rows}==set(SEEDS) and all(row['status']=='complete' for row in rows)
        values=[row['value'] for row in rows if number(row['value'])]
        group_rows.append(dict(zip(GROUP_KEYS,key),n_seeds=sum(row['status']=='complete' for row in rows),expected_seeds=3,
                               mean=statistics.mean(values) if complete else None,
                               sample_std=statistics.stdev(values) if complete else None,status='complete' if complete else 'pending'))
    output.mkdir(parents=True,exist_ok=False)
    base_fields=list(GROUP_KEYS[:2])+['seed']+list(GROUP_KEYS[2:])
    write_csv(output/'per-repeat.csv',repeat_rows,base_fields+['repeat','value','status'])
    write_csv(output/'per-seed.csv',seed_rows,base_fields+['n_observations_expected','n_observations','value','maximum_observed','status'])
    write_csv(output/'across-seeds.csv',group_rows,list(GROUP_KEYS)+['n_seeds','expected_seeds','mean','sample_std','status'])
    write_csv(output/'frozen-budgets.csv',budget_rows,['dataset','model','tier','status','uniform_fanout','labor_fanout','cost_match_feasible','uniform_median_seconds','labor_median_seconds','labor_over_uniform_time_ratio'])
    write_json(output/'hardware-and-usage-context.json',context)
    write_json(output/'provenance.json',{'input_manifest_sha256':expected_sha256,'artifacts':provenance,'no_payload_references_followed':True})
    report={'schema':'vldb_final_scalar_summary_v1','status':'complete' if not pending and all(row['status']=='complete' for row in group_rows) else 'pending',
            'pending_artifacts':pending,'pending_group_metrics':sum(row['status']=='pending' for row in group_rows),
            'training_seeds':list(SEEDS),'repeats_per_training_seed':4,'between_seed_std_ddof':1,
            'units':'accuracy/interaction fractions, native margins, counts, seconds and bytes; no normalization learned',
            'interpretation':['Four repeats are paired observations within a training seed, not independent training seeds.',
                'Full/RAM timing uses the original three-repeat medians and original q=1/10/100 formulas.',
                'Per-seed value is the four-repeat mean where applicable; maximum_observed preserves each seed worst observed peak.',
                'Oracle mechanism states are not deployment states; interaction is not an additive contribution fraction.',
                'Worker/science/scheduler usage scopes overlap; no cumulative total is invented.',
                'Missing/unbound fields are pending, not zero; no labels, NPZ, checkpoint or per-node arrays were read.'],
            'source_manifest_sha256':SOURCE_SHA,'process_CPU_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-started}
    write_json(output/'summary.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Regenerate final scalar tables from public per-cell records.')
    parser.add_argument('--inputs',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run_job(args.inputs,hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest(),args.output)

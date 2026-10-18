"""Fixed descriptive reduction of existing global-diagnostic mechanism arrays.

No model/checkpoint/torch/scientific imports, scores, new predictions or tests.
Only ids, three errors, and four existing correct/margin pairs are loaded.
The same Root-released CPU scope may call this after scalar-summary v1.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROUND='vldb-sampling-mechanisms-v1'
SOURCE_SHA='ece3665a84b76ac7801639561f0d07ab98af02d94c55f28aa8bc1dca82bb5e75'
BRANCHES=('sampled','local_restored','upstream_restored','both_restored')
ERRORS=('local_estimator_sq_l2_sampled_inputs','local_estimator_sq_l2_full_inputs','upstream_input_sq_l2')
PAIRS=(('local_restored','sampled'),('upstream_restored','sampled'),('both_restored','sampled'),
       ('both_restored','upstream_restored'),('both_restored','local_restored'),('upstream_restored','local_restored'))
SEEDS=(11,23,37)
KEYS=('dataset','model','sampler','section','contrast','metric','unit')


def require(ok,message):
    if not ok:raise ValueError(message)


def reject_constant(value):
    raise ValueError('nonfinite JSON constant: '+value)


def write_csv(path,rows,fields):
    with path.open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def write_json(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def run_job(manifest_path,expected_sha256,raw_root,output_dir):
    started=time.perf_counter();cpu=time.process_time()
    raw=Path(manifest_path).resolve().read_bytes()
    require(hashlib.sha256(raw).hexdigest()==expected_sha256,'exact Root input manifest required')
    manifest=json.loads(raw,parse_constant=reject_constant)
    require(manifest['schema']=='vldb_public_mechanism_inputs_v1' and manifest['source_manifest_sha256']==SOURCE_SHA, 'Public source/input identity differs')
    root=Path(raw_root).resolve();output=Path(output_dir).resolve()
    require(not output.exists(),'Fresh output required')
    expected={('ogbn-arxiv',model,seed) for model in ('hgcn','hybonet','sage','gat') for seed in SEEDS}
    expected|={('PubMed','sage',seed) for seed in SEEDS}
    require(len(manifest['entries'])==15 and {(entry['dataset'],entry['model'],entry['seed']) for entry in manifest['entries']}==expected,
            'exact fifteen training-seed cells required')
    import numpy as np
    require(np.__version__ == '1.24.4', 'Use the recorded NumPy 1.24.4 for exact reductions')
    rows=[];pending=[];provenance=[]

    def emit(entry,sampler,repeat,section,contrast,metric,unit,value):
        rows.append({'dataset':entry['dataset'],'model':entry['model'],'seed':entry['seed'],'sampler':sampler,
                     'repeat':repeat,'section':section,'contrast':contrast,'metric':metric,'unit':unit,
                     'value':value,'status':'complete' if value is not None else 'pending'})

    for entry in manifest['entries']:
        records=entry['files']
        require(len(records)==8 and {(r['sampler'],r['repeat']) for r in records}=={(s,r) for s in ('uniform','labor') for r in range(4)},
                'exact two samplers by four existing repeats required')
        identities={};cell_rows_start=len(rows)
        for record in records:
            sampler=record['sampler'];repeat=record['repeat'];binding=record['file'];arrays={};identity=None
            cell=[entry['dataset'],entry['model'],entry['seed'],sampler,repeat]
            path=(root/binding['path']).resolve()
            require(path.is_relative_to(root/'raw/mechanisms') and path.name==f'mechanism-{sampler}-repeat{repeat}.npz',
                    'only existing mechanism NPZ names inside runs permitted')
            reason=None
            if binding['sha256'] is None:reason='unbound original mechanism artifact'
            elif not path.is_file():reason='bound original mechanism artifact missing'
            else:
                require(len(binding['sha256'])==64 and all(c in '0123456789abcdef' for c in binding['sha256']),'invalid artifact digest')
                with path.open('rb') as stream:
                    digest=hashlib.sha256()
                    for chunk in iter(lambda:stream.read(2**20),b''):digest.update(chunk)
                    require(digest.hexdigest()==binding['sha256'],'bound mechanism artifact changed')
                    stream.seek(0)
                    with np.load(stream,allow_pickle=False) as archive:
                        fields=('ids',)+ERRORS+tuple(branch+'__'+suffix for branch in BRANCHES for suffix in ('correct','true_class_margin'))
                        for field in fields:
                            if field not in archive.files:
                                pending.append({'cell':cell,'field':field,'reason':'original array field absent; no forward fallback'});continue
                            try:value=archive[field]
                            except (ValueError,TypeError) as exc:
                                pending.append({'cell':cell,'field':field,'reason':'unreadable allowed array: '+type(exc).__name__});continue
                            count=48603 if entry['dataset']=='ogbn-arxiv' else 1000
                            valid=value.ndim==1 and len(value)==count
                            if field=='ids':valid=valid and value.dtype.kind in 'iu' and len(np.unique(value))==count
                            elif field.endswith('__correct'):valid=valid and value.dtype.kind=='b'
                            else:valid=valid and value.dtype.kind=='f' and bool(np.isfinite(value).all())
                            if not valid:
                                pending.append({'cell':cell,'field':field,'reason':'original dtype/shape/finite values invalid'});continue
                            arrays[field]=value
                        if 'ids' in arrays:
                            identity=hashlib.sha256(arrays['ids'].astype('<i8',copy=False).tobytes()).hexdigest()
                        else:arrays={}
                provenance.append({'cell':cell,'file':binding,'ordered_ids_sha256':identity,
                                   'loaded_fields':sorted(arrays),'weights_sha256_metadata_only':entry['weights_sha256']})
            if reason:pending.append({'cell':cell,'field':'artifact','reason':reason})
            identities[(sampler,repeat)]=identity

            def mean(field):
                return float(np.mean(arrays[field],dtype=np.float64)) if field in arrays else None

            def difference(a,b):
                if a not in arrays or b not in arrays:return None
                return float(np.mean(arrays[a].astype(np.float64)-arrays[b].astype(np.float64)))

            for field in ERRORS:emit(entry,sampler,repeat,'error_mean',field,'mean','native_squared_l2',mean(field))
            emit(entry,sampler,repeat,'error_input_contrast','full_inputs_minus_sampled_inputs','mean','native_squared_l2',difference(ERRORS[1],ERRORS[0]))
            for branch in BRANCHES:
                emit(entry,sampler,repeat,'corner',branch,'accuracy','fraction',mean(branch+'__correct'))
                emit(entry,sampler,repeat,'corner',branch,'mean_true_class_margin','native_score',mean(branch+'__true_class_margin'))
            for after,before in PAIRS:
                for suffix,metric,unit in (('correct','accuracy_delta','fraction'),('true_class_margin','margin_delta','native_score')):
                    emit(entry,sampler,repeat,'paired_corner',after+'_minus_'+before,metric,unit,difference(after+'__'+suffix,before+'__'+suffix))
            for suffix,metric,unit in (('correct','accuracy_interaction','fraction'),('true_class_margin','margin_interaction','native_score')):
                fields=[branch+'__'+suffix for branch in BRANCHES]
                value=None
                if all(field in arrays for field in fields):
                    a,b,c,d=(arrays[field].astype(np.float64) for field in fields)
                    value=float(np.mean(d-b-c+a))
                emit(entry,sampler,repeat,'interaction','both-local-upstream+sampled',metric,unit,value)
            for branch in BRANCHES[1:]:
                before=arrays.get('sampled__correct');after=arrays.get(branch+'__correct')
                fixed=int(np.count_nonzero((~before)&after)) if before is not None and after is not None else None
                broken=int(np.count_nonzero(before&(~after))) if before is not None and after is not None else None
                emit(entry,sampler,repeat,'fix_break',branch+'_vs_sampled','fixed','nodes',fixed)
                emit(entry,sampler,repeat,'fix_break',branch+'_vs_sampled','introduced','nodes',broken)
            del arrays
        # All fixed same-CP/repeat sampler contrasts, without sign selection.
        for repeat in range(4):
            selected=[row for row in rows[cell_rows_start:] if row['repeat']==repeat and row['sampler'] in ('uniform','labor')]
            uniform={(row['section'],row['contrast'],row['metric']):row for row in selected if row['sampler']=='uniform'}
            for row in (r for r in selected if r['sampler']=='labor'):
                other=uniform[(row['section'],row['contrast'],row['metric'])]
                aligned=identities[('uniform',repeat)] is not None and identities[('uniform',repeat)]==identities[('labor',repeat)]
                value=row['value']-other['value'] if aligned and row['value'] is not None and other['value'] is not None else None
                emit(entry,'labor_minus_uniform',repeat,row['section'],row['contrast'],row['metric'],row['unit'],value)
                if not aligned and row['section']=='error_mean':
                    pending.append({'cell':[entry['dataset'],entry['model'],entry['seed'],repeat],
                                    'field':'sampler_pair','reason':'identical ordered original ids required; no reorder or imputation'})
    seed_groups={}
    for row in rows:seed_groups.setdefault(tuple(row[k] for k in KEYS)+(row['seed'],),[]).append(row)
    seed_rows=[]
    for key,group in seed_groups.items():
        complete=len(group)==4 and {r['repeat'] for r in group}=={0,1,2,3} and all(r['value'] is not None for r in group)
        seed_rows.append(dict(zip(KEYS+('seed',),key),n_repeats=sum(r['value'] is not None for r in group),expected_repeats=4,
                              value=statistics.mean(r['value'] for r in group) if complete else None,status='complete' if complete else 'pending'))
    groups={}
    for row in seed_rows:groups.setdefault(tuple(row[k] for k in KEYS),[]).append(row)
    across=[]
    for key,group in groups.items():
        complete=len(group)==3 and {r['seed'] for r in group}==set(SEEDS) and all(r['value'] is not None for r in group)
        values=[r['value'] for r in group if r['value'] is not None]
        across.append(dict(zip(KEYS,key),n_seeds=len(values),expected_seeds=3,
                           mean=statistics.mean(values) if complete else None,sample_std=statistics.stdev(values) if complete else None,
                           status='complete' if complete else 'pending'))
    output.mkdir(parents=True,exist_ok=False)
    write_csv(output/'per-repeat.csv',rows,list(KEYS)+['seed','repeat','value','status'])
    write_csv(output/'per-seed.csv',seed_rows,list(KEYS)+['seed','n_repeats','expected_repeats','value','status'])
    write_csv(output/'across-seeds.csv',across,list(KEYS)+['n_seeds','expected_seeds','mean','sample_std','status'])
    joint={};columns=[]
    for row in rows:
        if row['section'] not in ('error_mean','corner','paired_corner','interaction','fix_break'):continue
        identity=tuple(row[k] for k in ('dataset','model','seed','sampler','repeat'))
        item=joint.setdefault(identity,dict(zip(('dataset','model','seed','sampler','repeat'),identity),status='complete'))
        column=row['section']+'__'+row['contrast']+'__'+row['metric']
        if column not in columns:columns.append(column)
        item[column]=row['value']
        if row['status']!='complete':item['status']='pending'
    write_csv(output/'error-accuracy-pairs.csv',list(joint.values()),['dataset','model','seed','sampler','repeat']+columns+['status'])
    write_json(output/'provenance.json',{'manifest_sha256':expected_sha256,'artifacts':provenance,'non_allowlisted_arrays_loaded':False})
    report={'schema':'vldb_mechanism_descriptive_summary_v1','status':'complete' if not pending and all(r['status']=='complete' for r in across) else 'pending',
            'pending':pending,'pending_group_metrics':sum(r['status']=='pending' for r in across),
            'scope':'Existing global sampled-prefix diagnostic at original target ids; not dependency-cone deployment states or deployment errors.',
            'interpretation':['Native error/margin scales remain separated by dataset and model; no cross-model pooling or ranking.',
                'All same-CP/repeat LABOR-minus-Uniform scalar contrasts are retained, including negative or mixed effects.',
                'Four-corner differences are paired by original stored row; interaction is D-B-C+A, not an additive attribution fraction.',
                'Stored correctness/margins are reduced, never recalculated from scores. Fix/break are Boolean transitions from sampled.',
                'Means use float64 accumulation of existing values; no threshold, fitting, filtering, correlations or inferential tests.',
                'Error reduction and accuracy change are exported together descriptively; no monotonic link is assumed or asserted before data.',
                'Four repeats enter each training-seed mean; only seeds11/23/37 enter sample std(ddof1). Missing stays pending/null.'],
            'process_CPU_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-started}
    write_json(output/'summary.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Reduce published mechanism arrays without running a model.')
    parser.add_argument('--inputs',required=True);parser.add_argument('--raw-root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run_job(args.inputs,hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest(),args.raw_root,args.output)

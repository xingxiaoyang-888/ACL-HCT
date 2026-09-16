"""Explicit offline preparation only; never starts training or downloads."""
import argparse
import gzip
import hashlib
import json
import platform
from pathlib import Path
import tarfile
import time
import numpy as np
from acl_hct.taxonomy import wordnet_records, mesh_records
from acl_hct.protocols import grouped_split, observed_neighbors, training_candidates, fit_text_features, digest


def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def prepare(taxonomy, output_root, split_seed=20260914, dimension=128, feature_seed=11, negative_seed=11):
    """Produce separated model inputs and evaluator-only labels/path metadata."""
    output_root=Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError('output directory must be empty; never overwrite a prepared split')
    nodes=sorted(taxonomy.records); split=grouped_split(nodes,taxonomy.edges,split_seed)
    features,feature_manifest,state=fit_text_features(taxonomy.records,split['entities']['train'],dimension,feature_seed)
    graph=observed_neighbors(nodes,split['relations']['train'])
    queries=training_candidates(nodes,split['entities']['train'],split['relations']['train'],seed=negative_seed)
    degrees=[len(row) for row in graph]; total=sum(degrees)
    coverage=[{'fanout':f,'affected_nodes':sum(d>f for d in degrees),
               'affected_node_fraction':sum(d>f for d in degrees)/len(nodes),
               'removed_message_fraction':sum(max(0,d-f) for d in degrees)/total if total else 0.,
               'all_full_equivalent':all(d<=f for d in degrees),
               'mean_k_over_N_nonempty':sum(min(d,f)/d for d in degrees if d)/max(1,sum(d>0 for d in degrees))}
              for f in (4,8,16,32,64)]
    from acl_hct import taxonomy as taxonomy_module, protocols as protocols_module, data as data_module
    source_paths=[Path(__file__),Path(taxonomy_module.__file__),Path(protocols_module.__file__),Path(data_module.__file__)]
    source_hashes={p.name:hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest() for p in source_paths}
    children={b for a,b in taxonomy.edges}; incident={v for edge in taxonomy.edges for v in edge}
    manifest={'source_sha256_normalized_lf':source_hashes,'python':platform.python_version(),'numpy':np.__version__,
              'protocol':split['protocol'],'split_hash':split['split_hash'],'feature_manifest':feature_manifest,
              'graph_hash':digest(graph),'train_queries_hash':digest(queries),'node_order_hash':digest(nodes),
              'coverage':coverage,'isolated_observed_nodes':sum(d==0 for d in degrees),
              'split_seed':split_seed,'feature_seed':feature_seed,'negative_seed':negative_seed,
              'entity_counts':{key:len(value) for key,value in split['entities'].items()},
              'positive_relation_counts':{key:len(value) for key,value in split['relations'].items()},
              'supervised_child_counts':{key:len({b for a,b in value}) for key,value in split['relations'].items()},
              'roots_per_split':{key:sum(node not in children for node in value) for key,value in split['entities'].items()},
              'isolates_per_split':{key:sum(node not in incident for node in value) for key,value in split['entities'].items()},
              'text_fit_entities':split['entities']['train'], 'structure':taxonomy.validate(),
              'status':'prepared under frozen protocol V1; training-runner acceptance required before real training'}
    output_root.mkdir(parents=True,exist_ok=True)
    payloads={'input_manifest.json':manifest,'observed_graph.json':{'nodes':nodes,'neighbors':graph},
              'train_queries.json':queries,'entity_split.json':split['entities'],
              'evaluator_truth.json':{'records':taxonomy.records,'edges':sorted(taxonomy.edges),'diagnostics':taxonomy.diagnostics},
              'evaluator_valid.json':split['relations']['valid'],'evaluator_test.json':split['relations']['test'],
              'text_vocabulary.json':state['vocabulary']}
    for name,payload in payloads.items():
        (output_root/name).write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    np.savez_compressed(output_root/'features.npz',features=features)
    np.savez_compressed(output_root/'text_transform.npz',idf=state['idf'],components=state['components'])
    return manifest


def main():
    cli=argparse.ArgumentParser();cli.add_argument('--dataset',choices=['wordnet','mesh'],required=True)
    cli.add_argument('--raw',type=Path,required=True);cli.add_argument('--source-manifest',type=Path,required=True)
    cli.add_argument('--output-root',type=Path,required=True);cli.add_argument('--split-seed',type=int,default=20260914)
    cli.add_argument('--feature-seed',type=int,default=11);cli.add_argument('--negative-seed',type=int,default=11)
    cli.add_argument('--dimension',type=int,default=128);cli.add_argument('--threads',type=int,choices=[1,2,4,8],default=2);args=cli.parse_args()
    started=time.perf_counter()
    source=json.loads(args.source_manifest.read_text(encoding='utf-8'))
    if file_hash(args.raw)!=source['sha256']: raise ValueError('raw data hash differs from source manifest')
    if args.dataset=='wordnet':
        with tarfile.open(args.raw) as archive:
            members=[m for m in archive.getmembers() if m.name.endswith('/dict/data.noun')]
            if len(members)!=1: raise ValueError('unexpected WordNet archive layout')
            with archive.extractfile(members[0]) as stream:
                taxonomy=wordnet_records(line.decode('utf-8') for line in stream)
    else:
        with gzip.open(args.raw,'rb') as stream: taxonomy=mesh_records(stream)
    if taxonomy.validate()!=source['structure']: raise ValueError('rich parser structure differs from source manifest')
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.threads):
        result=prepare(taxonomy,args.output_root,args.split_seed,args.dimension,args.feature_seed,args.negative_seed)
    (args.output_root/'source_manifest.json').write_text(json.dumps(source,indent=2)+'\n',encoding='utf-8')
    (args.output_root/'preprocessing_runtime.json').write_text(json.dumps({'threads':args.threads, 'elapsed_seconds':time.perf_counter()-started, 'raw_sha256':source['sha256']},indent=2)+'\n')
    print(json.dumps({'protocol':result['protocol'],'split_hash':result['split_hash'],
                      'nodes':len(taxonomy.records),'output':str(args.output_root)}))


if __name__=='__main__':main()

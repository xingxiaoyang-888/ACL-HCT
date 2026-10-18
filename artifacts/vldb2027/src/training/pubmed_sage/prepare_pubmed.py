"""New PubMed public/train-fit-dev preparation. Never opens ind.pubmed.ty.

The immutable official Planetoid pickle files are hash-bound before loading.
Graph/features can include test nodes in this transductive task. Only the first
1060 ally label rows (60 train + official 1000 validation) are semantically
selected; the complete ally container is deserialized, and this is reported.
Official test indices are public metadata; test labels are not an input.
"""
from pathlib import Path
import pickle
import numpy as np
import scipy.sparse as sp
from common import ROUND, release, hash_order, batches, sha, save


RAW_NAMES = {'x','y','tx','allx','ally','graph','test.index'}


def run_job(release_path, expected_release_sha256, guard_callback):
    if not callable(guard_callback):
        raise ValueError('real Ops guard callback required')
    guard_callback()
    r, inputs, cfg, output = release(release_path, expected_release_sha256, 'prepare_pubmed')
    if set(inputs) != RAW_NAMES or r['seed'] is not None:
        raise ValueError('exact non-test-label raw inputs required')
    if cfg['schema'] != 'vldb_pubmed_sage_config_v1':
        raise ValueError('wrong frozen configuration')
    expected_names = {k: 'ind.pubmed.'+k for k in RAW_NAMES}
    if any(inputs[k].name != expected_names[k] for k in RAW_NAMES):
        raise ValueError('raw role basename mismatch')
    def load(name):
        with inputs[name].open('rb') as f:
            return pickle.load(f, encoding='latin1')
    x, tx, allx = load('x'), load('tx'), load('allx')
    if x.shape != (60,500) or tx.shape != (1000,500) or allx.shape != (18717,500):
        raise ValueError('official PubMed feature shapes differ')
    test_ids = np.array([int(v) for v in inputs['test.index'].read_text('ascii').splitlines()], dtype=np.int64)
    if len(test_ids) != 1000 or len(np.unique(test_ids)) != 1000 or np.any(test_ids < 1060) or np.any(test_ids >= 19717):
        raise ValueError('official PubMed test IDs malformed')
    features = sp.vstack((allx,tx),format='lil')
    features[test_ids,:] = features[np.sort(test_ids),:]
    features = features.tocsr().astype(np.float32)
    row_sum = np.asarray(features.sum(1)).ravel()
    inverse = np.divide(1., row_sum, out=np.zeros_like(row_sum), where=row_sum != 0)
    dense = (sp.diags(inverse) @ features).toarray().astype(np.float32)
    if dense.shape != (19717,500) or not np.isfinite(dense).all():
        raise ValueError('nonfinite prepared features')
    graph = load('graph')
    edges = [(int(v),int(w)) for v,neighbors in graph.items() for w in neighbors if int(v) != int(w)]
    if any(min(v,w)<0 or max(v,w)>=19717 for v,w in edges):
        raise ValueError('graph ID outside official range')
    packed = np.unique(np.array([v*19717+w for v,w in edges] + [w*19717+v for v,w in edges],dtype=np.int64))
    rows, indices = packed//19717, packed%19717
    indptr = np.r_[0,np.cumsum(np.bincount(rows,minlength=19717))].astype(np.int64)
    if len(indices) != 88648:
        raise ValueError('canonical PubMed nonself edge count differs')
    train_y = np.asarray(load('y'))
    ally = np.asarray(load('ally'))
    if train_y.shape != (60,3) or ally.shape != (18717,3):
        raise ValueError('official label container shape differs')
    allowed = ally[:1060]
    if not np.array_equal(train_y,allowed[:60]) or not np.all(allowed.sum(1)==1) or not np.all((allowed==0)|(allowed==1)):
        raise ValueError('allowed one-hot labels malformed')
    labels = allowed.argmax(1).astype(np.int64)
    ordered = hash_order(range(60,1060), cfg['split']['hash_prefix'])
    ids = {'train':list(range(60)), 'fit':ordered[:500], 'dev':ordered[500:], 'test':test_ids.tolist()}
    output.mkdir(parents=True,exist_ok=False)
    public, development = output/'public',output/'development'
    public.mkdir();development.mkdir()
    np.savez(public/'graph.npz',x=dense,indptr=indptr,indices=indices,values=np.ones(len(indices),dtype=np.float32),diagonal=np.zeros(19717,dtype=np.float32))
    save(public/'ids.json',{'schema':'vldb_pubmed_ids_v1','dataset':'PubMed','nodes':19717,'features':500,'classes':3,'ids':ids,
                           'batches':{s:batches(v) for s,v in ids.items() if s != 'train'},'split_hash_prefix':cfg['split']['hash_prefix']})
    for split in ('train','fit','dev'):
        a=np.asarray(ids[split],dtype=np.int64)
        np.savez(development/(split+'.npz'),ids=a,y=labels[a])
    receipt={'schema':'vldb_pubmed_preparation_v1','round':ROUND,'release_sha256':expected_release_sha256,
        'source_manifest_sha256':r['source_manifest']['sha256'],'config_sha256':r['config']['sha256'],
        'raw':{k:sha(p) for k,p in inputs.items()},'public':{p.name:sha(p) for p in public.iterdir()},
        'development':{p.name:sha(p) for p in development.iterdir()},'test_label_file_opened':False,
        'label_container_deserialized':'y 60x3 and ally 18717x3; only ally rows 0..1059 selected/decoded',
        'semantic_label_rows_used':1060,'test_scores_computed':False,
        'graph_policy':'simple symmetric nonself graph; SAGE owns separate root transform',
        'feature_policy':'fixed Planetoid row-L1 normalization, zero rows remain zero',
        'historical_test_scope':'Prior HyboNet/LResNet studies evaluated PubMed test; this new SAGE config is prospectively fixed independently of those scores.'}
    save(public/'preparation.json',receipt)
    guard_callback()
    return receipt

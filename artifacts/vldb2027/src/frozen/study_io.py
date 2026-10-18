"""Exact frozen study bindings; public graph and one released test label file.

No dataset download, raw OGB/Planetoid loader, training label access or source
discovery. Resource guards and scientific cell replay barriers belong to the
released caller. All paths/hashes arrive through the explicit job.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re

ROUND='vldb-sampling-mechanisms-v1'


def require(value,message):
    if not bool(value):raise ValueError(message)


def path(value):
    require(isinstance(value,str) and Path(value).is_absolute(),'absolute path required')
    return Path(value).resolve()


def ref(value):
    require(isinstance(value,dict) and set(value)=={'path','sha256'},'exact path/hash binding required')
    require(isinstance(value['sha256'],str) and re.fullmatch('[0-9a-f]{64}',value['sha256']),'lowercase SHA256 required')
    return path(value['path'])


@contextmanager
def bound(value):
    with ref(value).open('rb') as stream:
        digest=hashlib.sha256()
        for chunk in iter(lambda:stream.read(2**20),b''):digest.update(chunk)
        require(digest.hexdigest()==value['sha256'],'bound bytes changed: '+value['path'])
        stream.seek(0);yield stream


def read_json(value):
    with bound(value) as stream:return json.loads(stream.read().decode('utf-8-sig'))


def save(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def npz(value,keys):
    import numpy as np
    with bound(value) as stream,np.load(stream,allow_pickle=False) as archive:
        require(len(archive.files)==len(keys) and set(archive.files)==set(keys),'unexpected NPZ array fields')
        return {key:archive[key].copy() for key in keys}


def tree(value,role):
    require(set(value)=={'root','manifest'},'exact root/manifest tree binding required')
    root=path(value['root']);manifest=read_json(value['manifest']);seen=set()
    expected='vldb_frozen_source_manifest_v1' if role=='code' else 'dasfaa_training_tree_manifest_v1'
    require(manifest['schema']==expected,'wrong source manifest schema')
    if role!='code':require(manifest['role']==role,'wrong immutable tree role')
    for entry in manifest['files']:
        relative=Path(entry['path']);member=(root/relative).resolve()
        require(not relative.is_absolute() and '..' not in relative.parts and member.is_relative_to(root)
                and member not in seen and member.suffix=='.py','source member escaped/duplicated')
        with bound({'path':str(member),'sha256':entry['sha256']}):pass
        seen.add(member)
    if role=='sampler':require(seen=={root/'blocks.py',root/'common.py'} and not (root/'.deps').exists(),
                               'sampler exact two-file source without unbound .deps required')
    return root,seen


class PublicGraph:
    kind='sage'
    def __init__(self,x,indptr,indices):
        import numpy as np
        self.x=x;self.indptr=indptr;self.indices=indices;self.degree=np.diff(indptr);self.n=len(x);self._dgl=None
    def dgl_graph(self):
        if self._dgl is None:
            import numpy as np
            import torch
            import dgl
            with torch.inference_mode(False):
                rows=np.repeat(np.arange(self.n,dtype=np.int64),self.degree)
                self._dgl=dgl.graph((torch.tensor(self.indices),torch.from_numpy(rows)),num_nodes=self.n)
                self._dgl.create_formats_()
        return self._dgl


def load_public(job,plan):
    import numpy as np
    import torch
    refs=job['public'];require(set(refs)=={'graph','ids','preparation'},'exact public input roles required')
    shape=plan['datasets'][job['dataset']]['shape'];n=shape['nodes']
    meta=read_json(refs['ids']);receipt=read_json(refs['preparation'])
    require(meta['dataset']==job['dataset'] and meta['classes']==shape['classes'],'public dataset metadata differs')
    for role in ('graph','ids'):
        filename=role+('.npz' if role=='graph' else '.json')
        require(receipt['public'][filename]==refs[role]['sha256'],'public preparation hash differs')
    arrays=npz(refs['graph'],{'x','indptr','indices','values','diagonal'})
    x,ptr,idx=(arrays[k] for k in ('x','indptr','indices'))
    require(x.dtype==np.float32 and x.shape==(n,shape['features']) and np.isfinite(x).all(),'public feature shape/dtype differs')
    require(ptr.dtype==idx.dtype==np.int64 and ptr.shape==(n+1,) and idx.ndim==1 and ptr[0]==0 and ptr[-1]==len(idx)
            and np.all(np.diff(ptr)>=0) and np.all((idx>=0)&(idx<n)),'public CSR malformed')
    rows=np.repeat(np.arange(n,dtype=np.int64),np.diff(ptr))
    require(not np.any(rows==idx) and not np.any((rows[1:]==rows[:-1])&(idx[1:]<=idx[:-1])),'nonself sorted distinct CSR required')
    require(arrays['values'].dtype==np.float32 and arrays['values'].shape==idx.shape and np.all(arrays['values']==1)
            and arrays['diagonal'].dtype==np.float32 and arrays['diagonal'].shape==(n,) and np.all(arrays['diagonal']==0),'unweighted public graph required')
    return PublicGraph(torch.from_numpy(x),ptr,idx),meta,receipt


def hash_order(values,prefix):
    return sorted(map(int,values),key=lambda n:(hashlib.sha256((prefix+str(n)).encode()).digest(),n))


def calibration_batches(meta,dataset,plan):
    import numpy as np
    split='gate' if dataset=='ogbn-arxiv' else 'dev'
    values=meta['ids'][split]
    require(len(values)==(4967 if dataset=='ogbn-arxiv' else 500),'fixed calibration development split differs')
    ordered=hash_order(values,plan['calibration']['target_hash_prefix']+dataset+':')
    chosen=np.array(ordered[:1024] if dataset=='ogbn-arxiv' else ordered,dtype=np.int64)
    batches=tuple(np.array_split(chosen,8))
    require(all(0<len(b)<=128 for b in batches),'exact eight bounded calibration batches required')
    return batches


def test_targets(job,meta,shape):
    import numpy as np
    target=read_json(job['targets'])
    require(set(target)=={'schema','dataset','split','ids','batches','batch_size','source_provenance'}
            and target['schema']=='vldb_sampling_test_targets_v1' and target['dataset']==job['dataset']
            and target['split']=='official_test' and target['batch_size']==128,'exact official test target manifest required')
    ids=np.asarray(target['ids'],dtype=np.int64)
    require(ids.ndim==1 and len(ids)==shape['test'] and len(np.unique(ids))==len(ids)
            and np.all((ids>=0)&(ids<shape['nodes'])),'test target IDs malformed')
    existing=np.concatenate([np.asarray(v,dtype=np.int64) for k,v in meta['ids'].items() if k!='test'])
    require(not np.intersect1d(existing,ids).size,'test IDs overlap training/development IDs')
    if job['dataset']=='PubMed':require(np.array_equal(np.sort(ids),np.sort(meta['ids']['test'])),'PubMed official test IDs differ')
    batches=tuple(np.asarray(v,dtype=np.int64) for v in target['batches'])
    require(batches and all(len(v)==128 for v in batches[:-1]) and 0<len(batches[-1])<=128
            and np.array_equal(np.concatenate(batches),ids),'test batches must partition IDs once in exact order')
    return ids,batches


def test_labels(binding,ids,classes):
    import numpy as np
    arrays=npz(binding,{'ids','y'});source,labels=arrays['ids'],arrays['y']
    require(source.dtype==labels.dtype==np.int64 and source.shape==labels.shape==ids.shape
            and len(np.unique(source))==len(source) and np.array_equal(np.sort(source),np.sort(ids))
            and np.all((labels>=0)&(labels<classes)),'released official test label IDs/values differ')
    order=np.argsort(source);return labels[order][np.searchsorted(source[order],ids)]

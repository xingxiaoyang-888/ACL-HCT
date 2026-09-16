"""Lossless NPZ arrays plus compact JSON manifests for private diagnostic archives."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .evaluate_checkpoint import file_sha256


def write_archive(directory, name, payload):
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    if not name.replace('_','').replace('-','').isalnum():raise ValueError('simple artifact name required')
    target=root/(name+'.npz');manifest=root/(name+'.json')
    if target.exists() or manifest.exists():raise ValueError('artifact already exists; no overwrite/resume')
    arrays={};metadata={}
    def encode(value):
        if isinstance(value,torch.Tensor):value=value.detach().cpu().numpy()
        if isinstance(value,np.ndarray):
            if value.dtype.hasobject:raise ValueError('object arrays forbidden')
            if value.ndim==0:return value.item()
            key=f'a{len(arrays):05d}';value=np.ascontiguousarray(value)
            arrays[key]=value
            metadata[key]={'shape':list(value.shape),'dtype':value.dtype.str,
                           'data_sha256':hashlib.sha256(value.tobytes()).hexdigest()}
            return {'npz_array':key}
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,dict):return {str(k):encode(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):
            if len(value)>=64 and all(type(v) in (int,float,bool,str) for v in value):
                return encode(np.asarray(value))
            return [encode(v) for v in value]
        return value
    tree=encode(payload)
    temporary=root/(name+'.npz.tmp')
    with temporary.open('xb') as stream:np.savez_compressed(stream,**arrays)
    temporary.replace(target)
    record={'schema_version':1,'array_file':target.name,'array_file_sha256':file_sha256(target),
            'arrays':metadata,'payload':tree,'array_bytes':target.stat().st_size}
    with manifest.open('x',encoding='utf-8',newline='\n') as stream:json.dump(record,stream,allow_nan=False,separators=(',',':'))
    return {'manifest':manifest.name,'manifest_sha256':file_sha256(manifest),
            'array_file':target.name,'array_file_sha256':record['array_file_sha256'],
            'array_bytes':record['array_bytes'],'arrays':len(arrays)}


def read_archive(directory, descriptor):
    root=Path(directory).resolve()
    def child(name):
        path=(root/name).resolve()
        if path.parent!=root:raise ValueError('artifact path escapes directory')
        return path
    manifest=child(descriptor['manifest'])
    if file_sha256(manifest)!=descriptor['manifest_sha256']:raise ValueError('manifest hash mismatch')
    record=json.loads(manifest.read_text(encoding='utf-8'));path=child(record['array_file'])
    if file_sha256(path)!=record['array_file_sha256']:raise ValueError('NPZ hash mismatch')
    with np.load(path,allow_pickle=False) as loaded:
        if set(loaded.files)!=set(record['arrays']):raise ValueError('array inventory mismatch')
        arrays={}
        for key,meta in record['arrays'].items():
            value=loaded[key]
            if (list(value.shape)!=meta['shape'] or value.dtype.str!=meta['dtype']
                    or hashlib.sha256(value.tobytes()).hexdigest()!=meta['data_sha256']):
                raise ValueError('array identity mismatch')
            arrays[key]=value
    def decode(value):
        if isinstance(value,dict):
            if set(value)=={'npz_array'}:return arrays[value['npz_array']]
            return {k:decode(v) for k,v in value.items()}
        if isinstance(value,list):return [decode(v) for v in value]
        return value
    return decode(record['payload'])

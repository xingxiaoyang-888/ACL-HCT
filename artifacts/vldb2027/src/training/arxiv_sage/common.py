"""Small round-local contracts. No old prediction, test or ledger imports."""
from pathlib import Path
import hashlib
import json
import os
import sys
import numpy as np

CODE = Path(__file__).resolve().parent
PROJECT = CODE.parents[2]
ROUND = CODE.parent
sys.dont_write_bytecode = True
if (CODE / '.deps').is_dir():
    sys.path.insert(0, str(CODE / '.deps'))
os.environ.setdefault('DGLBACKEND', 'pytorch')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '7')
import torch


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def domain(stage, seed, repeat=0, batch=0, layer=0, path='main'):
    # k/action intentionally absent: U/P/L share row priorities; E gets a new path.
    text = f'icml-compute-allocation-v1:{stage}:{seed}:{repeat}:{batch}:{layer}:{path}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little') & ((1 << 63)-1)


def hash_order(ids, prefix):
    return sorted(map(int, ids), key=lambda n: (hashlib.sha256((prefix + str(n)).encode()).digest(), n))


def splits(valid):
    ids = hash_order(valid, 'icml-ca-v1:valid-split:')
    development, confirmation = ids[:len(ids)//2], ids[len(ids)//2:]
    development = hash_order(development, 'icml-ca-v1:dev-split:')
    return {'fit': development[:2*len(development)//3], 'gate': development[2*len(development)//3:], 'confirmation': confirmation}


def batches(ids, size=128):
    ids = hash_order(ids, 'icml-ca-v1:batch:')
    return [ids[i:i+size] for i in range(0, len(ids), size)]


def configure():
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def load_npz(path, keys):
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != set(keys):
            raise ValueError(f'Unexpected input fields: {path}')
        return {k: z[k].copy() for k in z.files}


def release(path, stage, seed, inputs):
    """Root writes releases; this consumer only checks exact bound artifacts."""
    r = read_json(path)
    if r.get('schema') != 'icml_ca_release_v1' or r.get('stage') != stage or r.get('seed') != seed:
        raise ValueError('Exact Root release stage/seed required')
    if r.get('authorized') is not True:
        raise ValueError('Root release is not authorized')
    for name, file in inputs.items():
        if r.get('inputs', {}).get(name) != sha(file):
            raise ValueError('Released input differs: ' + name)
    manifest = Path(r['source_manifest'])
    if sha(manifest) != r['source_manifest_sha256']:
        raise ValueError('Released source manifest differs')
    for rec in read_json(manifest)['files']:
        file = CODE / rec['path']
        if not file.resolve().is_relative_to(CODE) or sha(file) != rec['sha256']:
            raise ValueError('Released source changed: ' + rec['path'])
    return {'release_sha256': sha(path), 'source_manifest_sha256': sha(manifest), 'inputs': r['inputs']}


def fresh_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def accuracy_counts(output, y):
    if not bool(torch.isfinite(output).all()) or len(y) != len(output):
        raise ValueError('Nonfinite output or target mismatch')
    return int((output.argmax(-1).cpu() == torch.as_tensor(y)).sum())

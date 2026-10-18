"""VLDB round-local source binding and RNG; no execution or ledger authority."""
from pathlib import Path
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault('DGLBACKEND', 'pytorch')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

ROUND = 'vldb-sampling-mechanisms-v1'
CODE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def bound(ref):
    if set(ref) != {'path', 'sha256'} or not Path(ref['path']).is_absolute():
        raise ValueError('exact absolute path and SHA256 binding required')
    p = Path(ref['path']).resolve()
    if sha(p) != ref['sha256']:
        raise ValueError('bound input changed: ' + str(p))
    return p


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def domain(stage, seed, repeat=0, batch=0, layer=0, path='main'):
    value = f'{ROUND}:pubmed:{stage}:{seed}:{repeat}:{batch}:{layer}:{path}'
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little') & ((1 << 63)-1)


def hash_order(ids, prefix):
    return sorted(map(int, ids), key=lambda n: (hashlib.sha256((prefix+str(n)).encode()).digest(), n))


def batches(ids, size=128):
    ordered = hash_order(ids, ROUND + ':pubmed:batch:')
    return [ordered[i:i+size] for i in range(0, len(ordered), size)]


def configure():
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def release(path, expected_sha256, stage):
    """Expected hash comes from Root's separate signed handoff to Ops."""
    path = Path(path).resolve()
    if sha(path) != expected_sha256:
        raise ValueError('release hash mismatch')
    r = read_json(path)
    required = {'schema','round','authorized','stage','seed','inputs','source_manifest','config','output'}
    if set(r) != required or r['schema'] != 'vldb_pubmed_release_v1' or r['round'] != ROUND:
        raise ValueError('new exact VLDB release required')
    if r['authorized'] is not True or r['stage'] != stage:
        raise ValueError('wrong stage or unreleased entry')
    manifest = read_json(bound(r['source_manifest']))
    if manifest['schema'] != 'vldb_pubmed_source_manifest_v1' or {rec['path'] for rec in manifest['files']} != {'common.py','blocks.py','sage.py','prepare_pubmed.py','train_pubmed.py'}:
        raise ValueError('wrong source manifest')
    for rec in manifest['files']:
        member = (CODE / rec['path']).resolve()
        if not member.is_relative_to(CODE) or sha(member) != rec['sha256']:
            raise ValueError('source binding differs')
    inputs = {k: bound(v) for k,v in r['inputs'].items()}
    config = read_json(bound(r['config']))
    output = Path(r['output'])
    if not output.is_absolute() or output.exists():
        raise ValueError('fresh absolute output directory required')
    return r, inputs, config, output

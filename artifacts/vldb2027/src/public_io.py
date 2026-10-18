"""Portable file contracts. Standard library only; no experiment on import."""
from pathlib import Path
import hashlib
import json

PACKAGE = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def binding(value, base):
    if not isinstance(value, dict) or set(value) != {'path', 'sha256'}:
        raise ValueError('Each file requires exactly path and sha256')
    digest = value['sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Supply a lowercase SHA256; placeholders are not accepted')
    p = Path(value['path'])
    if not p.is_absolute():
        p = Path(base) / p
    p = p.resolve()
    if not p.is_file() or sha(p) != digest:
        raise ValueError('Missing or changed input: ' + str(p))
    return {'path': str(p), 'sha256': digest}


def file_ref(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': sha(p)}


def verify_sources():
    """Check only the published text/source closure; no third-party imports."""
    manifest = read(PACKAGE / 'config/source-manifest.json')
    seen = set()
    for entry in manifest['files']:
        p = (PACKAGE / entry['path']).resolve()
        if not p.is_relative_to(PACKAGE) or p in seen or sha(p) != entry['published_sha256']:
            raise ValueError('Published source closure differs: ' + entry['path'])
        seen.add(p)
    return manifest


def native_root(model, value, base):
    """Only reconstructed exact native trees; no installation or source discovery."""
    if model not in ('hgcn', 'hybonet'):
        if value is not None:
            raise ValueError('Euclidean cells have no native source tree')
        return None
    if not isinstance(value, str):
        raise ValueError('Provide the reconstructed native tree path')
    p = Path(value)
    if not p.is_absolute():
        p = Path(base) / p
    p = p.resolve()
    inventory = read(PACKAGE / 'config/native-sources.json')
    group = next(g for g in inventory['groups'] if g['model'] == model)
    allowed = {(p / entry['path']).resolve() for entry in group['files']}
    if set(p.rglob('*.py')) != allowed or (p / '.deps').exists():
        raise ValueError('Native tree must be the exact reconstructed source allowlist')
    for entry in group['files']:
        member = (p / entry['path']).resolve()
        if not member.is_relative_to(p) or sha(member) != entry['accepted_sha256']:
            raise ValueError('Native reconstruction differs: ' + entry['path'])
    return p


def configure_cuda():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly one allocated CUDA device with CUDA_VISIBLE_DEVICES')
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(7)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def local_boundary():
    """Synchronization only. This does NOT certify exclusive hardware ownership."""
    import torch
    torch.cuda.synchronize(0)


"""Bind an unmodified external HazyResearch/hgcn checkout, without vendoring it."""
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


UPSTREAM_COMMIT = 'a526385744da25fc880f3da346e17d0fe33817f8'
MANIFEST_SHA256 = '7b8e2a18d40d91147f31cc09d03739b18c9335f727aad45c0e3e8d9c0954e6e8'
_PACKAGES = ('layers', 'manifolds', 'models', 'utils', 'optimizers', 'config')
_CACHE = {}


def verify_checkout(root, manifest_path=None):
    root = Path(root).resolve(strict=True)
    manifest_path = Path(manifest_path) if manifest_path else Path(__file__).parents[2] / 'configs/hgcn_upstream.json'
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError('unregistered upstream manifest')
    manifest = json.loads(raw)
    if manifest['commit'] != UPSTREAM_COMMIT:
        raise ValueError('unregistered upstream revision')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, timeout=5).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError('upstream checkout HEAD differs from fixed revision')
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=root, text=True, timeout=5).splitlines())
    if tracked != set(manifest['files']):
        raise ValueError('upstream tracked inventory differs from fixed revision')
    # Reject shadow modules, including untracked __init__.py in namespace packages.
    actual_python = {p.relative_to(root).as_posix() for p in root.rglob('*.py') if '.git' not in p.parts}
    if actual_python != {p for p in tracked if p.endswith('.py')}:
        raise ValueError('upstream Python inventory contains missing or shadow modules')
    hashes = {}
    for name, record in manifest['files'].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError('unsafe or missing upstream file: ' + name)
        content = path.read_bytes()
        if record['normalization'] == 'LF':
            content = content.replace(b'\r\n', b'\n')
        digest = hashlib.sha256(content).hexdigest()
        if digest != record['sha256'] or len(content) != record['bytes']:
            raise ValueError('modified upstream file: ' + name)
        hashes[name] = digest
    return {'repository': manifest['repository'], 'commit': commit,
            'manifest_sha256': MANIFEST_SHA256, 'file_sha256': hashes,
            'license_status': manifest['license_status'], 'checkout_modified': False}


def _belongs(name):
    return any(name == p or name.startswith(p + '.') for p in _PACKAGES)


@contextmanager
def _import_scope(root):
    saved = {k: v for k, v in sys.modules.items() if _belongs(k)}
    for name in saved:
        del sys.modules[name]
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        for name in list(sys.modules):
            if _belongs(name):
                del sys.modules[name]
        sys.modules.update(saved)
        sys.path.remove(str(root))


def load_upstream(root, manifest_path=None):
    """No downloads or patches; isolate upstream's generic absolute import names.

    Use within a single-threaded worker. Save only state_dicts, never pickle these
    class objects under the upstream's temporary module names.
    """
    root = Path(root).resolve(strict=True)
    identity = verify_checkout(root, manifest_path)
    key = str(root)
    if key not in _CACHE:
        with _import_scope(root):
            base = importlib.import_module('models.base_models')
            encoders = importlib.import_module('models.encoders')
            data = importlib.import_module('utils.data_utils')
            config = importlib.import_module('config')
            _CACHE[key] = SimpleNamespace(BaseModel=base.BaseModel, LPModel=base.LPModel,
                                          HGCN=encoders.HGCN, data=data, parser=config.parser)
    api = _CACHE[key]
    return SimpleNamespace(**vars(api), identity=identity)

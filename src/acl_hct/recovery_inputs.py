"""Verified immutable bytes for unchanged legacy loader bytecode.

Dependency injection uses private function globals, never mutating old modules.
Large original paths are read/hashed once; subsequent parsing uses those bytes.
"""
import hashlib
import io
from pathlib import Path
from types import FunctionType

import numpy as np

from .backbone_frozen_inputs import load_train_prepared
from .encoder_matched_control import load_prepared
from .evaluate_checkpoint import load_verified_checkpoint


class VerifiedBytes(io.BytesIO):
    def __init__(self, raw, digest):
        super().__init__(raw)
        self.verified_sha256 = digest


class CachedPath:
    def __init__(self, cache, value):
        self.cache = cache
        self.value = Path(value.value if isinstance(value, CachedPath) else value)

    def __truediv__(self, value):
        return CachedPath(self.cache, self.value / value)

    def __fspath__(self):
        return str(self.value)

    @property
    def name(self):
        return self.value.name

    def resolve(self):
        # Legacy training-source inspection remains an ordinary, small read.
        return self.value.resolve()

    def read_bytes(self):
        return self.cache.bytes(self.value)

    def read_text(self, encoding='utf-8'):
        return self.read_bytes().decode(encoding)

    def open(self, mode='rb'):
        if mode != 'rb':
            raise ValueError('immutable binary read only')
        return VerifiedBytes(self.read_bytes(), self.cache.sha(self.value))


class NumpyCache:
    def __init__(self, cache):
        self.cache = cache

    def __getattr__(self, name):
        return getattr(np, name)

    def load(self, path, *, allow_pickle=False):
        if allow_pickle:
            raise ValueError('pickle inputs forbidden')
        return np.load(io.BytesIO(self.cache.bytes(path)), allow_pickle=False)


class HashCache:
    """Legacy raw-byte SHA calls reuse the already computed hash object.

    Canonical payloads and decoded array bytes still receive their own hashes.
    Identity matching is safe because cached immutable bytes remain alive.
    """
    def __init__(self, cache):
        self.cache = cache

    def __getattr__(self, name):
        return getattr(hashlib, name)

    def sha256(self, data=b''):
        for key, raw in self.cache.raw.items():
            if data is raw:
                return self.cache.hash_objects[key].copy()
        return hashlib.sha256(data)


class VerifiedInputs:
    def __init__(self):
        self.allowed = {}
        self.raw = {}
        self.hash_objects = {}
        self.receipts = {}

    def allow(self, path, expected, alias):
        key = Path(path).resolve()
        if key in self.allowed and self.allowed[key] == (expected, alias):
            return
        if key in self.allowed or alias in {item[1] for item in self.allowed.values()}:
            raise ValueError('duplicate input declaration')
        self.allowed[key] = (expected, alias)

    def bytes(self, path):
        key = Path(path).resolve()
        if key not in self.allowed:
            raise ValueError('input outside explicit worker allowlist')
        if key not in self.raw:
            expected, alias = self.allowed[key]
            raw = key.read_bytes()
            hasher = hashlib.sha256(raw)
            digest = hasher.hexdigest()
            if digest != expected:
                raise ValueError('original input raw SHA mismatch: ' + alias)
            self.raw[key] = raw
            self.hash_objects[key] = hasher
            self.receipts[alias] = {'raw_sha256': digest, 'original_path_reads': 1,
                                    'full_file_hash_checks': 1, 'decoded_from_verified_cached_bytes': True,
                                    'bytes': len(raw)}
        return self.raw[key]

    def sha(self, path):
        self.bytes(path)
        return self.allowed[Path(path).resolve()][0]

    def path(self, value):
        return CachedPath(self, value)


def _injected(function, **overrides):
    return FunctionType(function.__code__, {**function.__globals__, **overrides},
                        function.__name__, function.__defaults__, function.__closure__)


def prepared_from_cache(cache, root, config):
    train = _injected(load_train_prepared, Path=cache.path, np=NumpyCache(cache), hashlib=HashCache(cache))
    complete = _injected(load_prepared, Path=cache.path, file_sha256=cache.sha,
                         load_train_prepared=train, hashlib=HashCache(cache))
    return complete(root, config)


def checkpoint_from_cache(cache, path, expected, commit, training_release):
    def digest(stream):
        if not isinstance(stream, VerifiedBytes):
            raise ValueError('checkpoint must use verified cached bytes')
        return stream.verified_sha256
    loader = _injected(load_verified_checkpoint, Path=cache.path, _stream_sha256=digest)
    return loader(path, expected, commit, training_release)

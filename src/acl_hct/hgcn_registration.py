"""Stdlib-only binding for the engineering quality gate, not science release."""
import hashlib
import json
from pathlib import Path
import re
import subprocess


PROTOCOL = 'MATURE-HGCN-quality-v1'
CONFIG_SHA256 = 'a5fbaceaa1a735b224fa111153bbcffe56b6c9a8f8209ecdd97393b153322ec5'
_SOURCE_NAMES = ('__init__ aggregation backbone protocols geometry diagnostic_archive evaluate_checkpoint '
                 'mechanisms train ranking text_capacity backbone_frozen_inputs hgcn_upstream hgcn_sampling '
                 'mature_hgcn hgcn_fixture hgcn_registration hgcn_quality hgcn_entry').split()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_hashes():
    root = Path(__file__).parent
    return {'acl_hct/' + name + '.py': hashlib.sha256((root / (name + '.py')).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for name in _SOURCE_NAMES}


def validate_config(config):
    if config.get('protocol') != PROTOCOL or config.get('science_released') is not False:
        raise ValueError('engineering-only quality configuration required')
    value = canonical(config)
    if value != CONFIG_SHA256:
        raise ValueError('unregistered quality configuration')
    return value


def verify_release(config, phase, source_commit, record_path):
    config_hash = validate_config(config)
    if phase not in ('cpu_quality', 'cuda_quality'):
        raise ValueError('only registered engineering phases released')
    if not isinstance(source_commit, str) or not re.fullmatch('[0-9a-f]{40}', source_commit):
        raise ValueError('exact source commit required')
    raw = Path(record_path).read_bytes(); record = json.loads(raw)
    expected = {'protocol': PROTOCOL, 'accepted': True, 'authorized_scope': 'engineering_only',
                'phase': phase, 'source_commit': source_commit, 'config_sha256': config_hash,
                'upstream_commit': config['upstream_commit'], 'source_sha256_normalized_lf': source_hashes()}
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('supervisor quality release does not bind this phase/source/config')
    repo = Path(__file__).parents[2]
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True, timeout=5).strip()
    if head != source_commit:
        raise ValueError('execution checkout HEAD differs from reviewed source commit')
    return {'release_sha256': hashlib.sha256(raw).hexdigest(), **expected}

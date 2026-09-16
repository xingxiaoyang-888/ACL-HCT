"""Frozen E2 confirmation registration and development-only planning audit (stdlib)."""
import hashlib
import json
import math
from pathlib import Path

PROTOCOL = 'E2-independent-confirmation-v1'
BASE_SEED = 2026091702
FANOUTS = (4, 8, 16, 32, 64)
CONDITIONS = ('local_L1', 'F/S', 'S/F', 'S/S')
REGISTRATION_SHA256 = '04f625d9bb5d61402c8f8ad1440ab4de21fad7f73ce64abb112a1138080f4d12'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def child_path(directory, name):
    root = Path(directory).resolve()
    path = (root / name).resolve()
    if path.parent != root or path.name != name:
        raise ValueError('artifact must be a direct child of its directory')
    return path


def validate_config(config):
    # A changed design requires a new reviewed source/registration, never a CLI override.
    if digest(config) != REGISTRATION_SHA256:
        raise ValueError('fixed independent confirmation registration required')
    return REGISTRATION_SHA256


def shard(config, seed, fanout):
    validate_config(config)
    matches = [s for s in config['shards'] if (s['seed'], s['fanout']) == (seed, fanout)]
    if len(matches) != 1:
        raise ValueError('unregistered checkpoint/fanout')
    return matches[0]


def verify_development(config, index_path):
    validate_config(config)
    binding = config['development']
    if file_sha256(index_path) != binding['index_raw_sha256']:
        raise ValueError('development index hash mismatch')
    index = json.loads(Path(index_path).read_text(encoding='utf-8'))
    if any(index[k] != binding[k] for k in ('source_commit', 'config_sha256')):
        raise ValueError('development source/config mismatch')
    rows = {(r['seed'], r['fanout']): r for r in index['shards']}
    if len(rows) != 10 or len(index['shards']) != 10:
        raise ValueError('all ten development shards required')
    targets = config['inference']['halfwidth_targets']
    def count(se, n, family):
        if not math.isfinite(se) or se < 0:
            raise ValueError('invalid development variance')
        return math.ceil((1.96 * se * math.sqrt(n) / targets[family]) ** 2)
    for reference in binding['summaries']:
        key = reference['seed'], reference['fanout']
        row = rows[key]
        if any(row[k] != reference[k] for k in ('summary_file', 'summary_sha256')):
            raise ValueError('development summary identity mismatch')
        path = child_path(Path(index_path).parent, reference['summary_file'])
        if file_sha256(path) != reference['summary_sha256']:
            raise ValueError('development summary hash mismatch')
        planned = {'geometry': {c: count(row['geometry_groups'][c]['V']['projection']['mc_se'], 128, 'geometry')
                                for c in CONDITIONS},
                   'structure': {k: count(row['structure_V']['S/S'][f'{k}/V/score_change']['mc_se'], 128, 'structure')
                                 for k in ('direct', 'distant')},
                   'task': count(row['task_statistics']['mc_se'][1], 8, 'task')}
        if planned != shard(config, *key)['planning_required_counts']:
            raise ValueError('variance-only planning audit failed')
    return {'status': 'passed', 'index_raw_sha256': binding['index_raw_sha256'], 'summaries_verified': 10}

"""Separate, stdlib-only release binding for bounded mature-HGCN validation."""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from .hgcn_registration import canonical, source_hashes as quality_sources

PROTOCOL = 'MATURE-HGCN-validation-v1'
CONFIG_SHA256 = '4a3b003a39b6bfcb74957e74a62a5a2b0329735f1dda0ae622172fbc205a3edc'
NEW_SOURCES = ('development_view hgcn_geometry hgcn_panels hgcn_evidence hgcn_statistics '
               'hgcn_analysis hgcn_official hgcn_synthetic hgcn_validation_registration hgcn_validation_entry '
               'hgcn_validation hgcn_replay').split()
PHASES = ('cpu_fixture', 'cuda_fixture', 'train', 'replay', 'eval', 'official', 'analyze')


def source_hashes():
    root = Path(__file__).parent
    return {**quality_sources(), **{'acl_hct/' + n + '.py': hashlib.sha256(
        (root / (n + '.py')).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for n in NEW_SOURCES}}


def validate_config(config):
    if config.get('protocol') != PROTOCOL or canonical(config) != CONFIG_SHA256:
        raise ValueError('exact registered validation configuration required')
    if quality_sources() != config['quality_source_sha256_normalized_lf']:
        raise ValueError('previously accepted quality source19 changed')
    return CONFIG_SHA256


def role(config, phase, seed=None, repeat_start=None):
    r = config['resources']
    if phase not in PHASES:
        raise ValueError('unregistered phase')
    selectors = {}
    if phase in ('train', 'replay', 'eval'):
        if type(seed) is not int or seed not in config['training']['seeds']:
            raise ValueError('registered training seed required')
        selectors['seed'] = seed
    elif seed is not None:
        raise ValueError('seed selector only for WordNet train/replay/eval')
    if phase == 'eval':
        if type(repeat_start) is not int or not any(s['seed'] == seed and s['repeats'] == list(
                range(repeat_start, repeat_start + 4)) for s in config['sampling']['shards']):
            raise ValueError('exact registered four-repeat shard required')
        selectors['repeat_start'] = repeat_start
    elif repeat_start is not None:
        raise ValueError('repeat selector only for frozen shard')
    if phase == 'replay':
        return {'selectors': selectors, 'worker_seconds': 240, 'allocation_gpu_seconds': 300}
    key = {'cpu_fixture': 'cpu_fixture', 'cuda_fixture': 'science_fixture',
           'train': 'train', 'eval': 'eval', 'official': 'official', 'analyze': 'analysis'}[phase]
    gpu = phase not in ('cpu_fixture', 'analyze')
    return {'selectors': selectors, 'worker_seconds': r[key + '_worker_seconds'],
            'allocation_gpu_seconds': r[key + '_allocation_seconds'] if gpu else 0}


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def verify_release(config, phase, source_commit, record_path, seed=None, repeat_start=None, replay_policy=None):
    digest = validate_config(config); job = role(config, phase, seed, repeat_start)
    if not isinstance(source_commit, str) or not re.fullmatch('[0-9a-f]{40}', source_commit):
        raise ValueError('exact reviewed source commit required')
    raw = Path(record_path).read_bytes(); record = json.loads(raw)
    scope = 'engineering_only' if phase.endswith('fixture') else 'bounded_mature_hgcn_validation'
    expected = {'protocol': PROTOCOL, 'accepted': True, 'authorized_scope': scope,
                'phase': phase, 'source_commit': source_commit, 'config_sha256': digest,
                'upstream_commit': config['upstream_commit'], 'source_sha256_normalized_lf': source_hashes(), **job}
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('supervisor release does not bind exact phase/role/source/config')
    if phase in ('train', 'official'):
        raise ValueError('new scientific training is not authorized by the numerical replay amendment; reuse preserved originals')
    policy_digest = None
    if phase in ('replay', 'eval', 'analyze') or replay_policy is not None:
        if replay_policy is None:
            raise ValueError('separately frozen numerical replay policy required')
        from .hgcn_replay import validate_policy
        policy_digest = validate_policy(replay_policy, config)
        if record.get('replay_policy_sha256') != policy_digest:
            raise ValueError('release does not bind numerical replay policy')
    if phase in ('replay', 'eval', 'official', 'analyze'):
        for gate in ('cpu_fixture', 'cuda_fixture'):
            g = record.get('quality_gates', {}).get(gate, {})
            if (g.get('accepted') is not True or g.get('source_commit') != source_commit
                    or g.get('config_sha256') != digest or not _sha(g.get('report_sha256'))
                    or (policy_digest is not None and g.get('replay_policy_sha256') != policy_digest)):
                raise ValueError('both final-source original-runtime fixture acceptances required')
    inputs = record.get('inputs', {})
    required = {'replay': ('training_run_sha256', 'best_checkpoint_sha256'),
                'eval': ('training_run_sha256', 'best_checkpoint_sha256'),
                'analyze': ('artifact_index_sha256',)}.get(phase, ())
    if any(not _sha(inputs.get(k)) for k in required):
        raise ValueError('reviewed phase input hashes required')
    if phase == 'replay':
        origin = replay_policy['origins'][str(seed)]
        if any(inputs[k] != origin[k] for k in required):
            raise ValueError('replay must bind original frozen training lineage')
    baseline = record.get('baseline_acceptance', {})
    amendment = replay_policy.get('rank_sensitive_amendment') if replay_policy else None
    baseline_policy = amendment['v1_policy_sha256'] if amendment else policy_digest
    if phase == 'eval' and amendment and (
            inputs['training_run_sha256'] != amendment['v1_accepted_replays'][str(seed)]
            or baseline.get('amendment_policy_sha256') != policy_digest):
        raise ValueError('v2 release must bind exact old accepted replay and current amendment')
    if phase == 'eval' and (any(baseline.get(k) is not True for k in
            ('accepted', 'complete_4096_and_scheduled_valid_reviewed', 'matching_best_reload_reviewed',
             'learning_state_reviewed', 'full_hierarchy_reviewed', 'order_above_half_necessary_not_sufficient'))
            or baseline.get('training_run_sha256') != inputs['training_run_sha256']
            or baseline.get('best_checkpoint_sha256') != inputs['best_checkpoint_sha256']
            or baseline.get('replay_policy_sha256') != baseline_policy
            or baseline.get('original_training_run_sha256') != replay_policy['origins'][str(seed)]['training_run_sha256']
            or not _sha(baseline.get('review_sha256'))):
        raise ValueError('independent complete baseline/learning/full-hierarchy acceptance required before sampling')
    accounting = record.get('accounting', {})
    spent = accounting.get('spent_gpu_seconds'); reserved = accounting.get('reserved_including_this_gpu_seconds')
    concurrent = accounting.get('concurrent_gpu_jobs_including_this')
    if (type(spent) not in (int, float) or type(reserved) not in (int, float)
            or not math.isfinite(spent) or not math.isfinite(reserved)
            or spent < (replay_policy['resources']['minimum_spent_GPU_seconds'] if replay_policy else config['resources']['spent_quality_GPU_seconds'])
            or reserved < job['allocation_gpu_seconds']
            or spent + reserved > config['resources']['whole_round_GPU_seconds_limit']
            or type(concurrent) is not int or not 0 <= concurrent <= config['resources']['max_concurrency']
            or (job['allocation_gpu_seconds'] and concurrent < 1)):
        raise ValueError('bounded cumulative GPU reservation ledger required')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parents[2],
                                   text=True, timeout=5).strip()
    if head != source_commit:
        raise ValueError('execution HEAD differs from reviewed commit')
    return {'release_sha256': hashlib.sha256(raw).hexdigest(), **expected,
            'replay_policy_sha256': policy_digest,
            'inputs': inputs, 'quality_gates': record.get('quality_gates', {}),
            'baseline_acceptance': baseline, 'accounting': accounting}

"""Isolated registration, named random streams and Stage A decision rules."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from .hgcn_registration import canonical
from .hgcn_statistics import holm, student_summary
from .hgcn_validation_registration import source_hashes as original_source_hashes


PROTOCOL = 'R-TSC-HGCN-stage-A-v1'
PHASES = ('cpu_quality', 'cuda_quality', 'probe', 'train', 'final')
SOURCES = ('hgcn_tangent_correction', 'hgcn_rtsc_stage_a', 'hgcn_rtsc_protocol',
           'hgcn_rtsc_entry', 'hgcn_rtsc_worker')


def source_hashes():
    root = Path(__file__).parent
    return {**original_source_hashes(), **{'acl_hct/' + name + '.py': hashlib.sha256(
        (root / (name + '.py')).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for name in SOURCES}}


def validate_config(config, original, policy):
    """Reject local changes to scientific settings before any execution."""
    required = {
        'status': 'candidate_requires_independent_review_and_release',
        'protocol': PROTOCOL,
        'upstream_commit': original['upstream_commit'],
        'original_config_sha256': canonical(original),
        'replay_policy_sha256': canonical(policy),
        'seeds': [11, 23], 'original_best_steps': {'11': 4096, '23': 3584},
        'arms': {'task_only': 0.0, 'task_relation': 0.1},
        'module': {'hidden': 16, 'directions': 3, 'feature_count': 10,
                   'parameters_per_layer': 227, 'layers': 2, 'tau': 0.05,
                   'margin': 0.1, 'eps': 1e-8, 'chunk_edges': 4096},
        'training': {'steps': 1024, 'batch_positives': 128,
                     'negatives_per_positive': 4, 'fanout': 4, 'optimizer': 'Adam',
                     'lr': 0.003, 'weight_decay': 0.0, 'grad_clip_norm': 1.0,
                     'step_penalty_weight': 0.001},
        'selection': {'steps': [0, 256, 512, 768, 1024], 'repeats': 2,
                      'fanout': 4, 'direct_order_delta_floor': -0.002,
                      'rule': 'among qualified candidates first strict maximum mean full filtered micro MRR'},
        'final': {'repeats': 16, 'shard_repeats': 4, 'fanouts': [4, 8, 16], 'primary_family_size': 24,
                  'relation_contrast_family_size': 12, 'alpha': 0.05},
        'rng': {'protocol': PROTOCOL,
                'formula': 'lower 63 bits of SHA256 of canonical JSON [protocol, phase, seed, step, fanout, repeat, layer]',
                'streams': ['module_init', 'train_batch', 'train_layer', 'selection_layer', 'final_layer']},
        'ranking': {'candidate_chunk': 4096, 'max_seconds': 60},
    }
    if config != required or original['protocol'] != 'MATURE-HGCN-validation-v1' or policy['protocol'] != 'MATURE-HGCN-numeric-replay-v2':
        raise ValueError('exact separate reviewed Stage A configuration and original lineage required')
    return canonical(config)


def verify_release(config, original, policy, phase, record_path, source_commit,
                   *, seed=None, arm=None, repeat_start=None, original_run=None, task_run=None,
                   relation_run=None):
    """Bind each executable phase to an independently accepted exact release."""
    from .hgcn_evidence import file_hash
    digest = validate_config(config, original, policy)
    if phase not in PHASES or not isinstance(source_commit, str) or not re.fullmatch('[0-9a-f]{40}', source_commit):
        raise ValueError('registered phase and reviewed source commit required')
    selector = {'seed': seed, 'arm': arm, 'repeat_start': repeat_start}
    if phase in ('probe', 'train', 'final'):
        if seed not in config['seeds'] or (phase != 'final' and arm not in config['arms']) or (phase == 'final' and arm is not None):
            raise ValueError('registered model seed and arm selector required')
    elif seed is not None or arm is not None:
        raise ValueError('quality phase cannot choose training seed/arm')
    if (phase == 'final' and repeat_start not in (0, 4, 8, 12)
            or phase != 'final' and repeat_start is not None):
        raise ValueError('exact four-repeat final shard selector required')
    raw = Path(record_path).read_bytes(); record = json.loads(raw)
    expected = {'protocol': PROTOCOL, 'accepted': True, 'phase': phase,
                'source_commit': source_commit, 'source_sha256_normalized_lf': source_hashes(),
                'config_sha256': digest, 'upstream_commit': config['upstream_commit'],
                'selector': selector}
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('separate release does not bind phase/source/config/selector')
    seconds = record.get('worker_seconds')
    if type(seconds) is not int or not 1 <= seconds <= 14400:
        raise ValueError('positive bounded worker deadline required')
    if phase != 'cpu_quality':
        required_gates = ('cpu_quality',) if phase == 'cuda_quality' else ('cpu_quality', 'cuda_quality')
        for gate in required_gates:
            value = record.get('quality_gates', {}).get(gate, {})
            if (value.get('accepted') is not True or value.get('source_commit') != source_commit
                    or value.get('config_sha256') != digest
                    or not isinstance(value.get('report_sha256'), str)
                    or not re.fullmatch('[0-9a-f]{64}', value['report_sha256'])):
                raise ValueError('same-source CPU/CUDA quality acceptance required')
    if phase in ('probe', 'train', 'final'):
        if original_run is None or file_hash(original_run) != policy['origins'][str(seed)]['training_run_sha256']:
            raise ValueError('exact original failed training lineage required')
        inputs = record.get('inputs', {})
        if (inputs.get('original_training_run_sha256') != file_hash(original_run)
                or inputs.get('original_best_checkpoint_sha256') != policy['origins'][str(seed)]['best_checkpoint_sha256']):
            raise ValueError('release original checkpoint binding mismatch')
        accepted = record.get('baseline_acceptance', {})
        if (accepted.get('accepted') is not True
                or accepted.get('v2_policy_sha256') != canonical(policy)
                or accepted.get('original_training_run_sha256') != file_hash(original_run)
                or accepted.get('original_best_checkpoint_sha256') != policy['origins'][str(seed)]['best_checkpoint_sha256']
                or not isinstance(accepted.get('review_sha256'), str)
                or not re.fullmatch('[0-9a-f]{64}', accepted['review_sha256'])):
            raise ValueError('independently accepted original HGCN baseline required')
        if phase == 'final':
            if task_run is None or relation_run is None or any(
                inputs.get(key) != file_hash(path) for key, path in
                (('task_result_sha256', task_run), ('relation_result_sha256', relation_run))):
                raise ValueError('both selected complete training arm inputs required')
    if phase in ('cuda_quality', 'probe', 'train', 'final'):
        accounting = record.get('accounting', {})
        if (type(accounting.get('reserved_gpu_seconds')) is not int
                or accounting['reserved_gpu_seconds'] < seconds
                or type(accounting.get('concurrent_jobs_including_this')) is not int
                or not 1 <= accounting['concurrent_jobs_including_this'] <= 3):
            raise ValueError('bounded GPU accounting and concurrency required')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                   cwd=Path(__file__).parents[2], text=True, timeout=5).strip()
    if head != source_commit:
        raise ValueError('execution HEAD differs from reviewed source')
    return {'release_sha256': hashlib.sha256(raw).hexdigest(),
            'worker_seconds': seconds, 'source_sha256_normalized_lf': expected['source_sha256_normalized_lf']}


class NamedStreams:
    def __init__(self, protocol=PROTOCOL):
        if protocol != PROTOCOL:
            raise ValueError('registered random stream protocol required')
        self.seen = {}

    def seed(self, phase, seed, step=None, fanout=None, repeat=None, layer=None):
        if phase not in ('module_init', 'train_batch', 'train_layer',
                         'selection_layer', 'final_layer') or seed not in (11, 23):
            raise ValueError('registered stream name and model seed required')
        fields = [PROTOCOL, phase, seed, step, fanout, repeat, layer]
        raw = json.dumps(fields, separators=(',', ':'), ensure_ascii=True).encode('ascii')
        value = int.from_bytes(hashlib.sha256(raw).digest()[-8:], 'big') & ((1 << 63) - 1)
        key = tuple(fields)
        if value in self.seen and self.seen[value] != key:
            raise ValueError('named random stream collision')
        self.seen[value] = key
        return value

    def manifest(self):
        return [{'seed63': value, 'fields': list(fields)} for value, fields in sorted(self.seen.items())]


def choose_checkpoint(rows, floor=-0.002):
    """Fixed two-repeat mean direct-order gate, then first strict MRR max."""
    if [row['step'] for row in rows] != [0, 256, 512, 768, 1024]:
        raise ValueError('all five scheduled checkpoints required in order')
    best = None
    for row in rows:
        differences = np.asarray(row['direct_order_delta'], dtype=np.float64)
        mrr = np.asarray(row['micro_mrr'], dtype=np.float64)
        if differences.shape != (2,) or mrr.shape != (2,) or not (np.isfinite(differences).all() and np.isfinite(mrr).all()):
            raise ValueError('two complete finite paired development repeats required')
        if float(differences.mean()) >= floor and (best is None or float(mrr.mean()) > best['mean_micro_mrr']):
            best = {'step': row['step'], 'mean_micro_mrr': float(mrr.mean()),
                    'mean_direct_order_delta': float(differences.mean())}
    return best


def final_families(records, seeds=(11, 23), fanouts=(4, 8, 16), repeats=16):
    """Compute paired graph-repeat Student contrasts in fixed Holm24/Holm12 families."""
    keys = {(seed, fanout, repeat, arm) for seed in seeds for fanout in fanouts
            for repeat in range(repeats) for arm in ('S', 'task_only', 'task_relation')}
    if set(records) != keys:
        raise ValueError('complete paired S/task/relation final grid required')
    primary, secondary = [], []
    for seed in seeds:
        for fanout in fanouts:
            for metric in ('micro_mrr', 'direct_order'):
                for arm in ('task_only', 'task_relation'):
                    values = np.array([records[seed, fanout, r, arm][metric] -
                                       records[seed, fanout, r, 'S'][metric]
                                       for r in range(repeats)], dtype=np.float64)
                    primary.append({'comparison': f'seed{seed}-f{fanout}-{arm}-S-{metric}',
                                    'seed': seed, 'fanout': fanout, 'arm': arm,
                                    'metric': metric, 'contrast': f'{arm}-S', **student_summary(values)})
                values = np.array([records[seed, fanout, r, 'task_relation'][metric] -
                                   records[seed, fanout, r, 'task_only'][metric]
                                   for r in range(repeats)], dtype=np.float64)
                secondary.append({'comparison': f'seed{seed}-f{fanout}-relation-task-{metric}',
                                  'seed': seed, 'fanout': fanout, 'metric': metric,
                                  'contrast': 'task_relation-task_only', **student_summary(values)})
    return holm(primary, family_size=24), holm(secondary, family_size=12)


def analyze_final_shards(shards):
    """Accept exactly eight complete four-repeat shards and state the fixed rule."""
    expected = {(seed, start) for seed in (11, 23) for start in (0, 4, 8, 12)}
    if len(shards) != len(expected):
        raise ValueError('exactly eight Stage A final shards required')
    records = {}
    fixed = {}
    selected = {}
    observed = set()
    for shard in shards:
        key = (shard.get('seed'), shard.get('repeat_start'))
        if (shard.get('status') != 'complete' or key not in expected or key in observed
                or shard.get('base_state_unchanged') is not True
                or shard.get('original_F_qualification', {}).get('accepted') is not True
                or any(shard.get('full_arm_qualifications', {}).get(arm, {}).get('accepted') is not True
                       for arm in ('task_only', 'task_relation'))):
            raise ValueError('complete independent frozen-F-qualified shard required')
        observed.add(key)
        seed, start = key
        if seed in fixed and fixed[seed] != shard['fixed_F']:
            raise ValueError('fixed F changed between shards')
        fixed[seed] = shard['fixed_F']
        if set(shard.get('selected', {})) != {'task_only', 'task_relation'}:
            raise ValueError('both selected module checkpoints required')
        if seed in selected and selected[seed] != shard['selected']:
            raise ValueError('selected module checkpoint changed between shards')
        selected[seed] = shard['selected']
        if len(shard.get('rows', [])) != 36:
            raise ValueError('all 36 shard condition records required')
        for row in shard['rows']:
            name = (row['seed'], row['fanout'], row['repeat'], row['arm'])
            if (row['seed'] != seed or row['fanout'] not in (4, 8, 16)
                    or not start <= row['repeat'] < start + 4
                    or row['arm'] not in ('S', 'task_only', 'task_relation')
                    or name in records):
                raise ValueError('duplicate or out-of-shard final record')
            records[name] = {'micro_mrr': row['micro_mrr'],
                             'direct_order': row['direct_order']}
    if observed != expected:
        raise ValueError('all fixed model/repeat shards required')
    primary, secondary = final_families(records)
    joint = []
    recovery = []
    for seed in (11, 23):
        for metric in ('micro_mrr', 'direct_order'):
            row = next(r for r in primary if r['seed'] == seed and r['fanout'] == 4
                       and r['arm'] == 'task_relation' and r['metric'] == metric)
            joint.append(row['mean'] > 0 and row['reject_holm'])
            sample_mean = float(np.mean([records[seed, 4, r, 'S'][metric] for r in range(16)]))
            damage = fixed[seed][metric] - sample_mean
            recovery.append({'seed': seed, 'metric': metric, 'fixed_F_minus_mean_S': damage,
                             'C_relation_minus_S': row['mean'],
                             'fraction_of_observed_loss': row['mean'] / damage if damage > 0 else None,
                             'meets_20pct_development_target': row['mean'] >= .2 * damage if damage > 0 else None})
    return {'status': 'complete', 'scope': 'two fixed checkpoints and 16 paired graph repeats each',
            'primary_holm24': primary, 'relation_vs_task_holm12': secondary,
            'joint_recovery_rule_met': all(joint), 'recovery': recovery,
            'fixed_F': fixed, 'selected': selected}

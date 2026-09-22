"""Exact registration, mixed streams and predeclared amplitude contrast families."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from scipy.stats import t as student

from .hgcn_registration import canonical
from .hgcn_rtsc_protocol import (SOURCES as OLD_SOURCES,
                                  source_hashes as old_source_hashes,
                                  choose_checkpoint)
from .hgcn_statistics import holm, student_summary


PROTOCOL = 'R-TSC-HGCN-amplitude-v1'
OLD_PROTOCOL = 'R-TSC-HGCN-stage-A-v1'
PHASES = ('cpu_quality', 'cuda_quality', 'probe', 'train', 'final')
SOURCES = ('hgcn_tangent_amplitude', 'hgcn_rtsc_amplitude',
           'hgcn_rtsc_amplitude_protocol', 'hgcn_rtsc_amplitude_entry',
           'hgcn_rtsc_amplitude_worker')
CONDITIONS = ('S', 'old_task_only', 'old_task_relation',
              'new_task_only', 'new_task_relation')


def source_hashes():
    root = Path(__file__).parent
    return {**old_source_hashes(), **{'acl_hct/' + name + '.py': hashlib.sha256(
        (root / (name + '.py')).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for name in SOURCES}}


def validate_config(config, original, policy, old):
    required = {
        'status': 'candidate_requires_independent_review_and_release',
        'protocol': PROTOCOL, 'old_protocol': OLD_PROTOCOL,
        'old_source_commit': 'aadbecaa06b40da2f27042473eadf852290f92bd',
        'old_config_sha256': canonical(old),
        'upstream_commit': original['upstream_commit'],
        'original_config_sha256': canonical(original),
        'replay_policy_sha256': canonical(policy),
        'seeds': [11, 23], 'original_best_steps': {'11': 4096, '23': 3584},
        'arms': {'task_only': 0.0, 'task_relation': 0.1},
        'module': {'hidden': 16, 'directions': 3, 'feature_count': 10,
                   'parameters_per_layer': 227, 'layers': 2, 'tau': 0.05,
                   'margin': 0.1, 'eps': 1e-8, 'scale': 4, 'chunk_edges': 4096,
                   'penalty': 'mean physical raw norm squared over tau squared'},
        'training': old['training'], 'selection': old['selection'],
        'final': {'repeats': 16, 'shard_repeats': 4, 'fanouts': [4, 8, 16],
                  'conditions': list(CONDITIONS), 'main_family_size': 24,
                  'support_family_size': 24, 'relation_family_size': 12,
                  'hierarchy_noninferiority_margin': 0.002, 'alpha': 0.05},
        'rng': {'train_selection_protocol': OLD_PROTOCOL,
                'final_protocol': PROTOCOL,
                'formula': 'lower 63 bits of SHA256 of canonical JSON [protocol, phase, seed, step, fanout, repeat, layer]',
                'streams': ['module_init', 'train_batch', 'train_layer',
                            'selection_layer', 'final_layer']},
        'ranking': old['ranking']}
    if (config != required or old['protocol'] != OLD_PROTOCOL
            or canonical(old) != '4047865a86d6dcf85939cfb40f0e34546156445c192e699f88ee7eef079a13fd'
            or original['protocol'] != 'MATURE-HGCN-validation-v1'
            or policy['protocol'] != 'MATURE-HGCN-numeric-replay-v2'):
        raise ValueError('exact separate reviewed amplitude and old lineage required')
    return canonical(config)


class MixedStreams:
    def __init__(self):
        self.seen = {}

    def seed(self, phase, seed, step=None, fanout=None, repeat=None, layer=None):
        if phase not in ('module_init', 'train_batch', 'train_layer',
                         'selection_layer', 'final_layer') or seed not in (11, 23):
            raise ValueError('registered stream and fixed model seed required')
        protocol = PROTOCOL if phase == 'final_layer' else OLD_PROTOCOL
        fields = [protocol, phase, seed, step, fanout, repeat, layer]
        raw = json.dumps(fields, separators=(',', ':'), ensure_ascii=True).encode('ascii')
        value = int.from_bytes(hashlib.sha256(raw).digest()[-8:], 'big') & ((1 << 63) - 1)
        key = tuple(fields)
        if value in self.seen and self.seen[value] != key:
            raise ValueError('mixed random stream collision')
        self.seen[value] = key
        return value

    def manifest(self):
        return [{'seed63': value, 'fields': list(fields)}
                for value, fields in sorted(self.seen.items())]


def one_sided_summary(values, *, margin=0.):
    """Greater-than test; report unshifted effect and its marginal 95% interval."""
    values = np.asarray(values, dtype=np.float64)
    summary = student_summary(values)
    if summary['t'] is None:
        summary['p'] = 1.
        return summary
    statistic = (summary['mean'] + margin) / summary['SE']
    summary['t'] = float(statistic)
    summary['p'] = float(student.sf(statistic, summary['df']))
    return summary


def final_families(records):
    expected = {(seed, fanout, repeat, arm)
                for seed in (11, 23) for fanout in (4, 8, 16)
                for repeat in range(16) for arm in CONDITIONS}
    if set(records) != expected:
        raise ValueError('complete five-condition paired final grid required')
    main, support, relation = [], [], []
    for seed in (11, 23):
        for fanout in (4, 8, 16):
            for arm in ('task_only', 'task_relation'):
                old_arm, new_arm = 'old_' + arm, 'new_' + arm
                for metric in ('micro_mrr', 'direct_order'):
                    old_new = np.asarray([records[seed, fanout, r, new_arm][metric] -
                                          records[seed, fanout, r, old_arm][metric]
                                          for r in range(16)], dtype=np.float64)
                    new_sample = np.asarray([records[seed, fanout, r, new_arm][metric] -
                                             records[seed, fanout, r, 'S'][metric]
                                             for r in range(16)], dtype=np.float64)
                    margin = .002 if metric == 'direct_order' else 0.
                    main.append({'comparison': f'seed{seed}-f{fanout}-{arm}-new-old-{metric}',
                                 'seed': seed, 'fanout': fanout, 'arm': arm,
                                 'metric': metric, 'contrast': 'new-old',
                                 'test': 'one-sided hierarchy noninferiority' if margin else 'one-sided MRR superiority',
                                 'noninferiority_margin': margin,
                                 **one_sided_summary(old_new, margin=margin)})
                    support.append({'comparison': f'seed{seed}-f{fanout}-{arm}-new-S-{metric}',
                                    'seed': seed, 'fanout': fanout, 'arm': arm,
                                    'metric': metric, 'contrast': 'new-S',
                                    'test': 'two-sided improvement requires positive mean',
                                    **student_summary(new_sample)})
            for metric in ('micro_mrr', 'direct_order'):
                values = np.asarray([records[seed, fanout, r, 'new_task_relation'][metric] -
                                     records[seed, fanout, r, 'new_task_only'][metric]
                                     for r in range(16)], dtype=np.float64)
                relation.append({'comparison': f'seed{seed}-f{fanout}-new-relation-task-{metric}',
                                 'seed': seed, 'fanout': fanout, 'metric': metric,
                                 'contrast': 'new_task_relation-new_task_only',
                                 **student_summary(values)})
    return holm(main, family_size=24), holm(support, family_size=24), holm(relation, family_size=12)


def analyze_final_shards(shards):
    expected_shards = {(seed, start) for seed in (11, 23) for start in (0, 4, 8, 12)}
    if len(shards) != 8:
        raise ValueError('exactly eight final shards required')
    records, selected, fixed, observed = {}, {}, {}, set()
    for shard in shards:
        key = (shard.get('seed'), shard.get('repeat_start'))
        if (shard.get('status') != 'complete' or key not in expected_shards or key in observed
                or shard.get('base_state_unchanged') is not True
                or shard.get('original_F_qualification', {}).get('accepted') is not True
                or set(shard.get('full_arm_qualifications', {})) != set(CONDITIONS[1:])
                or any(not v.get('accepted') for v in shard['full_arm_qualifications'].values())):
            raise ValueError('complete independently qualified final shard required')
        observed.add(key)
        seed, start = key
        if seed in fixed and fixed[seed] != shard['fixed_F']:
            raise ValueError('fixed F changed between shards')
        fixed[seed] = shard['fixed_F']
        if set(shard.get('selected', {})) != set(CONDITIONS[1:]):
            raise ValueError('four selected module checkpoints required')
        if seed in selected and selected[seed] != shard['selected']:
            raise ValueError('selected checkpoints changed between shards')
        selected[seed] = shard['selected']
        if len(shard.get('rows', [])) != 60:
            raise ValueError('all 60 shard condition records required')
        for row in shard['rows']:
            name = (row['seed'], row['fanout'], row['repeat'], row['arm'])
            if (row['seed'] != seed or row['fanout'] not in (4, 8, 16)
                    or not start <= row['repeat'] < start + 4
                    or row['arm'] not in CONDITIONS or name in records):
                raise ValueError('duplicate or out-of-shard final record')
            records[name] = {'micro_mrr': row['micro_mrr'],
                             'direct_order': row['direct_order']}
    if observed != expected_shards:
        raise ValueError('all fixed model/repeat shards required')
    main, support, relation = final_families(records)
    arm_rules = {}
    for arm in ('task_only', 'task_relation'):
        rule = []
        for seed in (11, 23):
            for metric in ('micro_mrr', 'direct_order'):
                primary = next(r for r in main if r['seed'] == seed and r['fanout'] == 4
                               and r['arm'] == arm and r['metric'] == metric)
                secondary = next(r for r in support if r['seed'] == seed and r['fanout'] == 4
                                 and r['arm'] == arm and r['metric'] == metric)
                rule.extend((primary['reject_holm'], secondary['reject_holm'] and secondary['mean'] > 0))
        arm_rules[arm] = all(rule)
    recovery = []
    for seed in (11, 23):
        for fanout in (4, 8, 16):
            for metric in ('micro_mrr', 'direct_order'):
                sample_mean = float(np.mean([records[seed, fanout, r, 'S'][metric]
                                             for r in range(16)]))
                damage = fixed[seed][metric] - sample_mean
                for arm in CONDITIONS[1:]:
                    corrected = float(np.mean([records[seed, fanout, r, arm][metric]
                                               for r in range(16)]))
                    improvement = corrected - sample_mean
                    recovery.append({'seed': seed, 'fanout': fanout, 'arm': arm,
                                     'metric': metric, 'fixed_F_minus_mean_S': damage,
                                     'C_minus_S': improvement,
                                     'fraction_of_observed_loss': improvement / damage
                                     if damage > 0 else None,
                                     'meets_20pct_development_target': improvement >= .2 * damage
                                     if damage > 0 else None})
    return {'status': 'complete', 'scope': 'two fixed checkpoints and 16 new paired graph repeats each',
            'main_new_vs_old_holm24': main, 'support_new_vs_S_holm24': support,
            'new_relation_vs_task_holm12': relation,
            'method_revision_rule_met': any(arm_rules.values()), 'arm_rules': arm_rules,
            'recovery': recovery, 'fixed_F': fixed, 'selected': selected}


def verify_release(config, original, policy, old, phase, record_path, source_commit,
                   *, seed=None, arm=None, repeat_start=None, original_run=None,
                   new_task_run=None, new_relation_run=None,
                   old_task_run=None, old_relation_run=None):
    from .hgcn_evidence import file_hash

    digest = validate_config(config, original, policy, old)
    root = Path(__file__).parents[2]
    for name in OLD_SOURCES:
        relative = 'src/acl_hct/' + name + '.py'
        archived = subprocess.check_output(
            ['git', 'show', config['old_source_commit'] + ':' + relative],
            cwd=root, timeout=5).replace(b'\r\n', b'\n')
        current = (root / relative).read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(current).digest() != hashlib.sha256(archived).digest():
            raise ValueError('Stage A source differs from accepted old module behavior')
    if phase not in PHASES or not isinstance(source_commit, str) or not re.fullmatch('[0-9a-f]{40}', source_commit):
        raise ValueError('registered phase and reviewed source commit required')
    selector = {'seed': seed, 'arm': arm, 'repeat_start': repeat_start}
    if phase in ('probe', 'train', 'final'):
        if seed not in (11, 23) or (phase != 'final' and arm not in config['arms']) or (phase == 'final' and arm is not None):
            raise ValueError('registered fixed seed/arm selector required')
    elif seed is not None or arm is not None:
        raise ValueError('quality phase cannot select seed/arm')
    if (phase == 'final' and repeat_start not in (0, 4, 8, 12)
            or phase != 'final' and repeat_start is not None):
        raise ValueError('exact final shard selector required')
    raw = Path(record_path).read_bytes(); record = json.loads(raw)
    expected = {'protocol': PROTOCOL, 'accepted': True, 'phase': phase,
                'source_commit': source_commit,
                'source_sha256_normalized_lf': source_hashes(),
                'config_sha256': digest, 'upstream_commit': config['upstream_commit'],
                'selector': selector}
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('amplitude release does not bind phase/source/config/selector')
    seconds = record.get('worker_seconds')
    if type(seconds) is not int or not 1 <= seconds <= 14400:
        raise ValueError('positive bounded worker deadline required')
    if phase != 'cpu_quality':
        for gate in (('cpu_quality',) if phase == 'cuda_quality' else ('cpu_quality', 'cuda_quality')):
            value = record.get('quality_gates', {}).get(gate, {})
            if (value.get('accepted') is not True or value.get('source_commit') != source_commit
                    or value.get('config_sha256') != digest
                    or not re.fullmatch('[0-9a-f]{64}', str(value.get('report_sha256')))):
                raise ValueError('same-source CPU/CUDA quality acceptance required')
    if phase in ('probe', 'train', 'final'):
        if original_run is None or file_hash(original_run) != policy['origins'][str(seed)]['training_run_sha256']:
            raise ValueError('exact original training lineage required')
        inputs = record.get('inputs', {})
        if (inputs.get('original_training_run_sha256') != file_hash(original_run)
                or inputs.get('original_best_checkpoint_sha256') != policy['origins'][str(seed)]['best_checkpoint_sha256']):
            raise ValueError('released original HGCN binding required')
        accepted = record.get('baseline_acceptance', {})
        if (accepted.get('accepted') is not True
                or accepted.get('v2_policy_sha256') != canonical(policy)
                or accepted.get('original_training_run_sha256') != file_hash(original_run)
                or accepted.get('original_best_checkpoint_sha256') != policy['origins'][str(seed)]['best_checkpoint_sha256']
                or not re.fullmatch('[0-9a-f]{64}', str(accepted.get('review_sha256')))):
            raise ValueError('independently accepted original frozen HGCN required')
        if phase == 'final':
            paths = {'new_task_result_sha256': new_task_run,
                     'new_relation_result_sha256': new_relation_run,
                     'old_task_result_sha256': old_task_run,
                     'old_relation_result_sha256': old_relation_run}
            if any(p is None or inputs.get(k) != file_hash(p) for k, p in paths.items()):
                raise ValueError('four exact selected arm inputs required')
    if phase != 'cpu_quality':
        accounting = record.get('accounting', {})
        if (type(accounting.get('reserved_gpu_seconds')) is not int
                or accounting['reserved_gpu_seconds'] < seconds
                or type(accounting.get('concurrent_jobs_including_this')) is not int
                or not 1 <= accounting['concurrent_jobs_including_this'] <= 3):
            raise ValueError('bounded GPU accounting and concurrency required')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                   cwd=root, text=True, timeout=5).strip()
    if head != source_commit:
        raise ValueError('execution HEAD differs from reviewed source')
    return {'release_sha256': hashlib.sha256(raw).hexdigest(),
            'worker_seconds': seconds,
            'source_sha256_normalized_lf': expected['source_sha256_normalized_lf']}

"""Exact registration and inference for matched HGCN joint training."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from scipy.stats import t as student

from .hgcn_registration import canonical
from .hgcn_rtsc_protocol import SOURCES as OLD_SOURCES
from .hgcn_rtsc_protocol import source_hashes as old_source_hashes
from .hgcn_statistics import holm, student_summary


PROTOCOL = 'R-TSC-HGCN-joint-v1'
PHASES = ('cpu_quality', 'cuda_quality', 'probe', 'train', 'final', 'benchmark')
ARMS = ('plain_task', 'plain_relation', 'three_task', 'three_relation',
        'single_task', 'single_relation')
STRUCTURES = ('plain', 'three', 'single')
METRICS = ('micro_mrr', 'direct_order')
SOURCES = ('hgcn_rtsc_joint', 'hgcn_rtsc_joint_protocol',
           'hgcn_rtsc_joint_entry', 'hgcn_rtsc_joint_worker')
OLD_SOURCE_COMMIT = 'aadbecaa06b40da2f27042473eadf852290f92bd'
OLD_CONFIG_SHA256 = '4047865a86d6dcf85939cfb40f0e34546156445c192e699f88ee7eef079a13fd'
DOCUMENT_SHA256 = 'b140bd63f73e9b418de5cab7f9a06d8f08dddaa665b8957e0b5b1bc27f4261a8'


def source_hashes():
    root = Path(__file__).parent
    return {**old_source_hashes(), **{
        'acl_hct/' + name + '.py': hashlib.sha256(
            (root / (name + '.py')).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for name in SOURCES}}


def validate_config(config, original, policy, old):
    required = {
        'status': 'candidate_requires_independent_review_and_release',
        'protocol': PROTOCOL, 'protocol_document_sha256': DOCUMENT_SHA256,
        'old_source_commit': OLD_SOURCE_COMMIT,
        'old_config_sha256': OLD_CONFIG_SHA256,
        'upstream_commit': original['upstream_commit'],
        'original_config_sha256': canonical(original),
        'replay_policy_sha256': canonical(policy),
        'seeds': [11, 23], 'original_best_steps': {'11': 4096, '23': 3584},
        'arms': {
            'plain_task': {'structure': 'plain', 'relation_weight': 0.},
            'plain_relation': {'structure': 'plain', 'relation_weight': .1},
            'three_task': {'structure': 'three', 'relation_weight': 0.},
            'three_relation': {'structure': 'three', 'relation_weight': .1},
            'single_task': {'structure': 'single', 'relation_weight': 0.},
            'single_relation': {'structure': 'single', 'relation_weight': .1}},
        'module': {'hidden': 16, 'feature_count': 10,
                   'three_parameters_per_layer': 227,
                   'single_parameters_per_layer': 193, 'layers': 2,
                   'tau': .05, 'margin': .1, 'eps': 1e-8,
                   'chunk_edges': 4096,
                   'penalty': 'mean clipped physical step norm squared over tau squared'},
        'training': {'steps': 1024, 'batch_positives': 128,
                     'negatives_per_positive': 4, 'fanout': 4,
                     'optimizer': 'Adam', 'betas': [.9, .999], 'eps': 1e-8,
                     'base_lr': .0003, 'module_lr': .003,
                     'weight_decay': 0., 'grad_clip_norm': 1.,
                     'step_penalty_weight': .001},
        'selection': {
            'steps': [0, 256, 512, 768, 1024], 'repeats': 2, 'fanout': 4,
            'direct_order_from_common_S0_floor': -.002,
            'rule': 'among candidates above common initial S0 direct-order mean minus 0.002, first strict maximum mean complete filtered micro MRR'},
        'final': {'repeats': 16, 'shard_repeats': 4, 'fanouts': [4, 8, 16],
                  'conditions': list(ARMS), 'main_family_size': 16,
                  'protection_family_size': 32, 'relation_family_size': 36,
                  'mrr_protection_margin': .0005,
                  'hierarchy_noninferiority_margin': .002, 'alpha': .05},
        'benchmark': {'fanout': 4, 'batch_positives': 128, 'warmups': 5,
                      'repetitions': 20, 'order': 'interleaved',
                      'inference_overhead_limit': .1,
                      'training_overhead_limit': .15,
                      'memory_overhead_limit': .1,
                      'module_parameters_limit': 1000},
        'rng': {'protocol': PROTOCOL,
                'formula': 'lower 63 bits of SHA256 of canonical JSON [protocol, phase, seed, step, fanout, repeat, layer]',
                'streams': ['module_init', 'train_batch', 'train_layer',
                            'selection_layer', 'final_layer', 'benchmark']},
        'ranking': old['ranking']}
    if (config != required or canonical(old) != OLD_CONFIG_SHA256
            or original['protocol'] != 'MATURE-HGCN-validation-v1'
            or policy['protocol'] != 'MATURE-HGCN-numeric-replay-v2'
            or canonical(original) != config['original_config_sha256']
            or canonical(policy) != config['replay_policy_sha256']):
        raise ValueError('exact separately registered joint configuration and lineage required')
    return canonical(config)


class NamedStreams:
    def __init__(self):
        self.seen = {}

    def seed(self, phase, seed, step=None, fanout=None, repeat=None, layer=None):
        if phase not in ('module_init', 'train_batch', 'train_layer',
                         'selection_layer', 'final_layer', 'benchmark') or seed not in (11, 23):
            raise ValueError('registered joint random stream and seed required')
        fields = [PROTOCOL, phase, seed, step, fanout, repeat, layer]
        raw = json.dumps(fields, separators=(',', ':'), ensure_ascii=True).encode('ascii')
        value = int.from_bytes(hashlib.sha256(raw).digest()[-8:], 'big') & ((1 << 63) - 1)
        key = tuple(fields)
        if value in self.seen and self.seen[value] != key:
            raise ValueError('joint named stream collision')
        self.seen[value] = key
        return value

    def manifest(self):
        return [{'seed63': value, 'fields': list(fields)}
                for value, fields in sorted(self.seen.items())]


def choose_checkpoint(rows, common_s0_direct_order, floor=-.002):
    if [row.get('step') for row in rows] != [0, 256, 512, 768, 1024]:
        raise ValueError('all five joint candidate checkpoints required')
    if not np.isfinite(common_s0_direct_order):
        raise ValueError('finite shared initial S0 gate required')
    best = None
    for row in rows:
        mrr = np.asarray(row.get('micro_mrr'), dtype=np.float64)
        order = np.asarray(row.get('direct_order'), dtype=np.float64)
        if (mrr.shape != (2,) or order.shape != (2,)
                or not np.isfinite(mrr).all() or not np.isfinite(order).all()):
            raise ValueError('two complete finite matched selection graphs required')
        mean_order, mean_mrr = float(order.mean()), float(mrr.mean())
        if mean_order >= common_s0_direct_order + floor and (
                best is None or mean_mrr > best['mean_micro_mrr']):
            best = {'step': row['step'], 'mean_micro_mrr': mean_mrr,
                    'mean_direct_order': mean_order,
                    'common_S0_direct_order': common_s0_direct_order,
                    'direct_order_gate': common_s0_direct_order + floor}
    return best


def one_sided(values, margin=0.):
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
                for repeat in range(16) for arm in ARMS}
    if set(records) != expected:
        raise ValueError('complete paired six-arm 576-condition grid required')
    if any(set(value) != set(METRICS) or not all(np.isfinite(value[m]) for m in METRICS)
           for value in records.values()):
        raise ValueError('finite two-metric record required')
    main, protection, relation = [], [], []

    def differences(seed, fanout, first, second, metric):
        return np.asarray([records[seed, fanout, r, first][metric]
                           - records[seed, fanout, r, second][metric]
                           for r in range(16)], dtype=np.float64)

    for seed in (11, 23):
        for loss in ('task', 'relation'):
            for contrast, second in (('three-plain', 'plain'),
                                     ('three-single', 'single')):
                for metric in METRICS:
                    margin = .002 if contrast == 'three-single' and metric == 'direct_order' else 0.
                    main.append({
                        'comparison': f'seed{seed}-f4-{loss}-{contrast}-{metric}',
                        'seed': seed, 'fanout': 4, 'loss': loss,
                        'contrast': contrast, 'metric': metric,
                        'test': 'one-sided noninferiority' if margin else 'one-sided superiority',
                        'noninferiority_margin': margin,
                        **one_sided(differences(seed, 4, 'three_' + loss,
                                                second + '_' + loss, metric), margin)})
        for fanout in (8, 16):
            for loss in ('task', 'relation'):
                for contrast, second in (('three-plain', 'plain'),
                                         ('three-single', 'single')):
                    for metric in METRICS:
                        margin = .0005 if metric == 'micro_mrr' else .002
                        protection.append({
                            'comparison': f'seed{seed}-f{fanout}-{loss}-{contrast}-{metric}',
                            'seed': seed, 'fanout': fanout, 'loss': loss,
                            'contrast': contrast, 'metric': metric,
                            'test': 'one-sided noninferiority',
                            'noninferiority_margin': margin,
                            **one_sided(differences(seed, fanout, 'three_' + loss,
                                                    second + '_' + loss, metric), margin)})
        for fanout in (4, 8, 16):
            for structure in STRUCTURES:
                for metric in METRICS:
                    relation.append({
                        'comparison': f'seed{seed}-f{fanout}-{structure}-relation-task-{metric}',
                        'seed': seed, 'fanout': fanout, 'structure': structure,
                        'metric': metric, 'contrast': 'relation-task',
                        **student_summary(differences(
                            seed, fanout, structure + '_relation',
                            structure + '_task', metric))})
    return (holm(main, family_size=16),
            holm(protection, family_size=32),
            holm(relation, family_size=36))


def analyze_final_shards(shards):
    expected = {(seed, start) for seed in (11, 23) for start in (0, 4, 8, 12)}
    if len(shards) != 8:
        raise ValueError('eight complete joint final shards required')
    records, selected, fixed, observed = {}, {}, {}, set()
    for shard in shards:
        key = shard.get('seed'), shard.get('repeat_start')
        if (shard.get('status') != 'complete' or key not in expected or key in observed
                or shard.get('original_F_qualification', {}).get('accepted') is not True
                or set(shard.get('full_qualifications', {})) != set(ARMS)
                or any(v.get('accepted') is not True
                       for v in shard['full_qualifications'].values())
                or set(shard.get('module_off_full_controls', {}))
                != {arm for arm in ARMS if not arm.startswith('plain_')}
                or any(v.get('accepted') is not True
                       for v in shard['module_off_full_controls'].values())):
            raise ValueError('complete exact full-qualified joint shard required')
        observed.add(key)
        seed, start = key
        if set(shard.get('selected', {})) != set(ARMS) or set(shard.get('fixed_F', {})) != set(ARMS):
            raise ValueError('six selected checkpoint and own-F identities required')
        if seed in selected and selected[seed] != shard['selected']:
            raise ValueError('selected joint checkpoints changed between shards')
        if seed in fixed and fixed[seed] != shard['fixed_F']:
            raise ValueError('selected joint full references changed between shards')
        selected[seed], fixed[seed] = shard['selected'], shard['fixed_F']
        if len(shard.get('rows', [])) != 72:
            raise ValueError('all 72 joint shard conditions required')
        for row in shard['rows']:
            name = row['seed'], row['fanout'], row['repeat'], row['arm']
            if (row['seed'] != seed or row['fanout'] not in (4, 8, 16)
                    or not start <= row['repeat'] < start + 4
                    or row['arm'] not in ARMS or name in records):
                raise ValueError('duplicate or out-of-shard joint condition')
            records[name] = {metric: row[metric] for metric in METRICS}
    if observed != expected:
        raise ValueError('missing joint final shard')
    main, protection, relation = final_families(records)
    module_increment, extra_directions = {}, {}
    for loss in ('task', 'relation'):
        module_increment[loss] = all(
            row['reject_holm'] and row['mean'] > 0
            for row in main if row['loss'] == loss and row['contrast'] == 'three-plain')
        extra_directions[loss] = module_increment[loss] and all(
            row['reject_holm'] and (row['mean'] > 0 if row['metric'] == 'micro_mrr' else True)
            for row in main if row['loss'] == loss and row['contrast'] == 'three-single')
    return {'status': 'complete',
            'scope': 'two fixed starting checkpoints, six separately selected models, 16 paired graphs per budget',
            'main_holm16': main, 'protection_holm32': protection,
            'relation_holm36': relation, 'module_increment_by_loss': module_increment,
            'extra_directions_by_loss': extra_directions,
            'selected': selected, 'fixed_F': fixed}


def verify_release(config, original, policy, old, phase, record_path, source_commit,
                   *, seed=None, arm=None, repeat_start=None, original_run=None,
                   training_runs=None):
    from .hgcn_evidence import file_hash

    digest = validate_config(config, original, policy, old)
    root = Path(__file__).parents[2]
    document = root / 'docs/operations/RTSC_HGCN_JOINT_V1.md'
    if file_hash(document) != DOCUMENT_SHA256:
        raise ValueError('registered joint protocol document changed')
    for name in OLD_SOURCES:
        relative = 'src/acl_hct/' + name + '.py'
        archived = subprocess.check_output(
            ['git', 'show', OLD_SOURCE_COMMIT + ':' + relative],
            cwd=root, timeout=5).replace(b'\r\n', b'\n')
        current = (root / relative).read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(current).digest() != hashlib.sha256(archived).digest():
            raise ValueError('accepted Stage A scientific source changed')
    if phase not in PHASES or not isinstance(source_commit, str) or not re.fullmatch('[0-9a-f]{40}', source_commit):
        raise ValueError('registered phase and reviewed source commit required')
    if phase == 'probe' and (seed, arm) != (11, 'three_relation'):
        raise ValueError('only registered seed11 three-relation discarded probe allowed')
    if phase == 'train' and (seed not in (11, 23) or arm not in ARMS):
        raise ValueError('registered joint training selector required')
    if phase in ('final', 'benchmark') and (seed not in (11, 23) or arm is not None):
        raise ValueError('registered selected six-arm seed required')
    if phase in ('cpu_quality', 'cuda_quality') and (seed is not None or arm is not None):
        raise ValueError('quality phase cannot select science arm')
    if (phase == 'final' and repeat_start not in (0, 4, 8, 12)
            or phase != 'final' and repeat_start is not None):
        raise ValueError('exact four-repeat joint final selector required')
    selector = {'seed': seed, 'arm': arm, 'repeat_start': repeat_start}
    raw = Path(record_path).read_bytes()
    record = json.loads(raw)
    expected = {'protocol': PROTOCOL, 'accepted': True, 'phase': phase,
                'source_commit': source_commit,
                'source_sha256_normalized_lf': source_hashes(),
                'config_sha256': digest, 'upstream_commit': config['upstream_commit'],
                'selector': selector}
    if any(record.get(k) != value for k, value in expected.items()):
        raise ValueError('joint release source/config/selector mismatch')
    seconds = record.get('worker_seconds')
    limit = {'cpu_quality': 600, 'cuda_quality': 600, 'probe': 1200}.get(phase, 86400)
    if type(seconds) is not int or not 1 <= seconds <= limit:
        raise ValueError('bounded joint worker deadline required')
    if phase != 'cpu_quality':
        gates = ('cpu_quality',) if phase == 'cuda_quality' else ('cpu_quality', 'cuda_quality')
        for gate in gates:
            value = record.get('quality_gates', {}).get(gate, {})
            if (value.get('accepted') is not True or value.get('source_commit') != source_commit
                    or value.get('config_sha256') != digest
                    or not re.fullmatch('[0-9a-f]{64}', str(value.get('report_sha256')))):
                raise ValueError('same-source joint CPU/CUDA quality acceptance required')
    if phase in ('probe', 'train', 'final', 'benchmark'):
        if original_run is None or file_hash(original_run) != policy['origins'][str(seed)]['training_run_sha256']:
            raise ValueError('exact original accepted training lineage required')
        inputs = record.get('inputs', {})
        if (inputs.get('original_training_run_sha256') != file_hash(original_run)
                or inputs.get('original_best_checkpoint_sha256')
                != policy['origins'][str(seed)]['best_checkpoint_sha256']):
            raise ValueError('released original best input mismatch')
        baseline = record.get('baseline_acceptance', {})
        if (baseline.get('accepted') is not True
                or baseline.get('v2_policy_sha256') != canonical(policy)
                or baseline.get('original_training_run_sha256') != file_hash(original_run)
                or baseline.get('original_best_checkpoint_sha256')
                != policy['origins'][str(seed)]['best_checkpoint_sha256']
                or not re.fullmatch('[0-9a-f]{64}', str(baseline.get('review_sha256')))):
            raise ValueError('independently accepted original HGCN required')
    if phase in ('train', 'final', 'benchmark'):
        probe = record.get('probe_acceptance', {})
        if (probe.get('accepted') is not True
                or probe.get('source_commit') != source_commit
                or probe.get('config_sha256') != digest
                or not re.fullmatch('[0-9a-f]{64}', str(probe.get('review_sha256')))):
            raise ValueError('same-source discarded joint probe acceptance required')
        budget = record.get('phase_budget', {})
        if (budget.get('accepted') is not True
                or budget.get('phase') != phase
                or not re.fullmatch('[0-9a-f]{64}', str(budget.get('record_sha256')))):
            raise ValueError('post-probe phase resource bound required')
    if phase in ('final', 'benchmark'):
        if training_runs is None or set(training_runs) != set(ARMS):
            raise ValueError('six selected joint training results required')
        inputs = record.get('inputs', {})
        reviews = record.get('training_reviews', {})
        if set(reviews) != set(ARMS):
            raise ValueError('six independently accepted joint training reviews required')
        for name in ARMS:
            digest_run = file_hash(training_runs[name])
            review = reviews[name]
            if (inputs.get(name + '_result_sha256') != digest_run
                    or review.get('accepted') is not True
                    or review.get('result_sha256') != digest_run
                    or not re.fullmatch('[0-9a-f]{64}', str(review.get('review_sha256')))):
                raise ValueError('selected joint arm input/review mismatch')
    if phase != 'cpu_quality':
        accounting = record.get('accounting', {})
        if (type(accounting.get('reserved_gpu_seconds')) is not int
                or accounting['reserved_gpu_seconds'] < seconds
                or type(accounting.get('concurrent_jobs_including_this')) is not int
                or not 1 <= accounting['concurrent_jobs_including_this'] <= 3
                or phase in ('cuda_quality', 'probe')
                and accounting['concurrent_jobs_including_this'] != 1):
            raise ValueError('bounded joint GPU accounting and concurrency required')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                   cwd=root, text=True, timeout=5).strip()
    if head != source_commit:
        raise ValueError('execution HEAD differs from reviewed joint source')
    return {'release_sha256': hashlib.sha256(raw).hexdigest(),
            'worker_seconds': seconds,
            'source_sha256_normalized_lf': expected['source_sha256_normalized_lf']}

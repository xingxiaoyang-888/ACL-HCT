"""Stdlib registration/review gate for the independent frozen recovery pilot."""
import json
from pathlib import Path
import subprocess

from .encoder_matched_registration import canonical, file_sha256, source_identity as _identity

PROTOCOL = 'E2-frozen-recovery-v1'
CONFIG_SHA256 = '8ecd462845cf6f413bf57186506cabf044dcd87a52b4b6d490744db4def27172'
SOURCE_NAMES = ('__init__ backbone geometry aggregation protocols ranking train mechanisms '
                'backbone_frozen_inputs encoder_matched encoder_matched_registration encoder_matched_control '
                'evaluate_checkpoint development_view frozen_forward frozen_stats frozen_structure frozen_fixture '
                'vector_structure diagnostic_archive e2_pilot model data text_capacity taxonomy smoke '
                'frozen_bias_intervention frozen_recovery recovery_inputs recovery_registration recovery_entry recovery_analysis').split()


def source_hashes():
    import hashlib
    root = Path(__file__).parent
    return {name + '.py': hashlib.sha256((root / (name + '.py')).read_text(encoding='utf-8').encode()).hexdigest()
            for name in SOURCE_NAMES}


def validate_config(config):
    if config.get('protocol') != PROTOCOL or canonical(config) != CONFIG_SHA256:
        raise ValueError('exact frozen recovery registration required')
    return CONFIG_SHA256


def source_identity(declared):
    identity = _identity(declared)
    if identity['git']['available']:
        import hashlib
        root = Path(__file__).resolve().parents[2]
        for name, expected in source_hashes().items():
            try:
                blob = subprocess.check_output(['git', '-C', str(root), 'show', declared + ':src/acl_hct/' + name],
                                               stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as error:
                raise ValueError('unpublished recovery source: ' + name) from error
            actual = hashlib.sha256(blob.decode().replace('\r\n', '\n').replace('\r', '\n').encode()).hexdigest()
            if actual != expected:
                raise ValueError('execution source differs from declared Git tree: ' + name)
    return identity


def verify_release(config, identity, approval, quality_path, phase):
    validate_config(config)
    required = dict(user_authorized=True, quality_review_passed=True, entry_criteria_frozen=True,
                    supervisor_released=True, numerical_limits_reviewed=True,
                    scope=PROTOCOL, config_sha256=CONFIG_SHA256,
                    source_commit=identity['source_commit'], execution_phase=phase)
    if (any(approval.get(k) != v or (isinstance(v, bool) and approval.get(k) is not v) for k, v in required.items())
            or not isinstance(approval.get('user_message_reference'), str)
            or not approval['user_message_reference'].strip()):
        raise ValueError('exact user/review/source/phase and numerical review release required')
    if file_sha256(quality_path) != approval.get('quality_record_sha256'):
        raise ValueError('reviewed quality raw SHA mismatch')
    quality = json.loads(Path(quality_path).read_text(encoding='utf-8'))
    actual = source_hashes()
    if (quality.get('status') != 'passed' or quality.get('config_canonical_sha256') != CONFIG_SHA256
            or quality.get('source_lf_sha256') != actual):
        raise ValueError('quality must bind the exact recovery source/config')
    if any(actual[name] != expected for name, expected in config['frozen_source_lf_sha256'].items()):
        raise ValueError('original science or accepted adapter changed')
    return {'quality_raw_sha256': file_sha256(quality_path), 'approval_canonical_sha256': canonical(approval)}


def expected_receipts(config, seeds):
    files = {'prepared/' + key: value for key, value in config['prepared']['input_raw_sha256'].items()}
    files.update({'prepared/features.npz': config['prepared']['features_npz_raw_sha256'],
                  'prepared/evaluator_valid.json': config['prepared']['evaluator_valid_raw_sha256']})
    for seed in seeds:
        spec, cal = config['checkpoints'][str(seed)], config['calibration'][str(seed)]
        files.update({f'seed{seed}/checkpoint': spec['sha256'], f'seed{seed}/baseline': spec['baseline_report_sha256']})
        for stem, prefix in (('entry', 'entry'), ('fanout_4', 'fanout')):
            for suffix, field in (('.json', 'manifest'), ('.npz', 'npz')):
                files[f'seed{seed}/calibration/{stem}{suffix}'] = cal[f'{prefix}_{field}_raw_sha256']
    return files


def validate_receipts(receipts, expected):
    if set(receipts) != set(expected):
        raise ValueError('exact selected original worker receipt inventory required')
    for name, digest in expected.items():
        row = receipts[name]
        if (row.get('raw_sha256') != digest or type(row.get('original_path_reads')) is not int
                or row['original_path_reads'] != 1 or type(row.get('full_file_hash_checks')) is not int
                or row['full_file_hash_checks'] != 1 or row.get('decoded_from_verified_cached_bytes') is not True
                or type(row.get('bytes')) is not int or row['bytes'] <= 0):
            raise ValueError('single-read/hash verified-byte decode receipt required: ' + name)


def verify_gate(path, expected, phase, identity, config):
    if file_sha256(path) != expected:
        raise ValueError('reviewed phase artifact raw SHA mismatch')
    row = json.loads(Path(path).read_text(encoding='utf-8'))
    supervisor = json.loads((Path(path).parent / 'supervisor.json').read_text(encoding='utf-8'))
    if (supervisor.get('status') != 'complete' or supervisor.get('worker_exit_code') != 0
            or supervisor.get('deadline_seconds') != 240 or row.get('status') != 'complete'
            or row.get('phase') != phase or row.get('config_sha256') != CONFIG_SHA256
            or row.get('source_commit') != identity['source_commit']
            or row.get('source_lf_sha256') != source_hashes() or row.get('torch') != '2.5.1+cu124'
            or row.get('seeds') != [11, 23]
            or supervisor.get('context', {}).get('phase') != phase
            or supervisor['context'].get('protocol') != PROTOCOL
            or supervisor['context'].get('config_sha256') != CONFIG_SHA256
            or supervisor['context'].get('source_commit') != identity['source_commit']):
        raise ValueError('complete same-release native phase evidence required')
    if phase == 'cpu_preflight':
        if (row.get('forward_backward_optimizer_called') is not False
                or row.get('device') != 'cpu' or row.get('declared_resources') != config['resources']['cpu_preflight']
                or set(row.get('models', {})) != {'11', '23'}):
            raise ValueError('complete two-model input preflight required')
        if row.get('all_bulk_inputs_verified_inside_worker') is not True:
            raise ValueError('bulk inputs must be verified inside the supervised worker')
        validate_receipts(row['worker_input_receipts'], expected_receipts(config, (11, 23)))
        for seed in (11, 23):
            proof = row['models'][str(seed)]
            spec, cal = config['checkpoints'][str(seed)], config['calibration'][str(seed)]
            if (proof.get('checkpoint_raw_sha256') != spec['sha256']
                    or proof.get('baseline_raw_sha256') != spec['baseline_report_sha256']
                    or proof.get('prepared_manifest_hash') != config['prepared']['manifest_hash']
                    or proof.get('valid_hash') != config['prepared']['valid_queries_hash']
                    or proof.get('nodes_count') != config['prepared']['nodes_count']
                    or proof.get('valid_count') != config['valid_query_count'] or proof.get('calibration_R') != cal['R']
                    or proof.get('parameter_count') != config['model']['parameter_count']
                    or proof.get('native_dtype') != 'float32' or proof.get('gradients_present') is not False
                    or proof.get('all_bulk_inputs_verified_inside_worker') is not True
                    or proof.get('selection', {}).get('status') != 'passed'
                    or proof['selection'].get('selected_step') != spec['step']):
                raise ValueError('original selected model/calibration input identity mismatch')
            validate_receipts(proof['worker_input_receipts'], expected_receipts(config, (seed,)))
            if set(proof.get('calibration_fields', {})) != set(cal['fields']) - {'nodes'}:
                raise ValueError('all calibrated fields required')
            for name, actual in proof['calibration_fields'].items():
                if actual != {k: cal['fields'][name][k] for k in ('shape', 'dtype', 'data_sha256')}:
                    raise ValueError('original calibrated array identity mismatch')
    else:
        required = {'FP64_reference', 'zero_native_identity', 'cache_scores', 'complete_ranking',
                    'shared_C_plan', 'q_plan_rng_isolation', 'cast_limits', 'archive_complete'}
        if (phase != 'cuda_fixture' or row.get('device') != 'NVIDIA L40'
                or row.get('original_runtime_inputs_opened') != [] or set(row.get('checks', {})) != required
                or any(row['checks'][name] is not True for name in required)):
            raise ValueError('complete fixed CUDA fixture checks required')
        from .diagnostic_archive import read_archive
        if set(row.get('archives', {})) != {'11', '23'}:
            raise ValueError('both fixture archives required')
        for descriptor in row['archives'].values():
            data = read_archive(Path(path).parent, descriptor)
            if not {'features', 'weights', 'base', 'bias', 'q', 'plans', 'neighbors', 'FP64_reference_points',
                    'native_points', 'direct_scores', 'cached_scores', 'rankings'} <= set(data):
                raise ValueError('full independent fixture recomputation arrays required')
            import numpy as np
            from .encoder_matched_control import validate_ranking
            if (data['features'].shape != (40, 128) or data['features'].dtype != np.float32 or not np.isfinite(data['features']).all()
                    or any(data[k].shape != (40, 129) or data[k].dtype != np.float64 or not np.isfinite(data[k]).all()
                           for k in ('base', 'bias', 'q'))
                    or len(data['weights']) != 8 or sum(v.size for v in data['weights'].values()) != 82433
                    or any(v.dtype != np.float32 or not np.isfinite(v).all() for v in data['weights'].values())
                    or set(data['plans']) != {'0', '8'} or any(len(v) != 2 or any(len(layer) != 40 for layer in v) for v in data['plans'].values())):
                raise ValueError('complete production-dimension two-repeat fixture arrays required')
            keys = {f'{repeat}/{condition}' for repeat in (0, 8) for condition in ('F', 'S', 'O', 'C', 'Q')}
            if any(set(data[k]) != keys for k in ('native_points', 'direct_scores', 'cached_scores', 'rankings')):
                raise ValueError('all five fixture conditions for both global repeat IDs required')
            for key in keys:
                if (data['native_points'][key].shape != (40, 129) or data['native_points'][key].dtype != np.float32
                        or not np.isfinite(data['native_points'][key]).all()
                        or any(data[k][key].shape != (40, 40) or not np.isfinite(data[k][key]).all()
                               for k in ('direct_scores', 'cached_scores'))):
                    raise ValueError('full finite native fixture points/scores required')
                validate_ranking(data['rankings'][key], {'nodes': data['nodes'], 'valid': [tuple(map(int, p)) for p in data['valid']]})
    return row

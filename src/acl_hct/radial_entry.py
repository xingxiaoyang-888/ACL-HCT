"""Stdlib-only bounded entry for saved-sample radial-component diagnostics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .encoder_matched_registration import canonical, file_sha256, source_identity
from .recovery_entry import supervise
from .recovery_registration import source_hashes as old_source_hashes

PROTOCOL = 'E2-radial-components-v1/B'
CONFIG_SHA256 = '576b9d2984b2ce10c5e3c0bae07a570fcf95e57c10db6b76aca16740e25d351d'
MODULES = ('radial_components.py', 'radial_recovery.py', 'radial_entry.py', 'radial_analysis.py')
FIXTURE_CHECKS = {'FP64_reference', 'zero_native_identity', 'cache_scores', 'complete_ranking',
                  'head_and_RNG_unchanged', 'cast_limits', 'undefined_unapplied_bias', 'archive_complete'}


def lf_hash(path):
    return hashlib.sha256(Path(path).read_text(encoding='utf-8').encode()).hexdigest()


def source_hashes():
    return {name: lf_hash(Path(__file__).parent / name) for name in MODULES}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_config(config):
    require(config.get('protocol') == PROTOCOL and canonical(config) == CONFIG_SHA256,
            'exact frozen B configuration required')
    require(old_source_hashes() == config['old_source_lf_sha256'], 'original 32 source dependencies changed')
    require(all(lf_hash(Path(__file__).parent / name) == digest
                for name, digest in config['A_source_lf_sha256'].items()), 'accepted A dependency changed')
    return CONFIG_SHA256


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'old-config', 'bindings', 'protocol', 'approval', 'quality',
                 'output', 'fixture-record', 'fixture-review'):
        cli.add_argument('--' + name, type=Path)
    cli.add_argument('--source-commit')
    cli.add_argument('--phase', choices=('cuda_fixture', 'science'))
    cli.add_argument('--seed', type=int, choices=(11, 23))
    cli.add_argument('--repeat-start', type=int, choices=(0, 8))
    cli.add_argument('--shards', type=Path, nargs=4)
    cli.add_argument('--execute', action='store_true')
    cli.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    return cli


def validate_arguments(args):
    require(all(getattr(args, k) is not None for k in (
        'config', 'protocol', 'approval', 'quality', 'output', 'source_commit', 'phase')),
        'explicit reviewed phase, source, quality, protocol and output required')
    inputs = ('old_config', 'bindings', 'shards', 'seed', 'repeat_start', 'fixture_record', 'fixture_review')
    if args.phase == 'cuda_fixture':
        require(all(getattr(args, k) is None for k in inputs), 'synthetic fixture accepts no original inputs')
    else:
        require(all(getattr(args, k) is not None for k in inputs), 'all four original roots and independent fixture review required')


def verify_fixture(args, config, approval, sources):
    """Supervisor acceptance is separate from the fixture worker's own checks."""
    require(file_sha256(args.fixture_record) == approval['fixture_raw_sha256']
            and file_sha256(args.fixture_review) == approval['fixture_review_raw_sha256'],
            'reviewed fixture/review bytes differ')
    row = json.loads(args.fixture_record.read_bytes())
    review = json.loads(args.fixture_review.read_bytes())
    sup = json.loads((args.fixture_record.parent / 'supervisor.json').read_bytes())
    require(row['status'] == 'complete' and row['phase'] == 'cuda_fixture'
            and row['protocol'] == PROTOCOL and row['config_sha256'] == CONFIG_SHA256
            and row['source_commit'] == args.source_commit and row['source_lf_sha256'] == sources
            and row['original_runtime_inputs_opened'] == [] and row['seeds'] == [11, 23]
            and row['torch'] == config['runtime']['torch'] and row['device'] == 'NVIDIA L40'
            and sup['status'] == 'complete' and sup['worker_exit_code'] == 0 and sup['deadline_seconds'] == 240
            and sup['context']['protocol'] == PROTOCOL and sup['context']['phase'] == 'cuda_fixture'
            and sup['context']['config_sha256'] == CONFIG_SHA256
            and sup['context']['source_commit'] == args.source_commit
            and set(row['checks']) == FIXTURE_CHECKS and all(v is True for v in row['checks'].values())
            and set(row['archives']) == {'11/mixed', '11/zero', '23/mixed', '23/zero'}
            and review['status'] == 'independent_fixture_review_passed'
            and review['fixture_raw_sha256'] == file_sha256(args.fixture_record)
            and review['source_commit'] == args.source_commit and review['config_sha256'] == CONFIG_SHA256
            and review['source_lf_sha256'] == sources,
            'complete same-release fixture and independent supervisor acceptance required')
    # Full archive verification is inside the already bounded tensor worker.
    from .diagnostic_archive import read_archive
    import numpy as np
    from .encoder_matched_control import validate_ranking
    for key, descriptor in row['archives'].items():
        evidence = read_archive(args.fixture_record.parent, descriptor)
        require({'base', 'bias', 'floor', 'root_index', 'sample64', 'weights', 'nodes', 'valid',
                 'outward_unit', 'defined', 'unapplied_bias', 'removed_fields', 'offsets',
                 'FP64_reference_points', 'GPU_FP64_points', 'native_points', 'casting',
                 'direct_scores', 'cached_scores', 'rankings', 'structure'} <= set(evidence),
                'full independent fixture arrays required')
        require(f"{evidence['seed']}/{evidence['case']}" == key and evidence['root_index'] == 0,
                'fixed fixture identity required')
        shape = (16, 129)
        require(all(evidence[name].shape == shape and evidence[name].dtype == np.float64
                    and np.isfinite(evidence[name]).all() for name in ('base', 'bias', 'sample64', 'outward_unit', 'unapplied_bias'))
                and evidence['floor'].shape == (16,) and evidence['floor'].dtype == np.float64
                and evidence['defined'].shape == (16,) and evidence['defined'].dtype == np.bool_,
                'complete production-dimension FP64 fixture required')
        for name in ('S', 'R', 'T'):
            require(evidence['native_points'][name].shape == shape and evidence['native_points'][name].dtype == np.float32
                    and np.isfinite(evidence['native_points'][name]).all()
                    and all(evidence[k][name].shape == (16, 16) and evidence[k][name].dtype == np.float32
                            and np.isfinite(evidence[k][name]).all() for k in ('direct_scores', 'cached_scores')),
                    'all finite native points and full scores required')
            validate_ranking(evidence['rankings'][name],
                             {'nodes': evidence['nodes'], 'valid': [tuple(map(int, p)) for p in evidence['valid']]})
    return {'fixture_raw_sha256': file_sha256(args.fixture_record),
            'fixture_review_raw_sha256': file_sha256(args.fixture_review)}


def verify_release(args):
    validate_arguments(args)
    config = json.loads(args.config.read_bytes())
    validate_config(config)
    identity = source_identity(args.source_commit)
    sources = source_hashes()
    if identity['git']['available']:
        root = Path(__file__).resolve().parents[2]
        for name, digest in sources.items():
            blob = subprocess.check_output(['git', '-C', str(root), 'show',
                                             args.source_commit + ':src/acl_hct/' + name])
            require(hashlib.sha256(blob.decode().replace('\r\n', '\n').replace('\r', '\n').encode()).hexdigest() == digest,
                    'unpublished or modified B source')
    approval = json.loads(args.approval.read_bytes())
    quality = json.loads(args.quality.read_bytes())
    require(approval['status'] == 'approved_B_' + args.phase and approval['protocol'] == PROTOCOL
            and approval['execution_phase'] == args.phase and approval['source_commit'] == args.source_commit
            and approval['user_authorized'] is True and approval['quality_review_passed'] is True
            and approval['supervisor_released'] is True and approval['GPU_requested'] == 1
            and approval['config_canonical_sha256'] == CONFIG_SHA256
            and approval['new_source_lf_sha256'] == sources,
            'exact authorized/reviewed B phase release required')
    require(quality['status'] == 'passed' and quality['phase'] == 'B'
            and quality['config_canonical_sha256'] == CONFIG_SHA256
            and quality['new_source_lf_sha256'] == sources
            and approval['quality_raw_sha256'] == file_sha256(args.quality)
            and approval['config_raw_sha256'] == file_sha256(args.config)
            and quality['config_raw_sha256'] == file_sha256(args.config)
            and approval['protocol_lf_sha256'] == lf_hash(args.protocol)
            and quality['protocol_lf_sha256'] == lf_hash(args.protocol),
            'reviewed configuration, quality and current protocol bytes required')
    provenance = {**identity, 'new_source_lf_sha256': sources,
                  'approval_raw_sha256': file_sha256(args.approval), 'quality_raw_sha256': file_sha256(args.quality),
                  'protocol_lf_sha256': lf_hash(args.protocol), 'release_verified': True}
    if args.phase == 'science':
        digest = file_sha256(args.bindings)
        bindings = json.loads(args.bindings.read_bytes())
        require(digest == config['original_input_catalog_raw_sha256'] == approval['bindings_raw_sha256']
                and bindings['old_science_source_commit'] == config['old_science_source_commit']
                and bindings['no_new_graph_samples'] is True and bindings['A_runtime_prepared_graph_files'] == []
                and file_sha256(args.old_config) == bindings['old_config_raw_sha256']
                and canonical(json.loads(args.old_config.read_bytes())) == config['old_config_canonical_sha256'],
                'exact original catalog/configuration and zero new samples required')
        provenance.update(bindings_raw_sha256=digest, **verify_fixture(args, config, approval, sources))
    return config, provenance


def main():
    args = parser().parse_args()
    if not args.execute:
        print(json.dumps({'status': 'blocked_until_exact_quality_release', 'protocol': PROTOCOL,
                          'config_sha256': CONFIG_SHA256, 'fixture_worker_seconds': 240,
                          'science_worker_seconds': 840, 'new_graph_samples': 0, 'GNN_forward_calls': 0}))
        return 0
    validate_arguments(args)
    if not args.worker:
        command = [sys.executable, '-m', 'acl_hct.radial_entry', *sys.argv[1:], '--worker']
        return supervise(command, args.output, seconds=240 if args.phase == 'cuda_fixture' else 840,
                         metadata={'phase': args.phase, 'protocol': PROTOCOL, 'config_sha256': CONFIG_SHA256,
                                   'source_commit': args.source_commit})
    require(os.environ.get('ACL_FROZEN_RECOVERY_SUPERVISOR_PID') == str(os.getppid()),
            'direct bounded supervisor required')
    deadline = float(os.environ['ACL_FROZEN_RECOVERY_DEADLINE'])
    require(time.monotonic() < deadline, 'whole-worker deadline expired')
    config, provenance = verify_release(args)
    from .radial_recovery import worker
    return worker(args, config, provenance, deadline)


if __name__ == '__main__':
    sys.exit(main())

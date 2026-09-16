"""Registered independent E2 confirmation; separate from development and training."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from . import e2_development as dev
from . import e2_pilot as entry
from . import confirmation_registration as registration
from .mechanisms import source_identity


def source_hashes():
    # Stable across fixture, scientific CLI and test import order; no Git dependency.
    names = ('__init__ aggregation backbone data development_view diagnostic_archive diagnostics '
             'e1_audit e1_closure e1_controls e2_development e2_pilot evaluate_checkpoint '
             'frozen_fixture frozen_forward frozen_stats frozen_structure geometry mechanisms '
             'model protocols ranking smoke taxonomy train vector_structure '
             'e2_confirmation confirmation_registration confirmation_analysis').split()
    hashes = {}
    for stem in names:
        name = stem + '.py'
        hashes['acl_hct/' + name] = hashlib.sha256((Path(__file__).parent / name).read_text(encoding='utf-8').encode()).hexdigest()
    return hashes


def design(config):
    return dev.DiagnosticDesign(protocol=registration.PROTOCOL, panel='diagnostic_confirmation',
                                base_seed=registration.BASE_SEED, repetition_limit=512, task_limit=128,
                                panel_binding=config['panels'], registration_sha256=registration.validate_config(config))


def verify_cuda_evidence(path, approval, identity):
    dev.verify_cuda_evidence(path, approval, identity)
    row = json.loads(Path(path).read_text(encoding='utf-8'))
    if (row.get('confirmation_fixture_passed') is not True
            or row.get('registration_sha256') != registration.REGISTRATION_SHA256
            or row.get('source_sha256_normalized_lf') != source_hashes()):
        raise ValueError('new confirmation CUDA fixture for exact source and registration required')


def cuda_fixture(directory, source_commit, config):
    from .frozen_fixture import synthetic_fixture
    identity = source_identity(source_commit)
    if identity['source_commit'] is None:
        raise ValueError('fixed source required')
    device = torch.device('cuda'); dev.require_cuda(device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    model, features, view = synthetic_fixture()
    panels = entry.make_panels(view, 1000)
    binding = {'combined_hash': panels['hash'], **{k: {s: panels['panels'][k][s] for s in ('panel_hash', 'relation_hash')}
               for k in ('development', 'diagnostic_confirmation')}}
    fixture_design = dev.DiagnosticDesign(protocol=registration.PROTOCOL, panel='diagnostic_confirmation',
                        base_seed=registration.BASE_SEED, panel_binding=binding,
                        registration_sha256=registration.validate_config(config))
    result = dev.diagnose(model.to(device), features.to(device), view, directory=directory,
                         fanouts=[4, 16], repetitions=6, task_repetitions=3, max_seconds=120,
                         budget=128, candidate_chunk=11, design=fixture_design,
                         namespace=registration.PROTOCOL + '/engineering-fixture')
    result.update(scope='40-node engineering fixture only; no real confirmation outputs', source=identity,
                  source_sha256_normalized_lf=source_hashes(), device=torch.cuda.get_device_name(0),
                  cuda_fixture_passed=result['status'] == 'complete',
                  confirmation_fixture_passed=result['status'] == 'complete')
    dev.write_json(Path(directory) / 'cuda-quality.json', result)
    if not result['confirmation_fixture_passed']:
        raise ValueError('confirmation CUDA fixture failed')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--cuda-fixture', action='store_true')
    parser.add_argument('--seed', type=int, choices=(11, 23))
    parser.add_argument('--fanout', type=int, choices=registration.FANOUTS)
    parser.add_argument('--source-commit')
    for name in ('prepared', 'checkpoint', 'training-release', 'baseline-report', 'approval-record',
                 'cuda-fixture-record', 'development-index', 'output-dir'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    config_hash = registration.validate_config(config)
    if args.cuda_fixture:
        if args.execute or args.output_dir is None or not args.source_commit:
            raise ValueError('fixture requires new output directory and fixed source')
        args.output_dir.mkdir(parents=True, exist_ok=False)
        cuda_fixture(args.output_dir, args.source_commit, config)
        return
    if not args.execute:
        audit = registration.verify_development(config, args.development_index) if args.development_index else None
        print(json.dumps({'status': 'static_only', 'registration_sha256': config_hash, 'development_audit': audit}))
        return
    required = ('seed', 'fanout', 'source_commit', 'prepared', 'checkpoint', 'training_release',
                'baseline_report', 'approval_record', 'cuda_fixture_record', 'development_index', 'output_dir')
    if any(getattr(args, key) is None for key in required):
        raise ValueError('fixed source, approval, new CUDA evidence and all registered inputs required')
    spec = registration.shard(config, args.seed, args.fanout)
    audit = registration.verify_development(config, args.development_index)
    identity = source_identity(args.source_commit)
    approval = json.loads(args.approval_record.read_text(encoding='utf-8'))
    entry.verify_approval(approval, config, identity)
    if approval.get('registration_sha256') != config_hash or approval.get('cuda_fixture_passed') is not True:
        raise ValueError('reviewed confirmation registration and CUDA gate required')
    verify_cuda_evidence(args.cuda_fixture_record, approval, identity)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / 'summary.json'
    evidence = {'development_audit': audit, 'registration_sha256': config_hash,
                'cuda_fixture_artifact_sha256': registration.file_sha256(args.cuda_fixture_record)}
    def save(row):
        dev.write_json(output, {**row, **evidence})
    save({'status': 'verification_started', 'config_sha256': config_hash})
    def runner(*positional, **keywords):
        return dev.diagnose(*positional, directory=args.output_dir, fanouts=[args.fanout],
                            repetitions=spec['repetitions'], task_repetitions=spec['task_repetitions'],
                            design=design(config), **keywords)
    try:
        result = entry.run(config, args.seed, args.prepared, args.checkpoint, args.training_release,
                           args.baseline_report, approval, args.source_commit, device='cuda', progress=save,
                           diagnostic_runner=runner, configuration_check=registration.validate_config,
                           execution_seconds=spec['internal_seconds'])
        result['source_sha256_normalized_lf'] = source_hashes()
        save(result)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        previous = json.loads(output.read_text(encoding='utf-8'))
        previous.update(status='failed', wrapper_failure=str(error)); save(previous)
        raise
    print(json.dumps({'status': result['status'], 'seed': args.seed, 'fanout': args.fanout}))
    if result['status'] != 'complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()

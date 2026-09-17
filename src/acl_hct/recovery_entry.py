"""Stdlib supervisor: the deadline starts before the tensor worker imports."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .recovery_registration import PROTOCOL, validate_config


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--config', type=Path, required=True)
    cli.add_argument('--phase', choices=('cpu_preflight', 'cuda_fixture', 'science'))
    cli.add_argument('--execute', action='store_true')
    cli.add_argument('--seed', type=int, choices=(11, 23))
    cli.add_argument('--repeat-start', type=int, choices=(0, 8))
    for name in ('source-commit', 'approval-record', 'quality-record', 'output', 'prepared-root',
                 'checkpoint-root', 'training-release', 'reference-root', 'calibration-root',
                 'cpu-preflight-record', 'cuda-fixture-record'):
        cli.add_argument('--' + name)
    return cli


def validate_arguments(args):
    if any(getattr(args, k) is None for k in ('phase', 'source_commit', 'approval_record', 'quality_record', 'output')):
        raise ValueError('reviewed phase/source/approval/quality and new output required')
    original = ('prepared_root', 'checkpoint_root', 'training_release', 'reference_root', 'calibration_root')
    if args.phase == 'cuda_fixture':
        if any(getattr(args, k) is not None for k in (*original, 'seed', 'repeat_start', 'cpu_preflight_record', 'cuda_fixture_record')):
            raise ValueError('synthetic fixture accepts no original runtime inputs or shard arguments')
    else:
        if any(getattr(args, k) is None for k in original):
            raise ValueError('all five explicit original input roots required')
        if args.phase == 'science':
            if any(getattr(args, k) is None for k in ('seed', 'repeat_start', 'cpu_preflight_record', 'cuda_fixture_record')):
                raise ValueError('fixed shard and both reviewed native phase records required')
        elif any(getattr(args, k) is not None for k in ('seed', 'repeat_start', 'cpu_preflight_record', 'cuda_fixture_record')):
            raise ValueError('two-model preflight has no shard or fixture-record arguments')


def phase_seconds(config, phase):
    if phase == 'cpu_preflight':
        return config['resources']['cpu_preflight']['worker_seconds']
    return config['resources']['science_worker_seconds' if phase == 'science' else 'fixture_worker_seconds']


def supervise(command, output, *, seconds, metadata=None, environment=None):
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not 0 < seconds <= 1080:
        raise ValueError('positive bounded whole-worker deadline required')
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError('new empty output required; no overwrite/resume')
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + seconds
    env = dict(os.environ if environment is None else environment)
    env['ACL_FROZEN_RECOVERY_SUPERVISOR_PID'] = str(os.getpid())
    env['ACL_FROZEN_RECOVERY_DEADLINE'] = str(deadline)
    timed_out = False
    with (root / 'worker.log').open('xb') as stream:
        worker = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            status = worker.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
            worker.kill()
            status = worker.wait()
        except BaseException:
            if worker.poll() is None:
                worker.kill()
                worker.wait()
            raise
    elapsed = time.monotonic() - started
    row = {'status': 'timeout' if timed_out else 'complete' if status == 0 else 'failed',
           'worker_exit_code': status, 'deadline_seconds': seconds, 'elapsed_seconds': elapsed,
           'deadline_overrun_seconds': max(0, elapsed - seconds), 'context': metadata,
           'deadline_scope': 'tensor imports, release checks, original input verification/loading, inference, rankings and archives',
           'kill_scope': 'direct tensor worker only; source Git probes are read-only; no GPU/training child process',
           'deadline_caveat': 'OS scheduling/reaping latency is measured; not a real-time guarantee'}
    (root / 'supervisor.json').write_text(json.dumps(row, indent=2) + '\n', encoding='utf-8', newline='\n')
    if row['status'] != 'complete':
        progress = {}
        for name in ('progress.json', 'run.json'):
            try:
                progress.update(json.loads((root / name).read_text(encoding='utf-8')))
            except (OSError, ValueError):
                pass
        failure = {'status': row['status'], 'worker_exit_code': status, 'context': metadata,
                   'completed_repetitions': progress.get('completed_repetitions', 0),
                   'active_repeat': progress.get('active_repeat'), 'active_phase': progress.get('active_phase'),
                   'partial_results_are_complete': False}
        (root / 'supervisor-failure.json').write_text(json.dumps(failure, indent=2) + '\n', encoding='utf-8', newline='\n')
    return 124 if timed_out else 0 if status == 0 else 2


def main():
    args = parser().parse_args()
    try:
        config = json.loads(args.config.read_text(encoding='utf-8'))
        config_hash = validate_config(config)
        if not args.execute:
            print(json.dumps({'status': 'static_only', 'protocol': PROTOCOL, 'config_sha256': config_hash,
                              'phases': ['cpu_preflight', 'cuda_fixture', 'science']}))
            return 0
        validate_arguments(args)
        command = [sys.executable, '-m', 'acl_hct.frozen_recovery']
        for name, value in vars(args).items():
            if name != 'execute' and value is not None:
                command.extend(['--' + name.replace('_', '-'), str(value)])
        return supervise(command, args.output, seconds=phase_seconds(config, args.phase),
                         metadata={'protocol': PROTOCOL, 'phase': args.phase, 'seed': args.seed,
                                   'repeat_start': args.repeat_start, 'config_sha256': config_hash,
                                   'source_commit': args.source_commit})
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

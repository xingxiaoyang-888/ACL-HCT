"""Stdlib process deadline for each reviewed matched-encoder execution phase."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .encoder_matched_registration import PROTOCOL, validate_config


def supervise(command, output, *, seconds, metadata=None, environment=None):
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not 0 < seconds <= 840:
        raise ValueError('positive bounded worker deadline required')
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError('new empty output required; no overwrite/resume')
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); deadline = started + seconds
    env = dict(os.environ if environment is None else environment)
    env['ACL_ENCODER_MATCHED_SUPERVISOR_PID'] = str(os.getpid())
    env['ACL_ENCODER_MATCHED_DEADLINE'] = str(deadline)
    timed_out = False
    with (root / 'worker.log').open('xb') as log:
        worker = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            status = worker.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True; worker.kill(); status = worker.wait()
        except BaseException:
            if worker.poll() is None:
                worker.kill(); worker.wait()
            raise
    elapsed = time.monotonic() - started
    row = {'status': 'timeout' if timed_out else 'complete' if status == 0 else 'failed',
           'worker_exit_code': status, 'deadline_seconds': seconds, 'elapsed_seconds': elapsed,
           'deadline_overrun_seconds': max(0, elapsed - seconds),
           'deadline_scope': 'imports, all verification, fixture or training, validations, reload and archives',
           'kill_scope': 'this direct worker only; worker launches no child processes', 'context': metadata,
           'deadline_caveat': 'OS scheduling/reaping latency measured; not a real-time guarantee'}
    (root / 'supervisor.json').write_text(json.dumps(row, indent=2) + '\n', encoding='utf-8', newline='\n')
    if row['status'] != 'complete':
        progress = {}; completed = 0
        for name in ('run.json', 'progress.json'):
            try:
                item = json.loads((root / name).read_text(encoding='utf-8'))
                completed = max(completed, item.get('completed_steps', 0)); progress.update(item)
            except (OSError, ValueError):
                pass
        failure = {'status': row['status'], 'worker_exit_code': status, 'context': metadata,
                   'completed_steps': completed,
                   'active_step': progress.get('active_step'), 'active_phase': progress.get('active_phase'),
                   'completed_validation_steps': [r['step'] for r in progress.get('evaluations', []) if r['status'] == 'complete'],
                   'partial_results_are_complete': False}
        (root / 'supervisor-failure.json').write_text(json.dumps(failure, indent=2) + '\n', encoding='utf-8', newline='\n')
    return 124 if timed_out else 0 if status == 0 else 2


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--config', type=Path, required=True)
    cli.add_argument('--phase', choices=('cpu_preflight', 'cuda_fixture', 'science'))
    cli.add_argument('--execute', action='store_true')
    cli.add_argument('--seed', type=int, choices=(11, 23))
    for name in ('source-commit', 'approval-record', 'quality-record', 'output', 'prepared-root',
                 'reference-root', 'cpu-preflight-record', 'cuda-fixture-record'):
        cli.add_argument('--' + name)
    args = cli.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding='utf-8')); config_hash = validate_config(config)
        if not args.execute:
            print(json.dumps({'status': 'static_only', 'protocol': PROTOCOL, 'config_sha256': config_hash,
                              'phases': ['cpu_preflight', 'cuda_fixture', 'science']})); return 0
        if any(getattr(args, k) is None for k in ('phase', 'source_commit', 'approval_record', 'quality_record', 'output')):
            raise ValueError('exact reviewed phase/source/approval/quality and new output required')
        command = [sys.executable, '-m', 'acl_hct.encoder_matched_control']
        for name, value in vars(args).items():
            if name != 'execute' and value is not None:
                command.extend(['--' + name.replace('_', '-'), str(value)])
        seconds = config['resources']['science_worker_seconds'] if args.phase == 'science' else config['resources']['fixture_worker_seconds']
        return supervise(command, args.output, seconds=seconds, metadata={'protocol': PROTOCOL, 'phase': args.phase, 'seed': args.seed})
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())

"""Separate release-gated supervisor for R-TSC/HGCN Stage A."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .hgcn_rtsc_protocol import PHASES, validate_config, verify_release


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--original-config', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--phase', choices=PHASES)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--seed', type=int)
    p.add_argument('--arm', choices=('task_only', 'task_relation'))
    p.add_argument('--repeat-start', type=int)
    p.add_argument('--probe-steps', type=int)
    for name in ('source-commit', 'release-record', 'upstream-root', 'upstream-manifest',
                 'prepared-root', 'original-training-run', 'task-result',
                 'relation-result', 'output'):
        p.add_argument('--' + name)
    return p


def supervise(command, output, seconds):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('fresh empty Stage A output required; no overwrite or resume')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    env = dict(os.environ)
    # Existing complete-ranking helper has a live-process deadline guard.
    env['ACL_HGCN_VALIDATION_PID'] = str(os.getpid())
    env['ACL_HGCN_VALIDATION_DEADLINE'] = str(started + seconds)
    timed_out = False
    with (output / 'worker.log').open('xb') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            status = child.wait(timeout=max(0., started + seconds - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True; child.kill(); status = child.wait()
        except BaseException:
            if child.poll() is None:
                child.kill(); child.wait()
            raise
    result = {'status': 'timeout' if timed_out else 'complete' if status == 0 else 'failed',
              'worker_exit_code': status, 'deadline_seconds': seconds,
              'elapsed_seconds': time.monotonic() - started}
    (output / 'supervisor.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n',
                                            encoding='utf-8', newline='\n')
    return 124 if timed_out else 0 if status == 0 else 2


def main():
    args = parser().parse_args()
    try:
        config = json.loads(args.config.read_bytes())
        original = json.loads(args.original_config.read_bytes())
        policy = json.loads(args.policy.read_bytes())
        digest = validate_config(config, original, policy)
        if not args.execute:
            print(json.dumps({'status': 'static_only', 'protocol': config['protocol'],
                              'config_sha256': digest, 'requires_reviewed_release': True}))
            return 0
        required = ['phase', 'source_commit', 'release_record', 'upstream_root', 'output']
        if args.phase in ('probe', 'train', 'final'):
            required += ['prepared_root', 'original_training_run']
        if args.phase == 'final':
            required += ['task_result', 'relation_result']
        if any(getattr(args, name) is None for name in required):
            raise ValueError('exact phase, released inputs and fresh output required')
        if (args.phase == 'probe' and args.probe_steps is None
                or args.phase != 'probe' and args.probe_steps is not None):
            raise ValueError('explicit 1..32 probe steps only on discarded probe phase')
        release = verify_release(config, original, policy, args.phase,
                                 args.release_record, args.source_commit,
                                 seed=args.seed, arm=args.arm, repeat_start=args.repeat_start,
                                 original_run=args.original_training_run,
                                 task_run=args.task_result,
                                 relation_run=args.relation_result)
        command = [sys.executable, '-m', 'acl_hct.hgcn_rtsc_worker']
        for name, value in vars(args).items():
            if value is not None and name != 'execute':
                command.extend(['--' + name.replace('_', '-'), str(value)])
        return supervise(command, args.output, release['worker_seconds'])
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

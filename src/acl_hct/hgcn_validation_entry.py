"""Fresh-output bounded supervisor; scientific execution needs its own release."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .hgcn_validation_registration import PHASES, PROTOCOL, role, validate_config, verify_release


def supervise(command, output, seconds, context, environment=None):
    if type(seconds) is not int or not 0 < seconds <= 2400:
        raise ValueError('registered <=2400s worker required')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('fresh empty output required, no overwrite or resume')
    output.mkdir(parents=True, exist_ok=True); start = time.monotonic()
    env = dict(os.environ if environment is None else environment)
    env['ACL_HGCN_VALIDATION_PID'] = str(os.getpid())
    env['ACL_HGCN_VALIDATION_DEADLINE'] = str(start + seconds)
    timeout = False
    with (output / 'worker.log').open('xb') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            status = child.wait(timeout=max(0., start + seconds - time.monotonic()))
        except subprocess.TimeoutExpired:
            timeout = True; child.kill(); status = child.wait()
        except BaseException:
            if child.poll() is None:
                child.kill(); child.wait()
            raise
    elapsed = time.monotonic() - start
    result = {'status': 'timeout' if timeout else 'complete' if status == 0 else 'failed',
              'worker_exit_code': status, 'context': context, 'deadline_seconds': seconds,
              'elapsed_seconds': elapsed, 'deadline_overrun_seconds': max(0., elapsed - seconds),
              'kill_scope': 'direct ACL worker; short read-only Git children; scheduled step owns cleanup',
              'deadline_caveat': 'OS reaping latency measured, not real-time guarantee'}
    (output / 'supervisor.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n',
                                           encoding='utf-8', newline='\n')
    return 124 if timeout else 0 if status == 0 else 2


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--phase', choices=PHASES)
    p.add_argument('--execute', action='store_true'); p.add_argument('--seed', type=int)
    p.add_argument('--repeat-start', type=int)
    for name in ('source-commit', 'release-record', 'upstream-root', 'upstream-manifest',
                 'prepared-root', 'training-run', 'artifact-index', 'replay-policy', 'output'):
        p.add_argument('--' + name)
    return p


def main():
    args = parser().parse_args()
    try:
        config = json.loads(args.config.read_bytes()); digest = validate_config(config)
        if not args.execute:
            print(json.dumps({'status': 'static_only', 'protocol': PROTOCOL, 'config_sha256': digest,
                              'requires_separate_supervisor_release': True})); return 0
        if any(getattr(args, k) is None for k in ('phase', 'source_commit', 'release_record', 'output')):
            raise ValueError('exact phase/source/release and fresh output required')
        if args.phase != 'analyze' and args.upstream_root is None:
            raise ValueError('verified external upstream checkout required')
        if args.phase in ('train', 'replay', 'eval') and args.prepared_root is None:
            raise ValueError('frozen prepared inputs required')
        if args.phase in ('replay', 'eval') and args.training_run is None:
            raise ValueError('reviewed training lineage required')
        if args.phase == 'analyze' and (args.artifact_index is None or args.prepared_root is None):
            raise ValueError('reviewed complete artifact index and development inputs required')
        from .hgcn_replay import load_policy
        policy = load_policy(args.replay_policy, config) if args.replay_policy else None
        released = verify_release(config, args.phase, args.source_commit, args.release_record,
                                  args.seed, args.repeat_start, policy)
        command = [sys.executable, '-m', 'acl_hct.hgcn_validation']
        for name, value in vars(args).items():
            if name != 'execute' and value is not None:
                command.extend(['--' + name.replace('_', '-'), str(value)])
        return supervise(command, args.output, role(config, args.phase, args.seed, args.repeat_start)['worker_seconds'], released)
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())

"""Process deadline and explicit supervisor gate for HGCN engineering only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .hgcn_registration import PROTOCOL, validate_config, verify_release


def supervise(command, output, seconds, context, environment=None):
    if type(seconds) not in (int, float) or not 0 < seconds <= 840:
        raise ValueError('bounded <=840s engineering worker required')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('fresh empty output required; no overwrite/resume')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    env = dict(os.environ if environment is None else environment)
    env['ACL_HGCN_SUPERVISOR_PID'] = str(os.getpid())
    env['ACL_HGCN_DEADLINE'] = str(started + seconds)
    timed_out = False
    with (output / 'worker.log').open('xb') as log:
        worker = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            status = worker.wait(timeout=max(0., started + seconds - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True; worker.kill(); status = worker.wait()
        except BaseException:
            if worker.poll() is None:
                worker.kill(); worker.wait()
            raise
    elapsed = time.monotonic() - started
    row = {'status': 'timeout' if timed_out else 'complete' if status == 0 else 'failed',
           'worker_exit_code': status, 'context': context, 'deadline_seconds': seconds,
           'elapsed_seconds': elapsed, 'deadline_overrun_seconds': max(0., elapsed - seconds),
           'scope': 'engineering only; imports, input/source checks, fixtures and cost included',
           'kill_scope': 'direct ACL worker only; read-only local Git queries are short subprocesses; Slurm step owns cleanup',
           'deadline_caveat': 'OS reaping latency measured; not a real-time guarantee'}
    (output / 'supervisor.json').write_text(json.dumps(row, indent=2) + '\n', encoding='utf-8', newline='\n')
    return 124 if timed_out else 0 if status == 0 else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--phase', choices=('cpu_quality', 'cuda_quality'))
    parser.add_argument('--execute', action='store_true')
    for name in ('source-commit', 'release-record', 'upstream-root', 'upstream-manifest', 'prepared-root', 'output'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_bytes()); config_hash = validate_config(config)
        if not args.execute:
            print(json.dumps({'status': 'static_only', 'protocol': PROTOCOL, 'config_sha256': config_hash,
                              'phases': ['cpu_quality', 'cuda_quality'], 'science_released': False})); return 0
        if any(getattr(args, k) is None for k in ('phase', 'source_commit', 'release_record', 'upstream_root',
                                                'prepared_root', 'output')):
            raise ValueError('exact phase/source/release/upstream/prepared and fresh output required')
        release = verify_release(config, args.phase, args.source_commit, args.release_record)
        command = [sys.executable, '-m', 'acl_hct.hgcn_quality']
        for name, value in vars(args).items():
            if name != 'execute' and value is not None:
                command.extend(['--' + name.replace('_', '-'), str(value)])
        seconds = config['resources']['cpu_worker_seconds' if args.phase == 'cpu_quality' else 'cuda_worker_seconds']
        return supervise(command, args.output, seconds, release)
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())

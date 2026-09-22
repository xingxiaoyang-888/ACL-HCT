"""Separate reviewed-release supervisor for the HGCN amplitude comparison."""
import argparse
import json
from pathlib import Path
import sys

from .hgcn_rtsc_amplitude_protocol import (PHASES, validate_config,
                                           verify_release)
from .hgcn_rtsc_entry import supervise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'old-config', 'original-config', 'policy'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--phase', choices=PHASES)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--seed', type=int)
    p.add_argument('--arm', choices=('task_only', 'task_relation'))
    p.add_argument('--repeat-start', type=int)
    p.add_argument('--probe-steps', type=int)
    for name in ('source-commit', 'release-record', 'upstream-root', 'upstream-manifest',
                 'prepared-root', 'original-training-run', 'old-task-result',
                 'old-relation-result', 'new-task-result', 'new-relation-result',
                 'output'):
        p.add_argument('--' + name)
    return p


def main():
    args = parser().parse_args()
    try:
        config = json.loads(args.config.read_bytes())
        old = json.loads(args.old_config.read_bytes())
        original = json.loads(args.original_config.read_bytes())
        policy = json.loads(args.policy.read_bytes())
        digest = validate_config(config, original, policy, old)
        if not args.execute:
            print(json.dumps({'status': 'static_only', 'protocol': config['protocol'],
                              'config_sha256': digest,
                              'requires_reviewed_release': True}))
            return 0
        required = ['phase', 'source_commit', 'release_record', 'upstream_root', 'output']
        if args.phase in ('probe', 'train', 'final'):
            required += ['prepared_root', 'original_training_run']
        if args.phase == 'final':
            required += ['old_task_result', 'old_relation_result',
                         'new_task_result', 'new_relation_result']
        if any(getattr(args, name) is None for name in required):
            raise ValueError('exact phase, released inputs and fresh output required')
        if (args.phase == 'probe') != (args.probe_steps is not None):
            raise ValueError('explicit 1..32 probe steps only on discarded probe phase')
        if args.phase == 'probe' and not 1 <= args.probe_steps <= 32:
            raise ValueError('discarded probe limited to 32 steps')
        release = verify_release(config, original, policy, old, args.phase,
                                 args.release_record, args.source_commit,
                                 seed=args.seed, arm=args.arm, repeat_start=args.repeat_start,
                                 original_run=args.original_training_run,
                                 new_task_run=args.new_task_result,
                                 new_relation_run=args.new_relation_result,
                                 old_task_run=args.old_task_result,
                                 old_relation_run=args.old_relation_result)
        command = [sys.executable, '-m', 'acl_hct.hgcn_rtsc_amplitude_worker']
        for name, value in vars(args).items():
            if value is not None and name != 'execute':
                command.extend(['--' + name.replace('_', '-'), str(value)])
        return supervise(command, args.output, release['worker_seconds'])
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

"""Release-gated entry for the six-arm HGCN joint comparison."""
import argparse
import json
from pathlib import Path
import sys

from .hgcn_rtsc_entry import supervise
from .hgcn_rtsc_joint_protocol import (ARMS, PHASES, validate_config,
                                       verify_release)


RESULT_FLAGS = {
    'plain_task': 'plain_task_result',
    'plain_relation': 'plain_relation_result',
    'three_task': 'three_task_result',
    'three_relation': 'three_relation_result',
    'single_task': 'single_task_result',
    'single_relation': 'single_relation_result',
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'old-config', 'original-config', 'policy'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--phase', choices=PHASES)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--seed', type=int)
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--repeat-start', type=int)
    p.add_argument('--probe-steps', type=int)
    for name in ('source-commit', 'release-record', 'upstream-root',
                 'upstream-manifest', 'prepared-root', 'original-training-run',
                 'plain-task-result', 'plain-relation-result',
                 'three-task-result', 'three-relation-result',
                 'single-task-result', 'single-relation-result', 'output'):
        p.add_argument('--' + name)
    return p


def training_result_paths(args):
    return {arm: getattr(args, flag) for arm, flag in RESULT_FLAGS.items()}


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
        required = ['phase', 'source_commit', 'release_record',
                    'upstream_root', 'output']
        if args.phase in ('probe', 'train', 'final', 'benchmark'):
            required += ['prepared_root', 'original_training_run']
        if args.phase in ('final', 'benchmark'):
            required += list(RESULT_FLAGS.values())
        if any(getattr(args, name) is None for name in required):
            raise ValueError('exact phase, released inputs and fresh output required')
        if (args.phase == 'probe') != (args.probe_steps is not None):
            raise ValueError('explicit discarded probe steps required only on probe')
        if args.phase == 'probe' and not 1 <= args.probe_steps <= 32:
            raise ValueError('discarded probe limited to 32 updates')
        paths = training_result_paths(args) if args.phase in ('final', 'benchmark') else None
        release = verify_release(
            config, original, policy, old, args.phase, args.release_record,
            args.source_commit, seed=args.seed, arm=args.arm,
            repeat_start=args.repeat_start, original_run=args.original_training_run,
            training_runs=paths)
        command = [sys.executable, '-m', 'acl_hct.hgcn_rtsc_joint_worker']
        for name, value in vars(args).items():
            if value is not None and name != 'execute':
                command.extend(['--' + name.replace('_', '-'), str(value)])
        return supervise(command, args.output, release['worker_seconds'])
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

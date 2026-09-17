"""Process deadline for the registered fixed-weight diagnostic; no tensor imports.

Use this entry, rather than invoking the scientific worker directly. All worker
imports, verification, equivalence passes and archival share the 720 s deadline.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def failure_inventory(root, metadata):
    """Report atomic completed descriptors and the remaining fixed matrix."""
    summary = {}
    try:
        summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        pass
    try:
        progress = json.loads((root / 'progress.json').read_text(encoding='utf-8'))
        for name in ('active_cell', 'active_phase', 'pass_execution'):
            if name in progress:
                summary[name] = progress[name]
    except (OSError, ValueError):
        pass
    completed = []
    for batch in summary.get('batches', []):
        for cell in batch.get('cells', []):
            archive = cell.get('archive', {})
            if archive.get('manifest') and archive.get('array_file'):
                completed.append({'batch_step': batch['step'], **cell['identity']})
    expected = []
    if metadata:
        for step in (769, 770):
            for graph in ('unmasked', 'masked'):
                expected.append({'batch_step': step, 'graph': graph, 'fanout': 'full', 'replicate': None})
                expected.extend({'batch_step': step, 'graph': graph, 'fanout': 'f16', 'replicate': r} for r in range(8))
    def identity(cell):
        return tuple(cell[name] for name in ('batch_step', 'graph', 'fanout', 'replicate'))
    completed_ids = {identity(cell) for cell in completed}
    return {'context': metadata, 'completed_archived_cells': completed,
            'active_cell': summary.get('active_cell'),
            'active_phase': summary.get('active_phase'), 'pass_execution': summary.get('pass_execution'),
            'missing_registered_cells': [cell for cell in expected if identity(cell) not in completed_ids],
            'partial_results_are_complete': False,
            'completion_basis': 'atomic summary cell with archive descriptor; orphan files/tmp are not counted'}


def supervise(command, output, *, seconds=720, environment=None, metadata=None):
    """Kill only our direct worker at the deadline; retain its incremental files."""
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError('output must be empty; no overwrite/resume')
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    env = dict(os.environ if environment is None else environment)
    env['ACL_BACKBONE_FROZEN_SUPERVISOR_PID'] = str(os.getpid())
    env['ACL_BACKBONE_FROZEN_DEADLINE'] = str(started + seconds)
    with (root / 'worker.log').open('xb') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        timed_out = False
        try:
            status = process.wait(timeout=max(0, started + seconds - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            status = process.wait()
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
    elapsed = time.monotonic() - started
    record = {'status': 'timeout' if timed_out else ('complete' if status == 0 else 'failed'),
              'worker_exit_code': status, 'deadline_seconds': seconds,
              'elapsed_seconds': elapsed, 'deadline_overrun_seconds': max(0, elapsed - seconds),
              'deadline_scope': 'worker imports, verification, all passes and archive writes',
              'kill_scope': 'direct diagnostic worker only',
              'deadline_caveat': 'OS scheduling and process reaping latency are measured; not a real-time guarantee'}
    (root / 'supervisor.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    if timed_out or status != 0:
        failure = {'status': record['status'], 'worker_exit_code': status, **failure_inventory(root, metadata)}
        (root / 'supervisor-failure.json').write_text(json.dumps(failure, indent=2) + '\n', encoding='utf-8')
    return 124 if timed_out else (0 if status == 0 else 2)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'prepared-root', 'checkpoint', 'training-release',
                 'batch-history', 'approval-record', 'quality-record', 'source-commit', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--seed', type=int, choices=(11, 23), required=True)
    return parser.parse_args()


def main():
    args = arguments()
    command = [sys.executable, '-m', 'acl_hct.backbone_frozen_diagnosis']
    for name, value in vars(args).items():
        command.extend(['--' + name.replace('_', '-'), str(value)])
    try:
        return supervise(command, args.output, metadata={'seed': args.seed, 'protocol': 'E2-BACKBONE-FROZEN-v1'})
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

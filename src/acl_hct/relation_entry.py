"""Stdlib-only, bounded CPU entry for the released A archive localization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


PROTOCOL = 'E2-radial-components-v1/A'
MODULES = ('recovery_relations.py', 'relation_entry.py')


def raw_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lf_hash(path):
    return hashlib.sha256(Path(path).read_text(encoding='utf-8').encode()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'bindings', 'protocol', 'approval', 'quality', 'output'):
        cli.add_argument('--' + name, type=Path)
    cli.add_argument('--shards', nargs=4, type=Path)
    cli.add_argument('--source-commit')
    cli.add_argument('--execute', action='store_true')
    cli.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    return cli


def verify_release(args):
    require(all(getattr(args, name) is not None for name in (
        'config', 'bindings', 'protocol', 'approval', 'quality', 'output', 'shards', 'source_commit')),
        'explicit inputs, reviewed source/quality/approval and new output required')
    require(re.fullmatch('[0-9a-f]{40}', args.source_commit) is not None, 'full source commit required')
    approval = json.loads(args.approval.read_bytes())
    quality = json.loads(args.quality.read_bytes())
    bindings = json.loads(args.bindings.read_bytes())
    sources = {name: lf_hash(Path(__file__).parent / name) for name in MODULES}
    require(approval['status'] == 'approved_A_CPU' and approval['protocol'] == PROTOCOL
            and approval['source_commit'] == args.source_commit and approval['GPU_requested'] == 0,
            'released A CPU scope required')
    require(quality['status'] == 'passed' and quality['phase'] == 'A'
            and quality['new_source_lf_sha256'] == sources and approval['new_source_lf_sha256'] == sources,
            'reviewed new source bytes required')
    require(approval['quality_raw_sha256'] == raw_hash(args.quality)
            and approval['bindings_raw_sha256'] == raw_hash(args.bindings)
            and approval['protocol_lf_sha256'] == lf_hash(args.protocol)
            and quality['protocol_lf_sha256'] == lf_hash(args.protocol),
            'reviewed quality/bindings/protocol bytes required')
    require(raw_hash(args.config) == bindings['old_config_raw_sha256']
            and bindings['old_config_canonical_sha256'] == '8ecd462845cf6f413bf57186506cabf044dcd87a52b4b6d490744db4def27172'
            and bindings['old_science_source_commit'] == '293ca0beaa50dd9bedacf6f06001ec3b73b2b9b6'
            and bindings['no_new_graph_samples'] is True and bindings['A_runtime_prepared_graph_files'] == [],
            'frozen original configuration/archives only required')
    return {'source_commit': args.source_commit, 'new_source_lf_sha256': sources,
            'approval_raw_sha256': raw_hash(args.approval), 'quality_raw_sha256': raw_hash(args.quality),
            'bindings_raw_sha256': raw_hash(args.bindings), 'protocol_lf_sha256': lf_hash(args.protocol),
            'input_bindings_protocol_lf_sha256': bindings['supervisor_protocol_LF_sha256'],
            'release_verified': True}


def supervise(args, started):
    """1680 seconds covers release checks, worker imports, all I/O and archives."""
    provenance = verify_release(args)
    deadline = started + 1680
    require(not args.output.exists(), 'fresh A output required; no overwrite/resume')
    args.output.mkdir(parents=True)
    command = [sys.executable, '-m', 'acl_hct.relation_entry', '--execute', '--worker']
    for name in ('config', 'bindings', 'protocol', 'approval', 'quality', 'output', 'source_commit'):
        command.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
    command.extend(['--shards', *(str(p) for p in args.shards)])
    environment = dict(os.environ)
    environment['ACL_RELATION_LOCALIZATION_SUPERVISOR_PID'] = str(os.getpid())
    environment['ACL_RELATION_LOCALIZATION_DEADLINE'] = str(deadline)
    timed_out = False
    with (args.output / 'worker.log').open('xb') as stream:
        child = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        try:
            code = child.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
            child.kill()
            code = child.wait()
        except BaseException:
            if child.poll() is None:
                child.kill()
                child.wait()
            raise
    elapsed = time.monotonic() - started
    status = 'timeout' if timed_out else 'complete' if code == 0 else 'failed'
    record = {'status': status, 'worker_exit_code': code, 'deadline_seconds': 1680,
              'elapsed_seconds': elapsed, 'deadline_overrun_seconds': max(0, elapsed - 1680),
              'GPU_requested': 0, 'maximum_CPU_threads': 4, 'provenance': provenance,
              'deadline_scope': 'release checks, numeric imports, full old archive verification, analysis and output',
              'partial_results_are_complete': False if status != 'complete' else None}
    (args.output / 'supervisor.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8', newline='\n')
    return 124 if timed_out else 0 if code == 0 else 2


def main():
    started = time.monotonic()
    args = parser().parse_args()
    try:
        if not args.execute:
            require(not args.worker, 'worker requires execute')
            print(json.dumps({'status': 'static_only', 'protocol': PROTOCOL, 'GPU_requested': 0,
                              'maximum_CPU_threads': 4, 'worker_seconds': 1680, 'allocation_seconds': 1800}))
            return 0
        if args.worker:
            provenance = verify_release(args)
            from .recovery_relations import worker
            worker(args, provenance)
            return 0
        return supervise(args, started)
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

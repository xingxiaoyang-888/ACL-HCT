"""Read-only complete validation of an externally identified protocol-B checkpoint."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import time

import torch
from .backbone import LorentzMeanNetwork
from .mechanisms import dependency_hashes, source_identity
from .train import TrainConfig, load_prepared, synchronize


REQUIRED_TRAINING_SOURCES = {
    'acl_hct/'+name for name in ('__init__.py', 'train.py', 'backbone.py',
    'ranking.py', 'protocols.py', 'mechanisms.py', 'geometry.py', 'aggregation.py')
}


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return _stream_sha256(stream)


def _stream_sha256(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        result.update(block)
    return result.hexdigest()


def load_verified_checkpoint(path, expected_sha256, expected_training_commit, training_release):
    """Verify artifact before loading; verify historical source without importing it.

    The externally supplied artifact hash is the trust anchor. A release directory
    verifies the checkpoint's recorded source bytes, not an independent Git claim.
    """
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256):
        raise ValueError('expected checkpoint SHA256 must be 64 lowercase hex digits')
    if not re.fullmatch(r'[0-9a-f]{40}', expected_training_commit):
        raise ValueError('expected training commit must be a full lowercase SHA')
    with Path(path).open('rb') as stream:
        if _stream_sha256(stream) != expected_sha256:
            raise ValueError('checkpoint SHA256 mismatch')
        stream.seek(0)
        checkpoint = torch.load(stream, map_location='cpu', weights_only=True)
    if checkpoint['source']['source_commit'] != expected_training_commit:
        raise ValueError('training source commit mismatch')
    hashes = checkpoint['source_sha256_normalized_lf']
    if not REQUIRED_TRAINING_SOURCES.issubset(hashes):
        raise ValueError('checkpoint lacks required training source hashes')
    root = (Path(training_release) / 'src').resolve()
    for name, expected in hashes.items():
        if not re.fullmatch(r'acl_hct/[A-Za-z_][A-Za-z_0-9]*\.py', name):
            raise ValueError('invalid training source path')
        source = (root / name).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError('missing or escaping training release source')
        actual = hashlib.sha256(source.read_text(encoding='utf-8').encode()).hexdigest()
        if actual != expected:
            raise ValueError(f'training release source hash mismatch: {name}')
    if checkpoint['run_status'] == 'failed':
        raise ValueError('failed checkpoints require explicit recovery review')
    if type(checkpoint['completed_steps']) is not int or checkpoint['completed_steps'] < 0:
        raise ValueError('invalid checkpoint step count')
    config = TrainConfig(**checkpoint['config'])
    config.validate()
    return checkpoint, config


def evaluate(prepared_root, checkpoint_path, expected_checkpoint_sha256,
             expected_training_commit, training_release, *, device='cpu',
             max_seconds=600., candidate_chunk=4096, threads=2, source_commit=None):
    """Evaluate all valid queries and candidates; never train, resume or save weights.

    The deadline is checked between verification, encoding and ranking children.
    One indivisible phase/child can overrun it; allocation wall time is the hard cap.
    Partial ranks are explicitly labeled and cannot become a full-valid score.
    """
    from .ranking import filtered_parent_ranks
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 3600:
        raise ValueError('max_seconds must be positive and <=3600')
    if type(candidate_chunk) is not int or not 1 <= candidate_chunk <= 8192:
        raise ValueError('candidate_chunk must be in [1,8192]')
    if type(threads) is not int or not 1 <= threads <= 8:
        raise ValueError('threads must be in [1,8]')
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda') or device.index not in (None, 0):
        raise ValueError('CPU or the single allocated CUDA device only')
    if device.type == 'cuda':
        if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise ValueError('CUDA evaluation requires Slurm allocation and preserved GPU binding')
        if torch.cuda.device_count() != 1:
            raise ValueError('exactly one visible allocated GPU required')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
    started = time.perf_counter()
    identity = source_identity(source_commit)
    if identity['source_commit'] is None:
        raise ValueError('archive evaluation requires --source-commit')
    torch.set_num_threads(threads)
    checkpoint, config = load_verified_checkpoint(
        checkpoint_path, expected_checkpoint_sha256, expected_training_commit, training_release)
    data = load_prepared(prepared_root)
    if checkpoint['manifest_hash'] != data['manifest_hash'] or checkpoint['valid_hash'] != data['valid_hash']:
        raise ValueError('checkpoint/prepared manifest or validation hash mismatch')
    source_hashes = dependency_hashes()
    source_hashes['acl_hct/evaluate_checkpoint.py'] = hashlib.sha256(
        Path(__file__).read_text(encoding='utf-8').encode()).hexdigest()
    report = {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'read-only full validation for cost/development; no checkpoint selection',
        'status': 'incomplete_time_limit', 'selection_performed': False,
        'checkpoint_sha256': expected_checkpoint_sha256,
        'checkpoint_bytes': Path(checkpoint_path).stat().st_size,
        'checkpoint_completed_steps': checkpoint['completed_steps'],
        'checkpoint_run_status': checkpoint['run_status'],
        'checkpoint_selection_status_recorded': checkpoint['selection_status'],
        'checkpoint_selection_caveat': 'historical metadata does not establish that these weights are selected best',
        'training_source': checkpoint['source'],
        'training_source_sha256_normalized_lf': checkpoint['source_sha256_normalized_lf'],
        'training_source_verification': 'all recorded source hashes matched supplied release; commit is checkpoint/caller metadata',
        'evaluation_source': identity, 'evaluation_source_sha256_normalized_lf': source_hashes,
        'training_config': checkpoint['config'], 'manifest_hash': data['manifest_hash'],
        'validation_queries_hash': data['valid_hash'], 'protocol': data['manifest']['protocol'],
        'entity_count': len(data['nodes']), 'full_valid_query_count': len(data['valid']),
        'full_valid_child_count': len({child for _, child in data['valid']}),
        'encoding': 'both layers full observed neighborhoods; no sampling or query masking',
        'candidate_scope': 'all entity IDs; filter self and other valid true parents per child',
        'precision': 'FP32; CUDA TF32 disabled; no mixed precision',
        'checkpoint_device_type': checkpoint['device_type'], 'evaluation_device_type': device.type,
        'device': torch.cuda.get_device_name(0) if device.type == 'cuda' else platform.processor(),
        'torch': str(torch.__version__), 'python': platform.python_version(),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'max_seconds': max_seconds, 'candidate_chunk': candidate_chunk, 'threads': threads,
        'budget_check': 'between phases and ranking children; an indivisible phase/child may overrun',
        'verification_seconds': time.perf_counter() - started,
        'model_setup_seconds': None, 'full_graph_encoding_seconds': None,
        'ranking_seconds': None, 'ranking': None, 'full_valid_query_micro_mrr': None,
        'peak_cuda_allocated_bytes': None, 'peak_cuda_reserved_bytes': None,
    }
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    def remaining():
        return max_seconds - (time.perf_counter() - started)

    def finish():
        if file_sha256(checkpoint_path) != expected_checkpoint_sha256:
            raise ValueError('checkpoint changed during read-only evaluation')
        report['total_seconds'] = time.perf_counter() - started
        if device.type == 'cuda':
            report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
        return report

    if remaining() <= 0:
        report['stopped_before'] = 'model_setup'
        return finish()
    begin = time.perf_counter()
    model = LorentzMeanNetwork(data['features'].shape[1], config.hidden, config.c,
                               config.scaled_radius, config.head_hidden)
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all()
           for value in checkpoint['model'].values()):
        raise ValueError('checkpoint model must contain finite FP32 weights')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device).eval().requires_grad_(False)
    features = data['features'].to(device)
    synchronize(device)
    report['model_setup_seconds'] = time.perf_counter() - begin
    if remaining() <= 0:
        report['stopped_before'] = 'full_encoding'
        return finish()
    begin = time.perf_counter()
    with torch.no_grad():
        points, _, _ = model.encode(features, data['neighbors'], [data['neighbors']] * 2,
                                    method='none', max_padded_messages=config.max_padded_messages)
    synchronize(device)
    report['full_graph_encoding_seconds'] = time.perf_counter() - begin
    if remaining() <= 0:
        report['stopped_before'] = 'ranking'
        return finish()
    true_parents = defaultdict(set)
    for parent, child in data['valid']:
        true_parents[child].add(parent)
    begin = time.perf_counter()
    result = filtered_parent_ranks(model, points, data['valid'], true_parents,
                                   candidate_chunk=candidate_chunk, max_seconds=remaining())
    synchronize(device)
    report['ranking_seconds'] = time.perf_counter() - begin
    report['ranking'] = result
    report['status'] = result['status']
    if result['status'] == 'complete':
        report['full_valid_query_micro_mrr'] = result['query_micro_mrr']
    return finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'checkpoint', 'training-release', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-checkpoint-sha256', required=True)
    parser.add_argument('--expected-training-commit', required=True)
    parser.add_argument('--source-commit', help='Full evaluation release commit; mandatory without Git metadata')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--max-seconds', type=float, default=600.)
    parser.add_argument('--candidate-chunk', type=int, default=4096)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output must be a new JSON path')
    if args.output.suffix.lower() != '.json':
        raise ValueError('output must have a .json extension')
    result = evaluate(args.prepared, args.checkpoint, args.expected_checkpoint_sha256,
                      args.expected_training_commit, args.training_release, device=args.device,
                      max_seconds=args.max_seconds, candidate_chunk=args.candidate_chunk,
                      threads=args.threads, source_commit=args.source_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'status': result['status'],
                      'full_valid_queries': result['full_valid_query_count'],
                      'selection_performed': False}))


if __name__ == '__main__':
    main()

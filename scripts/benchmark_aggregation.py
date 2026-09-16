"""Bounded CPU-only aggregation microbenchmark; no GPU or training jobs."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path
import torch
from acl_hct.aggregation import correct, correct_batched
from acl_hct.geometry import from_spatial


def run(nodes=32, repeats=3):
    if not 1 <= nodes <= 128 or not 1 <= repeats <= 10:
        raise ValueError('bounded benchmark: nodes 1..128, repeats 1..10')
    torch.set_num_threads(2)
    g = torch.Generator().manual_seed(91)
    initial = torch.randn(nodes, 8, 4, dtype=torch.float64, generator=g)*.1
    mask = torch.ones(nodes, 8, dtype=torch.bool)
    result = {}
    for method in ('none', 'third', 'jackknife'):
        result[method] = {}
        for batched in (False, True):
            times = []
            for iteration in range(repeats+1):
                u = initial.clone().requires_grad_()
                started = time.perf_counter()
                points = from_spatial(u)
                if batched:
                    out, _ = correct_batched(points, 16, mask, method=method)
                else:
                    out = torch.stack([correct(row, 16, method=method)[0] for row in points])
                out[...,1:].square().sum().backward()
                elapsed = time.perf_counter()-started
                if not torch.isfinite(u.grad).all():
                    raise RuntimeError('nonfinite gradient')
                if iteration: times.append(elapsed)
            result[method]['batched' if batched else 'reference'] = {
                'seconds': times, 'median_seconds': statistics.median(times)}
    return {'scope': 'CPU FP64 forward+backward including validation and diagnostics; equal padded lengths; no production/GPU speed claim',
            'nodes': nodes, 'k': 8, 'N': 16, 'spatial_dim': 4, 'threads': 2,
            'warmups': 1, 'repeats': repeats, 'python': platform.python_version(),
            'torch': torch.__version__, 'platform': platform.platform(), 'results': result}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nodes', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    result = run(args.nodes, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))

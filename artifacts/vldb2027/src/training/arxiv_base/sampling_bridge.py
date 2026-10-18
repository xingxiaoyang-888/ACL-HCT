"""Source-only bridge to the accepted native uniform sampler, with a new RNG domain.

The runner injects the accepted uniform_edges and native_planning_threads;
this module does not import old runners or read public/private task files.
Graph.kind='sage' selects only the legacy sampler's unweighted public-graph
storage convention. Native model self/aggregation semantics belong to adapters.
All planning, block validation and transfers must be charged to the request.
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable, Sequence

import numpy as np
import torch

from shared_blocks import NativeBlock, validate_block_chain


ROUND_DOMAIN = "dasfaa-cross-model-correction-v1"


def plan_seed(*, stage: str, training_seed: int, epoch: int, repeat: int,
              batch: int, layer: int, path: str = "main") -> int:
    """Arms deliberately do not enter the key: their sampling is paired.

    The native DGL single-thread sampler owns its process-local RNG. This
    deterministic key never draws from NumPy/Torch/dropout global RNGs.
    """
    if not stage or not path or not isinstance(stage, str) or not isinstance(path, str):
        raise ValueError("explicit stage and path required")
    numbers = (training_seed, epoch, repeat, batch, layer)
    if any(type(v) is not int or v < 0 for v in numbers):
        raise ValueError("nonnegative integer random-domain fields required")
    value = json.dumps([ROUND_DOMAIN, stage, *numbers, path], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


class UniformPlanningBridge:
    def __init__(self, native_uniform_edges: Callable, native_thread_scope: Callable):
        self.native_uniform_edges = native_uniform_edges
        self.native_thread_scope = native_thread_scope

    def plan(self, graph, targets: np.ndarray, fanouts: Sequence[int], *, stage: str,
             training_seed: int, epoch: int = 0, repeat: int = 0,
             batch: int = 0, path: str = "main") -> list[NativeBlock]:
        if graph.kind != "sage":
            raise ValueError("unweighted nonself public graph storage required")
        if not fanouts or any(type(k) is not int or (k != -1 and k < 1) for k in fanouts):
            raise ValueError("positive fanouts or exact full (-1) required")
        dst = np.asarray(targets)
        if (dst.dtype != np.int64 or dst.ndim != 1 or len(dst) == 0
                or len(np.unique(dst)) != len(dst)
                or np.any(dst < 0) or np.any(dst >= graph.n)):
            raise ValueError("distinct, nonempty, in-range int64 target vector required")
        blocks: list[NativeBlock] = []
        # One scope for a whole dependency cone, preserving accepted DGL/OpenMP
        # restoration behavior; individual native calls can nest in this scope.
        with self.native_thread_scope():
            for layer in reversed(range(len(fanouts))):
                salt = plan_seed(stage=stage, training_seed=training_seed, epoch=epoch,
                                 repeat=repeat, batch=batch, layer=layer, path=path)
                source, rows, _legacy_coefficients, reported_counts = self.native_uniform_edges(
                    graph, dst, fanouts[layer], salt)
                source = np.asarray(source, dtype=np.int64)
                rows = np.asarray(rows, dtype=np.int64)
                src = np.unique(np.concatenate((source, dst)))
                # Only plain topology passes the bridge. Legal self/HT/attention
                # coefficients are recomputed by each registered native adapter.
                block = NativeBlock(
                    torch.from_numpy(src), torch.from_numpy(dst.copy()),
                    torch.from_numpy(np.searchsorted(src, source)), torch.from_numpy(rows),
                    torch.from_numpy(np.searchsorted(src, dst)),
                    torch.from_numpy(graph.degree[dst].astype(np.int64, copy=True)),
                    sampler="full" if fanouts[layer] == -1 else "uniform_wor")
                if not np.array_equal(block.counts.numpy(), reported_counts):
                    raise ValueError("native count/edge alignment mismatch")
                expected = graph.degree[dst] if fanouts[layer] == -1 else np.minimum(
                    graph.degree[dst], fanouts[layer])
                if not np.array_equal(reported_counts, expected):
                    raise ValueError("native sampler did not return the registered WOR counts")
                blocks.insert(0, block)
                dst = src
        validate_block_chain(blocks)
        return blocks

"""Required real-upstream artificial-graph gates; no datasets downloaded."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch

from .hgcn_sampling import make_plan, mask_graph
from .mature_hgcn import MatureHGCN, encoder_args, to_lorentz_fp64
from .text_capacity import filtered_parent_ranks, prepare_scores, score_cached


@contextmanager
def default_dtype(dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _close(a, b, dtype):
    atol, rtol = (2e-6, 2e-5) if dtype == torch.float32 else (1e-10, 1e-9)
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    return float((a - b).abs().max()) if a.numel() else 0.


def run_fixture(upstream, device='cpu', archive_dir=None):
    device = torch.device(device)
    neighbors = [[], [2], [1, 3, 4, 5, 6, 7, 8], [2, 4], [2, 3], [2], [2], [2], [2]]
    rows = [i for i, row in enumerate(neighbors) for _ in row]
    columns = [j for row in neighbors for j in row]
    scipy_adj = sp.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(9, 9))
    cases = []; evidence = []; checkpoints = []
    torch.manual_seed(4417)
    for dtype in (torch.float32, torch.float64):
        with default_dtype(dtype):
            source_features = torch.randn(9, 4, dtype=dtype) * .1
            official_adj, _ = upstream.data.process(scipy_adj, source_features.numpy(), True, False)
            full = make_plan(neighbors, None, torch.Generator().manual_seed(101))
            full_adj = full.matrix(device=device, dtype=dtype)
            adjacency_error = _close(full_adj.to_dense().cpu(), official_adj.to_dense(), dtype)
            model = MatureHGCN(upstream, 4, 6, 5, device=device).to(dtype=dtype)
            official = upstream.HGCN(model.curvature.clone(), encoder_args(4, 6, 1., 0., device)).to(device=device, dtype=dtype)
            official.load_state_dict(model.encoder.state_dict(), strict=True)
            model.eval(); official.eval()
            features = source_features.to(device).detach().clone().requires_grad_()
            other_features = source_features.to(device).detach().clone().requires_grad_()
            actual = model.encode(features, [full_adj, full_adj])
            expected = official.encode(other_features, official_adj.to(device=device, dtype=dtype))
            output_error = _close(actual, expected, dtype)
            multiplier = torch.arange(actual.numel(), dtype=dtype, device=device).reshape_as(actual) / actual.numel()
            (actual * multiplier).sum().backward(); (expected * multiplier).sum().backward()
            gradient_error = _close(features.grad, other_features.grad, dtype)
            for (name, param), (other_name, other_param) in zip(model.encoder.named_parameters(), official.named_parameters()):
                if name != other_name or param.grad is None or other_param.grad is None:
                    raise ValueError('official gradient parameter coverage mismatch')
                if not torch.isfinite(param.grad).all():
                    raise ValueError('nonfinite official gradient')
                gradient_error = max(gradient_error, _close(param.grad, other_param.grad, dtype))
            output_gradients = {name: p.grad.detach().clone() for name, p in model.encoder.named_parameters()}
            official_gradients = {name: p.grad.detach().clone() for name, p in official.named_parameters()}
            # f >= max degree is identical and consumes no random state.
            rng = torch.Generator().manual_seed(101); before = rng.get_state().clone()
            complete = make_plan(neighbors, 7, rng)
            if not torch.equal(before, rng.get_state()) or not torch.equal(complete.indices, full.indices):
                raise ValueError('full sampling changed order or random state')
            if not torch.equal(complete.weights, full.weights):
                raise ValueError('full sampling changed adjacency weights')
            model.zero_grad(set_to_none=True)
            complete_features = features.detach().clone().requires_grad_()
            complete_output = model.encode(complete_features, [complete.matrix(device=device, dtype=dtype)] * 2)
            full_sampling_error = _close(complete_output, actual.detach(), dtype)
            (complete_output * multiplier).sum().backward()
            full_sampling_gradient_error = _close(complete_features.grad, features.grad, dtype)
            complete_gradients = {name: p.grad.detach().clone() for name, p in model.encoder.named_parameters()}
            for name, param in model.encoder.named_parameters():
                full_sampling_gradient_error = max(full_sampling_gradient_error, _close(param.grad, output_gradients[name], dtype))
            # Different plans must actually reach their respective official layers.
            p0 = make_plan(neighbors, 2, torch.Generator().manual_seed(121))
            p1 = make_plan(neighbors, 2, torch.Generator().manual_seed(129))
            if torch.equal(p0.indices, p1.indices):
                raise ValueError('fixture needs different independent layer plans')
            a0, a1 = [p.matrix(device=device, dtype=dtype) for p in (p0, p1)]
            sampled = model.encode(features.detach(), [a0, a1])
            manifold = official.manifold
            h = manifold.proj(manifold.expmap0(manifold.proj_tan0(features.detach(), official.curvatures[0]),
                                              c=official.curvatures[0]), c=official.curvatures[0])
            h, _ = official.layers[0]((h, a0)); h, _ = official.layers[1]((h, a1))
            layer_plan_error = _close(sampled, h, dtype)
            if not torch.isfinite(sampled).all():
                raise ValueError('nonfinite sampled points')
            native = actual.detach()
            lorentz = to_lorentz_fp64(native)
            converted = manifold.to_hyperboloid(native.double(), model.curvature.double())
            conversion_error = _close(lorentz, converted, torch.float64)
            # Direct, shared-head cache and complete filtered ranking check.
            query = torch.tensor([[1, 3], [2, 3], [2, 8]], dtype=torch.long, device=device)
            coordinates = model.tangent(native)
            cache = prepare_scores(model, coordinates)
            cache_error = _close(model.score(native, query), score_cached(model, cache, query[:, 0], query[:, 1]), dtype)
            ranking = filtered_parent_ranks(model, coordinates, query.cpu().tolist(), {3: {1, 2}, 8: {2}}, 3)
            if ranking['status'] != 'complete' or ranking['completed_queries'] != 3:
                raise ValueError('artificial ranking coverage failure')
            evidence.append({'dtype': str(dtype), 'features': features.detach().clone(),
                             'full_plan': full.archive(), 'full_adjacency': {'indices': full_adj.coalesce().indices(),
                                                                            'values': full_adj.coalesce().values()},
                             'official_adjacency': {'indices': official_adj._indices(), 'values': official_adj._values()},
                             'layer_plans': [p0.archive(), p1.archive()],
                             'official_output': expected.detach(), 'adapted_output': native,
                             'sampled_output': sampled.detach(), 'manual_official_sampled_output': h.detach(),
                             'loss_multiplier': multiplier, 'adapted_input_gradient': features.grad.clone(),
                             'official_input_gradient': other_features.grad.clone(),
                             'adapted_parameter_gradients': output_gradients, 'official_parameter_gradients': official_gradients,
                             'full_sampling_input_gradient': complete_features.grad.clone(),
                             'full_sampling_parameter_gradients': complete_gradients,
                             'model_state': {k: v.detach().clone() for k, v in model.state_dict().items()},
                             'query': query, 'orthonormal_tangent': coordinates.detach(),
                             'direct_scores': model.score(native, query).detach(),
                             'cached_scores': score_cached(model, cache, query[:, 0], query[:, 1]),
                             'filtered_truth': {'3': [1, 2], '8': [2]}, 'ranking': ranking,
                             'lorentz_fp64': lorentz, 'official_isometry_fp64': converted})
            # Checkpoint state_dict roundtrip (all fixed curvatures move/cast).
            reloaded = MatureHGCN(upstream, 4, 6, 5, device=device).to(dtype=dtype)
            state = model.state_dict()
            if archive_dir is not None:
                root = Path(archive_dir); root.mkdir(parents=True, exist_ok=True)
                path = root / ('fixture-weights-' + str(dtype).split('.')[-1] + '.pt')
                if path.exists():
                    raise ValueError('fresh fixture checkpoint required')
                torch.save({'model': state, 'upstream_commit': upstream.identity['commit'],
                            'model_definition': 'PoincareBall fixed c1; official encoder 4->6->6; directed head 5'}, path)
                checkpoint = torch.load(path, map_location=device, weights_only=True)
                if checkpoint['upstream_commit'] != upstream.identity['commit']:
                    raise ValueError('checkpoint upstream provenance mismatch')
                state = checkpoint['model']
                checkpoints.append({'name': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            reloaded.load_state_dict(state, strict=True); reloaded.eval()
            reload_error = _close(reloaded.encode(features.detach(), [full_adj] * 2), native, dtype)
            reload_error = max(reload_error, _close(reloaded.score(native, query), model.score(native, query), dtype))
            # Finite BCE gradients through both official encoder and directed head.
            model.zero_grad(set_to_none=True)
            encoded = model.encode(features.detach(), [a0, a1])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model.score(encoded, query),
                                                                        torch.tensor([1., 0., 1.], device=device, dtype=dtype))
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('finite all-parameter directed training gradients required')
            # Duplicate queries remain duplicate supervised observations; mask is set-valued.
            masked = mask_graph(neighbors, [[1, 2], [1, 2]])
            if masked != mask_graph(neighbors, [[1, 2]]) or masked[1] or 1 in masked[2]:
                raise ValueError('postmask self/degree or duplicate target semantics failed')
            postmask = make_plan(masked, 2, torch.Generator().manual_seed(1))
            if postmask.populations[1] != 0 or postmask.matrix(dtype=dtype).to_dense()[1, 1] != 1:
                raise ValueError('empty postmask row must retain exact self identity')
            duplicate_query = query[[0, 0, 2]]
            duplicate_logits = model.score(native, duplicate_query)
            if not torch.equal(duplicate_logits[0], duplicate_logits[1]):
                raise ValueError('duplicate supervised query score changed')
            cases.append({'dtype': str(dtype), 'official_adjacency_max_error': adjacency_error,
                          'official_output_max_error': output_error, 'official_gradient_max_error': gradient_error,
                          'full_sampling_max_error': full_sampling_error, 'layer_plan_max_error': layer_plan_error,
                          'full_sampling_gradient_max_error': full_sampling_gradient_error,
                          'isometry_max_error': conversion_error, 'score_cache_max_error': cache_error,
                          'checkpoint_reload_max_error': reload_error, 'finite_directed_bce': float(loss.detach())})
    result = {'status': 'passed', 'scope': 'artificial CPU/CUDA engineering; not real-task training or sampling effect',
              'upstream': upstream.identity, 'device': str(device), 'cases': cases,
              'disk_checkpoint_roundtrips': checkpoints,
              'adjacency_order': 'adapted COO coalesces/sorts; official process can retain unsorted COO; tolerance includes roundoff'}
    if archive_dir is not None:
        from .diagnostic_archive import write_archive
        result['reproduction_archive'] = write_archive(archive_dir, 'official-fixture',
                                                       {'neighbors': neighbors, 'cases': evidence})
    return result

"""Faithful single-seed Disease LP example with matching best-state saving."""
from pathlib import Path

import numpy as np
import torch

from .diagnostic_archive import write_archive
from .hgcn_evidence import atomic_json, deadline, load_checkpoint, save_checkpoint
from .hgcn_registration import canonical
from .hgcn_quality import synchronize


def metric_row(metrics):
    result = {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in metrics.items()}
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError('finite original loss/ROC/AP required')
    return result


def official_early_stop(counter, epoch, settings):
    # Preserve the exact original unusual == and zero-based > behavior.
    return counter == settings['patience'] and epoch > settings['min_epochs']


def train_official(upstream, config, device, output, release):
    s = config['official_task']; args = upstream.parser.parse_args([])
    for name, value in s.items():
        if hasattr(args, name):
            setattr(args, name, value)
    args.cuda = -1 if device.type == 'cpu' else 0; args.device = str(device)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    source_data = Path(upstream.data.__file__).parents[1] / 'data' / args.dataset
    data = upstream.data.load_data(args, str(source_data))
    args.n_nodes, args.feat_dim = data['features'].shape
    args.nb_false_edges = len(data['train_edges_false']); args.nb_edges = len(data['train_edges'])
    model = upstream.LPModel(args).to(device)
    data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_reduce_freq, gamma=args.gamma)
    initial_np_rng = np.random.get_state(); history = []; best = None; best_metrics = model.init_metric_dict()
    counter = 0; stop = 'max_epochs'; best_points = None

    def cp_binding(epoch):
        return {'protocol': config['protocol'], 'config_sha256': canonical(config),
                'source_commit': release['source_commit'], 'upstream_commit': config['upstream_commit'],
                'task': 'official-disease-lp', 'seed': args.seed, 'epoch': epoch}

    for epoch in range(args.epochs):
        deadline(); model.train(); optimizer.zero_grad(set_to_none=True)
        points = model.encode(data['features'], data['adj_train_norm'])
        metrics = model.compute_metrics(points, data, 'train'); loss = metrics['loss']; loss.backward()
        if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('finite all-parameter official gradient gate failed')
        optimizer.step(); scheduler.step()
        row = {'epoch': epoch + 1, 'train': metric_row(metrics), 'lr': optimizer.param_groups[0]['lr']}
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            with torch.no_grad():
                points = model.encode(data['features'], data['adj_train_norm'])
                val = model.compute_metrics(points, data, 'val')
            row['val'] = metric_row(val)
            if model.has_improved(best_metrics, val):
                best_metrics = val; counter = 0; best_points = points.detach().cpu().clone()
                checkpoint = save_checkpoint(output, f'official-best-epoch-{epoch + 1}.pt', model, cp_binding(epoch + 1))
                best = {'epoch': epoch + 1, 'validation': metric_row(val), 'checkpoint': checkpoint}
            else:
                counter += 1
                if official_early_stop(counter, epoch, s):
                    stop = 'original_early_stop'
        row['counter'] = counter; history.append(row)
        if (epoch + 1) % 25 == 0 or stop == 'original_early_stop':
            atomic_json(output / 'official-progress.json', {'status': 'running', 'epochs': epoch + 1,
                                                          'best': best, 'history': history})
        if stop == 'original_early_stop':
            break
    if best is None:
        raise ValueError('official strict validation best not established')
    last = save_checkpoint(output, 'official-last.pt', model, cp_binding(len(history)),
                           {'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict()})
    loaded = load_checkpoint(output, best['checkpoint'], cp_binding(best['epoch']))
    model.load_state_dict(loaded['model_state'], strict=True); model.eval(); deadline()
    with torch.no_grad():
        points = model.encode(data['features'], data['adj_train_norm'])
        torch.testing.assert_close(points.cpu(), best_points, atol=0, rtol=0)
        val = metric_row(model.compute_metrics(points, data, 'val'))
        test = metric_row(model.compute_metrics(points, data, 'test'))  # Test first evaluated after final selection.
    if val != best['validation']:
        raise ValueError('official best validation replay mismatch')
    adj = data['adj_train_norm'].coalesce(); synchronize(device)
    artifact = write_archive(output / 'arrays', 'official-full-example',
                             {'features': data['features'], 'adjacency_indices': adj.indices(), 'adjacency_values': adj.values(),
                              'edge_sets': {k: data[k] for k in ('train_edges', 'train_edges_false', 'val_edges',
                                                               'val_edges_false', 'test_edges', 'test_edges_false')},
                              'best_native_ball_points': points, 'validation': val, 'test': test,
                              'initial_training_numpy_rng': initial_np_rng, 'final_numpy_rng': np.random.get_state(),
                              'history': history, 'best_checkpoint': best['checkpoint']})
    return {'status': 'complete', 'epochs': len(history), 'stop': stop, 'best': best, 'last': last,
            'test_after_final_best_reload': test, 'archive': artifact,
            'negative_pool_boundary': s['negative_pool'], 'driver_differences': s['driver_differences'],
            'claim_scope': s['claim_scope'], 'Slurm_binding_preserved': True}

"""Exactly three authorized training seeds. No OGB test()/dataset import."""
import argparse
from pathlib import Path
import time
import numpy as np
from common import torch, configure, domain, release, fresh_output, accuracy_counts, write_json, sha
from runtime import load_public, labels, path_forward
from sage import SAGE
from blocks import plan
from torch.nn import functional as F


CONFIG = {'layers':3, 'hidden':256, 'dropout':.5, 'lr':.01, 'weight_decay':0.,
          'epochs':500, 'fanouts':[15,15,15], 'batch':1024, 'selection_every':5, 'selection_repeats':4}


def epoch_ids(ids, seed, epoch):
    return np.random.default_rng(domain('train-shuffle', seed, epoch)).permutation(np.asarray(ids, dtype=np.int64))


@torch.no_grad()
def select(model, graph, ids, fit_labels, seed, device):
    model.eval()
    counts = []
    for repeat in range(4):
        correct = 0
        for batch, target in enumerate(ids['batches']['fit']):
            # Stable across ALL selection epochs; no gate or confirmation input.
            output, _ = path_forward(model, graph, target, {'family':'S','fanouts':[15]*3},
                                      'cp-selection', seed, repeat, batch, device)
            correct += accuracy_counts(output, [fit_labels[n] for n in target])
        counts.append(correct)
    return counts


def run(args):
    public, dev = Path(args.public), Path(args.development)
    bound = {'graph':public/'graph.npz', 'ids':public/'ids.json', 'preparation':public/'preparation.json',
             'train_labels':dev/'train.npz', 'fit_labels':dev/'fit.npz'}
    bindings = release(args.release, 'train_sage', args.seed, bound)
    configure()
    graph, ids = load_public(public)
    if graph.kind != 'sage':
        raise ValueError('Training is only GraphSAGE/ogbn-arxiv')
    train_y, fit_y = labels(dev/'train.npz', ids['ids']['train']), labels(dev/'fit.npz', ids['ids']['fit'])
    out = fresh_output(args.output)
    torch.manual_seed(args.seed)
    if torch.device(args.device).type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    model = SAGE().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=0.)
    best_sum, best_epoch, updates = -1, None, 0
    start = time.perf_counter()
    with (out/'curve.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(1, 501):
            model.train()
            shuffled = epoch_ids(ids['ids']['train'], args.seed, epoch)
            loss_total, count, epoch_updates = 0., 0, 0
            for batch, offset in enumerate(range(0, len(shuffled), 1024)):
                targets = shuffled[offset:offset+1024]
                blocks = plan(graph, targets, [15]*3, 'train-sampling', args.seed, epoch, batch)
                x = graph.x[torch.from_numpy(blocks[0].src)].to(args.device)
                y = torch.tensor([train_y[int(n)] for n in targets], device=args.device)
                optimizer.zero_grad(set_to_none=True)
                output = model(x, blocks)
                loss = F.nll_loss(output, y)
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite training loss; no automatic retry')
                loss.backward()
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise ValueError('Nonfinite training gradient; no automatic retry')
                optimizer.step()
                loss_total += float(loss.detach())*len(targets)
                count += len(targets)
                updates += 1
                epoch_updates += 1
            record = {'epoch':epoch, 'train_nll':loss_total/count, 'train_nodes':count, 'epoch_updates':epoch_updates, 'updates':updates}
            if epoch % 5 == 0:
                counts = select(model, graph, ids, fit_y, args.seed, args.device)
                record.update(fit_correct=counts, fit_count=len(fit_y), selected=sum(counts)>best_sum)
                if sum(counts) > best_sum:
                    best_sum, best_epoch = sum(counts), epoch
                    payload = {'schema':'icml_ca_sage_weights_v1', 'seed':args.seed, 'epoch':epoch,
                               'model_state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, 'bindings':bindings}
                    temp = out/'selected.writing.pt'
                    torch.save(payload, temp)
                    temp.replace(out/'label-free-weights.pt')
            log.write(__import__('json').dumps(record)+'\n')
            log.flush()
            if epoch % 5 == 0:
                print(f'epoch={epoch} updates={updates} selected_epoch={best_epoch}', flush=True)
    write_json(out/'completion.json', {'schema':'icml_ca_training_complete_v1', 'seed':args.seed, 'config':CONFIG,
        'epochs_executed':500, 'updates':updates, 'selected_epoch':best_epoch, 'selected_correct_sum':best_sum,
        'selection_denominator':4*len(fit_y), 'wall_seconds':time.perf_counter()-start,
        'weights_sha256':sha(out/'label-free-weights.pt'), 'bindings':bindings, 'gate_labels_loaded':False,
        'confirmation_labels_loaded':False, 'test_labels_parsed':False})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('public', 'development', 'output', 'release'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--seed', type=int, choices=[11,23,37], required=True)
    p.add_argument('--device', default='cuda:0')
    run(p.parse_args())

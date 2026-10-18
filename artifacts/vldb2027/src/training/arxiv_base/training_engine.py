"""NEW, UNEXECUTED training engine. No CLI or implicit runtime permission.

The separately released runner supplies already authorized train/fit data,
one built native model, a freshly initialized optimizer and the bound native
planning bridge. No dataset downloads, gate/confirmation/test readers or
checkpoint loading are present here. It saves only this run's chosen weights.
Training timing includes selection and saving and is NOT inference latency.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable, Mapping

import numpy as np
import torch

from sampling_bridge import plan_seed


@dataclass(frozen=True)
class TrainingSpec:
    model: str
    arm: str
    seed: int
    stage: str
    epochs: int
    fanouts: tuple[int, int, int]
    batch_size: int
    selection_every: int
    selection_repeats: int
    gradient_clip: float | None = None
    # Identical within a model/stage across arms; no effect-dependent schedule.
    warmup_epochs: int = 0

    def validate(self):
        if self.model not in ('sage', 'gat', 'hgcn', 'hybonet'):
            raise ValueError('unregistered backbone')
        if self.arm not in ('plain', 'candidate', 'generic') or self.seed not in (11, 23, 37):
            raise ValueError('unregistered arm or training seed')
        if self.stage not in ('pretrain', 'adapt') or (self.stage == 'pretrain' and self.arm != 'plain'):
            raise ValueError('only plain can create a pretraining warm start')
        integers = (self.epochs, self.batch_size, self.selection_every, self.selection_repeats)
        if any(type(x) is not int or x < 1 for x in integers):
            raise ValueError('explicit positive training and selection counts required')
        if len(self.fanouts) != 3 or any(type(k) is not int or k < 1 for k in self.fanouts):
            raise ValueError('one finite positive training fanout per actual message layer')
        if type(self.warmup_epochs) is not int or not 0 <= self.warmup_epochs <= self.epochs:
            raise ValueError('invalid frozen learning-rate warmup')
        if self.gradient_clip is not None and (not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0):
            raise ValueError('invalid registered gradient clip')


def paired_batches(ids, *, spec: TrainingSpec, epoch: int):
    """Private permutation with no arm field and no global NumPy/Torch draws."""
    value = np.asarray(ids)
    if value.dtype != np.int64 or value.ndim != 1 or len(value) == 0:
        raise ValueError('nonempty int64 training IDs required')
    salt = plan_seed(stage=spec.stage + '-batch-order', training_seed=spec.seed,
                     epoch=epoch, repeat=0, batch=0, layer=0)
    shuffled = np.random.default_rng(salt).permutation(value)
    for offset in range(0, len(shuffled), spec.batch_size):
        yield shuffled[offset:offset + spec.batch_size]


def paired_dropout_seed(spec: TrainingSpec, epoch: int, batch: int) -> int:
    return plan_seed(stage=spec.stage + '-dropout', training_seed=spec.seed,
                     epoch=epoch, repeat=0, batch=batch, layer=0) % (2**63 - 1)


def fit_is_better(counts, best_sum):
    # Strictly greater selects the first maximum; epoch zero is not a candidate.
    return sum(counts) > best_sum


def _forward(model, data, bridge, score_fn, targets, spec, device, *,
             selection: bool, epoch: int, repeat: int, batch: int):
    # Integer metadata must be created outside inference_mode so immutable
    # tensor version fingerprints remain available through device transfers.
    with torch.inference_mode(False):
        blocks = bridge.plan(data.graph, targets, spec.fanouts,
                             stage=spec.stage + ('-fit-selection' if selection else '-train'),
                             training_seed=spec.seed, epoch=0 if selection else epoch,
                             repeat=repeat, batch=batch)
        features = data.graph.x.index_select(0, blocks[0].src_ids).to(device)
        moved = [block.to(device) for block in blocks]
    output = score_fn(model, features, moved)
    if output.ndim != 2 or output.shape[0] != len(targets) or not bool(torch.isfinite(output).all()):
        raise ValueError('invalid native task scores; stop without retry')
    return output


@torch.no_grad()
def select_fit(model, data, bridge, score_fn, spec: TrainingSpec, device):
    """Same fixed fit batches and streams across epochs/arms, at train fanout."""
    model.eval()
    counts = []
    fit_ids = np.asarray(data.fit_ids, dtype=np.int64)
    for repeat in range(spec.selection_repeats):
        correct = 0
        for batch, targets in enumerate(data.fit_batches):
            output = _forward(model, data, bridge, score_fn, targets, spec, device,
                              selection=True, epoch=0, repeat=repeat, batch=batch)
            labels = torch.as_tensor(data.labels_for('fit', targets), device=device, dtype=torch.long)
            correct += int((output.argmax(-1) == labels).sum())
        counts.append(correct)
    return counts


def _finite_gradients(model):
    # One Boolean reduction transfer rather than a separate GPU sync per tensor.
    checks = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    if not checks or not bool(torch.stack(checks).all()):
        raise ValueError('missing or nonfinite training gradient; stop without retry')


def _file_sha(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run_training(*, spec: TrainingSpec, data, model, optimizer, bridge,
                 score_fn: Callable, loss_fn: Callable, output_directory: Path,
                 bindings: Mapping, device: str):
    """One exact run. Caller must validate frozen release/source/input bindings.

    loss_fn(scores, labels) is the model's frozen native task loss only. There
    is no correction-size penalty, teacher, cached representation or auxiliary
    target. The selected model is never loaded back to continue optimization.
    """
    spec.validate()
    if not bindings or any(len(group['params']) == 0 for group in optimizer.param_groups):
        raise ValueError('bound inputs and nonempty fresh optimizer required')
    if optimizer.state:
        raise ValueError('all arms require fresh optimizer state at shared warm start')
    out = Path(output_directory)
    if out.exists():
        raise FileExistsError('new run directory required; no overwrite or resume')
    out.mkdir(parents=True, exist_ok=False)
    if any(p.device != torch.device(device) for p in model.parameters()):
        raise ValueError('runner must place the full model on the released device')
    learning_rates = [float(group['lr']) for group in optimizer.param_groups]
    model.train()
    updates, best_sum, best_epoch = 0, -1, None
    started = time.perf_counter()
    with (out / 'curve.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(1, spec.epochs + 1):
            model.train()
            factor = min(1., epoch / spec.warmup_epochs) if spec.warmup_epochs else 1.
            for group, rate in zip(optimizer.param_groups, learning_rates):
                group['lr'] = rate * factor
            total_loss, count, epoch_updates = 0., 0, 0
            for batch, targets in enumerate(paired_batches(data.train_ids, spec=spec, epoch=epoch)):
                # Reset per step, so correction initialization and intervening
                # fit evaluation cannot shift baseline dropout RNG streams.
                salt = paired_dropout_seed(spec, epoch, batch)
                torch.default_generator.manual_seed(salt)
                if torch.device(device).type == 'cuda':
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(salt)
                optimizer.zero_grad(set_to_none=True)
                output = _forward(model, data, bridge, score_fn, targets, spec, device,
                                  selection=False, epoch=epoch, repeat=0, batch=batch)
                labels = torch.as_tensor(data.labels_for('train', targets), device=device, dtype=torch.long)
                loss = loss_fn(output, labels)
                if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                    raise ValueError('invalid scalar native loss; stop without retry')
                loss.backward()
                _finite_gradients(model)
                if spec.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), spec.gradient_clip,
                                                   error_if_nonfinite=True)
                optimizer.step()
                total_loss += float(loss.detach()) * len(targets)
                count += len(targets)
                updates += 1
                epoch_updates += 1
            if count != len(data.train_ids):
                raise ValueError('epoch did not cover all authorized train IDs')
            record = {'epoch': epoch, 'updates': updates, 'epoch_updates': epoch_updates,
                      'train_nodes': count, 'native_train_loss': total_loss / count,
                      'learning_rates': [g['lr'] for g in optimizer.param_groups]}
            if epoch % spec.selection_every == 0 or epoch == spec.epochs:
                counts = select_fit(model, data, bridge, score_fn, spec, device)
                chosen = fit_is_better(counts, best_sum)
                record.update(fit_correct=counts, fit_count=len(data.fit_ids), selected=chosen)
                if chosen:
                    best_sum, best_epoch = sum(counts), epoch
                    payload = {'schema': 'dasfaa_shared_correction_weights_v1',
                               'model': spec.model, 'arm': spec.arm, 'seed': spec.seed,
                               'epoch': epoch, 'model_state': {k: v.detach().cpu().clone()
                                                             for k, v in model.state_dict().items()},
                               'config': asdict(spec), 'bindings': dict(bindings)}
                    temp = out / 'selected.writing.pt'
                    torch.save(payload, temp)
                    temp.replace(out / 'label-free-weights.pt')
                print(json.dumps({'epoch': epoch, 'updates': updates,
                                  'selected_epoch': best_epoch}), flush=True)
            log.write(json.dumps(record) + '\n')
            log.flush()
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    completion = {'schema': 'dasfaa_shared_training_complete_v1', 'spec': asdict(spec),
                  'epochs_executed': spec.epochs, 'updates': updates,
                  'selected_epoch': best_epoch, 'selected_correct_sum': best_sum,
                  'selection_denominator': spec.selection_repeats * len(data.fit_ids),
                  'program_wall_seconds': time.perf_counter() - started,
                  'program_wall_is_inference_latency': False,
                  'weights_sha256': _file_sha(out / 'label-free-weights.pt'),
                  'bindings': dict(bindings), 'gate_labels_loaded': False,
                  'confirmation_labels_loaded': False, 'test_labels_parsed': False}
    (out / 'training-completion.json').write_text(json.dumps(completion, indent=2) + '\n', encoding='utf-8')
    return completion

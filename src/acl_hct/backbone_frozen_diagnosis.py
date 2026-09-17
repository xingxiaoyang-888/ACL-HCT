"""Registered fixed-weight train-batch observations, with no optimizer or ranking.

Real-data use requires the separately reviewed approval/quality records and the
process supervisor. CPU tests call the observation helper on synthetic tensors.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from .backbone import LorentzMeanNetwork, make_plan
from .diagnostic_archive import write_archive
from .evaluate_checkpoint import file_sha256, load_verified_checkpoint
from .mechanisms import source_identity
from .protocols import digest, mask_indexed_queries


CONFIG_SHA256 = '38ae7ad55156fd2b9922eff424460c5961f9d7e43dab8135bd5a23d117e7f784'
DIAGNOSTIC_SOURCES = ('backbone_frozen_entry.py', 'backbone_frozen_diagnosis.py',
                      'backbone_frozen_inputs.py', 'backbone_frozen_readings.py',
                      'diagnostic_archive.py', 'evaluate_checkpoint.py', 'mechanisms.py',
                      'train.py', '__init__.py')


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_registration(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if digest(config) != CONFIG_SHA256:
        raise ValueError('configuration differs from the registered fixed matrix')
    return config


def sampling_seed(seed, batch_step, replicate):
    if type(replicate) is not int or not 0 <= replicate < 8:
        raise ValueError('replicate must be 0..7')
    token = f'E2-BACKBONE-FROZEN-v1|{seed}|{batch_step}|{replicate}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(token).digest()[:8], 'big') % (2 ** 63)


def two_layer_plans(neighbors, fanout, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return [make_plan(neighbors, fanout, generator) for _ in range(2)]


def tensor_hash(value):
    array = value.detach().cpu().contiguous().numpy()
    return digest({'shape': list(array.shape), 'dtype': array.dtype.str,
                   'bytes_sha256': hashlib.sha256(array.tobytes()).hexdigest()})


def weight_hash(model):
    return digest({name: tensor_hash(value) for name, value in model.state_dict().items()})


def condition_input_hash(features, neighbors, plans, queries, labels):
    return digest({'features': tensor_hash(features), 'neighbors': neighbors, 'plans': plans,
                   'queries': tensor_hash(queries), 'labels': tensor_hash(labels)})


def finite(value, name):
    if not torch.isfinite(value).all():
        raise ValueError(f'nonfinite {name}')


def observe_condition(model, features, neighbors, plans, queries, labels, *,
                      max_padded_messages=32768, equivalence=False, atol=2e-6, rtol=2e-5,
                      progress=None):
    """One mean-BCE backward with passive Linear hooks and optional plain pass.

    Hooks never replace or mutate an activation. Weights require autograd but
    are never updated. The plain pass uses the same stored plans and loss scale.
    """
    from .backbone_frozen_readings import layer_readings, parameter_gradient_readings
    if len(model.layers) != 2 or queries.shape != (labels.numel(), 2) or labels.ndim != 1:
        raise ValueError('two layers and flattened query/label records required')
    if len(labels) % 5 or not torch.equal(labels.reshape(-1, 5),
                                         labels.new_tensor([1, 0, 0, 0, 0]).expand(len(labels) // 5, 5)):
        raise ValueError('one positive and four negatives in each fixed group required')
    if model.training or not all(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('eval model with differentiable fixed parameters required')
    before_weights = weight_hash(model)
    before_inputs = condition_input_hash(features, neighbors, plans, queries, labels)
    plain = None

    def mark(phase, kind):
        if progress is not None:
            progress(phase, kind)

    def forward_backward(kind):
        mark('pass_started', kind)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            points, _, _ = model.encode(features, neighbors, plans, method='none',
                                         max_padded_messages=max_padded_messages)
            finite(points, 'points')
            logits = model.score(points, queries)
            bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            finite(logits, 'logits')
            finite(bce, 'BCE')
            loss = bce.mean()
            mark('forward_returned', kind)
            mark('backward_started', kind)
            loss.backward()
            mark('backward_returned', kind)
        parameter_gradient_readings(model)  # missing/nonfinite gradients fail both paths
        return points, logits, bce, loss

    if equivalence:
        points, logits, bce, loss = forward_backward('equivalence')
        plain = {'points': points.detach().clone(), 'logits': logits.detach().clone(),
                 'loss': loss.detach().clone(),
                 'grads': {name: parameter.grad.detach().clone()
                           for name, parameter in model.named_parameters()}}
        del points, logits, bce, loss
    activations = {}
    handles = []

    def hook(index):
        def collect(module, inputs, output):
            if index in activations:
                raise ValueError('Linear observed more than once')
            output.retain_grad()
            activations[index] = output
        return collect

    try:
        for index, layer in enumerate(model.layers):
            handles.append(layer.linear.register_forward_hook(hook(index)))
        points, logits, bce, loss = forward_backward('scientific')
        if set(activations) != {0, 1}:
            raise ValueError('incomplete two-layer observation')
        equivalence_record = {'performed': equivalence, 'atol': atol, 'rtol': rtol}
        if plain is not None:
            mark('equivalence_gate', 'scientific')
            deltas = {}
            for name, observed in (('points', points), ('logits', logits), ('loss', loss)):
                torch.testing.assert_close(observed.detach(), plain[name], atol=atol, rtol=rtol)
                deltas[name + '_max_abs'] = float((observed.detach() - plain[name]).abs().max())
            gradient_delta = 0.
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter.grad, plain['grads'][name], atol=atol, rtol=rtol)
                gradient_delta = max(gradient_delta, float((parameter.grad - plain['grads'][name]).abs().max()))
            equivalence_record.update(status='passed', parameter_grad_max_abs=gradient_delta, **deltas)
        positive = queries.reshape(-1, 5, 2)[:, 0]
        mark('readings', 'scientific')
        layers = []
        for index, layer in enumerate(model.layers):
            u = activations[index]
            if u.grad is None:
                raise ValueError('missing dL/du')
            finite(u, 'Linear output')
            finite(u.grad, 'dL/du')
            layers.append(layer_readings(u.detach(), u.grad.detach(), layer.c, layer.scaled_radius,
                                         positive[:, 0].cpu(), positive[:, 1].cpu()))
        logit_array = logits.detach().cpu().numpy().copy()
        bce_array = bce.detach().cpu().numpy().copy()
        grouped = logit_array.reshape(-1, 5)
        result = {'logits': logit_array, 'bce': bce_array,
                  'margins': grouped[:, 0] - grouped[:, 1:].mean(axis=1),
                  'mean_bce': float(loss.detach()), 'layers': layers,
                  'parameter_gradients': parameter_gradient_readings(model),
                  'equivalence': equivalence_record,
                  'weight_hash': before_weights, 'input_hash': before_inputs,
                  'scientific_forward_backward_count': 1,
                  'equivalence_forward_backward_count': int(equivalence)}
        if weight_hash(model) != before_weights or condition_input_hash(features, neighbors, plans, queries, labels) != before_inputs:
            raise ValueError('weights or condition inputs changed')
        return result
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)


def execution_source_hashes(config):
    root = Path(__file__).parent
    for name in config['execution_frozen_sources']:
        actual = hashlib.sha256((root / name).read_text(encoding='utf-8').encode()).hexdigest()
        if actual != config['training_source_lf_sha256'][name]:
            raise ValueError(f'frozen execution source differs: {name}')
    return {name: hashlib.sha256((root / name).read_text(encoding='utf-8').encode()).hexdigest()
            for name in sorted(set(DIAGNOSTIC_SOURCES) | set(config['execution_frozen_sources']))}


def verify_approval(approval_path, quality_path, config, identity, sources):
    """Fail closed before opening any prepared data or checkpoint."""
    approval = json.loads(Path(approval_path).read_text(encoding='utf-8'))
    quality = json.loads(Path(quality_path).read_text(encoding='utf-8'))
    required = {'user_authorized': True, 'scope': config['protocol'], 'config_sha256': CONFIG_SHA256,
                'source_commit': identity['source_commit'], 'source_lf_sha256': sources,
                'quality_review_accepted': True, 'entry_criteria_frozen': True,
                'quality_record_sha256': file_sha256(quality_path)}
    for name, expected in required.items():
        if approval.get(name) != expected:
            raise ValueError(f'approval missing or mismatched {name}')
    if not isinstance(approval.get('user_message_reference'), str) or not approval['user_message_reference'].strip():
        raise ValueError('exact user authorization reference required')
    if (quality.get('status') != 'passed' or quality.get('config_sha256') != CONFIG_SHA256
            or quality.get('source_lf_sha256') != sources or quality.get('real_data_executed') is not False):
        raise ValueError('reviewed engineering quality record mismatched')
    return {'approval_sha256': file_sha256(approval_path), 'quality_record_sha256': file_sha256(quality_path),
            'user_message_reference': approval['user_message_reference']}


def check_deadline():
    supervisor = os.environ.get('ACL_BACKBONE_FROZEN_SUPERVISOR_PID', '')
    deadline = float(os.environ.get('ACL_BACKBONE_FROZEN_DEADLINE', 'nan'))
    if not supervisor.isdigit() or int(supervisor) != os.getppid() or not math.isfinite(deadline):
        raise ValueError('real scientific worker requires the process supervisor')
    if time.monotonic() >= deadline:
        raise TimeoutError('registered diagnostic deadline exceeded')


def run_matrix(model, features, data, batches, config, training_seed, output, report, *, deadline=check_deadline):
    """Shared matrix/archival path; also exercised with synthetic CPU fixtures.

    Production callers pass verified data/weights/approval and the supervised
    deadline. This helper does not load files or authorize a real experiment.
    """
    from .backbone_frozen_inputs import immutable_input_hash
    from .backbone_frozen_readings import degree_readings, paired_comparisons
    whole_weights = weight_hash(model)
    whole_inputs = immutable_input_hash(data)
    report.update(status='incomplete', scientific_forward_backward_count=0,
                  equivalence_forward_backward_count=0, batches=[],
                  whole_job_before={'weights': whole_weights, 'inputs': whole_inputs},
                  count_basis='quality-passed scientific conditions and their plain passes',
                  pass_execution={kind: {'attempted': 0, 'forward_returned': 0, 'backward_returned': 0}
                                  for kind in ('scientific', 'equivalence')})
    def progress(phase, kind):
        report['active_phase'] = {'phase': phase, 'kind': kind}
        field = 'attempted' if phase == 'pass_started' else phase
        if field in report['pass_execution'][kind]:
            report['pass_execution'][kind][field] += 1
        # Phase updates stay small instead of repeatedly serializing all prior
        # layer summaries. Completed cell descriptors remain in summary.json.
        atomic_json(output / 'progress.json', {'active_cell': report.get('active_cell'),
                    'active_phase': report['active_phase'], 'pass_execution': report['pass_execution']})
    atomic_json(output / 'summary.json', report)
    for step in config['batch_steps']:
        chosen = batches[step]
        groups = data['query_groups'][chosen]
        queries = groups.reshape(-1, 2).to(features.device)
        labels = data['labels'][chosen].reshape(-1).to(features.device)
        positive = groups[:, 0]
        graphs = {'unmasked': data['neighbors'],
                  'masked': mask_indexed_queries(data['neighbors'], positive.tolist())}
        batch = {'step': step, 'group_ids_sha256': digest(chosen.tolist()),
                 'cells': [], 'comparisons': None}
        report['batches'].append(batch)
        batch['batch_archive'] = write_archive(output / 'private', f'batch-{step}',
                                               {'group_ids': chosen, 'query_groups': groups,
                                                'labels': labels, 'unmasked_positive_degree': np.asarray(
                                                    [[len(data['neighbors'][a]), len(data['neighbors'][b])] for a, b in positive.tolist()])})
        cells = {}
        # All four first-batch equivalence gates precede the other realizations.
        schedule = [(graph, 'full', None) for graph in graphs]
        schedule += [(graph, 'f16', replicate) for replicate in range(config['replicates']) for graph in graphs]
        for graph_name, fanout_name, replicate in schedule:
            neighbors = graphs[graph_name]
            plan_seed = 0 if replicate is None else sampling_seed(training_seed, step, replicate)
            report['active_cell'] = {'batch_step': step, 'graph': graph_name, 'fanout': fanout_name,
                                     'replicate': replicate, 'sampling_seed': plan_seed}
            progress('plan_setup', 'scientific')
            deadline()
            plans = two_layer_plans(neighbors, None if replicate is None else config['fanout'], plan_seed)
            cell_identity = {'graph': graph_name, 'fanout': fanout_name, 'replicate': replicate,
                             'sampling_seed': plan_seed, 'plan_sha256': digest(plans)}
            report['active_cell'] = {'batch_step': step, **cell_identity}
            atomic_json(output / 'summary.json', report)
            equivalent = step == config['equivalence']['batch_step'] and (
                replicate is None or replicate == config['equivalence']['sample_replicate'])
            result = observe_condition(model, features, neighbors, plans, queries, labels,
                                       max_padded_messages=config['max_padded_messages'], equivalence=equivalent,
                                       atol=config['equivalence']['atol'], rtol=config['equivalence']['rtol'],
                                       progress=progress)
            if weight_hash(model) != whole_weights or immutable_input_hash(data) != whole_inputs:
                raise ValueError('whole-job weights or prepared inputs changed')
            report['scientific_forward_backward_count'] += 1
            report['equivalence_forward_backward_count'] += int(equivalent)
            layers = result.pop('layers')
            degrees = degree_readings(data['neighbors'], neighbors, plans, positive.cpu().numpy())
            raw = {'plans': plans, 'readings': result,
                   'layers': [layer['raw'] for layer in layers],
                   'degrees': degrees['raw'], 'identity': cell_identity}
            progress('archive', 'scientific')
            descriptor = write_archive(output / 'private',
                                        f'{step}-{graph_name}-{fanout_name}-{replicate if replicate is not None else "all"}', raw)
            # Raw node/query arrays remain private, independently hashed.
            public = {'identity': cell_identity, 'archive': descriptor,
                      'mean_bce': result['mean_bce'], 'equivalence': result['equivalence'],
                      'weight_hash': result['weight_hash'], 'input_hash': result['input_hash'],
                      'parameter_gradients': result['parameter_gradients'],
                      'layers': [layer['summary'] for layer in layers],
                      'degrees': degrees['summary']}
            batch['cells'].append(public)
            cells[(graph_name, fanout_name, replicate)] = {name: result[name] for name in ('logits', 'bce', 'margins')}
            cells[(graph_name, fanout_name, replicate)]['metrics'] = {
                'parameter_gradient_norm/' + name: value['norm']
                for name, value in result['parameter_gradients'].items()}
            for index, layer in enumerate(layers):
                all_nodes = layer['summary']['all_unique_nodes']
                cells[(graph_name, fanout_name, replicate)]['metrics'].update({
                    f'layer{index + 1}/all_unique_gradient_norm_mean': all_nodes['gradient_norm']['mean'],
                    f'layer{index + 1}/all_unique_near_bound_fraction': all_nodes['near_bound_fraction']})
            atomic_json(output / 'summary.json', report)
        comparisons = paired_comparisons(cells, positive.cpu().numpy(), data['neighbors'])
        batch['comparisons'] = comparisons['summary']
        batch['comparison_archive'] = write_archive(output / 'private', f'comparisons-{step}', comparisons['raw'])
        atomic_json(output / 'summary.json', report)
    if report['scientific_forward_backward_count'] != 36 or report['equivalence_forward_backward_count'] != 4:
        raise ValueError('registered realization matrix incomplete')
    progress('whole_job_hash_check', 'scientific')
    deadline()
    if weight_hash(model) != whole_weights or immutable_input_hash(data) != whole_inputs:
        raise ValueError('whole-job final weights or inputs changed')
    report['whole_job_after'] = {'weights': whole_weights, 'inputs': whole_inputs}
    return report


def verify_checkpoint_metadata(checkpoint, training, anchor, config):
    if (digest(asdict(training)) != digest(anchor['training_config']) or checkpoint['completed_steps'] != anchor['step']
            or checkpoint['selection_status'] != 'selected by complete filtered all-candidate validation query-micro MRR'
            or checkpoint['manifest_hash'] != config['prepared']['manifest_hash']
            or checkpoint['valid_hash'] != config['prepared']['valid_queries_hash']):
        raise ValueError('checkpoint selection, configuration or data identity mismatched')
    if any(checkpoint['source_sha256_normalized_lf'].get('acl_hct/' + name) != expected
           for name, expected in config['training_source_lf_sha256'].items()):
        raise ValueError('checkpoint training source hashes differ from frozen registration')


def run_registered(args, output):
    from .backbone_frozen_inputs import load_train_prepared, load_batch_history, replay_batches, immutable_input_hash
    from .backbone_frozen_readings import degree_readings, paired_comparisons
    check_deadline()
    config = load_registration(args.config)
    sources = execution_source_hashes(config)
    identity = source_identity(args.source_commit)
    approval = verify_approval(args.approval_record, args.quality_record, config, identity, sources)
    # Environment checks follow approval and precede data/checkpoint loading.
    if str(torch.__version__) != config['torch_version']:
        raise ValueError('original PyTorch 2.5.1+cu124 runtime required')
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('fresh single-GPU Slurm allocation required')
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) not in ('NVIDIA L40', 'L40'):
        raise ValueError('exactly one allocated L40 required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(config['threads'])
    anchor = config['checkpoints'][str(args.seed)]
    checkpoint, training = load_verified_checkpoint(args.checkpoint, anchor['sha256'],
                                                     config['training_commit'], args.training_release)
    verify_checkpoint_metadata(checkpoint, training, anchor, config)
    data = load_train_prepared(args.prepared_root, config['prepared'])
    history = load_batch_history(args.batch_history, config['history_raw_sha256'][str(args.seed)], args.seed)
    batches = replay_batches(len(data['query_groups']), args.seed, config['batch_steps'],
                              config['batch_positives'], history)
    torch.manual_seed(args.seed)
    model = LorentzMeanNetwork(128, 128, config['c'], config['radius'], 128)
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in checkpoint['model'].values()):
        raise ValueError('finite original FP32 weights required')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to('cuda').eval()
    features = data['features'].to('cuda')
    whole_weights = weight_hash(model)
    whole_inputs = immutable_input_hash(data)
    report = {'status': 'incomplete', 'protocol': config['protocol'], 'seed': args.seed,
              'config_sha256': CONFIG_SHA256, 'source': identity, 'source_lf_sha256': sources,
              'approval': approval, 'checkpoint_sha256': anchor['sha256'],
              'prepared': config['prepared'], 'batch_history_sha256': config['history_raw_sha256'][str(args.seed)],
              'software': {'torch': str(torch.__version__), 'python': platform.python_version(),
                           'device': torch.cuda.get_device_name(0), 'dtype': 'FP32', 'TF32': False},
              'interpretation': 'exploratory fixed-weight train-query path sensitivity; no validation performance or retraining claim',
              'scientific_forward_backward_count': 0, 'equivalence_forward_backward_count': 0,
              'batches': [], 'whole_job_before': {'weights': whole_weights, 'inputs': whole_inputs},
              'optimizer_created': False, 'validation_ranking_performed': False, 'test_truth_opened': False}
    atomic_json(output / 'summary.json', report)
    run_matrix(model, features, data, batches, config, args.seed, output, report)
    atomic_json(output / 'progress.json', {'active_phase': {'phase': 'final_provenance_verification'},
                'active_cell': None, 'pass_execution': report['pass_execution']})
    check_deadline()
    if (weight_hash(model) != whole_weights or immutable_input_hash(data) != whole_inputs
            or file_sha256(args.checkpoint) != anchor['sha256'] or execution_source_hashes(config) != sources):
        raise ValueError('whole-job final provenance changed')
    if immutable_input_hash(load_train_prepared(args.prepared_root, config['prepared'])) != whole_inputs:
        raise ValueError('prepared input files changed during diagnostic')
    report['whole_job_after'] = {'weights': weight_hash(model), 'inputs': immutable_input_hash(data)}
    report['status'] = 'complete'
    report.pop('active_cell', None)
    report.pop('active_phase', None)
    atomic_json(output / 'summary.json', report)
    atomic_json(output / 'progress.json', {'status': 'complete', 'active_cell': None,
                'active_phase': None, 'pass_execution': report['pass_execution']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'prepared-root', 'checkpoint', 'training-release', 'batch-history',
                 'approval-record', 'quality-record', 'source-commit', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--seed', type=int, choices=(11, 23), required=True)
    args = parser.parse_args()
    output = Path(args.output)
    try:
        check_deadline()
        run_registered(args, output)
        return 0
    except Exception as error:
        traceback.print_exc()  # private worker.log; raw assertion details are not public
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / 'failure.json', {'status': 'failed', 'error_type': type(error).__name__,
                    'reason': 'diagnostic worker failed; see private worker.log and incremental summary',
                    'private_failure_log': 'worker.log', 'partial_results_are_complete': False})
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

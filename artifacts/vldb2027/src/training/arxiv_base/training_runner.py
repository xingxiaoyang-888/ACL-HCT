"""Portable input/controller layer around the unchanged original training engine.

No cluster submissions, retries, resumption or corrections are exposed here.
Plain matched-training settings, initialization, minibatch/sample/dropout RNG,
native losses, optimizer and fit checkpoint selection remain in original files.
"""
from public_io import PACKAGE, file_ref, read, sha


def run_plain_pretrain(model_key, seed, files, native, output):
    import sys
    import torch
    from model_builders import build_model, native_scores, task_loss, make_optimizer
    from sampling_bridge import UniformPlanningBridge
    from training_data import load_training_data
    from training_engine import TrainingSpec, run_training
    if model_key not in ('hgcn', 'hybonet', 'gat'):
        raise ValueError('Only the three original new arxiv baseline models')
    config = read(PACKAGE / 'config/arxiv-models.json')
    if native:
        config['runtime_native_roots'][model_key] = str(native)
    data = load_training_data({'schema': 'dasfaa_training_inputs_v1', 'dataset': 'ogbn-arxiv',
        'shape': {'nodes': 169343, 'features': 128, 'classes': 40},
        'split_counts': {'train': 90941, 'fit': 9932, 'gate': 4967, 'confirmation': 14900}, 'files': files})
    sys.path.insert(0, str(PACKAGE / 'src/sampler'))
    from blocks import uniform_edges, native_planning_threads
    bridge = UniformPlanningBridge(uniform_edges, native_planning_threads)
    shared = config['shared_training']
    stage = config['models'][model_key]['stages']['pretrain']
    spec = TrainingSpec(model=model_key, arm='plain', seed=seed, stage='pretrain', epochs=stage['epochs'],
        fanouts=tuple(shared['fanouts_forward_encoding_order']), batch_size=shared['batch_size'],
        selection_every=shared['selection_every_epochs'], selection_repeats=shared['selection_paired_repeats'],
        gradient_clip=None, warmup_epochs=stage['warmup_epochs'])
    if spec.epochs != 500:
        raise ValueError('Original 500-epoch budget required')
    model = build_model(model_key, 'plain', seed, config, 'cuda:0')
    optimizer = make_optimizer(model_key, model, config, 'pretrain')
    return run_training(spec=spec, data=data, model=model, optimizer=optimizer, bridge=bridge,
        score_fn=native_scores, loss_fn=lambda scores, labels: task_loss(model_key, scores, labels, config),
        output_directory=output, device='cuda:0', bindings={
            'portable_reproduction': True, 'config_sha256': sha(PACKAGE / 'config/arxiv-models.json'),
            'source_manifest_sha256': sha(PACKAGE / 'config/source-manifest.json'),
            'training_input_sha256': {k: v['sha256'] for k, v in files.items()}})


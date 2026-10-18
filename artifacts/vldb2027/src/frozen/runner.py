"""Portable replacement of the campaign-specific metadata-v4 controller.

The other sixteen frozen scientific modules are byte-identical. This entry
retains exact input/CP checks and original CUDA/RNG/scoring calls, but does not
recreate the private scheduler, one-campaign authorization or failure ledger.
Each reader's invocation is an independent reproduction, not a paper result.
"""
import gc
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

from public_io import PACKAGE, binding, configure_cuda, file_ref, local_boundary
from public_io import native_root, read, sha, verify_sources, write_new


def run_portable(args):
    verify_sources()
    manifest_path = Path(args.inputs).resolve()
    manifest = read(manifest_path)
    required = {'schema', 'dataset', 'model', 'seed', 'public', 'checkpoint', 'native_root', 'targets', 'labels', 'budget_lock'}
    if set(manifest) != required or manifest['schema'] != 'public_frozen_inputs_v1':
        raise ValueError('Use the exact public_frozen_inputs_v1 schema')
    dataset, model_key, seed = (manifest[k] for k in ('dataset', 'model', 'seed'))
    if (dataset not in ('ogbn-arxiv', 'PubMed') or model_key not in ('hgcn', 'hybonet', 'sage', 'gat')
            or type(seed) is not int or seed not in (11, 23, 37)
            or (dataset == 'PubMed' and model_key != 'sage')):
        raise ValueError('Not a registered model/dataset/seed')
    if set(manifest['public']) != {'graph', 'ids', 'preparation'} or set(manifest['checkpoint']) != {'weights', 'metadata'}:
        raise ValueError('Exact public and checkpoint roles required')
    base = manifest_path.parent
    public = {k: binding(v, base) for k, v in manifest['public'].items()}
    cp = {'weights': binding(manifest['checkpoint']['weights'], base), 'metadata': manifest['checkpoint']['metadata']}
    if args.checkpoint_mode == 'paper':
        entries = read(PACKAGE / 'config/checkpoints.json')['entries']
        expected = next(e for e in entries if (e['dataset'], e['model'], e['seed']) == (dataset, model_key, seed))
        if cp['weights']['sha256'] != expected['sha256']:
            raise ValueError('Paper mode requires the exact recorded checkpoint bytes')
    elif not isinstance(cp['metadata'], dict):
        raise ValueError('Regenerated mode requires exact metadata from checkpoint-metadata.json')
    native = native_root(model_key, manifest['native_root'], base)
    plan_ref = file_ref(PACKAGE / 'config/plan.json')
    plan = read(plan_ref['path'])
    config = read(PACKAGE / ('config/pubmed-sage.json' if dataset == 'PubMed' else 'config/arxiv-models.json'))
    if native:
        config.setdefault('runtime_native_roots', {})[model_key] = str(native)
    output = Path(args.output).resolve()
    if output.exists() or output.is_relative_to(PACKAGE / 'src'):
        raise ValueError('Use a fresh output outside the source tree')
    source = file_ref(PACKAGE / 'config/source-manifest.json')
    job = {'round': plan['round'], 'dataset': dataset, 'model': model_key, 'seed': seed,
           'checkpoint': cp, 'public': public, 'plan': plan_ref, 'source': {'manifest': source}}
    if args.stage == 'calibrate':
        if seed != 11 or any(manifest[k] is not None for k in ('targets', 'labels', 'budget_lock')):
            raise ValueError('Calibration uses seed11, with no target-label or budget-lock bindings')
    else:
        for k in ('targets', 'labels', 'budget_lock'):
            job[k] = binding(manifest[k], base)
        lock = read(job['budget_lock']['path'])
        if (lock['schema'] != 'vldb_sampling_budget_lock_v1' or lock['dataset'] != dataset or lock['model'] != model_key
                or lock['calibration_seed'] != 11 or lock['labels_read'] is not False or lock['plan_sha256'] != plan_ref['sha256']
                or [t['tier'] for t in lock['tiers']] != ['low', 'middle', 'high']
                or [t['uniform_fanout'] for t in lock['tiers']] != plan['datasets'][dataset]['uniform_fanouts']
                or any(t['labor_fanout'] not in plan['datasets'][dataset]['labor_grid'] for t in lock['tiers'])):
            raise ValueError('Frozen label-free budget lock differs')
    # Inputs were hash-bound before model construction. Output creation is exclusive.
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'portable-input-commitment.json', {
        'input_manifest_sha256': sha(manifest_path), 'source_manifest_sha256': source['sha256'],
        'plan_sha256': plan_ref['sha256'], 'checkpoint_sha256': cp['weights']['sha256'],
        'checkpoint_mode': args.checkpoint_mode, 'stage': args.stage,
        'paper_result': False, 'scheduler_or_exclusivity_certified': False})
    configure_cuda()
    import torch
    from study_io import bound, load_public, calibration_batches, test_targets, test_labels
    from model_builders import build_model, build_pubmed_sage
    from sampling_bridge import PlanningBridge
    sys.path.insert(0, str(PACKAGE / 'src/sampler'))
    blocks = importlib.import_module('blocks')
    planner = PlanningBridge(blocks.uniform_edges, blocks.native_planning_threads)
    started = time.perf_counter()
    graph, meta, _ = load_public(job, plan)
    with blocks.native_planning_threads():
        graph.dgl_graph()
    graph_setup = time.perf_counter() - started
    started = time.perf_counter()
    model = build_pubmed_sage(seed, 'cuda:0') if dataset == 'PubMed' else build_model(model_key, 'plain', seed, config, 'cuda:0')
    with bound(cp['weights']) as stream:
        payload = torch.load(stream, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or 'model_state' not in payload:
        raise ValueError('Checkpoint payload must contain model_state')
    actual = {k: v for k, v in payload.items() if k != 'model_state'}
    # JSON canonicalization retains metadata-v4 tuple/list equivalence.
    canonical = json.dumps(actual, sort_keys=True, allow_nan=False)
    if args.checkpoint_mode == 'paper' and hashlib.sha256(canonical.encode()).hexdigest() != expected['metadata_sha256']:
        raise ValueError('Paper checkpoint metadata identity differs')
    if args.checkpoint_mode == 'regenerated' and canonical != json.dumps(cp['metadata'], sort_keys=True, allow_nan=False):
        raise ValueError('Checkpoint metadata differs')
    if payload['seed'] != seed or type(payload['epoch']) is not int or not 1 <= payload['epoch'] <= 500 or payload['epoch'] % 5:
        raise ValueError('Checkpoint seed/selection epoch differs')
    if payload['schema'] == 'dasfaa_shared_correction_weights_v1':
        if payload['model'] != model_key or payload['arm'] != 'plain' or payload['config']['stage'] != 'pretrain' or payload['config']['epochs'] != 500:
            raise ValueError('Only original plain500 checkpoints')
    elif model_key != 'sage' or payload['schema'] != ('vldb_pubmed_sage_weights_v1' if dataset == 'PubMed' else 'icml_ca_sage_weights_v1'):
        raise ValueError('Wrong checkpoint origin')
    if any('.correction.' in key for key in payload['model_state']):
        raise ValueError('Correction weights are outside the primary study')
    model.load_state_dict(payload['model_state'], strict=True)
    del payload
    model.eval()
    local_boundary()
    model_setup = time.perf_counter() - started
    if args.stage == 'calibrate':
        from calibration import calibrate
        result = calibrate(job, plan, model, graph, planner, calibration_batches(meta, dataset, plan), output, local_boundary)
    else:
        ids, batches = test_targets(job, meta, plan['datasets'][dataset]['shape'])
        labels = test_labels(job['labels'], ids, plan['datasets'][dataset]['shape']['classes'])
        from final_evaluation import evaluate
        result = evaluate(job, plan, model, graph, planner, ids, batches, labels, lock, output, local_boundary)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    write_new(output / 'portable-completion.json', {'status': 'completed', 'dataset': dataset, 'model': model_key, 'seed': seed,
        'public_graph_setup_seconds': graph_setup, 'CP_model_setup_seconds': model_setup,
        'portable_reproduction_not_original_campaign': True, 'continuous_contention_certification': False})
    return result

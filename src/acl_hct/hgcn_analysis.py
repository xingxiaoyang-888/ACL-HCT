"""CPU archive replay and complete conditional Holm12; never select models."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from .diagnostic_archive import read_archive, write_archive
from .hgcn_evidence import (array_hash, atomic_json, checkpoint_binding, compare_full, deadline, file_hash,
                           hierarchy_evaluator, load_checkpoint, load_panel_design, validate_ranking)
from .hgcn_geometry import mean_error_statistics
from .hgcn_quality import load_train, load_valid
from .hgcn_registration import canonical
from .hgcn_sampling import make_plan
from .hgcn_statistics import primary_family, student_summary


def equal_tree(actual, expected, path='payload'):
    if isinstance(expected, np.ndarray):
        if (not isinstance(actual, np.ndarray) or actual.dtype != expected.dtype
                or actual.shape != expected.shape or not np.array_equal(actual, expected)):
            raise ValueError('archive replay array mismatch: ' + path)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError('archive replay keys mismatch: ' + path)
        for key in expected:
            equal_tree(actual[key], expected[key], path + '/' + str(key))
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError('archive replay sequence mismatch: ' + path)
        for i, value in enumerate(expected):
            equal_tree(actual[i], value, path + '/' + str(i))
    elif actual != expected:
        raise ValueError('archive replay scalar mismatch: ' + path)


def numpy_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {k: numpy_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [numpy_tree(v) for v in value]
    return value


def checked_run(item, phase, config, release):
    path = Path(item['run_file'])
    if file_hash(path) != item['sha256']:
        raise ValueError('artifact index report hash mismatch')
    report = json.loads(path.read_bytes())
    if (report['status'] != 'complete' or report['phase'] != phase
            or report['config_sha256'] != canonical(config) or report['source_commit'] != release['source_commit']):
        raise ValueError('complete same-source/config archived phase required')
    return path.parent, report['result']


def replay_training(root, trained, data, config, source_commit):
    t = config['training']; seed = trained['seed']
    if trained['completed_steps'] != t['steps']:
        raise ValueError('incomplete trained baseline')
    valid, truth = data['valid'], data['truth']; n = len(data['nodes'])
    evaluations = trained['evaluations']; expected_steps = [0, *t['complete_valid_selection_steps']]
    if [e['step'] for e in evaluations] != expected_steps:
        raise ValueError('all initial/scheduled complete valid archives required')
    best = None; maximum = -1.
    for evaluation in evaluations:
        deadline(); saved = read_archive(root / 'arrays', evaluation['ranking_archive'])
        if saved['step'] != evaluation['step'] or set(map(tuple, saved['queries'])) != set(valid):
            raise ValueError('saved validation query/step identity mismatch')
        mrr = validate_ranking(saved['ranking'], valid, truth, n)
        if mrr != evaluation['micro_mrr'] or evaluation['selection_eligible'] != (evaluation['step'] != 0):
            raise ValueError('saved selection score/eligibility mismatch')
        if evaluation['step'] and mrr > maximum:
            maximum = mrr; best = evaluation['step']
    if trained['best']['step'] != best or trained['best']['micro_mrr'] != maximum:
        raise ValueError('first strict full-validation maximum not reproduced')
    cp = load_checkpoint(root, trained['best']['checkpoint'], checkpoint_binding(config, source_commit, seed, best))
    reference = read_archive(root / 'arrays', trained['best']['reference'])
    if reference['model_state_sha256'] != cp['model_state_sha256'] or reference['binding'] != cp['binding']:
        raise ValueError('best reference not bound to serialized best state')
    validate_ranking(reference['ranking'], valid, truth, n)
    if reference['native_ball_points'].dtype != np.float32 or reference['native_ball_points'].shape != (n, config['model']['hidden']):
        raise ValueError('best complete native FP32 point shape required')
    equal_tree(reference['full_plan'], numpy_tree(make_plan(data['neighbors'], None, torch.Generator()).archive()))
    initial = read_archive(root / 'arrays', trained['initial_reference'])
    if initial['model_state_sha256'] != trained['initial_state_sha256']:
        raise ValueError('initial reference not bound to fresh initialization')
    validate_ranking(initial['ranking'], valid, truth, n)
    history = read_archive(root / 'arrays', trained['batch_history'])
    batches = history['batch_indices']
    if batches.dtype != np.int64 or batches.shape != (t['steps'], t['batch_positives']):
        raise ValueError('complete FP32 training batch history shape/dtype mismatch')
    rng = torch.Generator().manual_seed(seed + 1)
    equal_tree(history['initial_batch_rng_state'], rng.get_state().numpy())
    if len(history['steps']) != t['steps']:
        raise ValueError('all finite training steps required')
    full_messages = sum(map(len, data['neighbors']))
    for step, chosen in enumerate(batches, start=1):
        if step % 256 == 0:
            deadline()
        expected = torch.randperm(len(data['query_groups']), generator=rng)[:t['batch_positives']].numpy()
        equal_tree(chosen, expected)
        targets = data['query_groups'][torch.from_numpy(chosen), 0].tolist()
        messages = full_messages - 2 * len(set(map(tuple, targets)))
        row = history['steps'][step - 1]
        if row['step'] != step or not np.isfinite(row['loss']) or row['postmask_nonself_messages'] != messages:
            raise ValueError('batch target masking or finite step history mismatch')
    equal_tree(history['final_batch_rng_state'], rng.get_state().numpy())
    load_checkpoint(root, trained['last'], checkpoint_binding(config, source_commit, seed, t['steps']))
    return reference


def analyze(artifact_index, prepared_root, config, release, output):
    if file_hash(artifact_index) != release['inputs']['artifact_index_sha256']:
        raise ValueError('reviewed complete artifact index hash mismatch')
    index = json.loads(Path(artifact_index).read_bytes())
    if index['protocol'] != config['protocol'] or index['config_sha256'] != canonical(config):
        raise ValueError('artifact index protocol/config mismatch')
    if len(index['training']) != 2 or len(index['evaluation']) != 8:
        raise ValueError('both models and all eight evaluation shards required')
    data = load_train(prepared_root, config)
    data['valid'], data['truth'] = load_valid(prepared_root, data, config)
    view, panel = load_panel_design(prepared_root, config)
    trained = {}; references = {}; evaluators = {}; baselines = {}; training_hashes = {}
    for item in index['training']:
        root, run = checked_run(item, 'train', config, release); seed = run['seed']
        if seed not in config['training']['seeds'] or seed in trained:
            raise ValueError('distinct registered trained model identities required')
        ref = replay_training(root, run, data, config, release['source_commit'])
        evaluator, full = hierarchy_evaluator(view, panel, ref['native_ball_points'], config)
        baseline = read_archive(root / 'arrays', run['full_baseline']['hierarchy_archive'])
        equal_tree(baseline['panel_design'], evaluator.archive_design())
        equal_tree(baseline['hierarchy'], full)
        if (baseline['native_full_points_sha256'] != array_hash(ref['native_ball_points'])
                or baseline['model_state_sha256'] != run['best']['checkpoint']['model_state_sha256']
                or run['full_baseline']['direct_order'] != full['direct']['metrics']['score']):
            raise ValueError('pre-sampling full hierarchy baseline identity mismatch')
        trained[seed] = run; references[seed] = ref; evaluators[seed] = evaluator; baselines[seed] = full
        training_hashes[seed] = item['sha256']
    if len({run['initial_state_sha256'] for run in trained.values()}) != 2:
        raise ValueError('fresh seeds did not initialize distinct weights')
    samples = {}; seen_shards = set()
    for item in index['evaluation']:
        root, shard = checked_run(item, 'eval', config, release); seed = shard['seed']; start = shard['repeat_start']
        if (seed, start) in seen_shards or (seed, start) not in {(s['seed'], s['repeats'][0]) for s in config['sampling']['shards']}:
            raise ValueError('all distinct registered shards required')
        seen_shards.add((seed, start)); evaluator = evaluators[seed]
        if shard['best_checkpoint'] != trained[seed]['best']['checkpoint'] or len(shard['samples']) != 12:
            raise ValueError('same best weights and complete shard required')
        control = read_archive(root / 'arrays', shard['full_control'])
        validate_ranking(control['ranking'], data['valid'], data['truth'], len(data['nodes']))
        compare_full(control['native_ball_points'], control['ranking'], references[seed])
        equal_tree(control['panel_design'], evaluator.archive_design())
        equal_tree(control['hierarchy'], baselines[seed])
        equal_tree(control['full_plan'], references[seed]['full_plan'])
        for item_sample in shard['samples']:
            deadline(); sample = read_archive(root / 'arrays', item_sample['artifact'])
            repeat, budget = sample['repeat'], sample['fanout']; key = (seed, budget, repeat)
            if (key in samples or repeat not in range(start, start + 4) or budget not in (4, 8, 16)
                    or sample['seed'] != seed or item_sample['repeat'] != repeat or item_sample['fanout'] != budget
                    or sample['checkpoint_sha256'] != trained[seed]['best']['checkpoint']['sha256']
                    or sample['model_state_sha256'] != trained[seed]['best']['checkpoint']['model_state_sha256']
                    or sample['full_native_points_sha256'] != array_hash(references[seed]['native_ball_points'])):
                raise ValueError('sample identity, weights or fixed full points mismatch')
            graph_seed = 202609190000 + seed * 10000 + repeat * 100 + budget
            if sample['graph_seed'] != graph_seed or len(sample['plans']) != 2:
                raise ValueError('registered independent graph generator mismatch')
            rng = torch.Generator().manual_seed(graph_seed)
            equal_tree(sample['rng_before'], rng.get_state().numpy())
            plans = [make_plan(data['neighbors'], budget, rng) for _ in range(2)]
            equal_tree(sample['plans'], numpy_tree([p.archive() for p in plans]))
            equal_tree(sample['rng_after'], rng.get_state().numpy())
            mrr = validate_ranking(sample['ranking'], data['valid'], data['truth'], len(data['nodes']))
            native = sample['native_ball_points']
            if native.dtype != np.float32 or native.shape != references[seed]['native_ball_points'].shape:
                raise ValueError('saved complete native FP32 sampled points required')
            hierarchy = evaluator.evaluate(native); equal_tree(sample['hierarchy'], hierarchy)
            if item_sample['micro_mrr'] != mrr or item_sample['direct_order'] != hierarchy['direct']['metrics']['score']:
                raise ValueError('shard summary differs from full arrays')
            samples[key] = {'micro_mrr': mrr, 'hierarchy': hierarchy}
        atomic_json(output / 'analysis-progress.json', {'status': 'running', 'replayed_shards': len(seen_shards),
                                                       'replayed_sample_arrays': len(samples)})
    expected = {(s, b, r) for s in (11, 23) for b in (4, 8, 16) for r in range(16)}
    if set(samples) != expected:
        raise ValueError('all 96 complete graph samples required, no optional stopping')
    differences = {}; bias = []; descriptive = []
    for seed in (11, 23):
        for budget in (4, 8, 16):
            group = [samples[seed, budget, r] for r in range(16)]; full = baselines[seed]
            direct = [x['hierarchy']['direct']['metrics']['score'] for x in group]
            if full['direct']['metrics']['score'] is None or any(x is None for x in direct):
                raise ValueError('registered direct coverage must remain fixed and nonempty')
            differences[seed, budget, 'direct_order'] = np.array(direct) - full['direct']['metrics']['score']
            differences[seed, budget, 'micro_mrr'] = np.array([x['micro_mrr'] for x in group]) - references[seed]['ranking']['query_micro_mrr']
            errors = np.stack([x['hierarchy']['bias']['nodewise_error_fp64'] for x in group])
            moments = mean_error_statistics(errors); weights = evaluators[seed].weights
            corrected = float(np.sum(weights * moments['bias_squared_unbiased_per_node']) / weights.sum())
            radial = [x['hierarchy']['bias']['weighted_outward_projection'] for x in group]
            artifact = write_archive(output / 'arrays', f'conditional-moments-seed{seed}-f{budget}',
                                     {'seed': seed, 'fanout': budget, 'errors': errors, 'moments': moments,
                                      'fixed_panel_design': evaluators[seed].archive_design(),
                                      'S_minus_F_direct_order': differences[seed, budget, 'direct_order'],
                                      'S_minus_F_micro_mrr': differences[seed, budget, 'micro_mrr']})
            bias.append({'seed': seed, 'fanout': budget, 'signed_unbiased_bias_squared_weighted': corrected,
                         'outward_projection_exploratory': student_summary(radial) if all(x is not None for x in radial) else None,
                         'archive': artifact, 'mean_norm_alone_is_bias_proof': False})
            descriptive.append({'seed': seed, 'fanout': budget,
                                'distant_order': [x['hierarchy']['distant']['metrics']['score'] for x in group],
                                'direct_gaps': [x['hierarchy']['direct']['metrics']['gap'] for x in group],
                                'root_coverage': {k: full['bias'][k] for k in ('root_known_nodes', 'direction_known_nodes', 'weighted_direction_known_mass')}})
    primary = primary_family(differences)
    _, official = checked_run(index['official'], 'official', config, release)
    # Official example provenance is distinct from WordNet's cleaner protocol.
    official_root = Path(index['official']['run_file']).parent
    official_data = read_archive(official_root / 'arrays', official['archive'])
    equal_tree(official_data['validation'], official['best']['validation'])
    equal_tree(official_data['test'], official['test_after_final_best_reload'])
    load_checkpoint(official_root, official['best']['checkpoint'], official['best']['checkpoint']['binding'])
    return {'status': 'complete', 'replayed_sample_arrays': len(samples), 'primary_family': primary,
            'full_baselines': {str(seed): {'direct_order': baselines[seed]['direct']['metrics']['score'],
                               'positive_order_necessary_premise': baselines[seed]['direct']['metrics']['score'] > .5,
                               'premise_limit': '>0.5 is necessary, not sufficient; inspect order, gaps, coverage and learning state, retain both models',
                               'micro_mrr': references[seed]['ranking']['query_micro_mrr'],
                               'hierarchy_coverage': {kind: {k: v for k, v in baselines[seed][kind].items()
                                                     if k not in ('gap', 'score', 'child_values')} for kind in ('direct', 'distant')}}
                               for seed in (11, 23)},
            'bias_exploratory': bias, 'secondary_descriptive': descriptive, 'official_example': official,
            'rank_replay_scope': 'all saved query identities, candidates, exact-tie rank bounds and aggregates; full logits were checked by source fixtures and GPU deterministic F replay, not independently CPU-rescored here',
            'generality': config['statistics']['generality'], 'stopping_boundary': config['stopping']['end'],
            'training_run_sha256': {str(s): h for s, h in training_hashes.items()}}

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


def retrieval_sensitivity_decision(original, adjusted, observed_max):
    same_direction = np.sign(original['mean']) == np.sign(adjusted['mean'])
    same_holm = original['reject_holm'] == adjusted['reject_holm']
    return {'same_direction': bool(same_direction), 'same_Holm_decision': same_holm,
            'original_mean_abs_exceeds_observed_max': abs(original['mean']) > observed_max,
            'adjusted_mean_abs_exceeds_observed_max': abs(adjusted['mean']) > observed_max,
            'robust_retrieval_damage_claim_allowed': bool(same_direction and same_holm and
                original['reject_holm'] and adjusted['reject_holm'] and
                original['mean'] < 0 and adjusted['mean'] < 0 and
                abs(original['mean']) > observed_max and abs(adjusted['mean']) > observed_max)}


def checked_run(item, phase, config, release, historical_official=False, accepted_source=None):
    path = Path(item['run_file'])
    if file_hash(path) != item['sha256']:
        raise ValueError('artifact index report hash mismatch')
    report = json.loads(path.read_bytes())
    from .hgcn_replay import ORIGINAL_SOURCE
    if historical_official and (phase != 'official' or not release.get('replay_policy_sha256')):
        raise ValueError('historical official source requires explicit amended analysis context')
    accepted_source = ORIGINAL_SOURCE if historical_official else accepted_source or release['source_commit']
    if (report['status'] != 'complete' or report['phase'] != phase
            or report['config_sha256'] != canonical(config) or report['source_commit'] != accepted_source):
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


def analyze(artifact_index, prepared_root, config, release, output, policy=None, historical_policy=None):
    if file_hash(artifact_index) != release['inputs']['artifact_index_sha256']:
        raise ValueError('reviewed complete artifact index hash mismatch')
    index = json.loads(Path(artifact_index).read_bytes())
    if index['protocol'] != config['protocol'] or index['config_sha256'] != canonical(config):
        raise ValueError('artifact index protocol/config mismatch')
    if policy is not None and index.get('replay_policy_sha256') != canonical(policy):
        raise ValueError('artifact index does not bind amended replay policy')
    if policy is not None and release.get('replay_policy_sha256') != canonical(policy):
        raise ValueError('amended analysis policy differs from verified release context')
    if len(index['training']) != 2 or len(index['evaluation']) != 8:
        raise ValueError('both models and all eight evaluation shards required')
    from .hgcn_replay import is_v2, load_policy
    amendment = policy['rank_sensitive_amendment'] if policy and is_v2(policy) else None
    if historical_policy is not None and not amendment:
        raise ValueError('historical policy override only for v2 mixed archive validation')
    v1_policy = ((historical_policy or load_policy(Path(__file__).parents[2] / 'configs/mature_hgcn_replay_policy.json', config))
                 if amendment else None)
    if amendment and (index.get('v1_policy_sha256') != amendment['v1_policy_sha256']
                      or canonical(v1_policy) != amendment['v1_policy_sha256']):
        raise ValueError('exact older qualification policy required for mixed shards')
    if amendment and not set(amendment['v1_completed_shards'].values()).issubset(
            {item['sha256'] for item in index['evaluation']}):
        raise ValueError('all three preserved v1 completed shards required in mixed batch')
    data = load_train(prepared_root, config)
    data['valid'], data['truth'] = load_valid(prepared_root, data, config)
    view, panel = load_panel_design(prepared_root, config)
    trained = {}; references = {}; evaluators = {}; baselines = {}; training_hashes = {}; numeric_replays = []
    full_observations = []; shard_drifts = {}
    if amendment:
        basis = v1_policy['basis']
        full_observations.append({'scope': 'frozen_v1_full_only_diagnostic_basis_maximum_of_three_complete_ranks',
                                  'seed': basis['seed'], 'diagnosis_sha256': basis['diagnosis_sha256'],
                                  'archive_manifest_sha256': basis['archive_manifest_sha256'],
                                  'archive_npz_sha256': basis['archive_npz_sha256'],
                                  'observed_abs_micro_mrr_drift': basis['observed_maxima']['micro_mrr_delta'],
                                  'observed_abs_macro_mrr_drift': basis['observed_maxima']['macro_mrr_delta'],
                                  'not_independent_graph_repeats': True})
    for item in index['training']:
        root, run = checked_run(item, 'replay' if policy else 'train', config, release,
                                accepted_source=amendment['v1_source_commit'] if amendment else None); seed = run['seed']
        if seed not in config['training']['seeds'] or seed in trained:
            raise ValueError('distinct registered trained model identities required')
        if policy:
            from .hgcn_replay import ORIGINAL_SOURCE, historical_training, qualify_full, require_qualified
            if amendment and item['sha256'] != amendment['v1_accepted_replays'][str(seed)]:
                raise ValueError('baseline replay is not exact whitelisted v1 artifact')
            origin_root, original = historical_training(run['original_training']['run_file'], config, policy, seed)
            for key in ('best', 'evaluations', 'initial_reference', 'initial_state_sha256', 'batch_history', 'last', 'original_training'):
                equal_tree(run[key], original[key])
            baseline_policy = v1_policy if amendment else policy
            if run['replay_policy_sha256'] != canonical(baseline_policy):
                raise ValueError('recovered baseline policy mismatch')
            ref = replay_training(origin_root, run, data, config, ORIGINAL_SOURCE)
        else:
            ref = replay_training(root, run, data, config, release['source_commit'])
        evaluator, full = hierarchy_evaluator(view, panel, ref['native_ball_points'], config)
        if policy:
            saved_replay = read_archive(root / 'arrays', run['replay_archive'])
            qualified = qualify_full(saved_replay['native_ball_points'], saved_replay['ranking'], ref, evaluator, config,
                                     baseline_policy, data['valid'], data['truth'], saved_replay['model_state_sha256'],
                                     checkpoint_binding(config, ORIGINAL_SOURCE, seed, run['best']['step']), saved_replay['full_plan'])
            equal_tree(saved_replay['qualification'], qualified); equal_tree(run['replay_qualification'], qualified)
            equal_tree(saved_replay['panel_design'], evaluator.archive_design())
            equal_tree(saved_replay['replay_hierarchy'], evaluator.evaluate(saved_replay['native_ball_points']))
            require_qualified(qualified); numeric_replays.append({'scope': 'baseline', 'seed': seed, **qualified})
            if amendment:
                full_observations.append({'scope': 'accepted_full_only', 'seed': seed, 'run_sha256': item['sha256'],
                                          'policy_sha256': canonical(baseline_policy),
                                          'signed_micro_mrr_drift': saved_replay['ranking']['query_micro_mrr'] - ref['ranking']['query_micro_mrr'],
                                          'signed_macro_mrr_drift': saved_replay['ranking']['child_macro_mrr'] - ref['ranking']['child_macro_mrr']})
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
    if amendment:
        failed_item = index['failed_v1_full_control']
        failed_path = Path(failed_item['run_file']); failed_qpath = Path(failed_item['qualification_file'])
        if (file_hash(failed_path) != amendment['failed_v1_seed11_r4_run_sha256']
                or file_hash(failed_qpath) != amendment['failed_v1_seed11_r4_qualification_sha256']
                or failed_item['sha256'] != amendment['failed_v1_seed11_r4_run_sha256']
                or failed_item['qualification_sha256'] != amendment['failed_v1_seed11_r4_qualification_sha256']):
            raise ValueError('preserved failed v1 full-control evidence identity mismatch')
        failure = json.loads(failed_path.read_bytes()); failed_q = json.loads(failed_qpath.read_bytes())
        if (failure['status'] != 'failed' or failure['phase'] != 'eval'
                or failure['source_commit'] != amendment['v1_source_commit']
                or failure['config_sha256'] != canonical(config)
                or failure['error'] != 'ValueError: numerical full replay qualification failed: micro_mrr_abs_delta, macro_mrr_abs_delta'
                or failed_path.parent != failed_qpath.parent):
            raise ValueError('old full-control failure must remain failed and local to its archive')
        failed_control = read_archive(failed_path.parent / 'arrays', failed_q['archive'])
        seed = 11; evaluator = evaluators[seed]
        diagnosed = qualify_full(failed_control['native_ball_points'], failed_control['ranking'], references[seed],
                                 evaluator, config, v1_policy, data['valid'], data['truth'],
                                 failed_control['model_state_sha256'],
                                 checkpoint_binding(config, ORIGINAL_SOURCE, seed, trained[seed]['best']['step']),
                                 failed_control['full_plan'])
        equal_tree(failed_control['qualification'], diagnosed); equal_tree(failed_q['qualification'], diagnosed)
        if diagnosed['accepted'] or [v['metric'] for v in diagnosed['violations']] != [
                'micro_mrr_abs_delta', 'macro_mrr_abs_delta']:
            raise ValueError('preserved v1 MRR-only failure changed')
        full_observations.append({'scope': 'preserved_failed_v1_control', 'seed': seed,
                                  'run_sha256': failed_item['sha256'], 'policy_sha256': canonical(v1_policy),
                                  'signed_micro_mrr_drift': failed_control['ranking']['query_micro_mrr'] - references[seed]['ranking']['query_micro_mrr'],
                                  'signed_macro_mrr_drift': failed_control['ranking']['child_macro_mrr'] - references[seed]['ranking']['child_macro_mrr']})
    samples = {}; seen_shards = set()
    for item in index['evaluation']:
        older = bool(amendment and item['sha256'] in amendment['v1_completed_shards'].values())
        root, shard = checked_run(item, 'eval', config, release,
                                  accepted_source=amendment['v1_source_commit'] if older else None)
        seed = shard['seed']; start = shard['repeat_start']
        if amendment and older and item['sha256'] != amendment['v1_completed_shards'].get(f'{seed}:{start}'):
            raise ValueError('old completed shard is not exact source/seed/start whitelist')
        if (seed, start) in seen_shards or (seed, start) not in {(s['seed'], s['repeats'][0]) for s in config['sampling']['shards']}:
            raise ValueError('all distinct registered shards required')
        seen_shards.add((seed, start)); evaluator = evaluators[seed]
        if shard['best_checkpoint'] != trained[seed]['best']['checkpoint'] or len(shard['samples']) != 12:
            raise ValueError('same best weights and complete shard required')
        control = read_archive(root / 'arrays', shard['full_control'])
        validate_ranking(control['ranking'], data['valid'], data['truth'], len(data['nodes']))
        if policy:
            shard_policy = v1_policy if older else policy
            if shard['replay_policy_sha256'] != canonical(shard_policy) or shard['original_training'] != trained[seed]['original_training']:
                raise ValueError('shard amended policy/original lineage mismatch')
            qualified = qualify_full(control['native_ball_points'], control['ranking'], references[seed], evaluator, config,
                                     shard_policy, data['valid'], data['truth'], control['model_state_sha256'],
                                     checkpoint_binding(config, ORIGINAL_SOURCE, seed, trained[seed]['best']['step']), control['full_plan'])
            equal_tree(control['qualification'], qualified); equal_tree(shard['full_replay'], qualified); require_qualified(qualified)
            if control['fixed_full_native_points_sha256'] != array_hash(references[seed]['native_ball_points']):
                raise ValueError('shard fixed F replacement detected')
            numeric_replays.append({'scope': 'shard', 'seed': seed, 'repeat_start': start, **qualified})
            if amendment:
                signed_micro = control['ranking']['query_micro_mrr'] - references[seed]['ranking']['query_micro_mrr']
                signed_macro = control['ranking']['child_macro_mrr'] - references[seed]['ranking']['child_macro_mrr']
                if not older and (signed_micro != qualified['v1_diagnostic']['signed_micro_mrr_drift']
                                  or signed_macro != qualified['v1_diagnostic']['signed_macro_mrr_drift']):
                    raise ValueError('signed v2 full-control drift identity mismatch')
                full_observations.append({'scope': 'completed_shard', 'seed': seed, 'repeat_start': start,
                                          'run_sha256': item['sha256'], 'policy_sha256': canonical(shard_policy),
                                          'signed_micro_mrr_drift': signed_micro,
                                          'signed_macro_mrr_drift': signed_macro})
                for repeat in range(start, start + 4):
                    shard_drifts[seed, repeat] = signed_micro
        else:
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
    differences = {}; adjusted_differences = {}; bias = []; descriptive = []
    for seed in (11, 23):
        for budget in (4, 8, 16):
            group = [samples[seed, budget, r] for r in range(16)]; full = baselines[seed]
            direct = [x['hierarchy']['direct']['metrics']['score'] for x in group]
            if full['direct']['metrics']['score'] is None or any(x is None for x in direct):
                raise ValueError('registered direct coverage must remain fixed and nonempty')
            differences[seed, budget, 'direct_order'] = np.array(direct) - full['direct']['metrics']['score']
            differences[seed, budget, 'micro_mrr'] = np.array([x['micro_mrr'] for x in group]) - references[seed]['ranking']['query_micro_mrr']
            if amendment:
                adjusted_differences[seed, budget, 'direct_order'] = differences[seed, budget, 'direct_order'].copy()
                adjusted_differences[seed, budget, 'micro_mrr'] = differences[seed, budget, 'micro_mrr'] - np.array(
                    [shard_drifts[seed, r] for r in range(16)], dtype=np.float64)
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
    sensitivity = None
    if amendment:
        adjusted = primary_family(adjusted_differences)
        by_comparison = {row['comparison']: row for row in adjusted}
        maxima = {str(seed): {metric: max(abs(o['signed_' + metric + '_mrr_drift'])
                                                if 'signed_' + metric + '_mrr_drift' in o
                                                else o['observed_abs_' + metric + '_mrr_drift']
                                                for o in full_observations if o['seed'] == seed)
                              for metric in ('micro', 'macro')} for seed in config['training']['seeds']}
        retrieval = []
        for original in primary:
            if original['metric'] != 'micro_mrr':
                continue
            corrected = by_comparison[original['comparison']]
            largest = maxima[str(original['seed'])]['micro']
            retrieval.append({'comparison': original['comparison'], 'original_mean': original['mean'],
                              'adjusted_mean': corrected['mean'], 'original_marginal_student95': original['marginal_student95'],
                              'adjusted_marginal_student95': corrected['marginal_student95'],
                              'original_reject_holm': original['reject_holm'],
                              'adjusted_reject_holm': corrected['reject_holm'],
                              'max_observed_full_micro_drift_for_seed': largest,
                              **retrieval_sensitivity_decision(original, corrected, largest)})
        sensitivity = {'scope': 'descriptive_full_control_basis_sensitivity_not_extra_graph_repeats',
                       'adjusted_primary_family_Holm12': adjusted, 'retrieval_interpretation': retrieval,
                       'full_control_observations_including_failed': full_observations,
                       'max_observed_abs_full_MRR_drift_by_seed': maxima,
                       'full_drift_is_not_guaranteed_error_bound': True,
                       'adjustment': 'each micro S-original-F minus its shard signed full-control-original-F; direct unchanged',
                       'original_primary_family_unchanged': True}
    if amendment and index['official']['sha256'] != amendment['v1_official_run_sha256']:
        raise ValueError('historical official example is not exact v1 whitelist')
    # The sidecar also accompanies current-source artificial quality fixtures.
    # Only actual amended analysis uses the preserved historical example.
    _, official = checked_run(index['official'], 'official', config, release, historical_official=policy is not None)
    # Official example provenance is distinct from WordNet's cleaner protocol.
    official_root = Path(index['official']['run_file']).parent
    official_data = read_archive(official_root / 'arrays', official['archive'])
    equal_tree(official_data['validation'], official['best']['validation'])
    equal_tree(official_data['test'], official['test_after_final_best_reload'])
    load_checkpoint(official_root, official['best']['checkpoint'], official['best']['checkpoint']['binding'])
    return {'status': 'complete', 'replayed_sample_arrays': len(samples), 'primary_family': primary,
            **({'replay_policy_sha256': canonical(policy), 'full_numeric_replays': numeric_replays,
                **({'ranking_sensitive_v2': sensitivity, 'v1_policy_sha256': amendment['v1_policy_sha256']}
                   if amendment else {}),
                'full_only_diagnostic_observations': policy['basis'],
                'numeric_interpretation': 'report observed full repeat errors alongside effect sizes; engineering limits are not guaranteed error bounds, sampled-graph bounds or statistical null bands; tiny significance alone is not substantive damage',
                'original_training_lineage': {str(s): trained[s]['original_training'] for s in trained}} if policy else {}),
            'full_baselines': {str(seed): {'direct_order': baselines[seed]['direct']['metrics']['score'],
                               'positive_order_necessary_premise': baselines[seed]['direct']['metrics']['score'] > .5,
                               'premise_limit': '>0.5 is necessary, not sufficient; inspect order, gaps, coverage and learning state, retain both models',
                               'micro_mrr': references[seed]['ranking']['query_micro_mrr'],
                               'hierarchy_coverage': {kind: {k: v for k, v in baselines[seed][kind].items()
                                                     if k not in ('gap', 'score', 'child_values')} for kind in ('direct', 'distant')}}
                               for seed in (11, 23)},
            'bias_exploratory': bias, 'secondary_descriptive': descriptive, 'official_example': official,
            'rank_replay_scope': 'all saved query identities, candidates, exact-tie rank bounds and aggregates; full logits checked by source fixtures and explicit full replay qualification, not independently CPU-rescored here',
            'generality': config['statistics']['generality'], 'stopping_boundary': config['stopping']['end'],
            'training_run_sha256': {str(s): h for s, h in training_hashes.items()}}

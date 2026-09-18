"""Revised qualification rejects independent failure modes without changing F."""
from copy import deepcopy
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from acl_hct.hgcn_panels import HierarchyPanel
from acl_hct.hgcn_replay import (ORIGINAL_FAILURE, ORIGINAL_SOURCE, POLICY_SHA256, historical_training,
                                qualify_full, require_qualified, validate_policy)


def policy():
    return json.loads((Path(__file__).parents[1] / 'configs/mature_hgcn_replay_policy.json').read_bytes())


def example(points=None, both_known=False):
    full = np.array([[.05, 0.], [.2, 0.], [.35, 0.], [.3, 0.]], dtype=np.float32) if points is None else points
    view = SimpleNamespace(nodes=['r', 'p', 'c', 'u'], root='r', reachable={'r', 'p', 'c', 'u'} if both_known else {'r', 'p', 'c'})
    panel = {'pool_size': 2, 'rows': [{'id': n, 'index': i, 'inclusion_probability': 1., 'pool_mean_weight': .5}
                                    for n, i in [('c', 2), ('u', 3)]],
             'relations': [{'child': n, 'direct_parents': ['p'], 'positive_distant_ancestors': ['r']} for n in ('c', 'u')]}
    evaluator = HierarchyPanel(view, panel, full)
    queries = [(1, 2), (0, 3)]; truth = {2: {1}, 3: {0}}
    ranking = {'status': 'complete', 'expected_queries': 2, 'completed_queries': 2, 'completed_children': 2,
               'query_micro_mrr': (1/2 + 1/3)/2, 'child_macro_mrr': (1/2 + 1/3)/2,
               'hits': {'1': 0., '3': 1., '10': 1.},
               'rows': [{'parent': 1, 'child': 2, 'rank': 2., 'candidates': 3},
                        {'parent': 0, 'child': 3, 'rank': 3., 'candidates': 3}]}
    binding = {'source_commit': ORIGINAL_SOURCE, 'config_sha256': 'a' * 64, 'seed': 11, 'step': 4}
    plan = {'fanout': None, 'indices': np.array([[0], [0]], dtype=np.int64), 'weights_fp64': np.ones(1, dtype=np.float64)}
    reference = {'native_ball_points': full, 'ranking': ranking, 'binding': binding,
                 'model_state_sha256': 'b' * 64, 'full_plan': plan}
    config = {'model': {'hidden': 2, 'c': 1.}, 'prepared': {'nodes_count': 4}}
    return full, reference, evaluator, config, queries, truth, binding, plan


def qualify(example_data, points=None, ranking=None, candidate_policy=None, state=None, binding=None, plan=None):
    full, ref, ev, cfg, queries, truth, original_binding, original_plan = example_data
    return qualify_full(full if points is None else points, ref['ranking'] if ranking is None else ranking,
                        ref, ev, cfg, policy() if candidate_policy is None else candidate_policy, queries, truth,
                        ref['model_state_sha256'] if state is None else state,
                        original_binding if binding is None else binding, original_plan if plan is None else plan)


def violations(result):
    return {v['metric'] for v in result['violations']}


def test_exact_policy_is_separate_from_unchanged_training_config():
    cfg = json.loads((Path(__file__).parents[1] / 'configs/mature_hgcn_validation.json').read_bytes())
    assert validate_policy(policy(), cfg) == POLICY_SHA256
    modified = policy(); modified['limits']['point_max_abs'] *= 100
    with pytest.raises(ValueError, match='exact separately frozen'):
        validate_policy(modified, cfg)


def test_small_numeric_difference_and_reordered_queries_keep_original_F():
    data = example(); points = data[0].copy(); points[2, 0] += np.float32(1e-7)
    rank = deepcopy(data[1]['ranking']); rank['rows'].reverse()
    result = qualify(data, points, rank)
    assert result['accepted'] and not result['original_bit_exact']['native_points']
    assert result['original_bit_exact']['rank_rows'] and result['limits_are_guaranteed_error_bounds'] is False
    assert np.array_equal(data[2].full, data[0].astype(np.float64))


def test_coordinate_max_and_RMS_each_have_a_gate():
    data = example(); point = data[0].copy(); point[2, 0] += np.float32(4e-6)
    assert 'point_max_abs' in violations(qualify(data, point))
    common = data[0] + np.float32(1e-7); result = qualify(data, common)
    assert 'point_RMS' in violations(result) and 'point_max_abs' not in violations(result)


def test_geodesic_gate_catches_boundary_amplification_below_coordinate_limit():
    points = example()[0].copy(); points[2, 0] = np.float32(.996); data = example(points)
    changed = points.copy(); changed[2, 0] += np.float32(2e-6)
    result = qualify(data, changed)
    assert 'max_geodesic' in violations(result) and 'point_max_abs' not in violations(result)


def test_pair_gap_gate_is_separate_from_geodesic_limit():
    points = example()[0].copy(); points[1, 0] = .99; points[2, 0] = .993; data = example(points)
    changed = points.copy(); changed[1, 0] += np.float32(1e-6)
    result = qualify(data, changed)
    assert 'direct/pair_gap_max_abs_delta' in violations(result) and 'max_geodesic' not in violations(result)


def test_balanced_sign_flips_reject_even_when_order_aggregate_unchanged():
    points = example()[0].copy(); points[2, 0] = np.nextafter(points[1, 0], np.float32(1.))
    points[3, 0] = np.nextafter(points[1, 0], np.float32(0.)); data = example(points, both_known=True)
    changed = points.copy(); changed[[2, 3]] = changed[[3, 2]]
    result = qualify(data, changed)
    assert result['measured']['hierarchy']['direct']['order_abs_delta'] == 0
    assert 'direct/pair_sign_changes' in violations(result) and 'point_max_abs' not in violations(result)


def test_finite_interior_FP32_and_fixed_evaluator_basis_are_strict():
    data = example()
    for changed in (data[0].astype(np.float64), np.full_like(data[0], np.nan), np.ones_like(data[0])):
        with pytest.raises(ValueError): qualify(data, changed)
    wrong = list(data); changed = data[0].copy(); changed[0, 0] += np.float32(1e-7)
    wrong[2] = example(changed)[2]
    with pytest.raises(ValueError, match='original fixed F basis'): qualify(wrong)


def test_weight_binding_and_full_adjacency_have_no_numeric_tolerance():
    data = example()
    with pytest.raises(ValueError, match='weight/config'): qualify(data, state='0' * 64)
    with pytest.raises(ValueError, match='weight/config'): qualify(data, binding={**data[6], 'seed': 23})
    plan = deepcopy(data[7]); plan['weights_fp64'][0] += 1e-12
    with pytest.raises(ValueError, match='adjacency identity'): qualify(data, plan=plan)


@pytest.mark.parametrize('mutation', ['query', 'candidate', 'row_nan', 'aggregate_nan', 'stale_mrr'])
def test_complete_rank_identity_finite_values_and_aggregates_are_strict(mutation):
    data = example(); rank = deepcopy(data[1]['ranking'])
    if mutation == 'query': rank['rows'][0]['parent'] = 0
    elif mutation == 'candidate': rank['rows'][0]['candidates'] += 1
    elif mutation == 'row_nan': rank['rows'][0]['rank'] = float('nan')
    elif mutation == 'aggregate_nan': rank['query_micro_mrr'] = float('nan')
    else: rank['query_micro_mrr'] += .001
    with pytest.raises(ValueError): qualify(data, ranking=rank)


def test_rank_count_rank_size_and_MRR_are_distinct_limits():
    data = example(); rank = deepcopy(data[1]['ranking']); rank['rows'][0]['rank'] = 3.
    rank['query_micro_mrr'] = rank['child_macro_mrr'] = 1/3
    candidate = policy(); candidate['limits']['rank_changed_queries'] = 0; candidate['limits']['rank_max_abs_delta'] = 0
    result = qualify(data, ranking=rank, candidate_policy=candidate)
    assert {'rank_changed_queries', 'rank_max_abs_delta', 'micro_mrr_abs_delta', 'macro_mrr_abs_delta'} <= violations(result)
    with pytest.raises(ValueError, match='qualification failed'): require_qualified(result)


def test_contrived_runtime_fixture_saves_both_acceptance_and_rejection(tmp_path):
    from acl_hct.diagnostic_archive import read_archive
    from acl_hct.hgcn_replay import qualification_fixture
    result = qualification_fixture(tmp_path, torch.device('cpu'), policy())
    saved = read_archive(tmp_path, result['archive'])
    assert saved['qualifications'][0]['accepted'] and not saved['qualifications'][1]['accepted']
    assert saved['policy_sha256'] == POLICY_SHA256


def test_historical_official_exception_requires_explicit_analysis_mode(tmp_path):
    from acl_hct.hgcn_analysis import analyze, checked_run
    from acl_hct.hgcn_evidence import atomic_json, file_hash
    from acl_hct.hgcn_registration import canonical
    cfg = {'scope': 'artificial metadata test'}; current = '1' * 40
    release = {'source_commit': current, 'replay_policy_sha256': POLICY_SHA256}
    path = tmp_path/'run.json'
    def saved(source, phase='official'):
        atomic_json(path, {'status': 'complete', 'phase': phase, 'source_commit': source,
                          'config_sha256': canonical(cfg), 'result': {'scope': 'artificial'}})
        return {'run_file': str(path), 'sha256': file_hash(path)}
    item = saved(current)
    checked_run(item, 'official', cfg, release)  # Sidecar alone must not change source.
    with pytest.raises(ValueError, match='same-source/config'):
        checked_run(item, 'official', cfg, release, historical_official=True)
    item = saved(ORIGINAL_SOURCE)
    checked_run(item, 'official', cfg, release, historical_official=True)
    with pytest.raises(ValueError, match='same-source/config'):
        checked_run(item, 'official', cfg, release)
    with pytest.raises(ValueError, match='explicit amended'):
        checked_run(item, 'official', cfg, {'source_commit': current}, historical_official=True)
    with pytest.raises(ValueError, match='explicit amended'):
        checked_run(saved(ORIGINAL_SOURCE, 'eval'), 'eval', cfg, release, historical_official=True)
    # The explicit analysis argument must match the verified release sidecar.
    cfg['protocol'] = 'artificial'
    index = tmp_path/'index.json'
    atomic_json(index, {'protocol': cfg['protocol'], 'config_sha256': canonical(cfg),
                       'replay_policy_sha256': POLICY_SHA256, 'training': [], 'evaluation': []})
    wrong = {**release, 'replay_policy_sha256': '0'*64, 'inputs': {'artifact_index_sha256': file_hash(index)}}
    with pytest.raises(ValueError, match='verified release context'):
        analyze(index, tmp_path/'not-opened', cfg, wrong, tmp_path, policy())


def test_artificial_failed_train_recovery_preserves_source_files_and_fixed_F(tmp_path, monkeypatch):
    from acl_hct.hgcn_evidence import atomic_json, file_hash
    from acl_hct.hgcn_quality import load_train
    from acl_hct.hgcn_synthetic import miniature_inputs
    from acl_hct.hgcn_upstream import load_upstream
    import acl_hct.hgcn_validation as driver
    upstream_path = os.environ.get('ACL_HGCN_UPSTREAM_PATH')
    if not upstream_path: pytest.fail('explicit pinned upstream required for recovery test')
    upstream = load_upstream(upstream_path)
    monkeypatch.setenv('ACL_HGCN_VALIDATION_PID', str(os.getppid()))
    monkeypatch.setenv('ACL_HGCN_VALIDATION_DEADLINE', str(time.monotonic() + 120))
    torch.set_num_threads(2)
    cfg = json.loads((Path(__file__).parents[1] / 'configs/mature_hgcn_validation.json').read_bytes())
    prepared = tmp_path/'inputs'; cfg = miniature_inputs(prepared, cfg); data = load_train(prepared, cfg)
    original = tmp_path/'original'; original.mkdir()
    def injected(*args): raise ValueError(ORIGINAL_FAILURE.removeprefix('ValueError: '))
    monkeypatch.setattr(driver, 'compare_full', injected)
    with pytest.raises(ValueError, match='full native point replay differs'):
        driver.train_wordnet(data, cfg, upstream, torch.device('cpu'), original, prepared, {'source_commit': ORIGINAL_SOURCE}, 11)
    run = original/'run.json'; atomic_json(run, {'status':'failed','phase':'train','source_commit':ORIGINAL_SOURCE,
         'config_sha256':driver.canonical(cfg),'error':ORIGINAL_FAILURE,'scope':'ARTIFICIAL four-step CPU quality test'})
    progress=json.loads((original/'training-progress.json').read_bytes()); best=progress['best']; candidate=policy()
    origin={'source_commit':ORIGINAL_SOURCE,'training_run_sha256':file_hash(run),
            'training_progress_sha256':file_hash(original/'training-progress.json'),
            'best_checkpoint_sha256':best['checkpoint']['sha256'],'best_reference_manifest_sha256':best['reference']['manifest_sha256'],
            'batch_history_manifest_sha256':file_hash(original/'arrays/training-history.json'),
            'initial_reference_manifest_sha256':file_hash(original/'arrays/initial-full-reference.json'),
            'last_checkpoint_sha256':file_hash(original/'last.pt'),'best_step':best['step']}
    candidate['origins']['11']=origin
    before={str(p):file_hash(p) for p in original.rglob('*') if p.is_file()}
    recovered=tmp_path/'recovered';recovered.mkdir()
    release={'source_commit':'1'*40,'inputs':{'training_run_sha256':file_hash(run),'best_checkpoint_sha256':best['checkpoint']['sha256']}}
    result=driver.replay_wordnet(data,cfg,upstream,torch.device('cpu'),recovered,prepared,run,release,11,candidate)
    assert result['replay_qualification']['accepted'] and result['original_failure_preserved']
    assert result['parameters_updated'] is False and result['optimizer_created'] is False
    assert before=={str(p):file_hash(p) for p in original.rglob('*') if p.is_file()}
    assert json.loads(run.read_bytes())['status']=='failed'
    assert result['full_baseline']['micro_mrr']==best['micro_mrr']
    recovered_run=recovered/'run.json';atomic_json(recovered_run,{'status':'complete','phase':'replay','source_commit':'1'*40,
               'config_sha256':driver.canonical(cfg),'result':result})
    released={**release,'inputs':{**release['inputs'],'training_run_sha256':file_hash(recovered_run)}}
    original_root,trained=driver.training_input(recovered_run,cfg,released,11,candidate)
    assert original_root==original and trained['best']==best
    # A numerical outlier preserves fresh raw evidence before rejecting;
    # it must never overwrite the preserved old failure or widen the policy.
    original_factory=driver.new_model
    def outlier_factory(*args):
        model=original_factory(*args);original_encode=model.encode
        def encode(*a,**k):
            points=original_encode(*a,**k).clone();points[:,0]+=1e-4;return points
        model.encode=encode;return model
    monkeypatch.setattr(driver,'new_model',outlier_factory)
    rejected=tmp_path/'rejected';rejected.mkdir()
    with pytest.raises(ValueError,match='numerical full replay qualification failed'):
        driver.replay_wordnet(data,cfg,upstream,torch.device('cpu'),rejected,prepared,run,release,11,candidate)
    qualification=json.loads((rejected/'replay-qualification.json').read_bytes())
    assert qualification['qualification']['accepted'] is False
    from acl_hct.diagnostic_archive import read_archive
    assert read_archive(rejected/'arrays',qualification['archive'])['qualification']['accepted'] is False
    assert before=={str(p):file_hash(p) for p in original.rglob('*') if p.is_file()}
    # Tampered historical bytes cannot silently inherit amended acceptance.
    with (original/'training-progress.json').open('ab') as stream:stream.write(b' ')
    with pytest.raises(ValueError,match='artifact identity'):historical_training(run,cfg,candidate,11)


def test_complete_amended_artificial_pipeline_keeps_two_original_F_and_Holm12(tmp_path, monkeypatch):
    """Exercise revised recovery/shard/analysis branches with actual official NN."""
    from acl_hct.diagnostic_archive import read_archive
    from acl_hct.hgcn_analysis import analyze
    from acl_hct.hgcn_evidence import atomic_json, file_hash
    from acl_hct.hgcn_official import train_official
    from acl_hct.hgcn_quality import load_train
    from acl_hct.hgcn_synthetic import miniature_inputs
    from acl_hct.hgcn_upstream import load_upstream
    import acl_hct.hgcn_validation as driver
    external=os.environ.get('ACL_HGCN_UPSTREAM_PATH')
    if not external:pytest.fail('explicit pinned upstream required for amended pipeline')
    upstream=load_upstream(external);torch.set_num_threads(2)
    monkeypatch.setenv('ACL_HGCN_VALIDATION_PID',str(os.getppid()))
    monkeypatch.setenv('ACL_HGCN_VALIDATION_DEADLINE',str(time.monotonic()+180))
    registered=json.loads((Path(__file__).parents[1]/'configs/mature_hgcn_validation.json').read_bytes())
    prepared=tmp_path/'inputs';cfg=miniature_inputs(prepared,registered);data=load_train(prepared,cfg);candidate=policy()
    new_source='1'*40;old_runs={};old_hashes={}
    def injected(*args):raise ValueError(ORIGINAL_FAILURE.removeprefix('ValueError: '))
    monkeypatch.setattr(driver,'compare_full',injected)
    for seed in (11,23):
        root=tmp_path/f'original-{seed}';root.mkdir()
        with pytest.raises(ValueError,match='full native point replay differs'):
            driver.train_wordnet(data,cfg,upstream,torch.device('cpu'),root,prepared,{'source_commit':ORIGINAL_SOURCE},seed)
        run=root/'run.json';atomic_json(run,{'status':'failed','phase':'train','source_commit':ORIGINAL_SOURCE,
             'config_sha256':driver.canonical(cfg),'error':ORIGINAL_FAILURE,'scope':'ARTIFICIAL simulated old four-step training lineage'})
        progress=json.loads((root/'training-progress.json').read_bytes());best=progress['best']
        candidate['origins'][str(seed)]={'source_commit':ORIGINAL_SOURCE,'training_run_sha256':file_hash(run),
             'training_progress_sha256':file_hash(root/'training-progress.json'),'best_checkpoint_sha256':best['checkpoint']['sha256'],
             'best_reference_manifest_sha256':best['reference']['manifest_sha256'],
             'batch_history_manifest_sha256':file_hash(root/'arrays/training-history.json'),
             'initial_reference_manifest_sha256':file_hash(root/'arrays/initial-full-reference.json'),
             'last_checkpoint_sha256':file_hash(root/'last.pt'),'best_step':best['step']}
        old_runs[seed]=run;old_hashes.update({str(p):file_hash(p) for p in root.rglob('*') if p.is_file()})
    index={'protocol':cfg['protocol'],'config_sha256':driver.canonical(cfg),'replay_policy_sha256':driver.canonical(candidate),
           'training':[],'evaluation':[]}
    def save(root,phase,result,source=new_source):
        path=root/'run.json';atomic_json(path,{'status':'complete','phase':phase,'source_commit':source,
             'config_sha256':driver.canonical(cfg),'scope':'ARTIFICIAL engineering only','result':result})
        return {'run_file':str(path.resolve()),'sha256':file_hash(path)}
    for seed in (11,23):
        root=tmp_path/f'recovered-{seed}';root.mkdir();origin=candidate['origins'][str(seed)]
        release={'source_commit':new_source,'inputs':{'training_run_sha256':file_hash(old_runs[seed]),
                 'best_checkpoint_sha256':origin['best_checkpoint_sha256']}}
        result=driver.replay_wordnet(data,cfg,upstream,torch.device('cpu'),root,prepared,old_runs[seed],release,seed,candidate)
        item=save(root,'replay',result);index['training'].append(item)
        origin_root,old=historical_training(old_runs[seed],cfg,candidate,seed)
        fixed=read_archive(origin_root/'arrays',old['best']['reference'])['native_ball_points']
        for start in (0,4,8,12):
            shard_root=tmp_path/f'eval-{seed}-{start}';shard_root.mkdir()
            shard_release={**release,'inputs':{**release['inputs'],'training_run_sha256':item['sha256']}}
            shard=driver.evaluate_shard(data,cfg,upstream,torch.device('cpu'),shard_root,prepared,item['run_file'],shard_release,seed,start,candidate)
            assert shard['full_replay']['accepted']
            for sampled in shard['samples']:
                saved=read_archive(shard_root/'arrays',sampled['artifact'])
                assert saved['full_native_points_sha256']==driver.array_hash(fixed)
            index['evaluation'].append(save(shard_root,'eval',shard))
    official_root=tmp_path/'official';official_root.mkdir()
    official=train_official(upstream,cfg,torch.device('cpu'),official_root,{'source_commit':ORIGINAL_SOURCE})
    index['official']=save(official_root,'official',official,ORIGINAL_SOURCE)
    index_path=tmp_path/'index.json';atomic_json(index_path,index);analysis_root=tmp_path/'analysis';analysis_root.mkdir()
    result=analyze(index_path,prepared,cfg,{'source_commit':new_source,'replay_policy_sha256':driver.canonical(candidate),
                   'inputs':{'artifact_index_sha256':file_hash(index_path)}},analysis_root,candidate)
    assert result['replayed_sample_arrays']==96 and len(result['primary_family'])==12
    assert len(result['full_numeric_replays'])==10 and set(result['original_training_lineage'])=={'11','23'}
    assert all(q['accepted'] for q in result['full_numeric_replays'])
    assert all(file_hash(Path(p))==h for p,h in old_hashes.items())

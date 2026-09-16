import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest
import torch
from acl_hct import capacity_control as capacity
from acl_hct import train as original_train
from acl_hct.backbone import LorentzMeanNetwork
from acl_hct.protocols import digest
from acl_hct.text_capacity import TextMLP, prepare_scores, score_cached, filtered_parent_ranks
from test_checkpoint_evaluation import prepared

ROOT = Path(__file__).parents[1]
IDENTITY = {'source_commit':'1'*40, 'source_commit_basis':'synthetic_engineering_fixture_not_git', 'git':{'available':False}}


def config():
    return json.loads((ROOT/'configs/e2_model_capacity_control.json').read_text(encoding='utf-8'))


def tiny_settings(**changes):
    return replace(capacity.TrainSettings(input_dim=4, hidden=4, head_hidden=5, batch_positives=2,
                   max_steps=4, evaluate_every=1, save_every=1, max_seconds=60, evaluation_max_seconds=10,
                   candidate_chunk=7), **changes)


def test_production_dimensions_gradients_and_parameter_count():
    torch.set_num_threads(2); torch.manual_seed(18)
    model = TextMLP()
    features = torch.randn(9,128); queries = torch.tensor([[0,1],[2,1],[3,4],[5,4]])
    logits = model(features, queries)
    assert model.encode(features).shape == (9,128) and logits.shape == (4,)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([1.,0.,1.,0.])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not any(isinstance(m, (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.LayerNorm)) for m in model.modules())
    assert sum(p.numel() for p in model.parameters()) == 82433
    assert sum(p.numel() for p in LorentzMeanNetwork(128).parameters()) == 82433
    with pytest.raises(ValueError): model(features, torch.tensor([[-1,0]]))
    with pytest.raises(ValueError): model(features, queries.float())


def test_query_only_encoding_matches_full_loss_and_gradients():
    torch.manual_seed(29); model = TextMLP(4,5,6).double(); other = copy.deepcopy(model)
    features = torch.randn(10,4,dtype=torch.float64); queries = torch.tensor([[0,1],[2,1],[0,3],[5,3]])
    query_loss = model(features, queries).sum(); full_loss = other.score(other.encode(features), queries).sum()
    torch.testing.assert_close(query_loss, full_loss, atol=1e-14, rtol=1e-13)
    query_loss.backward(); full_loss.backward()
    for a,b in zip(model.parameters(),other.parameters()):
        torch.testing.assert_close(a.grad,b.grad,atol=1e-14,rtol=1e-13)
    # Adding unrelated entities cannot affect a query's embedding or score.
    torch.testing.assert_close(model(features,queries), model(torch.cat([features,torch.randn(3,4,dtype=torch.float64)]),queries), atol=1e-14,rtol=1e-13)


def test_production_width_fixture_cache_check():
    torch.manual_seed(53); model = TextMLP().eval()
    result = capacity.cache_check(model, torch.randn(40,128)*.1)
    assert result['status'] == 'passed' and result['pairs'] == 1600
    assert result['maximum_absolute_score_difference'] <= result['absolute_tolerance']
    failed = capacity.cache_check(model, torch.full((40,128),float('nan')))
    assert failed['status'] == 'failed' and failed['nonfinite_scores']
    assert failed['maximum_absolute_score_difference'] is None
    json.dumps(failed,allow_nan=False)


@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
def test_cache_matches_direct_score_and_invalidates_on_encoder_update(dtype):
    torch.manual_seed(31); model = TextMLP(4,5,6).to(dtype=dtype).eval()
    embeddings = model.encode(torch.randn(9,4,dtype=dtype)); queries = torch.cartesian_prod(torch.arange(9),torch.arange(9))
    cache = prepare_scores(model, embeddings)
    torch.testing.assert_close(score_cached(model,cache,queries[:,0],queries[:,1]), model.score(embeddings,queries),
                               atol=3e-8 if dtype == torch.float32 else 1e-14, rtol=1e-6 if dtype == torch.float32 else 1e-13)
    with torch.no_grad(): model.encoder[0].weight.add_(.01)
    with pytest.raises(ValueError,match='stale'): score_cached(model,cache,queries[:,0],queries[:,1])


def test_all_candidate_ranks_against_exhaustive_reference_and_ties():
    torch.manual_seed(37); model = TextMLP(3,4,5).double().eval()
    embeddings = model.encode(torch.randn(8,3,dtype=torch.float64))
    truth = {3:{0,1},7:{2}}; queries = [(0,3),(1,3),(2,7)]
    result = filtered_parent_ranks(model,embeddings,queries,truth,candidate_chunk=3)
    for row in result['rows']:
        parent, child = row['parent'], row['child']
        candidates = [p for p in range(8) if p != child and (p not in truth[child] or p == parent)]
        scores = model.score(embeddings,torch.tensor([[p,child] for p in candidates]))
        target = scores[candidates.index(parent)]
        assert row['rank'] == 1+int((scores>target).sum())+.5*(int((scores==target).sum())-1)
        assert row['candidates'] == len(candidates)
    with torch.no_grad():
        for p in model.relation_head.parameters(): p.zero_()
    tied = filtered_parent_ranks(model,embeddings,queries,truth,candidate_chunk=3)
    assert [r['rank'] for r in tied['rows']] == [3.5,3.5,4.]
    assert tied['query_micro_mrr'] == pytest.approx((2/3.5+1/4)/3)
    assert tied['child_macro_mrr'] == pytest.approx((1/3.5+1/4)/2)
    partial = filtered_parent_ranks(model,embeddings,queries,truth,max_seconds=0)
    assert partial['status'] == 'incomplete_time_limit' and partial['query_micro_mrr'] is None
    with pytest.raises(ValueError): filtered_parent_ranks(model,embeddings,queries+queries,truth)


def test_fixed_registration_approval_and_cuda_binding(tmp_path):
    cfg = config(); assert capacity.validate_config(cfg) == capacity.CONFIG_SHA256
    for section,key in [('training','max_steps'),('training','evaluate_every'),('model','hidden'),('data','split_seed')]:
        changed = copy.deepcopy(cfg); changed[section][key] += 1
        with pytest.raises(ValueError): capacity.validate_config(changed)
    settings = capacity.settings_from_config(cfg,23)
    assert settings.seed == 23 and settings.max_steps == 1024 and settings.evaluate_every == 256
    with pytest.raises(ValueError): capacity.settings_from_config(cfg,17)
    row = {'capacity_fixture_passed':True, 'status':'complete', 'source':IDENTITY,
           'config_sha256':capacity.CONFIG_SHA256, 'source_sha256_normalized_lf':capacity.source_hashes()}
    path = tmp_path/'cuda.json'; path.write_text(json.dumps(row))
    approval = {'user_authorized':True,'user_message_reference':'offline fixture only', 'scope':capacity.PROTOCOL,
                'config_sha256':capacity.CONFIG_SHA256,'source_commit':IDENTITY['source_commit'],
                'quality_review_passed':True,'entry_criteria_frozen':True,'cuda_fixture_passed':True,
                'cuda_fixture_artifact_sha256':capacity.file_sha256(path)}
    capacity.verify_approval(approval,cfg,IDENTITY); capacity.verify_cuda_evidence(path,approval,IDENTITY)
    for key in ('user_authorized','user_message_reference','scope','config_sha256','source_commit','quality_review_passed','entry_criteria_frozen','cuda_fixture_passed'):
        bad = dict(approval); bad.pop(key)
        with pytest.raises(ValueError): capacity.verify_approval(bad,cfg,IDENTITY)
    row['source_sha256_normalized_lf']['acl_hct/text_capacity.py'] = '0'*64; path.write_text(json.dumps(row))
    approval['cuda_fixture_artifact_sha256'] = capacity.file_sha256(path)
    with pytest.raises(ValueError,match='exact source'): capacity.verify_cuda_evidence(path,approval,IDENTITY)


def tree_hashes(path):
    return {str(p.relative_to(path)):capacity.file_sha256(p) for p in path.rglob('*') if p.is_file()}


def test_runner_matches_original_batches_labels_and_ignores_graph(prepared,tmp_path,monkeypatch):
    original_read = Path.read_text; reads = []
    def checked_read(path,*a,**kw):
        if path.parent == prepared:
            assert path.name in {'input_manifest.json','observed_graph.json','train_queries.json','evaluator_valid.json'}
            reads.append(path.name)
        return original_read(path,*a,**kw)
    monkeypatch.setattr(Path,'read_text',checked_read)
    before = tree_hashes(prepared); data = original_train.load_prepared(prepared)
    group_index = {tuple(g[0]):i for i,g in enumerate(data['query_groups'].tolist())}
    batches = []; mask = original_train.mask_indexed_queries
    def capture(neighbors, positives):
        batches.append([group_index[tuple(p)] for p in positives])
        return mask(neighbors,positives)
    monkeypatch.setattr(original_train,'mask_indexed_queries',capture)
    settings = tiny_settings()
    old_settings = original_train.TrainConfig(hidden=4,head_hidden=5,batch_positives=2,max_steps=4,
        max_seconds=60,evaluation_max_seconds=10,save_every=1,evaluate_every=1,evaluation='full_validation',candidate_chunk=7)
    original_train.run(prepared,tmp_path/'original',old_settings,identity=IDENTITY)
    seen_queries = []; forward = TextMLP.forward
    def capture_mlp(model,features,queries):
        seen_queries.append(queries.cpu().tolist()); return forward(model,features,queries)
    monkeypatch.setattr(TextMLP,'forward',capture_mlp)
    report = capacity.run(data,tmp_path/'text',settings,identity=IDENTITY)
    assert report['status'] == 'complete' and len(report['evaluations']) == 4
    assert [s['batch_group_ids_sha256'] for s in report['steps']] == [digest(b) for b in batches]
    assert seen_queries == [data['query_groups'][b].reshape(-1,2).tolist() for b in batches]
    original = torch.load(tmp_path/'original/last.pt',map_location='cpu',weights_only=True)
    saved = torch.load(tmp_path/'text/last.pt',map_location='cpu',weights_only=True)
    assert torch.equal(saved['batch_rng'],original['batch_rng'])
    assert saved['source_sha256_normalized_lf'] == capacity.source_hashes()
    assert saved['completed_steps'] == 4 and saved['run_status'] == 'complete' and saved['optimizer']['state']
    assert saved['weights_sha256'] == {k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in saved['model'].items()}
    altered = dict(data); altered['neighbors'] = object()  # Never consulted by the text runner.
    again = capacity.run(altered,tmp_path/'text-other-graph',settings,identity=IDENTITY)
    last = torch.load(tmp_path/'text-other-graph/last.pt',map_location='cpu',weights_only=True)
    assert [s['loss'] for s in report['steps']] == [s['loss'] for s in again['steps']]
    for k,v in last['model'].items(): assert torch.equal(v,saved['model'][k])
    for row in report['evaluations']: capacity.validate_ranking(row,data)
    assert tree_hashes(prepared) == before
    assert set(reads) == {'input_manifest.json','observed_graph.json','train_queries.json','evaluator_valid.json'}
    with pytest.raises(FileExistsError): capacity.run(data,tmp_path/'text',settings,identity=IDENTITY)


def constant_ranking(data,rank=2.):
    truth = {child:{a for a,b in data['valid'] if b == child} for _,child in data['valid']}
    rows = [{'parent':a,'child':b,'rank':rank,'candidates':len(data['nodes'])-len(truth[b])} for a,b in data['valid']]
    return {'status':'complete','metric_scope':'filtered_all_entity_candidates','expected_queries':len(rows),
            'completed_queries':len(rows),'completed_children':len(truth),'query_micro_mrr':1/rank,
            'child_macro_mrr':1/rank,'rows':rows}


def test_first_best_tie_and_saved_selected_weights(prepared,tmp_path,monkeypatch):
    data = original_train.load_prepared(prepared)
    monkeypatch.setattr(capacity,'filtered_parent_ranks',lambda *a,**kw:copy.deepcopy(constant_ranking(data)))
    report = capacity.run(data,tmp_path/'tie',tiny_settings(),identity=IDENTITY)
    assert report['status'] == 'complete' and report['best_step'] == 1 and report['best_full_valid_mrr'] == .5
    best = torch.load(tmp_path/'tie/best.pt',map_location='cpu',weights_only=True)
    last = torch.load(tmp_path/'tie/last.pt',map_location='cpu',weights_only=True)
    assert best['completed_steps'] == best['best_step'] == 1 and best['selection_evaluation_complete']
    assert any(not torch.equal(v,last['model'][k]) for k,v in best['model'].items())
    assert capacity.file_sha256(tmp_path/'tie/best.pt') == report['checkpoints']['best.pt']['sha256']


def test_partial_valid_stops_without_selection_or_later_training(prepared,tmp_path,monkeypatch):
    data = original_train.load_prepared(prepared); calls = []
    def partial(*a,**kw):
        calls.append(1); result = constant_ranking(data); result['status'] = 'incomplete_time_limit'; return result
    monkeypatch.setattr(capacity,'filtered_parent_ranks',partial)
    report = capacity.run(data,tmp_path/'partial',tiny_settings(),identity=IDENTITY)
    assert report['status'] == 'incomplete_time_limit' and report['completed_steps'] == 1 and len(calls) == 1
    assert report['best_step'] is None and not (tmp_path/'partial/best.pt').exists()
    assert report['evaluations'][0]['status'] == 'incomplete_time_limit'
    saved = torch.load(tmp_path/'partial/last.pt',weights_only=True,map_location='cpu')
    assert saved['run_status'] == 'incomplete_time_limit'


def test_nonfinite_gradient_and_expired_budget_preserve_failure(prepared,tmp_path,monkeypatch):
    data = original_train.load_prepared(prepared); constructor = capacity.TextMLP
    def invalid(*a,**kw):
        model = constructor(*a,**kw)
        model.encoder[0].weight.register_hook(lambda g:torch.full_like(g,float('nan')))
        return model
    monkeypatch.setattr(capacity,'TextMLP',invalid)
    report = capacity.run(data,tmp_path/'failed',tiny_settings(),identity=IDENTITY)
    assert report['status'] == 'failed' and report['completed_steps'] == 0 and 'gradient' in report['error']
    assert torch.load(tmp_path/'failed/last.pt',weights_only=True,map_location='cpu')['run_status'] == 'failed'
    monkeypatch.setattr(capacity,'TextMLP',constructor)
    expired = capacity.run(data,tmp_path/'expired',tiny_settings(),identity=IDENTITY,started=time.perf_counter()-61)
    assert expired['status'] == 'incomplete_time_limit' and expired['steps'] == [] and expired['evaluations'] == []


def test_baseline_binding_ranks_and_per_step_comparison(prepared,tmp_path,monkeypatch):
    data = original_train.load_prepared(prepared); cfg = config()
    cfg['model']['input_dim'] = 4; cfg['data']['prepared_manifest_hash'] = data['manifest_hash']; cfg['data']['valid_queries_hash'] = data['valid_hash']
    baseline = {'config':cfg['baseline_anchors']['11']['training_config'], 'source':{'source_commit':cfg['baseline_anchors']['11']['training_commit']},
                'status':'step_limit_reached','completed_steps':1024,'manifest_hash':data['manifest_hash'],
                'validation_queries_hash':data['valid_hash'],'best_full_valid_mrr':.5,'elapsed_seconds':12.,
                'evaluations':[{**constant_ranking(data), 'step':s,'purpose':'full_validation'} for s in (256,512,768,1024)]}
    cfg['baseline_anchors']['11']['step'] = 256
    path = tmp_path/'baseline.json'; path.write_text(json.dumps(baseline))
    cfg['baseline_anchors']['11']['baseline_report_sha256'] = capacity.file_sha256(path)
    monkeypatch.setattr(capacity,'validate_config',lambda _:capacity.CONFIG_SHA256)  # Synthetic test binding only.
    assert capacity.verify_baseline(path,cfg,11,data) == baseline
    wrong = dict(data); wrong['valid_hash'] = '0'*64
    with pytest.raises(ValueError,match='identity'): capacity.verify_baseline(path,cfg,11,wrong)
    baseline['evaluations'][0]['rows'][0]['rank'] = 0; path.write_text(json.dumps(baseline))
    cfg['baseline_anchors']['11']['baseline_report_sha256'] = capacity.file_sha256(path)
    with pytest.raises(ValueError,match='rank'): capacity.verify_baseline(path,cfg,11,data)
    report = {'evaluations':[{**constant_ranking(data,4.),'step':256}], 'best_step':256}
    baseline['evaluations'][0] = {**constant_ranking(data),'step':256,'purpose':'full_validation'}
    comparison = capacity.compare_to_baseline(report,baseline)
    assert comparison['best']['query_micro_mrr']['text_minus_gnn'] == -.25
    assert comparison['best']['gnn_step'] == 256


def test_cpu_engineering_fixture_and_no_git_static_cli(tmp_path):
    data = capacity.engineering_data()
    result = capacity.run(data,tmp_path/'cpu-fixture',replace(capacity.TrainSettings(),max_steps=4,evaluate_every=1,
        save_every=1,batch_positives=4,max_seconds=60,evaluation_max_seconds=10,candidate_chunk=11),identity=IDENTITY,engineering=True)
    assert result['status'] == 'complete' and result['parameter_count'] == 82433 and len(result['evaluations']) == 4
    release = tmp_path/'release'; package = release/'src/acl_hct'; package.mkdir(parents=True)
    for name in capacity.SOURCE_NAMES: shutil.copyfile(ROOT/'src/acl_hct'/(name+'.py'),package/(name+'.py'))
    (release/'configs').mkdir(); shutil.copyfile(ROOT/'configs/e2_model_capacity_control.json',release/'configs/e2_model_capacity_control.json')
    assert subprocess.run(['git','rev-parse','--show-toplevel'],cwd=release,capture_output=True).returncode != 0
    command = [sys.executable,'-m','acl_hct.capacity_control','--config','configs/e2_model_capacity_control.json']
    env = {**os.environ,'PYTHONPATH':str(release/'src')}
    result = subprocess.run(command,cwd=release,env=env,capture_output=True,text=True)
    assert result.returncode == 0 and json.loads(result.stdout)['status'] == 'static_only'
    assert subprocess.run(command+['--execute'],cwd=release,env=env,capture_output=True).returncode != 0


@pytest.mark.parametrize('status',['complete','failed','incomplete_time_limit'])
def test_cli_incomplete_result_exits_nonzero(tmp_path,monkeypatch,status):
    identity = capacity.source_identity()['source_commit'] or '1'*40
    path = tmp_path/'approval.json'; path.write_text('{}')
    argv = ['capacity','--config',str(ROOT/'configs/e2_model_capacity_control.json'),'--execute','--seed','11',
            '--source-commit',identity,'--prepared',str(tmp_path/'prepared'),'--output-dir',str(tmp_path/'output'),
            '--baseline-report',str(path),'--approval-record',str(path),'--cuda-fixture-record',str(path)]
    monkeypatch.setattr(sys,'argv',argv)
    for name in ('verify_approval','verify_cuda_evidence','verify_baseline'): monkeypatch.setattr(capacity,name,lambda *a: {})
    monkeypatch.setattr(capacity,'load_prepared',lambda *a: {})
    def fake_run(*a,**kw):
        assert a[2].max_steps == 1024 and a[2].evaluate_every == 256
        output = a[1]; output.mkdir(); capacity.atomic_json(output/'run.json',{'status':status})
        return {'status':status,'completed_steps':1024 if status == 'complete' else 1,'best_step':None}
    monkeypatch.setattr(capacity,'run',fake_run)
    if status == 'complete': capacity.main()
    else:
        with pytest.raises(SystemExit) as stop: capacity.main()
        assert stop.value.code == 2
    assert json.loads((tmp_path/'output/run.json').read_text())['status'] == status

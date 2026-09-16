import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from scipy.stats import ttest_1samp
from acl_hct import confirmation_analysis as analysis
from acl_hct import confirmation_registration as reg
from acl_hct import e2_confirmation as confirm
from acl_hct import e2_development as dev
from acl_hct.diagnostic_archive import write_archive
from acl_hct.frozen_fixture import synthetic_fixture
from acl_hct.frozen_forward import PlanStreams

ROOT = Path(__file__).parents[1]


def config():
    return json.loads((ROOT/'configs/e2_independent_confirmation.json').read_text(encoding='utf-8'))


def test_registration_and_variance_only_plan():
    cfg = config()
    assert reg.validate_config(cfg) == reg.REGISTRATION_SHA256
    assert reg.verify_development(cfg, ROOT/'reports/e2-multibudget-all-results-index.json')['summaries_verified'] == 10
    assert sum(s['repetitions'] for s in cfg['shards']) == 2816
    assert sum(s['task_repetitions'] for s in cfg['shards']) == 448
    for field, value in [('repetitions', 128), ('task_repetitions', 8), ('internal_seconds', 9000)]:
        changed = copy.deepcopy(cfg); changed['shards'][0][field] = value
        with pytest.raises(ValueError): reg.validate_config(changed)
    changed = copy.deepcopy(cfg); changed['inference']['family_size'] = 40
    with pytest.raises(ValueError): reg.validate_config(changed)
    with pytest.raises(ValueError): reg.shard(cfg, 42, 4)


def test_panel_binding_and_independent_random_streams(tmp_path):
    model, features, view = synthetic_fixture()
    panels = dev.entry.make_panels(view, 1000)
    binding = {'combined_hash': panels['hash'], **{k: {s: panels['panels'][k][s] for s in ('panel_hash', 'relation_hash')}
               for k in ('development', 'diagnostic_confirmation')}}
    design = dev.DiagnosticDesign(protocol=reg.PROTOCOL, panel='diagnostic_confirmation', base_seed=reg.BASE_SEED,
                                 panel_binding=binding, registration_sha256=reg.REGISTRATION_SHA256)
    dev.validate_panels(panels, design)
    bad = copy.deepcopy(panels); bad['panels']['diagnostic_confirmation']['panel_hash'] = '0'*64
    with pytest.raises(ValueError, match='identity'): dev.validate_panels(bad, design)
    bad = copy.deepcopy(panels); bad['panels']['diagnostic_confirmation']['rows'] = bad['panels']['development']['rows']
    with pytest.raises(ValueError, match='overlap'): dev.validate_panels(bad, design)
    old = PlanStreams(dev.BASE_SEED, 'E2-multibudget-development-v1/seed11/fanout4')
    new = PlanStreams(reg.BASE_SEED, reg.PROTOCOL+'/seed11/fanout4')
    assert not set(old.seeds) & set(new.seeds)
    assert old.draw(view.neighbors, 4) != new.draw(view.neighbors, 4)


@pytest.fixture
def engineering_result(tmp_path):
    torch.set_num_threads(2)
    model, features, view = synthetic_fixture()
    panels = dev.entry.make_panels(view, 1000)
    binding = {'combined_hash': panels['hash'], **{k: {s: panels['panels'][k][s] for s in ('panel_hash', 'relation_hash')}
               for k in ('development', 'diagnostic_confirmation')}}
    design = dev.DiagnosticDesign(protocol=reg.PROTOCOL, panel='diagnostic_confirmation', base_seed=reg.BASE_SEED,
                                 panel_binding=binding, registration_sha256=reg.REGISTRATION_SHA256)
    result = dev.diagnose(model, features, view, directory=tmp_path, fanouts=[4], repetitions=6,
                         task_repetitions=3, budget=128, candidate_chunk=11, design=design,
                         namespace=reg.PROTOCOL+'/seed11')
    assert result['status'] == 'complete', result['failures']
    return result, tmp_path, binding, model, features, view


def test_confirmation_archive_independent_recomputation(engineering_result):
    result, directory, binding, model, features, view = engineering_result
    assert result['confirmation_evaluated'] and result['evaluated_panel'] == 'diagnostic_confirmation'
    entry = analysis.read_archive(directory, result['entry_artifact'])
    budget = analysis.read_archive(directory, result['budgets']['4']['artifact'])
    values, meta = analysis.extract_repeats(entry, budget, {'repetitions': 6, 'task_repetitions': 3})
    for condition in reg.CONDITIONS:
        observed = values['geometry/'+condition]
        reference = result['budgets']['4']['geometry_groups'][condition]['V']['projection']
        assert observed.mean() == pytest.approx(reference['mean'], abs=1e-14)
        assert observed.std(ddof=1)/math.sqrt(6) == pytest.approx(reference['mc_se'], abs=1e-14)
    assert np.mean(values['task/micro_mrr_change']) == pytest.approx(result['budgets']['4']['task_statistics']['mean'][1], abs=1e-14)
    # Confirm selected structure is computed using sealed synthetic children, not development children.
    layer = entry['numerical_layers'][1]
    panel = dev.RadialPanel(view, entry['panels']['panels']['diagnostic_confirmation'], torch.from_numpy(layer['base']),
                           budget['groups'], numerical_floor=torch.from_numpy(layer['empirical_numerical_floor']))
    expected = panel.evaluate(torch.from_numpy(layer['base']))
    for kind in ('direct', 'distant'):
        assert expected[kind]['groups']['V'] == budget['full_structure']['S/S'][kind]['groups']['V']
    wrong = copy.deepcopy(budget); wrong['per_repeat_structure_and_task'][3]['task'] = wrong['per_repeat_structure_and_task'][0]['task']
    with pytest.raises(ValueError, match='registered repeats'): analysis.extract_repeats(entry, wrong, {'repetitions': 6, 'task_repetitions': 3})
    wrong = copy.deepcopy(budget); wrong['per_repeat_structure_and_task'][0]['task']['ranks'][0] = 0
    with pytest.raises(ValueError, match='ranking'): analysis.extract_repeats(entry, wrong, {'repetitions': 6, 'task_repetitions': 3})
    for audit in budget['numerical_audits']:
        analysis.check_numerics(audit['checks'], config()['numerical_limits'])
    with (directory/result['budgets']['4']['artifact']['array_file']).open('ab') as stream: stream.write(b'tamper')
    with pytest.raises(ValueError, match='NPZ'): analysis.read_archive(directory, result['budgets']['4']['artifact'])


def test_student_t_holm_and_degenerate_policy():
    got = analysis.estimate([1., 3.], .001)
    assert got['mean'] == 2. and got['se'] == pytest.approx(1.)
    # df=1 Student-t is standard Cauchy: independent analytic quantile.
    assert got['marginal_halfwidth'] == pytest.approx(math.tan(math.pi*(.975-.5)), rel=1e-10)
    assert got['p_raw'] == pytest.approx(1-2*math.atan(2)/math.pi)
    samples = np.random.default_rng(19).normal(.2, .5, 256)
    assert analysis.estimate(samples, .001)['p_raw'] == pytest.approx(ttest_1samp(samples, 0).pvalue)
    for x in ([0.]*32, [1.]*32, None, [np.nan, 1.]):
        r = analysis.estimate(x, .001)
        assert not r['inferable'] and r['p_for_adjustment'] == 1 and r['marginal_ci95'] is None
    rows = [{'id': str(i), **analysis.estimate(None, .001)} for i in range(70)]
    for i, p in enumerate((.0001, .0002, .01)):
        rows[i].update(p_for_adjustment=p, inferable=True)
    analysis.holm(rows)
    assert [r['p_holm'] for r in rows[:3]] == pytest.approx([.007, .0138, .68])
    assert [r['reject_holm_05'] for r in rows[:3]] == [True, True, False]
    with pytest.raises(ValueError): analysis.holm(rows[:69])


def test_missing_shards_retain_whole_family_and_reject_development(tmp_path):
    cfg = config()
    report = analysis.analyze(cfg, {'shards': []}, tmp_path, 'a'*40, 'b'*64, {})
    assert report['status'] == 'incomplete_evidence' and len(report['primary_hypotheses']) == 70
    assert all(r['p_holm'] == 1 and not r['inferable'] for r in report['primary_hypotheses'])
    development = json.loads((ROOT/'reports/e2-multibudget-f4-seed11.json').read_text())
    with pytest.raises(ValueError, match='provenance'):
        analysis.validate_summary(development, cfg, cfg['shards'][0], 'a'*40, 'b'*64, {})


def test_analysis_imports_no_model_and_static_cli(tmp_path):
    env = {**os.environ, 'PYTHONPATH': str(ROOT/'src')}
    result = subprocess.run([sys.executable, '-c', 'import sys; import acl_hct.confirmation_analysis; assert "torch" not in sys.modules; assert "acl_hct.e2_pilot" not in sys.modules'],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    command = [sys.executable, '-m', 'acl_hct.e2_confirmation', '--config', str(ROOT/'configs/e2_independent_confirmation.json')]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0 and json.loads(result.stdout)['status'] == 'static_only'
    assert subprocess.run(command+['--execute'], cwd=tmp_path, env=env, capture_output=True).returncode != 0


def test_confirmation_cuda_binding_requires_new_fixture(tmp_path):
    row = {'cuda_fixture_passed': True, 'confirmation_fixture_passed': True, 'status': 'complete',
           'source': {'source_commit': 'a'*40}, 'source_sha256_normalized_lf': confirm.source_hashes(),
           'registration_sha256': reg.REGISTRATION_SHA256}
    path = tmp_path/'fixture.json'; path.write_text(json.dumps(row))
    approval = {'cuda_fixture_artifact_sha256': reg.file_sha256(path)}
    confirm.verify_cuda_evidence(path, approval, {'source_commit': 'a'*40})
    row['confirmation_fixture_passed'] = False; path.write_text(json.dumps(row))
    approval['cuda_fixture_artifact_sha256'] = reg.file_sha256(path)
    with pytest.raises(ValueError, match='new confirmation'): confirm.verify_cuda_evidence(path, approval, {'source_commit': 'a'*40})


def test_complete_analysis_path_on_small_engineering_registration(engineering_result, monkeypatch):
    result, directory, binding, *_ = engineering_result
    cfg = config(); cfg['panels'] = binding
    spec = cfg['shards'][0]; spec['repetitions'] = 6; spec['task_repetitions'] = 3
    # Only this offline test uses a toy registration. Production validator remains hash-pinned.
    monkeypatch.setattr(reg, 'validate_config', lambda _: reg.REGISTRATION_SHA256)
    result.update(config=cfg, config_sha256=reg.REGISTRATION_SHA256, checkpoint_seed=11,
                  checkpoint_sha256=cfg['checkpoints']['11']['sha256'], source={'source_commit': 'a'*40},
                  source_sha256_normalized_lf={}, cuda_fixture_artifact_sha256='b'*64,
                  checkpoint_and_weights_unchanged=True, execution_internal_seconds=3300,
                  development_audit={'status':'passed', 'index_raw_sha256': cfg['development']['index_raw_sha256'], 'summaries_verified':10})
    archived = analysis.read_archive(directory, result['entry_artifact'])
    pairs_hash = reg.digest(sorted(map(tuple, archived['full_valid']['row_ids'][:, :2].tolist())))
    result['entry_full_valid_reproduction'] = {'status':'passed', 'query_coverage_identical':True,
        'current_queries':7, 'historical_queries':7, 'current_query_pairs_hash':pairs_hash,
        'historical_query_pairs_hash':pairs_hash, 'absolute_tolerance':1e-8,
        'metric_differences':{'query_micro_mrr':0., 'child_macro_mrr':0.}}
    path = directory/'summary.json'; dev.write_json(path, result)
    def run():
        return analysis.analyze(cfg, {'shards':[{'seed':11, 'fanout':4, 'summary_file':path.name,
            'summary_sha256':reg.file_sha256(path), 'artifact_directory':'.'}]}, directory, 'a'*40, 'b'*64, {})
    report = run()
    assert report['shard_audits'][0]['status'] == 'passed', report['shard_audits'][0]
    assert len(report['primary_hypotheses']) == 70
    assert [r['n'] for r in report['primary_hypotheses'][:7]] == [6,6,6,6,6,6,3]
    json.dumps(report, allow_nan=False)
    result['numerical_entry']['checks'][0]['observed']['maximum'] = 1.
    dev.write_json(path, result)
    failed = run()
    assert failed['shard_audits'][0]['status'].startswith('invalid_for_inference')
    assert all(r['p_for_adjustment'] == 1 and not r['inferable'] for r in failed['primary_hypotheses'])


@pytest.mark.parametrize('status', ['complete', 'failed', 'incomplete_time_limit'])
def test_cli_fixed_shard_deadline_and_nonzero_failure(tmp_path, monkeypatch, status):
    cfg = config(); identity = dev.source_identity()['source_commit'] or 'a'*40
    approval = {'scope':reg.PROTOCOL, 'user_authorized':True, 'user_message_reference':'engineering fixture only',
                'quality_review_passed':True, 'entry_criteria_frozen':True, 'config_sha256':reg.REGISTRATION_SHA256,
                'registration_sha256':reg.REGISTRATION_SHA256, 'source_commit':identity,
                'cuda_fixture_passed':True, 'cuda_fixture_artifact_sha256':'b'*64}
    path = tmp_path/'approval.json'; path.write_text(json.dumps(approval))
    fixture = tmp_path/'fixture.json'; fixture.write_text('{}')
    output = tmp_path/'output'
    args = ['confirmation', '--config', str(ROOT/'configs/e2_independent_confirmation.json'), '--execute',
            '--seed', '23', '--fanout', '4', '--source-commit', identity, '--approval-record', str(path),
            '--output-dir', str(output), '--cuda-fixture-record', str(fixture),
            '--development-index', str(ROOT/'reports/e2-multibudget-all-results-index.json')]
    for name in ('prepared', 'checkpoint', 'training-release', 'baseline-report'):
        args += ['--'+name, str(tmp_path/'unused')]
    monkeypatch.setattr(sys, 'argv', args)
    monkeypatch.setattr(confirm, 'verify_cuda_evidence', lambda *a: None)
    def runner(*a, **kw):
        assert kw['execution_seconds'] == 6600
        return {'status':status}
    monkeypatch.setattr(confirm.entry, 'run', runner)
    if status == 'complete': confirm.main()
    else:
        with pytest.raises(SystemExit) as stopped: confirm.main()
        assert stopped.value.code == 2
    assert json.loads((output/'summary.json').read_text())['status'] == status
    with pytest.raises(FileExistsError): confirm.main()

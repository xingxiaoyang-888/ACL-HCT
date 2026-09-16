import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch
import acl_hct.evaluate_checkpoint as evaluator
from acl_hct.train import run, load_prepared
from acl_hct.taxonomy import Taxonomy
from test_training_runner import small_config

TRAINING_FIXTURE_COMMIT = '1111111111111111111111111111111111111111'
EVALUATION_FIXTURE_COMMIT = '2222222222222222222222222222222222222222'


@pytest.fixture
def prepared(tmp_path):
    pytest.importorskip('sklearn')
    nodes = [f'n{i:02d}' for i in range(40)]
    records = {node: {'name': f'word{i}', 'definition': f'entity category{i%3} common'}
               for i, node in enumerate(nodes)}
    taxonomy = Taxonomy(records, {(nodes[(i-1)//2], nodes[i]) for i in range(1, 40)}, {})
    spec = importlib.util.spec_from_file_location('eval_prep', Path(__file__).parents[1]/'scripts/prepare_benchmark.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    path = tmp_path/'prepared'; module.prepare(taxonomy, path, dimension=4)
    return path


@pytest.fixture
def evaluation_inputs(prepared, tmp_path):
    # Synthetic checkpoint provenance is explicit and never claims a Git commit.
    fixture_identity = {'source_commit': TRAINING_FIXTURE_COMMIT,
                        'source_commit_basis': 'synthetic_test_fixture_not_a_repository_revision',
                        'git': {'available': False, 'commit': None, 'dirty': None}}
    run(prepared, tmp_path/'pilot', small_config(max_steps=1, probe_queries=1), identity=fixture_identity)
    checkpoint = tmp_path/'pilot/last.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    release = tmp_path/'training-release'
    package = release/'src/acl_hct'
    package.mkdir(parents=True)
    current_package = Path(evaluator.__file__).parent
    for name in saved['source_sha256_normalized_lf']:
        shutil.copyfile(current_package/Path(name).name, release/'src'/name)
    return {
        'prepared_root': prepared, 'checkpoint_path': checkpoint,
        'expected_checkpoint_sha256': evaluator.file_sha256(checkpoint),
        'expected_training_commit': saved['source']['source_commit'],
        'training_release': release,
        # A real checkout must retain its actual revision: production rejects
        # false declarations. A Git-free fixture archive uses this synthetic ID.
        'source_commit': evaluator.source_identity()['source_commit'] or EVALUATION_FIXTURE_COMMIT,
    }


def tree_hashes(root):
    return {str(path.relative_to(root)): evaluator.file_sha256(path)
            for path in root.rglob('*') if path.is_file()}


def test_full_validation_is_read_only_and_never_selects(evaluation_inputs, monkeypatch):
    args = evaluation_inputs
    original_read = Path.read_text
    def checked_read(path, *a, **kw):
        assert path.name not in ('evaluator_test.json', 'evaluator_truth.json')
        return original_read(path, *a, **kw)
    monkeypatch.setattr(Path, 'read_text', checked_read)
    def forbidden(*a, **kw):
        raise AssertionError('read-only evaluator attempted training or checkpoint save')
    monkeypatch.setattr(torch, 'save', forbidden)
    monkeypatch.setattr(torch.optim, 'Adam', forbidden)
    before = tree_hashes(args['prepared_root']), tree_hashes(args['checkpoint_path'].parent)
    result = evaluator.evaluate(**args, candidate_chunk=7)
    data = load_prepared(args['prepared_root'])
    assert len(data['valid']) > 1  # The training probe had only one query.
    assert result['status'] == 'complete'
    assert result['ranking']['completed_queries'] == len(data['valid'])
    assert set((r['parent'], r['child']) for r in result['ranking']['rows']) == set(data['valid'])
    assert all(r['candidates'] == len(data['nodes']) - sum(b == r['child'] for _, b in data['valid'])
               for r in result['ranking']['rows'])
    assert result['full_valid_query_micro_mrr'] == result['ranking']['query_micro_mrr']
    assert result['selection_performed'] is False
    assert not (args['checkpoint_path'].parent/'best.pt').exists()
    assert before == (tree_hashes(args['prepared_root']), tree_hashes(args['checkpoint_path'].parent))
    assert result['training_source']['source_commit'] == args['expected_training_commit']
    assert 'acl_hct/evaluate_checkpoint.py' in result['evaluation_source_sha256_normalized_lf']
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('mismatch', ['artifact', 'commit', 'source', 'valid', 'manifest'])
def test_provenance_mismatches_rejected(evaluation_inputs, mismatch):
    args = dict(evaluation_inputs)
    if mismatch == 'artifact':
        args['expected_checkpoint_sha256'] = '0'*64
    elif mismatch == 'commit':
        args['expected_training_commit'] = '0'*40
    elif mismatch == 'source':
        path = args['training_release']/'src/acl_hct/backbone.py'
        path.write_text(path.read_text()+'\n# changed historical source\n')
    elif mismatch == 'valid':
        path = args['prepared_root']/'evaluator_valid.json'
        path.write_text(json.dumps(json.loads(path.read_text())[::-1]))
    else:
        path = args['prepared_root']/'input_manifest.json'
        data = json.loads(path.read_text()); data['extra'] = 'changed manifest'
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='mismatch'):
        evaluator.evaluate(**args)


def test_budget_stop_has_no_complete_metric(evaluation_inputs, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError('expired evaluation attempted encoding')
    monkeypatch.setattr(evaluator.LorentzMeanNetwork, 'encode', forbidden)
    result = evaluator.evaluate(**evaluation_inputs, max_seconds=1e-9)
    assert result['status'] == 'incomplete_time_limit'
    assert result['ranking'] is None and result['full_valid_query_micro_mrr'] is None
    assert result['selection_performed'] is False and result['stopped_before'] == 'model_setup'


def test_partial_ranking_cannot_become_complete_score(evaluation_inputs, monkeypatch):
    import acl_hct.ranking as ranking
    original = ranking.filtered_parent_ranks
    def expired(*args, **kwargs):
        kwargs['max_seconds'] = 0
        return original(*args, **kwargs)
    monkeypatch.setattr(ranking, 'filtered_parent_ranks', expired)
    result = evaluator.evaluate(**evaluation_inputs)
    assert result['status'] == 'incomplete_time_limit'
    assert result['ranking']['completed_queries'] == 0
    assert result['full_valid_query_micro_mrr'] is None


def test_cuda_requires_allocation(evaluation_inputs, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(ValueError, match='Slurm allocation'):
        evaluator.evaluate(**evaluation_inputs, device='cuda')


def test_archive_cli_separates_training_and_evaluation_sources(evaluation_inputs, tmp_path):
    args = evaluation_inputs
    release = tmp_path/'evaluation-release'
    shutil.copytree(args['training_release'], release)
    # A distinct evaluation source byte demonstrates independent provenance.
    source = release/'src/acl_hct/evaluate_checkpoint.py'
    source.write_text(source.read_text()+'\n# archive fixture\n')
    output = tmp_path/'full-valid.json'
    declared = EVALUATION_FIXTURE_COMMIT
    command = [sys.executable, '-m', 'acl_hct.evaluate_checkpoint',
        '--prepared', str(args['prepared_root']), '--checkpoint', str(args['checkpoint_path']),
        '--training-release', str(args['training_release']), '--output', str(output),
        '--expected-checkpoint-sha256', args['expected_checkpoint_sha256'],
        '--expected-training-commit', args['expected_training_commit'], '--source-commit', declared]
    env = {**os.environ, 'PYTHONPATH': str(release/'src')}
    missing_commit = command[:-2]  # Remove --source-commit and its value.
    rejected = subprocess.run(missing_commit, cwd=release, env=env, capture_output=True)
    assert rejected.returncode != 0 and b'archive evaluation requires --source-commit' in rejected.stderr
    assert not output.exists()
    subprocess.run(command, cwd=release, env=env, check=True, capture_output=True)
    result = json.loads(output.read_text())
    assert result['status'] == 'complete'
    assert result['evaluation_source']['source_commit'] == declared
    assert result['evaluation_source']['git']['available'] is False
    assert result['training_source']['source_commit'] == args['expected_training_commit']
    assert result['evaluation_source_sha256_normalized_lf']['acl_hct/evaluate_checkpoint.py'] == hashlib.sha256(source.read_text().encode()).hexdigest()
    before = output.read_bytes()
    failed = subprocess.run(command, cwd=release, env=env, capture_output=True)
    assert failed.returncode != 0 and b'output must be a new JSON path' in failed.stderr
    assert output.read_bytes() == before

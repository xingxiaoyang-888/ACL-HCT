import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import pytest
import torch
from acl_hct.taxonomy import Taxonomy
from acl_hct.train import TrainConfig, run, load_prepared


@pytest.fixture
def prepared(tmp_path):
    pytest.importorskip('sklearn')
    nodes=[f'n{i:02d}' for i in range(20)]
    records={node:{'name':f'word{i}', 'definition':f'entity category{i%3} common'} for i,node in enumerate(nodes)}
    taxonomy=Taxonomy(records,{(nodes[(i-1)//2],nodes[i]) for i in range(1,20)}, {})
    spec=importlib.util.spec_from_file_location('prep',Path(__file__).parents[1]/'scripts/prepare_benchmark.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    path=tmp_path/'prepared';module.prepare(taxonomy,path,dimension=4)
    return path


def small_config(**kwargs):
    return replace(TrainConfig(hidden=4,head_hidden=5,batch_positives=2,max_steps=2,max_seconds=60.,
                               evaluation_max_seconds=10.,save_every=1,candidate_chunk=7,fanouts=(3,3)),**kwargs)


def test_runner_probe_saves_state_without_reading_test(prepared,tmp_path,monkeypatch):
    original=Path.read_text
    def checked(path,*args,**kwargs):
        assert path.name not in ('evaluator_test.json','evaluator_truth.json')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',checked)
    result=run(prepared,tmp_path/'run',small_config())
    assert result['status']=='step_limit_reached' and result['completed_steps']==2
    assert result['evaluations'][0]['purpose']=='validation_probe'
    assert not (tmp_path/'run/best.pt').exists() and result['best_full_valid_mrr'] is None
    checkpoint=torch.load(tmp_path/'run/last.pt',map_location='cpu')
    assert checkpoint['optimizer']['state'] and checkpoint['sampling_rng'].dtype==torch.uint8
    assert checkpoint['completed_steps']==2
    assert json.loads((tmp_path/'run/run.json').read_text())['completed_steps']==2


def test_full_validation_selection_and_exact_resume(prepared,tmp_path):
    config=small_config(evaluation='full_validation')
    continuous=run(prepared,tmp_path/'continuous',config)
    first=run(prepared,tmp_path/'first',replace(config,max_steps=1))
    resumed=run(prepared,tmp_path/'resumed',config,resume_checkpoint=tmp_path/'first/last.pt')
    assert (tmp_path/'resumed/best.pt').exists() and first['best_full_valid_mrr'] is not None
    a=torch.load(tmp_path/'continuous/last.pt',map_location='cpu')
    b=torch.load(tmp_path/'resumed/last.pt',map_location='cpu')
    for name,value in a['model'].items():assert torch.equal(value,b['model'][name])
    assert torch.equal(a['sampling_rng'],b['sampling_rng'])
    assert continuous['steps'][-1]['loss']==resumed['steps'][-1]['loss']
    assert resumed['resumed_from_step']==1


def test_budget_stop_and_prepared_tamper(prepared,tmp_path):
    result=run(prepared,tmp_path/'bounded',small_config(max_seconds=.000001,evaluation_max_seconds=.000001))
    assert result['status']=='wall_limit_reached' and result['completed_steps']==0
    assert (tmp_path/'bounded/last.pt').exists()
    graph=json.loads((prepared/'observed_graph.json').read_text());graph['neighbors'][0].append(19)
    (prepared/'observed_graph.json').write_text(json.dumps(graph))
    with pytest.raises(ValueError,match='hash mismatch'):load_prepared(prepared)


def test_cuda_requires_explicit_allocation(prepared,tmp_path,monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(ValueError,match='Slurm allocation'):
        run(prepared,tmp_path/'unallocated',small_config(),device='cuda')
    assert not (tmp_path/'unallocated').exists()


def test_archive_runner_cli_records_entry_hash(prepared,tmp_path):
    import hashlib, os, shutil, subprocess, sys
    from dataclasses import asdict
    root=Path(__file__).parents[1];release=tmp_path/'release';package=release/'src/acl_hct'
    package.mkdir(parents=True)
    for name in ('__init__.py','train.py','backbone.py','ranking.py','protocols.py','mechanisms.py','geometry.py','aggregation.py'):
        shutil.copyfile(root/'src/acl_hct'/name,package/name)
    config_path=release/'config.json';config_path.write_text(json.dumps(asdict(small_config(max_steps=1))))
    output=tmp_path/'cli-run';declared='2ca02d6033fd5503b82bbcd5b72c5f172b07eb6e'
    subprocess.run([sys.executable,'-m','acl_hct.train','--prepared',str(prepared),'--output',str(output),
                    '--config',str(config_path),'--source-commit',declared],cwd=release,
                   env={**os.environ,'PYTHONPATH':str(release/'src')},check=True,capture_output=True)
    result=json.loads((output/'run.json').read_text())
    assert result['source']['git']['available'] is False and result['source']['source_commit_basis']=='caller_declared'
    assert result['source_sha256_normalized_lf']['acl_hct/train.py']==hashlib.sha256((package/'train.py').read_text().encode()).hexdigest()
    assert result['completed_steps']==1 and result['best_full_valid_mrr'] is None

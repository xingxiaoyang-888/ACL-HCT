import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def test_archive_cli_and_plot_outside_git(tmp_path):
    root=Path(__file__).parents[1]
    archive=tmp_path/'release'; archive.mkdir()
    package=archive/'src'/'acl_hct'; package.mkdir(parents=True)
    for name in ('__init__.py','mechanisms.py','geometry.py','aggregation.py'):
        shutil.copyfile(root/'src'/'acl_hct'/name,package/name)
    config={'enumeration_threshold':20,'mc_draws':4,'chunk_size':3,
            'cases':[{'name':'archive','N':4,'k':2,'spread':.1}]}
    config_path=archive/'config.json'; config_path.write_text(json.dumps(config))
    output=archive/'result.json'; env={**os.environ,'PYTHONPATH':str(archive/'src')}
    declared='9e610cbc16acfeff0d0790c63a0a11d28c4b9cb6'
    subprocess.run([sys.executable,'-m','acl_hct.mechanisms','--config',str(config_path),
                    '--output',str(output),'--source-commit',declared],cwd=archive,env=env,check=True,capture_output=True)
    result=json.loads(output.read_text())
    assert result['git']=={'available':False,'commit':None,'dirty':None}
    assert result['source_commit']==declared and result['source_commit_basis']=='caller_declared'
    assert result['config_sha256']==hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert set(result['source_sha256_normalized_lf'])=={'acl_hct/'+name for name in ('__init__.py','mechanisms.py','geometry.py','aggregation.py')}
    assert all(result['source_sha256_normalized_lf']['acl_hct/'+p.name]==hashlib.sha256(p.read_text().encode()).hexdigest()
               for p in package.glob('*.py'))
    assert result['cases'][0]['result']['status']=='ok'
    # Plot reconstruction is optional when the plots extra is absent.
    import importlib.util
    if importlib.util.find_spec('matplotlib') is None:
        return
    shutil.copyfile(root/'scripts'/'plot_e1.py',archive/'plot_e1.py')
    subprocess.run([sys.executable,str(archive/'plot_e1.py'),'--input',str(output),'--output-dir',str(archive/'figures')],
                   cwd=archive,env=env,check=True,capture_output=True)
    assert len(list((archive/'figures').glob('*.png')))==3


def test_formal_s1_all_exact_budget():
    import math
    config=json.loads((Path(__file__).parents[1]/'configs/e1_s1.json').read_text())
    counts=[math.comb(row['N'],row['k']) for row in config['cases']]
    assert len(counts)==54 and max(counts)<=config['enumeration_threshold']
    assert sum(counts)==138483

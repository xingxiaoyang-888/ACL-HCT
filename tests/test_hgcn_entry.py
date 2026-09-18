from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

from acl_hct.hgcn_entry import supervise
from acl_hct.hgcn_registration import (CONFIG_SHA256, PROTOCOL, source_hashes, validate_config, verify_release)


ROOT = Path(__file__).parents[1]


def configuration():
    return json.loads((ROOT / 'configs/mature_hgcn_quality.json').read_bytes())


def test_fixed_configuration_and_static_cli_cannot_release_science():
    config = configuration(); assert validate_config(config) == CONFIG_SHA256
    result = subprocess.run([sys.executable, '-m', 'acl_hct.hgcn_entry', '--config',
                             str(ROOT / 'configs/mature_hgcn_quality.json')],
                            env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')}, capture_output=True, text=True, check=True)
    row = json.loads(result.stdout)
    assert row['science_released'] is False and row['status'] == 'static_only'
    assert row['phases'] == ['cpu_quality', 'cuda_quality']
    changed = deepcopy(config); changed['wordnet_quality']['quality_steps'] += 1
    with pytest.raises(ValueError, match='unregistered'):
        validate_config(changed)
    changed = deepcopy(config); changed['science_released'] = True
    with pytest.raises(ValueError, match='engineering'):
        validate_config(changed)


def test_source_scope_and_phase_are_bound_before_execution(tmp_path):
    config = configuration(); commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    record = {'protocol': PROTOCOL, 'accepted': True, 'authorized_scope': 'engineering_only', 'phase': 'cpu_quality',
              'source_commit': commit, 'config_sha256': CONFIG_SHA256, 'upstream_commit': config['upstream_commit'],
              'source_sha256_normalized_lf': source_hashes()}
    path = tmp_path / 'release.json'; path.write_text(json.dumps(record))
    assert verify_release(config, 'cpu_quality', commit, path)['accepted'] is True
    with pytest.raises(ValueError, match='bind'):
        verify_release(config, 'cuda_quality', commit, path)
    with pytest.raises(ValueError, match='engineering'):
        verify_release(config, 'science', commit, path)
    record['source_sha256_normalized_lf']['acl_hct/mature_hgcn.py'] = '0' * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='bind'):
        verify_release(config, 'cpu_quality', commit, path)


def test_deadline_stops_only_bounded_worker_and_preserves_failure(tmp_path):
    output = tmp_path / 'timeout'
    assert supervise([sys.executable, '-c', 'import time; time.sleep(10)'], output, .1, {'phase': 'test'}) == 124
    row = json.loads((output / 'supervisor.json').read_bytes())
    assert row['status'] == 'timeout' and row['deadline_seconds'] == .1
    with pytest.raises(ValueError, match='fresh'):
        supervise([sys.executable, '-c', 'pass'], output, 1, {})
    assert supervise([sys.executable, '-c', 'raise SystemExit(3)'], tmp_path / 'failed', 2, {}) == 2

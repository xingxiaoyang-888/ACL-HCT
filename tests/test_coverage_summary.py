import importlib.util
from pathlib import Path
import pytest


def test_sampling_coverage_summary():
    spec=importlib.util.spec_from_file_location('coverage',Path(__file__).parents[1]/'scripts/audit_sampling_coverage.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result=module.summarize([0,1,4,8,12])
    row=result['fanouts'][0]
    assert row['affected_nodes']==2 and row['removed_messages']==12
    assert row['affected_node_fraction']==.4
    assert row['removed_message_fraction']==pytest.approx(12/25)
    assert result['fanouts'][-1]['removed_messages']==0
    assert module.summarize([])['fanouts'][0]['affected_node_fraction'] is None

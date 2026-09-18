import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from acl_hct.hgcn_evidence import (array_hash, compare_full, file_hash, load_checkpoint,
                                   save_checkpoint, validate_ranking)
from acl_hct.hgcn_official import official_early_stop
from acl_hct.hgcn_upstream import load_upstream
from acl_hct.hgcn_validation import integration_fixture
from acl_hct.text_capacity import TextMLP, filtered_parent_ranks


def test_checkpoint_roundtrip_tamper_and_owning_directory(tmp_path):
    model = TextMLP(3, 4, 5); binding = {'seed': 11, 'step': 2}
    descriptor = save_checkpoint(tmp_path, 'step-2.pt', model, binding, {'batch_rng': torch.Generator().get_state()})
    loaded = load_checkpoint(tmp_path, descriptor, binding)
    assert all(torch.equal(v, loaded['model_state'][k]) for k, v in model.state_dict().items())
    with pytest.raises(ValueError, match='immutable'):
        save_checkpoint(tmp_path, 'step-2.pt', model, binding)
    with pytest.raises(ValueError, match='binding'):
        load_checkpoint(tmp_path, descriptor, {'seed': 11, 'step': 4})
    bad = {**descriptor, 'file': '../step-2.pt'}
    with pytest.raises(ValueError, match='escapes'):
        load_checkpoint(tmp_path, bad, binding)
    with (tmp_path / 'step-2.pt').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='file'):
        load_checkpoint(tmp_path, descriptor, binding)


def test_query_identity_replay_not_row_zip_and_candidate_tie_validation():
    model = TextMLP(3, 4, 5).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    queries = [(0, 4), (1, 4), (2, 3)]; truth = {4: {0, 1}, 3: {2}}
    ranking = filtered_parent_ranks(model, torch.zeros(5, 4), queries, truth, 2)
    assert validate_ranking(ranking, queries, truth, 5) == pytest.approx((.5 + .5 + .4) / 3)
    native = np.zeros((5, 4), dtype=np.float32); reordered = copy.deepcopy(ranking)
    reordered['rows'].reverse()
    compare_full(native, reordered, {'native_ball_points': native, 'ranking': ranking})
    for change in (lambda r: r['rows'][0].update(candidates=5),
                   lambda r: r['rows'][0].update(rank=1.25),
                   lambda r: r.update(query_micro_mrr=.9),
                   lambda r: r['rows'].pop()):
        bad = copy.deepcopy(ranking); change(bad)
        with pytest.raises(ValueError):
            validate_ranking(bad, queries, truth, 5)


def test_original_early_stop_uses_exact_equal_and_zero_based_strict_minimum():
    settings = {'patience': 100, 'min_epochs': 100}
    assert not official_early_stop(100, 100, settings)
    assert official_early_stop(100, 101, settings)
    assert not official_early_stop(101, 101, settings)


def test_artificial_end_to_end_training_selection_shards_reload_and_CPU_replay(tmp_path, monkeypatch):
    external = os.environ.get('ACL_HGCN_UPSTREAM_PATH')
    if not external:
        pytest.skip('explicit pinned external checkout required for official integration')
    monkeypatch.setenv('ACL_HGCN_VALIDATION_PID', str(os.getppid()))
    monkeypatch.setenv('ACL_HGCN_VALIDATION_DEADLINE', str(time.monotonic() + 180))
    previous_open = Path.open
    allowed = {'input_manifest.json', 'observed_graph.json', 'train_queries.json', 'features.npz',
               'entity_split.json', 'evaluator_valid.json'}
    def guarded(path, mode='r', *args, **kwargs):
        if path.parent.name == 'inputs':
            assert path.name in allowed, 'new drivers opened forbidden evaluator input'
        return previous_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    torch.set_num_threads(2)
    config = json.loads((Path(__file__).parents[1] / 'configs/mature_hgcn_validation.json').read_bytes())
    from acl_hct.hgcn_validation import fixture
    from acl_hct.hgcn_replay import POLICY_SHA256
    policy = json.loads((Path(__file__).parents[1] / 'configs/mature_hgcn_replay_policy.json').read_bytes())
    output = tmp_path / 'fixture'; output.mkdir()
    result = fixture(load_upstream(external), torch.device('cpu'), output, config,
                     {'source_commit': '0' * 40, 'replay_policy_sha256': POLICY_SHA256}, policy)
    assert result['status'] == 'passed' and result['numeric_qualification_fixture']['policy_sha256'] == POLICY_SHA256
    integration = result['integration_fixture']
    assert integration['complete_samples_replayed'] == 96 and integration['primary_comparisons_replayed'] == 12
    analysis = json.loads((output / 'artificial-integration/analysis/run.json').read_bytes())['result']
    assert len(analysis['primary_family']) == 12
    assert analysis['official_example']['epochs'] == 3

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from acl_hct.backbone import LorentzMeanNetwork, aggregate_plan, make_plan
from acl_hct.backbone_frozen_diagnosis import (CONFIG_SHA256, condition_input_hash,
    execution_source_hashes, load_registration, observe_condition, sampling_seed,
    two_layer_plans, verify_approval, weight_hash, run_matrix, verify_checkpoint_metadata)
from acl_hct.backbone_frozen_entry import supervise, failure_inventory
from acl_hct.diagnostic_archive import read_archive, write_archive
from acl_hct.protocols import digest, mask_indexed_queries
from acl_hct.train import TrainConfig


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module', autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tiny(dtype=torch.float64):
    torch.manual_seed(311)
    model = LorentzMeanNetwork(5, 4, head_hidden=6).to(dtype).eval()
    features = torch.randn(22, 5, dtype=dtype) * .2
    neighbors = [[] for _ in range(22)]
    neighbors[0] = list(range(1, 19))
    for i in range(1, 19):
        neighbors[i] = [0]
    neighbors[19], neighbors[20] = [20], [19]
    groups = torch.tensor([[[0, 1], [2, 1], [3, 1], [4, 1], [21, 1]],
                           [[19, 20], [1, 20], [2, 20], [3, 20], [21, 20]]])
    labels = torch.tensor([1, 0, 0, 0, 0] * 2, dtype=dtype)
    return model, features, neighbors, groups.reshape(-1, 2), labels


def test_registration_and_source_binding(tmp_path):
    config = load_registration(ROOT / 'configs/e2_backbone_frozen_diagnosis.json')
    assert digest(config) == CONFIG_SHA256
    assert config['total_scientific_forward_backward'] == 72
    assert config['total_equivalence_forward_backward'] == 8
    assert 'evaluator_valid.json' not in config['prepared']['input_raw_sha256']
    hashes = execution_source_hashes(config)
    assert len(hashes) >= 13
    changed = dict(config, replicates=9)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='registered'):
        load_registration(path)
    changed = dict(config, training_source_lf_sha256=dict(config['training_source_lf_sha256'], **{'backbone.py': '0' * 64}))
    with pytest.raises(ValueError, match='frozen execution'):
        execution_source_hashes(changed)


def test_frozen_checkpoint_metadata_tuple_list_and_rejection():
    config = load_registration(ROOT / 'configs/e2_backbone_frozen_diagnosis.json')
    anchor = config['checkpoints']['11']
    training = TrainConfig(**dict(anchor['training_config'], fanouts=(16, 16)))
    checkpoint = {'completed_steps': 768,
                  'selection_status': 'selected by complete filtered all-candidate validation query-micro MRR',
                  'manifest_hash': config['prepared']['manifest_hash'],
                  'valid_hash': config['prepared']['valid_queries_hash'],
                  'source_sha256_normalized_lf': {'acl_hct/' + k: v for k, v in config['training_source_lf_sha256'].items()}}
    verify_checkpoint_metadata(checkpoint, training, anchor, config)
    for name, value in [('completed_steps', 1024), ('selection_status', 'last'), ('manifest_hash', '0' * 64)]:
        with pytest.raises(ValueError, match='identity mismatched'):
            verify_checkpoint_metadata(dict(checkpoint, **{name: value}), training, anchor, config)
    state = tiny()[0].state_dict()
    del state['layers.0.linear.bias']
    with pytest.raises(RuntimeError, match='Missing key'):
        tiny()[0].load_state_dict(state, strict=True)


def test_seed_and_sequential_sampling():
    _, _, neighbors, _, _ = tiny()
    seed = sampling_seed(11, 769, 0)
    expected = int.from_bytes(hashlib.sha256(b'E2-BACKBONE-FROZEN-v1|11|769|0').digest()[:8], 'big') % 2**63
    assert seed == expected
    generator = torch.Generator().manual_seed(seed)
    direct = [make_plan(neighbors, 16, generator) for _ in range(2)]
    assert two_layer_plans(neighbors, 16, seed) == direct
    assert direct[0][0] != direct[1][0]
    assert all(plan[i] == neighbors[i] for plan in direct for i in range(1, 22))
    generator = torch.Generator().manual_seed(seed)
    before = generator.get_state().clone()
    assert make_plan(neighbors, None, generator) == neighbors
    assert torch.equal(before, generator.get_state())
    with pytest.raises(ValueError):
        sampling_seed(11, 769, 8)


@pytest.mark.parametrize('dtype,atol,rtol', [(torch.float64, 1e-12, 1e-11), (torch.float32, 2e-6, 2e-5)])
@pytest.mark.parametrize('masked', [False, True])
@pytest.mark.parametrize('fanout', [None, 16])
def test_passive_observation_matches_plain_and_preserves_inputs(dtype, atol, rtol, masked, fanout, tmp_path):
    model, features, neighbors, queries, labels = tiny(dtype)
    if masked:
        neighbors = mask_indexed_queries(neighbors, queries.reshape(-1, 5, 2)[:, 0].tolist())
        assert neighbors[19] == neighbors[20] == []
    plans = two_layer_plans(neighbors, fanout, sampling_seed(11, 769, 0))
    before = weight_hash(model), condition_input_hash(features, neighbors, plans, queries, labels)
    # Existing gradients must not accumulate into either backward.
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    first = observe_condition(model, features, neighbors, plans, queries, labels,
                               equivalence=True, atol=atol, rtol=rtol, max_padded_messages=128)
    second = observe_condition(model, features, neighbors, plans, queries, labels,
                                equivalence=False, max_padded_messages=128)
    assert first['equivalence']['status'] == 'passed'
    assert first['equivalence']['parameter_grad_max_abs'] == 0
    assert first['scientific_forward_backward_count'] == 1
    assert first['equivalence_forward_backward_count'] == 1
    np.testing.assert_array_equal(first['logits'], second['logits'])
    assert first['parameter_gradients'] == second['parameter_gradients']
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not layer.linear._forward_hooks for layer in model.layers)
    assert before == (weight_hash(model), condition_input_hash(features, neighbors, plans, queries, labels))
    descriptor = write_archive(tmp_path, 'readings', first)
    restored = read_archive(tmp_path, descriptor)
    np.testing.assert_array_equal(restored['bce'], first['bce'])


def test_empty_row_fallback_is_self_message():
    model, features, neighbors, queries, _ = tiny()
    masked = mask_indexed_queries(neighbors, queries.reshape(-1, 5, 2)[:, 0].tolist())
    messages = model.layers[0].messages(features)
    output, _ = aggregate_plan(messages, masked, make_plan(masked, None, torch.Generator()))
    torch.testing.assert_close(output[19:22], messages[19:22], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(output[2], messages[0], atol=1e-12, rtol=1e-12)


def test_nonfinite_stops_and_removes_hooks():
    model, features, neighbors, queries, labels = tiny()
    features[0, 0] = float('nan')
    with pytest.raises(ValueError):
        observe_condition(model, features, neighbors, [neighbors] * 2, queries, labels)
    assert all(not layer.linear._forward_hooks for layer in model.layers)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_approval_gate_binds_quality_source_and_user(tmp_path):
    config = load_registration(ROOT / 'configs/e2_backbone_frozen_diagnosis.json')
    sources = execution_source_hashes(config)
    identity = {'source_commit': 'a' * 40}
    quality = {'status': 'passed', 'config_sha256': CONFIG_SHA256, 'source_lf_sha256': sources,
               'real_data_executed': False}
    quality_path = tmp_path / 'quality.json'
    quality_path.write_text(json.dumps(quality))
    approval = {'user_authorized': True, 'scope': config['protocol'], 'config_sha256': CONFIG_SHA256,
                'source_commit': identity['source_commit'], 'source_lf_sha256': sources,
                'quality_review_accepted': True, 'entry_criteria_frozen': True,
                'quality_record_sha256': hashlib.sha256(quality_path.read_bytes()).hexdigest(),
                'user_message_reference': 'synthetic gate fixture only'}
    approval_path = tmp_path / 'approval.json'
    approval_path.write_text(json.dumps(approval))
    verify_approval(approval_path, quality_path, config, identity, sources)
    for field in ('user_authorized', 'quality_review_accepted', 'entry_criteria_frozen'):
        changed = dict(approval, **{field: False})
        approval_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=field):
            verify_approval(approval_path, quality_path, config, identity, sources)


def test_supervisor_deadline_keeps_partial_evidence_and_nonzero(tmp_path):
    output = tmp_path / 'timed'
    script = 'from pathlib import Path; import time; Path("' + output.as_posix() + '/partial.json").write_text("{}\\n"); time.sleep(10)'
    assert supervise([sys.executable, '-c', script], output, seconds=.5,
                     metadata={'seed': 11, 'protocol': 'E2-BACKBONE-FROZEN-v1'}) == 124
    assert (output / 'partial.json').exists()
    report = json.loads((output / 'supervisor.json').read_text())
    assert report['status'] == 'timeout'
    assert report['kill_scope'] == 'direct diagnostic worker only'
    assert report['elapsed_seconds'] < 3
    failure = json.loads((output / 'supervisor-failure.json').read_text())
    assert len(failure['missing_registered_cells']) == 36
    assert failure['completed_archived_cells'] == []
    with pytest.raises(ValueError, match='empty'):
        supervise([sys.executable, '-c', 'pass'], output, seconds=1)
    failed = tmp_path / 'failed'
    assert supervise([sys.executable, '-c', 'raise ValueError("fixture")'], failed, seconds=2) == 2
    assert json.loads((failed / 'supervisor.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('training_seed', [11, 23])
def test_complete_synthetic_matrix_seeds_counts_and_private_archives(training_seed, tmp_path):
    model, features, neighbors, queries, labels = tiny(torch.float64)
    data = {'manifest': {'fixture': True}, 'manifest_hash': 'synthetic', 'valid_hash': 'identity-only',
            'nodes': list(range(22)), 'neighbors': neighbors, 'features': features,
            'query_groups': queries.reshape(-1, 5, 2), 'labels': labels.reshape(-1, 5), 'input_raw_sha256': {}}
    config = load_registration(ROOT / 'configs/e2_backbone_frozen_diagnosis.json')
    report = {'fixture': True, 'real_data_executed': False}
    run_matrix(model, features, data, {769: torch.tensor([0, 1]), 770: torch.tensor([1, 0])},
               config, training_seed, tmp_path, report, deadline=lambda: None)
    assert report['scientific_forward_backward_count'] == 36
    assert report['equivalence_forward_backward_count'] == 4
    assert report['pass_execution']['scientific'] == {'attempted': 36, 'forward_returned': 36, 'backward_returned': 36}
    assert report['whole_job_before'] == report['whole_job_after']
    for batch in report['batches']:
        assert len(batch['cells']) == 18
        assert [(c['identity']['graph'], c['identity']['fanout'], c['identity']['replicate']) for c in batch['cells'][:4]] == [
            ('unmasked', 'full', None), ('masked', 'full', None), ('unmasked', 'f16', 0), ('masked', 'f16', 0)]
        equivalents = [c for c in batch['cells'] if c['equivalence']['performed']]
        assert len(equivalents) == (4 if batch['step'] == 769 else 0)
        for cell in batch['cells']:
            identity = cell['identity']
            replicate = identity['replicate']
            expected = 0 if replicate is None else sampling_seed(training_seed, batch['step'], replicate)
            assert identity['sampling_seed'] == expected
            raw = read_archive(tmp_path / 'private', cell['archive'])
            graph = neighbors if identity['graph'] == 'unmasked' else mask_indexed_queries(neighbors, queries.reshape(-1, 5, 2)[:, 0].tolist())
            assert raw['plans'] == two_layer_plans(graph, None if replicate is None else 16, expected)
            assert 'logits' not in cell and 'raw' not in cell['layers'][0]
        read_archive(tmp_path / 'private', batch['comparison_archive'])


def test_partial_matrix_keeps_incremental_cells(tmp_path):
    model, features, neighbors, queries, labels = tiny()
    data = {'manifest': {'fixture': True}, 'manifest_hash': 'synthetic', 'valid_hash': 'identity-only',
            'nodes': list(range(22)), 'neighbors': neighbors, 'features': features,
            'query_groups': queries.reshape(-1, 5, 2), 'labels': labels.reshape(-1, 5), 'input_raw_sha256': {}}
    config = load_registration(ROOT / 'configs/e2_backbone_frozen_diagnosis.json')
    calls = 0
    def deadline():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise TimeoutError('synthetic interruption')
    with pytest.raises(TimeoutError):
        run_matrix(model, features, data, {769: torch.tensor([0, 1]), 770: torch.tensor([0, 1])},
                   config, 11, tmp_path, {'fixture': True}, deadline=deadline)
    report = json.loads((tmp_path / 'summary.json').read_text())
    assert report['status'] == 'incomplete'
    assert len(report['batches'][0]['cells']) == 2
    for cell in report['batches'][0]['cells']:
        read_archive(tmp_path / 'private', cell['archive'])
    failure = failure_inventory(tmp_path, {'seed': 11, 'protocol': 'E2-BACKBONE-FROZEN-v1'})
    assert len(failure['missing_registered_cells']) == 34
    assert len(failure['completed_archived_cells']) == 2
    assert failure['active_phase']['phase'] == 'plan_setup'

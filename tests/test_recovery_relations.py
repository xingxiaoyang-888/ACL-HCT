"""Analytic scalar/geometry checks; no real archives, checkpoint or GPU."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from acl_hct.recovery_relations import (RelationLocalization, quantile_groups, child_reciprocal_ranks,
                                       validated_headers, bound_archive)


def line_points(radii):
    r = torch.tensor(radii, dtype=torch.float64)
    return torch.stack((r.cosh(), r.sinh()), -1)


def design():
    nodes = ['r', 'a', 'b', 'z', 'unknown']
    view = SimpleNamespace(nodes=nodes, root='r', reachable=set(nodes[:-1]))
    panel = {'rows': [
        {'id': 'b', 'stratum': '1-2', 'pool_mean_weight': .3, 'h_dev_shortest': 2},
        {'id': 'z', 'stratum': '3-4', 'pool_mean_weight': .2, 'h_dev_shortest': 9},
        {'id': 'unknown', 'stratum': '0', 'pool_mean_weight': .5, 'h_dev_shortest': None}],
        'relations': [{'child': 'b', 'direct_parents': ['a', 'r'], 'positive_distant_ancestors': ['r']},
                      {'child': 'z', 'direct_parents': ['a'], 'positive_distant_ancestors': ['r']},
                      {'child': 'unknown', 'direct_parents': ['r'], 'positive_distant_ancestors': ['r']}]}
    base = line_points([0, .2, .4, .6, .8])
    floor = torch.full((5,), .001, dtype=torch.float64)
    return view, panel, base, torch.zeros_like(base), floor


def test_analytic_flip_decomposition_and_original_child_weights():
    analyzer = RelationLocalization(*design())
    s = np.array([.1, .2, .4, .6, .8])
    o = np.array([.15, .5, .45, .35, .8])
    result = analyzer.compare_radials(s, o)['direct']
    changes = result['changes']
    np.testing.assert_allclose(changes['gap_change'], changes['child_radial_change'] - changes['parent_radial_change'], atol=1e-15)
    np.testing.assert_allclose(changes['parent_radial_change'], [.3, .05, .3], atol=1e-15)
    np.testing.assert_allclose(changes['child_radial_change'], [.05, .05, -.25], atol=1e-15)
    metrics = result['groups']['all']['metrics']
    assert metrics['sampled_score'] == 1
    assert metrics['moved_score'] == pytest.approx(.3)
    assert metrics['score_change'] == pytest.approx(-.7)
    assert metrics['correct_to_error'] == pytest.approx(.7)
    assert result['groups']['all']['weighted_relation_mass_in_pool'] == pytest.approx(.5)
    assert result['coverage']['unknown_relations'] == 1
    assert result['coverage']['no_covered_relation_children'] == 1
    assert result['children'][-1]['score_change'] is None
    for prefix in ('transition_', 'F_to_S_', 'F_to_X_'):
        assert sum(v for k, v in metrics.items() if k.startswith(prefix)) == pytest.approx(1)


def test_fixed_anchor_analytic_native_geometry_and_accepted_primary_reproduction():
    analyzer = RelationLocalization(*design())
    native = line_points([.1, .2, .4, .6, .8]).float()
    radial = analyzer.radial(native)
    np.testing.assert_allclose(radial, [.1, .2, .4, .6, .8], atol=1e-7)
    # A candidate root is .1 from the FIXED reference root: it is not reset to0.
    assert radial[0] == pytest.approx(.1, abs=1e-7)
    output = analyzer.compare_radials(radial, radial)
    from acl_hct.frozen_recovery import promoted
    old = analyzer.panel.evaluate(promoted(native, 1.))
    for kind in output:
        assert output[kind]['coverage'] == {k: v for k, v in old[kind]['groups']['V'].items()
                                           if k != 'weighted_covered_child_metrics'}
        assert output[kind]['groups']['all']['metrics']['sampled_score'] == pytest.approx(
            old[kind]['groups']['V']['weighted_covered_child_metrics']['score'], abs=1e-12)
        assert output[kind]['groups']['all']['metrics']['score_change'] == 0


def test_quartile_ties_reference_groups_and_unknown_depth_retained():
    groups, cuts = quantile_groups(np.array([1., 1., 1., 1., 4., 4., 4., 4.]), np.ones(8, bool), 'fixed')
    assert len(cuts) == len(set(cuts))
    assert sum(mask.astype(int) for mask in groups.values()).tolist() == [1] * 8
    for mask in groups.values():
        assert len(set(mask[:4])) == len(set(mask[4:])) == 1
    analyzer = RelationLocalization(*design())
    mask = np.array([True, False, False])
    analyzer.add_reference_groups({'direct': {'custom_reference': mask}})
    mask[:] = False
    first = analyzer.compare_radials(np.array([0, .2, .4, .6, .8]), np.array([0, .2, .4, .6, .8]))
    assert first['direct']['groups']['custom_reference']['covered_relations'] == 1
    assert first['direct']['groups']['depth/unknown']['metrics']['score_change'] is None
    assert first['direct']['groups']['bias_zero']['covered_relations'] == 3
    with pytest.raises(ValueError, match='freeze'):
        analyzer.add_reference_groups({'direct': {'late': np.ones(3, bool)}})


def test_retrieval_divergence_alignment_and_missing_not_zero():
    analyzer = RelationLocalization(*design())
    result = analyzer.compare_radials(np.array([.1, .2, .4, .6, .8]), np.array([.15, .5, .45, .35, .8]),
                                     sampled_child_mrr={2: .1, 3: .1, 4: .1},
                                     moved_child_mrr={2: .2, 3: .2, 4: .2})['direct']
    assert result['joint']['fractions']['structure_loss_retrieval_gain'] == pytest.approx(1)
    assert result['joint']['paired_covered_children'] == 2
    assert result['joint']['unpaired_sampled_children'] == 1
    assert result['children'][-1]['mrr_change'] == pytest.approx(.1)
    assert result['children'][-1]['score_change'] is None
    missing = analyzer.compare_radials(np.array([.1, .2, .4, .6, .8]), np.array([.15, .5, .45, .35, .8]),
                                      sampled_child_mrr={2: .1}, moved_child_mrr={2: .2})['direct']
    assert missing['children'][1]['mrr_change'] is None
    assert missing['coverage'] == result['coverage']


@pytest.mark.parametrize('case', ['wrong_shape', 'nonfinite', 'non_tangent', 'missing_floor'])
def test_invalid_fixed_input_rejected(case):
    view, panel, base, bias, floor = design()
    if case == 'wrong_shape':
        bias = bias[:-1]
    elif case == 'nonfinite':
        bias[1, 1] = float('nan')
    elif case == 'non_tangent':
        bias[1, 0] = 1
    else:
        floor = None
    with pytest.raises(ValueError):
        RelationLocalization(view, panel, base, bias, floor)


def test_node_order_native_precision_and_candidate_invalid_values_rejected():
    analyzer = RelationLocalization(*design())
    with pytest.raises(ValueError, match='node order'):
        analyzer.analyze(line_points([0, .2, .4, .6, .8]).float(), line_points([0, .2, .4, .6, .8]).float(),
                         {}, {}, {'nodes': list(reversed(analyzer.nodes))})
    with pytest.raises(ValueError, match='FP32'):
        analyzer.radial(line_points([0, .2, .4, .6, .8]))
    with pytest.raises(ValueError, match='finite'):
        analyzer.compare_radials(np.zeros(5), np.array([0, 0, float('nan'), 0, 0]))


def test_full_rank_coverage_reciprocal_rank_child_alignment():
    data = {'nodes': ['r', 'a', 'b', 'z', 'unknown'], 'valid': [(0, 2), (1, 2), (1, 3)]}
    rows = [{'parent': 1, 'child': 3, 'rank': 4., 'candidates': 4},
            {'parent': 1, 'child': 2, 'rank': 2., 'candidates': 3},
            {'parent': 0, 'child': 2, 'rank': 1., 'candidates': 3}]
    ranking = {'status': 'complete', 'metric_scope': 'filtered_all_entity_candidates', 'expected_queries': 3,
               'completed_queries': 3, 'completed_children': 2, 'rows': rows, 'query_micro_mrr': 1.75 / 3,
               'child_macro_mrr': .5, 'hits': {'1': 1 / 3, '3': 2 / 3, '10': 1.}}
    assert child_reciprocal_ranks(ranking, data) == {3: .25, 2: .75}
    wrong = copy.deepcopy(ranking)
    wrong['rows'][0]['child'] = 4
    with pytest.raises(ValueError, match='coverage'):
        child_reciprocal_ranks(wrong, data)


def headers_fixture(tmp_path):
    from acl_hct.recovery_registration import PROTOCOL, CONFIG_SHA256, source_hashes
    bindings = {'shards': [], 'immutable_file_inventory': {}, 'old_science_source_commit': '2' * 40}
    roots = []
    for seed in (11, 23):
        for start in (0, 8):
            root = tmp_path / f'{seed}-{start}'
            root.mkdir()
            row = {'seed': seed, 'status': 'complete', 'protocol': PROTOCOL, 'config_sha256': CONFIG_SHA256,
                   'engineering_fixture_only': False, 'completed_repetitions': 8, 'weights_unchanged': True,
                   'input_unchanged': True, 'model_updates': 0, 'provenance': {'release_verified': True,
                    'source_commit': '2' * 40, 'release': {'phase': 'science'}}, 'identity': {'q': f'fixed-{seed}'},
                   'repeat_range': [start, start + 8], 'observations': [{'repeat': i} for i in range(start, start + 8)],
                   'repeat_archives': [{} for i in range(8)]}
            supervisor = {'status': 'complete', 'worker_exit_code': 0, 'deadline_seconds': 1080}
            progress = {'protocol': PROTOCOL, 'phase': 'science', 'config_sha256': CONFIG_SHA256,
                        'source_commit': '2' * 40, 'source_lf_sha256': source_hashes(), 'release': {'phase': 'science'}}
            spec = {'seed': seed, 'run_raw_sha256': '', 'remote_output': root.as_posix(), 'frozen_identity': row['identity'],
                    'repeat_range': row['repeat_range'], 'global_repeat_ids': list(range(start, start + 8)),
                    'archives': [{} for i in range(9)]}
            for name, value in (('run.json', row), ('supervisor.json', supervisor), ('progress.json', progress)):
                path = root / name
                path.write_text(json.dumps(value), encoding='utf-8')
                h = hashlib.sha256(path.read_bytes()).hexdigest()
                bindings['immutable_file_inventory'][root.as_posix() + '/' + name] = {'raw_sha256': h}
                if name == 'run.json':
                    spec['run_raw_sha256'] = h
            roots.append(root)
            bindings['shards'].append(spec)
    return roots, bindings


@pytest.mark.parametrize('case', ['positive', 'duplicate', 'identity_drift', 'incomplete', 'raw_drift'])
def test_four_shard_header_identity_completeness_and_raw_gates(tmp_path, case):
    roots, bindings = headers_fixture(tmp_path)
    if case == 'positive':
        assert len(validated_headers(roots, bindings, {})) == 4
        return
    row = json.loads((roots[1] / 'run.json').read_bytes())
    spec = bindings['shards'][1]
    if case == 'duplicate':
        row['repeat_range'] = spec['repeat_range'] = [0, 8]
        row['observations'] = [{'repeat': i} for i in range(8)]
        spec['global_repeat_ids'] = list(range(8))
    elif case == 'identity_drift':
        row['identity']['q'] = 'changed-Q'
        spec['frozen_identity'] = copy.deepcopy(row['identity'])
    elif case == 'incomplete':
        spec['archives'].pop()
    else:
        row['status'] = 'failed'
    (roots[1] / 'run.json').write_text(json.dumps(row), encoding='utf-8')
    if case != 'raw_drift':
        spec['run_raw_sha256'] = hashlib.sha256((roots[1] / 'run.json').read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        validated_headers(roots, bindings, {})


def test_archive_descriptor_drift_rejected_before_npz_load(tmp_path):
    descriptor = {'manifest': 'entry.json', 'manifest_sha256': '1' * 64, 'array_file': 'entry.npz',
                  'array_file_sha256': '2' * 64, 'array_bytes': 20, 'arrays': 1}
    expected = {'manifest_path': '/bound/entry.json', 'manifest_raw_sha256': '1' * 64,
                'NPZ_path': '/bound/entry.npz', 'NPZ_raw_sha256': '3' * 64, 'NPZ_bytes': 20,
                'arrays_count': 1}
    with pytest.raises(ValueError, match='descriptor'):
        bound_archive(tmp_path, descriptor, expected, time.monotonic() + 10)


def test_bound_archive_positive_full_array_validation_and_deadline(tmp_path):
    from acl_hct.diagnostic_archive import write_archive
    descriptor = write_archive(tmp_path, 'reference', {'vector': np.array([1., 2., 3.])})
    record = json.loads((tmp_path / descriptor['manifest']).read_bytes())
    expected = {'manifest_path': '/fixed/' + descriptor['manifest'], 'manifest_raw_sha256': descriptor['manifest_sha256'],
                'NPZ_path': '/fixed/' + descriptor['array_file'], 'NPZ_raw_sha256': descriptor['array_file_sha256'],
                'NPZ_bytes': descriptor['array_bytes'], 'arrays_count': descriptor['arrays'], 'array_metadata': record['arrays']}
    payload = bound_archive(tmp_path, descriptor, expected, time.monotonic() + 10)
    np.testing.assert_array_equal(payload['vector'], [1., 2., 3.])
    with pytest.raises(TimeoutError):
        bound_archive(tmp_path, descriptor, expected, time.monotonic() - 1)


def test_fixed_design_signature_detects_reference_group_changes():
    first = RelationLocalization(*design())
    second = RelationLocalization(*design())
    assert first.frozen_signature() == second.frozen_signature()
    second.add_reference_groups({'direct': {'extra': np.ones(3, bool)}})
    assert first.frozen_signature() != second.frozen_signature()


@pytest.mark.parametrize('field', ['source_lf_sha256', 'source_commit', 'config_sha256', 'release'])
def test_real_progress_schema_source_config_release_drift_rejected(tmp_path, field):
    roots, bindings = headers_fixture(tmp_path)
    row = json.loads((roots[0] / 'run.json').read_bytes())
    assert 'source_lf_sha256' not in row['provenance']
    assert len(validated_headers(roots, bindings, {})) == 4
    progress = json.loads((roots[0] / 'progress.json').read_bytes())
    progress[field] = {} if field in ('source_lf_sha256', 'release') else 'drift'
    (roots[0] / 'progress.json').write_text(json.dumps(progress), encoding='utf-8')
    spec = bindings['shards'][0]
    bindings['immutable_file_inventory'][spec['remote_output'] + '/progress.json']['raw_sha256'] = hashlib.sha256(
        (roots[0] / 'progress.json').read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='original scientific'):
        validated_headers(roots, bindings, {})


def test_stdlib_supervisor_imports_no_numeric_modules_and_static_cpu_budget():
    from acl_hct import relation_entry
    env_code = "import acl_hct.relation_entry,sys; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
    task_environment = dict(os.environ)
    task_environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
    subprocess.run([sys.executable, '-c', env_code], check=True, env=task_environment)
    output = subprocess.check_output([sys.executable, '-m', 'acl_hct.relation_entry'], text=True, env=task_environment)
    status = json.loads(output)
    assert status['GPU_requested'] == 0 and status['worker_seconds'] == 1680
    with pytest.raises(ValueError, match='explicit'):
        relation_entry.verify_release(relation_entry.parser().parse_args(['--execute']))


def release_fixture(tmp_path):
    from acl_hct import relation_entry as entry
    root = Path(__file__).resolve().parents[1]
    protocol = tmp_path / 'protocol.md'
    protocol.write_text('synthetic fixed A protocol', encoding='utf-8')
    quality = tmp_path / 'quality.json'
    bindings = tmp_path / 'bindings.json'
    approval = tmp_path / 'approval.json'
    config = root / 'configs/e2_frozen_recovery_pilot.json'
    sources = {name: entry.lf_hash(root / 'src/acl_hct' / name) for name in entry.MODULES}
    q = {'status': 'passed', 'phase': 'A', 'new_source_lf_sha256': sources,
         'protocol_lf_sha256': entry.lf_hash(protocol)}
    quality.write_text(json.dumps(q), encoding='utf-8')
    b = {'old_config_raw_sha256': entry.raw_hash(config),
         'old_config_canonical_sha256': '8ecd462845cf6f413bf57186506cabf044dcd87a52b4b6d490744db4def27172',
         'old_science_source_commit': '293ca0beaa50dd9bedacf6f06001ec3b73b2b9b6',
         'no_new_graph_samples': True, 'A_runtime_prepared_graph_files': [], 'supervisor_protocol_LF_sha256': 'f' * 64}
    bindings.write_text(json.dumps(b), encoding='utf-8')
    a = {'status': 'approved_A_CPU', 'protocol': entry.PROTOCOL, 'source_commit': '1' * 40, 'GPU_requested': 0,
         'new_source_lf_sha256': sources, 'quality_raw_sha256': entry.raw_hash(quality),
         'bindings_raw_sha256': entry.raw_hash(bindings), 'protocol_lf_sha256': entry.lf_hash(protocol)}
    approval.write_text(json.dumps(a), encoding='utf-8')
    args = entry.parser().parse_args(['--execute', '--config', str(config), '--bindings', str(bindings),
                                     '--protocol', str(protocol), '--approval', str(approval), '--quality', str(quality),
                                     '--source-commit', '1' * 40, '--output', str(tmp_path / 'output'),
                                     '--shards', 'a', 'b', 'c', 'd'])
    return entry, args


@pytest.mark.parametrize('case', ['positive', 'protocol_drift', 'quality_drift', 'bindings_drift', 'source_label_drift'])
def test_exact_source_quality_input_and_current_protocol_release(tmp_path, case):
    entry, args = release_fixture(tmp_path)
    if case == 'positive':
        record = entry.verify_release(args)
        assert record['release_verified'] is True
        assert record['input_bindings_protocol_lf_sha256'] != record['protocol_lf_sha256']
        return
    if case == 'source_label_drift':
        args.source_commit = '2' * 40
    else:
        target = {'protocol_drift': args.protocol, 'quality_drift': args.quality, 'bindings_drift': args.bindings}[case]
        with target.open('ab') as stream:
            stream.write(b'\n ')
    with pytest.raises(ValueError):
        entry.verify_release(args)

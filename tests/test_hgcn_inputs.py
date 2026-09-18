import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import pytest

from acl_hct.hgcn_quality import load_train, load_valid, wordnet_quality
from acl_hct.hgcn_registration import canonical
from acl_hct.hgcn_upstream import load_upstream
from acl_hct.taxonomy import Taxonomy


@pytest.fixture
def prepared(tmp_path):
    nodes = [f'n{i:02d}' for i in range(30)]
    records = {n: {'name': f'word{i}', 'definition': f'entity category{i % 3} common'} for i, n in enumerate(nodes)}
    taxonomy = Taxonomy(records, {(nodes[(i-1)//2], nodes[i]) for i in range(1, len(nodes))}, {})
    spec = importlib.util.spec_from_file_location('hgcn_test_prepare', Path(__file__).parents[1] / 'scripts/prepare_benchmark.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    root = tmp_path / 'prepared'; module.prepare(taxonomy, root, dimension=4)
    def read(name):
        return json.loads((root / name).read_bytes())
    manifest = read('input_manifest.json'); graph = read('observed_graph.json'); queries = read('train_queries.json')
    valid = read('evaluator_valid.json')
    config = json.loads((Path(__file__).parents[1] / 'configs/mature_hgcn_quality.json').read_bytes())
    config['model'].update(input_dim=4, hidden=6, head_hidden=5)
    config['wordnet_quality'].update(batch_positives=2, quality_steps=2)
    config['prepared'] = {'manifest_hash': canonical(manifest), 'node_order_hash': canonical(graph['nodes']),
                          'graph_hash': canonical(graph['neighbors']), 'train_queries_hash': canonical(queries),
                          'valid_queries_hash': canonical(valid), 'nodes_count': len(nodes),
                          'train_groups_count': len(queries['labels']) // 5,
                          'input_raw_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                               for name in ('input_manifest.json', 'observed_graph.json', 'train_queries.json')},
                          'features_npz_raw_sha256': hashlib.sha256((root / 'features.npz').read_bytes()).hexdigest(),
                          'evaluator_valid_raw_sha256': hashlib.sha256((root / 'evaluator_valid.json').read_bytes()).hexdigest()}
    config['valid_query_count'] = len(valid)
    return root, config


def test_train_and_evaluator_reads_are_separate_allowlists(prepared, monkeypatch):
    root, config = prepared; prior = Path.read_bytes; opened = []
    def checked(path):
        if path.parent == root:
            assert path.name in ('input_manifest.json', 'observed_graph.json', 'train_queries.json', 'features.npz')
            opened.append(path.name)
        return prior(path)
    monkeypatch.setattr(Path, 'read_bytes', checked)
    data = load_train(root, config)
    assert set(opened) == {'input_manifest.json', 'observed_graph.json', 'train_queries.json', 'features.npz'}
    assert set(data['input_files_opened']) == set(opened)
    monkeypatch.setattr(Path, 'read_bytes', prior)
    valid, truth = load_valid(root, data, config)
    assert len(valid) == config['valid_query_count'] and all(a in truth[b] for a, b in valid)


def test_heldout_loader_rejects_training_child_even_with_matching_hashes(prepared):
    root, config = prepared; data = load_train(root, config)
    a, b = data['query_groups'][0, 0].tolist(); bad = [[data['nodes'][a], data['nodes'][b]]]
    p = root / 'evaluator_valid.json'; p.write_text(json.dumps(bad))
    config['prepared'].update(valid_queries_hash=canonical(bad), evaluator_valid_raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    config['valid_query_count'] = 1
    with pytest.raises(ValueError, match='heldout'):
        load_valid(root, data, config)


def test_train_feature_bytes_and_registered_graph_cannot_drift(prepared):
    root, config = prepared
    with (root / 'features.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='raw hash'):
        load_train(root, config)


def test_cpu_quality_uses_no_heldout_labels_or_selection(prepared, tmp_path, monkeypatch):
    external = os.environ.get('ACL_HGCN_UPSTREAM_PATH')
    if not external:
        pytest.skip('explicit pinned external checkout needed')
    root, config = prepared; data = load_train(root, config); upstream = load_upstream(external)
    monkeypatch.setenv('ACL_HGCN_SUPERVISOR_PID', str(os.getppid()))
    monkeypatch.setenv('ACL_HGCN_DEADLINE', str(time.monotonic() + 60))
    prior = Path.read_bytes
    def checked(path):
        assert path.name not in ('evaluator_valid.json', 'evaluator_test.json', 'evaluator_truth.json')
        return prior(path)
    monkeypatch.setattr(Path, 'read_bytes', checked)
    output = tmp_path / 'quality'; output.mkdir()
    result = wordnet_quality(data, config, upstream, __import__('torch').device('cpu'), output, root)
    assert result['status'] == 'passed' and len(result['steps']) == 2
    assert result['complete_valid_ranking'] is None
    assert not (output / 'best.pt').exists() and set(result['sample_forward_cost']) == {'4', '8', '16'}

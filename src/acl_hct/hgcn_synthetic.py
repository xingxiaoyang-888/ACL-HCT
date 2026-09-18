"""Tiny artificial protocol-B inputs; random features, never scientific data."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .development_view import load_development_view, make_panels
from .hgcn_evidence import file_hash
from .hgcn_panels import HierarchyPanel
from .hgcn_registration import canonical
from .protocols import grouped_split, observed_neighbors, training_candidates


def miniature_inputs(root, registered):
    root = Path(root); root.mkdir(parents=True)
    nodes = [f'artificial-node-{i:02d}' for i in range(40)]
    edges = [(nodes[0], n) for n in nodes[1:]]
    split = grouped_split(nodes, edges, 20260914)
    graph = {'nodes': nodes, 'neighbors': observed_neighbors(nodes, split['relations']['train'])}
    queries = training_candidates(nodes, split['entities']['train'], split['relations']['train'])
    features = np.random.default_rng(777).normal(0, .1, (40, 4)).astype(np.float32)
    manifest = {'protocol': split['protocol'], 'split_seed': 20260914,
                'text_fit_entities': split['entities']['train'], 'node_order_hash': canonical(nodes),
                'graph_hash': canonical(graph['neighbors']), 'train_queries_hash': canonical(queries),
                'feature_manifest': {'schema': 'ARTIFICIAL-random-features-QA-only', 'effective_dimension': 4,
                                     'nodes': nodes, 'train_entities': split['entities']['train'],
                                     'feature_sha256': hashlib.sha256(features.tobytes()).hexdigest()}}
    valid = split['relations']['valid']
    for name, payload in {'input_manifest.json': manifest, 'observed_graph.json': graph, 'train_queries.json': queries,
                          'entity_split.json': split['entities'], 'evaluator_valid.json': valid}.items():
        (root / name).write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8', newline='\n')
    np.savez_compressed(root / 'features.npz', features=features)
    config = copy.deepcopy(registered); config['model'].update(input_dim=4, hidden=6, head_hidden=5)
    config['training'].update(steps=4, batch_positives=2, evaluate_every=2, complete_valid_selection_steps=[2, 4])
    config['official_task']['epochs'] = 3
    config['prepared'] = {'manifest_hash': canonical(manifest), 'node_order_hash': canonical(nodes),
                          'graph_hash': canonical(graph['neighbors']), 'train_queries_hash': canonical(queries),
                          'valid_queries_hash': canonical(valid), 'nodes_count': len(nodes),
                          'train_groups_count': len(queries['labels']) // 5,
                          'input_raw_sha256': {name: file_hash(root / name) for name in
                                               ('input_manifest.json', 'observed_graph.json', 'train_queries.json')},
                          'features_npz_raw_sha256': file_hash(root / 'features.npz'),
                          'evaluator_valid_raw_sha256': file_hash(root / 'evaluator_valid.json')}
    config['valid_query_count'] = len(valid)
    view = load_development_view(root); panels = make_panels(view, target=2); panel = panels['panels']['development']
    fake_points = np.zeros((len(nodes), 6), dtype=np.float32)
    fixed = HierarchyPanel(view, panel, fake_points).evaluate(fake_points)
    r = {'combined_hash': panels['hash'], 'reference_root': view.root, 'reference_root_index': view.nodes.index(view.root),
         'h_dev_hash': view.metadata['h_dev_hash'], 'panel_name': 'development', 'panel_hash': panel['panel_hash'],
         'relation_hash': panel['relation_hash'], 'target': 2, 'reachable_including_root': len(view.reachable),
         'entity_split_hash': view.metadata['entity_split_hash'], 'entity_split_raw_sha256': file_hash(root / 'entity_split.json'),
         'selected_root_reachable_children': fixed['bias']['root_known_nodes'],
         'coverage': {kind: {**{k: fixed[kind][k] for k in ('covered_pairs', 'unknown_pairs', 'covered_children')},
                             'weighted_covered_child_mass_in_pool': fixed[kind]['weighted_covered_child_mass']}
                      for kind in ('direct', 'distant')}}
    config['hierarchy']['registration'] = r
    return config

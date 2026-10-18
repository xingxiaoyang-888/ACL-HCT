"""Reproduce one original 500-epoch plain baseline, never correction training."""
import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from public_io import PACKAGE, binding, configure_cuda, file_ref, local_boundary
from public_io import native_root, read, sha, verify_sources, write_new


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', required=True, help='public_training_inputs_v1 JSON')
    p.add_argument('--output', required=True, help='Fresh output directory')
    a = p.parse_args()
    verify_sources()
    source = Path(a.inputs).resolve()
    m = read(source)
    if set(m) != {'schema', 'dataset', 'model', 'seed', 'files', 'native_root'} or m['schema'] != 'public_training_inputs_v1':
        raise ValueError('Wrong training input schema')
    if type(m['seed']) is not int or m['seed'] not in (11, 23, 37):
        raise ValueError('Use the registered three seeds')
    if set(m['files']) != {'graph', 'ids', 'preparation', 'train', 'fit'}:
        raise ValueError('Only graph/ID/preparation/train/fit inputs')
    files = {k: binding(v, source.parent) for k, v in m['files'].items()}
    native = native_root(m['model'], m['native_root'], source.parent)
    out = Path(a.output).resolve()
    if out.exists() or out.is_relative_to(PACKAGE / 'src'):
        raise ValueError('Fresh output outside source tree required')
    out.parent.mkdir(parents=True, exist_ok=True)
    control = out.parent / (out.name + '-input-contract')
    control.mkdir(exist_ok=False)
    os.environ.setdefault('DGLBACKEND', 'pytorch')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('OMP_NUM_THREADS', '7')
    sys.dont_write_bytecode = True
    configure_cuda()
    if m['dataset'] == 'ogbn-arxiv' and m['model'] in ('hgcn', 'hybonet', 'gat'):
        sys.path.insert(0, str(PACKAGE / 'src/training/arxiv_base'))
        from training_runner import run_plain_pretrain
        run_plain_pretrain(m['model'], m['seed'], files, native, out)
        weights = out / 'label-free-weights.pt'
    elif m['model'] == 'sage' and m['dataset'] in ('ogbn-arxiv', 'PubMed'):
        folder = PACKAGE / ('src/training/arxiv_sage' if m['dataset'] == 'ogbn-arxiv' else 'src/training/pubmed_sage')
        sys.path.insert(0, str(folder))
        names = ['blocks.py', 'common.py', 'runtime.py', 'sage.py', 'train_sage.py', 'prepare_data.py'] if m['dataset'] == 'ogbn-arxiv' else ['blocks.py', 'common.py', 'sage.py', 'train_pubmed.py', 'prepare_pubmed.py']
        tree = {'schema': 'icml_ca_source_manifest_v1' if m['dataset'] == 'ogbn-arxiv' else 'vldb_pubmed_source_manifest_v1',
                'files': [{'path': n, 'sha256': sha(folder / n)} for n in names]}
        tree_path = control / 'source-contract.json'
        write_new(tree_path, tree)
        if m['dataset'] == 'ogbn-arxiv':
            roles = {'graph': 'graph', 'ids': 'ids', 'preparation': 'preparation', 'train_labels': 'train', 'fit_labels': 'fit'}
            contract = {'schema': 'icml_ca_release_v1', 'stage': 'train_sage', 'seed': m['seed'], 'authorized': True,
                        'inputs': {k: files[v]['sha256'] for k, v in roles.items()},
                        'source_manifest': str(tree_path), 'source_manifest_sha256': sha(tree_path)}
            # The unmodified trainer takes two directories; do not silently join unrelated role paths.
            public = Path(files['graph']['path']).parent
            development = Path(files['train']['path']).parent
            for role, folder_, filename in [('graph', public, 'graph.npz'), ('ids', public, 'ids.json'),
                    ('preparation', public, 'preparation.json'), ('train', development, 'train.npz'), ('fit', development, 'fit.npz')]:
                if Path(files[role]['path']) != folder_ / filename:
                    raise ValueError('SAGE inputs must use the documented public/development filenames')
            contract_path = control / 'input-contract.json'
            write_new(contract_path, contract)
            from train_sage import run
            run(SimpleNamespace(public=str(public), development=str(development), output=str(out),
                                release=str(contract_path), seed=m['seed'], device='cuda:0'))
        else:
            contract = {'schema': 'vldb_pubmed_release_v1', 'round': 'vldb-sampling-mechanisms-v1', 'authorized': True,
                        'stage': 'train_pubmed_sage', 'seed': m['seed'], 'inputs': files,
                        'source_manifest': file_ref(tree_path), 'config': file_ref(PACKAGE / 'config/pubmed-sage.json'), 'output': str(out)}
            contract_path = control / 'input-contract.json'
            write_new(contract_path, contract)
            from train_pubmed import run_job
            run_job(str(contract_path), sha(contract_path), local_boundary)
        weights = out / 'label-free-weights.pt'
    else:
        raise ValueError('Only the original 12 arxiv / 3 PubMed plain training cells')
    import torch
    payload = torch.load(weights, map_location='cpu', weights_only=True)
    metadata = {k: v for k, v in payload.items() if k != 'model_state'}
    write_new(out / 'checkpoint-metadata.json', {'sha256': sha(weights), 'metadata': metadata,
        'regenerated_not_original_paper_bytes': True})


if __name__ == '__main__':
    main()


"""Prepare user-acquired official inputs; never downloads or scores a model."""
import argparse
import gzip
from pathlib import Path
import sys
import zipfile
from public_io import PACKAGE, file_ref, read, sha, verify_sources, write_new


def check(path, expected):
    if sha(path) != expected:
        raise ValueError('Official input identity differs: ' + str(path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('arxiv')
    a.add_argument('--raw-root', required=True, help='Extracted arxiv directory with raw/ and split/')
    a.add_argument('--output', required=True)
    a = sub.add_parser('pubmed')
    a.add_argument('--raw-root', required=True, help='Directory containing pinned ind.pubmed.* files')
    a.add_argument('--output', required=True)
    a = sub.add_parser('test-labels')
    a.add_argument('--dataset', choices=('ogbn-arxiv', 'PubMed'), required=True)
    a.add_argument('--raw', required=True, help='Official arxiv.zip OR pinned PubMed data directory')
    a.add_argument('--public-ids', required=True)
    a.add_argument('--output', required=True)
    a.add_argument('--confirm-frozen-evaluation', action='store_true')
    args = p.parse_args()
    verify_sources()
    provenance = read(PACKAGE / 'config/data-provenance.json')
    out = Path(args.output).resolve()
    if out.exists() or out.is_relative_to(PACKAGE / 'src'):
        raise ValueError('Fresh output outside the source tree required')
    sys.dont_write_bytecode = True
    if args.command == 'arxiv':
        raw = Path(args.raw_root).resolve()
        for name, digest in provenance['arxiv']['raw_member_hashes'].items():
            check(raw / name, digest)
        sys.path.insert(0, str(PACKAGE / 'src/training/arxiv_sage'))
        from prepare_data import prepare
        prepare(raw, out / 'public', out / 'development', out / 'historical-confirmation')
    elif args.command == 'pubmed':
        raw = Path(args.raw_root).resolve()
        files = {x['name']: x for x in provenance['pubmed']['raw_files']}
        roles = ['x', 'y', 'tx', 'allx', 'ally', 'graph', 'test.index']
        inputs = {}
        for role in roles:
            name = 'ind.pubmed.' + role
            check(raw / name, files[name]['sha256'])
            inputs[role] = file_ref(raw / name)
        folder = PACKAGE / 'src/training/pubmed_sage'
        control = out.parent / (out.name + '-input-contract')
        control.mkdir(parents=True, exist_ok=False)
        tree = {'schema': 'vldb_pubmed_source_manifest_v1', 'files': [
            {'path': n, 'sha256': sha(folder / n)} for n in ['blocks.py', 'common.py', 'sage.py', 'prepare_pubmed.py', 'train_pubmed.py']]}
        write_new(control / 'source-contract.json', tree)
        contract = {'schema': 'vldb_pubmed_release_v1', 'round': 'vldb-sampling-mechanisms-v1', 'authorized': True,
                    'stage': 'prepare_pubmed', 'seed': None, 'inputs': inputs,
                    'source_manifest': file_ref(control / 'source-contract.json'),
                    'config': file_ref(PACKAGE / 'config/pubmed-sage.json'), 'output': str(out)}
        write_new(control / 'input-contract.json', contract)
        sys.path.insert(0, str(folder))
        from prepare_pubmed import run_job
        run_job(str(control / 'input-contract.json'), sha(control / 'input-contract.json'), lambda: None)
    else:
        if not args.confirm_frozen_evaluation:
            p.error('Fix methods, weights, budgets and statistical plan before opening test labels')
        sys.path.insert(0, str(PACKAGE / 'src/data'))
        from extract_final_labels import target_manifest, parse_arxiv_labels, parse_pubmed_labels, public_ids_check
        meta = read(Path(args.public_ids).resolve())
        raw = Path(args.raw).resolve()
        if args.dataset == 'ogbn-arxiv':
            check(raw, provenance['arxiv']['archive_sha256'])
            with zipfile.ZipFile(raw) as z:
                ids = [int(x) for x in gzip.decompress(z.read('arxiv/split/time/test.csv.gz')).splitlines()]
                ys = parse_arxiv_labels(z.read('arxiv/raw/node-label.csv.gz'), ids)
            origin = {'official_archive_sha256': provenance['arxiv']['archive_sha256']}
        else:
            files = {x['name']: x for x in provenance['pubmed']['raw_files']}
            for name in ['ind.pubmed.test.index', 'ind.pubmed.ty']:
                check(raw / name, files[name]['sha256'])
            ids = [int(x) for x in (raw / 'ind.pubmed.test.index').read_text('ascii').splitlines()]
            ys = parse_pubmed_labels((raw / 'ind.pubmed.ty').read_bytes(), ids)
            origin = {'source_commit': provenance['pubmed']['source_commit'], 'prior_dataset_test_exposure': True}
        public_ids_check(meta, args.dataset, ids)
        target = target_manifest(args.dataset, ids, origin)
        out.mkdir(parents=True, exist_ok=False)
        write_new(out / 'targets.json', target)
        import numpy as np
        with (out / 'labels.npz').open('xb') as f:
            np.savez(f, ids=np.asarray(ids, dtype=np.int64), y=np.asarray(ys, dtype=np.int64))
        write_new(out / 'input-hashes.json', {k: sha(out / k) for k in ['targets.json', 'labels.npz']})


if __name__ == '__main__':
    main()

"""Raw OGB preparation. NEVER constructs an OGB dataset or all-node y."""
import argparse
import gzip
from pathlib import Path
import numpy as np
from common import splits, batches, sha, write_json, fresh_output


def read_ids(path):
    with gzip.open(path, 'rt', encoding='ascii') as f:
        ids = [int(row.strip()) for row in f]
    if len(ids) != len(set(ids)) or any(n < 0 for n in ids):
        raise ValueError('Invalid official split IDs')
    return ids


def stream_allowed_labels(path, allowed, n, parser=int):
    # Iteration/decompression touches compressed bytes for every row. Prohibited
    # values are never stripped, decoded, tokenized, parsed, stored, or scored.
    allowed = set(map(int, allowed))
    labels, count = {}, 0
    with gzip.open(path, 'rb') as f:
        for i, raw_line in enumerate(f):
            count += 1
            if i in allowed:
                value = parser(raw_line)
                if not 0 <= value < 40:
                    raise ValueError('Allowed label out of official range')
                labels[i] = value
    if count != n or set(labels) != allowed:
        raise ValueError('Raw label row/split count differs')
    return labels


def symmetric_csr(edges, n):
    edges = np.asarray(edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.min() < 0 or edges.max() >= n:
        raise ValueError('Raw edges malformed')
    packed = np.unique(np.concatenate((edges[:, 1]*n + edges[:, 0], edges[:, 0]*n + edges[:, 1])))
    rows, cols = packed//n, packed%n
    indptr = np.concatenate(([0], np.cumsum(np.bincount(rows, minlength=n))))
    return indptr, cols


def prepare(raw, public, development, sealed):
    raw = Path(raw)
    train, valid = read_ids(raw/'split/time/train.csv.gz'), read_ids(raw/'split/time/valid.csv.gz')
    if set(train) & set(valid):
        raise ValueError('Official train/valid overlap')
    x = np.loadtxt(raw/'raw/node-feat.csv.gz', delimiter=',', dtype=np.float32)
    if x.shape != (169343, 128) or not np.isfinite(x).all():
        raise ValueError('Official ogbn-arxiv v1 feature shape/values differ')
    if max(train+valid) >= len(x) or (len(train), len(valid)) != (90941, 29799):
        raise ValueError('Official arxiv temporal split dimensions differ')
    split = splits(valid)
    edges = np.loadtxt(raw/'raw/edge.csv.gz', delimiter=',', dtype=np.int64)
    indptr, indices = symmetric_csr(edges, len(x))
    labels = stream_allowed_labels(raw/'raw/node-label.csv.gz', train+valid, len(x))
    # Create all destinations only after validation; no overwrite of prior runs.
    public, development, sealed = map(fresh_output, (public, development, sealed))
    np.savez(public/'graph.npz', x=x, indptr=indptr, indices=indices, values=np.ones(len(indices), dtype=np.float32),
             diagonal=np.zeros(len(x), dtype=np.float32))
    ids = {'train': train, **split}
    manifest = {'schema': 'icml_ca_ids_v1', 'dataset': 'ogbn-arxiv', 'classes': 40, 'ids': ids,
                'batches': {s: batches(v) for s, v in split.items()}, 'official_test_ids_read': False}
    write_json(public/'ids.json', manifest)
    for name in ('train', 'fit', 'gate', 'confirmation'):
        dest = sealed if name == 'confirmation' else development
        # Labels are aligned to the fixed ID list. Runtime gathers only IDs
        # authorized for its stage, with no all-node label array.
        np.savez(dest/(name+'.npz'), ids=np.asarray(ids[name], dtype=np.int64), y=np.array([labels[n] for n in ids[name]], dtype=np.int64))
    raw_files = ['raw/node-feat.csv.gz', 'raw/edge.csv.gz', 'raw/node-label.csv.gz', 'split/time/train.csv.gz', 'split/time/valid.csv.gz']
    receipt = {'schema': 'icml_ca_preparation_v1', 'dataset_version': 'ogbn-arxiv v1',
               'raw': {name: sha(raw/name) for name in raw_files},
               'public': {p.name: sha(p) for p in public.iterdir()},
               'development': {p.name: sha(p) for p in development.iterdir()},
               'sealed_sha256': sha(sealed/'confirmation.npz'),
               'parsed_label_rows': len(labels), 'skipped_label_rows': len(x)-len(labels),
               'semantic_test_label_reads': 0, 'compressed_bytes_touched': 'entire node-label gzip stream',
               'confirmation_label_path_exposed_to_development': False}
    write_json(public/'preparation.json', receipt)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw-root', required=True)
    p.add_argument('--public-output', required=True)
    p.add_argument('--development-output', required=True)
    p.add_argument('--confirmation-output', required=True, help='Ops-controlled sealed destination, outside public/development')
    a = p.parse_args()
    destinations = [Path(x).resolve() for x in (a.public_output, a.development_output, a.confirmation_output)]
    if any(x == y or x in y.parents or y in x.parents for i, x in enumerate(destinations) for y in destinations[i+1:]):
        p.error('Public, development and sealed stores must be separate sibling directories')
    print(prepare(a.raw_root, *destinations)['schema'])

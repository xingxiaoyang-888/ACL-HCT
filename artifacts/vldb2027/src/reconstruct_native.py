"""Reconstruct exact native bytes from a user-provided pinned upstream checkout.

No repository, data or model is downloaded; no upstream code is executed.
Only our short changes are distributed. Upstream licensing remains separate.
"""
import argparse
from pathlib import Path
from public_io import PACKAGE, read
import hashlib


def digest(data):
    return hashlib.sha256(data).hexdigest()


def reconstruct(model, upstream, output):
    group = next(g for g in read(PACKAGE / 'config/native-sources.json')['groups'] if g['model'] == model)
    upstream, output = Path(upstream).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(upstream) or upstream.is_relative_to(output):
        raise ValueError('Use a new output separate from the upstream checkout')
    products = []
    for entry in group['files']:
        source = (upstream / group['source_prefix'] / entry['path']).resolve()
        if not source.is_relative_to(upstream):
            raise ValueError('Source escaped the supplied checkout')
        raw = source.read_bytes()
        lf = raw.decode('utf-8-sig').replace('\r\n', '\n').encode('utf-8')
        if digest(lf) != entry['upstream_lf_sha256']:
            raise ValueError('Wrong pinned source: ' + entry['path'])
        result = lf.replace(b'\n', b'\r\n')
        if model == 'hybonet' and entry['path'] == 'models/base_models.py':
            marker = b'from os import POSIX_FADV_NOREUSE\r\n'
            if result.count(marker) != 1:
                raise ValueError('Unique compatibility import anchor missing')
            result = result.replace(marker, b'')
        elif model == 'hybonet' and entry['path'] == 'layers/hyp_layers.py':
            marker = b'        output = support_t / denom\r\n        return output\r\n'
            addition = (b'        output = support_t / denom\n'
                        b"        observer = getattr(self, '_stage1_observer', None)\n"
                        b'        if observer is not None:\n'
                        b'            observer(self, x, adj, support_t, denom, output)\n'
                        b'        return output\n')
            if result.count(marker) != 1:
                raise ValueError('Unique observer anchor missing')
            result = result.replace(marker, addition)
        elif model == 'hybonet' and entry['path'] == 'optim/__init__.py':
            marker = b'from .radam import RiemannianAdam'
            if result.splitlines().count(marker) != 1:
                raise ValueError('Unique optimizer export anchor missing')
            result = marker + b'\n'
        if digest(result) != entry['accepted_sha256']:
            raise ValueError('Reconstruction does not match accepted bytes: ' + entry['path'])
        products.append((entry['path'], result))
    output.mkdir(parents=True, exist_ok=False)
    for name, content in products:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    print('Reconstructed', len(products), 'source files; upstream code was not executed.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=('hgcn', 'hybonet'), required=True)
    p.add_argument('--upstream', required=True, help='Pinned repository root (not its gcn subdirectory)')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    reconstruct(a.model, a.upstream, a.output)


if __name__ == '__main__':
    main()

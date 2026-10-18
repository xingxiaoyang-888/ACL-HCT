"""Download resumable release parts and reconstruct the original result ZIPs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def identity(path):
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()

def matches(path, spec):
    return path.is_file() and identity(path) == (spec['bytes'], spec['sha256'])

def safe_name(name):
    if not name or '/' in name or '\\' in name or name in {'.', '..'}:
        raise ValueError('Archive and part names must be plain filenames')
    return name

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'results/raw-downloads.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parts-dir', type=Path, help='Use existing local parts; do not access the network')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    part_root = args.parts_dir.resolve() if args.parts_dir else output/'.parts'
    part_root.mkdir(parents=True, exist_ok=True)
    for archive in manifest['archives']:
        name = safe_name(archive['name'])
        destination = output/name
        if destination.exists():
            if not matches(destination, archive):
                raise FileExistsError('Existing ZIP differs; choose another output directory: '+str(destination))
            print('Verified existing '+name, flush=True)
            continue
        parts = []
        for spec in archive['parts']:
            part_name = safe_name(spec['name'])
            local = part_root/part_name
            if not matches(local, spec):
                if args.parts_dir:
                    raise ValueError('Missing or corrupt local part: '+part_name)
                url = manifest['release_base_url'].rstrip('/')+'/'+urllib.parse.quote(part_name)
                request = urllib.request.Request(url, headers={'User-Agent':'ACL-HCT-artifact-downloader'})
                with tempfile.NamedTemporaryFile(prefix='.download-', dir=part_root, delete=False) as temporary:
                    temp_path = Path(temporary.name)
                    with urllib.request.urlopen(request, timeout=300) as response:
                        for block in iter(lambda: response.read(1024*1024), b''):
                            temporary.write(block)
                if not matches(temp_path, spec):
                    raise ValueError('Downloaded part failed SHA256; retained temporary file: '+str(temp_path))
                os.replace(temp_path, local)
            print('Verified '+part_name, flush=True)
            parts.append(local)
        with tempfile.NamedTemporaryFile(prefix='.assemble-', dir=output, delete=False) as temporary:
            temp_path = Path(temporary.name)
            for part in parts:
                with part.open('rb') as source:
                    for block in iter(lambda: source.read(1024*1024), b''):
                        temporary.write(block)
        if not matches(temp_path, archive):
            raise ValueError('Reconstructed ZIP failed original SHA256; retained '+str(temp_path))
        os.replace(temp_path, destination)
        print('Reconstructed and verified '+name, flush=True)

if __name__ == '__main__':
    main()

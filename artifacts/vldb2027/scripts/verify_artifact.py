"""Verify public artifact bytes using Python's standard library."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=ROOT/'MANIFEST.json')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    errors = []
    for row in manifest['files']:
        path = (ROOT/row['path']).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            errors.append(row['path'] + ': missing or invalid path')
            continue
        raw = path.read_bytes()
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            errors.append(row['path'] + ': size or SHA256 mismatch')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f"Verified {len(manifest['files'])} artifact files (size and SHA256).")

if __name__ == '__main__':
    main()

"""Verify public source/config byte identities without importing model code."""
import argparse
from public_io import verify_sources


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    manifest = verify_sources()
    print('Verified', len(manifest['files']), 'published source/config/document identities.')


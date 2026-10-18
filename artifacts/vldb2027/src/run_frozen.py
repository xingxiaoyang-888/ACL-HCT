"""Public CLI for one frozen model cell; explicitly supplied data/CP required."""
import argparse
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', required=True, help='Portable input manifest; paths relative to this JSON')
    p.add_argument('--output', required=True, help='New directory; never overwritten')
    p.add_argument('--stage', choices=('calibrate', 'final'), required=True)
    p.add_argument('--checkpoint-mode', choices=('paper', 'regenerated'), default='paper')
    p.add_argument('--confirm-frozen-evaluation', action='store_true', help='Required to open labels for final evaluation')
    a = p.parse_args()
    if a.stage == 'final' and not a.confirm_frozen_evaluation:
        p.error('Freeze checkpoint, budgets and input manifest before final scoring; pass --confirm-frozen-evaluation')
    os.environ.setdefault('DGLBACKEND', 'pytorch')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('OMP_NUM_THREADS', '7')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'frozen'))
    from runner import run_portable
    run_portable(a)


if __name__ == '__main__':
    main()


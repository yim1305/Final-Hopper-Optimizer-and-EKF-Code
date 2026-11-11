"""Launch navigation from the repository root without installing either project."""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import sys

sys.stdout.reconfigure(encoding='utf-8',errors='replace')
sys.stderr.reconfigure(encoding='utf-8',errors='replace')

GN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(GN), str(GN.parent / 'Astro')]

from gn_sim.scripts.run_nav import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=None)
    main(parser.parse_args().runs)

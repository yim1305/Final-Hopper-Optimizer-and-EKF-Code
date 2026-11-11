"""Navigation simulation; keep generated plotting caches inside this repository."""
from pathlib import Path
import os

os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[2]/'.maintenance/gn_runtime/matplotlib'))

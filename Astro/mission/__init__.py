"""Active DEM mission workflow; hopper is imported read-only."""

from pathlib import Path
import os

# Keep plotting's generated cache inside this repository as well.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / ".maintenance/astro_runtime/matplotlib"),
)

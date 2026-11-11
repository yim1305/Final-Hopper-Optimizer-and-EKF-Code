"""Report simple launch-base LOS using the accepted main DEM trajectory."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mission.los import main

if __name__ == "__main__":
    main()

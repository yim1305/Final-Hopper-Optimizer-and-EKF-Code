"""Native DEM interpolation and the unchanged terrain constraints of the accepted flight."""

from dataclasses import replace
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/runtime"))
import casadi as ca
import rasterio
from scipy.ndimage import map_coordinates
from hopper.problem import Problem
from hopper.environment.terrain import altitude

TERRAIN_FOLDER = ROOT / "data"


class DemTerrain:
    """Same native DEM bilinear interpolation in NumPy and CasADi, no smoothing."""

    def __init__(self, radius=1737400.0, fraction=1.0):
        self.radius, self.fraction = radius, float(fraction)
        with rasterio.open(TERRAIN_FOLDER / "terrain_corridor_30m.tif") as src:
            self.dem = src.read(1).astype(float)
            self.transform = src.transform
        self.inverse = ~self.transform
        rows, cols = self.dem.shape
        self.xgrid = self.transform.c + (np.arange(cols) + 0.5) * self.transform.a
        self.ygrid = (self.transform.f + (np.arange(rows) + 0.5) * self.transform.e)[::-1]
        if not np.isfinite(self.dem).all():
            raise ValueError("DEM includes missing pixels")
        # CasADi uses column-major data with x varying fastest.
        self.lut = ca.interpolant(
            "dem_native",
            "linear",
            [self.xgrid.tolist(), self.ygrid.tolist()],
            self.dem[::-1, :].ravel().tolist(),
        )
        self._maps = {}

    def projected(self, r):
        x, y, z = r
        n = (x * x + y * y + z * z) ** 0.5
        factor = 2 * self.radius / (n - z)
        return y * factor, x * factor

    def height(self, r_abs):
        x, y = self.projected(r_abs)
        if any(isinstance(v, (ca.SX, ca.MX, ca.DM)) for v in r_abs):
            n = int(x.numel())
            if n not in self._maps:
                self._maps[n] = self.lut.map(n)
            return self.fraction * self._maps[n](
                ca.vertcat(ca.reshape(x, 1, n), ca.reshape(y, 1, n))
            )
        col, row = self.inverse * (x, y)
        shape = np.shape(x)
        coords = np.array([np.asarray(row).ravel() - 0.5, np.asarray(col).ravel() - 0.5])
        values = map_coordinates(
            self.dem, coords, order=1, mode="constant", cval=np.nan, prefilter=False
        )
        if not np.isfinite(values).all():
            raise ValueError("Terrain query leaves retained DEM")
        out = self.fraction * values.reshape(shape)
        return float(out) if out.ndim == 0 else out

    def height_positions(self, r):
        a = np.asarray(r)
        return self.height(tuple(np.moveaxis(a, -1, 0)))

    def altitude(self, r):
        a = np.asarray(r)
        return np.linalg.norm(a, axis=-1) - self.radius - self.height_positions(a)


class TerrainProblem(Problem):
    """Original flight constraints plus DEM clearance at nodes/midpoints and camera rays."""

    def _build(self):
        super()._build()
        p, s, opti = self.p, self.s, self.opti
        self.original_constraint_count = int(opti.ng)
        for ph in self.phases:
            X, U, n = ph.X, ph.U, ph.n
            F = self.dynamics_of(ph).map(n + 1)(X, U)
            Xm = (X[:, :-1] + X[:, 1:]) / 2 + (ph.tdur / n / 8) * (F[:, :-1] - F[:, 1:])
            samples = ca.horzcat(X, Xm)
            absolute = ca.DM(p.mission.r_start) + samples[:3, :] * s.L
            xy = self.terrain.projected(tuple(absolute[i, :] for i in range(3)))
            for coord, grid in zip(xy, (self.terrain.xgrid, self.terrain.ygrid)):
                # Coordinates use km solely to scale numerical domain constraints.
                opti.subject_to(opti.bounded(grid[0] / 1000, coord / 1000, grid[-1] / 1000))
            if ph.kind in ("rise", "terminal"):
                start = 1 if ph.kind == "rise" else 0
                end = n if ph.kind == "terminal" else n + 1
                for positions in (X[:3, start:end], Xm[:3, :]):
                    h = altitude(p.mission.r_start, p.moon.radius, positions * s.L, self.terrain)
                    opti.subject_to(h / s.L >= 0)
            elif ph.kind != "observation":
                h = altitude(p.mission.r_start, p.moon.radius, Xm[:3, :] * s.L, self.terrain)
                opti.subject_to(h / s.L >= p.limits.min_altitude / s.L)
            if ph.kind == "observation":
                # Add terrain obstruction checks along the camera-to-ground rays.
                # These are collocation samples, verified with finer rays afterwards.
                for point in ph.sensor_sites:
                    target = ca.DM(p.mission.r_start) + point * s.L
                    for fraction in np.linspace(0.1, 0.9, 9):
                        ray = (1 - fraction) * target + fraction * absolute
                        h = (
                            ca.sqrt(ca.sum1(ray * ray))
                            - p.moon.radius
                            - self.terrain.height(tuple(ray[i, :] for i in range(3)))
                        )
                        opti.subject_to(h / s.L >= 0)
        self.terrain_constraint_count = int(opti.ng) - self.original_constraint_count


class GroundProblem(TerrainProblem):
    def __init__(self, p, branch="uprange", *, tw_max=None, **kwargs):
        """`tw_max` (default: mission.yaml descent_variant, off when null) adds thrust / local weight <= tw_max at
        the touchdown node of each hop."""
        from .config import descent_variant

        settings = descent_variant()
        self.tw_max = tw_max if tw_max is not None else settings["touchdown_thrust_to_weight_max"]
        margin = settings["solve_pointing_rate_deg_s"]
        if margin is not None:
            # Solver-side margin only: the independent camera check keeps the sensor limit of the caller's p.
            cap = min(p.sensor.pointing_rate_max, np.radians(margin))
            p = replace(p, sensor=replace(p.sensor, pointing_rate_max=cap))
        super().__init__(p, branch, **kwargs)

    def _build(self):
        super()._build()
        if self.tw_max is not None:
            p, s = self.p, self.s
            start = ca.DM(p.mission.r_start)
            for ph in (self.hop1[-1], self.hop2[-1]):
                r_abs = start + ph.X[0:3, -1] * s.L
                weight = ph.X[6, -1] * s.M * p.moon.mu / ca.dot(r_abs, r_abs)
                self.opti.subject_to(ph.U[3, -1] * s.F / weight <= self.tw_max)
        # Densify the existing terrain camera-ray screen without changing its FOV.
        p, s = self.p, self.s
        for ph in self.phases:
            if ph.kind != "observation":
                continue
            F = self.dynamics_of(ph).map(ph.n + 1)(ph.X, ph.U)
            mid = (ph.X[:, :-1] + ph.X[:, 1:]) / 2 + ph.tdur / ph.n / 8 * (F[:, :-1] - F[:, 1:])
            absolute = ca.DM(p.mission.r_start) + ca.horzcat(ph.X[:3], mid[:3]) * s.L
            for point in ph.sensor_sites:
                target = ca.DM(p.mission.r_start) + point * s.L
                for fraction in np.linspace(0, 1, 20 + 1)[1:-1]:
                    if abs(fraction * 10 - round(fraction * 10)) < 1e-8:
                        continue
                    ray = (1 - fraction) * target + fraction * absolute
                    height = (
                        ca.sqrt(ca.sum1(ray * ray))
                        - p.moon.radius
                        - self.terrain.height(tuple(ray[i, :] for i in range(3)))
                    )
                    self.opti.subject_to(height / s.L >= 0)

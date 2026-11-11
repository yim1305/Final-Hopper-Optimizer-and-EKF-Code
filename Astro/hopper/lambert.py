"""Universal-variable Lambert solver (short way, single revolution) and the impulsive hop baseline."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .params import Params

_Z_HI = 4.0 * math.pi ** 2 - 1e-3   # single-revolution upper limit (psi = 4 pi^2 is singular)


def _stumpff(z):
    if z > 1e-6:
        s = math.sqrt(z)
        return (1 - math.cos(s)) / z, (s - math.sin(s)) / s ** 3
    if z < -1e-6:
        s = math.sqrt(-z)
        return (math.cosh(s) - 1) / (-z), (math.sinh(s) - s) / s ** 3
    return 0.5 - z / 24 + z * z / 720, 1 / 6 - z / 120 + z * z / 5040


class _Geom:
    def __init__(self, r1, r2):
        self.r1, self.r2 = np.linalg.norm(r1), np.linalg.norm(r2)
        dnu = math.atan2(np.linalg.norm(np.cross(r1, r2)), float(r1 @ r2))   # in [0, pi]
        # A = sin(dnu) sqrt(r1 r2 / (1 - cos dnu)), written without 1 - cos for small angles
        self.A = math.sqrt(2.0 * self.r1 * self.r2) * math.cos(dnu / 2.0)

    def y(self, z):
        C, S = _stumpff(z)
        return self.r1 + self.r2 + self.A * (z * S - 1.0) / math.sqrt(C)

    def tof(self, z, mu):
        C, S = _stumpff(z)
        y = self.y(z)
        return (S * (y / C) ** 1.5 + self.A * math.sqrt(y)) / math.sqrt(mu)


def _z_lower(g: _Geom) -> float:
    """Smallest z with y(z) > 0 (y is increasing in z)."""
    if g.y(_Z_HI) <= 0:
        raise ValueError("no single-revolution Lambert solution")
    lo = -1.0
    while g.y(lo) > 0:
        lo *= 2.0
        if lo < -1e4:
            return lo
    return brentq(g.y, lo, _Z_HI, xtol=1e-14, rtol=1e-14)


def tof_range(mu, r1, r2):
    g = _Geom(r1, r2)
    zl = _z_lower(g)
    return g.tof(zl + 1e-9 * (1 + abs(zl)), mu), g.tof(_Z_HI, mu)


def lambert(mu: float, r1: np.ndarray, r2: np.ndarray, tof: float):
    """Inertial two-point boundary value problem; returns (v1, v2) for the short-way transfer."""
    g = _Geom(r1, r2)
    zl = _z_lower(g)
    z_lo = zl + 1e-9 * (1 + abs(zl))
    t_lo, t_hi = g.tof(z_lo, mu), g.tof(_Z_HI, mu)
    if not t_lo <= tof <= t_hi:
        raise ValueError(f"TOF {tof:.2f} s outside the single-rev range [{t_lo:.2f}, {t_hi:.2f}] s")
    z = brentq(lambda zz: g.tof(zz, mu) - tof, z_lo, _Z_HI, xtol=1e-14, rtol=1e-14, maxiter=500)
    y = g.y(z)
    f = 1.0 - y / g.r1
    gg = g.A * math.sqrt(y / mu)
    gdot = 1.0 - y / g.r2
    v1 = (r2 - f * r1) / gg
    v2 = (gdot * r2 - r1) / gg
    return v1, v2


def _rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def lambert_fixed(p: Params, r1_abs: np.ndarray, r2_abs: np.ndarray, tof: float):
    """Lambert between two Moon-fixed positions (at departure and arrival epochs).

    Solved in the inertial frame (frames aligned at departure): the arrival point has rotated by
    omega*tof. Returned velocities are expressed in the Moon-fixed frame at their own epochs.
    """
    w = p.moon.omega_vec
    Rz = _rot_z(p.moon.omega * tof)
    r2_in = Rz @ r2_abs
    v1_in, v2_in = lambert(p.moon.mu, r1_abs, r2_in, tof)
    v1_f = v1_in - np.cross(w, r1_abs)
    v2_f = Rz.T @ (v2_in - np.cross(w, r2_in))
    return v1_f, v2_f


@dataclass
class HopBaseline:
    dv: float
    tof: float
    dv_depart: float
    dv_arrive: float
    v1: np.ndarray        # fixed-frame departure velocity
    v2: np.ndarray        # fixed-frame arrival velocity


def hop_dv(p: Params, tof: float, r1=None, r2=None) -> float:
    """Impulsive delta-v of a hop; defaults to start site -> target."""
    r1 = p.mission.r_start if r1 is None else r1
    r2 = p.mission.r_target if r2 is None else r2
    v1, v2 = lambert_fixed(p, r1, r2, tof)
    return float(np.linalg.norm(v1) + np.linalg.norm(v2))


def min_dv_hop(p: Params, n_grid: int = 80, r1=None, r2=None) -> HopBaseline:
    """Impulsive surface-to-surface hop: sweep TOF for minimum total delta-v (the 'ideal' baseline).

    Defaults to start site -> target; pass Moon-fixed positions r1, r2 for any other hop (e.g. the return)."""
    r1 = p.mission.r_start if r1 is None else r1
    r2 = p.mission.r_target if r2 is None else r2
    t_lo, t_hi = tof_range(p.moon.mu, r1, r2)
    # the rotation moves the arrival point slightly; stay inside the range with a margin
    ts = np.geomspace(t_lo * 1.02, t_hi * 0.98, n_grid)
    dvs = []
    for t in ts:
        try:
            dvs.append(hop_dv(p, t, r1, r2))
        except ValueError:
            dvs.append(np.inf)
    i = int(np.argmin(dvs))
    a, b = ts[max(i - 1, 0)], ts[min(i + 1, n_grid - 1)]
    res = minimize_scalar(lambda t: hop_dv(p, t, r1, r2), bounds=(a, b), method="bounded",
                          options={"xatol": 1e-6})
    t_opt = float(res.x)
    v1, v2 = lambert_fixed(p, r1, r2, t_opt)
    return HopBaseline(float(np.linalg.norm(v1) + np.linalg.norm(v2)), t_opt,
                       float(np.linalg.norm(v1)), float(np.linalg.norm(v2)), v1, v2)

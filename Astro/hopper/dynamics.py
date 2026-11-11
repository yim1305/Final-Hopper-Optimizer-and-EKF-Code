"""3-DOF dynamics, written once and used by CasADi (collocation) and numpy/scipy (validation).

State x = [r(3), v(3), m]  (r is the offset from the start site, r_abs = r_start + r)
Control u = [T_vec(3), Gamma]
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from .environment.gravity import CentralGravity, Gravity
from .params import Params


def _cat(lst):
    return ca.vertcat(*lst)


def _rhs(x, u, p: Params, gravity: Gravity, stack, mode=None):
    """Dimensional state derivative. `stack` builds a vector from 7 scalars.

    `mode` is the burn stage's EngineMode: mass flow is mdot = c0 + c1 * Gamma. None means unpowered (coast), no
    propellant flows and the control is ignored for the mass."""
    rs = p.mission.r_start
    rx, ry, rz = rs[0] + x[0], rs[1] + x[1], rs[2] + x[2]
    vx, vy, vz, m = x[3], x[4], x[5], x[6]
    w = p.moon.omega
    gx, gy, gz = gravity.accel((rx, ry, rz))
    # -2 w x v - w x (w x r_abs), with w = [0, 0, w]
    cor_x, cor_y = 2.0 * w * vy, -2.0 * w * vx
    cen_x, cen_y = w * w * rx, w * w * ry
    ax = gx + u[0] / m + cor_x + cen_x
    ay = gy + u[1] / m + cor_y + cen_y
    az = gz + u[2] / m
    mdot = 0.0 if mode is None else -mode.mdot(u[3])
    return stack([vx, vy, vz, ax, ay, az, mdot])


def make_dynamics(p: Params, gravity: Gravity | None = None, mode=None) -> ca.Function:
    """CasADi f(x[7], u[4]) -> xdot[7], dimensional SI."""
    gravity = gravity or CentralGravity(p.moon.mu)
    x, u = ca.SX.sym("x", 7), ca.SX.sym("u", 4)
    return ca.Function("f_dim", [x, u], [_rhs(x, u, p, gravity, _cat, mode)])


def make_scaled_dynamics(p: Params, gravity: Gravity | None = None, mode=None) -> ca.Function:
    """Nondimensional f_nd(xn, un) = d(xn)/d(tn). Length L, time sqrt(L/g), mass m_wet, force m_wet*g."""
    gravity = gravity or CentralGravity(p.moon.mu)
    s = p.scales
    xn, un = ca.SX.sym("xn", 7), ca.SX.sym("un", 4)
    x = ca.vertcat(xn[0:3] * s.L, xn[3:6] * s.V, xn[6] * s.M)
    u = ca.vertcat(un[0:3] * s.F, un[3] * s.F)
    xd = _rhs(x, u, p, gravity, _cat, mode)
    xdn = ca.vertcat(xd[0:3] * s.T / s.L, xd[3:6] * s.T / s.V, xd[6] * s.T / s.M)
    return ca.Function("f_nd", [xn, un], [xdn])


def make_numpy_rhs(p: Params, gravity: Gravity | None = None, mode=None):
    """numpy f(x, u) -> xdot, dimensional SI (same equations as the CasADi version)."""
    gravity = gravity or CentralGravity(p.moon.mu)
    return lambda x, u: _rhs(x, u, p, gravity, lambda lst: np.array(lst, dtype=float), mode)


def env_accel(p: Params, r, v, gravity: Gravity | None = None):
    """Non-thrust acceleration at offset position r, velocity v (numpy)."""
    f = make_numpy_rhs(p, gravity)
    x = np.concatenate([r, v, [1.0]])
    return f(x, np.zeros(4))[3:6]


def jacobi_constant(p: Params, x: np.ndarray) -> float:
    """Rotating-frame energy, conserved on a coast: v^2/2 - mu/|r_abs| - |w x r_abs|^2/2."""
    r_abs = p.mission.r_start + x[0:3]
    v = x[3:6]
    wxr = np.cross(p.moon.omega_vec, r_abs)
    return 0.5 * v @ v - p.moon.mu / np.linalg.norm(r_abs) - 0.5 * wxr @ wxr

"""Cost of the divert descent: from a fixed gate state, solve the terminal descent to a landing site on its own."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp

from .constraints import GlideSlope, MassFloor, ThrustBounds, TiltRate, Touchdown, TouchdownUpright, LandingRegion
from .dynamics import make_numpy_rhs, make_scaled_dynamics
from .environment.terrain import FlatTerrain
from .initial_guess import _Builder
from .params import Params
from .phases import make_phase
from .postprocess import phase_dv, thrust_tilt_deg
from .transcription import HermiteSimpson


@dataclass
class BranchResult:
    branch: str
    success: bool
    status: str
    iterations: int
    phase: dict | None = None      # same layout as Solution.phases entries (dimensional, offset positions)
    dv: float = float("nan")       # finite-burn delta-v of the terminal descent
    propellant: float = float("nan")
    m_final: float = float("nan")
    touchdown_pos_err: float = float("nan")    # re-propagation vs optimized touchdown [m]
    touchdown_up_speed: float = float("nan")   # vertical speed at touchdown [m/s], negative = descending
    touchdown_tilt_deg: float = float("nan")   # final thrust angle to the local surface normal


def gate_state(sol) -> dict:
    """Moving gate = observation end, including its velocity and control for the terminal junction."""
    braking = sol.phases["observation"]
    return dict(x=braking["X"][:, -1].copy(), u=braking["U"][:, -1].copy())


def solve_branch(p: Params, gate: dict, branch: str, nodes: int | None = None, print_level: int = 0,
                 terrain=None, guess: dict | None = None) -> BranchResult:
    """Terminal descent from a fixed moving gate; optional same-mesh phase guess for a warm start."""
    s = p.scales
    n = nodes or (guess["X"].shape[1] - 1 if guess is not None else p.solver.nodes["terminal"])
    terrain = terrain or FlatTerrain()
    opti = ca.Opti()
    mode = p.engine.mode("terminal")
    f = make_scaled_dynamics(p, mode=mode)
    tr = HermiteSimpson()
    bnd = tuple(v / s.T for v in p.limits.phase_duration_bounds["terminal"])

    scale = np.array([s.L] * 3 + [s.V] * 3 + [s.M])
    site_abs = p.mission.sites[branch]
    site = ca.DM((site_abs - p.mission.r_start) / s.L)
    up = ca.DM(site_abs / np.linalg.norm(site_abs))
    # stand-in for observation, so the tilt-rate junction constraint sees the gate control
    prev = SimpleNamespace(powered=True, U=ca.DM(gate["u"] / np.array([s.F] * 4)).reshape((4, 1)),
                           mode=p.engine.mode("observation"))
    ph = make_phase(opti, f"terminal_{branch}", "terminal", n, True, t_bounds=bnd, branch=branch, site=site,
                    site_up=up, prev=prev, mode=mode)
    ph.landing_target = site
    if p.mission.landing_tolerance > 0:
        ph.site = ph.X[0:3, -1]
        r_end = ca.DM(p.mission.r_start / s.L) + ph.site
        ph.site_up = r_end / ca.norm_2(r_end)
    args = (p, terrain)
    ph.constraints = [ThrustBounds(*args), MassFloor(*args), GlideSlope(*args), TiltRate(*args),
                      Touchdown(*args), TouchdownUpright(*args), LandingRegion(*args)]

    opti.subject_to(ca.vec(tr.defects(f, ph.X, ph.U, ph.tdur, ph.n)) == 0)
    ph.apply_constraints(opti)
    opti.subject_to(ph.X[:, 0] == ca.DM(gate["x"] / scale))
    opti.minimize(-ph.X[6, -1])

    x0 = gate["x"]
    lo, hi = p.limits.phase_duration_bounds["terminal"]
    t_term = float(np.clip(3.0 * np.sqrt(2.0 * max(_alt(p, x0), 10.0) / p.moon.g_surf), lo * 1.05, hi * 0.95))
    m_term = x0[6] * (1.0 - p.moon.g_surf * t_term / mode.ve_hi)
    g = guess if guess is not None else _Builder(p).powered(
        x0[0:3] + p.mission.r_start, x0[3:6], x0[6], site_abs, np.zeros(3), m_term, t_term, n, mode)
    opti.set_initial(ph.X, g["X"] / scale[:, None])
    opti.set_initial(ph.U, g["U"] / np.array([s.F] * 4)[:, None])
    opti.set_initial(ph.tdur, g["t"] / s.T)

    opti.solver("ipopt", {"expand": True, "print_time": False},
                {"max_iter": int(p.solver.ipopt["max_iter"]), "tol": float(p.solver.ipopt["tol"]),
                 "acceptable_iter": 0, "bound_relax_factor": 0,
                 "print_level": print_level, "sb": "yes",
                 **{k: v for k, v in p.solver.ipopt.items() if k not in ("max_iter", "tol")}})
    try:
        val = opti.solve().value
        ok = True
    except RuntimeError:
        val = opti.debug.value
        ok = False
    st = opti.stats()
    status = st["return_status"]
    ok = ok and status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
    out = dict(X=np.atleast_2d(val(ph.X)) * scale[:, None], U=np.atleast_2d(val(ph.U)) * s.F,
               t=float(val(ph.tdur)) * s.T, powered=True, kind="terminal", branch=branch)
    res = BranchResult(branch, ok, status, int(st.get("iter_count", -1)), out)
    res.dv = float(phase_dv(p, out))
    res.m_final = float(out["X"][6, -1])
    res.propellant = float(out["X"][6, 0] - out["X"][6, -1])
    res.touchdown_pos_err, res.touchdown_up_speed = _repropagate(p, out, branch)
    site_abs = p.mission.r_start + out["X"][0:3, -1]
    res.touchdown_tilt_deg = thrust_tilt_deg(out["U"][0:3, -1], site_abs / np.linalg.norm(site_abs))
    return res


def _alt(p: Params, x: np.ndarray) -> float:
    r = x[0:3] + p.mission.r_start
    return float(np.linalg.norm(r) - np.linalg.norm(p.mission.r_start))


def _repropagate(p: Params, ph: dict, branch: str):
    """Integrate the optimised controls (linear between nodes) from the gate and compare with the site."""
    f = make_numpy_rhs(p, mode=p.engine.mode("terminal"))
    T, n = ph["t"], ph["X"].shape[1] - 1
    tn = np.linspace(0.0, T, n + 1)
    U = ph["U"]
    sol = solve_ivp(lambda t, x: f(x, np.array([np.interp(t, tn, U[i]) for i in range(4)])), (0.0, T),
                    ph["X"][:, 0], rtol=1e-10, atol=1e-9, method="DOP853")
    x = sol.y[:, -1]
    site = p.mission.r_start + ph["X"][0:3, -1]
    return (float(np.linalg.norm(x[0:3] - (site - p.mission.r_start))),
            float(x[3:6] @ (site / np.linalg.norm(site))))


def divert_descent_cost(p: Params, sol, branch: str) -> dict:
    """Cost of the divert descent alone: from the SAME moving gate (end of hop-1 observation), the terminal descent to
    the divert site against the terminal descent to the nominal site. Each is re-optimised from the gate."""
    gate = gate_state(sol)
    nom = solve_branch(p, gate, "nominal")
    div = solve_branch(p, gate, branch)
    ok = nom.success and div.success
    return dict(success=ok, gate_mass=float(gate["x"][6]),
                nominal=dict(dv=nom.dv, propellant=nom.propellant, duration=nom.phase["t"] if nom.phase else None),
                divert=dict(dv=div.dv, propellant=div.propellant, duration=div.phase["t"] if div.phase else None),
                extra_dv=div.dv - nom.dv, extra_propellant=div.propellant - nom.propellant)

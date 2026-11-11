"""Initial guess: Lambert coast arc + rocket-equation burn estimates, cubic-Hermite state interpolation.

Everything here is dimensional SI with *absolute* positions; `build_guess` converts to start-site offsets.
Each phase entry: dict(X=7x(n+1), U=4x(n+1), t=duration [s]).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import make_numpy_rhs
from .environment.gravity import CentralGravity
from .lambert import lambert_fixed, min_dv_hop
from .params import Params
from .phases import stage_name, stage_nodes


def _hermite(r0, v0, r1, v1, t, s):
    """Cubic Hermite position, velocity, acceleration at fractions s (array) over duration t."""
    s = np.asarray(s)[None, :]
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    d00, d10, d01, d11 = 6 * s**2 - 6 * s, 3 * s**2 - 4 * s + 1, -6 * s**2 + 6 * s, 3 * s**2 - 2 * s
    c00, c10, c01, c11 = 12 * s - 6, 6 * s - 4, -12 * s + 6, 6 * s - 2
    r0, v0, r1, v1 = (np.asarray(a)[:, None] for a in (r0, v0, r1, v1))
    r = h00 * r0 + h10 * t * v0 + h01 * r1 + h11 * t * v1
    v = (d00 * r0 + d01 * r1) / t + d10 * v0 + d11 * v1
    a = (c00 * r0 + c01 * r1) / t**2 + (c10 * v0 + c11 * v1) / t
    return r, v, a


class _Builder:
    def __init__(self, p: Params):
        self.p = p
        self.rhs = make_numpy_rhs(p, CentralGravity(p.moon.mu))
        self.rs = p.mission.r_start

    def a_env(self, r_off, v):
        x = np.concatenate([r_off, v, [1.0]])
        return self.rhs(x, np.zeros(4))[3:6]

    def powered(self, r0, v0, m0, r1, v1, m1, t, n, mode, span=(0.0, 1.0)):
        """Hermite states + thrust that realises them: T = m (r'' - a_env), clipped to the stage's thrust band.

        The Hermite arc runs over the whole burn (duration t); `span` picks the fraction of it this stage covers."""
        s = np.linspace(span[0], span[1], n + 1)
        r, v, a = _hermite(r0 - self.rs, v0, r1 - self.rs, v1, t, s)
        m = m0 + (m1 - m0) * s
        lo, hi = mode.gamma_lo, mode.gamma_hi
        T = np.zeros((3, n + 1))
        G = np.zeros(n + 1)
        for k in range(n + 1):
            Tk = m[k] * (a[:, k] - self.a_env(r[:, k], v[:, k]))
            nrm = np.linalg.norm(Tk)
            if nrm < 1e-9:
                Tk, nrm = (self.rs / np.linalg.norm(self.rs)) * lo, lo
            g = min(max(nrm, lo), hi)
            T[:, k] = Tk / nrm * g
            G[k] = g
        return dict(X=np.vstack([r, v, m[None, :]]), U=np.vstack([T, G[None, :]]), t=t * (span[1] - span[0]))

    def staged(self, kind, r0, v0, m0, r1, v1, m1, t, nodes, relaxed=False, hop=1):
        """One guess entry per burn stage of `kind`, keyed by stage index; stages share the burn time equally."""
        eng = self.p.engine
        K = eng.stage_count(kind, relaxed, hop)
        n = stage_nodes(nodes[kind], K)
        return {k: self.powered(r0, v0, m0, r1, v1, m1, t, n, eng.mode(kind, k, relaxed, hop), (k / K, (k + 1) / K))
                for k in range(K)}


def build_guess(p: Params, branch: str, h_gate: float | None = None, nodes: dict | None = None,
                h_gate2: float | None = None, relaxed: bool = False):
    """Round-trip guess: hop 1 (start site -> gate over the target -> divert site `branch`) keyed `rise`, ...,
    `terminal_<branch>`, then hop 2 (divert site -> start site) keyed `rise2_<branch>`, ..., `terminal2_<branch>`."""
    nodes = nodes or p.solver.nodes
    mis = p.mission
    lo, hi = mis.gate_alt_bounds
    if h_gate is None:
        h_gate = min(max(3.0 * mis.divert_radius, lo * 2.0), hi)
    guess = _hop_guess(p, mis.r_start, mis.r_target, mis.up, {branch: mis.sites[branch]}, p.vehicle.m_wet, nodes,
                       h_gate, relaxed)
    up_s = mis.r_start / np.linalg.norm(mis.r_start)
    h2 = h_gate2 if h_gate2 is not None else min(max(2.0 * lo, 100.0), hi)
    m_land = guess[f"terminal_{branch}"]["X"][6, -1]
    g2 = _hop_guess(p, mis.sites[branch], mis.r_start, up_s, {"home": mis.r_start}, m_land, nodes, h2, relaxed, hop=2)
    for k, v in g2.items():
        guess[_hop2_key(k, branch)] = v
    return guess


def _hop2_key(k: str, br: str) -> str:
    """Hop-1 guess key -> hop-2 phase name: ascent -> ascent2_<br>, ascent_s2 -> ascent2_<br>_s2."""
    if k == "terminal_home":
        return f"terminal2_{br}"
    base, _, stage = k.partition("_s")
    return f"{base}2_{br}" + (f"_s{stage}" if stage else "")


def _burn_reference(modes) -> tuple:
    """(mdot, ve) of a burn at full thrust, averaged over its stages; only seeds burn durations and masses."""
    mdot = float(np.mean([m.mdot(m.gamma_hi) for m in modes]))
    ve = float(np.mean([m.ve_hi for m in modes]))
    return mdot, ve


def _add_stages(guess: dict, base: str, stages: dict):
    for k, entry in stages.items():
        guess[stage_name(base, k)] = entry


def _hop_guess(p: Params, launch, gate_site, gate_up, sites: dict, m0: float, nodes: dict, h_gate: float,
               relaxed: bool = False, hop: int = 1):
    """Guess for one hop: launch site -> gate over `gate_site` -> a terminal descent to each site in `sites`."""
    b = _Builder(p)
    eng, lim, mis = p.engine, p.limits, p.mission
    g = p.moon.g_surf
    rs = launch
    up_s = rs / np.linalg.norm(rs)
    bounds = lim.phase_duration_bounds

    def clip(t, k):
        return float(min(max(t, bounds[k][0] * 1.05), bounds[k][1] * 0.95))

    # ---- 1. vertical rise at full thrust
    rise_mode = eng.mode("rise", 0, relaxed, hop)
    a_up = rise_mode.gamma_hi / m0 - g
    t_rise = clip(math.sqrt(2 * lim.h_rise / a_up), "rise")
    v_rise = 2 * lim.h_rise / t_rise
    r_rise = rs + lim.h_rise * up_s
    m_rise = m0 - rise_mode.mdot(rise_mode.gamma_hi) * t_rise
    guess = {}
    guess["rise"] = b.powered(rs, np.zeros(3), m0, r_rise, v_rise * up_s, m_rise, t_rise,
                              nodes["rise"], rise_mode)

    # ---- 2. Lambert baseline -> burn durations
    base = min_dv_hop(p, r1=launch, r2=gate_site)
    mdot_asc, ve_asc = _burn_reference([eng.mode("ascent", k, relaxed, hop) for k in range(eng.stage_count("ascent", relaxed, hop))])
    mdot_brk, ve_brk = _burn_reference([eng.mode("braking", k, relaxed, hop) for k in range(eng.stage_count("braking", relaxed, hop))])
    m_asc = m_rise * math.exp(-base.dv_depart / ve_asc)
    t_asc = clip((m_rise - m_asc) / mdot_asc, "ascent")
    m_brk = m_asc * math.exp(-base.dv_arrive / ve_brk)
    t_brk = clip((m_asc - m_brk) / mdot_brk, "braking")

    # ---- 3. coast arc: Lambert from the end of the ascent burn to the start of the braking burn
    r_gate = gate_site + h_gate * gate_up
    r_a = r_rise + 0.5 * base.v1 * t_asc
    v_a_guess = base.v1
    r_b = r_gate - 0.5 * base.v2 * t_brk
    t_coast = clip(base.tof - 0.5 * (t_asc + t_brk), "coast")
    try:
        v_a, _ = lambert_fixed(p, r_a, r_b, t_coast)
    except ValueError:
        v_a = v_a_guess
    r_off0 = mis.r_start          # offsets in the state are always relative to the mission start site
    x0 = np.concatenate([r_a - r_off0, v_a, [m_asc]])
    sol = solve_ivp(lambda _t, x: b.rhs(x, np.zeros(4)), (0, t_coast), x0, rtol=1e-11, atol=1e-9,
                    dense_output=True, method="DOP853")
    tau_c = np.linspace(0.0, 1.0, nodes["coast"] + 1)
    Xc = np.array([sol.sol(t_coast * s) for s in tau_c]).T
    guess["coast"] = dict(X=Xc, U=np.zeros((4, nodes["coast"] + 1)), t=t_coast)
    r_b_act, v_b_act = Xc[0:3, -1] + r_off0, Xc[3:6, -1]

    # ---- 2'. ascent burn: rise end -> coast start
    _add_stages(guess, "ascent", b.staged("ascent", r_rise, v_rise * up_s, m_rise, r_a, v_a, m_asc, t_asc, nodes,
                                          relaxed, hop))

    # ---- 4. Brake to a descending camera window, then reach the decision gate.
    t_obs = max(p.sensor.observation_time_min + p.sensor.processing_delay, bounds["observation"][0]) * 1.1
    v_gate = -min(10.0, p.sensor.descent_speed_bounds[1] / 2) * gate_up
    r_obs = r_gate - v_gate * t_obs
    obs_mode, term_mode = eng.mode("observation", 0, relaxed, hop), eng.mode("terminal", 0, relaxed, hop)
    m_obs = m_brk * math.exp(-g * t_obs / obs_mode.ve_hi)
    _add_stages(guess, "braking", b.staged("braking", r_b_act, v_b_act, m_asc, r_obs, v_gate, m_brk, t_brk, nodes,
                                           relaxed, hop))
    guess["observation"] = b.powered(r_obs, v_gate, m_brk, r_gate, v_gate, m_obs, t_obs,
                                     nodes["observation"], obs_mode)

    # ---- 6. Terminal descent begins with downward motion, without a hover.
    t_term = clip(3.0 * math.sqrt(2 * h_gate / g), "terminal")
    m_term = m_obs * (1.0 - g * t_term / term_mode.ve_hi)
    for br, site in sites.items():
        guess[f"terminal_{br}"] = b.powered(r_gate, v_gate, m_obs, site, np.zeros(3),
                                            m_term, t_term, nodes["terminal"], term_mode)
    return guess

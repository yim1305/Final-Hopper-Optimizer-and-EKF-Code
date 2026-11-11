"""The round-trip optimal control problem: hop 1 (start site -> gate over the target -> divert site) then hop 2
(divert site -> gate -> start region, no divert). Each hop is rise, ascent, coast, braking, observation, terminal
descent."""
from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from .constraints import (GateAltitude, GateDescending, GateOverTarget, GlideSlope, MassFloor, MinAltitude,
                          ThrustBounds, TiltRate, Touchdown, TouchdownUpright, VerticalRise, LandingRegion,
                          SensorObservation)
from .dynamics import make_scaled_dynamics
from .environment.terrain import FlatTerrain
from .initial_guess import build_guess
from .params import BURNS, Params
from .phases import make_phase, stage_name, stage_nodes
from .transcription import HermiteSimpson
from .sensor import camera_axes

HOP_KINDS = ("rise", "ascent", "coast", "braking", "observation", "terminal")


@dataclass
class Solution:
    phases: dict          # name -> dict(X, U, t, powered, kind, branch)  (dimensional, offset positions)
    z: float              # final mass of the round trip [kg]
    status: str
    success: bool
    iterations: int
    solve_time: float
    m_wet: float = float("nan")   # initial mass: the config value, or the solved value in size mode
    primal_infeasibility: float = float("nan")
    dual_infeasibility: float = float("nan")


class Problem:
    def __init__(self, p: Params, branch: str = "uprange", nodes: dict | None = None, terrain=None,
                 transcription=None, mode: str | None = None, relaxed: bool = False,
                 fixed_m_wet: float | None = None):
        """`branch` is the divert site hop 1 lands on (one of mission.sites: nominal, uprange, downrange, left, right).

        mode "maximise_margin": fixed m_wet (config), maximise the final mass.
        mode "size": m_wet is a decision variable; minimise it subject to loaded propellant >= reserve_factor x the
        propellant used (m_final >= m_wet - (m_wet - m_dry) / reserve_factor, linear in m_wet).
        Default: config mission.mode.

        relaxed: one stage per phase with every engine of the pool as one continuous throttle at the pool's average
        Isp. Used to find the burn structure before solving the staged problem; see staging.py."""
        self.relaxed = relaxed
        mode = mode or p.mission.mode
        if mode not in ("size", "maximise_margin"):
            raise ValueError("mode must be 'size' or 'maximise_margin'")
        self.mode = mode
        self.fixed_m_wet = p.vehicle.m_wet if fixed_m_wet is None else float(fixed_m_wet)
        if fixed_m_wet is not None and (mode != "maximise_margin" or self.fixed_m_wet <= p.vehicle.m_dry):
            raise ValueError("fixed_m_wet requires maximise_margin mode and wet mass above dry mass")
        self.p = p
        self.s = p.scales
        self.branch = branch
        self.nodes = dict(nodes or p.solver.nodes)
        self.terrain = terrain or FlatTerrain()
        self.tr = transcription or HermiteSimpson()
        self._dynamics = {}
        self.opti = ca.Opti()
        self._build()

    def dynamics_of(self, phase):
        """Scaled dynamics for the phase's burn stage (one CasADi function per distinct engine mode)."""
        key = None if phase.mode is None else (phase.hop, phase.kind, phase.stage)
        if key not in self._dynamics:
            self._dynamics[key] = make_scaled_dynamics(self.p, mode=phase.mode)
        return self._dynamics[key]

    # ---- construction -----------------------------------------------------------------------
    def _stage_nodes(self, kind, hop=1):
        return stage_nodes(self.nodes[kind], self.p.engine.stage_count(kind, self.relaxed, hop))

    def _make_hop(self, names, launch_up, site, site_up, gate_over_target, observed_target, hop=1):
        """Rise, ascent (one or more stages), coast, braking (one or more stages), observation, terminal descent, with
        descending sensing before the decision gate and an optimized touchdown point."""
        p, s, opti, n, eng = self.p, self.s, self.opti, self.nodes, self.p.engine
        bnd = {k: (lo / s.T, hi / s.T) for k, (lo, hi) in p.limits.phase_duration_bounds.items()}
        args = (p, self.terrain)
        phases = []

        def add(kind, base, **kw):
            """Append the stages of one kind, chaining each to the phase before it; returns them."""
            powered = kind != "coast"
            made = []
            for k in range(eng.stage_count(kind, self.relaxed, hop) if powered else 1):
                prev = (phases + made)[-1] if (phases or made) else None
                # later stages may shrink to a short minimum, so a stage the optimizer has no use for collapses
                bounds = bnd[kind] if k == 0 else (p.limits.stage_duration_min / s.T, bnd[kind][1])
                made.append(make_phase(opti, stage_name(base, k), kind, self._stage_nodes(kind, hop), powered,
                                       t_bounds=bounds, prev=prev, stage=k,
                                       mode=eng.mode(kind, k, self.relaxed, hop), hop=hop, **kw))
            phases.extend(made)
            return made

        rise = add("rise", names[0], launch_up=launch_up)[0]
        ascent = add("ascent", names[1])
        coast = add("coast", names[2])[0]
        braking = add("braking", names[3])
        observation = add("observation", names[4])[0]
        terminal = add("terminal", names[5], branch=self.branch, site=site, site_up=site_up)[0]
        terminal.landing_target = site
        if p.mission.landing_tolerance > 0:
            terminal.site = terminal.X[0:3, -1]
            site_abs = ca.DM(p.mission.r_start / s.L) + terminal.site
            terminal.site_up = site_abs / ca.norm_2(site_abs)
        observation.sensor_sites = [observed_target, terminal.site]
        rise.constraints = [ThrustBounds(*args), MassFloor(*args), TiltRate(*args), VerticalRise(*args)]
        for ph in ascent:
            ph.constraints = [ThrustBounds(*args), MassFloor(*args), MinAltitude(*args), TiltRate(*args)]
        coast.constraints = [MassFloor(*args), MinAltitude(*args)]
        for ph in braking:
            ph.constraints = [ThrustBounds(*args), MassFloor(*args), MinAltitude(*args), TiltRate(*args)]
        observation.constraints = [ThrustBounds(*args), MassFloor(*args), TiltRate(*args), GateAltitude(*args),
                                   GateDescending(*args), SensorObservation(*args)]
        if gate_over_target:
            observation.constraints.append(GateOverTarget(*args))
        terminal.constraints = [ThrustBounds(*args), MassFloor(*args), GlideSlope(*args), TiltRate(*args),
                                Touchdown(*args), TouchdownUpright(*args), LandingRegion(*args)]
        return phases

    def _build(self):
        p, s, opti, br = self.p, self.s, self.opti, self.branch
        m = p.mission
        site1 = m.sites[br]
        up_start = m.r_start / np.linalg.norm(m.r_start)
        up_site = site1 / np.linalg.norm(site1)

        # hop 1: start site -> divert site (gate over the target); hop 2: divert site -> start landing region
        self.hop1 = self._make_hop(("rise", "ascent", "coast", "braking", "observation", f"terminal_{br}"), up_start,
                                   ca.DM((site1 - m.r_start) / s.L), ca.DM(up_site), m.gate_over_target,
                                   ca.DM((m.r_target - m.r_start) / s.L))
        self.hop2 = self._make_hop(tuple(f"{k}2_{br}" for k in HOP_KINDS), self.hop1[-1].site_up,
                                   ca.DM.zeros(3, 1), ca.DM(up_start), False, ca.DM.zeros(3, 1), hop=2)
        self.phases = self.hop1 + self.hop2

        for ph in self.phases:
            opti.subject_to(ca.vec(self.tr.defects(self.dynamics_of(ph), ph.X, ph.U, ph.tdur, ph.n)) == 0)
            ph.apply_constraints(opti)

        # boundary conditions and linkage
        rise1, term1, rise2, term2 = self.hop1[0], self.hop1[-1], self.hop2[0], self.hop2[-1]
        # Counts only fall within a burn. The lowest-numbered unit therefore
        # fires longest; bounding its complete rise/ascent or descent firing
        # also bounds every unit that shuts down earlier. Coast/surface reset it.
        for hop_phases in (self.hop1, self.hop2):
            for burn in BURNS:
                for name, cls in p.engine.classes.items():
                    if cls.max_continuous_burn_s is None or self.relaxed:
                        continue
                    active = [ph for ph in hop_phases if ph.kind in burn and
                              dict(ph.mode.fixed + ph.mode.throttled).get(name, 0) > 0]
                    if active:
                        duration = sum(ph.tdur for ph in active) * s.T
                        opti.subject_to(duration / cls.max_continuous_burn_s <= 1.)
        self.initial_boundary_offset = int(opti.ng)
        if self.mode == "size":
            self.mw = opti.variable()                       # nondimensional wet mass (scaled by config m_wet)
            opti.subject_to(opti.bounded(1.05 * p.vehicle.m_dry / s.M, self.mw, 3.0))
            opti.subject_to(rise1.X[0:6, 0] == 0)
            opti.subject_to(rise1.X[6, 0] == self.mw)
        else:
            self.mw = None
            opti.subject_to(rise1.X[:, 0] == ca.DM([0, 0, 0, 0, 0, 0, self.fixed_m_wet / s.M]))
        for hop in (self.hop1, self.hop2):
            for a, b in zip(hop[:-1], hop[1:]):
                opti.subject_to(b.X[:, 0] == a.X[:, -1])
                # Tracker coverage spans every stage of the ascent. Engine
                # shutdowns may step magnitude, while body attitude is continuous.
                st = p.star_tracker
                if (st is not None and st.enabled and a.powered and b.powered and
                        a.kind in st.required_phase_kinds and b.kind in st.required_phase_kinds):
                    ex, ey, _ = camera_axes(a.U[:3, -1], ca.DM(m.downrange))
                    opti.subject_to(ca.dot(ex, b.U[:3, 0]) == 0)
                    opti.subject_to(ca.dot(ey, b.U[:3, 0]) == 0)
                    opti.subject_to(ca.dot(a.U[:3, -1], b.U[:3, 0]) >= 0)
        # the vehicle sits on the surface between the hops (time ignored): same place, same mass, at rest
        opti.subject_to(rise2.X[0:3, 0] == term1.X[0:3, -1])
        opti.subject_to(rise2.X[3:6, 0] == 0)
        opti.subject_to(rise2.X[6, 0] == term1.X[6, -1])

        self.final = term2
        if self.mode == "size":
            # carry reserve_factor x the propellant used: m_final >= m_wet - (m_wet - m_dry) / f
            f, md = p.mission.reserve_factor, p.vehicle.m_dry / s.M
            opti.subject_to(term2.X[6, -1] >= self.mw - (self.mw - md) / f)
            opti.minimize(self.mw)
        else:
            opti.minimize(-term2.X[6, -1])                   # maximise the final mass

    # ---- guess ------------------------------------------------------------------------------
    @staticmethod
    def guess_from_solution(sol: "Solution") -> dict:
        """Turn a solved Solution into a guess dict (warm start; phase names must match)."""
        return {k: dict(X=v["X"], U=v["U"], t=v["t"]) for k, v in sol.phases.items()}

    def set_guess(self, guess: dict | None = None, m_wet0: float | None = None):
        """Set the initial guess. In size mode `m_wet0` [kg] seeds the wet-mass variable (default: config m_wet)."""
        s = self.s
        guess = guess or build_guess(self.p, self.branch, nodes=self.nodes, relaxed=self.relaxed)
        scale = np.array([s.L] * 3 + [s.V] * 3 + [s.M])
        uscale = np.array([s.F] * 4)
        for ph in self.phases:
            g = guess[ph.name]
            def resample(values):
                if values.shape[1] == ph.n + 1:
                    return values
                old = np.linspace(0, 1, values.shape[1])
                new = np.linspace(0, 1, ph.n + 1)
                return np.array([np.interp(new, old, row) for row in values])
            self.opti.set_initial(ph.X, resample(g["X"]) / scale[:, None])
            if ph.powered:
                self.opti.set_initial(ph.U, resample(g["U"]) / uscale[:, None])
            if not isinstance(ph.tdur, float):
                self.opti.set_initial(ph.tdur, g["t"] / s.T)
        if self.mode == "size":
            self.opti.set_initial(self.mw, (m_wet0 if m_wet0 is not None else self.p.vehicle.m_wet) / s.M)
        self.guess = guess

    # ---- solve ------------------------------------------------------------------------------
    def solve(self, print_level: int = 5) -> Solution:
        extra = {k: v for k, v in self.p.solver.ipopt.items() if k not in ("max_iter", "tol")}
        opts = {"max_iter": int(self.p.solver.ipopt["max_iter"]), "tol": float(self.p.solver.ipopt["tol"]),
                "acceptable_iter": 0, "bound_relax_factor": 0,
                "print_level": print_level, "sb": "yes", **extra}     # any further IPOPT option can come from the config
        self.opti.solver("ipopt", {"expand": True, "print_time": False}, opts)
        try:
            sol = self.opti.solve()
            val = sol.value
            success = True
        except RuntimeError:
            val = self.opti.debug.value
            success = False
        st = self.opti.stats()
        s = self.s
        out = {}
        scale = np.array([s.L] * 3 + [s.V] * 3 + [s.M])[:, None]
        for ph in self.phases:
            X = np.atleast_2d(val(ph.X)) * scale
            U = (np.atleast_2d(val(ph.U)) * s.F) if ph.powered else np.zeros((4, ph.n + 1))
            t = (float(val(ph.tdur)) if not isinstance(ph.tdur, float) else ph.tdur) * s.T
            out[ph.name] = dict(X=X, U=U, t=t, powered=ph.powered, kind=ph.kind, branch=ph.branch, stage=ph.stage,
                                relaxed=self.relaxed, hop=ph.hop)
        z = float(val(self.final.X[6, -1])) * s.M
        m_wet = float(val(self.mw)) * s.M if self.mode == "size" else self.fixed_m_wet
        hist = st.get("iterations", {})
        primal = float(hist.get("inf_pr", [float("inf")])[-1])
        dual = float(hist.get("inf_du", [float("inf")])[-1])
        # IPOPT can report acceptable-level termination after tiny steps while
        # enormous multipliers hide a poor unscaled dual residual.
        residual_limit = max(1e-6, 100 * float(self.p.solver.ipopt["tol"]))
        return Solution(out, z, st["return_status"], success and
                        st["return_status"] in ("Solve_Succeeded", "Solved_To_Acceptable_Level") and
                        max(primal, dual) <= residual_limit,
                        int(st.get("iter_count", -1)), float(st["t_wall_total"]) if "t_wall_total" in st else float("nan"),
                        m_wet, primal, dual)

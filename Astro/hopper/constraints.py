"""Constraints: each is a class with apply(opti, phase). Add a new one with a class plus one line in problem.py.

All quantities are nondimensional (see Params.scales); positions are offsets from the start site.
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from .environment.terrain import Terrain, altitude
from .params import Params
from .dynamics import make_scaled_dynamics
from .sensor import camera_axes


class Constraint:
    def apply(self, opti, phase):
        raise NotImplementedError


class _WithParams(Constraint):
    def __init__(self, p: Params, terrain: Terrain | None = None):
        self.p = p
        self.s = p.scales
        self.terrain = terrain
        self.rs = p.mission.r_start / self.s.L          # nondimensional start-site position
        self.R = p.moon.radius / self.s.L

    def alt(self, r):
        """Nondimensional altitude of offset position r (3 x n)."""
        if self.terrain is None:
            raise RuntimeError("constraint needs a terrain model")
        # terrain.height takes dimensional r_abs; scale in/out
        class _T:
            def height(_, r_abs):
                return self.terrain.height(tuple(c * self.s.L for c in r_abs)) / self.s.L
        return altitude(self.rs, self.R, r, _T())


class ThrustBounds(_WithParams):
    """The burn stage's thrust band gamma_lo <= Gamma <= gamma_hi and |T_vec| = Gamma on powered phases."""

    def apply(self, opti, phase):
        if not phase.powered:
            return
        T = phase.U[0:3, :]
        G = phase.U[3, :]
        lo, hi = phase.mode.gamma_lo / self.s.F, phase.mode.gamma_hi / self.s.F
        opti.subject_to(opti.bounded(lo, G, hi))
        opti.subject_to(ca.sum1(T * T) == G * G)
        # Throttled engines share one per-engine command. Controls interpolate
        # linearly, so bounding each interval slope bounds its entire history.
        for name, count in phase.mode.throttled:
            cls = self.p.engine.classes[name]
            if not count or cls.throttle_rate_fraction_s is None:
                continue
            fraction = (G * self.s.F - phase.mode.thrust_fixed) / (count * cls.thrust)
            delta = fraction[:, 1:] - fraction[:, :-1]
            allowed = cls.throttle_rate_fraction_s * phase.tdur * self.s.T / phase.n
            opti.subject_to(opti.bounded(-allowed, delta, allowed))
            prev = phase.prev
            if prev is not None and prev.powered:
                previous_count = dict(prev.mode.throttled).get(name, 0)
                if previous_count:
                    previous = (prev.U[3, -1] * self.s.F - prev.mode.thrust_fixed) / (previous_count * cls.thrust)
                    opti.subject_to(fraction[0] == previous)


class MassFloor(_WithParams):
    """m >= m_dry at all nodes."""

    def apply(self, opti, phase):
        opti.subject_to(phase.X[6, :] >= self.p.vehicle.m_dry / self.s.M)


class MinAltitude(_WithParams):
    """h >= min_altitude (ascent, coast, braking)."""

    def apply(self, opti, phase):
        # The linked start node is already constrained by the preceding phase. In
        # particular, rise ends at exactly h_rise == min_altitude in the default
        # configuration. Repeating it as an active inequality leaves no strict
        # interior and drives IPOPT's multipliers to ~1e14.
        prev = phase.prev
        inherited = prev is not None and (
            any(isinstance(c, MinAltitude) for c in prev.constraints) or
            (prev.kind == "rise" and self.p.limits.h_rise >= self.p.limits.min_altitude))
        start = 1 if inherited else 0
        opti.subject_to(self.alt(phase.X[0:3, start:]) >= self.p.limits.min_altitude / self.s.L)


class GateAltitude(_WithParams):
    """Altitude window at the observation end (the decision gate)."""

    def apply(self, opti, phase):
        lo, hi = self.p.mission.gate_alt_bounds
        h = self.alt(phase.X[0:3, -1])
        opti.subject_to(h >= lo / self.s.L)
        opti.subject_to(h <= hi / self.s.L)


class GlideSlope(_WithParams):
    """h_rel >= tan(slope) * d_horiz relative to this branch's site (terminal descents).

    Written as h_rel >= 0 and h_rel^2 >= tan^2 * d_horiz^2. The final node (on the site, where both are
    zero) is skipped; it is pinned by the landing-site equality.
    """

    def apply(self, opti, phase):
        up = phase.site_up
        d = phase.X[0:3, 0:phase.n] - phase.site
        hrel = ca.mtimes(up.T, d)
        dh2 = ca.sum1(d * d) - hrel * hrel
        opti.subject_to(hrel >= 0)
        opti.subject_to(hrel * hrel >= np.tan(self.p.limits.glide_slope) ** 2 * dh2)


class TiltRate(_WithParams):
    """u_k . u_{k+1} >= cos(rate * dt) with u = T_vec / Gamma, inside a phase and across a junction."""

    def apply(self, opti, phase):
        if not phase.powered:
            return
        rate = self.p.limits.tilt_rate_max
        if phase.kind == "observation" and self.p.sensor.enforce_pointing:
            rate = min(rate, self.p.sensor.pointing_rate_max)
        rate_nd = rate * self.s.T      # rad per nondimensional time
        T, G = phase.U[0:3, :], phase.U[3, :]
        n = phase.n
        dt = phase.tdur / n
        dot = ca.sum1(T[:, 0:n] * T[:, 1:n + 1])
        opti.subject_to(dot >= G[0:n] * G[1:n + 1] * ca.cos(rate_nd * dt))
        prev = phase.prev
        if prev is not None and prev.powered:
            # across the junction the control moves from the last node of prev to the first node of this
            # phase; allow one interval of this phase for the rotation (a zero-length gap would force
            # exactly parallel thrust and make the constraint degenerate)
            d = ca.sum1(prev.U[0:3, -1] * T[:, 0])
            opti.subject_to(d >= prev.U[3, -1] * G[0] * ca.cos(rate_nd * dt))


class Touchdown(_WithParams):
    """Descending or level, with bounded downward and horizontal speeds at the last node."""

    def apply(self, opti, phase):
        v = phase.X[3:6, -1]
        up = phase.site_up
        vup = ca.dot(up, v)
        vlimit = self.p.limits.td_vert_speed_max / self.s.V
        hlimit = self.p.limits.td_horiz_speed_max / self.s.V
        if vlimit == 0:
            opti.subject_to(vup == 0)
        else:
            opti.subject_to(opti.bounded(-1, vup / vlimit, 0))
        if hlimit == 0:
            e1, e2 = _tangents(up, self.p.mission.r_start)
            opti.subject_to(ca.dot(e1, v) == 0)
            opti.subject_to(ca.dot(e2, v) == 0)
        else:
            # Divide by one power of the limit: order-one boundary gradients without a huge Hessian.
            vhorizontal = v - vup * up
            opti.subject_to(ca.dot(vhorizontal, vhorizontal) / hlimit <= hlimit)


class TouchdownUpright(_WithParams):
    """Bound the final thrust/body-up angle to the landing-site surface normal (no attitude dynamics).

    Since |T| <= Gamma, up.T*T >= cos(angle)*Gamma also guarantees the actual thrust angle.
    The zero-angle case uses two tangential equalities to avoid a degenerate zero-width cone.
    """

    def apply(self, opti, phase):
        T = phase.U[0:3, -1]
        up = phase.site_up
        angle = self.p.limits.td_tilt_max
        if angle == 0:
            e1, e2 = _tangents(up, self.p.mission.r_start)
            opti.subject_to(ca.dot(e1, T) == 0)
            opti.subject_to(ca.dot(e2, T) == 0)
            opti.subject_to(ca.dot(up, T) >= 0)
        else:
            opti.subject_to(ca.dot(up, T) >= np.cos(angle) * phase.U[3, -1])


# ---- phase-specific boundary conditions -------------------------------------------------------

class VerticalRise(_WithParams):
    """Phase 1: horizontal velocity zero at every node, ends at h = h_rise.

    Node 0 is skipped: the hop linkage already fixes the full launch velocity to zero there, and repeating it as
    a horizontal-velocity equality would make the constraint Jacobian rank-deficient."""

    def apply(self, opti, phase):
        e1, e2 = _tangents(phase.launch_up, self.p.mission.r_start)
        v = phase.X[3:6, 1:]
        opti.subject_to(ca.mtimes(e1.T, v) == 0)
        opti.subject_to(ca.mtimes(e2.T, v) == 0)
        opti.subject_to(self.alt(phase.X[0:3, -1]) == self.p.limits.h_rise / self.s.L)


class GateDescending(_WithParams):
    """The gate is reached descending or level; its velocity is otherwise free."""

    def apply(self, opti, phase):
        r_abs = self.rs + phase.X[0:3, -1]
        opti.subject_to(ca.dot(r_abs, phase.X[3:6, -1]) <= 0)


class GateOverTarget(_WithParams):
    """Hop-1 gate (end of observation) horizontally directly above the target: no tangential offset from the
    target's local vertical, so the trunk does not favour any one divert site. Altitude stays free."""

    def apply(self, opti, phase):
        m = self.p.mission
        e1, e2 = _tangents(m.up)
        d = phase.X[0:3, -1] - ca.DM((m.r_target - m.r_start) / self.s.L)
        opti.subject_to(ca.mtimes(ca.DM(e1).T, d) == 0)
        opti.subject_to(ca.mtimes(ca.DM(e2).T, d) == 0)


class LandingRegion(_WithParams):
    """Touch down on the surface within the configured distance of the selected landing point."""

    def apply(self, opti, phase):
        end = phase.X[0:3, -1]
        tolerance = self.p.mission.landing_tolerance / self.s.L
        if tolerance == 0:
            opti.subject_to(end == phase.landing_target)
        else:
            opti.subject_to(self.alt(end) == 0)
            d = end - phase.landing_target
            opti.subject_to(ca.dot(d, d) / tolerance <= tolerance)


class SensorObservation(_WithParams):
    """Descending acquisition plus processing, with square-FOV checks at nodes and collocation midpoints.

    With sensor.enforce_pointing false the FOV, line-of-sight depth, range and surface-visibility checks (and the
    observation pointing-rate cap in TiltRate) are dropped; the duration, descent-rate, ground-speed and minimum
    altitude limits remain."""

    def apply(self, opti, phase):
        se, s = self.p.sensor, self.s
        opti.subject_to(phase.tdur >= (se.observation_time_min + se.processing_delay) / s.T)
        f = make_scaled_dynamics(self.p, mode=phase.mode)
        X, U, n = phase.X, phase.U, phase.n
        F = f.map(n + 1)(X, U)
        Xm = (X[:, :-1] + X[:, 1:]) / 2 + (phase.tdur / n / 8) * (F[:, :-1] - F[:, 1:])
        Um = (U[:, :-1] + U[:, 1:]) / 2
        samples_x, samples_u = ca.horzcat(X, Xm), ca.horzcat(U, Um)
        heading = ca.DM(self.p.mission.downrange)
        tan_h, tan_v = (np.tan(a / 2) for a in se.fov)
        vmin, vmax = se.descent_speed_bounds
        horizontal_max = se.horizontal_speed_max / s.V
        range_max = se.max_range / s.L
        for k in range(2 * n + 1):
            r, v = samples_x[0:3, k], samples_x[3:6, k]
            r_abs = self.rs + r
            up = r_abs / ca.norm_2(r_abs)
            vup = ca.dot(up, v)
            opti.subject_to(opti.bounded(-1, vup / (vmax / s.V), -vmin / vmax))
            vhorizontal = v - vup * up
            opti.subject_to(ca.dot(vhorizontal, vhorizontal) / horizontal_max <= horizontal_max)
            opti.subject_to(self.alt(r) >= se.min_altitude / s.L)
            if not se.enforce_pointing:
                continue        # descent-rate, ground-speed and altitude limits stay; the camera pointing checks drop
            image_x, image_y, boresight = camera_axes(samples_u[0:3, k], heading)
            for point in phase.sensor_sites:
                los = point - r
                depth = ca.dot(boresight, los)
                opti.subject_to(depth >= 0)
                opti.subject_to(opti.bounded(-depth * tan_h, ca.dot(image_x, los), depth * tan_h))
                opti.subject_to(opti.bounded(-depth * tan_v, ca.dot(image_y, los), depth * tan_v))
                opti.subject_to(ca.dot(los, los) / range_max <= range_max)
                point_abs = self.rs + point
                # Convex spherical surface: the observer must lie above the point's tangent plane.
                opti.subject_to(ca.dot(point_abs / ca.norm_2(point_abs), -los) >= 0)


def _tangents(up, reference=None):
    """Two tangent directions, also supporting an optimized landing/launch normal."""
    reference = np.asarray(up if reference is None else reference, dtype=float).reshape(3)
    axis = ca.DM(np.eye(3)[int(np.argmin(np.abs(reference)))])
    up = ca.reshape(up, 3, 1)
    e1 = axis - ca.dot(axis, up) * up
    e1 = e1 / ca.norm_2(e1)
    return e1, ca.cross(up, e1)

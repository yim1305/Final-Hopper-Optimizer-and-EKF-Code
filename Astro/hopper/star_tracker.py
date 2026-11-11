"""Two fixed tracker directions, frames, mounting search and update schedules.

All internal angles are radians and times are absolute dimensional mission seconds.
No functions in this module add constraints or decision variables to IPOPT.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq, minimize
from scipy.spatial.transform import Rotation, Slerp

from .dynamics import make_numpy_rhs
from .postprocess import chain, phase_mode


class OrientationError(ValueError):
    """The prescribed orientation cannot support a resolved coverage claim."""


def rotation_z(angle):
    a = np.asarray(angle)
    result = np.zeros(a.shape + (3, 3))
    result[..., 0, 0] = result[..., 1, 1] = np.cos(a)
    result[..., 1, 0] = np.sin(a)
    result[..., 0, 1] = -np.sin(a)
    result[..., 2, 2] = 1
    return result


def angle_between(a, b):
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))


def mount_axis(tilt, azimuth):
    return np.array([math.sin(tilt) * math.cos(azimuth), math.sin(tilt) * math.sin(azimuth),
                     math.cos(tilt)])


def top_mount_axes(elevations, azimuths):
    e, beta = np.asarray(elevations), np.asarray(azimuths)
    return np.column_stack((np.cos(e) * np.cos(beta), np.cos(e) * np.sin(beta), np.sin(e)))


def candidate_axes(candidate):
    if "elevations" in candidate:
        return top_mount_axes(candidate["elevations"], candidate["azimuths"])
    axis = mount_axis(candidate["tilt"], candidate["azimuth"])
    return np.array([axis, -axis])


def mounting_report(candidate, settings):
    axes = candidate_axes(candidate)
    elevations = np.arcsin(np.clip(axes[:, 2], -1., 1.))
    return dict(model=settings.mount_model, boresights_body=axes.tolist(),
                elevations_above_xy_deg=np.degrees(elevations).tolist(),
                tilts_from_positive_z_deg=(90 - np.degrees(elevations)).tolist(),
                azimuths_deg=np.degrees(np.arctan2(axes[:, 1], axes[:, 0]) % (2 * np.pi)).tolist(),
                separation_deg=float(np.degrees(angle_between(axes[0], axes[1]))),
                minimum_fov_edge_elevations_deg=np.degrees(elevations - settings.full_cone_fov / 2).tolist(),
                fixed_for_both_hops=True, physical_mount_positions_modeled=False,
                equivalent_pair="Swapping identical head labels; no sign-negation equivalence for top mounts")


def canonical_mount(tilt, azimuth):
    """Canonical labels for identical antipodal trackers, including pole/equator duplicates."""
    b = mount_axis(tilt, azimuth)
    if b[2] < -1e-12 or (abs(b[2]) <= 1e-12 and (b[1] < -1e-12 or
                                               (abs(b[1]) <= 1e-12 and b[0] < 0))):
        b = -b
    a = math.atan2(np.linalg.norm(b[:2]), max(0., b[2]))
    beta = math.atan2(b[1], b[0]) % (2 * math.pi) if a > 1e-12 else 0.
    return a, beta


def powered_frame(thrust, thrust_dot, heading, moon_rate=0.):
    """Right-handed body->F frame and full inertial rate, including prescribed roll.

    For each orthonormal column e, e_dot = omega_F x e, so
    omega_F = 1/2 sum(e x e_dot). Add F's inertial rotation before taking the norm.
    """
    thrust, thrust_dot = np.asarray(thrust, float), np.asarray(thrust_dot, float)
    size = np.linalg.norm(thrust, axis=-1, keepdims=True)
    if np.any(size < 1e-10):
        raise OrientationError("Zero powered thrust cannot define body orientation")
    z = thrust / size
    zd = (thrust_dot - np.sum(z * thrust_dot, axis=-1, keepdims=True) * z) / size
    hz = np.sum(heading * z, axis=-1, keepdims=True)
    raw = heading - hz * z
    raw_size = np.linalg.norm(raw, axis=-1, keepdims=True)
    if np.any(raw_size < 1e-7):
        raise OrientationError("Projected-downrange roll singularity; choose a continuous roll policy")
    x = raw / raw_size
    raw_dot = -np.sum(heading * zd, axis=-1, keepdims=True) * z - hz * zd
    xd = (raw_dot - np.sum(x * raw_dot, axis=-1, keepdims=True) * x) / raw_size
    y, yd = np.cross(z, x), np.cross(zd, x) + np.cross(z, xd)
    omega = .5 * (np.cross(x, xd) + np.cross(y, yd) + np.cross(z, zd))
    omega = omega + np.array([0., 0., moon_rate])
    return np.stack((x, y, z), axis=-1), np.linalg.norm(omega, axis=-1)


class CoastOrientation:
    """Constant-speed, shortest inertial SLERP, with deterministic exact-pi axis sign."""
    def __init__(self, first, last, duration):
        if duration <= 0:
            raise ValueError("Coast duration must be positive")
        first_r = Rotation.from_matrix(first)
        rv = (first_r.inv() * Rotation.from_matrix(last)).as_rotvec()
        self.angle = float(np.linalg.norm(rv))
        self.pi_ambiguity = abs(self.angle - math.pi) < 1e-10
        if self.pi_ambiguity:
            for v in rv:
                if abs(v) > 1e-12:
                    if v < 0:
                        rv = -rv
                    break
        endpoint = first_r * Rotation.from_rotvec(rv)
        self.interpolator = Slerp([0., duration], Rotation.from_quat([first_r.as_quat(), endpoint.as_quat()]))
        self.duration = duration
        self.rate = self.angle / duration

    def __call__(self, times):
        return self.interpolator(np.clip(times, 0, self.duration)).as_matrix()


class PhaseHistory:
    def __init__(self, p, name, ph, start, independent=False):
        self.p, self.name, self.ph, self.start = p, name, ph, float(start)
        self.end = self.start + ph["t"]
        self.knots = np.linspace(0, ph["t"], ph["X"].shape[1])
        self.rhs = make_numpy_rhs(p, mode=phase_mode(p, ph))
        if independent:
            sol = solve_ivp(lambda t, x: self.rhs(x, self.controls(np.array([t]))[0]),
                            (0., ph["t"]), ph["X"][:, 0], dense_output=True,
                            method="DOP853", rtol=1e-11, atol=1e-9)
            if not sol.success:
                raise ValueError(f"Tracker validation propagation failed: {sol.message}")
            self.state = lambda t: sol.sol(t).T
        else:
            derivatives = np.array([self.rhs(x, u) for x, u in zip(ph["X"].T, ph["U"].T)])
            spline = CubicHermiteSpline(self.knots, ph["X"].T, derivatives)
            self.state = spline
        self.coast = None
        if ph["powered"]:
            # Check closest approach to the heading singularity on every linear thrust segment.
            t0, delta = ph["U"][:3, :-1].T, np.diff(ph["U"][:3], axis=1).T
            h = p.mission.downrange
            perpendicular = t0 - np.outer(t0 @ h, h)
            dperp = delta - np.outer(delta @ h, h)
            fraction = np.clip(-np.sum(perpendicular * dperp, axis=1) /
                               np.maximum(np.sum(dperp * dperp, axis=1), 1e-30), 0, 1)
            powered_frame(t0 + fraction[:, None] * delta, delta, h, p.moon.omega)

    def controls(self, local):
        return np.stack([np.interp(local, self.knots, row) for row in self.ph["U"]], axis=-1)

    def orientation(self, times):
        local = np.clip(np.asarray(times) - self.start, 0, self.ph["t"])
        if self.coast is not None:
            return self.coast(local), np.full(local.shape, self.coast.rate)
        if not self.ph["powered"]:
            raise OrientationError("Coast orientation has not been assigned")
        k = np.clip(np.searchsorted(self.knots, local, side="right") - 1, 0, len(self.knots) - 2)
        slopes = np.diff(self.ph["U"][:3], axis=1).T / np.diff(self.knots)[:, None]
        T = self.controls(local)[..., :3]
        C, rate = powered_frame(T, slopes[k], self.p.mission.downrange, self.p.moon.omega)
        # At control knots require both one-sided rates; do not exploit derivative jumps.
        left = np.clip(np.searchsorted(self.knots, local, side="left") - 1, 0, len(self.knots) - 2)
        _, rate_left = powered_frame(T, slopes[left], self.p.mission.downrange, self.p.moon.omega)
        return rotation_z(self.p.moon.omega * np.asarray(times)) @ C, np.maximum(rate, rate_left)


@dataclass
class Samples:
    times: np.ndarray
    orientations: np.ndarray
    positions_I: np.ndarray
    body_rates: np.ndarray
    moon_radii: np.ndarray


class CoverageBlock:
    def __init__(self, name, phases, settings):
        self.name, self.phases, self.settings = name, phases, settings
        self.start, self.end = phases[0].start, phases[-1].end
        self.knots = np.unique(np.concatenate([ph.knots + ph.start for ph in phases]))
        self.warnings = []
        for first, second in zip(phases[:-1], phases[1:]):
            C0, _ = first.orientation(np.array([first.end]))
            C1, _ = second.orientation(np.array([second.start]))
            jump = Rotation.from_matrix(C0[0].T @ C1[0]).magnitude()
            if jump > 1e-7:
                raise OrientationError(f"Unresolved attitude jump at {first.name}/{second.name}: "
                                       f"{np.degrees(jump):.6g} deg")

    def phase_at(self, time):
        return next((ph.name for ph in self.phases if time <= ph.end + 1e-10), self.phases[-1].name)

    def evaluate(self, times):
        times = np.atleast_1d(times).astype(float)
        if np.any(times < self.start - 1e-8) or np.any(times > self.end + 1e-8):
            raise ValueError("Sample outside coverage block")
        C, positions, rates = np.empty((len(times), 3, 3)), np.empty((len(times), 3)), np.zeros(len(times))
        # Both adjacent phases contribute their rate at the shared boundary.
        for ph in self.phases:
            select = (times >= ph.start - 1e-10) & (times <= ph.end + 1e-10)
            if not select.any():
                continue
            t = times[select]
            c, rate = ph.orientation(t)
            C[select] = c
            rates[select] = np.maximum(rates[select], rate)
            r_F = ph.p.mission.r_start + ph.state(np.clip(t - ph.start, 0, ph.ph["t"]))[:, :3]
            positions[select] = np.einsum("nij,nj->ni", rotation_z(ph.p.moon.omega * t), r_F)
        distance = np.linalg.norm(positions, axis=1)
        radius = self.phases[0].p.moon.radius
        if np.any(distance < radius - 1e-6):
            raise ValueError("Materially subsurface tracker trajectory")
        return Samples(times, C, positions, rates, np.arcsin(np.minimum(1., radius / distance)))

    def sample(self, step):
        # Midpoints explicitly search for short windows/outages between the initial grid samples.
        base = np.unique(np.r_[np.linspace(self.start, self.end,
                                         int(np.ceil((self.end - self.start) / step)) + 1), self.knots])
        return self.evaluate(np.unique(np.r_[base, (base[:-1] + base[1:]) / 2]))


def build_blocks(p, sol, branch, independent=False):
    if p.star_tracker is None or not p.star_tracker.enabled:
        raise ValueError("Enable star_tracker for the mounting study")
    if not getattr(sol, "success", False) or not hasattr(sol, "phases"):
        raise ValueError("A successful Solution with full trajectory arrays is required; summary JSON is insufficient")
    histories, start = {}, 0.
    active = set(p.star_tracker.required_phase_kinds)
    names = chain(branch, p)
    if list(sol.phases) != names:
        raise ValueError("Full trajectory phase/stage order does not match the engine configuration")
    for name in names:
        ph = sol.phases[name]
        if ph["kind"] in active or ph["kind"] == "braking":
            histories[name] = PhaseHistory(p, name, ph, start, independent and ph["kind"] in active)
        start += ph["t"]
    blocks = []
    for suffix, label in (("", "hop1"), (f"2_{branch}", "hop2")):
        group = [histories[name] for name in names if name in histories and
                 ("2_" in name) == bool(suffix)]
        ascent = [ph for ph in group if ph.ph["kind"] == "ascent"][-1]
        coast = next(ph for ph in group if ph.ph["kind"] == "coast")
        braking = next(ph for ph in group if ph.ph["kind"] == "braking")
        first = ascent.orientation(np.array([ascent.end]))[0][0]
        last = braking.orientation(np.array([braking.start]))[0][0]
        coast.coast = CoastOrientation(first, last, coast.ph["t"])
        block = CoverageBlock(label, [ph for ph in group if ph.ph["kind"] in active],
                              p.star_tracker)
        if coast.coast.pi_ambiguity:
            block.warnings.append("Exact 180-degree coast rotation: deterministic first-nonzero-positive "
                                  "relative body-axis branch selected")
        blocks.append(block)
    return blocks


def geometry_margins(samples, axis, settings):
    """N x 2 x 3 clearances for each same tracker.

    A single unit axis retains the legacy antipodal convention. An explicit 2 x 3
    array supplies two independent fixed body boresights, used by the top-mount study.
    """
    axes = np.asarray(axis, dtype=float)
    if axes.shape == (3,):
        axes = np.stack((axes, -axes))
    if axes.shape != (2, 3) or not np.isfinite(axes).all() or not np.allclose(
            np.linalg.norm(axes, axis=1), 1., atol=1e-10, rtol=0):
        raise ValueError("Expected one unit axis or two explicit unit body boresights")
    boresights = np.einsum("nij,kj->nki", samples.orientations, axes)
    nadir = -samples.positions_I / np.linalg.norm(samples.positions_I, axis=1)[:, None]
    sun = angle_between(boresights, np.asarray(settings.sun_direction_I)) - settings.sun_exclusion
    earth = angle_between(boresights, np.asarray(settings.earth_direction_I)) - settings.earth_exclusion
    moon = (angle_between(boresights, nadir[:, None, :]) - samples.moon_radii[:, None]
            - settings.full_cone_fov / 2 - settings.moon_limb_margin)
    return np.stack((sun, earth, moon), axis=-1)


def signed_margins(samples, axis, settings, extra_margin=0.):
    geom = geometry_margins(samples, axis, settings) - extra_margin + settings.angular_tolerance
    rate = (settings.tracking_body_rate_max - samples.body_rates + settings.rate_tolerance
            if settings.enforce_body_rate else np.ones_like(samples.body_rates))
    return np.concatenate((geom, np.broadcast_to(rate[:, None, None], geom.shape[:2] + (1,))), axis=-1)


def merge_intervals(intervals, tolerance=1e-10):
    out = []
    for a, b in sorted(intervals):
        if b < a:
            continue
        if out and a <= out[-1][1] + tolerance:
            out[-1][1] = max(out[-1][1], float(b))
        else:
            out.append([float(a), float(b)])
    return out


def complement(intervals, start, end):
    out, last = [], start
    for a, b in merge_intervals(intervals):
        if a > last:
            out.append([float(last), a])
        last = max(last, b)
    if end > last:
        out.append([float(last), float(end)])
    return out


def sampled_intervals(times, margins):
    """Conservative sampled windows for search only; winning transitions are reevaluated/root-refined."""
    valid = np.all(margins >= 0, axis=-1)
    change = np.diff(np.r_[False, valid, False].astype(int))
    return [[float(times[a]), float(times[b - 1])] for a, b in
            zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1))]


def make_schedule(intervals_by_tracker, start, end, max_gap):
    """Greedy latest-event schedule over labeled closed windows; initial estimate is NOT an event."""
    intervals = merge_intervals([interval for group in intervals_by_tracker for interval in group])
    outages = complement(intervals, start, end)
    max_outage = max((b - a for a, b in outages), default=0.)
    events, current, index, failure = [], float(start), 0, {}
    # A block >= max_gap must contain a physical measurement, even if the endpoint estimate age passes.
    while end - current > max_gap + 1e-10 or (not events and end - start >= max_gap - 1e-10):
        deadline = min(current + max_gap, end)
        while index < len(intervals) and intervals[index][1] < current - 1e-10:
            index += 1
        best = None
        for a, b in intervals[index:]:
            if a > deadline + 1e-10:
                break
            candidate = min(b, deadline)
            if candidate >= current - 1e-10 and (best is None or candidate > best):
                best = candidate
        if best is None or (events and best <= current + 1e-10):
            if not failure:
                failure = dict(failure_deadline_s=float(deadline),
                               failed_outage=next(([a, b] for a, b in outages
                                                  if a <= deadline + 1e-10 and b > current), None),
                               nearest_prior_opportunity_s=max((b for a, b in intervals if b <= deadline), default=None),
                               nearest_next_opportunity_s=min((a for a, b in intervals if a > deadline), default=None))
            # Keep a labeled best-effort schedule AFTER a missed deadline, for diagnostics.
            # Its failed cadence is retained; later valid events cannot erase the violation.
            best = min((a for a, b in intervals if a > deadline + 1e-10), default=None)
            if best is None:
                break
        tracker = next(j + 1 for j, group in enumerate(intervals_by_tracker)
                       if any(a - 1e-10 <= best <= b + 1e-10 for a, b in group))
        events.append(dict(time_s=float(best), tracker=tracker))
        current = best
    first = events[0]["time_s"] - start if events else None
    final = end - current
    inter = max((b["time_s"] - a["time_s"] for a, b in zip(events[:-1], events[1:])), default=0.)
    return dict(passed=not failure, events=events, first_update_latency_s=first,
                largest_inter_update_gap_s=float(inter), final_update_age_s=float(final),
                max_update_gap_s=float(max(first or 0., inter, final)), event_count=len(events),
                initialized_estimate_only=not events, max_unavailable_interval_s=float(max_outage), **failure)


def cadence_feasible(intervals_by_tracker, start, end, max_gap):
    """Exact window-based feasibility under zero dwell, without constructing search-time events.

    Each open outage is bounded by usable endpoints (or the initialized estimate / block end).
    Thus its length must fit within one update gap. A long block also needs a real event.
    """
    union = merge_intervals([v for group in intervals_by_tracker for v in group])
    outages = complement(union, start, end)
    worst = max((b - a for a, b in outages), default=0.)
    actual_event_needed = end - start >= max_gap - 1e-10
    return worst <= max_gap + 1e-10 and (bool(union) or not actual_event_needed), worst


def refine_intervals(block, axis, step, extra_margin=0.):
    """Find roots of EACH signed margin, then classify open cells independently.

    Midpoint-enhanced sampling searches for sub-grid windows. Later half-step comparison
    tests numerical convergence; this is deliberately not a continuous-time proof.
    """
    settings = block.settings
    samples = block.sample(step)
    values = signed_margins(samples, axis, settings, extra_margin)
    by_tracker = []
    for j in range(2):
        cuts = list(block.knots)
        for k in range(4):
            v = values[:, j, k]
            cross = np.flatnonzero((v[:-1] < 0) != (v[1:] < 0))
            for i in cross:
                a, b = samples.times[i:i + 2]
                # Shrink slightly at one-sided rate discontinuities/control knots; these
                # cannot be interpolated as smooth roots across a derivative jump.
                def scalar(t):
                    return signed_margins(block.evaluate([t]), axis, settings, extra_margin)[0, j, k]
                root = brentq(scalar, a, b, xtol=min(settings.transition_tolerance / 100, 1e-6))
                cuts.append(root)
        cuts = np.unique(np.r_[block.start, cuts, block.end])
        mid = (cuts[:-1] + cuts[1:]) / 2
        vmid = signed_margins(block.evaluate(mid), axis, settings, extra_margin)[:, j]
        good = np.all(vmid >= 0, axis=1)
        intervals = []
        for a, b in zip(cuts[:-1][good], cuts[1:][good]):
            # Stay inside each measured valid region. Recheck every eventual event exactly.
            pad = min(settings.transition_tolerance / 10, (b - a) / 10)
            edge = signed_margins(block.evaluate([a, b]), axis, settings, extra_margin)[:, j]
            intervals.append([a if np.all(edge[0] >= 0) else a + pad,
                              b if np.all(edge[1] >= 0) else b - pad])
        # Do not bridge a forbidden isolated knot (important for derivative jumps).
        by_tracker.append(merge_intervals(intervals))
    return by_tracker, samples


def evaluate_schedule(block, axis, intervals, extra_margin=0.):
    schedule = make_schedule(intervals, block.start, block.end, block.settings.update_gap_max)
    settings = block.settings
    for event in schedule["events"]:
        t, j = event["time_s"], event["tracker"] - 1
        sample = block.evaluate([t])
        margins = geometry_margins(sample, axis, settings)[0, j]
        valid = (np.all(margins >= extra_margin - settings.angular_tolerance) and
                 (not settings.enforce_body_rate or
                  sample.body_rates[0] <= settings.tracking_body_rate_max + settings.rate_tolerance))
        event.update(phase=block.phase_at(t), angular_margins_deg=dict(zip(
            ("sun", "earth", "moon_limb"), np.degrees(margins).tolist())),
                     body_rate_deg_s=float(np.degrees(sample.body_rates[0])), valid=bool(valid))
        if not valid:
            schedule["passed"] = False
            schedule["invalid_event"] = event.copy()
    schedule["start_s"], schedule["end_s"] = block.start, block.end
    schedule["body_rate_limit_enforced"] = settings.enforce_body_rate
    schedule["unavailable_intervals_s"] = {str(j + 1): complement(group, block.start, block.end)
                                            for j, group in enumerate(intervals)}
    union = merge_intervals([v for group in intervals for v in group])
    outages = complement(union, block.start, block.end)
    schedule["both_unavailable_intervals_s"] = outages
    schedule["availability_fraction"] = sum(b - a for a, b in union) / (block.end - block.start)
    schedule["worst_outage_causes"] = []
    if outages:
        a, b = max(outages, key=lambda v: v[1] - v[0])
        mid = (a + b) / 2
        sample = block.evaluate([mid])
        margins = signed_margins(sample, axis, settings)[0]
        for j in range(2):
            schedule["worst_outage_causes"].append(dict(
                tracker=j + 1, midpoint_s=mid, phase=block.phase_at(mid),
                failed_tests=[name for name, val in zip(("sun", "earth", "moon_limb", "body_rate"),
                                                        margins[j]) if val < 0]))
    if schedule["events"]:
        events = schedule["events"]
        limiting = min(((v, e, key) for e in events for key, v in e["angular_margins_deg"].items()),
                       key=lambda x: x[0])
        schedule["limiting_event"] = dict(margin_deg=limiting[0], time_s=limiting[1]["time_s"],
                                           tracker=limiting[1]["tracker"], phase=limiting[1]["phase"],
                                           exclusion=limiting[2])
        schedule["worst_accepted_body_rate_deg_s"] = max(e["body_rate_deg_s"] for e in events)
    return schedule


def rate_precheck(blocks, step):
    reports = []
    for block in blocks:
        samples = block.sample(step)
        settings = block.settings
        valid = (settings.tracking_body_rate_max - samples.body_rates + settings.rate_tolerance)[:, None]
        intervals = sampled_intervals(samples.times, valid)
        schedule = make_schedule([intervals], block.start, block.end, settings.update_gap_max)

        def refine_rate(spacing):
            data = block.sample(spacing)
            values = settings.tracking_body_rate_max + settings.rate_tolerance - data.body_rates
            cuts = list(block.knots)
            def margin(t):
                return settings.tracking_body_rate_max + settings.rate_tolerance - block.evaluate([t]).body_rates[0]
            for i in np.flatnonzero((values[:-1] < 0) != (values[1:] < 0)):
                cuts.append(brentq(margin, *data.times[i:i + 2],
                                   xtol=min(settings.transition_tolerance / 100, 1e-6)))
            cuts = np.unique(np.r_[block.start, cuts, block.end])
            middle = block.evaluate((cuts[:-1] + cuts[1:]) / 2).body_rates
            good = middle <= settings.tracking_body_rate_max + settings.rate_tolerance
            windows = []
            for a, b in zip(cuts[:-1][good], cuts[1:][good]):
                pad = min(settings.transition_tolerance / 10, (b - a) / 10)
                windows.append([a if margin(a) >= 0 else a + pad, b if margin(b) >= 0 else b - pad])
            return merge_intervals(windows)

        refined, finer = refine_rate(step / 2), refine_rate(step / 4)
        resolved = (len(refined) == len(finer) and
                    all(abs(a - b) <= settings.transition_tolerance
                        for aa, bb in zip(refined, finer) for a, b in zip(aa, bb)))
        refined_schedule = make_schedule([finer], block.start, block.end, settings.update_gap_max)
        reports.append(dict(block=block.name, sampled_rate_only_schedule_passed=schedule["passed"],
                            body_rate_limit_enforced=settings.enforce_body_rate,
                            refined_rate_only_schedule_passed=refined_schedule["passed"],
                            numerically_resolved=bool(resolved),
                            max_sampled_rate_deg_s=float(np.degrees(max(samples.body_rates))),
                            unavailable_intervals_s=complement(finer, block.start, block.end),
                            max_rate_only_outage_s=refined_schedule["max_unavailable_interval_s"],
                            first_rate_allowed_update_latency_s=refined_schedule["first_update_latency_s"],
                            note=("Mount-independent analytic full-frame rate, roots refined and half-step checked; "
                                  "a failed resolved rate-only schedule cannot be repaired by remounting"
                                  if settings.enforce_body_rate else
                                  "Diagnostic only: the user waived the rate gate assuming active attitude control; "
                                  "these reference-rate outages do not reject tracker events")))
    return reports


def _search_antipodal_mount(blocks):
    """Coarse hemisphere + bounded derivative-free refinement of robust cadence margin."""
    settings = blocks[0].settings
    cached = [b.sample(settings.sample_step) for b in blocks]
    candidates = {}

    def assess(angles):
        tilt, beta = map(float, angles)
        if not (settings.tilt_bounds[0] <= tilt <= settings.tilt_bounds[1] and
                settings.azimuth_bounds[0] <= beta <= settings.azimuth_bounds[1]):
            return -1e6
        axis = mount_axis(tilt, beta)
        geometry = [geometry_margins(s, axis, settings).min(axis=-1) for s in cached]
        eligible_rate = [(s.body_rates <= settings.tracking_body_rate_max + settings.rate_tolerance
                          if settings.enforce_body_rate else np.ones(len(s.times), dtype=bool)) for s in cached]

        def feasible(extra):
            worst, failures = 0., 0
            for block, s, g, rate in zip(blocks, cached, geometry, eligible_rate):
                windows = [sampled_intervals(s.times, np.c_[g[:, j] - extra + settings.angular_tolerance,
                                                          np.where(rate, 1., -1.)]) for j in range(2)]
                passed, outage = cadence_feasible(windows, block.start, block.end, settings.update_gap_max)
                worst = max(worst, outage)
                failures += not passed
            return failures == 0, failures, worst

        passed, failures, worst = feasible(0.)
        if passed:
            low, high = 0., math.pi
            for _ in range(13):
                mid = (low + high) / 2
                if feasible(mid)[0]:
                    low = mid
                else:
                    high = mid
            score = low
        else:
            score = -1. - failures - worst / settings.update_gap_max
        candidates[(tilt, beta)] = dict(tilt=tilt, azimuth=beta, sampled_passed=passed,
                                       sampled_robust_margin=low if passed else None, score=score)
        return score

    tilts = np.unique(np.r_[np.arange(*settings.tilt_bounds, settings.coarse_step), settings.tilt_bounds[1]])
    betas = np.arange(*settings.azimuth_bounds, settings.coarse_step)
    seen = set()
    for tilt in tilts:
        for beta in betas:
            key = tuple(round(v, 10) for v in canonical_mount(tilt, beta))
            if key not in seen:
                assess((tilt, beta))
                seen.add(key)
    coarse_count = len(candidates)
    best = sorted(candidates.values(), key=lambda c: c["score"], reverse=True)[:3]
    for candidate in best:
        minimize(lambda x: -assess(x), [candidate["tilt"], candidate["azimuth"]], method="Powell",
                 bounds=[settings.tilt_bounds, settings.azimuth_bounds],
                 options={"maxiter": 8, "maxfev": 100, "xtol": math.radians(.1), "ftol": 1e-4})
    ranked = sorted(candidates.values(), key=lambda c: c["score"], reverse=True)
    return ranked, dict(method="hemisphere grid then bounded Powell derivative-free refinement (top 3)",
                        coarse_step_deg=float(np.degrees(settings.coarse_step)), coarse_count=coarse_count,
                        tested_count=len(candidates), robustness_bisection_resolution_deg=180 / 2**13,
                        global_optimality_proven=False)


def search_mount(blocks):
    """Find one global pair, supporting historical antipodal and independent-top modes."""
    settings = blocks[0].settings
    if settings.mount_model == "antipodal":
        return _search_antipodal_mount(blocks)
    cached = [b.sample(settings.sample_step) for b in blocks]
    bounds = [settings.elevation_bounds[0], settings.paired_azimuth_bounds[0],
              settings.elevation_bounds[1], settings.paired_azimuth_bounds[1]]
    candidates = {}

    def assess(values):
        values = np.asarray(values, dtype=float)
        if any(not lo <= v <= hi for v, (lo, hi) in zip(values, bounds)):
            return -1e6
        candidate = dict(elevations=values[[0, 2]].tolist(), azimuths=values[[1, 3]].tolist())
        axes = candidate_axes(candidate)
        geometry = [geometry_margins(s, axes, settings).min(axis=-1) for s in cached]

        def feasible(extra):
            worst, failures = 0., 0
            for block, sample, g in zip(blocks, cached, geometry):
                eligible_rate = (sample.body_rates <= settings.tracking_body_rate_max + settings.rate_tolerance
                                 if settings.enforce_body_rate else np.ones(len(sample.times), dtype=bool))
                windows = [sampled_intervals(sample.times, np.c_[g[:, j] - extra + settings.angular_tolerance,
                                                               np.where(eligible_rate, 1., -1.)]) for j in range(2)]
                passed, outage = cadence_feasible(windows, block.start, block.end, settings.update_gap_max)
                worst = max(worst, outage)
                failures += not passed
            return failures == 0, failures, worst

        passed, failures, worst = feasible(0.)
        low = 0.
        if passed:
            high = np.pi
            for _ in range(13):
                middle = (low + high) / 2
                if feasible(middle)[0]:
                    low = middle
                else:
                    high = middle
            score = low
        else:
            score = -1. - failures - worst / settings.update_gap_max
        candidate.update(sampled_passed=passed, sampled_robust_margin=low if passed else None, score=score)
        candidates[tuple(values)] = candidate
        return score

    initial = np.array([settings.initial_elevations[0], settings.initial_azimuths[0],
                        settings.initial_elevations[1], settings.initial_azimuths[1]])
    assess(initial)
    for index in range(4):
        for delta in (-settings.coarse_step, settings.coarse_step):
            trial = initial.copy()
            trial[index] = np.clip(trial[index] + delta, *bounds[index])
            assess(trial)
    # Deterministic stratified multistart; no antipodal symmetry reduction is applicable.
    rng = np.random.default_rng(42)
    for unit in rng.random((48, 4)):
        assess([lo + f * (hi - lo) for f, (lo, hi) in zip(unit, bounds)])
    coarse_count = len(candidates)
    seeds = sorted(candidates.values(), key=lambda c: c["score"], reverse=True)[:3]
    for c in seeds:
        x0 = [c["elevations"][0], c["azimuths"][0], c["elevations"][1], c["azimuths"][1]]
        minimize(lambda x: -assess(x), x0, method="Powell", bounds=bounds,
                 options={"maxiter": 6, "maxfev": 120, "xtol": math.radians(.1), "ftol": 1e-4})
    return sorted(candidates.values(), key=lambda c: c["score"], reverse=True), dict(
        method="verified seed + perturbations + deterministic multistart, bounded Powell on four global angles",
        mount_model="independent_top", coarse_count=coarse_count, tested_count=len(candidates),
        robustness_bisection_resolution_deg=180 / 2**13, global_optimality_proven=False)


def validate_mount(blocks, independent_blocks, candidate, extra_margin=0., *, body_axes=None):
    """Independent propagation + finer grids + direct event checks; no sample-only guarantee."""
    settings = blocks[0].settings
    axis = (np.asarray(body_axes, dtype=float) if body_axes is not None else
            candidate_axes(candidate))
    pair = axis if axis.shape == (2, 3) else np.array([axis, -axis])
    plane_clearance = np.arcsin(np.clip(pair[:, 2], -1., 1.)) - settings.full_cone_fov / 2
    plane_ok = (settings.mount_model != "independent_top" or
                np.all(plane_clearance >= settings.top_plane_clearance_margin - settings.angular_tolerance))
    reports = {}
    for block, independent in zip(blocks, independent_blocks):
        coarse_windows, _ = refine_intervals(block, axis, settings.sample_step, extra_margin)
        windows, samples = refine_intervals(independent, axis, settings.sample_step / 2, extra_margin)
        finer_windows, finer_samples = refine_intervals(independent, axis, settings.sample_step / 4, extra_margin)
        coarse = evaluate_schedule(block, axis, coarse_windows, extra_margin)
        fine = evaluate_schedule(independent, axis, windows, extra_margin)
        final = evaluate_schedule(independent, axis, finer_windows, extra_margin)
        def difference(a, b):
            if len(a) != len(b):
                return float("inf")
            return max((abs(x - y) for aa, bb in zip(a, b) for x, y in zip(aa, bb)), default=0.)
        interval_change = max(difference(a, b) for a, b in zip(windows, finer_windows))
        agreement = (coarse["passed"] == fine["passed"] == final["passed"] and
                     interval_change <= settings.transition_tolerance and
                     abs(fine["max_unavailable_interval_s"] - final["max_unavailable_interval_s"])
                     <= settings.transition_tolerance)
        # Independent finite-rotation rate estimate checks the analytic full-frame derivative.
        rate_error = 0.
        probe = finer_samples.times
        epsilon = min(1e-4, settings.transition_tolerance / 10)
        clear = np.min(abs(probe[:, None] - independent.knots), axis=1) > 2 * epsilon
        probe = probe[clear][::max(1, np.count_nonzero(clear) // 1000)]
        if len(probe):
            first, last = independent.evaluate(probe - epsilon), independent.evaluate(probe + epsilon)
            relative = np.swapaxes(first.orientations, -1, -2) @ last.orientations
            numeric_rate = Rotation.from_matrix(relative).magnitude() / (2 * epsilon)
            exact = independent.evaluate(probe).body_rates
            rate_error = float(np.max(abs(numeric_rate - exact)))
        agreement = agreement and rate_error <= math.radians(.001)
        final["passed"] = bool(final["passed"] and plane_ok)
        final.update(numerically_resolved=bool(agreement),
                     top_plane_clearance_passed=bool(plane_ok),
                     minimum_fov_edge_elevations_deg=np.degrees(plane_clearance).tolist(),
                     required_extra_angular_margin_deg=float(np.degrees(extra_margin)),
                     status=("numerically_validated" if final["passed"] else "tested_mount_infeasible")
                     if agreement else "unresolved",
                     boundary_change_on_refinement_s=float(interval_change) if np.isfinite(interval_change) else None,
                     max_rate_crosscheck_error_deg_s=float(np.degrees(rate_error)),
                     validation_sample_step_s=settings.sample_step / 4,
                     midpoint_enhanced_max_spacing_s=settings.sample_step / 8,
                     transition_time_tolerance_s=settings.transition_tolerance,
                     warnings=independent.warnings)
        reports[block.name] = final
    return reports

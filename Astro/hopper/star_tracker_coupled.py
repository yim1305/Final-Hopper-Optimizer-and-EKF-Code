"""Local joint flight/mount NLP with fixed labeled update cells and a real-time cadence.

This subclass adds four global mount angles and one head's geometry at each
assigned event. The optional full-frame inertial rate gate is also enforced.
"""
from dataclasses import dataclass, replace
import time

import casadi as ca
import numpy as np

from .problem import Problem, Solution
from .star_tracker import (build_blocks, candidate_axes, evaluate_schedule, geometry_margins, refine_intervals)


def symbolic_rotation_z(angle):
    c, s = ca.cos(angle), ca.sin(angle)
    return ca.vertcat(ca.horzcat(c, -s, 0), ca.horzcat(s, c, 0), ca.horzcat(0, 0, 1))


def symbolic_powered_frame(thrust, heading):
    z = thrust / ca.sqrt(ca.fmax(ca.dot(thrust, thrust), 1e-20))
    raw = heading - ca.dot(heading, z) * z
    x = raw / ca.sqrt(ca.fmax(ca.dot(raw, raw), 1e-20))
    return ca.horzcat(x, ca.cross(z, x), z)


def symbolic_body_rate_squared(thrust, thrust_dot, heading, moon_rate):
    """Analytic full-frame inertial rate, matching star_tracker.powered_frame."""
    size = ca.sqrt(ca.fmax(ca.dot(thrust, thrust), 1e-20))
    z = thrust / size
    zd = (thrust_dot - ca.dot(z, thrust_dot) * z) / size
    hz = ca.dot(heading, z)
    raw = heading - hz * z
    raw_size = ca.sqrt(ca.fmax(ca.dot(raw, raw), 1e-20))
    x = raw / raw_size
    raw_dot = -ca.dot(heading, zd) * z - hz * zd
    xd = (raw_dot - ca.dot(x, raw_dot) * x) / raw_size
    y, yd = ca.cross(z, x), ca.cross(zd, x) + ca.cross(z, xd)
    omega = .5 * (ca.cross(x, xd) + ca.cross(y, yd) + ca.cross(z, zd))
    omega += ca.vertcat(0, 0, moon_rate)
    return ca.dot(omega, omega)


def quaternion_from_matrix(R):
    """Scalar-first unit quaternion via largest-component branches, including near pi."""
    tr = ca.trace(R)
    d = [1 + tr, 1 + 2 * R[0, 0] - tr, 1 + 2 * R[1, 1] - tr, 1 + 2 * R[2, 2] - tr]
    q = []
    for k in range(4):
        a = .5 * ca.sqrt(ca.fmax(d[k], 1e-20))
        divisor = 4 * a
        if k == 0:
            value = ca.vertcat(a, (R[2, 1] - R[1, 2]) / divisor,
                              (R[0, 2] - R[2, 0]) / divisor, (R[1, 0] - R[0, 1]) / divisor)
        elif k == 1:
            value = ca.vertcat((R[2, 1] - R[1, 2]) / divisor, a,
                              (R[0, 1] + R[1, 0]) / divisor, (R[0, 2] + R[2, 0]) / divisor)
        elif k == 2:
            value = ca.vertcat((R[0, 2] - R[2, 0]) / divisor,
                              (R[0, 1] + R[1, 0]) / divisor, a, (R[1, 2] + R[2, 1]) / divisor)
        else:
            value = ca.vertcat((R[1, 0] - R[0, 1]) / divisor,
                              (R[0, 2] + R[2, 0]) / divisor, (R[1, 2] + R[2, 1]) / divisor, a)
        q.append(value)
    chosen = q[0]
    largest = d[0]
    for k in range(1, 4):
        chosen = ca.if_else(d[k] > largest, q[k], chosen)
        largest = ca.fmax(largest, d[k])
    chosen = chosen / ca.sqrt(ca.dot(chosen, chosen))
    return ca.if_else(chosen[0] < 0, -chosen, chosen)


def matrix_from_quaternion(q):
    q = q / ca.sqrt(ca.dot(q, q))
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ca.vertcat(ca.horzcat(1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
                      ca.horzcat(2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
                      ca.horzcat(2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)))


def symbolic_slerp_data(first, last):
    """Shared endpoint work: build once per coast, not once per measurement event."""
    q = quaternion_from_matrix(first.T @ last)
    t = ca.dot(q[1:4], q[1:4])
    r = ca.sqrt(ca.fmax(t, 1e-20))
    phi = ca.atan2(r, q[0])
    return first, q, t, r, phi


def symbolic_slerp_from_data(data, fraction):
    first, q, t, r, phi = data
    f = fraction
    # Series in sin(phi)^2 remove zero-angle derivative singularities.
    ratio = ca.if_else(t < 1e-8, f + f * (1 - f*f) * t / 6 +
                      f * (9 - 10*f*f + f**4) * t*t / 120, ca.sin(f * phi) / r)
    scalar = ca.if_else(t < 1e-8, 1 - f*f * t / 2 + (f**4 / 24 - f*f / 6) * t*t,
                        ca.cos(f * phi))
    return first @ matrix_from_quaternion(ca.vertcat(scalar, ratio * q[1:4]))


def symbolic_slerp(first, last, fraction):
    return symbolic_slerp_from_data(symbolic_slerp_data(first, last), fraction)


def symbolic_mount_axes(elevations, azimuths):
    return [ca.vertcat(ca.cos(elevations[j]) * ca.cos(azimuths[j]),
                       ca.cos(elevations[j]) * ca.sin(azimuths[j]), ca.sin(elevations[j])) for j in range(2)]


@dataclass(frozen=True)
class AssignedEvent:
    phase: str
    cell: int
    fraction: float
    tracker: int
    block: str


def prepare_schedule(p, sol, branch, candidate, gap_fraction=.9, extra_margin_deg=10.):
    """Verified schedule with time slack; gap_fraction is a seed choice, not an acceptance limit."""
    axes = candidate_axes(candidate)
    seeded = replace(p, star_tracker=replace(p.star_tracker,
                                             update_gap_max=p.star_tracker.update_gap_max * gap_fraction))
    blocks = build_blocks(seeded, sol, branch, independent=True)
    events, reports = [], {}
    for block in blocks:
        windows, _ = refine_intervals(block, axes, p.star_tracker.sample_step / 2, np.radians(extra_margin_deg))
        report = evaluate_schedule(block, axes, windows, np.radians(extra_margin_deg))
        if not report["passed"]:
            raise ValueError(f"No feasible labeled seed schedule for {block.name}")
        reports[block.name] = report
        for event in report["events"]:
            name = event["phase"]
            history = next(ph for ph in block.phases if ph.name == name)
            n = sol.phases[name]["X"].shape[1] - 1
            location = np.clip((event["time_s"] - history.start) / sol.phases[name]["t"] * n, 0, n)
            cell = min(int(np.floor(location)), n - 1)
            events.append(AssignedEvent(name, cell, float(location - cell), event["tracker"], block.name))
    return events, reports


class TrackerProblem(Problem):
    def __init__(self, p, branch, events, mount_seed, *, fixed_mount=False, log_progress=False, **kwargs):
        if p.star_tracker is None or p.star_tracker.mount_model != "independent_top":
            raise ValueError("Coupled tracker solve requires independent_top mounting")
        self.assigned_events = list(events)
        self.mount_seed, self.fixed_mount = mount_seed, fixed_mount
        self.iteration_log = []
        self.has_dual_seed = False
        self._callback_start = time.monotonic()
        super().__init__(p, branch, **kwargs)
        if log_progress:
            self.opti.callback(self._record_iteration)

    def _record_iteration(self, iteration):
        if iteration % 100:
            return
        g = np.asarray(self.opti.debug.value(self.opti.g)).ravel()
        lo, hi = (np.asarray(self.opti.debug.value(v)).ravel() for v in (self.opti.lbg, self.opti.ubg))
        violation = float(np.max(np.maximum(np.maximum(lo - g, g - hi), 0.)))
        initial_mass = float(self.opti.debug.value(self.hop1[0].X[6, 0])) * self.s.M
        self.iteration_log.append(dict(iteration=int(iteration), max_primal_violation=violation,
                                       wet_mass_kg=initial_mass, elapsed_s=time.monotonic() - self._callback_start))
        print(f"  joint iter {iteration}: wet mass {initial_mass:.6f} kg, "
              f"max primal violation {violation:.3e}", flush=True)

    def initial_violation(self):
        initial = self.opti.initial()
        g = np.asarray(self.opti.debug.value(self.opti.g, initial)).ravel()
        lo, hi = (np.asarray(self.opti.debug.value(v, initial)).ravel() for v in (self.opti.lbg, self.opti.ubg))
        return float(np.max(np.maximum(np.maximum(lo - g, g - hi), 0.)))

    def _build(self):
        super()._build()
        self.base_constraint_count = int(self.opti.ng)
        p, settings, opti = self.p, self.p.star_tracker, self.opti
        self.mount_elevations, self.mount_azimuths = opti.variable(2), opti.variable(2)
        for j in range(2):
            opti.subject_to(opti.bounded(*[settings.elevation_bounds[j][0], self.mount_elevations[j],
                                          settings.elevation_bounds[j][1]]))
            opti.subject_to(opti.bounded(settings.paired_azimuth_bounds[j][0], self.mount_azimuths[j],
                                         settings.paired_azimuth_bounds[j][1]))
            opti.subject_to(self.mount_elevations[j] >= settings.full_cone_fov / 2 +
                            settings.top_plane_clearance_margin)
        if self.fixed_mount:
            opti.subject_to(self.mount_elevations == ca.DM(self.mount_seed["elevations"]))
            opti.subject_to(self.mount_azimuths == ca.DM(self.mount_seed["azimuths"]))
        opti.set_initial(self.mount_elevations, self.mount_seed["elevations"])
        opti.set_initial(self.mount_azimuths, self.mount_seed["azimuths"])
        axes = symbolic_mount_axes(self.mount_elevations, self.mount_azimuths)
        phases = {ph.name: ph for ph in self.phases}
        indices = {ph.name: i for i, ph in enumerate(self.phases)}
        durations = ca.vertcat(*[ph.tdur * self.s.T for ph in self.phases])
        self.duration_vector = durations
        starts, current = {}, 0.
        for ph in self.phases:
            starts[ph.name] = current
            current += ph.tdur * self.s.T
        heading = ca.DM(p.mission.downrange)
        coast_endpoints = {}
        for hop in (self.hop1, self.hop2):
            ascent = [ph for ph in hop if ph.kind == "ascent"][-1]
            coast = next(ph for ph in hop if ph.kind == "coast")
            braking = next(ph for ph in hop if ph.kind == "braking")
            C0 = symbolic_rotation_z(p.moon.omega * starts[coast.name]) @ symbolic_powered_frame(ascent.U[:3, -1], heading)
            C1 = symbolic_rotation_z(p.moon.omega * starts[braking.name]) @ symbolic_powered_frame(braking.U[:3, 0], heading)
            coast_endpoints[coast.name] = symbolic_slerp_data(C0, C1)
            if settings.enforce_body_rate:
                rate = 2 * coast_endpoints[coast.name][-1] / (coast.tdur * self.s.T)
                opti.subject_to(rate <= settings.tracking_body_rate_max)
        derivatives = {name: self.dynamics_of(ph).map(ph.n + 1)(ph.X, ph.U) for name, ph in phases.items()
                       if ph.kind in settings.required_phase_kinds}
        self.event_times, self.event_residuals = [], []
        coefficient_groups = {"hop1": [], "hop2": []}
        R = p.moon.radius / self.s.L
        fov_clear = settings.full_cone_fov / 2 + settings.moon_limb_margin
        for event in self.assigned_events:
            ph, k, u = phases[event.phase], event.cell, event.fraction
            if (event.tracker not in (1, 2) or event.block not in coefficient_groups or
                    ph.kind not in settings.required_phase_kinds or not (0 <= k < ph.n and 0 <= u <= 1)):
                raise ValueError("Event assignment outside active phase/collocation cell")
            fraction = (k + u) / ph.n
            t = starts[ph.name] + fraction * ph.tdur * self.s.T
            X0, X1 = ph.X[:, k], ph.X[:, k + 1]
            F0, F1 = derivatives[ph.name][:, k], derivatives[ph.name][:, k + 1]
            x = ((2*u**3 - 3*u*u + 1) * X0 + (u**3 - 2*u*u + u) * ph.tdur / ph.n * F0 +
                 (-2*u**3 + 3*u*u) * X1 + (u**3 - u*u) * ph.tdur / ph.n * F1)
            if ph.powered:
                thrust = (1 - u) * ph.U[:3, k] + u * ph.U[:3, k + 1]
                C = symbolic_rotation_z(p.moon.omega * t) @ symbolic_powered_frame(thrust, heading)
                if settings.enforce_body_rate:
                    cells = [k]
                    if u == 0 and k > 0:
                        cells.append(k - 1)
                    if u == 1 and k + 1 < ph.n:
                        cells.append(k + 1)
                    for cell in cells:
                        slope = (ph.U[:3, cell + 1] - ph.U[:3, cell]) / (ph.tdur * self.s.T / ph.n)
                        rate_sq = symbolic_body_rate_squared(thrust, slope, heading, p.moon.omega)
                        opti.subject_to(rate_sq / settings.tracking_body_rate_max**2 <= 1)
            else:
                C = symbolic_slerp_from_data(coast_endpoints[ph.name], fraction)
            b = C @ axes[event.tracker - 1]
            r_F = ca.DM(p.mission.r_start / self.s.L) + x[:3]
            radius = ca.norm_2(r_F)
            nadir_I = symbolic_rotation_z(p.moon.omega * t) @ (-r_F / radius)
            q = R / radius
            # Safe trial evaluation outside the feasible surface; the explicit radius
            # guard forbids subsurface points. On valid points this is cos(asin(q)+FOV).
            moon_cos = ca.sqrt(ca.fmax(1 - q*q, 1e-16)) * np.cos(fov_clear) - q * np.sin(fov_clear)
            residual = ca.vertcat(ca.dot(b, ca.DM(settings.sun_direction_I)) - np.cos(settings.sun_exclusion),
                                 ca.dot(b, ca.DM(settings.earth_direction_I)) - np.cos(settings.earth_exclusion),
                                 ca.dot(b, nadir_I) - moon_cos, R - radius)
            opti.subject_to(residual <= 0)
            self.event_times.append(t)
            self.event_residuals.append(residual)
            coeff = np.zeros(len(self.phases))
            coeff[:indices[ph.name]] = 1.
            coeff[indices[ph.name]] = fraction
            coefficient_groups[event.block].append(coeff)
        # Compress duplicate dimensional gap inequalities (regular event spacings)
        # rather than presenting hundreds of identical active bounds to IPOPT.
        gap_coefficients = {}
        for label, hop in (("hop1", self.hop1), ("hop2", self.hop2)):
            active = [ph for ph in hop if ph.kind in settings.required_phase_kinds]
            coefficients = coefficient_groups[label]
            if not coefficients:
                raise ValueError("Each coverage block requires an actual tracker event")
            first, last = np.zeros(len(self.phases)), np.zeros(len(self.phases))
            first[:indices[active[0].name]] = 1.
            last[:indices[active[-1].name] + 1] = 1.
            all_coeff = [first] + coefficients + [last]
            for a, b in zip(all_coeff[:-1], all_coeff[1:]):
                diff = b - a
                gap_coefficients[tuple(np.round(diff, 12))] = diff
        for coeff in gap_coefficients.values():
            opti.subject_to(ca.dot(ca.DM(coeff), durations) / settings.update_gap_max <= 1.)
        self.unique_gap_constraint_count = len(gap_coefficients)

    def set_dual_seed(self, baseline_duals):
        duals = np.asarray(baseline_duals).reshape(-1)
        if len(duals) != self.base_constraint_count:
            raise ValueError("Baseline multiplier count does not match the inherited flight model")
        self.opti.set_initial(self.opti.lam_g, np.r_[duals, np.zeros(self.opti.ng - len(duals))])
        self.has_dual_seed = True

    def solve(self, print_level=0):
        """Primal/dual warm-started IPOPT; keep tight feasibility, allow stated local optimality tolerance."""
        tolerance = float(self.p.solver.ipopt["tol"])
        options = dict(max_iter=int(self.p.solver.ipopt["max_iter"]), tol=tolerance,
                       constr_viol_tol=tolerance, acceptable_iter=1, acceptable_tol=max(1e-6, tolerance),
                       acceptable_constr_viol_tol=tolerance, acceptable_dual_inf_tol=1e-6,
                       acceptable_compl_inf_tol=1e-6, bound_relax_factor=0, print_level=print_level, sb="yes",
                       max_wall_time=float(self.p.solver.ipopt.get("max_wall_time", 60.)),
                       mu_strategy="adaptive", mu_init=1e-9,
                       warm_start_init_point="yes" if self.has_dual_seed else "no",
                       warm_start_bound_push=1e-14, warm_start_bound_frac=1e-14,
                       warm_start_slack_bound_push=1e-14, warm_start_slack_bound_frac=1e-14,
                       warm_start_mult_bound_push=1e-12, warm_start_mult_init_max=1e12)
        self.opti.solver("ipopt", {"expand": True, "print_time": False}, options)
        success = True
        try:
            raw = self.opti.solve()
            value = raw.value
        except RuntimeError:
            success = False
            value = self.opti.debug.value
        stats = self.opti.stats()
        self.solve_statistics = stats
        phases = {}
        scale = np.array([self.s.L] * 3 + [self.s.V] * 3 + [self.s.M])[:, None]
        for ph in self.phases:
            phases[ph.name] = dict(X=np.asarray(value(ph.X)) * scale,
                                   U=np.asarray(value(ph.U)) * self.s.F if ph.powered else np.zeros((4, ph.n + 1)),
                                   t=float(value(ph.tdur)) * self.s.T, powered=ph.powered,
                                   kind=ph.kind, branch=ph.branch, stage=ph.stage, relaxed=self.relaxed, hop=ph.hop)
        status = stats["return_status"]
        hist = stats.get("iterations", {})
        primal = float(hist.get("inf_pr", [float("inf")])[-1])
        dual = float(hist.get("inf_du", [float("inf")])[-1])
        return Solution(phases, float(value(self.final.X[6, -1])) * self.s.M, status,
                        success and status in ("Solve_Succeeded", "Solved_To_Acceptable_Level") and
                        max(primal, dual) <= max(1e-6, 100 * tolerance),
                        int(stats.get("iter_count", -1)), float(stats.get("t_wall_total", float("nan"))),
                        float(value(self.mw)) * self.s.M if self.mode == "size" else self.fixed_m_wet,
                        primal, dual)

    def mounting_candidate(self):
        return dict(elevations=np.asarray(self.opti.value(self.mount_elevations)).reshape(2).tolist(),
                    azimuths=np.asarray(self.opti.value(self.mount_azimuths)).reshape(2).tolist())

    def actual_events(self):
        times = np.asarray(self.opti.value(ca.vertcat(*self.event_times))).reshape(-1)
        return [dict(time_s=float(t), tracker=e.tracker, phase=e.phase, block=e.block,
                     cell=e.cell, fraction=e.fraction) for t, e in zip(times, self.assigned_events)]


def validate_assigned_events(p, sol, branch, candidate, actual_events):
    """Independently propagate and check the EXACT labeled events enforced by IPOPT."""
    settings = p.star_tracker
    axes = candidate_axes(candidate)
    plane = np.arcsin(np.clip(axes[:, 2], -1, 1)) - settings.full_cone_fov / 2
    plane_ok = np.all(plane >= settings.top_plane_clearance_margin - settings.angular_tolerance)
    angular_tolerance = np.radians(.0002)
    timing_tolerance = 2e-5
    reports = {}
    for block in build_blocks(p, sol, branch, independent=True):
        events = [dict(e) for e in actual_events if e["block"] == block.name]
        times = np.array([e["time_s"] for e in events])
        if not len(times) or np.any(np.diff(times) <= 0):
            raise ValueError("Missing/non-increasing coupled measurement events")
        samples = block.evaluate(times)
        margins = geometry_margins(samples, axes, settings)
        chosen = margins[np.arange(len(events)), np.array([e["tracker"] - 1 for e in events])]
        gaps = np.diff(np.r_[block.start, times, block.end])
        valid = (np.all(chosen >= -angular_tolerance, axis=1) &
                 (times >= block.start - timing_tolerance) & (times <= block.end + timing_tolerance))
        if settings.enforce_body_rate:
            valid &= samples.body_rates <= settings.tracking_body_rate_max + settings.rate_tolerance
        for i, event in enumerate(events):
            event.update(valid=bool(valid[i]), angular_margins_deg=dict(zip(
                ("sun", "earth", "moon_limb"), np.degrees(chosen[i]).tolist())),
                         body_rate_deg_s=float(np.degrees(samples.body_rates[i])))
        limiting = np.unravel_index(np.argmin(chosen), chosen.shape)
        reports[block.name] = dict(
            passed=bool(plane_ok and np.all(valid) and max(gaps) <= settings.update_gap_max + timing_tolerance),
            events=events, event_count=len(events), start_s=block.start, end_s=block.end,
            first_update_latency_s=float(gaps[0]), largest_inter_update_gap_s=float(max(gaps[1:-1], default=0)),
            final_update_age_s=float(gaps[-1]), max_update_gap_s=float(max(gaps)),
            body_rate_limit_enforced=settings.enforce_body_rate,
            worst_accepted_body_rate_deg_s=float(np.degrees(max(samples.body_rates))),
            worst_angular_margin_deg=float(np.degrees(chosen.min())),
            limiting_event=dict(time_s=float(times[limiting[0]]), tracker=events[limiting[0]]["tracker"],
                                exclusion=("sun", "earth", "moon_limb")[limiting[1]],
                                margin_deg=float(np.degrees(chosen[limiting]))),
            angular_validation_tolerance_deg=float(np.degrees(angular_tolerance)),
            timing_validation_tolerance_s=timing_tolerance,
            status="exact_coupled_events_validated" if np.all(valid) and plane_ok and
            max(gaps) <= settings.update_gap_max + timing_tolerance else "coupled_event_validation_failed",
            numerically_resolved=True)
    return reports

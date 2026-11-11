"""Independent check: re-propagate the optimised controls with scipy and compare with the collocation."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import make_numpy_rhs
from .params import Params
from .postprocess import chain, phase_mode


def repropagate(p: Params, sol, branch: str, rtol: float = 1e-10):
    """Integrate phase by phase from the optimiser's own phase-start state, with controls linearly
    interpolated between nodes (the collocation's control model). Returns per-phase end errors and the
    touchdown position/velocity error of the chained propagation: td["hop1"] at the divert site, then the
    vehicle restarts at rest on the surface and td at the start site (hop 2)."""
    report = []
    x_chain = None
    hop1_td = None
    for name in chain(branch, p):
        ph = sol.phases[name]
        f = make_numpy_rhs(p, mode=phase_mode(p, ph))
        if name.startswith("rise2_"):
            # surface stay between the hops: at rest in the Moon-fixed frame, same place, same mass
            hop1_td = _touchdown(p, x_chain, p.mission.r_start + sol.phases[f"terminal_{branch}"]["X"][0:3, -1])
            x_chain = x_chain.copy()
            x_chain[3:6] = 0.0
        T, n = ph["t"], ph["X"].shape[1] - 1
        tn = np.linspace(0.0, T, n + 1)

        def u_of(t, U=ph["U"], tn=tn):
            return np.array([np.interp(t, tn, U[i]) for i in range(4)])

        x0 = ph["X"][:, 0] if x_chain is None else x_chain
        s = solve_ivp(lambda t, x: f(x, u_of(t)), (0.0, T), x0, rtol=rtol, atol=1e-9, method="DOP853")
        x_chain = s.y[:, -1]
        # error of this phase's propagation against the collocation end node (from the same start state)
        s2 = solve_ivp(lambda t, x: f(x, u_of(t)), (0.0, T), ph["X"][:, 0], rtol=rtol, atol=1e-9,
                       method="DOP853")
        e = s2.y[:, -1] - ph["X"][:, -1]
        report.append(dict(phase=name, pos_err=float(np.linalg.norm(e[0:3])),
                           vel_err=float(np.linalg.norm(e[3:6])), mass_err=float(abs(e[6]))))
    td = _touchdown(p, x_chain, p.mission.r_start + sol.phases[f"terminal2_{branch}"]["X"][0:3, -1])
    td["hop1"] = hop1_td
    return report, td


def _touchdown(p: Params, x, site_abs):
    """Position error to the site and speeds at the end of a chained propagation (x is the offset-frame state)."""
    return dict(pos_err=float(np.linalg.norm(x[0:3] - (site_abs - p.mission.r_start))),
                vel=float(np.linalg.norm(x[3:6])),
                up_speed=float(x[3:6] @ (site_abs / np.linalg.norm(site_abs))))


def sensor_report(p: Params, sol, branch: str):
    """Dense independent integration of each observation window; angles use the square image axes."""
    from .sensor import image_angles
    reports = {}
    for name, target, terminal in (("observation", p.mission.r_target, f"terminal_{branch}"),
                                   (f"observation2_{branch}", p.mission.r_start, f"terminal2_{branch}")):
        ph = sol.phases[name]
        f = make_numpy_rhs(p, mode=phase_mode(p, ph))
        tn = np.linspace(0, ph["t"], ph["X"].shape[1])
        times = np.linspace(0, ph["t"], 401)
        def u_of(t):
            return np.array([np.interp(t, tn, row) for row in ph["U"]])
        integ = solve_ivp(lambda t, x: f(x, u_of(t)), (0, ph["t"]), ph["X"][:, 0],
                          t_eval=times, rtol=1e-10, atol=1e-9, method="DOP853")
        if not integ.success:
            raise RuntimeError(f"Observation integration failed: {integ.message}")
        points = (target, p.mission.r_start + sol.phases[terminal]["X"][0:3, -1])
        altitude, descent, horizontal, ranges, angles, depths, clearances, directions = [], [], [], [], [], [], [], []
        for t, x in zip(times, integ.y.T):
            r = p.mission.r_start + x[:3]
            up = r / np.linalg.norm(r)
            vup = x[3:6] @ up
            altitude.append(np.linalg.norm(r) - p.moon.radius)
            descent.append(-vup)
            horizontal.append(np.linalg.norm(x[3:6] - vup * up))
            thrust = u_of(t)[:3]
            directions.append(thrust / np.linalg.norm(thrust))
            for point in points:
                los = point - r
                ah, av, depth = image_angles(thrust, p.mission.downrange, los)
                ranges.append(np.linalg.norm(los))
                angles.append((abs(ah), abs(av)))
                depths.append(depth)
                clearances.append((r - point) @ (point / np.linalg.norm(point)))
        directions = np.array(directions)
        # atan2 remains accurate for the small angular steps of a dense sampling.
        slew = np.arctan2(np.linalg.norm(np.cross(directions[:-1], directions[1:]), axis=1),
                          np.sum(directions[:-1] * directions[1:], axis=1)) / np.diff(times)
        max_angles = np.max(angles, axis=0)
        se = p.sensor
        checks = [min(altitude) >= se.min_altitude - 0.02,
                  min(descent) >= se.descent_speed_bounds[0] - 0.001,
                  max(descent) <= se.descent_speed_bounds[1] + 0.001,
                  max(horizontal) <= se.horizontal_speed_max + 0.001,
                  ph["t"] >= se.observation_time_min + se.processing_delay - 1e-5]
        if se.enforce_pointing:
            checks += [max(ranges) <= se.max_range + 0.02,
                       all(max_angles <= np.degrees(se.fov) / 2 + 0.01),
                       min(depths) > 0, min(clearances) >= -0.02,
                       max(slew) <= min(se.pointing_rate_max, p.limits.tilt_rate_max) + np.radians(0.02)]
        else:
            checks.append(max(slew) <= p.limits.tilt_rate_max + np.radians(0.02))
        reports[name] = dict(passed=bool(all(checks)), pointing_enforced=se.enforce_pointing, duration=ph["t"],
                             acquisition_time=ph["t"] - se.processing_delay,
                             min_altitude=float(min(altitude)), min_descent_speed=float(min(descent)),
                             max_descent_speed=float(max(descent)), max_horizontal_speed=float(max(horizontal)),
                             max_range=float(max(ranges)), max_image_angle_deg=max_angles.tolist(),
                             max_pointing_rate_deg_s=float(np.degrees(max(slew))),
                             min_surface_visibility_clearance=float(min(clearances)))
    return reports

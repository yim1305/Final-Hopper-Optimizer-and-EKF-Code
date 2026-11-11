"""Independent control-knot integration and unchanged flight/camera acceptance limits."""

from types import SimpleNamespace
import numpy as np
from scipy.integrate import solve_ivp
from hopper.dynamics import make_numpy_rhs
from hopper.postprocess import phase_mode
from hopper.sensor import image_angles
from .config import SETTINGS


def propagate(f, stop, initial, knots, event=None):
    """Integrate separately across each control knot, retaining dense output."""
    edges = np.unique(np.r_[0.0, knots[(knots > 0) & (knots < stop)], stop])
    parts = []
    starts = []
    state = initial.copy()
    events = []
    for left, right in zip(edges[:-1], edges[1:]):
        part = solve_ivp(
            f,
            (left, right),
            state,
            events=event,
            dense_output=True,
            method="DOP853",
            rtol=1e-13,
            atol=1e-12,
            max_step=min(1.0, right - left),
        )
        if not part.success:
            raise ValueError(part.message)
        starts.append(left)
        parts.append(part)
        state = part.y[:, -1]
        if event is not None and len(part.t_events[0]):
            events = part.t_events[0]
            break

    def dense(times):
        times = np.atleast_1d(times)
        answer = np.empty((len(initial), len(times)))
        choices = np.clip(np.searchsorted(starts, times, side="right") - 1, 0, len(parts) - 1)
        for i in np.unique(choices):
            answer[:, choices == i] = parts[i].sol(times[choices == i])
        return answer

    return SimpleNamespace(success=True, y=state[:, None], sol=dense, t_events=[events])


def integrate(p, sol, terrain):
    rows, history, contacts = [], [], {}
    clock = 0.0
    chain = None
    for name, ph in sol.phases.items():
        if chain is None:
            chain = ph["X"][:, 0].copy()
        if name.startswith("rise2_"):
            chain[3:6] = 0.0
        knots = np.linspace(0, ph["t"], ph["X"].shape[1])

        def control(t):
            return np.array([np.interp(t, knots, row) for row in ph["U"]])

        rhs = make_numpy_rhs(p, mode=phase_mode(p, ph))

        def f(t, x):
            return rhs(x, control(t))

        local = propagate(f, ph["t"], ph["X"][:, 0], knots)

        def surface(t, x):
            return terrain.altitude(p.mission.r_start + x[:3])

        surface.direction = -1
        surface.terminal = True
        # Terminal propagation ends at the first descending DEM contact, not at
        # an imposed clock or mean-sphere crossing. Small extra time brackets it.
        terminal = ph["kind"] == "terminal"
        stop = ph["t"] + SETTINGS["contact_time_tolerance_s"] if terminal else ph["t"]
        integrated = propagate(f, stop, chain, knots, surface if terminal else None)
        if not integrated.success or not local.success:
            raise ValueError("Independent integration failed: " + name)
        end = ph["t"]
        if terminal:
            if not len(integrated.t_events[0]):
                raise ValueError("No first DEM contact: " + name)
            end = float(integrated.t_events[0][0])
        sample = np.unique(
            np.r_[
                np.arange(0, end, SETTINGS["independent_sample_step_s"]),
                end,
                knots[knots <= end],
                np.linspace(0, end, 401) if ph["kind"] == "observation" else [],
            ]
        )
        # Adding mission-clock offsets can collapse machine-close local times.
        # Keep one state per distinct absolute time before finite-rate checks.
        _, unique_indices = np.unique(clock + sample, return_index=True)
        sample = sample[unique_indices]
        states = integrated.sol(sample).T
        positions = p.mission.r_start + states[:, :3]
        altitude = terrain.altitude(positions)
        controls = np.array([control(t) for t in sample])
        local_error = float(np.linalg.norm(local.y[:3, -1] - ph["X"][:3, -1]))
        chain = states[-1].copy()
        chain_error = float(np.linalg.norm(chain[:3] - ph["X"][:3, -1]))
        floor = (
            p.limits.min_altitude
            if ph["kind"] in ("ascent", "coast", "braking")
            else p.sensor.min_altitude if ph["kind"] == "observation" else 0.0
        )
        row = dict(
            phase=name,
            minimum_AGL_m=float(altitude.min()),
            required_minimum_AGL_m=floor,
            local_position_error_m=local_error,
            chained_position_error_m=chain_error,
            passed=bool(
                local_error < 0.05 and chain_error < 0.05 and altitude.min() >= floor - 0.02
            ),
        )
        rows.append(row)
        history.append(
            dict(
                name=name,
                kind=ph["kind"],
                hop=ph["hop"],
                time=clock + sample,
                local_time=sample,
                state=states,
                r=positions,
                U=controls,
                AGL=altitude,
            )
        )
        if terminal:
            r = positions[-1]
            v = chain[3:6]
            up = r / np.linalg.norm(r)
            vertical = float(v @ up)
            horizontal = float(np.linalg.norm(v - vertical * up))
            target = p.mission.sites["uprange"] if ph["hop"] == 1 else p.mission.r_start
            offset = float(np.linalg.norm(r - target))
            tilt = float(
                np.degrees(
                    np.arccos(
                        np.clip(controls[-1, :3] @ up / np.linalg.norm(controls[-1, :3]), -1, 1)
                    )
                )
            )
            contacts[f'hop{ph["hop"]}'] = dict(
                event_detected=True,
                first_descending_surface_contact=True,
                local_contact_time_s=end,
                contact_time_difference_from_optimized_s=end - ph["t"],
                AGL_m=float(altitude[-1]),
                elevation_relative_reference_sphere_m=float(np.linalg.norm(r) - p.moon.radius),
                vertical_speed_m_s=vertical,
                horizontal_speed_m_s=horizontal,
                target_error_m=offset,
                touchdown_tilt_deg=tilt,
                passed=bool(
                    abs(end - ph["t"]) <= SETTINGS["contact_time_tolerance_s"]
                    and abs(altitude[-1]) < 1e-5
                    and vertical >= -p.limits.td_vert_speed_max - 0.001
                    and vertical <= 0.001
                    and horizontal <= p.limits.td_horiz_speed_max + 0.001
                    and offset <= p.mission.landing_tolerance + 0.02
                    and tilt <= np.degrees(p.limits.td_tilt_max) + 0.01
                ),
            )
        clock += end
        print(
            name,
            "min DEM clearance",
            float(altitude.min()),
            "chained error",
            chain_error,
            flush=True,
        )
    return rows, history, contacts


def ray_clearance(terrain, base, end, step=7.5):
    distance = np.linalg.norm(end - base)
    n = max(3, int(np.ceil(distance / step)) + 1)
    ray = base + np.linspace(0, 1, n)[:, None] * (end - base)
    return float(terrain.altitude(ray).min())


def camera_check(p, sol, terrain, history):
    reports = {}
    for item in history:
        if item["kind"] != "observation":
            continue
        hop = item["hop"]
        terminal = "terminal_uprange" if hop == 1 else "terminal2_uprange"
        ground_points = [
            p.mission.r_target if hop == 1 else p.mission.r_start,
            p.mission.r_start + sol.phases[terminal]["X"][:3, -1],
        ]
        angles = []
        depths = []
        ranges = []
        ray_min = []
        for r, u in zip(item["r"], item["U"]):
            for point in ground_points:
                ah, av, d = image_angles(u[:3], p.mission.downrange, point - r)
                angles.append([abs(ah), abs(av)])
                depths.append(d)
                ranges.append(np.linalg.norm(r - point))
                ray_min.append(ray_clearance(terrain, point, r, step=1.0))
        up = item["r"] / np.linalg.norm(item["r"], axis=1)[:, None]
        vup = np.einsum("ij,ij->i", item["state"][:, 3:6], up)
        vh = np.linalg.norm(item["state"][:, 3:6] - vup[:, None] * up, axis=1)
        direction = item["U"][:, :3] / np.linalg.norm(item["U"][:, :3], axis=1)[:, None]
        rates = np.arctan2(
            np.linalg.norm(np.cross(direction[:-1], direction[1:]), axis=1),
            np.einsum("ij,ij->i", direction[:-1], direction[1:]),
        ) / np.diff(item["time"])
        passed = (
            min(depths) > 0
            and max(ranges) <= p.sensor.max_range + 0.02
            and np.all(np.max(angles, axis=0) <= np.degrees(p.sensor.fov) / 2 + 0.01)
            and min(ray_min) >= -0.02
            and item["AGL"].min() >= p.sensor.min_altitude - 0.02
            and min(-vup) >= p.sensor.descent_speed_bounds[0] - 0.001
            and max(-vup) <= p.sensor.descent_speed_bounds[1] + 0.001
            and max(vh) <= p.sensor.horizontal_speed_max + 0.001
            and max(rates)
            <= min(p.sensor.pointing_rate_max, p.limits.tilt_rate_max) + np.radians(0.02)
        )
        reports[item["name"]] = dict(
            passed=bool(passed),
            max_image_angles_deg=np.max(angles, axis=0).tolist(),
            min_camera_ray_clearance_m=min(ray_min),
            max_range_m=max(ranges),
            minimum_AGL_m=float(item["AGL"].min()),
        )
    return reports

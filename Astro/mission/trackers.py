"""Terrain-aware star-tracker schedules on the solved reference flight."""

import hashlib
import numpy as np
from scipy.ndimage import map_coordinates
from hopper.star_tracker import (
    PhaseHistory,
    CoverageBlock,
    CoastOrientation,
    rotation_z,
    top_mount_axes,
    angle_between,
)
from .terrain import DemTerrain, ROOT


class SkyTerrain:
    def __init__(self):
        self.native = DemTerrain(fraction=1.0)
        path = ROOT / "data/ldem_16.img"
        raw = np.fromfile(path, dtype="<i2")
        if len(raw) != 2880 * 5760:
            raise ValueError("Wrong global LOLA dimensions")
        self.global_dem = raw.reshape(2880, 5760).astype(float) * 0.5
        self.outer = self.native.radius + max(
            float(self.native.dem.max()), float(self.global_dem.max())
        )
        self.inner = self.native.radius + min(
            float(self.native.dem.min()), float(self.global_dem.min())
        )
        self.global_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    def height(self, r):
        r = np.asarray(r)
        x, y = self.native.projected(tuple(r.T))
        col, row = self.native.inverse * (x, y)
        inside = (
            (col >= 0.5)
            & (col <= self.native.dem.shape[1] - 0.5)
            & (row >= 0.5)
            & (row <= self.native.dem.shape[0] - 0.5)
        )
        values = np.empty(len(r))
        if inside.any():
            values[inside] = self.native.height_positions(r[inside])
        if (~inside).any():
            rr = r[~inside]
            radius = np.linalg.norm(rr, axis=1)
            lat = np.degrees(np.arcsin(rr[:, 2] / radius))
            lon = np.degrees(np.arctan2(rr[:, 1], rr[:, 0])) % 360
            rows = np.clip((90 - lat) * 16 - 0.5, 0, 2879)
            cols = (lon * 16 - 0.5) % 5760
            values[~inside] = map_coordinates(
                self.global_dem, [rows, cols], order=1, mode="grid-wrap", prefilter=False
            )
        return values

    def cone(self, r, axis, half_angle, rings, azimuths):
        radius = np.linalg.norm(r)
        up = r / radius
        from_nadir = float(np.arccos(np.clip(axis @ (-up), -1, 1)))
        if radius > self.outer and from_nadir - half_angle > np.arcsin(self.outer / radius):
            return dict(
                clear=True,
                method="whole_cone_outside_global_DEM_bounding_sphere",
                minimum_clearance_m=None,
            )
        ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(axis, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)
        directions = [axis]
        for theta in np.linspace(0, half_angle, rings + 1)[1:]:
            for phi in np.linspace(0, 2 * np.pi, azimuths, endpoint=False):
                directions.append(
                    np.cos(theta) * axis + np.sin(theta) * (np.cos(phi) * e1 + np.sin(phi) * e2)
                )
        minimum = float("inf")
        for direction in directions:
            rd = float(r @ direction)
            closest2 = radius * radius - max(0.0, -rd) ** 2
            if closest2 < self.inner * self.inner:
                return dict(
                    clear=False,
                    method="ray_intersects_global_inner_solid_sphere",
                    minimum_clearance_m=None,
                )
            disc = rd * rd + self.outer * self.outer - radius * radius
            if disc < 0:
                continue
            length = -rd + np.sqrt(disc)
            if length <= 0:
                continue
            # Resolve native local terrain at 30 m and the remote 1.9 km grid
            # at 240 m. End beyond the data-derived outer bounding sphere.
            near = min(length, 5000.0)
            s = np.unique(np.r_[np.arange(0, near, 30.0), np.arange(near, length, 240.0), length])
            points = r + s[:, None] * direction
            clearance = np.linalg.norm(points, axis=1) - self.native.radius - self.height(points)
            minimum = min(minimum, float(clearance.min()))
            if minimum < -0.001:
                return dict(
                    clear=False, method="sampled_DEM_cone_rays", minimum_clearance_m=minimum
                )
        return dict(
            clear=True,
            method="sampled_DEM_cone_rays",
            minimum_clearance_m=minimum if np.isfinite(minimum) else None,
        )


def blocks(p, sol):
    histories = {}
    clock = 0.0
    for name, ph in sol.phases.items():
        if ph["kind"] in p.star_tracker.required_phase_kinds or ph["kind"] == "braking":
            histories[name] = PhaseHistory(
                p, name, ph, clock, independent=ph["kind"] in p.star_tracker.required_phase_kinds
            )
        clock += ph["t"]
    result = []
    for hop in (1, 2):
        group = [ph for ph in histories.values() if ph.ph["hop"] == hop]
        ascent = [ph for ph in group if ph.ph["kind"] == "ascent"][-1]
        coast = next(ph for ph in group if ph.ph["kind"] == "coast")
        braking = next(ph for ph in group if ph.ph["kind"] == "braking")
        coast.coast = CoastOrientation(
            ascent.orientation([ascent.end])[0][0],
            braking.orientation([braking.start])[0][0],
            coast.ph["t"],
        )
        result.append(
            CoverageBlock(
                f"hop{hop}",
                [ph for ph in group if ph.ph["kind"] in p.star_tracker.required_phase_kinds],
                p.star_tracker,
            )
        )
    return result


def tracker_checks(p, sol):
    terrain = SkyTerrain()
    settings = p.star_tracker
    mounts = top_mount_axes(settings.initial_elevations, settings.initial_azimuths)
    reports = {}
    for block in blocks(p, sol):
        # Explicit physical updates at <=1 s spacing, within the 2 s limit.
        times = np.linspace(
            block.start,
            block.end,
            int(np.ceil((block.end - block.start) / min(1.0, settings.update_gap_max / 2))) + 1,
        )
        events = []
        for k, t in enumerate(times):
            ph = next((ph for ph in block.phases if t <= ph.end + 1e-10), block.phases[-1])
            C, rate = ph.orientation(np.array([t]))
            r = (
                p.mission.r_start
                + ph.state(np.array([np.clip(t - ph.start, 0, ph.ph["t"])]))[0, :3]
            )
            accepted = None
            for j, axis in enumerate(mounts):
                b = C[0] @ axis
                sun = float(
                    angle_between(b, np.array(settings.sun_direction_I)) - settings.sun_exclusion
                )
                earth = float(
                    angle_between(b, np.array(settings.earth_direction_I))
                    - settings.earth_exclusion
                )
                if min(sun, earth) < -settings.angular_tolerance or (
                    settings.enforce_body_rate
                    and rate[0] > settings.tracking_body_rate_max + settings.rate_tolerance
                ):
                    continue
                direction = rotation_z(-p.moon.omega * t) @ b
                coarse = terrain.cone(
                    r, direction, settings.full_cone_fov / 2 + settings.moon_limb_margin, 2, 16
                )
                if not coarse["clear"]:
                    continue
                fine = terrain.cone(
                    r, direction, settings.full_cone_fov / 2 + settings.moon_limb_margin, 4, 32
                )
                if fine["clear"]:
                    accepted = dict(
                        time_s=float(t),
                        tracker=j + 1,
                        valid=True,
                        body_rate_deg_s=float(np.degrees(rate[0])),
                        sun_margin_deg=float(np.degrees(sun)),
                        earth_margin_deg=float(np.degrees(earth)),
                        terrain_cone_check=fine,
                    )
                    break
            if accepted is None:
                # The existing initialized estimate has zero age at block start.
                # A measurement is required before its age exceeds the configured
                # gap, rather than at every point of this denser candidate grid.
                last = events[-1]["time_s"] if events else block.start
                if t - last > settings.update_gap_max + 1e-8:
                    reports[block.name] = dict(
                        passed=False,
                        failure_time_s=float(t),
                        event_count=len(events),
                        events=events,
                    )
                    break
                continue
            events.append(accepted)
            if k % 100 == 0:
                print(block.name, "terrain tracker events", k, "/", len(times), flush=True)
        else:
            accepted_times = np.array([event["time_s"] for event in events])
            ages = np.diff(np.r_[block.start, accepted_times, block.end])
            gap = float(ages.max())
            reports[block.name] = dict(
                passed=bool(len(events) and gap <= settings.update_gap_max + 1e-8),
                event_count=len(events),
                max_update_gap_s=gap,
                events=events,
                first_update_age_s=float(accepted_times[0] - block.start) if len(events) else None,
                final_estimate_age_s=float(block.end - accepted_times[-1]) if len(events) else None,
            )
        print(block.name, "passed", reports[block.name]["passed"], flush=True)
    return dict(
        passed=all(row["passed"] for row in reports.values()),
        reports=reports,
        mounting_elevations_deg=np.degrees(settings.initial_elevations).tolist(),
        mounting_azimuths_deg=np.degrees(settings.initial_azimuths).tolist(),
    )

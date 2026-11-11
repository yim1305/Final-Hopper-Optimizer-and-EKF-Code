"""Fixed-mount tracker events coupled to the native DEM trajectory NLP.

Event labels/cells and low-altitude ray lengths are fixed within a local solve.
All times, attitudes, spacecraft positions and ray directions remain symbolic.
The global outer sphere is used only as a sufficient terrain bound, never as
the lunar surface. Exact assigned events require independent DEM validation.
"""

from dataclasses import asdict
import casadi as ca
import numpy as np
from scipy.interpolate import CubicHermiteSpline

from hopper.star_tracker import top_mount_axes, rotation_z, angle_between
from hopper.star_tracker_coupled import (
    AssignedEvent, symbolic_powered_frame, symbolic_rotation_z,
    symbolic_body_rate_squared, symbolic_slerp_data, symbolic_slerp_from_data,
)
from .terrain import GroundProblem
from .trackers import SkyTerrain, blocks
from .config import CONFIG


def cone_axes(axis, half, rings, azimuths):
    """Stable body-fixed rays; rotating these with the vehicle avoids frame jumps."""
    axis = np.asarray(axis)
    ref = np.array([0., 0., 1.]) if abs(axis[2]) < .9 else np.array([1., 0., 0.])
    first = np.cross(axis, ref)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    rays = [axis]
    for theta in np.linspace(0, half, rings + 1)[1:]:
        for phi in np.linspace(0, 2*np.pi, azimuths, endpoint=False):
            rays.append(np.cos(theta)*axis + np.sin(theta)*(np.cos(phi)*first + np.sin(phi)*second))
    return np.asarray(rays)


def minimum_cone_projection(r, axis, half):
    """Smallest r dot d over a circular cone (valid also when -r is inside)."""
    radius = np.linalg.norm(r)
    dot = float(r @ axis)
    if dot <= -radius*np.cos(half):
        return -radius
    return dot*np.cos(half) - np.sqrt(max(0., radius*radius-dot*dot))*np.sin(half)


def prepare_events(p, sol, sky, settings):
    """Use actual terrain and either head to seed a dense, labeled event schedule."""
    mounts = top_mount_axes(p.star_tracker.initial_elevations, p.star_tracker.initial_azimuths)
    half = p.star_tracker.full_cone_fov/2 + p.star_tracker.moon_limb_margin
    events, envelopes = [], []
    for block in blocks(p, sol):
        step = min(settings['seed_event_step_s'], p.star_tracker.update_gap_max/2)
        times = np.linspace(block.start, block.end, int(np.ceil((block.end-block.start)/step))+1)
        last = block.start  # Parser requires zero initial estimate age.
        count = 0
        for t in times:
            ph = next((ph for ph in block.phases if t <= ph.end + 1e-10), block.phases[-1])
            C, rates = ph.orientation(np.array([t]))
            r = p.mission.r_start + ph.state(np.array([np.clip(t-ph.start, 0, ph.ph['t'])]))[0, :3]
            candidates = []
            for head, axis in enumerate(mounts):
                b = C[0] @ axis
                margin = min(float(angle_between(b, np.asarray(p.star_tracker.sun_direction_I))) - p.star_tracker.sun_exclusion,
                             float(angle_between(b, np.asarray(p.star_tracker.earth_direction_I))) - p.star_tracker.earth_exclusion)
                d = rotation_z(-p.moon.omega*t) @ b
                if margin < 0 or (p.star_tracker.enforce_body_rate and rates[0] > p.star_tracker.tracking_body_rate_max):
                    continue
                if sky.cone(r, d, half, 2, 16)['clear']:
                    candidates.append((margin, head, d))
            if not candidates:
                if t-last > p.star_tracker.update_gap_max + 1e-8:
                    raise ValueError(f'No feasible DEM tracker seed at {block.name} {t:.6f} s')
                continue
            _, head, d = max(candidates, key=lambda x: x[0])
            n = ph.ph['X'].shape[1]-1
            location = np.clip((t-ph.start)/ph.ph['t']*n, 0, n)
            cell = min(int(np.floor(location)), n-1)
            event = AssignedEvent(ph.name, cell, float(location-cell), head+1, block.name)
            events.append(event)
            projection = minimum_cone_projection(r, d, half)
            # Entire cone misses the outer sphere: a cheap analytic certificate.
            sphere_clear = np.linalg.norm(r) > sky.outer and (projection >= 0 or
                            np.dot(r, r)-projection**2 > sky.outer**2)
            length = 0.
            if not sphere_clear:
                length = max(settings['ray_step_m'], settings['ray_length_seed_factor'] *
                             (-projection + np.sqrt(max(0., projection**2 + sky.outer**2 - np.dot(r, r)))))
            envelopes.append(dict(length_m=float(length), seed_r=r.tolist(), seed_C_F=(rotation_z(-p.moon.omega*t) @ C[0]).tolist()))
            last = t
            count += 1
        if not count or block.end-last > p.star_tracker.update_gap_max + 1e-8:
            raise ValueError(f'Incomplete tracker seed in {block.name}')
        print(block.name, 'seeded tracker events', count, flush=True)
    return events, envelopes


class SymbolicSky:
    """Exact native/global bilinear height, cropping the global table for memory.

    The crop covers all seeded rays plus a numerical domain pad; symbolic domain
    bounds forbid extrapolation. Global resolution is unchanged (16 pixels/deg).
    """
    def __init__(self, sky, points, pad_deg):
        self.native = sky.native
        points = np.asarray(points)
        lat = np.degrees(np.arcsin(points[:, 2]/np.linalg.norm(points, axis=1)))
        lon = np.degrees(np.arctan2(points[:, 1], points[:, 0])) % 360
        r0 = max(0, int(np.floor((90-lat.max()-pad_deg)*16-.5)))
        r1 = min(2879, int(np.ceil((90-lat.min()+pad_deg)*16-.5)))
        c0 = max(0, int(np.floor((lon.min()-pad_deg)*16-.5)))
        c1 = min(5759, int(np.ceil((lon.max()+pad_deg)*16-.5)))
        self.lat_grid = (90-(np.arange(r0, r1+1)+.5)/16)[::-1]
        self.lon_grid = (np.arange(c0, c1+1)+.5)/16
        values = sky.global_dem[r0:r1+1, c0:c1+1][::-1, :]
        self.lut = ca.interpolant('tracker_global_dem', 'linear',
                                  [self.lon_grid.tolist(), self.lat_grid.tolist()], values.ravel().tolist())

    def constrain(self, opti, points, scale):
        n = points.shape[1]
        x, y = self.native.projected(tuple(points[i, :] for i in range(3)))
        radius = ca.sqrt(ca.sum1(points*points))
        lat = ca.asin(points[2, :]/radius)*180/np.pi
        lon = ca.atan2(points[1, :], points[0, :])*180/np.pi
        lon = ca.if_else(lon < 0, lon+360, lon)
        opti.subject_to(opti.bounded(self.lat_grid[0], lat, self.lat_grid[-1]))
        opti.subject_to(opti.bounded(self.lon_grid[0], lon, self.lon_grid[-1]))
        nx, ny = self.native.xgrid, self.native.ygrid
        inside = ca.logic_and(ca.logic_and(x >= nx[0], x <= nx[-1]),
                              ca.logic_and(y >= ny[0], y <= ny[-1]))
        local = self.native.lut.map(n)(ca.vertcat(ca.fmin(ca.fmax(x, nx[0]), nx[-1]),
                                                 ca.fmin(ca.fmax(y, ny[0]), ny[-1])))
        remote = self.lut.map(n)(ca.vertcat(ca.fmin(ca.fmax(lon, self.lon_grid[0]), self.lon_grid[-1]),
                                           ca.fmin(ca.fmax(lat, self.lat_grid[0]), self.lat_grid[-1])))
        clearance = radius-self.native.radius-ca.if_else(inside, local, remote)
        opti.subject_to(clearance/scale >= 0)


class DemTrackerProblem(GroundProblem):
    """GroundProblem plus fixed-head event geometry, cadence, rate and DEM rays."""
    def __init__(self, p, branch, events, envelopes, sky, settings=None, **kwargs):
        self.events, self.envelopes, self.sky = list(events), list(envelopes), sky
        self.settings = settings or CONFIG['tracker_solver']
        for key in ('seed_event_step_s', 'ray_step_m', 'global_crop_pad_deg'):
            if not np.isfinite(self.settings[key]) or self.settings[key] <= 0:
                raise ValueError(f'tracker_solver.{key} must be positive and finite')
        for key in ('cone_rings', 'cone_azimuths', 'max_refinement_attempts'):
            if not isinstance(self.settings[key], int) or self.settings[key] < 1:
                raise ValueError(f'tracker_solver.{key} must be a positive integer')
        if not np.isfinite(self.settings['ray_length_seed_factor']) or self.settings['ray_length_seed_factor'] <= 1:
            raise ValueError('Ray length seed factor must exceed one')
        st = p.star_tracker
        if not st or not st.enabled or st.mount_model != 'independent_top':
            raise ValueError('DEM tracker solver requires enabled independent_top trackers')
        if tuple(st.required_phase_kinds) != ('ascent', 'coast'):
            raise ValueError('DEM tracker solver currently supports ascent/coast coverage')
        # The frozen configuration parser already requires zero acquisition,
        # switching delay and initial estimate age.
        if len(events) != len(envelopes):
            raise ValueError('Event/envelope count mismatch')
        mounts = top_mount_axes(st.initial_elevations, st.initial_azimuths)
        if np.any(np.arcsin(mounts[:, 2])-st.full_cone_fov/2 < st.top_plane_clearance_margin):
            raise ValueError('Installed tracker cone obstructed by top plane')
        self.mounts = mounts
        super().__init__(p, branch, **kwargs)

    def _build(self):
        super()._build()
        self.base_constraint_count = int(self.opti.ng)
        p, s, opti, st = self.p, self.s, self.opti, self.p.star_tracker
        half = st.full_cone_fov/2 + st.moon_limb_margin
        phases = {ph.name: ph for ph in self.phases}
        indices = {ph.name: k for k, ph in enumerate(self.phases)}
        starts, current = {}, 0.
        for ph in self.phases:
            starts[ph.name] = current
            current += ph.tdur*s.T
        durations = ca.vertcat(*[ph.tdur*s.T for ph in self.phases])
        heading = ca.DM(p.mission.downrange)
        coasts = {}
        for hop in (self.hop1, self.hop2):
            ascent = [ph for ph in hop if ph.kind == 'ascent'][-1]
            coast = next(ph for ph in hop if ph.kind == 'coast')
            braking = next(ph for ph in hop if ph.kind == 'braking')
            first = symbolic_rotation_z(p.moon.omega*starts[coast.name]) @ symbolic_powered_frame(ascent.U[:3, -1], heading)
            last = symbolic_rotation_z(p.moon.omega*starts[braking.name]) @ symbolic_powered_frame(braking.U[:3, 0], heading)
            coasts[coast.name] = symbolic_slerp_data(first, last)
            if st.enforce_body_rate:
                opti.subject_to(2*coasts[coast.name][-1]/(coast.tdur*s.T) <= st.tracking_body_rate_max)
        derivatives = {ph.name: self.dynamics_of(ph).map(ph.n+1)(ph.X, ph.U)
                       for ph in self.phases if ph.kind in st.required_phase_kinds}
        ray_axes = [cone_axes(axis, half, self.settings['cone_rings'], self.settings['cone_azimuths']) for axis in self.mounts]
        seeds = []
        for event, env in zip(self.events, self.envelopes):
            if env['length_m']:
                distances = np.linspace(0, env['length_m'], int(np.ceil(env['length_m']/self.settings['ray_step_m']))+1)
                rays = ray_axes[event.tracker-1] @ np.asarray(env['seed_C_F']).T
                seeds.extend((np.asarray(env['seed_r']) + rays[:, None, :]*distances[None, :, None]).reshape(-1, 3))
        self.symbolic_sky = SymbolicSky(self.sky, seeds, self.settings['global_crop_pad_deg']) if len(seeds) else None
        self.event_times = []
        groups = {'hop1': [], 'hop2': []}
        self.ray_sample_count = 0
        for event, env in zip(self.events, self.envelopes):
            ph = phases[event.phase]
            k, u = event.cell, event.fraction
            if (event.tracker not in (1, 2) or event.block != f'hop{ph.hop}' or
                ph.kind not in st.required_phase_kinds or not (0 <= k < ph.n and 0 <= u <= 1)):
                raise ValueError('Invalid event assignment')
            fraction = (k+u)/ph.n
            coeff = np.zeros(len(self.phases))
            coeff[:indices[ph.name]] = 1
            coeff[indices[ph.name]] = fraction
            if groups[event.block] and np.dot(coeff-groups[event.block][-1], np.ones(len(coeff))) <= 0:
                raise ValueError('Events must be strictly ordered in each hop')
            groups[event.block].append(coeff)
            t = starts[ph.name] + fraction*ph.tdur*s.T
            self.event_times.append(t)
            F = derivatives[ph.name]
            x = ((2*u**3-3*u*u+1)*ph.X[:, k] + (u**3-2*u*u+u)*ph.tdur/ph.n*F[:, k]
                 + (-2*u**3+3*u*u)*ph.X[:, k+1] + (u**3-u*u)*ph.tdur/ph.n*F[:, k+1])
            if ph.powered:
                thrust = (1-u)*ph.U[:3, k] + u*ph.U[:3, k+1]
                C_F = symbolic_powered_frame(thrust, heading)
                C = symbolic_rotation_z(p.moon.omega*t) @ C_F
                if st.enforce_body_rate:
                    cells = {k}
                    if u == 0 and k > 0:
                        cells.add(k-1)
                    if u == 1 and k+1 < ph.n:
                        cells.add(k+1)
                    for cell in cells:
                        slope = (ph.U[:3, cell+1]-ph.U[:3, cell])/(ph.tdur*s.T/ph.n)
                        opti.subject_to(symbolic_body_rate_squared(thrust, slope, heading, p.moon.omega)/st.tracking_body_rate_max**2 <= 1)
            else:
                C = symbolic_slerp_from_data(coasts[ph.name], fraction)
                C_F = symbolic_rotation_z(-p.moon.omega*t) @ C
            b = C @ ca.DM(self.mounts[event.tracker-1])
            opti.subject_to(ca.dot(b, ca.DM(st.sun_direction_I)) <= np.cos(st.sun_exclusion))
            opti.subject_to(ca.dot(b, ca.DM(st.earth_direction_I)) <= np.cos(st.earth_exclusion))
            r = ca.DM(p.mission.r_start) + x[:3]*s.L
            d = C_F @ ca.DM(self.mounts[event.tracker-1])
            radius = ca.norm_2(r)
            dot = ca.dot(r, d)
            projection = ca.if_else(dot <= -radius*np.cos(half), -radius,
                                   dot*np.cos(half)-ca.sqrt(ca.fmax(radius**2-dot**2, 1e-12))*np.sin(half))
            length = env['length_m']
            if length == 0:
                # Closest point on any forward ray must clear the global bound.
                opti.subject_to((radius**2-ca.fmin(projection, 0)**2-self.sky.outer**2)/s.L**2 >= 0)
            else:
                # After length, every ray is outside the outer sphere and moving out.
                opti.subject_to((projection+length)/s.L >= 0)
                opti.subject_to((radius**2+length**2+2*length*projection-self.sky.outer**2)/s.L**2 >= 0)
                distances = np.linspace(0, length, int(np.ceil(length/self.settings['ray_step_m']))+1)
                rays = C_F @ ca.DM(ray_axes[event.tracker-1].T)
                points = ca.horzcat(*[ca.repmat(r, 1, len(distances)) + rays[:, j] @ ca.DM(distances[None, :]) for j in range(rays.shape[1])])
                self.symbolic_sky.constrain(opti, points, s.L)
                self.ray_sample_count += points.shape[1]
        # Compress equivalent linear cadence bounds; times are dimensional seconds.
        gaps = {}
        for label, hop in (('hop1', self.hop1), ('hop2', self.hop2)):
            active = [ph for ph in hop if ph.kind in st.required_phase_kinds]
            if not groups[label]:
                raise ValueError('Each hop needs at least one actual update')
            first, last = np.zeros(len(self.phases)), np.zeros(len(self.phases))
            first[:indices[active[0].name]] = 1
            last[:indices[active[-1].name]+1] = 1
            vectors = [first] + groups[label] + [last]
            for j, (a, b) in enumerate(zip(vectors[:-1], vectors[1:])):
                age = 0.
                diff = b-a
                gaps[(tuple(np.round(diff, 12)), age)] = (diff, age)
        for diff, age in gaps.values():
            # Events at block endpoints have an identically zero boundary age.
            if not np.any(diff):
                if age > st.update_gap_max:
                    raise ValueError('Initial estimate exceeds maximum update age')
                continue
            opti.subject_to((ca.dot(ca.DM(diff), durations)+age)/st.update_gap_max <= 1)
        self.tracker_constraint_count = int(opti.ng)-self.base_constraint_count

    def actual_events(self):
        times = np.asarray(self.opti.value(ca.vertcat(*self.event_times))).ravel()
        return [dict(**asdict(event), time_s=float(t)) for event, t in zip(self.events, times)]


def validate_events(p, sol, events, sky, history=None):
    """Validate the exact labels/times, using chained independent states if supplied."""
    mounts = top_mount_axes(p.star_tracker.initial_elevations, p.star_tracker.initial_azimuths)
    st = p.star_tracker
    half = st.full_cone_fov/2 + st.moon_limb_margin
    reports = {}
    by_name = {h['name']: h for h in history} if history is not None else {}
    coverage = blocks(p, sol)
    labels_valid = all(e.get('block') in {b.name for b in coverage} for e in events)
    for block in coverage:
        rows = []
        assigned = [e for e in events if e['block'] == block.name]
        times = np.array([e['time_s'] for e in assigned])
        ordered = bool(len(times) and np.all(np.diff(times) > 0))
        for event in assigned:
            t = event['time_s']
            ph = next((ph for ph in block.phases if ph.name == event['phase']), None)
            if ph is None or not ph.start-1e-8 <= t <= ph.end+1e-8 or event['tracker'] not in (1, 2):
                rows.append(dict(**event, valid=False, reason='event outside assigned phase/head'))
                continue
            C, rate = ph.orientation(np.array([t]))
            # Check both adjacent phases at their shared boundary.
            for adjacent in block.phases:
                if adjacent.start-1e-10 <= t <= adjacent.end+1e-10:
                    rate = np.maximum(rate, adjacent.orientation(np.array([t]))[1])
            if ph.name in by_name:
                h = by_name[ph.name]
                curve = CubicHermiteSpline(h['local_time'], h['r'], h['state'][:, 3:6])
                r = curve(np.clip(t-ph.start, 0, h['local_time'][-1]))
                clock_shift = h['time'][0]-ph.start
                C = rotation_z(p.moon.omega*clock_shift) @ C
            else:
                r = p.mission.r_start + ph.state(np.array([np.clip(t-ph.start, 0, ph.ph['t'])]))[0, :3]
                clock_shift = 0.
            b = C[0] @ mounts[event['tracker']-1]
            sun = float(angle_between(b, np.asarray(st.sun_direction_I))-st.sun_exclusion)
            earth = float(angle_between(b, np.asarray(st.earth_direction_I))-st.earth_exclusion)
            cone = sky.cone(r, rotation_z(-p.moon.omega*(t+clock_shift)) @ b, half, 4, 32)
            valid = min(sun, earth) >= -st.angular_tolerance and cone['clear'] and (
                not st.enforce_body_rate or rate[0] <= st.tracking_body_rate_max+st.rate_tolerance)
            rows.append(dict(**event, actual_mission_time_s=float(t+clock_shift), valid=bool(valid), sun_margin_deg=float(np.degrees(sun)),
                             earth_margin_deg=float(np.degrees(earth)), body_rate_deg_s=float(np.degrees(rate[0])), terrain_cone_check=cone))
        gaps = np.diff(np.r_[block.start, times, block.end])
        passed = ordered and all(r['valid'] for r in rows) and np.min(gaps) >= -1e-8 and np.max(gaps) <= st.update_gap_max+1e-8
        reports[block.name] = dict(passed=bool(passed), event_count=len(rows), events=rows,
                                  max_update_gap_s=float(max(gaps)), first_update_age_s=float(gaps[0]),
                                  final_estimate_age_s=float(gaps[-1]))
    return dict(passed=bool(labels_valid and reports and all(r['passed'] for r in reports.values())), reports=reports,
                validation='exact_assigned_events_with_chained_independent_states' if history is not None else 'exact_assigned_events_seed_states')

"""6-DOF kinematic truth from the optimizer's validated trajectory.

Translational truth: each hop is re-propagated with the optimizer's own equations (hopper.dynamics) from
its first state, chaining phase to phase, using the linearly interpolated optimized controls. An extra
state integrates the inertial specific force T/m, so the IMU velocity increments are exactly consistent
with the truth trajectory.

Rotational truth: prescribed by the optimizer's attitude convention (hopper.star_tracker): body +Z along
thrust, roll from projected downrange, inertial SLERP during coast. No rotational dynamics.

Frames: F = Moon-fixed; I = synthetic inertial frame aligned with F at first launch, r_I = Rz(w t) r_F,
t = cumulative mission time with zero surface stay (same as the star-tracker analysis).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

import numpy as np

from hopper.dynamics import make_numpy_rhs
from hopper.postprocess import chain, phase_mode
from hopper.star_tracker import CoastOrientation, PhaseHistory
from hopper.tracker_cache import load_trajectory
from hopper.star_tracker import top_mount_axes
from mission.config import parameters as astro_parameters, physical_identity, digest, SETTINGS
from mission.terrain import DemTerrain
from mission.validation import propagate
from mission.trackers import SkyTerrain

from .config import NavConfig
from .rotations import exp_so3, log_so3, rot_z


@dataclass
class Case:
    p: object
    sol: object
    report: dict
    mounts: np.ndarray          # (2, 3) body boresights
    notes: list = field(default_factory=list)
    terrain: object = None
    sky: object = None
    tracker_geometry_cache: dict = field(default_factory=dict)


@dataclass
class HopTruth:
    hop: int
    t: np.ndarray               # (K+1,) cumulative mission time of nav epochs
    dt: np.ndarray              # (K,) step lengths
    r_I: np.ndarray             # (K+1, 3)
    v_I: np.ndarray             # (K+1, 3)
    C_IB: np.ndarray            # (K+1, 3, 3) body -> inertial
    body_rate: np.ndarray       # (K+1,) inertial body rate magnitude, rad/s
    dtheta_b: np.ndarray        # (K, 3) true body rotation over each step
    dv_b: np.ndarray            # (K, 3) true specific-force velocity increment, mid-step body axes
    altitude: np.ndarray        # (K+1,)
    phase: np.ndarray           # (K+1,) index into phase_names
    phase_names: list
    phase_bounds: list          # (name, kind, start, end) cumulative times
    checkpoints: dict           # name -> epoch index
    local_basis_F: np.ndarray   # (K+1, 3, 3) columns downrange, crossrange, up in F
    omega: float
    attitude_jumps: list
    max_rate_deg_s: float

    @property
    def local_basis_I(self):
        return rot_z(self.omega * self.t) @ self.local_basis_F


def load_case(cfg: NavConfig) -> Case:
    """Read the current validated DEM case without changing any Astro artifact."""
    terrain=DemTerrain()
    p=astro_parameters(cfg.raw['trajectory']['case_id'],terrain)
    report=json.loads((cfg.case_dir/'summary.json').read_text())
    saved=json.loads((cfg.case_dir/'parameters.json').read_text())
    if report.get('status')!='numerically_validated_dem_flight':
        raise ValueError(f"Case is not validated: {report.get('status')}")
    if physical_identity(p)!=report['physical_parameters_sha256'] or physical_identity(saved)!=physical_identity(p):
        raise ValueError('Astro physical inputs differ from the saved DEM flight')
    p=replace(p,solver=replace(p.solver,nodes=saved['solver']['nodes']))
    for path,expected in report['source_hashes'].items():
        if digest(cfg.case_dir.parents[1]/path)!=expected:
            raise ValueError('Astro source/data changed: '+path)
    if digest(cfg.case_dir/'sizing.npz')!=report['trajectory_sha256']:
        raise ValueError('Saved sizing source changed')
    sol,validation=load_trajectory(cfg.case_dir/'trajectory.npz',p,cfg.branch,report['baseline_identity'])
    flight=json.loads((cfg.case_dir/'validation.json').read_text())
    trackers=json.loads((cfg.case_dir/'trackers.json').read_text())
    if not validation.get('passed') or not flight['passed'] or not flight['fixed_mass']['passed'] or not trackers['passed']:
        raise ValueError('Complete Astro flight/reserve/tracker acceptance is required')
    mounts=top_mount_axes(np.radians(trackers['mounting_elevations_deg']),np.radians(trackers['mounting_azimuths_deg']))
    notes=['Validated 8-A2200 / 2000 kg dry DEM source; exact current physical/source/data identities verified.',
           'Truth altitude and terminal contact use DEM elevations; tracker cones use local/global lunar terrain.',
           'GN retains the 6 deg/s hardware tracker gate, prescribed attitude and the existing sensor placeholders.']
    return Case(p,sol,report,mounts,notes,terrain,SkyTerrain())


def _hop_phases(case: Case, branch: str, hop: int):
    every = chain(branch, case.p)
    names = [n for n in every if case.sol.phases[n]["hop"] == hop]
    t0 = sum(case.sol.phases[n]["t"] for n in every if case.sol.phases[n]["hop"] < hop)
    return names, t0


def build_hop(case: Case, cfg: NavConfig, hop: int) -> HopTruth:
    p, sol = case.p, case.sol
    names, t_hop0 = _hop_phases(case, cfg.branch, hop)
    omega = p.moon.omega
    rs = p.mission.r_start

    # --- translational truth: chained propagation with specific-force integral -------------------
    segments, bounds = [], []
    y = np.concatenate([sol.phases[names[0]]["X"][:, 0], np.zeros(3)])
    start = t_hop0
    for name in names:
        ph = sol.phases[name]
        T, n = ph["t"], ph["X"].shape[1]
        knots = np.linspace(0.0, T, n)
        U = ph["U"]
        rhs = make_numpy_rhs(p, mode=phase_mode(p, ph))

        def f(t, yy, U=U, knots=knots, rhs=rhs, t_abs0=start):
            u = np.array([np.interp(t, knots, row) for row in U])
            dx = rhs(yy[:7], u)
            fI = rot_z(omega * (t_abs0 + t)) @ u[:3] / yy[6]
            return np.concatenate([dx, fI])

        def surface(t,yy):
            return case.terrain.altitude(rs+yy[:3])
        surface.direction=-1
        surface.terminal=True
        terminal=ph['kind']=='terminal'
        stop=T+SETTINGS['contact_time_tolerance_s'] if terminal else T
        s=propagate(f,stop,y,knots,surface if terminal else None)
        if terminal:
            if not len(s.t_events[0]):raise RuntimeError('GN truth has no first DEM contact: '+name)
            T=float(s.t_events[0][0])
        segments.append((start, T, s.sol))
        bounds.append((name, ph["kind"], start, start + T))
        y = s.y[:, -1]
        start += T
    t_end = start

    # --- nav epochs -------------------------------------------------------------------------------
    dt_nom = cfg.dt
    k_full = int(np.floor((t_end - t_hop0) / dt_nom + 1e-9))
    t = t_hop0 + dt_nom * np.arange(k_full + 1)
    if t_end - t[-1] > 1e-6:
        t = np.append(t, t_end)
    else:
        t[-1] = t_end
    phase_idx = np.array([_phase_at(bounds, ti) for ti in t])

    Y = np.empty((len(t), 10))
    for j, (s0, T, dense) in enumerate(segments):
        sel = phase_idx == j
        if sel.any():
            Y[sel] = dense(np.clip(t[sel] - s0, 0, T)).T
    Rz = rot_z(omega * t)
    r_F = rs + Y[:, :3]
    w = np.array([0., 0., omega])
    r_I = np.einsum("nij,nj->ni", Rz, r_F)
    v_I = np.einsum("nij,nj->ni", Rz, Y[:, 3:6] + np.cross(w, r_F))
    V_I = Y[:, 7:10]

    # --- attitude truth ------------------------------------------------------------------------------
    histories = {}
    for (name, kind, s0, s1) in bounds:
        histories[name] = PhaseHistory(p, name, sol.phases[name], s0)
    ascent_last = [b for b in bounds if b[1] == "ascent"][-1]
    coast = next(b for b in bounds if b[1] == "coast")
    braking_first = next(b for b in bounds if b[1] == "braking")
    first = histories[ascent_last[0]].orientation(np.array([ascent_last[3]]))[0][0]
    last = histories[braking_first[0]].orientation(np.array([braking_first[2]]))[0][0]
    histories[coast[0]].coast = CoastOrientation(first, last, coast[3] - coast[2])

    C = np.empty((len(t), 3, 3))
    rate = np.empty(len(t))
    for j, (name, kind, s0, s1) in enumerate(bounds):
        sel = phase_idx == j
        if sel.any():
            c, rr = histories[name].orientation(t[sel])
            C[sel], rate[sel] = c, rr
    # The optimizer allows instantaneous thrust-direction steps at some phase joins (one interval of the
    # next phase for rotation). A real vehicle slews. Replace each step by a slew at the vehicle tilt-rate
    # limit: a decaying offset rotation applied to the post-jump attitude. Translational truth (thrust
    # vector) is unchanged; only the body attitude lags briefly. Recorded in `attitude_jumps`.
    slew_rate = p.limits.tilt_rate_max
    jumps = []
    for (a, b) in zip(bounds[:-1], bounds[1:]):
        Ca = histories[a[0]].orientation(np.array([a[3]]))[0][0]
        Cb = histories[b[0]].orientation(np.array([b[2]]))[0][0]
        phi = log_so3(Cb.T @ Ca)              # offset that maps post-jump attitude back to pre-jump
        ang = float(np.linalg.norm(phi))
        if ang < np.radians(1e-3):
            continue
        tau = ang / slew_rate
        sel = (t >= a[3] - 1e-9) & (t < a[3] + tau)
        s = (t[sel] - a[3]) / tau
        C[sel] = C[sel] @ exp_so3((1 - s)[:, None] * phi)
        jumps.append(dict(at=f"{a[0]}/{b[0]}", time_s=float(a[3]), angle_deg=float(np.degrees(ang)),
                          smoothed_over_s=float(tau), slew_rate_deg_s=float(np.degrees(slew_rate))))

    # --- IMU truth increments -------------------------------------------------------------------------
    dth = log_so3(np.swapaxes(C[:-1], 1, 2) @ C[1:])
    step_rate = np.linalg.norm(dth, axis=1) / np.diff(t)
    rate = np.maximum(rate, np.maximum(np.r_[step_rate, 0.0], np.r_[0.0, step_rate]))
    C_mid = C[:-1] @ exp_so3(dth / 2)
    dv_b = np.einsum("nji,nj->ni", C_mid, V_I[1:] - V_I[:-1])

    # --- altitude, checkpoints, local frame ---------------------------------------------------------
    alt = case.terrain.altitude(r_F)
    checkpoints = dict(launch=0)
    checkpoints["end_of_ascent_burn"] = int(np.argmin(np.abs(t - ascent_last[3])))
    checkpoints["end_of_coast"] = int(np.argmin(np.abs(t - coast[3])))
    after = np.where((t >= coast[3]) & (alt <= cfg.checkpoint_alt))[0]
    if len(after):
        checkpoints[f"descent_{cfg.checkpoint_alt:.0f}m"] = int(after[0])
    checkpoints["touchdown"] = len(t) - 1
    site0, site1 = r_F[0], r_F[-1]
    up = r_F / np.linalg.norm(r_F, axis=1)[:, None]
    d = site1 - site0
    dr = d - np.sum(d * up, axis=1)[:, None] * up
    dr /= np.linalg.norm(dr, axis=1)[:, None]
    basis = np.stack((dr, np.cross(up, dr), up), axis=-1)

    return HopTruth(hop=hop, t=t, dt=np.diff(t), r_I=r_I, v_I=v_I, C_IB=C, body_rate=rate, dtheta_b=dth,
                    dv_b=dv_b, altitude=alt, phase=phase_idx, phase_names=[b[0] for b in bounds],
                    phase_bounds=bounds, checkpoints=checkpoints, local_basis_F=basis, omega=omega,
                    attitude_jumps=jumps, max_rate_deg_s=float(np.degrees(rate.max())))


def _phase_at(bounds, ti):
    for j, (_, _, s0, s1) in enumerate(bounds):
        if ti < s1 - 1e-9:
            return j
    return len(bounds) - 1


def gravity(r, mu):
    rn = np.linalg.norm(r, axis=-1, keepdims=True)
    return -mu * r / rn**3


def dead_reckon(truth: HopTruth, mu, dth=None, dv=None):
    """Strapdown mechanization (same as the EKF's) from the exact initial state; perfect inputs by default."""
    dth = truth.dtheta_b if dth is None else dth
    dv = truth.dv_b if dv is None else dv
    r, v, C = truth.r_I[0].copy(), truth.v_I[0].copy(), truth.C_IB[0].copy()
    for k, h in enumerate(truth.dt):
        C_mid = C @ exp_so3(dth[k] / 2)
        dvI = C_mid @ dv[k]
        g0 = gravity(r, mu)
        r_pred = r + v * h + 0.5 * (dvI + g0 * h) * h
        g1 = gravity(r_pred, mu)
        v_new = v + dvI + 0.5 * (g0 + g1) * h
        r = r + 0.5 * (v + v_new) * h
        v = v_new
        C = C @ exp_so3(dth[k])
    return r, v, C

"""Measurement models: LN-200S IMU, two star tracker heads, SPLICE TRN position data."""
from __future__ import annotations

import numpy as np
import hashlib

from .config import NavConfig
from .rotations import exp_so3, frame_from_boresight,rot_z


# ---------------------------------------------------------------- IMU
def draw_imu_biases(rng, n, cfg: NavConfig):
    """Constant turn-on biases per run (no in-run drift spec available)."""
    bg = rng.normal(0.0, cfg.imu.gyro_bias, (n, 3))
    ba = rng.normal(0.0, cfg.imu.accel_bias, (n, 3))
    return bg, ba


def imu_measure(rng, truth, k, bg, ba, cfg: NavConfig):
    """Delta-theta and delta-v over nav step k: 400 Hz samples summed onboard. Summed white noise has the
    same statistics as integrating ARW/VRW over the step; constant bias integrates linearly."""
    h = truth.dt[k]
    n = bg.shape[0]
    dth = truth.dtheta_b[k] + bg * h + rng.normal(0.0, cfg.imu.gyro_arw * np.sqrt(h), (n, 3))
    dv = truth.dv_b[k] + ba * h + rng.normal(0.0, cfg.imu.accel_vrw * np.sqrt(h), (n, 3))
    return dth, dv


def imu_range_check(truth, cfg: NavConfig):
    rate = np.linalg.norm(truth.dtheta_b, axis=1) / truth.dt
    acc = np.linalg.norm(truth.dv_b, axis=1) / truth.dt
    return dict(max_body_rate_deg_s=float(np.degrees(rate.max())),
                gyro_range_deg_s=float(np.degrees(cfg.imu.gyro_range)),
                max_specific_force_m_s2=float(acc.max()),
                accel_range_m_s2=float(cfg.imu.accel_range),
                within_range=bool(rate.max() < cfg.imu.gyro_range and acc.max() < cfg.imu.accel_range))


# ---------------------------------------------------------------- star trackers
def tracker_frames(mounts):
    return [frame_from_boresight(b) for b in mounts]


def tracker_covariance(frame, cfg: NavConfig):
    """Body-frame attitude-measurement covariance for one head (cross-boresight / roll accuracy)."""
    s = cfg.tracker
    return frame @ np.diag([s.sigma_cross**2, s.sigma_cross**2, s.sigma_roll**2]) @ frame.T


def terrain_cone_clear(sky,r,axis,half_angle):
    """The Astro fine-cone test, batching its identical ray samples for GN's 16 Hz grid."""
    radius=np.linalg.norm(r)
    nadir=float(np.arccos(np.clip(axis@(-r/radius),-1.,1.)))
    if radius>sky.outer and nadir-half_angle>np.arcsin(sky.outer/radius):
        return True
    reference=np.array([0.,0.,1.]) if abs(axis[2])<.9 else np.array([1.,0.,0.])
    first=np.cross(axis,reference);first/=np.linalg.norm(first)
    second=np.cross(axis,first)
    phi=np.linspace(0.,2*np.pi,32,endpoint=False)
    directions=[axis[None]]
    for theta in np.linspace(0.,half_angle,5)[1:]:
        directions.append(np.cos(theta)*axis+np.sin(theta)*(np.cos(phi)[:,None]*first+np.sin(phi)[:,None]*second))
    directions=np.concatenate(directions)
    rd=directions@r
    if np.any(radius*radius-np.maximum(0.,-rd)**2<sky.inner*sky.inner):
        return False
    discriminant=rd*rd+sky.outer*sky.outer-radius*radius
    points=[]
    for direction,dot,disc in zip(directions,rd,discriminant):
        if disc<0:continue
        length=-dot+np.sqrt(disc)
        if length<=0:continue
        near=min(length,5000.)
        distance=np.unique(np.r_[np.arange(0.,near,30.),np.arange(near,length,240.),length])
        points.append(r+distance[:,None]*direction)
    if not points:return True
    points=np.concatenate(points)
    clearance=np.linalg.norm(points,axis=1)-sky.native.radius-sky.height(points)
    return bool(clearance.min()>=-.001)


def tracker_availability(truth, case, cfg: NavConfig):
    """(K+1, 2) bool: head j can deliver a measurement at epoch k (geometry, body rate, output schedule)."""
    key=hashlib.sha256(truth.t.tobytes()+truth.C_IB.tobytes()+truth.r_I.tobytes()).hexdigest()
    if key not in case.tracker_geometry_cache:
        settings=case.p.star_tracker
        axes=np.einsum('nij,kj->nki',truth.C_IB,case.mounts)
        sun=np.arccos(np.clip(axes@np.asarray(settings.sun_direction_I),-1.,1.))-settings.sun_exclusion
        earth=np.arccos(np.clip(axes@np.asarray(settings.earth_direction_I),-1.,1.))-settings.earth_exclusion
        celestial=(sun>0)&(earth>0)
        rotation=rot_z(-case.p.moon.omega*truth.t)
        positions=np.einsum('nij,nj->ni',rotation,truth.r_I)
        directions=np.einsum('nij,nkj->nki',rotation,axes)
        terrain_ok=np.zeros_like(celestial)
        for k,head in np.argwhere(celestial):
            terrain_ok[k,head]=terrain_cone_clear(case.sky,positions[k],directions[k,head],
                settings.full_cone_fov/2+settings.moon_limb_margin)
        geometry_ok=celestial&terrain_ok
        case.tracker_geometry_cache[key]=(geometry_ok,np.degrees(np.stack([sun,earth],axis=-1)),terrain_ok)
    geometry_ok,margins_deg,terrain_ok=case.tracker_geometry_cache[key]
    rate_ok = truth.body_rate <= cfg.tracker.max_body_rate
    schedule = (np.arange(len(truth.t)) % cfg.steps_per(cfg.tracker.rate)) == 0
    avail = geometry_ok & rate_ok[:, None] & schedule[:, None]
    avail[0] = False                                  # epoch 0 is the initialization
    return avail, dict(geometry_ok=geometry_ok, rate_ok=rate_ok,
                       sun_earth_margins_deg=margins_deg,terrain_ok=terrain_ok)


def tracker_measure(rng, C_true, frame, cfg: NavConfig, n):
    """Measured attitude C_meas = C_true exp(noise), noise in body axes from the head's frame."""
    s = cfg.tracker
    noise_t = rng.normal(0.0, 1.0, (n, 3)) * np.array([s.sigma_cross, s.sigma_cross, s.sigma_roll])
    return C_true @ exp_so3(noise_t @ frame.T)


def update_gaps(truth, avail_any):
    """Gaps between consecutive accepted star tracker updates over the flight (s)."""
    times = truth.t[avail_any]
    if len(times) == 0:
        return dict(updates=0, max_gap_s=float(truth.t[-1] - truth.t[0]), unavailable_s=float(truth.t[-1] - truth.t[0]))
    edges = np.r_[truth.t[0], times, truth.t[-1]]
    gaps = np.diff(edges)
    nominal = 1.0 / 16.0
    long_gaps = [(float(edges[i] - truth.t[0]), float(g)) for i, g in enumerate(gaps) if g > 2.0]
    return dict(updates=int(len(times)), max_gap_s=float(gaps.max()),
                unavailable_s=float(np.sum(gaps[gaps > 1.5 * nominal])),
                gaps_over_2s=[dict(start_s=s, length_s=g) for s, g in long_gaps])


# ---------------------------------------------------------------- SPLICE TRN
def trn_availability(truth, cfg: NavConfig, rate_hz=None):
    rate_hz = cfg.trn.rate if rate_hz is None else rate_hz
    k = np.arange(len(truth.t))
    lo, hi = cfg.trn.altitude_band
    avail = (k % cfg.steps_per(rate_hz) == 0) & (truth.altitude >= lo - 1e-6) & (truth.altitude <= hi)
    avail[0] = False
    return avail


def trn_measure(rng, r_true, sigma, n):
    return r_true + rng.normal(0.0, sigma, (n, 3))

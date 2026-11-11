import dataclasses

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from hopper.dynamics import make_numpy_rhs
from hopper.lambert import lambert, lambert_fixed, min_dv_hop


def test_two_body_lambert_roundtrip():
    mu = 3.986004418e14
    r1 = np.array([7000e3, 0.0, 0.0])
    r2 = np.array([0.0, 8000e3, 1000e3])
    tof = 2500.0
    v1, v2 = lambert(mu, r1, r2, tof)

    def rhs(_t, x):
        return np.concatenate([x[3:], -mu * x[:3] / np.linalg.norm(x[:3]) ** 3])

    sol = solve_ivp(rhs, (0, tof), np.concatenate([r1, v1]), rtol=1e-12, atol=1e-9)
    assert np.linalg.norm(sol.y[:3, -1] - r2) < 1.0
    assert np.linalg.norm(sol.y[3:, -1] - v2) < 1e-3


def test_lunar_hop_lambert_matches_rotating_frame_propagation(params):
    p = params
    tof = 300.0
    r1, r2 = p.mission.r_start, p.mission.r_target
    v1, v2 = lambert_fixed(p, r1, r2, tof)
    f = make_numpy_rhs(p)
    x0 = np.concatenate([np.zeros(3), v1, [1.0]])
    sol = solve_ivp(lambda _t, x: f(x, np.zeros(4)), (0, tof), x0, rtol=1e-12, atol=1e-10,
                    method="DOP853")
    r_end = p.mission.r_start + sol.y[:3, -1]
    assert np.linalg.norm(r_end - r2) < 0.5          # m, on a ~60 km hop
    assert np.linalg.norm(sol.y[3:6, -1] - v2) < 1e-3


def test_min_dv_hop_is_reasonable_and_a_minimum(params):
    base = min_dv_hop(params)
    # ballistic hop of L ~ 60 km: v ~ sqrt(g L) at departure and arrival
    v_est = np.sqrt(params.moon.g_surf * params.mission.L)
    assert 1.5 * v_est < base.dv < 3.0 * v_est
    from hopper.lambert import hop_dv
    assert hop_dv(params, base.tof * 1.1) >= base.dv - 1e-9
    assert hop_dv(params, base.tof * 0.9) >= base.dv - 1e-9
    # departure and arrival speeds are nearly equal (symmetric hop)
    assert base.dv_depart == pytest.approx(base.dv_arrive, rel=0.02)


def test_rotation_effect_is_small_near_pole(params):
    p0 = dataclasses.replace(params, moon=dataclasses.replace(params.moon, omega=0.0))
    assert abs(min_dv_hop(p0).dv - min_dv_hop(params).dv) < 0.5

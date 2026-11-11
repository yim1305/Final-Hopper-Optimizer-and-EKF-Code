import dataclasses

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from hopper.dynamics import (env_accel, jacobi_constant, make_dynamics, make_numpy_rhs,
                             make_scaled_dynamics)


def _propagate(p, x0, t, u=np.zeros(4), mode=None):
    f = make_numpy_rhs(p, mode=mode)
    return solve_ivp(lambda _t, x: f(x, u), (0, t), x0, rtol=1e-12, atol=1e-12, method="DOP853",
                     dense_output=True)


def test_coast_conserves_jacobi_constant(params):
    up = params.mission.r_start / np.linalg.norm(params.mission.r_start)
    x0 = np.concatenate([50 * up, [150.0, 80.0, 200.0], [800.0]])
    sol = _propagate(params, x0, 400.0)
    C = [jacobi_constant(params, sol.sol(t)) for t in np.linspace(0, 400, 9)]
    assert max(C) - min(C) < 1e-6 * abs(C[0])
    assert max(C) - min(C) < 1e-5    # absolute (m^2/s^2); C itself is ~ -2.8e6


def test_free_fall_from_rest(params):
    p = params
    p0 = dataclasses.replace(p, moon=dataclasses.replace(p.moon, omega=0.0))
    up = p.mission.r_start / np.linalg.norm(p.mission.r_start)
    h = 100.0
    x0 = np.concatenate([h * up, np.zeros(3), [800.0]])
    g = p.moon.g_surf
    t_fall = np.sqrt(2 * h / g)
    sol = _propagate(p0, x0, t_fall)
    x = sol.sol(t_fall)
    r_abs = p.mission.r_start + x[:3]
    assert np.linalg.norm(r_abs) - p.moon.radius == pytest.approx(0.0, abs=0.05)
    assert np.linalg.norm(x[3:6]) == pytest.approx(g * t_fall, rel=1e-3)


def test_mass_flow_matches_rocket_equation_for_a_pulsing_only_stage(params):
    # terminal stage: only pulsing engines (all 317 s), so mdot = Gamma / (g0 * 317) exactly
    mode = params.engine.mode("terminal")
    gamma = 3000.0
    x0 = np.concatenate([np.zeros(3), np.zeros(3), [1000.0]])
    sol = _propagate(params, x0, 10.0, np.array([0.0, 0.0, 0.0, gamma]), mode)
    assert sol.sol(10.0)[6] == pytest.approx(1000.0 - 10.0 * gamma / (params.engine.g0 * 317.0), rel=1e-12)


def test_mass_flow_with_large_engines_adds_their_lower_isp(params):
    # ascent stage 1: 3 large engines (277 s) fire; the rest of Gamma is pulsed at 317 s
    mode = params.engine.mode("ascent", 0)
    assert dict(mode.fixed)["large"] == 3
    gamma = 25000.0
    g0, large = params.engine.g0, params.engine.classes["large"]
    expected = 3 * large.thrust / (g0 * 277.0) + (gamma - 3 * large.thrust) / (g0 * 317.0)
    assert mode.mdot(gamma) == pytest.approx(expected, rel=1e-12)
    x0 = np.concatenate([np.zeros(3), np.zeros(3), [4000.0]])
    sol = _propagate(params, x0, 10.0, np.array([0.0, 0.0, 0.0, gamma]), mode)
    assert sol.sol(10.0)[6] == pytest.approx(4000.0 - 10.0 * expected, rel=1e-12)


def test_coast_burns_no_propellant_whatever_the_control(params):
    x0 = np.concatenate([np.zeros(3), np.zeros(3), [1000.0]])
    sol = _propagate(params, x0, 10.0, np.zeros(4), None)
    assert sol.sol(10.0)[6] == pytest.approx(1000.0, rel=1e-14)


def test_casadi_matches_numpy(params):
    mode = params.engine.mode("ascent", 1)       # one large engine firing plus pulsing: exercises c0 and c1
    f_ca = make_dynamics(params, mode=mode)
    f_np = make_numpy_rhs(params, mode=mode)
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(size=3) * 1e3, rng.normal(size=3) * 100, [750.0]])
    u = np.array([100.0, -50.0, 900.0, 9000.0])
    assert np.allclose(np.array(f_ca(x, u)).ravel(), f_np(x, u), rtol=1e-12, atol=1e-12)


def test_scaled_dynamics_consistent_with_dimensional(params):
    s = params.scales
    mode = params.engine.mode("braking", 0)
    f_nd = make_scaled_dynamics(params, mode=mode)
    f_np = make_numpy_rhs(params, mode=mode)
    x = np.array([1000.0, -2000.0, 500.0, 120.0, 30.0, -10.0, 800.0])
    u = np.array([300.0, -100.0, 1500.0, 22000.0])
    xn = np.concatenate([x[:3] / s.L, x[3:6] / s.V, [x[6] / s.M]])
    un = np.concatenate([u[:3] / s.F, [u[3] / s.F]])
    dn = np.array(f_nd(xn, un)).ravel()
    d = f_np(x, u)
    back = np.concatenate([dn[:3] * s.L / s.T, dn[3:6] * s.V / s.T, [dn[6] * s.M / s.T]])
    assert np.allclose(back, d, rtol=1e-10, atol=1e-10)


def test_env_accel_at_rest_is_about_g(params):
    a = env_accel(params, np.zeros(3), np.zeros(3))
    assert np.linalg.norm(a) == pytest.approx(params.moon.g_surf, rel=1e-3)

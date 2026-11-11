import numpy as np
import pytest

from hopper.divert import gate_state, solve_branch
from hopper.lambert import min_dv_hop
from hopper.postprocess import round_trip_summary, thrust_discard_flags
from hopper.problem import Problem
from hopper.postprocess import chain
from hopper.staging import solve_round_trip
from hopper.validate import repropagate, sensor_report

BRANCH = "uprange"


@pytest.fixture(scope="module")
def params(make_params):
    """Propulsion integration tests disable pointing, preserving the config default."""
    return make_params({"sensor": {"enforce_pointing": False}})


@pytest.fixture(scope="module")
def rt(params):
    """Fixed wet mass (config m_wet), maximise the final mass."""
    sol, _, _ = solve_round_trip(params, BRANCH, mode="maximise_margin")
    lam1 = min_dv_hop(params)
    lam2 = min_dv_hop(params, r1=params.mission.sites[BRANCH], r2=params.mission.r_start)
    return sol, round_trip_summary(params, sol, BRANCH, lam1.dv, lam2.dv), lam1, lam2


@pytest.fixture(scope="module")
def sized(params):
    sol, _, _ = solve_round_trip(params, BRANCH, mode="size")
    return sol


def test_round_trip_converges(rt):
    assert rt[0].success, rt[0].status


def test_hop2_returns_to_start_site(params, rt):
    x_end = rt[0].phases[f"terminal2_{BRANCH}"]["X"][:, -1]
    assert np.linalg.norm(x_end[0:3]) <= params.mission.landing_tolerance + 0.01                      # offset frame origin = start site
    assert x_end[6] >= params.vehicle.m_dry


def test_hop1_lands_on_divert_site(params, rt):
    x_end = rt[0].phases[f"terminal_{BRANCH}"]["X"][:, -1]
    site = params.mission.sites[BRANCH] - params.mission.r_start
    assert np.linalg.norm(x_end[0:3] - site) <= params.mission.landing_tolerance + 0.01


def test_delta_v_exceeds_lambert_and_legs_are_similar(rt):
    _, s, lam1, lam2 = rt
    assert s["hop1"]["finite_burn_dv"] > lam1.dv
    assert s["hop2"]["finite_burn_dv"] > lam2.dv
    assert s["total"]["finite_burn_dv"] >= lam1.dv + lam2.dv
    assert s["hop2"]["finite_burn_dv"] == pytest.approx(s["hop1"]["finite_burn_dv"], rel=0.10)
    assert lam2.dv == pytest.approx(lam1.dv, rel=0.01)


def test_four_engine_starts_and_no_thrust_discarded(params, rt):
    sol, s, *_ = rt
    assert s["total"]["engine_starts"] == 4
    assert thrust_discard_flags(params, sol) == {}


def test_gate_is_above_target_and_above_glide_floor(params, rt):
    gate = gate_state(rt[0])["x"]
    r_abs = gate[0:3] + params.mission.r_start
    d = r_abs - params.mission.r_target
    horiz = d - (d @ params.mission.up) * params.mission.up
    assert np.linalg.norm(horiz) < 1e-3
    h = np.linalg.norm(r_abs) - params.moon.radius
    # IPOPT's dimensionless feasibility tolerance corresponds to millimetres at hop length scale.
    h_tol = 2 * params.solver.ipopt["tol"] * params.scales.L
    assert h >= np.tan(params.limits.glide_slope) * params.mission.divert_radius - h_tol
    assert params.mission.gate_alt_bounds[0] - h_tol <= h <= params.mission.gate_alt_bounds[1] + h_tol
    vup = gate[3:6] @ params.mission.up
    assert vup < -1e-3


def test_no_hover_and_moving_gate_is_linked_to_terminal(params, rt, sized):
    for sol in (rt[0], sized):
        assert list(sol.phases) == chain(BRANCH, params)
        assert all(ph["kind"] != "hover" for ph in sol.phases.values())
        for braking, terminal in (("observation", f"terminal_{BRANCH}"),
                                  (f"observation2_{BRANCH}", f"terminal2_{BRANCH}")):
            gate = sol.phases[braking]["X"][:, -1]
            np.testing.assert_allclose(sol.phases[terminal]["X"][:, 0], gate, atol=1e-5, rtol=0)
            assert np.linalg.norm(gate[3:6]) > 1e-3


def test_both_touchdowns_are_upright_and_descending(params, rt, sized):
    for sol in (rt[0], sized):
        for name, site in ((f"terminal_{BRANCH}", params.mission.sites[BRANCH]),
                           (f"terminal2_{BRANCH}", params.mission.r_start)):
            ph = sol.phases[name]
            site = params.mission.r_start + ph["X"][0:3, -1]
            up = site / np.linalg.norm(site)
            T = ph["U"][0:3, -1]
            angle = np.arccos(np.clip((T @ up) / np.linalg.norm(T), -1, 1))
            assert angle <= params.limits.td_tilt_max + np.radians(1e-3)
            v = ph["X"][3:6, -1]
            vup = v @ up
            assert -params.limits.td_vert_speed_max - 1e-5 <= vup <= 1e-5
            assert np.linalg.norm(v - vup * up) <= params.limits.td_horiz_speed_max + 1e-5
        assert thrust_discard_flags(params, sol) == {}


def test_size_mode_meets_reserve_factor(params, sized):
    assert sized.success, sized.status
    md, f = params.vehicle.m_dry, params.mission.reserve_factor
    assert sized.m_wet > md
    assert (sized.m_wet - md) / (sized.m_wet - sized.z) == pytest.approx(f, rel=1e-4)


def test_sized_mass_is_smaller_than_the_oversized_config_mass(params, rt, sized):
    # at the 5000 kg placeholder the loaded/used ratio exceeds 1.1, so the sized wet mass must be lower
    assert rt[1]["total"]["loaded_over_used"] > params.mission.reserve_factor
    assert sized.m_wet < params.vehicle.m_wet


def test_sized_mass_reproduced_by_a_fixed_mass_run(params, make_params, sized):
    p2 = make_params({"vehicle": {"m_wet": sized.m_wet}, "sensor": {"enforce_pointing": False}})
    s2, _, _ = solve_round_trip(p2, BRANCH, mode="maximise_margin",
                               guess=Problem.guess_from_solution(sized))
    assert s2.success
    assert (s2.m_wet - p2.vehicle.m_dry) / (s2.m_wet - s2.z) == pytest.approx(params.mission.reserve_factor, rel=1e-3)
    assert s2.z == pytest.approx(sized.z, rel=1e-3)


def test_propellant_split_adds_up(params, rt):
    s = rt[1]
    for key in ("propellant_used_split", "propellant_loaded_split"):
        sp = s[key]
        assert sp["oxidizer"] / sp["fuel"] == pytest.approx(params.engine.mixture_ratio)
    used = s["propellant_used_split"]
    assert used["fuel"] + used["oxidizer"] == pytest.approx(s["total"]["propellant_used"])


def test_uprange_divert_reachable_from_moving_gate(params, sized):
    gate = gate_state(sized)
    res = solve_branch(params, gate, BRANCH, guess=sized.phases[f"terminal_{BRANCH}"])
    assert res.success, res.status
    np.testing.assert_allclose(res.phase["X"][:, 0], gate["x"], atol=1e-5, rtol=0)
    assert res.touchdown_pos_err < 0.05, res.touchdown_pos_err
    assert -params.limits.td_vert_speed_max - 1e-4 <= res.touchdown_up_speed <= 1e-4
    assert res.touchdown_tilt_deg <= np.degrees(params.limits.td_tilt_max) + 1e-3


def test_repropagation_matches_collocation(params, rt):
    rep, td = repropagate(params, rt[0], BRANCH)
    assert max(r["pos_err"] for r in rep) < 0.05
    assert td["hop1"]["pos_err"] < 0.05
    assert td["pos_err"] < 0.05
    assert -td["up_speed"] <= params.limits.td_vert_speed_max + 1e-4


def test_both_observation_windows(params, rt, sized):
    for sol in (rt[0], sized):
        report = sensor_report(params, sol, BRANCH)
        assert len(report) == 2
        for row in report.values():
            assert row["passed"], row
        for terminal in (f"terminal_{BRANCH}", f"terminal2_{BRANCH}"):
            r = params.mission.r_start + sol.phases[terminal]["X"][:3, -1]
            assert abs(np.linalg.norm(r) - params.moon.radius) < 0.01
        np.testing.assert_allclose(sol.phases[f"rise2_{BRANCH}"]["X"][:3, 0],
                                   sol.phases[f"terminal_{BRANCH}"]["X"][:3, -1], atol=1e-5)


def test_class_accounting_and_shutdown_schedule(params, rt, sized):
    from hopper.postprocess import class_usage, phase_mode, burn_activations
    for sol in (rt[0], sized):
        assert sol.success, sol.status
        assert sol.primal_infeasibility < 1e-6
        assert sol.dual_infeasibility < 1e-6
        prop, starts = class_usage(params, sol, BRANCH)
        assert sum(prop.values()) == pytest.approx(sol.m_wet-sol.z, abs=1e-4)
        assert all(v >= 0 for v in prop.values())
        assert starts == dict(large=12, small=None, rcs=None)
        assert burn_activations(params, sol, BRANCH) == dict(large=12, small=16, rcs=32)
        last = None
        for ph in sol.phases.values():
            if ph["kind"] == "rise" or not ph["powered"]:
                last = None
            if not ph["powered"]:
                continue
            mode = phase_mode(params, ph)
            count = dict(mode.fixed).get("large", 0)
            if last is not None:
                assert count <= last
            last = count
            assert min(ph["U"][3]) >= mode.gamma_lo-0.01
            assert max(ph["U"][3]) <= mode.gamma_hi+0.01
            if ph["kind"] in ("observation", "terminal"):
                assert count == 0

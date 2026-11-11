"""Engine model: classes, per-phase pools, burn-stage modes and the relight rule."""
import pytest

from hopper.params import load_params

LARGE, SMALL, RCS = 7060.0, 1310.0, 200.0


def test_pulsing_capacity_per_pool(params):
    e = params.engine
    assert e.mode("terminal").pulse_capacity == pytest.approx(4 * SMALL + 8 * RCS)
    assert e.mode("observation").pulse_capacity == pytest.approx(4 * SMALL + 8 * RCS)
    assert e.mode("ascent", 0).pulse_capacity == pytest.approx(4 * SMALL + 8 * RCS)


def test_stage_thrust_bands(params):
    e = params.engine
    m = e.mode("ascent", 0)                    # 3 large fire
    assert m.thrust_fixed == pytest.approx(3 * LARGE)
    assert m.gamma_lo == pytest.approx(3 * LARGE)
    assert m.gamma_hi == pytest.approx(3 * LARGE + 4 * SMALL + 8 * RCS)
    m = e.mode("terminal")                     # descent pool: pulsing only, floor is the placeholder fraction
    assert m.thrust_fixed == 0.0
    assert m.gamma_lo == pytest.approx(e.pulse_min_fraction * (4 * SMALL + 8 * RCS))
    assert m.gamma_hi == pytest.approx(4 * SMALL + 8 * RCS)


def test_descent_pool_has_no_large_engines(params):
    for kind in ("observation", "terminal"):
        assert "large" not in params.engine.pools[kind]
        assert all(n == 0 for _, n in params.engine.mode(kind).fixed)


def test_mass_flow_is_affine_and_matches_isp_at_full_thrust(params):
    e = params.engine
    m = e.mode("terminal")
    assert m.mdot(m.gamma_hi) == pytest.approx(m.gamma_hi / (e.g0 * 317.0))
    assert m.ve_hi == pytest.approx(e.g0 * 317.0)
    m = e.mode("rise")                          # all engines: mass-flow-weighted Isp
    full = 3 * e.classes["large"].mdot + 4 * e.classes["small"].mdot + 8 * e.classes["rcs"].mdot
    assert m.mdot(m.gamma_hi) == pytest.approx(full)


def test_large_engines_cost_more_propellant_per_newton(params):
    m = params.engine.mode("ascent", 0)
    assert m.c1 == pytest.approx(1.0 / (params.engine.g0 * 317.0))       # marginal cost is the pulsing engines'
    assert m.mdot(m.gamma_lo) / m.gamma_lo > m.c1                       # large engines are the expensive part


def test_one_stage_for_rise_observation_terminal(make_params):
    with pytest.raises(ValueError, match="exactly one stage"):
        make_params({"engine": {"stages": {"terminal": [{}, {}]}}})


def test_large_engines_cannot_relight_inside_a_burn(make_params):
    # ascent starting with more large engines than the rise that feeds it
    with pytest.raises(ValueError, match="cannot relight"):
        make_params({"engine": {"stages": {"rise": [{"large": 1}], "ascent": [{"large": 3}, {"large": 1}]}}})
    # counts rising inside the ascent
    with pytest.raises(ValueError, match="cannot relight"):
        make_params({"engine": {"stages": {"ascent": [{"large": 1}, {"large": 3}]}}})


def test_large_engines_may_relight_after_the_coast(params):
    e = params.engine
    assert dict(e.mode("ascent", 1).fixed)["large"] == 1
    assert dict(e.mode("braking", 0).fixed)["large"] == 3        # more than the last ascent stage: after the coast


def test_return_schedule_uses_its_own_modes_dynamics_and_cache(make_params, tmp_path):
    import numpy as np
    from hopper.problem import Problem
    from hopper.postprocess import chain, phase_mode
    from hopper.tracker_cache import save_trajectory, load_trajectory
    from test_star_tracker import synthetic_solution
    p = make_params({"engine": {"return_stages": {
        "rise": [{"large": 2}], "ascent": [{"large": 2}, {"large": 1}, {"large": 0}],
        "braking": [{"large": 2}, {"large": 0}]}}})
    assert p.engine.stage_count("ascent") == 2
    assert p.engine.stage_count("ascent", hop=2) == 3
    assert "ascent2_uprange_s3" in chain("uprange", p)
    assert "ascent_s3" not in chain("uprange", p)
    pr = Problem(p)
    a, b = pr.hop1[0], pr.hop2[0]
    x = np.array([0., 0., 0., 0., 0., 0., 1.])
    u = np.array([1., 0., 0., 1.])
    # The same thrust has different mass flow when two vs three fixed engines fire.
    assert float(pr.dynamics_of(a)(x, u)[6]) != pytest.approx(float(pr.dynamics_of(b)(x, u)[6]))
    sol = synthetic_solution(p)
    assert dict(phase_mode(p, sol.phases["rise2_uprange"]).fixed)["large"] == 2
    cache = tmp_path / "asymmetric.npz"
    save_trajectory(cache, p, sol, "uprange", "test", dict(passed=True))
    loaded, _ = load_trajectory(cache, p, "uprange", "test")
    assert loaded.phases["ascent2_uprange_s3"]["hop"] == 2
    assert loaded.phases["ascent2_uprange_s3"]["stage"] == 2
    from hopper.initial_guess import build_guess
    from hopper.staging import stage_guess
    assert set(build_guess(p, "uprange")) == set(sol.phases)
    relaxed = Problem(p, relaxed=True)
    relaxed.set_guess()
    assert [ph.hop for ph in relaxed.hop2] == [2] * 6
    # A single relaxed trajectory expands to the distinct schedules of both hops.
    from dataclasses import replace
    from test_star_tracker import synthetic_solution as synthetic
    plain = replace(p, engine=replace(p.engine, modes={k: [v[0]] for k, v in p.engine.modes.items()}, return_modes={}))
    assert list(stage_guess(p, synthetic(plain).phases)) == list(sol.phases)


def test_return_schedule_cannot_relight_inside_burn(make_params):
    with pytest.raises(ValueError, match="cannot relight"):
        make_params({"engine": {"return_stages": {"ascent": [{"large": 1}, {"large": 2}]}}})
    with pytest.raises(ValueError, match="cannot relight"):
        make_params({"engine": {"return_stages": {"rise": [{"large": 1}]}}})


def test_unknown_or_unavailable_engines_are_rejected(make_params):
    with pytest.raises(ValueError, match="unknown engine class"):
        make_params({"engine": {"stages": {"ascent": [{"huge": 1}]}}})
    with pytest.raises(ValueError, match="not a non-pulsing class"):
        make_params({"engine": {"stages": {"terminal": [{"large": 1}]}}})
    with pytest.raises(ValueError, match="count must be"):
        make_params({"engine": {"stages": {"ascent": [{"large": 4}, {"large": 1}]}}})


@pytest.mark.parametrize("kind,stage", [("terminal", 0), ("ascent", 0), ("ascent", 1)])
def test_mixed_pulsing_isp_conserves_individual_mass_flows(make_params, kind, stage):
    import numpy as np
    from hopper.postprocess import propellant_by_class
    from hopper.dynamics import make_dynamics, make_numpy_rhs
    p = make_params({"engine": {"classes": {"small": {"thrust": 1000., "isp": 321.}}}})
    e, duty, duration = p.engine, .37, 12.
    mode = e.mode(kind, stage)
    gamma = mode.thrust_fixed + duty * mode.pulse_capacity
    fixed_flow = sum(n * e.classes[c].mdot for c, n in mode.fixed)
    expected = fixed_flow + duty * (4 * 1000 / (e.g0 * 321) + 8 * 200 / (e.g0 * 317))
    assert mode.mdot(gamma) == pytest.approx(expected)
    X = np.zeros((7, 2))
    X[6] = [5000., 5000. - expected * duration]
    split = propellant_by_class(p, dict(kind=kind, stage=stage, hop=1, powered=True, X=X, t=duration))
    assert split["small"] == pytest.approx(duty * 4 * e.classes["small"].mdot * duration)
    assert split["rcs"] == pytest.approx(duty * 8 * e.classes["rcs"].mdot * duration)
    assert sum(split.values()) == pytest.approx(expected * duration)
    u = np.array([0., 0., gamma, gamma])
    assert make_numpy_rhs(p, mode=mode)(X[:, 0], u)[6] == pytest.approx(-expected)
    assert float(make_dynamics(p, mode=mode)(X[:, 0], u)[6]) == pytest.approx(-expected)


def test_relaxed_mixed_pulse_pool_matches_all_engine_flows(make_params):
    p = make_params({"engine": {"classes": {"small": {"isp": 321.}}}})
    mode = p.engine.relaxed["ascent"]
    expected = sum(c.count * c.mdot for c in p.engine.classes.values())
    assert mode.mdot(mode.gamma_hi) == pytest.approx(expected)


def test_continuous_burn_cap_spans_stages_and_resets_after_coast(make_params):
    import numpy as np
    from hopper.problem import Problem
    plain = Problem(make_params())
    p = make_params({"engine": {"classes": {"large": {"max_continuous_burn_s": 66.}}}})
    pr = Problem(p)
    assert pr.initial_boundary_offset == plain.initial_boundary_offset + 4
    for hop in (pr.hop1, pr.hop2):
        for ph in hop:
            if ph.powered:
                pr.opti.set_initial(ph.tdur, 10. / pr.s.T)
    cap_rows = pr.opti.g[pr.initial_boundary_offset-4:pr.initial_boundary_offset]
    values = np.asarray(pr.opti.debug.value(cap_rows, pr.opti.initial())).ravel()
    # Each ascent: rise + two active stages. Descent: two nonzero-AR46 stages;
    # the zero-count braking tail and small/RCS landing phases do not contribute.
    assert values == pytest.approx([30/66, 20/66, 30/66, 20/66])
    assert pr.initial_boundary_offset > 0


def test_mean_pulsing_does_not_define_a_continuous_burn_limit(make_params):
    with pytest.raises(ValueError, match="positive non-pulsing limit"):
        make_params({"engine": {"classes": {"rcs": {"max_continuous_burn_s": 66.}}}})


def test_prechecks_flag_a_vehicle_that_cannot_lift_off(make_params):
    from hopper.params import report_prechecks
    p = make_params({"vehicle": {"m_wet": 40000.0, "m_dry": 1500.0}})
    assert any("lift" in w for w in report_prechecks(p, out=lambda *_: None))


def test_star_tracker_accepts_multi_stage_burns(make_params):
    p = make_params({"star_tracker": {"enabled": True}})
    assert p.star_tracker.enabled
    assert p.engine.stage_count("ascent") == 2


def test_tracker_coverage_and_cache_include_every_stage(params, tmp_path):
    from dataclasses import replace
    from hopper.star_tracker import build_blocks
    from hopper.tracker_cache import save_trajectory, load_trajectory
    from test_star_tracker import synthetic_solution
    p = replace(params, star_tracker=replace(params.star_tracker, enabled=True))
    sol = synthetic_solution(p)
    blocks = build_blocks(p, sol, "uprange")
    assert [ph.name for ph in blocks[0].phases] == ["ascent", "ascent_s2", "coast"]
    assert [ph.name for ph in blocks[1].phases] == ["ascent2_uprange", "ascent2_uprange_s2", "coast2_uprange"]
    assert blocks[0].end == pytest.approx(8.)
    cache = tmp_path / "staged.npz"
    save_trajectory(cache, p, sol, "uprange", "test", dict(passed=True))
    loaded, _ = load_trajectory(cache, p, "uprange", "test")
    assert list(loaded.phases) == list(sol.phases)
    assert loaded.phases["ascent_s2"]["stage"] == 1
    assert loaded.phases["braking_s3"]["stage"] == 2


def test_tracker_rejects_missing_stage_instead_of_silently_skipping(params):
    from dataclasses import replace
    from hopper.star_tracker import build_blocks
    from test_star_tracker import synthetic_solution
    p = replace(params, star_tracker=replace(params.star_tracker, enabled=True))
    sol = synthetic_solution(p)
    sol.phases.pop("ascent_s2")
    with pytest.raises(ValueError, match="phase/stage order"):
        build_blocks(p, sol, "uprange")


def test_braking_seed_does_not_shut_down_from_one_low_first_node(params):
    import numpy as np
    from hopper.staging import _snap_counts, _stage_edges
    mode = params.engine.relaxed["braking"]
    gamma = np.array([mode.gamma_lo] + [mode.gamma_hi] * 15 + [mode.gamma_lo] * 3)
    ph = dict(kind="braking", U=np.vstack([np.zeros((3, len(gamma))), gamma]), t=80.0)
    counts = _snap_counts(params, ph, [3, 1, 0])
    assert np.all(counts[1:15] == 3)
    assert np.all(np.diff(counts) <= 0)
    assert counts[-1] == 0
    edges = _stage_edges(params, ph, [3, 1, 0])
    assert edges[1] > 50.0
    assert np.all(np.diff(edges) > 0)


def test_seed_failure_stops_before_fixed_engine_solve(params, monkeypatch):
    from types import SimpleNamespace
    import hopper.staging as staging
    calls = []
    class FailingProblem:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)
        def set_guess(self, *args, **kwargs):
            pass
        def solve(self, *args):
            return SimpleNamespace(success=False, status="Maximum_WallTime_Exceeded", iterations=1,
                                   m_wet=5000.0, primal_infeasibility=1., dual_infeasibility=1.)
    monkeypatch.setattr(staging, "Problem", FailingProblem)
    sol, _, _ = staging.solve_round_trip(params)
    assert not sol.success
    assert len(calls) == 1
    assert calls[0]["relaxed"]

"""Numerical tracker tests use explicit synthetic attitudes, never an optimizer per geometry case."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import casadi as ca
from scipy.spatial.transform import Rotation

from hopper.params import load_params
from hopper.postprocess import chain
from hopper.problem import Problem, Solution
from hopper.star_tracker import (
    CoastOrientation,
    OrientationError,
    Samples,
    angle_between,
    build_blocks,
    cadence_feasible,
    canonical_mount,
    evaluate_schedule,
    geometry_margins,
    make_schedule,
    mount_axis,
    powered_frame,
    refine_intervals,
    rotation_z,
    signed_margins,
    search_mount,
    validate_mount,
)
from hopper.tracker_cache import baseline_identity, load_trajectory, save_trajectory


@pytest.fixture
def make_params(single_stage_params_factory):
    return single_stage_params_factory


@pytest.fixture
def tracker_params(make_params):
    return make_params(
        {
            "star_tracker": {
                "enabled": True,
                "enforce_body_rate": True,
                "mount_model": "antipodal",
                "antipodal": True,
            }
        }
    )


def synthetic_solution(p, branch="uprange"):
    """Stationary above launch, fixed upward thrust; geometry/timing fixture, not a feasible flight."""
    phases = {}
    up = p.mission.r_start / np.linalg.norm(p.mission.r_start)
    from hopper.phases import stage_nodes

    counters = {}
    for name in chain(branch, p):
        kind = name.split("2_")[0] if "2_" in name else name.split("_")[0]
        hop = 2 if "2_" in name else 1
        n = stage_nodes(p.solver.nodes[kind], p.engine.stage_count(kind, hop=hop)) + 1
        group = ("2_" in name, kind)
        stage = counters.get(group, 0)
        counters[group] = stage + 1
        X = np.zeros((7, n))
        X[:3] = up[:, None] * 1000
        X[6] = p.vehicle.m_wet
        U = np.zeros((4, n))
        if kind != "coast":
            U[:3] = up[:, None] * p.engine.T_max
            U[3] = p.engine.T_max
        phases[name] = dict(
            X=X, U=U, t=2.0, kind=kind, powered=kind != "coast", branch=branch, stage=stage, hop=hop
        )
    return Solution(phases, p.vehicle.m_wet, "synthetic_test_only", True, 0, 0.0, p.vehicle.m_wet)


def samples_for_axis(axis, radius=10.0):
    return Samples(
        np.array([0.0]),
        np.eye(3)[None],
        np.array([[0.0, 0.0, radius]]),
        np.array([0.0]),
        np.array([np.pi / 2]),
    )


@pytest.mark.parametrize(
    "override",
    [
        {"n_trackers": 1},
        {"mount_model": "antipodal", "antipodal": False},
        {"full_cone_fov_deg": 0},
        {"enforce_body_rate": "false"},
        {"tracking_body_rate_max_deg_s": float("nan")},
        {"update_gap_max_s": 0},
        {"moon_limb_reference": "boresight"},
        {"acquisition_time_s": 0.1},
        {"switching_delay_s": 0.1},
        {"initial_estimate_age_s": 0.1},
        {"required_phase_kinds": ["observation"]},
        {"body_roll_policy": "free"},
        {"assumed_geometry": {"earth_direction_I": [0, 0, 0]}},
        {"assumed_geometry": {"surface_stay_s": 5}},
        {"assumed_geometry": {"directions_frozen_in": "F"}},
        {"mount_search": {"tilt_bounds_deg": [90, 0]}},
        {"analysis": {"sample_step_s": 0.1}},
    ],
)
def test_invalid_tracker_configuration(make_params, override):
    with pytest.raises(ValueError, match="star_tracker"):
        make_params({"star_tracker": override})


def test_missing_and_disabled_tracker_preserve_nlp(make_params, tmp_path):
    import yaml
    from conftest import CONFIG

    cfg = yaml.safe_load(open(CONFIG))
    cfg["engine"]["stages"].update(ascent=[dict(large=3)], braking=[dict(large=3)])
    cfg.pop("star_tracker")
    path = tmp_path / "old_config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    old = load_params(str(path))
    disabled = make_params({"star_tracker": {"enabled": False}})
    enabled = make_params({"star_tracker": {"enabled": True}})
    assert old.star_tracker is None
    problems = [Problem(p) for p in (old, disabled, enabled)]
    assert len({(pr.opti.nx, pr.opti.ng) for pr in problems}) == 1
    objective_values = [
        float(ca.Function("objective", [pr.opti.x], [pr.opti.f])(np.linspace(0.1, 1, pr.opti.nx)))
        for pr in problems
    ]
    assert objective_values == pytest.approx([objective_values[0]] * 3)


def test_units_directions_and_config(tracker_params):
    st = tracker_params.star_tracker
    assert st.full_cone_fov == pytest.approx(np.radians(20))
    assert st.tracking_body_rate_max == pytest.approx(np.radians(6))
    assert st.update_gap_max == 1.0
    assert np.linalg.norm(st.sun_direction_I) == pytest.approx(1.0)
    assert np.degrees(
        angle_between(np.array(st.sun_direction_I), np.array(st.earth_direction_I))
    ) == pytest.approx(87.1)


def test_exclusion_boundaries_and_full_fov(tracker_params):
    st = tracker_params.star_tracker
    b = np.array([0.0, 0.0, 1.0])
    boundary = np.array([np.sin(st.sun_exclusion), 0.0, np.cos(st.sun_exclusion)])
    settings = replace(st, sun_direction_I=tuple(boundary), earth_direction_I=tuple(boundary))
    sample = samples_for_axis(b)
    margins = geometry_margins(sample, b, settings)[0, 0]
    assert margins[:2] == pytest.approx([0.0, 0.0], abs=1e-14)
    assert np.all(signed_margins(sample, b, settings)[0, 0] >= 0)
    inside = replace(settings, sun_direction_I=tuple(mount_axis(st.sun_exclusion - 1e-4, 0)))
    assert signed_margins(sample, b, inside)[0, 0, 0] < 0
    # 100 degrees from nadir is exactly the full-FOV limb boundary at the surface.
    surface_axis = mount_axis(np.radians(80), 0)
    assert geometry_margins(sample, surface_axis, settings)[0, 0, 2] == pytest.approx(
        0.0, abs=1e-14
    )
    assert geometry_margins(sample, mount_axis(np.radians(81), 0), settings)[0, 0, 2] < 0


def test_must_clear_all_bodies_on_one_tracker(tracker_params):
    st = replace(
        tracker_params.star_tracker,
        sun_direction_I=(0.0, 0.0, 1.0),
        earth_direction_I=(0.0, 0.0, -1.0),
    )
    m = signed_margins(samples_for_axis(None), np.array([0.0, 0.0, 1.0]), st)[0]
    assert m[:, :2].max(axis=0).min() >= 0  # wrong per-body OR would pass
    assert not np.any(np.all(m >= 0, axis=1))


def test_right_handed_frame_and_antipodal_transforms():
    C, _ = powered_frame(np.array([0.2, 0.3, 1.0]), np.zeros(3), np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(C.T @ C, np.eye(3), atol=1e-14)
    assert np.linalg.det(C) == pytest.approx(1.0)
    b = mount_axis(0.8, 1.2)
    np.testing.assert_allclose(C @ (-b), -(C @ b))
    a, beta = canonical_mount(np.pi, 0.9)
    assert a == pytest.approx(0.0, abs=1e-12)
    assert beta == 0.0
    assert canonical_mount(np.pi / 2, 0.4) == pytest.approx(canonical_mount(np.pi / 2, 0.4 + np.pi))


def test_two_explicit_upward_boresights_do_not_create_downward_companion(tracker_params):
    settings = tracker_params.star_tracker
    sample = samples_for_axis(None)
    axes = np.array([mount_axis(np.radians(30), 0), mount_axis(np.radians(40), np.pi)])
    explicit = geometry_margins(sample, axes, settings)
    assert explicit[0, 0, 2] > 0 and explicit[0, 1, 2] > 0
    for j in range(2):
        np.testing.assert_allclose(
            explicit[:, j], geometry_margins(sample, axes[j], settings)[:, 0]
        )
    assert geometry_margins(sample, axes[0], settings)[0, 1, 2] < 0
    with pytest.raises(ValueError, match="unit"):
        geometry_margins(sample, axes * 2, settings)


def test_top_plane_clearance_covers_full_cone_not_only_boresight():
    from hopper.star_tracker import top_mount_axes

    top_axes = lambda angles: top_mount_axes(angles, [0.0, np.pi])
    plane_clearances = lambda axes, fov: np.arcsin(np.clip(np.asarray(axes)[:, 2], -1, 1)) - fov / 2
    axes = top_axes(np.radians([30, 40]))
    assert np.all(axes[:, 2] > 0)
    assert axes[0, 0] > 0 and axes[1, 0] < 0
    assert plane_clearances(axes, np.radians(20)) == pytest.approx(np.radians([20, 30]))
    boundary = top_axes(np.radians([10, 5]))
    assert plane_clearances(boundary, np.radians(20))[0] == pytest.approx(0, abs=1e-14)
    assert plane_clearances(boundary, np.radians(20))[1] < 0  # upward axis alone is insufficient


def test_powered_inertial_rate_includes_roll_and_frame_rotation():
    heading = np.array([1.0, 0.0, 0.0])
    T, Td = np.array([0.999, 0.01, 0.02]), np.array([0.0, 0.02, -0.01])
    C, rate = powered_frame(T, Td, heading, 0.003)
    eps = 1e-6
    first = rotation_z(-0.003 * eps) @ powered_frame(T - eps * Td, Td, heading)[0]
    last = rotation_z(0.003 * eps) @ powered_frame(T + eps * Td, Td, heading)[0]
    expected = Rotation.from_matrix(first.T @ last).magnitude() / (2 * eps)
    assert rate == pytest.approx(expected, rel=1e-6)
    boresight_rate = np.linalg.norm(np.cross(T, Td)) / (T @ T)
    assert rate > 10 * boresight_rate
    # A fixed inertial frame has zero rate, while its F representation rotates backwards.
    z = np.array([0.0, 0.0, 1.0])
    omega = 0.01
    frames = rotation_z(-omega * np.array([0.0, 1.0]))
    inertial = rotation_z(omega * np.array([0.0, 1.0])) @ frames
    assert Rotation.from_matrix(inertial[0].T @ inertial[1]).magnitude() < 1e-14
    with pytest.raises(OrientationError):
        powered_frame(heading, z, heading)


def test_coast_slerp_endpoints_constant_rate_and_pi():
    first = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    last = first @ Rotation.from_rotvec(np.array([0.1, 0.2, 0.3])).as_matrix()
    coast = CoastOrientation(first, last, 10.0)
    np.testing.assert_allclose(coast(np.array([0.0, 10.0])), [first, last], atol=1e-14)
    steps = coast(np.array([0.0, 1.0, 2.0]))
    assert Rotation.from_matrix(steps[0].T @ steps[1]).magnitude() == pytest.approx(coast.rate)
    assert Rotation.from_matrix(steps[1].T @ steps[2]).magnitude() == pytest.approx(coast.rate)
    assert CoastOrientation(first, first, 10.0).rate < 1e-14
    assert CoastOrientation(
        np.eye(3), Rotation.from_rotvec([np.pi, 0, 0]).as_matrix(), 5.0
    ).pi_ambiguity


def test_celestial_frame_sign_and_absolute_time(tracker_params):
    w, t = tracker_params.moon.omega, 1200.0
    e_I = np.array([1.0, 0.0, 0.0])
    e_F = rotation_z(-w * t) @ e_I
    assert e_F[1] < 0
    np.testing.assert_allclose(rotation_z(w * t) @ e_F, e_I, atol=1e-14)


def test_short_blocks_boundaries_alternation_and_noncontinuous_visibility():
    short = make_schedule([[]], 0.0, 0.9, 1.0)
    assert short["passed"] and short["initialized_estimate_only"]
    assert not make_schedule([[]], 0.0, 1.0, 1.0)["passed"]
    schedule = make_schedule([[[0.9, 1.0], [2.9, 3.0]], [[1.9, 2.0]]], 0.0, 3.9, 1.0)
    assert schedule["passed"]
    assert [v["tracker"] for v in schedule["events"]] == [1, 2, 1]
    assert schedule["max_update_gap_s"] == pytest.approx(1.0)
    assert schedule["first_update_latency_s"] == pytest.approx(1.0)
    assert schedule["final_update_age_s"] == pytest.approx(0.9)


def test_cadence_not_disjoint_bins_or_availability_percentage():
    assert not make_schedule([[[0.0, 0.05], [1.95, 4.0]]], 0.0, 4.0, 1.0)["passed"]
    assert not make_schedule([[[0.0, 49.0], [50.01, 100.0]]], 0.0, 100.0, 1.0)["passed"]
    # Simulated ascent/coast join at t=2 must not reset the clock.
    assert not make_schedule([[[0.0, 1.4], [2.5, 5.0]]], 0.0, 5.0, 1.0)["passed"]
    # Exact one-second outage is acceptable under instantaneous-measurement assumptions.
    assert make_schedule([[[0.0, 1.0], [2.0, 3.0]]], 0.0, 3.0, 1.0)["passed"]
    assert not make_schedule([[[1.001, 4.0]]], 0.0, 4.0, 1.0)["passed"]
    assert not make_schedule([[[0.0, 1.0]]], 0.0, 2.001, 1.0)["passed"]


def test_inclusive_rate_gate_applies_to_both_trackers(tracker_params):
    st = tracker_params.star_tracker
    sample = samples_for_axis(None)
    sample.body_rates[:] = st.tracking_body_rate_max
    assert np.all(signed_margins(sample, np.array([0.0, 0.0, 1.0]), st)[0, :, 3] >= 0)
    sample.body_rates[:] += np.radians(0.001)
    assert np.all(signed_margins(sample, np.array([0.0, 0.0, 1.0]), st)[0, :, 3] < 0)


def test_user_waiver_disables_only_rate_gate_and_keeps_raw_rate(tracker_params, make_params):
    assert make_params().star_tracker.enforce_body_rate is False
    settings = replace(tracker_params.star_tracker, enforce_body_rate=False)
    sample = samples_for_axis(None)
    sample.body_rates[:] = np.radians(12.0)
    axis = np.array([0.0, 0.0, 1.0])
    assert np.all(signed_margins(sample, axis, settings)[0, 0] >= 0)
    assert signed_margins(sample, axis, replace(settings, enforce_body_rate=True))[0, 0, 3] < 0
    blocked = replace(settings, sun_direction_I=(0.0, 0.0, 1.0))
    assert signed_margins(sample, axis, blocked)[0, 0, 0] < 0
    block = SimpleNamespace(
        settings=settings,
        start=0.0,
        end=2.0,
        name="test",
        phase_at=lambda t: "ascent",
        evaluate=lambda times: sample,
    )
    report = evaluate_schedule(block, axis, [[[0.0, 2.0]], []])
    assert report["passed"] and not report["body_rate_limit_enforced"]
    assert report["worst_accepted_body_rate_deg_s"] == pytest.approx(12.0)


def test_fast_window_cadence_matches_explicit_event_scheduler():
    rng = np.random.default_rng(42)
    examples = [[[], []], [[[0.0, 1.0], [2.0, 3.0]], []], [[[0.0, 1.0], [2.001, 3.0]], []]]
    for _ in range(100):
        raw = np.sort(rng.uniform(0, 3, (6, 2)), axis=1).tolist()
        examples.append([raw[:3], raw[3:]])
    for windows in examples:
        assert (
            cadence_feasible(windows, 0.0, 3.0, 1.0)[0]
            == make_schedule(windows, 0.0, 3.0, 1.0)["passed"]
        )


def test_midpoint_search_finds_outage_and_refines_individual_roots(tracker_params):
    st = replace(
        tracker_params.star_tracker,
        sun_direction_I=(0.0, 1.0, 0.0),
        earth_direction_I=(0.0, 1.0, 0.0),
    )

    class KnownWindow:
        start, end = 0.0, 0.05
        knots = np.array([0.0, 0.05])
        settings = st

        def evaluate(self, times):
            times = np.asarray(times)
            tilt = np.radians(81 - ((times - 0.025) / 0.01) ** 2)
            C = Rotation.from_rotvec(tilt[:, None] * np.array([0.0, 1.0, 0.0])).as_matrix()
            return Samples(
                times,
                C,
                np.tile([0.0, 0.0, 10.0], (len(times), 1)),
                np.zeros(len(times)),
                np.full(len(times), np.pi / 2),
            )

        def sample(self, step):
            base = np.linspace(
                self.start, self.end, int(np.ceil((self.end - self.start) / step)) + 1
            )
            return self.evaluate(np.unique(np.r_[base, (base[:-1] + base[1:]) / 2]))

    windows, _ = refine_intervals(KnownWindow(), np.array([0.0, 0.0, 1.0]), 0.05)
    assert len(windows[0]) == 2
    assert windows[0][0][1] == pytest.approx(0.015, abs=st.transition_tolerance)
    assert windows[0][1][0] == pytest.approx(0.035, abs=st.transition_tolerance)
    assert windows[1] == []


def test_blocks_use_same_design_coast_and_real_mission_clock(tracker_params):
    p = tracker_params
    sol = synthetic_solution(p)
    blocks = build_blocks(p, sol, "uprange")
    assert [(b.start, b.end) for b in blocks] == [(2.0, 6.0), (14.0, 18.0)]
    for block in blocks:
        samples = block.sample(0.05)
        np.testing.assert_allclose(np.linalg.det(samples.orientations), 1.0, atol=1e-13)
        expected = np.arcsin(p.moon.radius / (p.moon.radius + 1000))
        assert samples.moon_radii == pytest.approx(np.full(len(samples.times), expected))
        margins = geometry_margins(samples, mount_axis(0.5, 0.2), p.star_tracker)
        assert margins.shape == (len(samples.times), 2, 3)
    with pytest.raises(ValueError, match="full trajectory"):
        build_blocks(p, SimpleNamespace(success=True), "uprange")
    sol.phases["ascent"]["X"][:3] *= -1
    with pytest.raises(ValueError, match="subsurface"):
        build_blocks(p, sol, "uprange")[0].sample(0.05)


def test_complete_schedule_rechecked_on_independent_propagation(tracker_params):
    sol = synthetic_solution(tracker_params)
    blocks = build_blocks(tracker_params, sol, "uprange")
    independent = build_blocks(tracker_params, sol, "uprange", independent=True)
    report = validate_mount(blocks, independent, dict(tilt=0.0, azimuth=0.0))
    for row in report.values():
        assert row["passed"] and row["numerically_resolved"]
        assert row["event_count"] > 0
        assert all(e["valid"] for e in row["events"])
        assert row["max_update_gap_s"] <= 1.0 + 1e-10
        assert row["worst_accepted_body_rate_deg_s"] <= 6.0
        assert row["limiting_event"]["margin_deg"] >= 0


def test_explicit_non_antipodal_pair_has_independently_checked_events(tracker_params):
    from hopper.star_tracker import top_mount_axes

    top_axes = lambda angles: top_mount_axes(angles, [0.0, np.pi])
    p = replace(
        tracker_params, star_tracker=replace(tracker_params.star_tracker, enforce_body_rate=False)
    )
    sol = synthetic_solution(p)
    blocks, independent = build_blocks(p, sol, "uprange"), build_blocks(
        p, sol, "uprange", independent=True
    )
    axes = top_axes(np.radians([60, 50]))
    reports = validate_mount(blocks, independent, None, body_axes=axes)
    assert all(row["passed"] and row["numerically_resolved"] for row in reports.values())
    for row in reports.values():
        assert all(e["valid"] for e in row["events"])


def test_mount_search_positive_robustness_path_uses_both_hops(tracker_params):
    st = replace(
        tracker_params.star_tracker,
        tilt_bounds=(0.0, 0.1),
        azimuth_bounds=(0.0, 0.2),
        coarse_step=0.1,
    )
    p = replace(tracker_params, star_tracker=st)
    sol = synthetic_solution(p)
    blocks = build_blocks(p, sol, "uprange")
    ranked, method = search_mount(blocks)
    winner = ranked[0]
    assert winner["sampled_passed"] and winner["sampled_robust_margin"] > 0
    assert method["tested_count"] > method["coarse_count"]
    reports = validate_mount(
        blocks,
        build_blocks(p, sol, "uprange", independent=True),
        winner,
        extra_margin=winner["sampled_robust_margin"] / 2,
    )
    assert set(reports) == {"hop1", "hop2"}
    assert all(row["passed"] and row["numerically_resolved"] for row in reports.values())


def test_failed_schedule_retains_violation_and_reports_later_events():
    schedule = make_schedule([[[2.5, 10.0]]], 0.0, 10.0, 1.0)
    assert not schedule["passed"]
    assert schedule["first_update_latency_s"] == pytest.approx(2.5)
    assert schedule["max_update_gap_s"] == pytest.approx(2.5)
    assert schedule["event_count"] > 1
    assert schedule["final_update_age_s"] <= 1.0


def test_rise_join_jump_cannot_claim_coverage(tracker_params):
    p = replace(
        tracker_params,
        star_tracker=replace(
            tracker_params.star_tracker, required_phase_kinds=("rise", "ascent", "coast")
        ),
    )
    sol = synthetic_solution(p)
    sol.phases["rise"]["U"][:3] = (
        Rotation.from_euler("x", 2, degrees=True).as_matrix() @ sol.phases["rise"]["U"][:3]
    )
    with pytest.raises(OrientationError, match="jump"):
        build_blocks(p, sol, "uprange")


def test_numeric_only_cache_stale_checks_and_summary_rejection(tracker_params, tmp_path):
    from conftest import ROOT

    p = tracker_params
    identity = baseline_identity(p, "uprange", ROOT)
    assert baseline_identity(replace(p, star_tracker=None), "uprange", ROOT) == identity
    classes = dict(p.engine.classes)
    classes["small"] = replace(classes["small"], isp=318.0)
    changed = replace(p, engine=replace(p.engine, classes=classes))
    assert baseline_identity(changed, "uprange", ROOT) != identity
    path = tmp_path / "flight.npz"
    sol = synthetic_solution(p)
    save_trajectory(path, p, sol, "uprange", identity, {"passed": True})
    loaded, _ = load_trajectory(path, p, "uprange", identity)
    np.testing.assert_array_equal(loaded.phases["coast"]["X"], sol.phases["coast"]["X"])
    assert loaded.phases["terminal2_uprange"]["kind"] == "terminal"
    with pytest.raises(ValueError, match="Stale"):
        load_trajectory(path, p, "uprange", "wrong")
    summary = tmp_path / "summary.json"
    summary.write_text('{"summary": {}}')
    with pytest.raises(ValueError):
        load_trajectory(summary, p, "uprange", identity)

"""Symbolic/numeric agreement and dimensional timing checks for the coupled model."""
from dataclasses import replace
from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hopper.problem import Problem
from hopper.star_tracker import CoastOrientation, build_blocks, candidate_axes, geometry_margins, search_mount, validate_mount
from hopper.star_tracker_coupled import (TrackerProblem, prepare_schedule, symbolic_powered_frame,
                                        symbolic_rotation_z, symbolic_slerp, validate_assigned_events)
from test_star_tracker import synthetic_solution


@pytest.fixture
def make_params(single_stage_params_factory):
    return single_stage_params_factory


def test_symbolic_full_body_rate_matches_numeric():
    from hopper.star_tracker import powered_frame
    from hopper.star_tracker_coupled import symbolic_body_rate_squared
    heading = np.array([1., 0., 0.])
    for thrust, slope in (([.2, .3, 1.], [.1, -.2, .05]), ([0., 0., 2.], [.3, .4, 1.])):
        _, rate = powered_frame(np.array(thrust), np.array(slope), heading, 2.7e-6)
        symbolic = float(symbolic_body_rate_squared(ca.DM(thrust), ca.DM(slope), ca.DM(heading), 2.7e-6))
        assert np.sqrt(symbolic) == pytest.approx(float(rate), rel=1e-12)


def test_coupled_staged_coverage_enforces_rate():
    from hopper.params import load_params
    from conftest import CONFIG
    p = load_params(CONFIG, {"star_tracker": {"enabled": True, "enforce_body_rate": True}})
    sol = synthetic_solution(p)
    seed = dict(elevations=list(p.star_tracker.initial_elevations), azimuths=list(p.star_tracker.initial_azimuths))
    events, _ = prepare_schedule(p, sol, "uprange", seed, extra_margin_deg=0)
    pr = TrackerProblem(p, "uprange", events, seed)
    pr.set_guess(Problem.guess_from_solution(sol))
    assert np.isfinite(pr.initial_violation())
    checked = validate_assigned_events(p, sol, "uprange", seed, [dict(time_s=float(t), tracker=e.tracker,
        phase=e.phase, block=e.block) for e, t in zip(events, pr.opti.debug.value(ca.vertcat(*pr.event_times), pr.opti.initial()))])
    assert all(v["body_rate_limit_enforced"] and v["passed"] for v in checked.values())


@pytest.mark.parametrize("degrees", [0., 1e-8, 30., 110., 179.9999, 180.])
@pytest.mark.parametrize("axis", [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [.2, .8, .3]])
def test_symbolic_slerp_matches_numeric_shortest_path(degrees, axis):
    first = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    axis = np.array(axis) / np.linalg.norm(axis)
    last = first @ Rotation.from_rotvec(np.radians(degrees) * axis).as_matrix()
    numeric = CoastOrientation(first, last, 1.)
    for fraction in (0., .2, .7, 1.):
        symbolic = np.array(symbolic_slerp(ca.DM(first), ca.DM(last), fraction))
        np.testing.assert_allclose(symbolic, numeric(np.array([fraction]))[0], atol=1e-9)


def test_zero_rotation_slerp_has_finite_correct_derivative():
    a = ca.MX.sym("angle")
    frame = symbolic_slerp(ca.DM.eye(3), symbolic_rotation_z(a), .4)
    derivative = ca.Function("derivative", [a], [ca.jacobian(ca.vec(frame), a)])
    actual = np.array(derivative(0.)).reshape(3, 3, order="F")
    np.testing.assert_allclose(actual, [[0, -.4, 0], [.4, 0, 0], [0, 0, 0]], atol=1e-12)


def test_symbolic_frame_right_handed():
    frame = np.array(symbolic_powered_frame(ca.DM([.2, .3, 1]), ca.DM([1, 0, 0])))
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-14)
    assert np.linalg.det(frame) == pytest.approx(1.)


def test_independent_configuration_and_four_angle_search(make_params):
    p = make_params({"star_tracker": {"enabled": True}})
    assert p.star_tracker.mount_model == "independent_top"
    assert not p.star_tracker.enforce_body_rate
    sol = synthetic_solution(p)
    ranked, metadata = search_mount(build_blocks(p, sol, "uprange"))
    assert ranked[0]["sampled_passed"]
    assert len(ranked[0]["elevations"]) == len(ranked[0]["azimuths"]) == 2
    assert np.all(candidate_axes(ranked[0])[:, 2] > 0)
    assert metadata["mount_model"] == "independent_top"
    for override in ({"antipodal": True}, {"mount_model": "unknown"},
                     {"mount_search": {"elevation_bounds_deg": [[0, 90], [10, 90]]}},
                     {"mount_search": {"initial_elevations_deg": [-5, 30]}}):
        with pytest.raises(ValueError):
            make_params({"star_tracker": override})


def test_downward_installed_head_fails_even_if_other_head_covers_every_event(make_params):
    p = make_params({"star_tracker": {"enabled": True}})
    sol = synthetic_solution(p)
    pair = np.array([[0., 0., 1.], [0., 0., -1.]])
    report = validate_mount(build_blocks(p, sol, "uprange"),
                            build_blocks(p, sol, "uprange", independent=True), None, body_axes=pair)
    assert all(not row["passed"] and not row["top_plane_clearance_passed"] for row in report.values())


def test_coupled_adds_four_global_angles_and_matches_event_geometry(make_params):
    p = make_params({"star_tracker": {"enabled": True}})
    sol = synthetic_solution(p)
    seed = dict(elevations=list(p.star_tracker.initial_elevations), azimuths=list(p.star_tracker.initial_azimuths))
    events, _ = prepare_schedule(p, sol, "uprange", seed, extra_margin_deg=0)
    pr = TrackerProblem(p, "uprange", events, seed)
    baseline = Problem(p, "uprange")
    assert pr.opti.nx == baseline.opti.nx + 4
    pr.set_guess(Problem.guess_from_solution(sol), m_wet0=sol.m_wet)
    initial = pr.opti.initial()
    times = np.array([float(pr.opti.debug.value(t, initial)) for t in pr.event_times])
    blocks = {b.name: b for b in build_blocks(p, sol, "uprange")}
    settings = p.star_tracker
    for event, time, symbolic in zip(events, times, pr.event_residuals):
        sample = blocks[event.block].evaluate([time])
        b = sample.orientations[0] @ candidate_axes(seed)[event.tracker - 1]
        r = np.linalg.norm(sample.positions_I[0])
        nadir = -sample.positions_I[0] / r
        expected = [b @ settings.sun_direction_I - np.cos(settings.sun_exclusion),
                    b @ settings.earth_direction_I - np.cos(settings.earth_exclusion),
                    b @ nadir - np.cos(sample.moon_radii[0] + settings.full_cone_fov / 2 + settings.moon_limb_margin),
                    (p.moon.radius - r) / p.scales.L]
        np.testing.assert_allclose(np.array(pr.opti.debug.value(symbolic, initial)).reshape(4), expected, atol=1e-10)
    assert pr.unique_gap_constraint_count < len(events) + 2
    # All timing inequalities use dimensional seconds and must pass this 0.9 s seed.
    assert max(np.diff(times[:4])) <= .9 + 1e-12
    # Duration changes must move events; times are not frozen absolute seconds.
    duration_jacobian = ca.Function("time_gradient", [pr.opti.x],
                                    [ca.jacobian(pr.event_times[0], pr.opti.x)])
    state = pr.opti.debug.value(pr.opti.x, initial)
    assert np.count_nonzero(np.array(duration_jacobian(state))) > 0
    actual_events = [dict(time_s=float(t), tracker=e.tracker, phase=e.phase, block=e.block,
                          cell=e.cell, fraction=e.fraction) for t, e in zip(times, events)]
    checked = validate_assigned_events(p, sol, "uprange", seed, actual_events)
    assert all(row["passed"] for row in checked.values())
    actual_events[0]["time_s"] += .2  # geometry remains clear, but initialized-estimate latency exceeds 1 s
    assert not validate_assigned_events(p, sol, "uprange", seed, actual_events)["hop1"]["passed"]



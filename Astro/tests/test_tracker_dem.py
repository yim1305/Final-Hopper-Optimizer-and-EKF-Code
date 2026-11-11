"""DEM tracker regression checks: below-sphere geometry, real-time events and labels."""

from dataclasses import replace
from types import SimpleNamespace
import casadi as ca
import numpy as np
import pytest

from hopper.problem import Problem
from hopper.star_tracker_coupled import AssignedEvent
from mission.config import CONFIG, parameters
from mission.terrain import DemTerrain, GroundProblem
from mission.trackers import SkyTerrain
from mission import tracker_solver as coupled
from test_star_tracker import synthetic_solution


def test_minimum_cone_projection_includes_interior_extreme():
    half = np.radians(10)
    r = np.array([0., 0., 100.])
    assert coupled.minimum_cone_projection(r, np.array([0., 0., -1.]), half) == -100
    for axis in ([0., 0., 1.], [.8, 0., .6], [.8, 0., -.6]):
        sampled = coupled.cone_axes(np.asarray(axis), half, 4, 720) @ r
        expected = coupled.minimum_cone_projection(r, np.asarray(axis), half)
        assert sampled.min() == pytest.approx(expected, abs=1e-9)


def test_body_fixed_cone_samples_are_unit_and_contain_boresight():
    axis = np.array([.3, .4, np.sqrt(.75)])
    rays = coupled.cone_axes(axis, .2, 2, 16)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1., atol=1e-14)
    np.testing.assert_allclose(rays[0], axis)
    assert np.min(rays @ axis) == pytest.approx(np.cos(.2))


@pytest.fixture(scope='module')
def dem_inputs():
    terrain = DemTerrain()
    p = parameters('a2200_08', terrain, {'solver': {'nodes': dict(rise=3, ascent=6, coast=3, braking=6, observation=3, terminal=3)}})
    return terrain, p


@pytest.mark.parametrize('event_fraction', [0., .5])
def test_joint_nlp_keeps_dem_and_moves_events_with_duration(dem_inputs, event_fraction):
    terrain, p = dem_inputs
    seed = synthetic_solution(p)
    events = []
    for hop in (1, 2):
        name = next(name for name, ph in seed.phases.items() if ph['hop'] == hop and ph['kind'] == 'ascent')
        events.append(AssignedEvent(name, 0, event_fraction, 1, f'hop{hop}'))
    env = [dict(length_m=0.) for _ in events]
    sky = SimpleNamespace(outer=p.moon.radius+10000)
    problem = coupled.DemTrackerProblem(p, 'uprange', events, env, sky, terrain=terrain)
    ground = GroundProblem(p, 'uprange', terrain=terrain)
    # Fixed hardware adds constraints, not mount-angle decision variables.
    assert problem.opti.nx == ground.opti.nx
    assert problem.base_constraint_count == ground.opti.ng
    assert problem.opti.ng > ground.opti.ng
    problem.set_guess(Problem.guess_from_solution(seed), m_wet0=seed.m_wet)
    initial = problem.opti.initial()
    value = np.asarray(problem.opti.debug.value(problem.opti.x, initial)).ravel()
    jacobian = ca.Function('event_time_jacobian', [problem.opti.x], [ca.jacobian(problem.event_times[0], problem.opti.x)])
    assert np.count_nonzero(np.asarray(jacobian(value))) >= (1 if event_fraction == 0 else 2)
    assert np.isfinite(np.asarray(problem.opti.debug.value(problem.opti.g, initial))).all()


def test_symbolic_sky_accepts_native_terrain_below_reference_sphere(dem_inputs):
    terrain, p = dem_inputs
    sky = SkyTerrain()
    r = p.mission.r_start + p.mission.r_start/np.linalg.norm(p.mission.r_start)*50
    assert np.linalg.norm(r) < p.moon.radius
    assert sky.cone(r, r/np.linalg.norm(r), np.radians(10), 2, 16)['clear']
    symbolic = coupled.SymbolicSky(sky, [r], 1.)
    opti = ca.Opti()
    point = opti.variable(3)
    symbolic.constrain(opti, point, p.scales.L)
    opti.set_initial(point, r)
    clearance = float(opti.debug.value(opti.g[-1], opti.initial()))*p.scales.L
    assert clearance == pytest.approx(50., abs=1e-6)


@pytest.mark.parametrize('wrong_head,too_late,passed', [(False, False, True), (True, False, False), (False, True, False)])
def test_exact_event_validation_cannot_substitute_head_or_schedule(monkeypatch, wrong_head, too_late, passed):
    class Phase:
        name = 'ascent'
        start, end = 0., 3.
        ph = {'t': 3.}

        def orientation(self, times):
            return np.repeat(np.eye(3)[None], len(times), axis=0), np.zeros(len(times))

        def state(self, times):
            return np.zeros((len(times), 7))

    # Head 1 points up; head 2 points toward the forbidden Sun direction.
    st = SimpleNamespace(initial_elevations=(np.pi/2, 0.), initial_azimuths=(0., 0.),
                         full_cone_fov=.2, moon_limb_margin=0., sun_direction_I=(1., 0., 0.),
                         earth_direction_I=(0., 1., 0.), sun_exclusion=.4, earth_exclusion=.4,
                         angular_tolerance=1e-9, enforce_body_rate=True, tracking_body_rate_max=.3,
                         rate_tolerance=1e-9, update_gap_max=2.)
    p = SimpleNamespace(star_tracker=st, mission=SimpleNamespace(r_start=np.array([0., 0., 100.])), moon=SimpleNamespace(omega=0.))
    block = SimpleNamespace(name='hop1', start=0., end=3., phases=[Phase()])
    monkeypatch.setattr(coupled, 'blocks', lambda *args: [block])
    sky = SimpleNamespace(cone=lambda *args: dict(clear=True))
    event = dict(block='hop1', phase='ascent', tracker=2 if wrong_head else 1, time_s=2.1 if too_late else 1.5)
    assert coupled.validate_events(p, None, [event], sky)['passed'] is passed


def test_missing_updates_fail_even_for_a_short_block(monkeypatch):
    st = SimpleNamespace(initial_elevations=(np.pi/2, np.pi/2), initial_azimuths=(0., 0.), full_cone_fov=.2,
                         moon_limb_margin=0., update_gap_max=2.)
    p = SimpleNamespace(star_tracker=st)
    monkeypatch.setattr(coupled, 'blocks', lambda *args: [SimpleNamespace(name='hop1', start=0., end=.5, phases=[])])
    assert not coupled.validate_events(p, None, [], None)['passed']

"""Acceptance checks for the G&N simulation.

Run from the repository root:  .\\.venv\\Scripts\\python.exe -B -m pytest -q -p no:cacheprovider GN/tests
Needs the generated Astro flight (Astro/scripts/run_case.py a2200_08).
"""
from types import SimpleNamespace

import numpy as np
import pytest

from gn_sim.config import load_config
from gn_sim.montecarlo import run_case
from gn_sim.rotations import exp_so3, log_so3, skew
from gn_sim.sensors import imu_range_check, tracker_availability,trn_availability
from gn_sim.truth import build_hop, dead_reckon, load_case

from gn_sim.scripts.run_nav import window_gaps


@pytest.fixture(scope="module")
def setup():
    cfg = load_config()
    case = load_case(cfg)
    return cfg, case, {hop: build_hop(case, cfg, hop) for hop in (1, 2)}


def test_current_input_is_the_validated_2000kg_dem_design(setup):
    cfg,case,truths=setup
    assert cfg.case_dir.name=='a2200_08'
    assert case.p.vehicle.m_dry==2000.
    assert case.p.limits.td_horiz_speed_max==.5
    assert case.report['status']=='numerically_validated_dem_flight'
    assert abs(np.linalg.norm(case.p.mission.r_start)-case.p.moon.radius+2734.004134)<1e-5


@pytest.mark.parametrize('hop',[1,2])
def test_trn_uses_dem_height_below_the_reference_sphere(setup,hop):
    cfg,case,truths=setup
    truth=truths[hop]
    below=np.linalg.norm(truth.r_I,axis=1)<case.p.moon.radius
    assert below.any()
    assert np.any(trn_availability(truth,cfg)&below)
    assert truth.altitude.min()>-.02


# ---------------------------------------------------------------- rotations
def test_exp_log_roundtrip():
    rng = np.random.default_rng(1)
    phi = rng.normal(size=(50, 3)) * 0.8
    assert np.allclose(log_so3(exp_so3(phi)), phi, atol=1e-10)
    v, w = rng.normal(size=3), rng.normal(size=3)
    assert np.allclose(skew(v) @ w, np.cross(v, w))


# ---------------------------------------------------------------- truth (checks 1 and 2)
@pytest.mark.parametrize("hop", [1, 2])
def test_perfect_imu_reproduces_truth(setup, hop):
    cfg, case, truths = setup
    truth = truths[hop]
    r, v, C = dead_reckon(truth, case.p.moon.mu)
    assert np.linalg.norm(r - truth.r_I[-1]) < 1.0               # m at touchdown
    assert np.linalg.norm(v - truth.v_I[-1]) < 1e-3              # m/s
    assert np.linalg.norm(log_so3(C.T @ truth.C_IB[-1])) < 1e-6  # rad


@pytest.mark.parametrize("hop", [1, 2])
def test_truth_runs_launch_to_touchdown(setup, hop):
    cfg, case, truths = setup
    truth = truths[hop]
    assert abs(truth.altitude[0]) < 1e-3 and abs(truth.altitude[-1]) < 1e-2
    assert truth.checkpoints["touchdown"] == len(truth.t) - 1
    assert "descent_1000m" in truth.checkpoints
    assert imu_range_check(truth, cfg)["within_range"]


# ---------------------------------------------------------------- check 3: hand calculation
def test_accel_bias_dead_reckoning_matches_half_b_t_squared():
    """Coast with fixed attitude: an accelerometer bias b gives position error 1/2 b t^2."""
    mu, R = 4.9028e12, 1737.4e3
    h, n = 1 / 16, 16 * 60
    r0 = np.array([R + 20e3, 0, 0])
    v0 = np.array([0, np.sqrt(mu / np.linalg.norm(r0)), 0])
    zeros = np.zeros((n, 3))
    truth = SimpleNamespace(r_I=r0[None], v_I=v0[None], C_IB=np.eye(3)[None], dt=np.full(n, h),
                            dtheta_b=zeros, dv_b=zeros)
    r_true, _, _ = dead_reckon(truth, mu)
    b = np.array([0, 0, 3e-3])                      # 0.3 mg along body z (out of the orbit plane)
    r_bias, _, _ = dead_reckon(truth, mu, dv=zeros + b * h)
    t = n * h
    expected = 0.5 * 3e-3 * t**2
    assert abs(np.linalg.norm(r_bias - r_true) / expected - 1) < 0.02


# ---------------------------------------------------------------- check 5: star tracker availability
def test_higher_tracker_rate_gate_cannot_reduce_availability(setup):
    """GN's smoothed attitude is distinct from Astro's prescribed event frame.

    Raising the rate gate must preserve all updates allowed by the hardware gate;
    it does not certify the optimizer's cadence on a different attitude history.
    """
    cfg, case, truths = setup
    cfg15 = load_config(overrides=dict(star_tracker=dict(max_body_rate_deg_s=15.5)))
    for truth in truths.values():
        hardware,_=tracker_availability(truth,case,cfg)
        avail, _ = tracker_availability(truth, case, cfg15)
        assert not np.any(hardware&~avail)
        assert avail.any()
        assert np.all(truth.body_rate[avail.any(axis=1)]<=cfg15.tracker.max_body_rate)


def test_batched_terrain_cone_matches_astro_fine_ray_test(setup):
    from gn_sim.sensors import terrain_cone_clear
    _,case,_=setup
    half=case.p.star_tracker.full_cone_fov/2+case.p.star_tracker.moon_limb_margin
    rng=np.random.default_rng(4700)
    for center in (case.p.mission.r_start,case.p.mission.r_target):
        up=center/np.linalg.norm(center)
        for altitude in (100.,10000.,60000.):
            r=center+altitude*up
            for direction in (up,-up,*rng.normal(size=(3,3))):
                direction=direction/np.linalg.norm(direction)
                expected=case.sky.cone(r,direction,half,4,32)['clear']
                assert terrain_cone_clear(case.sky,r,direction,half)==expected


# ---------------------------------------------------------------- check 4: filter consistency
def test_filter_consistency_case_c(setup):
    cfg, case, truths = setup
    truth = truths[2]
    avail, _ = tracker_availability(truth, case, cfg)
    res = run_case(truth, case, cfg, cfg.cases["C"], avail, runs=30, case_id="C")
    for block in ("position", "velocity", "attitude"):
        assert res["consistency"][block]["fraction_within_3sigma"] > 0.98
        assert 2.0 < res["consistency"][block]["mean_nees"] < 4.0

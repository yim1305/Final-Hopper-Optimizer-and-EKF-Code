"""Descent settings of the final run: 2224 N thrust floor, touchdown thrust-to-weight limit, solve-side margin."""

import pytest
from mission.config import CONFIG, _lower_floor, descent_variant, parameters
from mission.terrain import DemTerrain, GroundProblem

OFF = {"thrust_floor_n": None}


@pytest.fixture(scope="module")
def terrain():
    return DemTerrain()


def test_mission_yaml_carries_the_final_descent_settings():
    settings = descent_variant()
    assert settings["thrust_floor_n"] == 2224.0
    assert settings["floor_phases"] == ["observation", "terminal"]
    assert settings["touchdown_thrust_to_weight_max"] == 1.0
    assert settings["solve_pointing_rate_deg_s"] == 5.0
    with pytest.raises(ValueError):
        descent_variant({"thrust_floor_n": -1.0})
    with pytest.raises(ValueError):
        descent_variant({"solve_pointing_rate_deg_s": 0.0})


def test_floor_lowers_only_the_listed_phases_on_both_hops(terrain):
    base = parameters("a2200_08", terrain, None, OFF)
    low = parameters("a2200_08", terrain)
    for table in ("modes", "return_modes"):
        for kind, stages in getattr(base.engine, table).items():
            for before, after in zip(stages, getattr(low.engine, table)[kind]):
                assert after.gamma_hi == before.gamma_hi
                assert after.gamma_lo == (2224.0 if kind in ("observation", "terminal") else before.gamma_lo)
    assert base.engine.modes["terminal"][0].gamma_lo == 4448.0


def test_floor_never_raises_and_stays_below_the_stage_maximum(terrain):
    modes = parameters("a2200_08", terrain, None, OFF).engine.modes
    assert _lower_floor(modes, ["terminal"], 5000.0)["terminal"][0].gamma_lo == 4448.0
    with pytest.raises(ValueError):
        _lower_floor(modes, ["terminal"], 8896.0)


def test_every_case_still_builds_with_the_settings(terrain):
    from mission.config import CASES

    for case in CASES:
        parameters(case, terrain)


def test_touchdown_thrust_to_weight_adds_one_constraint_per_hop(terrain, monkeypatch):
    p = parameters("a2200_08", terrain)
    nodes = dict(rise=4, ascent=8, coast=6, braking=8, observation=4, terminal=8)
    limited = GroundProblem(p, "uprange", terrain=terrain, nodes=nodes)
    assert limited.tw_max == 1.0
    monkeypatch.setitem(CONFIG["descent_variant"], "touchdown_thrust_to_weight_max", None)
    plain = GroundProblem(p, "uprange", terrain=terrain, nodes=nodes)
    assert plain.tw_max is None
    assert int(limited.opti.ng) - int(plain.opti.ng) == 2


def test_solve_margin_does_not_change_the_checked_sensor_limit(terrain):
    import numpy as np

    p = parameters("a2200_08", terrain)
    nodes = dict(rise=4, ascent=8, coast=6, braking=8, observation=4, terminal=8)
    problem = GroundProblem(p, "uprange", terrain=terrain, nodes=nodes)
    assert problem.p.sensor.pointing_rate_max == pytest.approx(np.radians(5.0))
    assert p.sensor.pointing_rate_max == pytest.approx(np.radians(30.0))

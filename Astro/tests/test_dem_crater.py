"""Current DEM geometry and the selected physical assumptions of the a2200_08 case."""

import numpy as np
import pytest
from mission.terrain import DemTerrain
from mission.trackers import SkyTerrain
from mission.config import parameters as load_case_parameters, physical_identity


def parameters(terrain):
    return load_case_parameters("a2200_08", terrain)


@pytest.fixture(scope="module")
def terrain():
    return DemTerrain()


def test_launch_location_is_preserved_but_uses_actual_dem_height(terrain):
    p = parameters(terrain)
    r = p.mission.r_start
    unit = r / np.linalg.norm(r)
    assert np.degrees(np.arcsin(unit[2])) == pytest.approx(-85.5, abs=1e-10)
    assert np.degrees(np.arctan2(unit[1], unit[0])) == pytest.approx(-87.1, abs=1e-10)
    assert np.linalg.norm(r) - p.moon.radius == pytest.approx(-2734.004134, abs=1e-5)
    assert abs(terrain.altitude(r)) < 1e-7


def test_all_landing_targets_are_registered_to_dem(terrain):
    p = parameters(terrain)
    points = np.array([p.mission.r_start, p.mission.r_target, *p.mission.sites.values()])
    assert np.max(np.abs(terrain.altitude(points))) < 1e-7
    assert np.linalg.norm(p.mission.r_target) - p.moon.radius == pytest.approx(
        -4818.455566, abs=1e-5
    )


def test_below_mean_radius_is_not_an_optical_subsurface_error():
    sky = SkyTerrain()
    p = parameters(sky.native)
    up = p.mission.r_start / np.linalg.norm(p.mission.r_start)
    observer = p.mission.r_start + 100 * up
    assert np.linalg.norm(observer) < p.moon.radius
    assert sky.cone(observer, up, np.radians(10), 2, 16)["clear"]
    assert not sky.cone(observer, -up, np.radians(10), 2, 16)["clear"]


def test_main_physical_inputs_match_current_a2200_08_descent_variant_case(terrain):
    """Pin the physical-input identity of the current a2200_08 case (the 5,363.13 kg flight).

    The hash covers every physical input of the case with the descent-variant settings of
    mission.yaml (2224 N floor, touchdown thrust/weight limit, 5 s observation, 5 m/s descent cap)
    but not the numerical solver settings. Update it deliberately when a physical input changes.
    """
    assert (
        physical_identity(parameters(terrain))
        == "6b9fc43a59bf878ab3d34badb58d301869cc130eeac9ba4483eb1bdf4914873f"
    )

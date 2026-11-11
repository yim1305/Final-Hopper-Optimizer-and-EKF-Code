"""Focused correctness checks for the isolated terrain-registration experiment."""

import sys
from pathlib import Path
import numpy as np
import casadi as ca
import pytest
from mission.terrain import DemTerrain
from mission.config import parameters as load_case_parameters


def parameters(terrain):
    return load_case_parameters("a2200_08", terrain)


@pytest.fixture(scope="module")
def terrain():
    return DemTerrain()


def test_native_numeric_and_symbolic_terrain_agree(terrain):
    p = parameters(terrain)
    r = np.column_stack([p.mission.r_start, p.mission.r_target, p.mission.sites["uprange"]])
    expected = terrain.height(tuple(r))
    symbol = ca.SX.sym("position", 3, 3)
    f = ca.Function(
        "native_height_test", [symbol], [terrain.height(tuple(symbol[i, :] for i in range(3)))]
    )
    assert np.allclose(np.asarray(f(r)).ravel(), expected, atol=1e-7)


def test_registered_endpoints_are_on_dem(terrain):
    p = parameters(terrain)
    points = np.array([p.mission.r_start, p.mission.r_target, *p.mission.sites.values()])
    assert np.max(np.abs(terrain.altitude(points))) < 1e-7
    assert np.linalg.norm(p.mission.r_start) < p.moon.radius
    assert np.linalg.norm(p.mission.r_target) < p.moon.radius


def test_altitude_is_radial_height_above_dem(terrain):
    p = parameters(terrain)
    r = p.mission.r_start
    lifted = r / np.linalg.norm(r) * (np.linalg.norm(r) + 50.0)
    assert terrain.altitude(lifted) == pytest.approx(50.0, abs=1e-7)


def test_missing_terrain_is_rejected(terrain):
    with pytest.raises(ValueError, match="leaves retained DEM"):
        terrain.height((terrain.radius, 0.0, 0.0))


def test_new_ray_trace_matches_spherical_horizon():
    from mission.validation import ray_clearance

    radius = 1737400.0

    class Sphere:
        def altitude(self, r):
            return np.linalg.norm(r, axis=-1) - radius

    base = np.array([radius + 2.0, 0.0, 0.0])
    rh = radius + 10000.0
    horizon = np.arccos(radius / np.linalg.norm(base)) + np.arccos(radius / rh)
    near, far = horizon - np.radians(0.05), horizon + np.radians(0.05)
    assert (
        ray_clearance(Sphere(), base, np.array([rh * np.cos(near), rh * np.sin(near), 0.0]), 7.5)
        > 0
    )
    assert (
        ray_clearance(Sphere(), base, np.array([rh * np.cos(far), rh * np.sin(far), 0.0]), 7.5) < 0
    )


def test_new_ray_trace_detects_intervening_ridge():
    from mission.validation import ray_clearance

    radius = 1737400.0

    class Ridge:
        def altitude(self, r):
            angle = np.arctan2(r[..., 1], r[..., 0])
            height = np.where(abs(angle - np.radians(1.0)) < np.radians(0.15), 5000.0, 0.0)
            return np.linalg.norm(r, axis=-1) - radius - height

    base = np.array([radius + 2.0, 0.0, 0.0])
    rh = radius + 10000.0
    a = np.radians(2.0)
    assert ray_clearance(Ridge(), base, np.array([rh * np.cos(a), rh * np.sin(a), 0.0]), 7.5) < 0

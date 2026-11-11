import numpy as np
import pytest
import casadi as ca

from hopper.sensor import camera_axes, image_angles
from hopper.environment.terrain import altitude, FlatTerrain


def test_square_camera_includes_diagonal_corner():
    thrust, heading = np.array([0., 0., 1.]), np.array([1., 0., 0.])
    angles = image_angles(thrust, heading, np.array([.9, .9, -1.]))
    assert max(abs(a) for a in angles[:2]) < 45
    assert np.degrees(np.arctan(np.sqrt(2) * .9)) > 45
    assert abs(image_angles(thrust, heading, np.array([1.1, 0., -1.]))[0]) > 45
    x, y, b = camera_axes(ca.DM(thrust), ca.DM(heading))
    np.testing.assert_allclose(np.array(ca.horzcat(x, y, b)).T @ np.array(ca.horzcat(x, y, b)), np.eye(3))


def test_altitude_checks_all_columns():
    r = ca.DM([[0, 0], [0, 0], [1, 2]])
    h = altitude(np.array([0, 0, 100]), 100, r, FlatTerrain())
    np.testing.assert_allclose(np.array(h).ravel(), [1, 2])


@pytest.mark.parametrize("overrides", [
    {"sensor": {"fov_deg": [180, 90]}},
    {"sensor": {"descent_speed_bounds": [0, 100]}},
    {"sensor": {"processing_delay": -1}},
    {"sensor": {"observation_time_min": 500}},
    {"mission": {"landing_tolerance": -1}},
])
def test_reject_invalid_sensor_assumptions(make_params, overrides):
    with pytest.raises(ValueError):
        make_params(overrides)


def test_fov_margin_preserves_hardware_field_of_view(make_params):
    p = make_params({"sensor": {"fov_deg": [90, 90], "edge_margin_deg": 5}})
    np.testing.assert_allclose(np.degrees(p.sensor.fov), [80, 80])
    assert np.degrees(p.sensor.edge_margin) == pytest.approx(5)
    np.testing.assert_allclose(np.degrees(np.array(p.sensor.fov) + 2 * p.sensor.edge_margin), [90, 90])

import numpy as np
import pytest

from hopper.params import report_prechecks


def test_sites_on_sphere(params):
    R = params.moon.radius
    assert np.linalg.norm(params.mission.r_start) == pytest.approx(R, abs=1e-6)
    assert np.linalg.norm(params.mission.r_target) == pytest.approx(R, abs=1e-6)


def test_start_is_in_southern_hemisphere(params):
    assert params.mission.r_start[2] < 0


def test_target_frame_orthonormal(params):
    m = params.mission
    for a, b in ((m.up, m.downrange), (m.up, m.left), (m.downrange, m.left)):
        assert abs(a @ b) < 1e-12
    for v in (m.up, m.downrange, m.left):
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-12)
    assert np.cross(m.up, m.downrange) @ m.left == pytest.approx(1.0, abs=1e-12)


def test_downrange_points_toward_start_to_target(params):
    m = params.mission
    assert (m.r_target - m.r_start) @ m.downrange > 0


def test_divert_sites_150m_from_target_on_surface(params):
    m = params.mission
    Rt = np.linalg.norm(m.r_target)
    for name in ("uprange", "downrange", "left", "right"):
        site = m.sites[name]
        assert np.linalg.norm(site) == pytest.approx(Rt, abs=1e-6)
        assert np.linalg.norm(site - m.r_target) == pytest.approx(150.0, abs=1e-3)


def test_divert_directions(params):
    m = params.mission
    d = {k: m.sites[k] - m.r_target for k in ("uprange", "downrange", "left", "right")}
    assert d["downrange"] @ m.downrange > 149.9
    assert d["uprange"] @ m.downrange < -149.9
    assert d["left"] @ m.left > 149.9
    assert d["right"] @ m.left < -149.9


def test_derived_values(params):
    e, v = params.engine, params.vehicle
    assert e.T_max == pytest.approx(3 * 7060.0 + 4 * 1310.0 + 8 * 200.0)
    for cls in e.classes.values():
        assert cls.ve == pytest.approx(e.g0 * cls.isp)
        assert cls.mdot == pytest.approx(cls.thrust / cls.ve)
    assert v.m_prop == pytest.approx(v.m_wet - v.m_dry)
    assert params.tw0 == pytest.approx(e.T_max / (v.m_wet * params.moon.g_surf))


def test_prechecks_pass_for_default(params):
    assert report_prechecks(params, out=lambda *_: None) == []


def test_rejects_bad_units(make_params):
    with pytest.raises(ValueError):
        make_params({"moon": {"radius": 1737.4}})   # km instead of m
    with pytest.raises(ValueError):
        make_params({"sites": {"start": {"lat": -189.9}}})
    with pytest.raises(ValueError):
        make_params({"vehicle": {"m_dry": 1e6}})        # dry mass above wet mass


def test_divert_radius_override(make_params):
    p = make_params({"mission": {"divert_radius": 50.0}})
    assert np.linalg.norm(p.mission.sites["left"] - p.mission.r_target) == pytest.approx(50.0, abs=1e-3)


def test_touchdown_tilt_units_and_range(make_params):
    p = make_params({"limits": {"td_tilt_max_deg": 20.0}})
    assert p.limits.td_tilt_max == pytest.approx(np.radians(20.0))
    assert make_params({"limits": {"td_tilt_max_deg": 0.0}}).limits.td_tilt_max == 0.0
    for angle in (-1, 90, float("nan")):
        with pytest.raises(ValueError, match="td_tilt_max_deg"):
            make_params({"limits": {"td_tilt_max_deg": angle}})

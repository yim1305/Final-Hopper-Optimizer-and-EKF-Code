"""Common comparison inputs and tracker update-age boundary acceptance (limit read from mission.yaml)."""

from dataclasses import asdict
from types import SimpleNamespace
import json
import numpy as np
import pytest
from mission.config import CASES, parameters
from mission.terrain import DemTerrain
from mission import trackers
from hopper.tracker_cache import json_value


def test_reports_reject_a_different_sizing_source(terrain, tmp_path, monkeypatch):
    from mission import report
    from mission.config import physical_identity

    p = parameters('a2200_08', terrain)
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    out = tmp_path/'results/a2200_08'
    out.mkdir(parents=True)
    (out/'sizing.npz').write_bytes(b'different sizing source')
    summary = dict(status='numerically_validated_dem_flight',
                   physical_parameters_sha256=physical_identity(p),trajectory_sha256='old source')
    for name,value in [('summary.json',summary),('validation.json',dict(passed=True)),('trackers.json',dict(passed=True))]:
        (out/name).write_text(json.dumps(value))
    with pytest.raises(ValueError,match='inputs changed'):
        report.report_case('a2200_08',p,terrain)


@pytest.fixture(scope="module")
def terrain():
    return DemTerrain()


def test_only_requested_architectures_are_active(terrain):
    assert CASES == ("a2200_08", "a2200_10", "a2200_12", "a2200_14", "ar46_leros_rcs")
    expected = parameters("a2200_08", terrain)
    for case in CASES:
        p = parameters(case, terrain)
        for section in ("moon", "vehicle", "limits", "sensor", "mission", "star_tracker"):
            a = asdict(getattr(p, section))
            b = asdict(getattr(expected, section))
            assert json.dumps(a, default=json_value, sort_keys=True) == json.dumps(
                b, default=json_value, sort_keys=True
            )
        assert p.vehicle.m_dry == 2000
        assert p.limits.td_vert_speed_max == 1.5 and p.limits.td_horiz_speed_max == 0.5
        if case.startswith("a2200"):
            n = int(case[-2:])
            assert p.engine.classes["a2200"].count == n
            assert [dict(mode.throttled)["a2200"] for mode in p.engine.modes["ascent"]] == [
                n,
                (n + 4) // 2,
                4,
            ]
        else:
            assert p.engine.classes["large"].max_continuous_burn_s == 66.0
            assert p.engine.classes["small"].thrust == 1000.0


@pytest.mark.parametrize(
    "unavailable_fraction,passed",
    [(0.2, True), (1.0, True), (1.6, False)],
    ids=["well-inside-limit", "exactly-at-limit", "beyond-limit"],
)
def test_tracker_boundary_ages_obey_configured_update_gap_limit(
    terrain, monkeypatch, unavailable_fraction, passed
):
    """The first update age at a block start must not exceed `star_tracker.update_gap_max`.

    The limit is read from the loaded configuration (mission.yaml), so the three cases always straddle
    the real limit: the trackers are unusable (body rate too high) for 0.2, 1.0 and 1.6 times the limit
    at the start of a block that lasts three times the limit.
    """
    p = parameters("a2200_08", terrain)
    limit = p.star_tracker.update_gap_max
    unavailable_until = unavailable_fraction * limit
    duration = 3.0 * limit

    class Phase:
        start = 0.0
        end = duration
        ph = {"t": duration}

        def orientation(self, t):
            rate = np.where(t < unavailable_until, p.star_tracker.tracking_body_rate_max * 2, 0.0)
            return np.repeat(np.eye(3)[None], len(t), axis=0), rate

        def state(self, t):
            return np.tile(
                np.r_[
                    p.mission.r_start / np.linalg.norm(p.mission.r_start) * 50.0,
                    np.zeros(3),
                    p.vehicle.m_wet,
                ],
                (len(t), 1),
            )

    class Sky:
        def cone(self, *args):
            return dict(clear=True)

    monkeypatch.setattr(trackers, "SkyTerrain", Sky)
    monkeypatch.setattr(
        trackers,
        "blocks",
        lambda *args: [SimpleNamespace(name="test", start=0.0, end=duration, phases=[Phase()])],
    )
    result = trackers.tracker_checks(p, None)
    assert result["passed"] == passed
    if passed:
        report = result["reports"]["test"]
        assert report["first_update_age_s"] <= limit
        assert report["final_estimate_age_s"] <= limit
        assert report["max_update_gap_s"] <= limit

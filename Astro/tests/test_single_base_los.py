"""Single-station interval accounting for the Comms/C&DH handoff."""

import numpy as np
import pytest
from mission.los import summarize


def test_blackout_crosses_zero_duration_surface_handoff_without_splitting():
    result = summarize([0.0, 10.0, 20.0, 20.0, 30.0, 40.0], [2.0, -2.0, -2.0, -2.0, -2.0, 2.0])
    assert result["visible_duration_s"] == pytest.approx(10.0)
    assert result["longest_blackout_s"] == pytest.approx(30.0)
    assert len(result["outages"]) == 1
    assert result["outages"][0]["start_s"] == pytest.approx(5.0)
    assert result["outages"][0]["end_s"] == pytest.approx(35.0)


def test_duration_weighting_uses_crossings_and_not_sample_counts():
    result = summarize([0.0, 1.0, 10.0], [1.0, -1.0, -1.0])
    assert result["visible_fraction"] == pytest.approx(0.05)
    assert result["blocked_duration_s"] == pytest.approx(9.5)


def test_clearance_zero_counts_as_geometric_visibility():
    result = summarize([0.0, 10.0], [0.0, 0.0])
    assert result["visible_fraction"] == 1.0
    assert result["longest_blackout_s"] == 0.0


def test_missing_ray_result_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        summarize([0.0, 10.0], [np.nan, 1.0])

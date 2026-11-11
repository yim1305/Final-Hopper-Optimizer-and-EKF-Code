"""Optional numerical mounting-study settings; angles are converted to radians here."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class StarTracker:
    enabled: bool
    full_cone_fov: float
    sun_exclusion: float
    earth_exclusion: float
    moon_limb_margin: float
    tracking_body_rate_max: float
    update_gap_max: float
    required_phase_kinds: tuple
    tilt_bounds: tuple
    azimuth_bounds: tuple
    coarse_step: float
    earth_direction_I: tuple
    sun_direction_I: tuple
    geometry_label: str
    sample_step: float
    transition_tolerance: float
    angular_tolerance: float = math.radians(1e-7)
    rate_tolerance: float = math.radians(1e-7)
    enforce_body_rate: bool = True
    mount_model: str = "antipodal"
    elevation_bounds: tuple = ((math.radians(10), math.pi / 2),) * 2
    paired_azimuth_bounds: tuple = ((0., 2 * math.pi),) * 2
    initial_elevations: tuple = (math.radians(72.66012643997279), math.radians(54.935656370286914))
    initial_azimuths: tuple = (0., math.pi)
    top_plane_clearance_margin: float = 0.


def load_star_tracker(cfg):
    """Missing configuration disables the study; unsupported conventions fail explicitly."""
    if cfg is None:
        return None

    def require(ok, message):
        if not ok:
            raise ValueError(f"config error: star_tracker.{message}")

    def number(obj, key, default, lo, hi, closed_lo=False):
        value = float(obj.get(key, default))
        require(math.isfinite(value) and (value >= lo if closed_lo else value > lo) and value <= hi,
                f"{key} outside supported range")
        return value

    require(isinstance(cfg.get("enabled", False), bool), "enabled must be boolean")
    require(isinstance(cfg.get("enforce_body_rate", True), bool), "enforce_body_rate must be boolean")
    model = cfg.get("mount_model", "antipodal")
    require(model in ("antipodal", "independent_top"), "mount_model must be antipodal or independent_top")
    require(cfg.get("n_trackers", 2) == 2, "requires exactly two trackers")
    require(cfg.get("antipodal", model == "antipodal") is (model == "antipodal"),
            "antipodal flag must agree with mount_model")
    policies = {"bright_body_exclusion_reference": "boresight_to_center",
                "moon_limb_reference": "fov_edge", "body_roll_policy": "projected_downrange",
                "coast_attitude_policy": "inertial_endpoint_slerp"}
    for key, expected in policies.items():
        require(cfg.get(key, expected) == expected, f"{key} must be {expected}")
    for key in ("acquisition_time_s", "switching_delay_s", "initial_estimate_age_s"):
        require(cfg.get(key, 0.0) == 0.0, f"{key} must be zero in version 1")
    kinds = tuple(cfg.get("required_phase_kinds", ["ascent", "coast"]))
    require(kinds in (("ascent", "coast"), ("rise", "ascent", "coast")),
            "required_phase_kinds must be [ascent, coast] or [rise, ascent, coast]")
    search, geometry, analysis = (cfg.get(k, {}) for k in ("mount_search", "assumed_geometry", "analysis"))
    require(geometry.get("directions_frozen_in", "I") == "I", "directions_frozen_in must be I")
    require(geometry.get("surface_stay_s", 0.0) == 0.0, "surface_stay_s must be zero in version 1")

    def direction(key, default):
        a = np.asarray(geometry.get(key, default), dtype=float)
        require(a.shape == (3,) and np.isfinite(a).all() and np.isfinite(np.linalg.norm(a)) and np.linalg.norm(a) > 0,
                f"{key} must be a finite nonzero 3-vector")
        return tuple(a / np.linalg.norm(a))

    def bounds(key, default, lo, hi):
        a = tuple(float(v) for v in search.get(key, default))
        require(len(a) == 2 and all(math.isfinite(v) for v in a) and lo <= a[0] < a[1] <= hi,
                f"{key} invalid bounds")
        return tuple(math.radians(v) for v in a)

    fov = number(cfg, "full_cone_fov_deg", 20.0, 0, 179.999)
    limb = number(cfg, "moon_limb_margin_deg", 0.0, 0, 90, True)
    require(fov / 2 + limb < 90, "FOV half-angle + Moon limb margin must be < 90 deg")
    step = number(analysis, "sample_step_s", .05, 0, .05)
    tolerance = number(analysis, "transition_time_tolerance_s", .001, 0, step / 2)
    gap = number(cfg, "update_gap_max_s", 1.0, 0, float("inf"))
    require(step <= gap / 4, "sample_step_s must be <= update_gap_max_s / 4")
    plane_margin = number(cfg, "top_plane_fov_clearance_margin_deg", 0., 0, 89., True)
    min_elevation = fov / 2 + plane_margin
    require(min_elevation < 90, "full FOV must fit above the top plane")

    def paired_bounds(key, default, low, high):
        rows = search.get(key, default)
        require(isinstance(rows, (list, tuple)) and len(rows) == 2, f"{key} requires two bound pairs")
        result = []
        for row in rows:
            require(isinstance(row, (list, tuple)) and len(row) == 2, f"{key} requires two bound pairs")
            a, b = map(float, row)
            require(math.isfinite(a) and math.isfinite(b) and low <= a < b <= high, f"{key} invalid bounds")
            result.append((math.radians(a), math.radians(b)))
        return tuple(result)

    elevations = paired_bounds("elevation_bounds_deg", [[min_elevation, 90]] * 2, min_elevation, 90.)
    azimuths = paired_bounds("paired_azimuth_bounds_deg", [[0, 360]] * 2, 0., 360.)

    def seed(key, default, limits):
        row = search.get(key, default)
        require(isinstance(row, (list, tuple)) and len(row) == 2, f"{key} needs two values")
        values = tuple(math.radians(float(v)) for v in row)
        require(all(math.isfinite(v) and lo <= v <= hi for v, (lo, hi) in zip(values, limits)),
                f"{key} must lie inside mount bounds")
        return values

    initial_e = seed("initial_elevations_deg", [72.66012643997279, 54.935656370286914], elevations)
    initial_b = seed("initial_azimuths_deg", [0, 180], azimuths)
    return StarTracker(
        cfg.get("enabled", False), math.radians(fov),
        math.radians(number(cfg, "sun_exclusion_deg", 23.5, 0, 180, True)),
        math.radians(number(cfg, "earth_exclusion_deg", 23.5, 0, 180, True)), math.radians(limb),
        math.radians(number(cfg, "tracking_body_rate_max_deg_s", 6, 0, 360)), gap, kinds,
        bounds("tilt_bounds_deg", [0, 90], 0, 90), bounds("azimuth_bounds_deg", [0, 360], 0, 360),
        math.radians(number(search, "coarse_step_deg", 10, 0, 90)),
        direction("earth_direction_I", [1, 0, 0]),
        direction("sun_direction_I", [.05059294007671333, -.9987193571841863, 0]),
        str(geometry.get("label", "synthetic_launch_aligned_geometry_not_ephemeris")), step, tolerance,
        enforce_body_rate=cfg.get("enforce_body_rate", True), mount_model=model,
        elevation_bounds=elevations, paired_azimuth_bounds=azimuths,
        initial_elevations=initial_e, initial_azimuths=initial_b,
        top_plane_clearance_margin=math.radians(plane_margin))

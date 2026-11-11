"""Load GN/nav_config.yaml and convert to SI units."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

GN_DIR = Path(__file__).resolve().parents[1]
ROOT = GN_DIR.parent
ARCSEC = math.pi / 180 / 3600
DEG = math.pi / 180


@dataclass(frozen=True)
class Imu:
    rate: float
    gyro_bias: float      # rad/s, 1-sigma per axis
    gyro_arw: float       # rad/sqrt(s)
    accel_bias: float     # m/s^2, 1-sigma per axis
    accel_vrw: float      # m/s/sqrt(s)
    gyro_range: float     # rad/s
    accel_range: float    # m/s^2


@dataclass(frozen=True)
class Tracker:
    rate: float
    sigma_cross: float    # rad
    sigma_roll: float     # rad
    max_body_rate: float  # rad/s


@dataclass(frozen=True)
class Trn:
    sigma: float          # m, 1-sigma per axis
    rate: float           # Hz
    altitude_band: tuple  # m


@dataclass(frozen=True)
class NavConfig:
    raw: dict
    case_dir: Path
    branch: str
    nav_rate: float
    checkpoint_alt: float
    store_rate: float
    imu: Imu
    tracker: Tracker
    trn: Trn
    position_sigma0: float
    velocity_sigma0: float
    runs: int
    seed: int
    cases: dict
    sweep: dict
    requirements: dict

    @property
    def dt(self):
        return 1.0 / self.nav_rate

    def steps_per(self, rate_hz):
        """Nav steps between updates at `rate_hz`; must be an integer."""
        n = self.nav_rate / rate_hz
        if abs(n - round(n)) > 1e-9 or round(n) < 1:
            raise ValueError(f"Update rate {rate_hz} Hz must divide the nav rate {self.nav_rate} Hz")
        return int(round(n))


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=None, overrides=None) -> NavConfig:
    path = Path(path) if path else GN_DIR / "nav_config.yaml"
    raw = yaml.safe_load(path.read_text())
    if overrides:
        raw = _merge(copy.deepcopy(raw), overrides)
    t, i, s, n, ini = raw["timing"], raw["imu"], raw["star_tracker"], raw["trn"], raw["initial"]
    g0 = 9.80665
    imu = Imu(rate=float(i["rate_hz"]),
              gyro_bias=float(i["gyro_bias_deg_hr"]) * DEG / 3600,
              gyro_arw=float(i["gyro_arw_deg_rthr"]) * DEG / 60,
              accel_bias=float(i["accel_bias_mg"]) * 1e-3 * g0,
              accel_vrw=float(i["accel_vrw_m_s_rthr"]) / 60,
              gyro_range=float(i["gyro_range_deg_s"]) * DEG,
              accel_range=float(i["accel_range_g"]) * g0)
    tracker = Tracker(rate=float(s["rate_hz"]), sigma_cross=float(s["sigma_cross_arcsec"]) * ARCSEC,
                      sigma_roll=float(s["sigma_roll_arcsec"]) * ARCSEC,
                      max_body_rate=float(s["max_body_rate_deg_s"]) * DEG)
    band = tuple(float(v) for v in n["altitude_band_m"])
    trn = Trn(sigma=float(n["sigma_m"]), rate=float(n["rate_hz"]), altitude_band=band)
    cfg = NavConfig(raw=raw, case_dir=ROOT / raw["trajectory"]["case_dir"], branch=raw["trajectory"]["branch"],
                    nav_rate=float(t["nav_rate_hz"]),
                    checkpoint_alt=float(t["descent_checkpoint_alt_m"]), store_rate=float(t["store_rate_hz"]),
                    imu=imu, tracker=tracker, trn=trn, position_sigma0=float(ini["position_sigma_m"]),
                    velocity_sigma0=float(ini["velocity_sigma_m_s"]), runs=int(raw["montecarlo"]["runs"]),
                    seed=int(raw["montecarlo"]["seed"]), cases=raw["cases"], sweep=raw["sweep"],
                    requirements=raw["requirements"])
    cfg.steps_per(tracker.rate)
    cfg.steps_per(trn.rate)
    cfg.steps_per(cfg.store_rate)
    return cfg

"""One mission configuration and five explicitly named architectures."""

from dataclasses import asdict, replace
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
import yaml
from hopper.params import load_params, _coerce_numbers, _deep_update
from hopper.problem import Solution
from hopper.tracker_cache import json_value

ROOT = Path(__file__).resolve().parents[1]
CONFIG = _coerce_numbers(yaml.safe_load((ROOT / "mission.yaml").read_text()))
SETTINGS = CONFIG["analysis"]
CASES = tuple(CONFIG["cases"])


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=json_value, allow_nan=False), encoding="utf-8"
    )


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_identity(p):
    values = copy.deepcopy(p) if isinstance(p, dict) else asdict(p)
    values.pop("solver")
    return hashlib.sha256(
        json.dumps(values, default=json_value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def descent_variant(overrides=None):
    """Opt-in descent settings from mission.yaml (all null = off), with explicit overrides on top."""
    settings = {**CONFIG["descent_variant"], **(overrides or {})}
    floor, tw = settings["thrust_floor_n"], settings["touchdown_thrust_to_weight_max"]
    if floor is not None and not floor > 0:
        raise ValueError("descent_variant.thrust_floor_n must be positive")
    if tw is not None and not tw > 0:
        raise ValueError("descent_variant.touchdown_thrust_to_weight_max must be positive")
    margin = settings.get("solve_pointing_rate_deg_s")
    if margin is not None and not margin > 0:
        raise ValueError("descent_variant.solve_pointing_rate_deg_s must be positive")
    return settings


def _lower_floor(modes, kinds, floor):
    """Engine-mode table with gamma_lo lowered to `floor` for `kinds`; the band may only widen, so a stage whose
    floor is already lower (for example a pulsing pool) is left as it is."""
    out = dict(modes)
    for kind in kinds:
        for mode in out.get(kind, ()):
            if not floor < mode.gamma_hi:
                raise ValueError(f"{kind}: thrust floor {floor} N is not below the stage maximum {mode.gamma_hi} N")
        if kind in out:
            out[kind] = [replace(mode, gamma_lo=min(mode.gamma_lo, float(floor))) for mode in out[kind]]
    return out


def parameters(case, terrain, updates=None, descent=None):
    override = copy.deepcopy(CONFIG["common"])
    definition = CONFIG["cases"][case]
    engine = copy.deepcopy(CONFIG["engine_types"][definition["architecture"]])
    if definition["architecture"] == "a2200":
        count = definition["count"]
        engine["classes"]["a2200"]["count"] = count
        for schedule in (engine["stages"], engine["return_stages"]):
            schedule["rise"] = [{"a2200": count}]
            for kind in ("ascent", "braking"):
                schedule[kind] = [{"a2200": n} for n in (count, (count + 4) // 2, 4)]
    override["engine"] = engine
    if "solver" in definition:
        _deep_update(override["solver"], definition["solver"])
    if updates:
        _deep_update(override, updates)
    engine = override["engine"]
    engine["classes"] = {**dict.fromkeys(("large", "small", "rcs", "a2200")), **engine["classes"]}
    for site in override["sites"].values():
        lat, lon = np.radians([site["lat"], site["lon"]])
        unit = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
        site["elev"] = float(terrain.height(tuple(unit * terrain.radius)))
    p = load_params(str(ROOT / "config.yaml"), override)
    sites = {
        name: r / np.linalg.norm(r) * (p.moon.radius + terrain.height(tuple(r)))
        for name, r in p.mission.sites.items()
    }
    p = replace(p, mission=replace(p.mission, sites=sites))
    settings = descent_variant(descent)
    if settings["thrust_floor_n"] is not None:
        floor, kinds = settings["thrust_floor_n"], settings["floor_phases"]
        engine = replace(
            p.engine,
            modes=_lower_floor(p.engine.modes, kinds, floor),
            return_modes=_lower_floor(p.engine.return_modes, kinds, floor),
        )
        p = replace(p, engine=engine)
    return p


def read_solution(path):
    """Read arrays; the caller must validate them before accepting a result."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"].item()))
        phases = {
            name: {
                **info,
                "hop": info.get("hop", 2 if "2_" in name else 1),
                "X": data[name + "_X"].copy(),
                "U": data[name + "_U"].copy(),
                "t": float(data[name + "_t"]),
            }
            for name, info in meta["phase_info"].items()
        }
    return Solution(
        phases,
        meta["z"],
        meta["status"],
        meta["success"],
        meta["iterations"],
        meta["solve_time"],
        meta["m_wet"],
    )

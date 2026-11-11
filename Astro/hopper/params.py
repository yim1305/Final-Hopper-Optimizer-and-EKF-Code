"""Single source of truth: loads config.yaml into frozen dataclasses and derives values."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import yaml

from .tracker_params import StarTracker, load_star_tracker

@dataclass(frozen=True)
class Moon:
    mu: float
    radius: float
    omega: float
    g_surf: float            # mu / R^2, used for scaling and the T/W checks

    @property
    def omega_vec(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.omega])


POWERED_KINDS = ("rise", "ascent", "braking", "observation", "terminal")
SINGLE_STAGE_KINDS = ("rise", "observation", "terminal")     # only ascent and braking may be split into stages
BURNS = (("rise", "ascent"), ("braking", "observation", "terminal"))   # continuous burns; coast sits between


@dataclass(frozen=True)
class EngineClass:
    name: str
    count: int
    thrust: float            # N per engine
    isp: float               # s
    pulsing: bool            # True: time-averaged thrust anywhere in [0, count * thrust]; False: full thrust or off
    ve: float                # g0 * Isp
    mdot: float              # kg/s per engine at full thrust
    throttle_min_fraction: float = 0.0  # >0: continuous throttle, with a prescribed active count per stage
    mixture_ratio: float | None = None  # optional class-specific O/F for propellant bookkeeping
    throttle_rate_fraction_s: float | None = None  # optional common per-engine command slew limit
    max_continuous_burn_s: float | None = None  # fixed/throttled engine's longest allowed uninterrupted firing


@dataclass(frozen=True)
class EngineMode:
    """One burn stage: a fixed set of non-pulsing engines firing, plus all pulsing engines of the pool.

    Thrust Gamma is continuous in [gamma_lo, gamma_hi]; the mass flow is affine in it, mdot = c0 + c1 * Gamma."""
    kind: str
    stage: int
    fixed: tuple             # ((class name, engines firing), ...) for the non-pulsing classes of the pool
    thrust_fixed: float      # N from the non-pulsing engines (constant)
    pulse_capacity: float    # N available from the pulsing engines
    gamma_lo: float
    gamma_hi: float
    c0: float
    c1: float
    throttled: tuple = ()    # ((class name, engines firing), ...), each between its throttle floor and full thrust

    def mdot(self, gamma):
        return self.c0 + self.c1 * gamma

    @property
    def ve_hi(self) -> float:
        """Effective exhaust velocity at maximum thrust of this stage."""
        return self.gamma_hi / self.mdot(self.gamma_hi)


@dataclass(frozen=True)
class Engine:
    classes: dict            # name -> EngineClass
    pools: dict              # kind -> tuple of class names allowed in that phase
    modes: dict              # kind -> list of EngineMode, one per burn stage
    g0: float
    pulse_min_fraction: float
    T_max: float             # largest thrust of any stage
    mixture_ratio: float = 1.65           # O/F by mass, only used to split the propellant into oxidizer and fuel
    relaxed: dict = field(default_factory=dict)   # kind -> single EngineMode: the non-pulsing engines as a throttle
    return_modes: dict = field(default_factory=dict)  # optional hop-2 schedule; hardware and pools stay shared

    def modes_for(self, hop: int = 1) -> dict:
        if hop not in (1, 2):
            raise ValueError("engine hop must be 1 or 2")
        return self.return_modes if hop == 2 and self.return_modes else self.modes

    def stage_count(self, kind: str, relaxed: bool = False, hop: int = 1) -> int:
        modes = self.modes_for(hop)
        return 1 if relaxed or kind not in modes else len(modes[kind])

    def mode(self, kind: str, stage: int = 0, relaxed: bool = False, hop: int = 1):
        """Engine mode of a phase; None for an unpowered phase (coast). `relaxed` picks the throttle relaxation."""
        modes = self.modes_for(hop)
        if kind not in modes:
            return None
        return self.relaxed[kind] if relaxed else modes[kind][stage]


@dataclass(frozen=True)
class Vehicle:
    m_wet: float
    m_dry: float
    m_prop: float


@dataclass(frozen=True)
class Mission:
    divert_radius: float
    gate_alt_bounds: tuple
    r_start: np.ndarray                  # absolute, Moon-fixed
    r_target: np.ndarray
    up: np.ndarray                       # target-local frame
    downrange: np.ndarray
    left: np.ndarray
    sites: dict                          # branch -> absolute site position
    L: float                             # start-to-target distance (chord)
    gate_over_target: bool = False       # hop-1 gate horizontally directly above the target
    mode: str = "maximise_margin"        # size | maximise_margin
    reserve_factor: float = 1.1          # loaded propellant / worst-case round-trip propellant used
    landing_tolerance: float = 0.0       # m from the selected nominal landing point


@dataclass(frozen=True)
class Limits:
    h_rise: float
    min_altitude: float
    glide_slope: float                   # rad
    tilt_rate_max: float                 # rad/s
    td_vert_speed_max: float
    td_horiz_speed_max: float
    phase_duration_bounds: dict
    td_tilt_max: float                   # rad; body-up approximated by resultant thrust direction
    stage_duration_min: float = 0.1      # s; shortest later burn stage, so a stage the optimizer does not need can collapse


@dataclass(frozen=True)
class Solver:
    nodes: dict
    ipopt: dict
    seed_nodes: dict | None = None


@dataclass(frozen=True)
class Sensor:
    fov: tuple                          # usable full FOV angles, radians after edge margin
    max_range: float
    min_altitude: float
    descent_speed_bounds: tuple
    horizontal_speed_max: float
    pointing_rate_max: float             # rad/s
    observation_time_min: float
    processing_delay: float
    edge_margin: float = 0.0             # rad per edge inside the hardware FOV
    enforce_pointing: bool = True        # False drops the camera FOV / line-of-sight / pointing-rate constraints


@dataclass(frozen=True)
class Params:
    moon: Moon
    engine: Engine
    vehicle: Vehicle
    mission: Mission
    limits: Limits
    solver: Solver
    sensor: Sensor
    star_tracker: StarTracker | None = None

    @property
    def tw0(self) -> float:
        """Initial lunar thrust-to-weight at wet mass."""
        return self.engine.T_max / (self.vehicle.m_wet * self.moon.g_surf)

    @property
    def scales(self) -> "Scales":
        g = self.moon.g_surf
        L = self.mission.L
        T = math.sqrt(L / g)
        M = self.vehicle.m_wet
        return Scales(L=L, T=T, M=M, V=L / T, F=M * g, g=g)


@dataclass(frozen=True)
class Scales:
    L: float   # length
    T: float   # time
    M: float   # mass
    V: float   # velocity
    F: float   # force
    g: float   # acceleration


def site_vector(lat_deg: float, lon_deg: float, elev: float, radius: float) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    return (radius + elev) * np.array(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])


def _unit(v):
    return v / np.linalg.norm(v)


def target_frame(r_start: np.ndarray, r_target: np.ndarray):
    up = _unit(r_target)
    d = r_target - r_start
    d = d - np.dot(d, up) * up
    if np.linalg.norm(d) < 1e-9:
        raise ValueError("start and target sites coincide; downrange is undefined")
    downrange = _unit(d)
    left = np.cross(up, downrange)
    return up, downrange, left


def divert_sites(r_target, downrange, left, radius_div):
    """Target moved radius_div along +/-downrange and +/-left, projected back to the surface radius."""
    Rt = np.linalg.norm(r_target)
    out = {}
    for name, direction in (("downrange", downrange), ("uprange", -downrange),
                            ("left", left), ("right", -left)):
        p = r_target + radius_div * direction
        out[name] = p / np.linalg.norm(p) * Rt
    return out


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _coerce_numbers(obj):
    """PyYAML (YAML 1.1) reads '4.9e12' (no exponent sign) as a string; turn such strings into floats."""
    if isinstance(obj, dict):
        return {k: _coerce_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numbers(v) for v in obj]
    if isinstance(obj, str):
        try:
            return float(obj)
        except ValueError:
            return obj
    return obj


def _require(cond: bool, msg: str):
    if not cond:
        raise ValueError(f"config error: {msg}")


def load_params(path: str, overrides: dict | None = None) -> Params:
    """Load and validate config.yaml. `overrides` is a nested dict deep-merged over the file
    (used for sweeps and sensitivity tests, so no module ever hardcodes a value)."""
    with open(path, "r") as fh:
        cfg = _coerce_numbers(yaml.safe_load(fh))
    if overrides:
        cfg = _deep_update(copy.deepcopy(cfg), overrides)

    m = cfg["moon"]
    _require(m["radius"] > 1e5, "moon.radius must be in metres (got < 100 km)")
    _require(m["mu"] > 1e9, "moon.mu must be in m^3/s^2")
    _require(0 <= m["omega"] < 1e-3, "moon.omega must be in rad/s")
    moon = Moon(m["mu"], m["radius"], m["omega"], m["mu"] / m["radius"] ** 2)

    g0 = cfg["g0"]
    engine = _load_engine(cfg["engine"], g0)

    v = cfg["vehicle"]
    _require(v["m_wet"] > v["m_dry"] > 0, "need m_wet > m_dry > 0 (kg)")
    vehicle = Vehicle(v["m_wet"], v["m_dry"], v["m_wet"] - v["m_dry"])

    s = cfg["sites"]
    for k in ("start", "target"):
        _require(-90 <= s[k]["lat"] <= 90, f"sites.{k}.lat must be in degrees [-90, 90]")
        _require(-360 <= s[k]["lon"] <= 360, f"sites.{k}.lon must be in degrees")
    r_start = site_vector(s["start"]["lat"], s["start"]["lon"], s["start"]["elev"], moon.radius)
    r_target = site_vector(s["target"]["lat"], s["target"]["lon"], s["target"]["elev"], moon.radius)
    up, downrange, left = target_frame(r_start, r_target)

    ms = cfg["mission"]
    _require(ms["divert_radius"] >= 0, "mission.divert_radius >= 0")
    _require(ms["landing_tolerance"] >= 0, "mission.landing_tolerance >= 0")
    _require(ms.get("mode", "maximise_margin") in ("size", "maximise_margin"),
             "mission.mode must be 'size' or 'maximise_margin'")
    _require(float(ms.get("propellant_reserve_factor", 1.1)) >= 1.0, "mission.propellant_reserve_factor >= 1")
    lo, hi = ms["gate_alt_bounds"]
    _require(0 <= lo < hi, "mission.gate_alt_bounds must be [lo, hi] with 0 <= lo < hi")
    sites = {"nominal": r_target}
    sites.update(divert_sites(r_target, downrange, left, ms["divert_radius"]))
    mission = Mission(ms["divert_radius"], (lo, hi), r_start, r_target,
                      up, downrange, left, sites, float(np.linalg.norm(r_target - r_start)),
                      bool(ms.get("gate_over_target", False)), ms.get("mode", "maximise_margin"),
                      float(ms.get("propellant_reserve_factor", 1.1)), float(ms["landing_tolerance"]))

    se = cfg["sensor"]
    margin = se["edge_margin_deg"]
    full_fov = tuple(float(a) for a in se["fov_deg"])
    _require(len(full_fov) == 2 and all(0 < a < 180 for a in full_fov), "sensor hardware FOV in (0, 180)")
    fov = tuple(a - 2 * margin for a in full_fov)
    _require(len(fov) == 2 and margin >= 0 and all(0 < a < 180 for a in fov), "sensor usable FOV in (0, 180)")
    _require(se["max_range"] > se["min_altitude"] > 0, "sensor range > minimum altitude > 0")
    vmin, vmax = se["descent_speed_bounds"]
    _require(0 < vmin < vmax, "sensor descent speed bounds require 0 < minimum < maximum")
    _require(se["horizontal_speed_max"] > 0 and se["pointing_rate_max_deg_s"] > 0, "sensor motion limits > 0")
    _require(se["observation_time_min"] > 0 and se["processing_delay"] >= 0, "sensor observation > 0, delay >= 0")
    sensor = Sensor(tuple(math.radians(a) for a in fov), se["max_range"], se["min_altitude"], (vmin, vmax),
                    se["horizontal_speed_max"], math.radians(se["pointing_rate_max_deg_s"]),
                    se["observation_time_min"], se["processing_delay"], math.radians(margin),
                    bool(se.get("enforce_pointing", True)))

    li = cfg["limits"]
    _require(0 < li["glide_slope_deg"] < 90, "limits.glide_slope_deg in (0, 90)")
    _require(li["tilt_rate_max_deg_s"] > 0, "limits.tilt_rate_max_deg_s > 0")
    _require(0 <= li["td_tilt_max_deg"] < 90, "limits.td_tilt_max_deg in [0, 90)")
    _require(li["td_vert_speed_max"] >= 0 and li["td_horiz_speed_max"] >= 0,
             "touchdown speed limits must be nonnegative")
    pdb = {k: tuple(vv) for k, vv in li["phase_duration_bounds"].items()}
    for k in ("rise", "ascent", "coast", "braking", "observation", "terminal"):
        _require(k in pdb and 0 < pdb[k][0] < pdb[k][1], f"limits.phase_duration_bounds.{k}")
    limits = Limits(li["h_rise"], li["min_altitude"], math.radians(li["glide_slope_deg"]),
                    math.radians(li["tilt_rate_max_deg_s"]), li["td_vert_speed_max"],
                    li["td_horiz_speed_max"], pdb, math.radians(li["td_tilt_max_deg"]),
                    float(li.get("stage_duration_min", 0.1)))
    _require(0 < limits.stage_duration_min <= min(pdb[k][0] for k in pdb),
             "limits.stage_duration_min must be positive and not above any phase minimum duration")
    _require(sensor.observation_time_min + sensor.processing_delay <= pdb["observation"][1],
             "observation duration ceiling must allow acquisition plus processing")

    so = cfg["solver"]
    for k in ("rise", "ascent", "coast", "braking", "observation", "terminal"):
        _require(int(so["nodes"][k]) >= 2, f"solver.nodes.{k} >= 2")
    seed_nodes = so.get("seed_nodes")
    if seed_nodes is not None:
        for k in so["nodes"]:
            _require(k in seed_nodes and int(seed_nodes[k]) >= 2, f"solver.seed_nodes.{k} >= 2")
        seed_nodes = {k: int(v) for k, v in seed_nodes.items()}
    solver = Solver({k: int(vv) for k, vv in so["nodes"].items()}, dict(so["ipopt"]), seed_nodes)

    star_tracker = load_star_tracker(cfg.get("star_tracker"))
    return Params(moon, engine, vehicle, mission, limits, solver, sensor, star_tracker)


def _make_mode(kind: str, stage: int, pool: tuple, firing: dict, classes: dict, pulse_min_fraction: float) -> EngineMode:
    fixed_names = [c for c in pool if not classes[c].pulsing and not classes[c].throttle_min_fraction]
    throttle_names = [c for c in pool if classes[c].throttle_min_fraction]
    pulsing = [classes[c] for c in pool if classes[c].pulsing]
    for name, n in firing.items():
        _require(name in classes, f"engine.stages.{kind}[{stage}]: unknown engine class '{name}'")
        _require(int(n) == n, f"engine.stages.{kind}[{stage}]: {name} count must be an integer")
        _require(int(n) == 0 or name in fixed_names + throttle_names,
                 f"engine.stages.{kind}[{stage}]: '{name}' is not a non-pulsing class of the {kind} pool")
        _require(0 <= int(n) <= classes[name].count, f"engine.stages.{kind}[{stage}]: {name} count must be 0..{classes[name].count}")
    fixed = tuple((c, int(firing.get(c, 0))) for c in fixed_names)
    throttled = tuple((c, int(firing.get(c, classes[c].count))) for c in throttle_names)
    thrust_fixed = sum(n * classes[c].thrust for c, n in fixed)
    mdot_fixed = sum(n * classes[c].mdot for c, n in fixed)
    variable = [(c, c.count) for c in pulsing] + [(classes[c], n) for c, n in throttled if n]
    _require(not throttled or len(variable) <= 1,
             "a continuously throttled pool supports only one variable-thrust class")
    capacity = sum(n * c.thrust for c, n in variable)
    throttle_floor = sum(n * c.thrust * c.throttle_min_fraction for c, n in variable)
    if capacity > 0:
        # Pulsing classes share one mean duty fraction. Their individual mass
        # flows therefore add even when their specific impulses differ.
        if len({c.ve for c, _ in variable}) == 1:
            ve = variable[0][0].ve
            c1, c0 = 1.0 / ve, mdot_fixed - thrust_fixed / ve
        else:
            c1 = sum(n * c.mdot for c, n in variable) / capacity
            c0 = mdot_fixed - thrust_fixed * c1
        lo = thrust_fixed + throttle_floor
        if lo == 0:
            lo = pulse_min_fraction * capacity
        hi = thrust_fixed + capacity
    else:
        c1, c0, lo, hi = 0.0, mdot_fixed, thrust_fixed, thrust_fixed
    _require(hi > 0, f"engine.stages.{kind}[{stage}] has no thrust available")
    return EngineMode(kind, stage, fixed, thrust_fixed, capacity, lo, hi, c0, c1, throttled)


def _load_engine(e: dict, g0: float) -> Engine:
    classes = {}
    for name, c in e["classes"].items():
        if c is None:       # an overlay can remove baseline hardware explicitly
            continue
        _require(int(c["count"]) >= 1 and c["thrust"] > 0 and c["isp"] > 0,
                 f"engine.classes.{name}: count >= 1, thrust > 0 and isp > 0")
        ve = g0 * c["isp"]
        throttle_min = float(c.get("throttle_min_fraction", 0.0))
        _require(0 <= throttle_min < 1, f"engine.classes.{name}.throttle_min_fraction must be in [0, 1)")
        _require(not (throttle_min and c.get("pulsing", False)),
                 f"engine.classes.{name} cannot be both pulsing and continuously throttled")
        mr = c.get("mixture_ratio")
        _require(mr is None or mr > 0, f"engine.classes.{name}.mixture_ratio must be positive")
        rate = c.get("throttle_rate_fraction_s")
        _require(rate is None or (throttle_min > 0 and rate > 0),
                 f"engine.classes.{name}.throttle_rate_fraction_s needs a throttled engine and positive rate")
        burn_cap = c.get("max_continuous_burn_s")
        _require(burn_cap is None or (burn_cap > 0 and not c.get("pulsing", False)),
                 f"engine.classes.{name}.max_continuous_burn_s needs a positive non-pulsing limit")
        classes[name] = EngineClass(name, int(c["count"]), float(c["thrust"]), float(c["isp"]),
                                    bool(c.get("pulsing", False)), ve, c["thrust"] / ve, throttle_min, mr, rate, burn_cap)
    frac = float(e.get("pulse_min_fraction", 0.0))
    _require(0 <= frac < 1, "engine.pulse_min_fraction must be in [0, 1)")
    pools, modes = {}, {}
    for kind in POWERED_KINDS:
        _require(kind in e["pools"], f"engine.pools.{kind} is missing")
        pools[kind] = tuple(e["pools"][kind])
        _require(all(c in classes for c in pools[kind]), f"engine.pools.{kind} names an unknown engine class")
        specs = e.get("stages", {}).get(kind) or [{}]
        _require(kind not in SINGLE_STAGE_KINDS or len(specs) == 1, f"engine.stages.{kind} must have exactly one stage")
        modes[kind] = [_make_mode(kind, i, pools[kind], dict(spec or {}), classes, frac) for i, spec in enumerate(specs)]
    return_modes = {}
    if e.get("return_stages"):
        _require(set(e["return_stages"]) <= set(POWERED_KINDS), "engine.return_stages has an unknown phase")
        for kind in POWERED_KINDS:
            specs = e["return_stages"].get(kind)
            if specs is None:
                return_modes[kind] = modes[kind]
                continue
            _require(bool(specs), f"engine.return_stages.{kind} cannot be empty")
            _require(kind not in SINGLE_STAGE_KINDS or len(specs) == 1,
                     f"engine.return_stages.{kind} must have exactly one stage")
            return_modes[kind] = [_make_mode(kind, i, pools[kind], dict(spec or {}), classes, frac)
                                  for i, spec in enumerate(specs)]
    # fixed-thrust engines cannot relight inside a burn: firing counts may only fall along rise->ascent and
    # braking->observation->terminal (they may rise again after the coast)
    for schedule in (modes, return_modes):
        if not schedule:
            continue
        for burn in BURNS:
            seq = [m for kind in burn for m in schedule[kind]]
            for a, b in zip(seq[:-1], seq[1:]):
                before, after = dict(a.fixed + a.throttled), dict(b.fixed + b.throttled)
                for name in set(before) | set(after):
                    _require(after.get(name, 0) <= before.get(name, 0),
                             f"engine.stages: {name} engines cannot relight inside a burn "
                             f"({a.kind} stage {a.stage + 1} -> {b.kind} stage {b.stage + 1})")
    T_max = max(m.gamma_hi for schedule in (modes, return_modes) for ms in schedule.values() for m in ms)
    relaxed = {kind: _make_relaxed(kind, pools[kind], classes, frac, modes[kind][0]) for kind in POWERED_KINDS}
    return Engine(classes, pools, modes, g0, frac, T_max, float(e.get("mixture_ratio", 1.65)), relaxed, return_modes)


def _make_relaxed(kind: str, pool: tuple, classes: dict, pulse_min_fraction: float, plain: EngineMode) -> EngineMode:
    """Throttle relaxation of a pool: every engine of the pool, the non-pulsing ones too, becomes part of one
    continuous throttle with a single mass-flow-weighted Isp (mdot = Gamma / ve_avg, from the all-engines-on flow).

    This is only a seed for the staged problem, so a plain linear mass flow is used: it keeps the problem as smooth
    and well conditioned as a single throttleable engine. Pools without any non-pulsing engine are already
    continuous, so their only mode is used as is."""
    fixed = [classes[c] for c in pool if not classes[c].pulsing and not classes[c].throttle_min_fraction]
    pulsing = [classes[c] for c in pool if classes[c].pulsing or classes[c].throttle_min_fraction]
    if not fixed:
        return plain
    _require(len(fixed) == 1 and pulsing, f"the {kind} pool needs one non-pulsing class and a pulsing one to be relaxed")
    f, capacity, ve_p = fixed[0], sum(c.count * c.thrust for c in pulsing), pulsing[0].ve
    hi = capacity + f.count * f.thrust
    pulse_flow = (capacity / ve_p if len({c.ve for c in pulsing}) == 1 else
                  sum(c.count * c.mdot for c in pulsing))
    mdot_hi = pulse_flow + f.count * f.mdot
    return EngineMode(kind, 0, ((f.name, f.count),), f.count * f.thrust, capacity, pulse_min_fraction * capacity,
                      hi, 0.0, mdot_hi / hi)


def report_prechecks(p: Params, out=print) -> list:
    """Print start-up pre-checks; returns the list of warning strings."""
    warnings = []
    g = p.moon.g_surf
    out(f"g_moon = {g:.4f} m/s^2   T_max = {p.engine.T_max:.1f} N")
    for kind, modes in p.engine.modes.items():
        for m in modes:
            fixed = ", ".join(f"{n} {c}" for c, n in m.fixed if n) or "no fixed-thrust engines"
            out(f"  {kind:<12} stage {m.stage + 1}: {fixed} + pulsing {m.pulse_capacity:.0f} N -> "
                f"Gamma {m.gamma_lo:.0f}..{m.gamma_hi:.0f} N, ve at max {m.ve_hi:.0f} m/s "
                f"(Isp {m.ve_hi / p.engine.g0:.1f} s)")
    out(f"m_wet = {p.vehicle.m_wet:.1f} kg   m_dry = {p.vehicle.m_dry:.1f} kg   "
        f"m_prop = {p.vehicle.m_prop:.1f} kg   hop distance L = {p.mission.L / 1e3:.2f} km")
    ok = p.tw0 > 1.0
    out(f"[{'OK' if ok else 'FAIL'}] initial lunar T/W = {p.tw0:.3f} (need > 1 at wet mass)")
    if not ok:
        warnings.append("T/W <= 1 at wet mass: vehicle cannot lift off")
    rise_hi = p.engine.mode("rise").gamma_hi
    ok = rise_hi > p.vehicle.m_wet * g
    out(f"[{'OK' if ok else 'FAIL'}] rise-stage T/W = {rise_hi / (p.vehicle.m_wet * g):.3f} (need > 1 at wet mass)")
    if not ok:
        warnings.append("the rise stage cannot lift the wet vehicle")
    desc_hi = p.engine.mode("terminal").gamma_hi
    ok = desc_hi > p.vehicle.m_dry * g
    out(f"[{'OK' if ok else 'FAIL'}] terminal-stage thrust {desc_hi:.0f} N vs dry weight "
        f"{p.vehicle.m_dry * g:.0f} N (must be able to hover the dry vehicle)")
    if not ok:
        warnings.append("terminal-stage thrust cannot hover the dry vehicle")
    h_min = math.tan(p.limits.glide_slope) * p.mission.divert_radius
    ok = p.mission.gate_alt_bounds[0] > h_min
    out(f"[{'OK' if ok else 'WARN'}] gate floor {p.mission.gate_alt_bounds[0]:.0f} m vs glide-slope minimum "
        f"{h_min:.0f} m (tan(slope) x divert radius) for a gate directly above the target")
    if not ok:
        warnings.append("gate_alt_bounds floor is not above the glide-slope minimum: divert branches infeasible")
    return warnings

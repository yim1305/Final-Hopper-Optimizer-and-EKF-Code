"""Delta-v, propellant and burn-timeline bookkeeping from a solved round trip."""
from __future__ import annotations

import numpy as np

from .params import Params
from .phases import stage_name

HOP_KINDS = ("rise", "ascent", "coast", "braking", "observation", "terminal")


def chain(branch: str, p: Params | None = None, relaxed: bool = False) -> list:
    """Phase names in flight order: hop 1 (start -> divert site `branch`), then hop 2 (-> start site).

    Pass `p` when ascent or braking are split into burn stages; without it (or for a relaxed solve) every kind has
    one phase."""
    def hop(base_names, hop_index):
        names = []
        for kind, base in zip(HOP_KINDS, base_names):
            names += [stage_name(base, k) for k in range(p.engine.stage_count(kind, relaxed, hop_index) if p else 1)]
        return names
    hop1 = hop(list(HOP_KINDS[:-1]) + [f"terminal_{branch}"], 1)
    hop2 = hop([f"{k}2_{branch}" for k in HOP_KINDS], 2)
    return hop1 + hop2


def phase_mode(p: Params, ph: dict):
    """EngineMode of a solved phase dict (None for coast)."""
    return p.engine.mode(ph["kind"], ph.get("stage", 0), ph.get("relaxed", False), ph.get("hop", 1)) if ph["powered"] else None


def phase_dv(p: Params, ph: dict) -> float:
    """Delta-v = integral of Gamma/m dt, trapezoid over the collocation nodes (Gamma is linear between them)."""
    if not ph["powered"]:
        return 0.0
    n = ph["X"].shape[1]
    t = np.linspace(0.0, ph["t"], n)
    return float(np.trapezoid(ph["U"][3] / ph["X"][6], t))


def propellant_by_class(p: Params, ph: dict) -> dict:
    """Propellant of a powered phase by engine class. Non-pulsing engines burn their full-thrust flow for the whole
    stage; the variable demand shares one duty/throttle fraction. Split its
    propellant by full-demand mass-flow capacity, accounting for each Isp."""
    if not ph["powered"]:
        return {}
    mode = phase_mode(p, ph)
    out = {name: n * p.engine.classes[name].mdot * ph["t"] for name, n in mode.fixed}
    pulsing_total = (ph["X"][6, 0] - ph["X"][6, -1]) - sum(out.values())
    variable = {c: p.engine.classes[c].count for c in p.engine.pools[ph["kind"]] if p.engine.classes[c].pulsing}
    variable.update(dict(mode.throttled))
    same_isp = len({p.engine.classes[c].ve for c, n in variable.items() if n}) <= 1
    weights = {c: n * (p.engine.classes[c].thrust if same_isp else p.engine.classes[c].mdot)
               for c, n in variable.items()}
    capacity = sum(weights.values())
    for c, n in variable.items():
        out[c] = pulsing_total * weights[c] / capacity if capacity else 0.0
    return out


def phase_table(p: Params, sol, branch: str, relaxed: bool = False):
    rows, t = [], 0.0
    for name in chain(branch, p, relaxed):
        ph = sol.phases[name]
        rows.append(dict(phase=name, t_start=t, t_end=t + ph["t"], duration=ph["t"],
                         engine="on" if ph["powered"] else "off", dv=phase_dv(p, ph),
                         propellant=ph["X"][6, 0] - ph["X"][6, -1]))
        t += ph["t"]
    return rows


def engine_starts(rows) -> int:
    starts, prev = 0, False
    for r in rows:
        on = r["engine"] == "on"
        starts += int(on and not prev)
        prev = on
    return starts


def propellant_split(p: Params, prop: float) -> dict:
    """Split a propellant mass into fuel and oxidizer with the mixture ratio O/F: fuel = P/(1+MR), ox = MR P/(1+MR)."""
    mr = p.engine.mixture_ratio
    return dict(fuel=prop / (1.0 + mr), oxidizer=prop * mr / (1.0 + mr))


def class_usage(p: Params, sol, branch: str):
    """(propellant kg by engine class, ignitions by non-pulsing class) over the whole round trip.

    A non-pulsing engine ignites once at the start of each burn (a burn is a run of powered phases; each hop starts
    a new one, as the engines are off on the surface) and only shuts down within it."""
    prop = {c: 0.0 for c in p.engine.classes}
    ignitions = {c: None if cls.pulsing else 0 for c, cls in p.engine.classes.items()}
    burning = False
    for name in chain(branch, p):
        ph = sol.phases[name]
        if name.startswith("rise"):
            burning = False                       # engines are off on the surface between hops
        if not ph["powered"]:
            burning = False
            continue
        for c, kg in propellant_by_class(p, ph).items():
            prop[c] += kg
        if not burning:
            m = phase_mode(p, ph)
            for c, n in m.fixed + m.throttled:
                ignitions[c] += n
        burning = True
    return prop, ignitions


def burn_activations(p: Params, sol, branch: str):
    """Individual engine activations per burn, using capacity-share pulse allocation.

    Pulsing classes share a duty fraction across their installed engines. Pulse
    frequency/minimum impulse bit are unspecified, so their actual ignition
    counts cannot be inferred from this time-averaged trajectory.
    """
    totals = {c: 0 for c in p.engine.classes}
    active = set()
    for name in chain(branch, p):
        ph = sol.phases[name]
        if ph["kind"] == "rise" or not ph["powered"]:
            active.clear()
        if not ph["powered"]:
            continue
        mode = phase_mode(p, ph)
        firing = dict(mode.fixed + mode.throttled)
        pulse_on = np.max(ph["U"][3] - mode.thrust_fixed) > 1e-3
        for c in p.engine.pools[ph["kind"]]:
            cls = p.engine.classes[c]
            count = cls.count if cls.pulsing and pulse_on else firing.get(c, 0)
            if count and c not in active:
                totals[c] += count
                active.add(c)
    return totals


def thrust_tilt_deg(thrust, up) -> float:
    """Actual thrust angle to the surface normal; used as the body-tilt approximation."""
    vertical = float(thrust @ up)
    horizontal = np.linalg.norm(thrust - vertical * up)
    return float(np.degrees(np.arctan2(horizontal, vertical)))


def round_trip_summary(p: Params, sol, branch: str, lambert_dv_hop1: float, lambert_dv_hop2: float) -> dict:
    """Per-hop and total numbers for a solved round trip (hop 2 = divert site -> start landing region).

    Hop 1 includes the divert terminal descent. The surface stay between the hops is ignored, so engine starts
    are counted per hop and summed (the engine is off on the surface)."""
    rows = phase_table(p, sol, branch)
    hop1 = [r for r in rows if "2_" not in r["phase"]]
    hop2 = [r for r in rows if "2_" in r["phase"]]
    m_start = sol.phases["rise"]["X"][6, 0]
    m_land1 = sol.phases[f"terminal_{branch}"]["X"][6, -1]
    m_final = sol.phases[f"terminal2_{branch}"]["X"][6, -1]
    out = {}
    for tag, hop, lam in (("hop1", hop1, lambert_dv_hop1), ("hop2", hop2, lambert_dv_hop2)):
        dv = sum(r["dv"] for r in hop)
        terminal_name = f"terminal_{branch}" if tag == "hop1" else f"terminal2_{branch}"
        site_abs = p.mission.sites[branch] if tag == "hop1" else p.mission.r_start
        touchdown = sol.phases[terminal_name]
        actual_site = p.mission.r_start + touchdown["X"][0:3, -1]
        up = actual_site / np.linalg.norm(actual_site)
        v = touchdown["X"][3:6, -1]
        vup = float(v @ up)
        out[tag] = dict(landing_target_offset=float(np.linalg.norm(actual_site - site_abs)), lambert_dv=lam, finite_burn_dv=dv,
                        gravity_steering_loss=dv - lam,
                        touchdown_tilt_deg=thrust_tilt_deg(touchdown["U"][0:3, -1], up),
                        touchdown_vertical_speed=vup, touchdown_horizontal_speed=float(np.linalg.norm(v - vup * up)),
                        propellant=sum(r["propellant"] for r in hop),
                        burn_time=sum(r["duration"] for r in hop if r["engine"] == "on"),
                        duration=sum(r["duration"] for r in hop), engine_starts=engine_starts(hop))
    prop_used = m_start - m_final
    loaded = m_start - p.vehicle.m_dry
    out["propellant_by_class"], out["ignitions"] = class_usage(p, sol, branch)
    out["burn_activations"] = burn_activations(p, sol, branch)
    out["ignition_note"] = ("Pulsing ignition counts are unresolved: pulse frequency and minimum impulse bit are not modeled."
                            if any(c.pulsing for c in p.engine.classes.values()) else
                            "Ignitions count prescribed active engines at each continuous burn; no pulsing is modeled.")
    fuel = sum(kg / (1 + (p.engine.classes[c].mixture_ratio or p.engine.mixture_ratio))
               for c, kg in out["propellant_by_class"].items())
    out["propellant_used_split"] = dict(fuel=fuel, oxidizer=prop_used-fuel)
    out["propellant_loaded_split"] = {k: v * loaded / prop_used for k, v in out["propellant_used_split"].items()}
    out["total"] = dict(m_start=m_start, m_land_hop1=m_land1, m_final=m_final, propellant_used=prop_used,
                        propellant_loaded=loaded,
                        finite_burn_dv=out["hop1"]["finite_burn_dv"] + out["hop2"]["finite_burn_dv"],
                        engine_starts=out["hop1"]["engine_starts"] + out["hop2"]["engine_starts"],
                        margin=m_final - p.vehicle.m_dry,
                        loaded_over_used=loaded / prop_used)
    return out


def thrust_discard_flags(p: Params, sol, tol: float = 0.01):
    """Powered nodes where |T_vec| < 0.99 Gamma (solver is discarding thrust it pays fuel for)."""
    flags = {}
    for name, ph in sol.phases.items():
        if not ph["powered"]:
            continue
        T = np.linalg.norm(ph["U"][0:3], axis=0)
        G = ph["U"][3]
        bad = np.where(T < (1 - tol) * G)[0]
        if len(bad):
            flags[name] = bad.tolist()
    return flags

"""Configured engine envelopes and retained operating-life checks."""

import numpy as np
from hopper.postprocess import phase_mode
from .config import descent_variant


def hardware_checks(p, sol):
    results = {}
    bands = True
    settings = descent_variant()

    def floor_fraction(kind, count, cls):
        """A2200 minimum throttle fraction: 0.5, except in the descent_variant floor phases. There a thrust floor of
        at least one engine's rated thrust is read as engines shut off with the rest at 50 % (assumption stated by
        the user), so the allowed fraction of the lit `count` engines is 1 / count."""
        floor_n = settings["thrust_floor_n"]
        if floor_n is not None and kind in settings["floor_phases"] and floor_n >= cls.thrust - 1e-6:
            return min(0.5, 1.0 / count)
        return 0.5

    for name, cls in p.engine.classes.items():
        units = [
            dict(engine=i + 1, firing_s=0.0, longest_burn_s=0.0, starts=0) for i in range(cls.count)
        ]
        fractions = []
        floors = []     # lowest allowed throttle fraction of each sample (0.5 unless the two-engine reading applies)
        slew = 0.0
        join = 0.0
        for hop in (1, 2):
            for kinds in (("rise", "ascent"), ("braking", "observation", "terminal")):
                times = np.zeros(cls.count)
                previous = None
                for ph in sol.phases.values():
                    if ph["hop"] != hop or ph["kind"] not in kinds:
                        continue
                    mode = phase_mode(p, ph)
                    gamma = ph["U"][3]
                    bands &= bool(
                        gamma.min() >= mode.gamma_lo - 1e-3 and gamma.max() <= mode.gamma_hi + 1e-3
                    )
                    count = dict(mode.fixed + mode.throttled).get(
                        name, cls.count if cls.pulsing and name in p.engine.pools[ph["kind"]] else 0
                    )
                    times[:count] += ph["t"]
                    if count and name in dict(mode.throttled):
                        frac = (gamma - mode.thrust_fixed) / (count * cls.thrust)
                        fractions.extend(frac.tolist())
                        floors.extend([floor_fraction(ph["kind"], count, cls)] * len(frac))
                        slew = max(
                            slew, float(np.abs(np.diff(frac) / (ph["t"] / (len(frac) - 1))).max())
                        )
                        if previous is not None:
                            join = max(join, abs(float(frac[0]) - previous))
                        previous = float(frac[-1])
                for unit, t in zip(units, times):
                    unit["firing_s"] += float(t)
                    unit["longest_burn_s"] = max(unit["longest_burn_s"], float(t))
                    unit["starts"] += int(t > 0)
        checks = []
        if cls.max_continuous_burn_s:
            checks.append(
                max(u["longest_burn_s"] for u in units) <= cls.max_continuous_burn_s + 1e-6
            )
        if name == "a2200":
            checks += [
                all(f >= floor - 1e-6 for f, floor in zip(fractions, floors)),
                max(fractions) <= 1 + 1e-6,
                slew <= 1 / 6 + 1e-6,
                join < 1e-5,
                all(
                    u["firing_s"] <= 1000 and u["longest_burn_s"] <= 500 and u["starts"] <= 30
                    for u in units
                ),
            ]
        if name == "large":
            checks += [
                all(
                    u["starts"] <= 2000
                    and u["firing_s"] * cls.thrust <= 465000.0 * 4.4482216152605 + 1e-3
                    for u in units
                )
            ]
        if name == "small":
            checks += [
                all(
                    u["firing_s"] <= 3600 and u["firing_s"] * cls.thrust <= 13600000.0
                    for u in units
                )
            ]
        results[name] = dict(
            passed=bool(all(checks)),
            per_engine=units,
            throttle_range=[min(fractions), max(fractions)] if fractions else None,
            maximum_throttle_slew_fraction_s=slew,
            maximum_throttle_join_error=join,
            pulse_ignition_count=None if cls.pulsing else "reported as burn starts",
        )
    return dict(
        passed=bool(bands and all(row["passed"] for row in results.values())),
        mode_bands_passed=bands,
        classes=results,
    )

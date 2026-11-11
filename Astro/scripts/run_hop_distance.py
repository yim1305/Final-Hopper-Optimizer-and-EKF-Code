"""Hop-distance demonstration: the selected 8-A2200 design flown to 50, 100 and 150 km destinations.

Same launch base, dry mass, engines, divert, landing/sensing limits and DEM as a2200_08; only the destination
changes. Destinations lie on the great circle from the launch base toward the Drygalski target, so every flight
stays inside the 30 m DEM corridor. Each case is seeded from the built-in Lambert-based guess on the coarse seed mesh (DEM
terrain), then solved on the full DEM mesh (GroundProblem) and passed through the same acceptance checks as the
main cases (independent propagation, camera, hardware, fixed-mass reproduction, star trackers).

Usage (repository root):  .\\.venv\\Scripts\\python.exe -B Astro/scripts/run_hop_distance.py [d050 d100 d150]
                           .\\.venv\\Scripts\\python.exe -B Astro/scripts/run_hop_distance.py --report-only
Results:                   Astro/results/hop_distance/<case>/ and Astro/results/hop_distance/REPORT.md
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from dataclasses import asdict, replace
from pathlib import Path
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hopper.problem import Problem
from hopper.lambert import min_dv_hop
from hopper.postprocess import round_trip_summary, phase_table, thrust_discard_flags, phase_mode
from hopper.tracker_cache import save_trajectory
from mission.config import parameters, write, physical_identity
from mission.run import numeric_settings, fixed_mass
from mission.terrain import DemTerrain, GroundProblem
from mission.validation import integrate, camera_check
from mission.hardware import hardware_checks
from mission.trackers import tracker_checks

LAUNCH = (-85.5, -87.1)
DRYGALSKI = (-78.82668404226864, -85.95852077297914)
CASES = {"d050": 50e3, "d100": 100e3, "d150": 150e3}
OUT = ROOT / "results" / "hop_distance"


def destination(distance, radius):
    """Point `distance` metres along the launch -> Drygalski great circle (reference sphere)."""
    def unit(lat, lon):
        lat, lon = np.radians([lat, lon])
        return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    a, b = unit(*LAUNCH), unit(*DRYGALSKI)
    th = np.arccos(np.clip(a @ b, -1, 1))
    s = distance / radius
    u = (np.sin(th - s) * a + np.sin(s) * b) / np.sin(th)
    return float(np.degrees(np.arcsin(u[2]))), float(np.degrees(np.arctan2(u[1], u[0])))


def solve(case, distance, terrain):
    out = OUT / case
    out.mkdir(parents=True, exist_ok=True)
    lat, lon = destination(distance, terrain.radius)
    p = numeric_settings(parameters("a2200_08", terrain, {"sites": {"target": {"lat": lat, "lon": lon}}}))
    log = dict(case=case, distance_km=distance / 1000, target_lat=lat, target_lon=lon, stages=[])
    print(case, f"target {lat:.6f}, {lon:.6f}", flush=True)

    # 1) Seed: the built-in Lambert-based initial guess (hopper/initial_guess.py) solved on the coarse seed mesh
    #    with the DEM terrain, then on the full mesh. (The relax/stage pipeline in staging.py only applies to
    #    fixed-thrust engine pools, not the throttled A2200.)
    tick = time.monotonic()
    guess, m0 = None, None
    for label, mesh in (("seed mesh", p.solver.seed_nodes), ("full mesh", p.solver.nodes)):
        t0 = time.monotonic()
        pr = Problem(p, "uprange", nodes=dict(mesh), terrain=terrain)
        pr.set_guess(guess, m_wet0=m0)
        seed = pr.solve(0)
        log["stages"].append(dict(stage=label, status=seed.status, success=seed.success,
                                  iterations=seed.iterations, wall_s=time.monotonic() - t0, m_wet=seed.m_wet,
                                  primal_infeasibility=seed.primal_infeasibility,
                                  dual_infeasibility=seed.dual_infeasibility))
        print(case, label, seed.status, seed.iterations, f"{seed.m_wet:.2f} kg", flush=True)
        # Seeds only need a converged IPOPT status; the strict residual test applies to the final DEM solve.
        if seed.status not in ("Solve_Succeeded", "Solved_To_Acceptable_Level"):
            write(out / "attempt.json", log)
            raise RuntimeError(case + ": " + label + " failed " + seed.status)
        guess, m0 = Problem.guess_from_solution(seed), seed.m_wet
        del pr

    # 2) Full DEM problem (midpoint clearance, domain, camera terrain rays) from the seed.
    problem = GroundProblem(p, "uprange", terrain=terrain)
    problem.set_guess(Problem.guess_from_solution(seed), m_wet0=seed.m_wet)
    t0 = time.monotonic()
    sol = problem.solve(0)
    duals = np.asarray(problem.opti.value(problem.opti.lam_g) if sol.success
                       else problem.opti.debug.value(problem.opti.lam_g)).ravel()
    log["stages"].append(dict(stage="DEM full mesh", status=sol.status, success=sol.success,
                              iterations=sol.iterations, wall_s=time.monotonic() - t0, m_wet=sol.m_wet,
                              primal_infeasibility=sol.primal_infeasibility,
                              dual_infeasibility=sol.dual_infeasibility))
    print(case, "DEM solve", sol.status, sol.iterations, sol.m_wet, flush=True)
    if not sol.success and sol.status in ("Solve_Succeeded", "Solved_To_Acceptable_Level"):
        # Same polish step as mission/run.py: warm-start from the converged point and multipliers.
        problem.set_guess(problem.guess_from_solution(sol), m_wet0=sol.m_wet)
        problem.opti.set_initial(problem.opti.lam_g, duals)
        problem.p = replace(p, solver=replace(p.solver, ipopt={**p.solver.ipopt, "warm_start_init_point": "yes",
                                                               "mu_init": 1e-10}))
        t0 = time.monotonic()
        sol = problem.solve(0)
        duals = np.asarray(problem.opti.value(problem.opti.lam_g) if sol.success
                           else problem.opti.debug.value(problem.opti.lam_g)).ravel()
        log["stages"].append(dict(stage="DEM warm-start polish", status=sol.status, success=sol.success,
                                  iterations=sol.iterations, wall_s=time.monotonic() - t0, m_wet=sol.m_wet,
                                  primal_infeasibility=sol.primal_infeasibility,
                                  dual_infeasibility=sol.dual_infeasibility))
        print(case, "DEM polish", sol.status, sol.iterations, sol.m_wet, flush=True)
    save_trajectory(out / "sizing.npz", p, sol, "uprange", physical_identity(p),
                    dict(passed=False, diagnostic_only=True))
    np.savez_compressed(out / "duals.npz", duals=duals)
    write(out / "parameters.json", asdict(p))
    sizing = dict(success=bool(sol.success), status=sol.status, wet_mass_kg=sol.m_wet,
                  iterations=sol.iterations, primal=sol.primal_infeasibility, dual=sol.dual_infeasibility,
                  variables=int(problem.opti.nx), constraints=int(problem.opti.ng),
                  total_wall_s=time.monotonic() - tick)
    write(out / "sizing.json", sizing)
    write(out / "attempt.json", log)
    if not sol.success:
        raise RuntimeError(case + ": DEM sizing failed " + sol.status)

    # 3) Same acceptance checks as the main cases.
    rows, history, contacts = integrate(p, sol, terrain)
    camera = camera_check(p, sol, terrain, history)
    hardware = hardware_checks(p, sol)
    checks = dict(passed=bool(all(r["passed"] for r in rows) and all(c["passed"] for c in contacts.values())
                              and all(c["passed"] for c in camera.values()) and hardware["passed"]
                              and not thrust_discard_flags(p, sol)),
                  propagation=rows, contacts=contacts, camera=camera, hardware=hardware)
    write(out / "validation.json", checks)
    if checks["passed"]:
        checks["fixed_mass"] = fm = fixed_mass(p, terrain, sol, problem, duals)
        checks["passed"] = bool(fm["passed"])
        write(out / "validation.json", checks)
    del problem
    trackers = tracker_checks(p, sol) if checks["passed"] else dict(passed=False, reports={})
    write(out / "trackers.json", trackers)
    checks["trackers_passed"] = bool(trackers["passed"])
    checks["passed"] = bool(checks["passed"] and trackers["passed"])
    write(out / "validation.json", checks)

    lam = (min_dv_hop(p).dv, min_dv_hop(p, r1=p.mission.sites["uprange"], r2=p.mission.r_start).dv)
    summary = round_trip_summary(p, sol, "uprange", *lam)
    table = phase_table(p, sol, "uprange")
    for row, ph in zip(table, sol.phases.values()):
        mode = phase_mode(p, ph)
        row["engine_counts"] = dict(mode.fixed + mode.throttled) if mode else {}
    great_circle = float(np.arccos(np.clip(
        p.mission.r_start @ p.mission.r_target / np.linalg.norm(p.mission.r_start) / np.linalg.norm(p.mission.r_target),
        -1, 1)) * terrain.radius)
    result = dict(status="numerically_validated_dem_flight" if checks["passed"] else "validation_failed",
                  case=case, distance_km=distance / 1000, great_circle_km=great_circle / 1000,
                  target_lat=lat, target_lon=lon,
                  target_dem_elevation_m=float(np.linalg.norm(p.mission.r_target) - terrain.radius),
                  summary=summary, table=table, contacts=contacts,
                  tracker_summary={k: {x: v[x] for x in ("passed", "event_count", "max_update_gap_s")}
                                   for k, v in trackers.get("reports", {}).items() if "max_update_gap_s" in v},
                  scope="Hop-distance demonstration of the 8-A2200 design; same inputs as a2200_08 except destination")
    write(out / "summary.json", result)
    save_trajectory(out / "trajectory.npz", p, sol, "uprange", physical_identity(p),
                    dict(passed=bool(checks["passed"])))
    np.savez_compressed(out / "independent_flight.npz",
                        time=np.concatenate([h["time"] for h in history]),
                        r=np.concatenate([h["r"] for h in history]),
                        state=np.concatenate([h["state"] for h in history]),
                        hop=np.concatenate([np.full(len(h["time"]), h["hop"]) for h in history]))
    print(case, "PASSED" if checks["passed"] else "FAILED", flush=True)
    return result


def report():
    main = json.loads((ROOT / "results/a2200_08/summary.json").read_text())
    rows = []
    for case in CASES:
        path = OUT / case / "summary.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
    rows.append(dict(case="a2200_08 (Drygalski)", status=main["status"], summary=main["summary"],
                     great_circle_km=202.402))
    lines = ["# Hop-distance demonstration — 8 A2200", "",
             "Same launch base, 2,000 kg dry mass, 8 A2200, 150 m uprange divert, 25 m landing region, "
             "1.5/0.5 m/s touchdown limits and 30 m DEM as `a2200_08`; only the destination changes. "
             "Destinations lie on the launch → Drygalski great circle. Each case passed the same acceptance checks "
             "(independent propagation, camera, hardware, fixed-mass reproduction, star trackers) unless marked.", "",
             "| Case | Distance (km) | Status | Wet mass (kg) | Propellant used (kg) | Finite-burn Δv (m/s) | "
             "Lambert Δv out/back (m/s) | Flight time (s) |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        s = r["summary"]
        lines.append(f"| {r['case']} | {r['great_circle_km']:.1f} | {r['status']} | {s['total']['m_start']:.1f} | "
                     f"{s['total']['propellant_used']:.1f} | {s['total']['finite_burn_dv']:.1f} | "
                     f"{s['hop1']['lambert_dv']:.1f} / {s['hop2']['lambert_dv']:.1f} | "
                     f"{s['hop1']['duration'] + s['hop2']['duration']:.1f} |")
    lines += ["", "![Hop-distance comparison](comparison.png)", "", "Nominal local optimization results; same scope and limitations as the main cases. "
              "Script: `Astro/scripts/run_hop_distance.py`."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot():
    """One summary figure: height profiles over downrange distance, wet mass and delta-v versus hop distance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mission.terrain import DemTerrain
    terrain = DemTerrain()
    cases = [(OUT / c, f"{int(d / 1000)} km") for c, d in CASES.items()] + [(ROOT / "results/a2200_08", "202 km (Drygalski)")]
    colors = ["#9DB4E8", "#5B82D6", "#2F5FB8", "#0046AC"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    dist, wet, dv, lam = [], [], [], []
    for (folder, label), color in zip(cases, colors):
        s = json.loads((folder / "summary.json").read_text())
        with np.load(folder / "independent_flight.npz") as f:
            r, hop = f["r"], f["hop"]
        start = r[0] / np.linalg.norm(r[0])
        for h, ls in ((1, "-"), (2, "--")):
            rr = r[hop == h]
            u = rr / np.linalg.norm(rr, axis=1)[:, None]
            x = np.arccos(np.clip(u @ start, -1, 1)) * terrain.radius / 1000
            ax[0].plot(x, terrain.altitude(rr) / 1000, ls, color=color, lw=1.8, label=label if h == 1 else None)
        S = s["summary"]
        dist.append(s.get("great_circle_km", 202.402)); wet.append(S["total"]["m_start"])
        dv.append(S["total"]["finite_burn_dv"]); lam.append(S["hop1"]["lambert_dv"] + S["hop2"]["lambert_dv"])
    ax[0].set(xlabel="Surface distance from launch (km)", ylabel="Height above DEM (km)",
              title="Reference flights (solid outbound, dashed return)")
    ax[0].legend(frameon=False)
    ax[1].plot(dist, wet, "o-", color="#0046AC")
    ax[1].set(xlabel="Hop distance (km)", ylabel="Wet mass (kg)", title="Minimum wet mass", ylim=(0, None))
    ax[2].plot(dist, dv, "o-", color="#0046AC", label="Finite burn (optimizer)")
    ax[2].plot(dist, lam, "s--", color="#888888", label="Lambert minimum")
    ax[2].set(xlabel="Hop distance (km)", ylabel="Round-trip Δv (m/s)", title="Round-trip Δv", ylim=(0, None))
    ax[2].legend(frameon=False)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=165, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        report()
        plot()
        sys.exit()
    terrain = DemTerrain()
    for case in (sys.argv[1:] or CASES):
        solve(case, CASES[case], terrain)
    report()
    plot()

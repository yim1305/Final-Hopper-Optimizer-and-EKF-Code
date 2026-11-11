"""Finish a hop-distance case from its saved DEM sizing when only the fixed-mass re-solve missed the 1e-6 residual gate.

The fixed-mass NLP is re-run exactly as mission/run.py fixed_mass, then (if IPOPT reported Solve_Succeeded but the unscaled
residual is just above 1e-6) polished once from its own converged point with warm-start multipliers and mu_init 1e-10,
the same polish run.py applies to sizing. Then the tracker check and summary are written exactly as run_hop_distance.solve.
Usage (repository root): .\\.venv\\Scripts\\python.exe -B Astro/scripts/finalize_hop_case.py d050 [d100 d150]
Needs results/hop_distance/<case>/ written earlier by run_hop_distance.py.
"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
from dataclasses import replace
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import run_hop_distance as H
from mission.config import parameters, read_solution, write, physical_identity
from mission.run import numeric_settings
from mission.terrain import DemTerrain, GroundProblem
from mission.validation import integrate
from mission.trackers import tracker_checks
from hopper.lambert import min_dv_hop
from hopper.postprocess import round_trip_summary, phase_table, phase_mode
from hopper.tracker_cache import save_trajectory


def fixed_mass_polished(p, terrain, sol, sizing_problem, duals):
    warm = replace(p, solver=replace(p.solver, ipopt={**p.solver.ipopt, "warm_start_init_point": "yes",
        "warm_start_bound_push": 1e-10, "warm_start_mult_bound_push": 1e-10, "warm_start_slack_bound_push": 1e-10,
        "mu_init": 1e-10, "max_wall_time": 1500.0}))
    reserve_index = sizing_problem.original_constraint_count - 1
    multiplier = abs(duals[reserve_index])
    problem = GroundProblem(warm, "uprange", terrain=terrain, mode="maximise_margin", fixed_m_wet=sol.m_wet)
    problem.set_guess(problem.guess_from_solution(sol))
    if multiplier > 1e-10:
        mapped = np.delete(duals, [sizing_problem.initial_boundary_offset, reserve_index]) / multiplier
        problem.opti.set_initial(problem.opti.lam_g, mapped)
    check = problem.solve(0)
    polished = False
    if not check.success and check.status in ("Solve_Succeeded", "Solved_To_Acceptable_Level"):
        lam = np.asarray(problem.opti.debug.value(problem.opti.lam_g)).ravel()
        problem.set_guess(problem.guess_from_solution(check))
        problem.opti.set_initial(problem.opti.lam_g, lam)
        check = problem.solve(0); polished = True
    ratio = (sol.m_wet - p.vehicle.m_dry) / (sol.m_wet - check.z)
    return dict(passed=bool(check.success and abs(ratio / p.mission.reserve_factor - 1) < 0.001 and abs(check.z - sol.z) < 0.05),
                status=check.status, iterations=check.iterations, primal=check.primal_infeasibility, dual=check.dual_infeasibility,
                loaded_over_used=ratio, final_mass_difference_kg=check.z - sol.z, polished=polished)


terrain = DemTerrain()
for case in sys.argv[1:]:
    out = H.OUT / case
    lat, lon = H.destination(H.CASES[case], terrain.radius)
    p = numeric_settings(parameters("a2200_08", terrain, {"sites": {"target": {"lat": lat, "lon": lon}}}))
    sol = read_solution(out / "sizing.npz"); duals = np.load(out / "duals.npz")["duals"]
    checks = json.loads((out / "validation.json").read_text())
    flight_ok = all(r["passed"] for r in checks["propagation"]) and all(c["passed"] for c in checks["contacts"].values()) \
        and all(c["passed"] for c in checks["camera"].values()) and checks["hardware"]["passed"]
    if not flight_ok:
        print(case, "flight checks failed; not finalized"); continue
    sizing_problem = GroundProblem(p, "uprange", terrain=terrain)
    checks["fixed_mass"] = fm = fixed_mass_polished(p, terrain, sol, sizing_problem, duals)
    print(case, "fixed mass", fm, flush=True)
    del sizing_problem
    trackers = tracker_checks(p, sol) if fm["passed"] else dict(passed=False, reports={})
    write(out / "trackers.json", trackers)
    checks["trackers_passed"] = bool(trackers["passed"])
    checks["passed"] = bool(fm["passed"] and trackers["passed"])
    write(out / "validation.json", checks)
    lam = (min_dv_hop(p).dv, min_dv_hop(p, r1=p.mission.sites["uprange"], r2=p.mission.r_start).dv)
    summary = round_trip_summary(p, sol, "uprange", *lam)
    table = phase_table(p, sol, "uprange")
    for row, ph in zip(table, sol.phases.values()):
        mode = phase_mode(p, ph); row["engine_counts"] = dict(mode.fixed + mode.throttled) if mode else {}
    great_circle = float(np.arccos(np.clip(p.mission.r_start @ p.mission.r_target / np.linalg.norm(p.mission.r_start) /
                                           np.linalg.norm(p.mission.r_target), -1, 1)) * terrain.radius)
    rows, history, contacts = integrate(p, sol, terrain)
    write(out / "summary.json", dict(status="numerically_validated_dem_flight" if checks["passed"] else "validation_failed",
        case=case, distance_km=H.CASES[case] / 1000, great_circle_km=great_circle / 1000, target_lat=lat, target_lon=lon,
        target_dem_elevation_m=float(np.linalg.norm(p.mission.r_target) - terrain.radius), summary=summary, table=table,
        contacts=contacts, tracker_summary={k: {x: v[x] for x in ("passed", "event_count", "max_update_gap_s")}
                                            for k, v in trackers.get("reports", {}).items() if "max_update_gap_s" in v},
        scope="Hop-distance demonstration of the 8-A2200 design; same inputs as a2200_08 except destination"))
    save_trajectory(out / "trajectory.npz", p, sol, "uprange", physical_identity(p), dict(passed=bool(checks["passed"])))
    np.savez_compressed(out / "independent_flight.npz", time=np.concatenate([h["time"] for h in history]),
                        r=np.concatenate([h["r"] for h in history]), state=np.concatenate([h["state"] for h in history]),
                        hop=np.concatenate([np.full(len(h["time"]), h["hop"]) for h in history]))
    print(case, "PASSED" if checks["passed"] else "FAILED", flush=True)
H.report(); H.plot()

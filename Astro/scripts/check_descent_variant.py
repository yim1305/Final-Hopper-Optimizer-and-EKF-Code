"""Full acceptance checks (same as run_case) on a saved descent-variant run; nothing is written outside its folder.

    .\\.venv\\Scripts\\python.exe -B Astro/scripts/check_descent_variant.py dv_final --floor 2224 --tw 1.0 --obs 5 --vcap 5

Reads results/TAG/sizing.npz and duals.npz written by run_descent_variant.py, then runs the independent flight
integration, camera/hardware/discard checks, the fixed-mass reserve reproduction and the terrain-aware tracker
schedule. Writes results/TAG/acceptance.json with every check, pass or fail.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
from pathlib import Path
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hopper.postprocess import thrust_discard_flags
from mission.config import CONFIG, parameters, read_solution, write
from mission.hardware import hardware_checks
from mission.run import fixed_mass, numeric_settings
from mission.terrain import DemTerrain, GroundProblem
from mission.trackers import tracker_checks
from mission.validation import camera_check, integrate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_hop_distance import destination

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tag")
    ap.add_argument("--case", default="a2200_08", choices=tuple(CONFIG["cases"]), help="mission.yaml case")
    ap.add_argument("--distance-km", type=float, help="hop-distance study target (see run_descent_variant.py)")
    ap.add_argument("--floor", type=float)
    ap.add_argument("--tw", type=float)
    ap.add_argument("--obs", type=float)
    ap.add_argument("--vcap", type=float)
    ap.add_argument("--obs-nodes", type=int)
    ap.add_argument("--solve-pointing-rate", type=float,
                    help="margin used when the run was solved; only the fixed-mass re-solve uses it, "
                         "every other check uses the real limits")
    a = ap.parse_args()
    CASE = a.case
    out = ROOT / "results" / a.tag
    terrain = DemTerrain()
    sensor = {}
    if a.obs is not None:
        sensor["observation_time_min"] = a.obs
    if a.vcap is not None:
        sensor["descent_speed_bounds"] = [0.1, a.vcap]
    updates = {"sensor": sensor} if sensor else {}
    if a.distance_km is not None:
        lat, lon = destination(a.distance_km * 1e3, terrain.radius)
        updates["sites"] = {"target": {"lat": lat, "lon": lon}}
    if a.obs_nodes is not None:
        updates["solver"] = {"nodes": {"observation": a.obs_nodes}}
    p = numeric_settings(parameters(CASE, terrain, updates or None, {"thrust_floor_n": a.floor}))
    if a.solve_pointing_rate is not None:
        solve_updates = {**updates, "sensor": {**sensor, "pointing_rate_max_deg_s": a.solve_pointing_rate}}
        p_solve = numeric_settings(parameters(CASE, terrain, solve_updates, {"thrust_floor_n": a.floor}))
    else:
        p_solve = p
    sol = read_solution(out / "sizing.npz")
    duals = np.load(out / "duals.npz")["duals"]
    report = dict(tag=a.tag, floor_n=a.floor, tw_max=a.tw, sizing_status=sol.status, m_wet_kg=sol.m_wet)

    def step(name, fn):
        try:
            report[name] = fn()
        except Exception as exc:  # record the failure and keep checking the rest
            report[name] = dict(passed=False, error=repr(exc), trace=traceback.format_exc(limit=3))
        print(name, "passed" if report[name].get("passed") else "FAILED", flush=True)
        write(out / "acceptance.json", report)

    state = {}

    def flight():
        rows, history, contacts = integrate(p, sol, terrain)
        state["history"] = history
        return dict(passed=bool(all(r["passed"] for r in rows) and all(r["passed"] for r in contacts.values())),
                    propagation=rows, contacts=contacts)

    step("flight", flight)
    step("camera", lambda: (lambda c: dict(passed=bool(all(r["passed"] for r in c.values())), reports=c))(
        camera_check(p, sol, terrain, state["history"])))
    step("hardware", lambda: hardware_checks(p, sol))
    step("thrust_discard", lambda: (lambda f: dict(passed=not f, flags=f))(thrust_discard_flags(p, sol)))

    def reserve():
        problem = GroundProblem(p_solve, "uprange", terrain=terrain, tw_max=a.tw)
        return fixed_mass(p_solve, terrain, sol, problem, duals)

    step("fixed_mass", reserve)
    step("trackers", lambda: tracker_checks(p, sol))
    report["passed"] = all(report[k].get("passed") for k in
                           ("flight", "camera", "hardware", "thrust_discard", "fixed_mass", "trackers"))
    write(out / "acceptance.json", report)
    print("ALL CHECKS PASSED" if report["passed"] else "ACCEPTANCE FAILED: see acceptance.json", flush=True)


if __name__ == "__main__":
    main()

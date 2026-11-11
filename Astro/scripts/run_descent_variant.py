"""Solve the a2200_08 round trip with a lower descent thrust floor and a touchdown thrust-to-weight limit.

Diagnostic optimizer run, seeded from a saved sizing.npz; it applies no acceptance checks and never touches the
validated a2200_08 folder. Output: Astro/results/<tag>/ (sizing.npz, duals.npz, diag.json).

Run from the repository root. The seed folder must hold a sizing.npz; the stored restart arrays give
--seed a2200_08 and --seed a2200_12 (a2200_08 already holds the final descent-variant solution):

    .\\.venv\\Scripts\\python.exe -B Astro/scripts/run_descent_variant.py TAG --seed a2200_08 \\
        --floor 2224 --tw 1.0 --obs 5 --vcap 5
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hopper.problem import Problem
from hopper.tracker_cache import save_trajectory
from mission.config import CONFIG, parameters, physical_identity, read_solution, write
from mission.run import numeric_settings
from mission.terrain import DemTerrain, GroundProblem

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_hop_distance import destination

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tag", help="output folder under Astro/results/ (may contain a subfolder, e.g. ladder_x/step1)")
    ap.add_argument("--case", default="a2200_08", choices=tuple(CONFIG["cases"]), help="mission.yaml case")
    ap.add_argument(
        "--distance-km",
        type=float,
        help="hop-distance study: move the target to this distance along the launch -> Drygalski great circle "
        "(same destination rule as run_hop_distance.py)",
    )
    ap.add_argument(
        "--yaml-off",
        action="store_true",
        help="for this process only, ignore mission.yaml descent_variant touchdown T/W limit and solve-side pointing "
        "margin (the file is not changed); the floor comes from --floor only",
    )
    ap.add_argument("--seed", required=True, help="Astro/results/ folder holding the warm-start sizing.npz")
    ap.add_argument("--floor", type=float, help="thrust floor [N] for observation and terminal, both hops")
    ap.add_argument("--tw", type=float, help="max thrust / weight at each hop's touchdown node")
    ap.add_argument("--obs", type=float, help="sensor.observation_time_min [s]")
    ap.add_argument("--vcap", type=float, help="observation vertical descent-speed cap [m/s]")
    ap.add_argument("--obs-nodes", type=int, help="solver.nodes.observation (mission.yaml: 36)")
    ap.add_argument(
        "--pointing-rate",
        type=float,
        help="solve-side observation pointing-rate cap [deg/s], a numerical margin below the 6 deg/s requirement; "
        "check_descent_variant.py still checks the real limit",
    )
    ap.add_argument(
        "--margin",
        type=float,
        help="solve-side pointing-rate margin [deg/s] for this process only (overrides mission.yaml, even with --yaml-off)",
    )
    ap.add_argument("--wall", type=float, default=600.0, help="IPOPT wall-time cap [s]")
    ap.add_argument("--print-level", type=int, default=0, help="IPOPT print level (5 shows every iteration)")
    a = ap.parse_args()
    CASE = a.case
    if a.yaml_off:
        CONFIG["descent_variant"]["touchdown_thrust_to_weight_max"] = None
        CONFIG["descent_variant"]["solve_pointing_rate_deg_s"] = None
    if a.margin is not None:
        CONFIG["descent_variant"]["solve_pointing_rate_deg_s"] = a.margin

    terrain = DemTerrain()
    sensor = {}
    if a.obs is not None:
        sensor["observation_time_min"] = a.obs
    if a.vcap is not None:
        sensor["descent_speed_bounds"] = [0.1, a.vcap]
    if a.pointing_rate is not None:
        sensor["pointing_rate_max_deg_s"] = a.pointing_rate
    updates = {"sensor": sensor} if sensor else {}
    if a.distance_km is not None:
        lat, lon = destination(a.distance_km * 1e3, terrain.radius)
        updates["sites"] = {"target": {"lat": lat, "lon": lon}}
    if a.obs_nodes is not None:
        updates["solver"] = {"nodes": {"observation": a.obs_nodes}}
    p = numeric_settings(parameters(CASE, terrain, updates or None, {"thrust_floor_n": a.floor}))
    p = replace(p, solver=replace(p.solver, ipopt={**p.solver.ipopt, "max_wall_time": a.wall}))
    seed = read_solution(ROOT / "results" / a.seed / "sizing.npz")
    problem = GroundProblem(p, "uprange", terrain=terrain, tw_max=a.tw)
    problem.set_guess(Problem.guess_from_solution(seed), m_wet0=seed.m_wet)
    print(f"{a.tag}: floor={a.floor} tw={a.tw} obs={a.obs} vcap={a.vcap} seed={a.seed}", flush=True)
    tick = time.monotonic()
    sol = problem.solve(a.print_level)
    if not sol.success and sol.status in ("Solve_Succeeded", "Solved_To_Acceptable_Level"):
        # Same warm-start polish as mission/run.py: IPOPT converged but missed the strict residual test.
        print(f"{a.tag}: {sol.status} primal={sol.primal_infeasibility:.2e}, polishing", flush=True)
        duals = problem.opti.debug.value(problem.opti.lam_g)
        problem.set_guess(problem.guess_from_solution(sol), m_wet0=sol.m_wet)
        problem.opti.set_initial(problem.opti.lam_g, duals)
        problem.p = replace(
            p, solver=replace(p.solver, ipopt={**p.solver.ipopt, "warm_start_init_point": "yes", "mu_init": 1e-10})
        )
        sol = problem.solve(a.print_level)
    wall = time.monotonic() - tick
    print(
        f"{a.tag}: {sol.status} success={sol.success} iters={sol.iterations} wall={wall:.0f}s "
        f"m_wet={sol.m_wet:.2f} primal={sol.primal_infeasibility:.3e} dual={sol.dual_infeasibility:.3e}",
        flush=True,
    )
    mu, rows = p.moon.mu, []
    for name, ph in sol.phases.items():
        X, U = ph["X"], ph["U"]
        r = p.mission.r_start + X[0:3, -1]
        rows.append(
            dict(
                phase=name,
                duration_s=ph["t"],
                thrust_min_n=float(U[3].min()),
                thrust_max_n=float(U[3].max()),
                mass_start_kg=float(X[6, 0]),
                speed_start_m_s=float(np.linalg.norm(X[3:6, 0])),
                speed_end_m_s=float(np.linalg.norm(X[3:6, -1])),
                end_thrust_to_weight=float(U[3, -1] / (X[6, -1] * mu / float(r @ r))),
            )
        )
        if ph["kind"] in ("observation", "terminal"):
            print(
                f"  {name}: {ph['t']:.1f} s  thrust {rows[-1]['thrust_min_n']:.0f}..{rows[-1]['thrust_max_n']:.0f} N  "
                f"end T/W {rows[-1]['end_thrust_to_weight']:.2f}  end speed {rows[-1]['speed_end_m_s']:.2f} m/s",
                flush=True,
            )
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    write(
        out / "diag.json",
        dict(
            tag=a.tag, seed=a.seed, floor_n=a.floor, tw_max=a.tw, obs=a.obs, vcap=a.vcap, status=sol.status,
            success=bool(sol.success), iterations=sol.iterations, wall_s=wall, m_wet_kg=sol.m_wet,
            primal=sol.primal_infeasibility, dual=sol.dual_infeasibility, table=rows,
        ),
    )
    save_trajectory(
        out / "sizing.npz", p, sol, "uprange", physical_identity(p), dict(passed=False, diagnostic_only=True)
    )
    duals = problem.opti.value(problem.opti.lam_g) if sol.success else problem.opti.debug.value(problem.opti.lam_g)
    np.savez_compressed(out / "duals.npz", duals=np.asarray(duals).ravel())


if __name__ == "__main__":
    main()

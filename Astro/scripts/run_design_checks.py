r"""Design-verification variants of the selected 8-A2200 DEM flight.

Variants (each re-optimized on the DEM GroundProblem, seeded from results/a2200_08, then put through the same
independent propagation, camera, hardware and fixed-mass acceptance checks as the main cases):

  dry1800, dry2200 : dry mass -10 % / +10 % (sensitivity of wet mass to dry-mass growth)
  mesh_other2x     : every phase except ascent doubled (ascent is refined separately in the ascent-mesh series)

Usage (repository root):  .\.venv\Scripts\python.exe -B Astro/scripts/run_design_checks.py [dry1800 dry2200 mesh_other2x]
                           .\.venv\Scripts\python.exe -B Astro/scripts/run_design_checks.py --report-only
Results:                   Astro/results/design_checks/<variant>/ and Astro/results/design_checks/REPORT.md
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
from scipy.interpolate import CubicHermiteSpline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hopper.dynamics import make_numpy_rhs
from hopper.problem import Problem
from hopper.postprocess import round_trip_summary, thrust_discard_flags
from hopper.lambert import min_dv_hop
from hopper.tracker_cache import save_trajectory
from mission.config import CONFIG, parameters, read_solution, write, physical_identity
from mission.run import numeric_settings, fixed_mass
from mission.terrain import DemTerrain, GroundProblem
from mission.validation import integrate, camera_check
from mission.hardware import hardware_checks

OUT = ROOT / "results" / "design_checks"
BASE = ROOT / "results" / "a2200_08"
NODES = CONFIG["common"]["solver"]["nodes"]
VARIANTS = {
    "dry1800": dict(label="Dry mass 1,800 kg (−10 %)", updates={"vehicle": {"m_dry": 1800.0}}),
    "dry2200": dict(label="Dry mass 2,200 kg (+10 %)", updates={"vehicle": {"m_dry": 2200.0}}),
    "mesh_other2x": dict(label="Mesh doubled in all phases except ascent",
                         updates={"solver": {"nodes": {k: (v if k == "ascent" else 2 * v) for k, v in NODES.items()}}}),
}


def resampled_guess(p, problem, seed):
    """Seed guess on the variant mesh: cubic Hermite states (same as collocation), linear controls."""
    guess = Problem.guess_from_solution(seed)
    for phase in problem.phases:
        src = guess[phase.name]
        if src["X"].shape[1] == phase.n + 1:
            continue
        t_old = np.linspace(0.0, src["t"], src["X"].shape[1])
        t_new = np.linspace(0.0, src["t"], phase.n + 1)
        rhs = make_numpy_rhs(p, mode=phase.mode)
        slopes = np.array([rhs(x, u) for x, u in zip(src["X"].T, src["U"].T)])
        src["X"] = CubicHermiteSpline(t_old, src["X"].T, slopes)(t_new).T
        src["U"] = np.array([np.interp(t_new, t_old, row) for row in src["U"]])
    return guess


def solve(name, terrain):
    spec = VARIANTS[name]
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    p = numeric_settings(parameters("a2200_08", terrain, spec["updates"]))
    # larger meshes need more than the 120 s default wall-time cap
    p = replace(p, solver=replace(p.solver, ipopt={**p.solver.ipopt, "max_wall_time": 1500.0}))
    seed = read_solution(BASE / "sizing.npz")
    tick = time.monotonic()
    problem = GroundProblem(p, "uprange", terrain=terrain)
    problem.set_guess(resampled_guess(p, problem, seed), m_wet0=seed.m_wet)
    sol = problem.solve(0)
    duals = np.asarray(problem.opti.value(problem.opti.lam_g) if sol.success
                       else problem.opti.debug.value(problem.opti.lam_g)).ravel()
    print(name, "solve", sol.status, sol.iterations, sol.m_wet, flush=True)
    if not sol.success and sol.status in ("Solve_Succeeded", "Solved_To_Acceptable_Level"):
        problem.set_guess(problem.guess_from_solution(sol), m_wet0=sol.m_wet)
        problem.opti.set_initial(problem.opti.lam_g, duals)
        problem.p = replace(p, solver=replace(p.solver, ipopt={**p.solver.ipopt, "warm_start_init_point": "yes",
                                                               "mu_init": 1e-10}))
        sol = problem.solve(0)
        duals = np.asarray(problem.opti.value(problem.opti.lam_g) if sol.success
                           else problem.opti.debug.value(problem.opti.lam_g)).ravel()
        print(name, "polish", sol.status, sol.iterations, sol.m_wet, flush=True)
    sizing = dict(success=bool(sol.success), status=sol.status, wet_mass_kg=sol.m_wet, iterations=sol.iterations,
                  primal=sol.primal_infeasibility, dual=sol.dual_infeasibility, variables=int(problem.opti.nx),
                  constraints=int(problem.opti.ng), wall_s=time.monotonic() - tick)
    write(out / "sizing.json", sizing)
    write(out / "parameters.json", asdict(p))
    save_trajectory(out / "sizing.npz", p, sol, "uprange", physical_identity(p), dict(passed=False, diagnostic_only=True))
    if not sol.success:
        write(out / "summary.json", dict(variant=name, label=spec["label"], status="not_converged", sizing=sizing))
        return
    rows, history, contacts = integrate(p, sol, terrain)
    camera = camera_check(p, sol, terrain, history)
    hardware = hardware_checks(p, sol)
    checks = dict(passed=bool(all(r["passed"] for r in rows) and all(c["passed"] for c in contacts.values())
                              and all(c["passed"] for c in camera.values()) and hardware["passed"]
                              and not thrust_discard_flags(p, sol)),
                  propagation=rows, contacts=contacts, camera=camera, hardware=hardware)
    if checks["passed"]:
        checks["fixed_mass"] = fm = fixed_mass(p, terrain, sol, problem, duals)
        checks["passed"] = bool(fm["passed"])
    write(out / "validation.json", checks)
    lam = (min_dv_hop(p).dv, min_dv_hop(p, r1=p.mission.sites["uprange"], r2=p.mission.r_start).dv)
    summary = round_trip_summary(p, sol, "uprange", *lam)
    write(out / "summary.json", dict(variant=name, label=spec["label"],
                                     status="validated" if checks["passed"] else "validation_failed",
                                     m_dry=p.vehicle.m_dry, sizing=sizing, summary=summary,
                                     max_chained_position_error_m=max(r["chained_position_error_m"] for r in rows)))
    print(name, "PASSED" if checks["passed"] else "FAILED", flush=True)


def report():
    base = json.loads((BASE / "summary.json").read_text())
    rows = [("a2200_08 (baseline)", "Baseline", 2000.0, base["summary"], "validated",
             json.loads((BASE / "sizing.json").read_text()))]
    for name, spec in VARIANTS.items():
        path = OUT / name / "summary.json"
        if path.exists():
            d = json.loads(path.read_text())
            rows.append((name, spec["label"], d.get("m_dry"), d.get("summary"), d["status"], d["sizing"]))
    L = ["# Design checks — 8 A2200 DEM flight", "",
         "Each variant is re-optimized from the baseline solution and passes the same acceptance checks unless marked.", "",
         "| Variant | Dry (kg) | Status | Wet mass (kg) | Propellant used (kg) | Finite-burn Δv (m/s) | Variables |",
         "|---|---:|---|---:|---:|---:|---:|"]
    for name, label, dry, s, status, sz in rows:
        if s:
            L.append(f"| {label} | {dry:.0f} | {status} | {s['total']['m_start']:.2f} | {s['total']['propellant_used']:.2f} | "
                     f"{s['total']['finite_burn_dv']:.1f} | {sz.get('variables', '')} |")
        else:
            L.append(f"| {label} | {dry} | {status} | — | — | — | {sz.get('variables', '')} |")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--report-only" not in sys.argv:
        terrain = DemTerrain()
        for name in args or VARIANTS:
            try:
                solve(name, terrain)
            except Exception as exc:  # keep going; the failure is recorded in the log
                print(name, "ERROR", repr(exc), flush=True)
    report()

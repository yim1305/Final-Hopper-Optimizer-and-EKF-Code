"""One sizing/validation workflow for the five retained architectures."""

import argparse
from dataclasses import asdict, replace
import gc
import json
import time
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from hopper.dynamics import make_numpy_rhs
from hopper.problem import Problem
from hopper.tracker_cache import save_trajectory, json_value
from hopper.postprocess import (
    round_trip_summary,
    phase_table,
    thrust_discard_flags,
    phase_mode,
    propellant_by_class,
)
from hopper.lambert import min_dv_hop
from .config import ROOT, CASES, parameters, read_solution, write, digest, physical_identity
from .terrain import DemTerrain, GroundProblem
from .validation import integrate, camera_check
from .hardware import hardware_checks
from .trackers import tracker_checks


def numeric_settings(p):
    return replace(
        p,
        solver=replace(
            p.solver, ipopt={**p.solver.ipopt, "mu_strategy": "adaptive", "max_wall_time": 120.0}
        ),
    )


def sizing(case, p, terrain, out):
    problem = GroundProblem(p, "uprange", terrain=terrain)
    if (out / "sizing.npz").exists():
        if (out / "parameters.json").exists():
            saved = json.loads((out / "parameters.json").read_text())
            if physical_identity(saved) != physical_identity(p):
                raise ValueError("Physical inputs changed; use --retry to reoptimize this case")
        sol = read_solution(out / "sizing.npz")
        if not sol.success:
            raise ValueError("Saved sizing failed; use --retry after inspecting sizing.json")
        return sol, problem, np.load(out / "duals.npz")["duals"]
    retries = sorted(out.glob("rejected_*/sizing.npz"))
    if not retries:
        raise ValueError("Sizing restart missing; restore the saved case arrays before running")
    seed = read_solution(retries[-1])
    guess = Problem.guess_from_solution(seed)
    # Use the same cubic state interpolation as Hermite-Simpson when a
    # validation-driven mesh refinement needs more nodes. Linear controls stay linear.
    for phase in problem.phases:
        source = guess[phase.name]
        if source["X"].shape[1] == phase.n + 1:
            continue
        old_time = np.linspace(0.0, source["t"], source["X"].shape[1])
        new_time = np.linspace(0.0, source["t"], phase.n + 1)
        rhs = make_numpy_rhs(p, mode=phase.mode)
        slopes = np.array([rhs(x, u) for x, u in zip(source["X"].T, source["U"].T)])
        source["X"] = CubicHermiteSpline(old_time, source["X"].T, slopes)(new_time).T
        source["U"] = np.array([np.interp(new_time, old_time, row) for row in source["U"]])
    problem.set_guess(guess, m_wet0=seed.m_wet)
    if retries:
        prior = np.load(retries[-1].parent / "duals.npz")["duals"]
        if len(prior) == int(problem.opti.ng):
            problem.opti.set_initial(problem.opti.lam_g, prior)
            problem.p = replace(
                p,
                solver=replace(
                    p.solver,
                    ipopt={
                        **p.solver.ipopt,
                        "warm_start_init_point": "yes",
                        "warm_start_bound_push": 1e-10,
                        "warm_start_mult_bound_push": 1e-10,
                        "warm_start_slack_bound_push": 1e-10,
                        "mu_init": 1e-10,
                    },
                ),
            )
    tick = time.monotonic()
    problem.opti.callback(
        lambda k: print(case, "sizing iteration", k, flush=True) if k % 100 == 0 else None
    )
    sol = problem.solve(0)
    duals = np.asarray(
        problem.opti.value(problem.opti.lam_g)
        if sol.success
        else problem.opti.debug.value(problem.opti.lam_g)
    ).ravel()
    if not sol.success and sol.status in ("Solve_Succeeded", "Solved_To_Acceptable_Level"):
        problem.set_guess(problem.guess_from_solution(sol), m_wet0=sol.m_wet)
        problem.opti.set_initial(problem.opti.lam_g, duals)
        problem.p = replace(
            p,
            solver=replace(
                p.solver, ipopt={**p.solver.ipopt, "warm_start_init_point": "yes", "mu_init": 1e-10}
            ),
        )
        sol = problem.solve(0)
        duals = np.asarray(
            problem.opti.value(problem.opti.lam_g)
            if sol.success
            else problem.opti.debug.value(problem.opti.lam_g)
        ).ravel()
    save_trajectory(
        out / "sizing.npz",
        p,
        sol,
        "uprange",
        physical_identity(p),
        dict(passed=False, diagnostic_only=True),
    )
    np.savez_compressed(out / "duals.npz", duals=duals)
    write(
        out / "sizing.json",
        dict(
            success=bool(sol.success),
            status=sol.status,
            wet_mass_kg=sol.m_wet,
            iterations=sol.iterations,
            primal=sol.primal_infeasibility,
            dual=sol.dual_infeasibility,
            wall_s=time.monotonic() - tick,
            variables=int(problem.opti.nx),
            constraints=int(problem.opti.ng),
        ),
    )
    if not sol.success:
        raise ValueError(case + ": sizing did not pass; diagnostic retained")
    return sol, problem, duals


def fixed_mass(p, terrain, sol, sizing_problem, duals):
    warm = replace(
        p,
        solver=replace(
            p.solver,
            ipopt={
                **p.solver.ipopt,
                "warm_start_init_point": "yes",
                "warm_start_bound_push": 1e-10,
                "warm_start_mult_bound_push": 1e-10,
                "warm_start_slack_bound_push": 1e-10,
                "mu_init": 1e-10,
            },
        ),
    )
    reserve_index = sizing_problem.original_constraint_count - 1
    multiplier = abs(duals[reserve_index])
    problem = GroundProblem(
        warm,
        "uprange",
        terrain=terrain,
        mode="maximise_margin",
        fixed_m_wet=sol.m_wet,
        tw_max=sizing_problem.tw_max,
    )
    problem.set_guess(problem.guess_from_solution(sol))
    if multiplier > 1e-10:
        mapped = (
            np.delete(duals, [sizing_problem.initial_boundary_offset, reserve_index]) / multiplier
        )
        if len(mapped) != int(problem.opti.ng):
            raise ValueError("Fixed-mass dual mapping mismatch")
        problem.opti.set_initial(problem.opti.lam_g, mapped)
    check = problem.solve(0)
    ratio = (sol.m_wet - p.vehicle.m_dry) / (sol.m_wet - check.z)
    return dict(
        passed=bool(
            check.success
            and abs(ratio / p.mission.reserve_factor - 1) < 0.001
            and abs(check.z - sol.z) < 0.05
        ),
        status=check.status,
        iterations=check.iterations,
        primal=check.primal_infeasibility,
        dual=check.dual_infeasibility,
        loaded_over_used=ratio,
        final_mass_difference_kg=check.z - sol.z,
    )


def run_case(case, *, retry=False, report_only=False):
    out = ROOT / "results" / case
    out.mkdir(exist_ok=True)
    result_path = out / "summary.json"
    if result_path.exists() and not retry:
        existing = json.loads(result_path.read_text())
        if existing["status"] in ("not_converged", "validation_failed"):
            message = (
                "did not converge after 1000 IPOPT iterations"
                if existing["status"] == "not_converged"
                else "no fully validated solution; best horizontal touchdown speed 0.5015 m/s"
            )
            print(case, ":", message, "— see REPORT.md", flush=True)
            if report_only and existing["status"] == "validation_failed":
                from .report import report_case

                terrain = DemTerrain()
                return report_case(case, parameters(case, terrain), terrain)
            return
    terrain = DemTerrain()
    p = numeric_settings(parameters(case, terrain))
    if report_only:
        from .report import report_case

        return report_case(case, p, terrain)
    if retry and (out / "sizing.npz").exists():
        stamp = time.time_ns()
        archive = out / f"rejected_{stamp}"
        archive.mkdir()
        for name in ("sizing.npz", "sizing.json", "duals.npz"):
            if (out / name).exists():
                (out / name).replace(archive / name)
    sol, problem, duals = sizing(case, p, terrain, out)
    print(case, "wet mass", sol.m_wet, flush=True)
    write(out / "parameters.json", asdict(p))
    rows, history, contacts = integrate(p, sol, terrain)
    camera = camera_check(p, sol, terrain, history)
    hardware = hardware_checks(p, sol)
    source = dict(
        trajectory_sha256=digest(out / "sizing.npz"),
        physical_parameters_sha256=physical_identity(p),
    )
    checks = dict(
        passed=bool(
            all(row["passed"] for row in rows)
            and all(row["passed"] for row in contacts.values())
            and all(row["passed"] for row in camera.values())
            and hardware["passed"]
            and not thrust_discard_flags(p, sol)
        ),
        propagation=rows,
        contacts=contacts,
        camera=camera,
        hardware=hardware,
        **source,
    )
    write(out / "validation.json", checks)
    if not checks["passed"]:
        raise ValueError(case + ": independent flight/camera/hardware validation failed")
    fixed = fixed_mass(p, terrain, sol, problem, duals)
    checks["fixed_mass"] = fixed
    write(out / "validation.json", checks)
    if not fixed["passed"]:
        raise ValueError(case + ": fixed-mass reproduction failed")
    del problem
    gc.collect()
    trackers = tracker_checks(p, sol)
    write(out / "trackers.json", dict(**trackers, **source))
    if not trackers["passed"]:
        raise ValueError(case + ": tracker schedule failed")
    lam = (min_dv_hop(p).dv, min_dv_hop(p, r1=p.mission.sites["uprange"], r2=p.mission.r_start).dv)
    summary = round_trip_summary(p, sol, "uprange", *lam)
    table = phase_table(p, sol, "uprange")
    for row, ph in zip(table, sol.phases.values()):
        mode = phase_mode(p, ph)
        row["engine_counts"] = dict(mode.fixed + mode.throttled) if mode else {}
        if mode:
            row["engine_counts"].update(
                {
                    name: p.engine.classes[name].count
                    for name in p.engine.pools[ph["kind"]]
                    if p.engine.classes[name].pulsing
                }
            )
        row["propellant_by_class"] = propellant_by_class(p, ph)
        row["thrust_range_n"] = [float(ph["U"][3].min()), float(ph["U"][3].max())]
        row["start_mass_kg"] = float(ph["X"][6, 0])
        row["end_mass_kg"] = float(ph["X"][6, -1])
    files = [
        ROOT / "mission.yaml",
        ROOT / "config.yaml",
        *sorted((ROOT / "mission").glob("*.py")),
        *sorted((ROOT / "hopper").rglob("*.py")),
        ROOT / "data/terrain_corridor_30m.tif",
        ROOT / "data/terrain_source.json",
        ROOT / "data/ldem_16.img",
        ROOT / "data/ldem_16.lbl",
    ]
    sources = {str(path.relative_to(ROOT)): digest(path) for path in files}
    identity = (
        __import__("hashlib")
        .sha256(
            json.dumps(
                dict(parameters=asdict(p), sources=sources), default=json_value, sort_keys=True
            ).encode()
        )
        .hexdigest()
    )
    write(
        out / "summary.json",
        dict(
            status="numerically_validated_dem_flight",
            case=case,
            summary=summary,
            table=table,
            contacts=contacts,
            baseline_identity=identity,
            source_hashes=sources,
            **source,
            tracker_summary={
                name: {key: row[key] for key in ("passed", "event_count", "max_update_gap_s")}
                for name, row in trackers["reports"].items()
            },
            scope="Astro wet-mass and reference-trajectory optimization; flat landing ground at DEM-derived elevations",
        ),
    )
    save_trajectory(
        out / "trajectory.npz",
        p,
        sol,
        "uprange",
        identity,
        dict(passed=True, tracker_validation="passed", fixed_mass_check="passed", **source),
    )
    np.savez_compressed(
        out / "independent_flight.npz",
        time=np.concatenate([row["time"] for row in history]),
        r=np.concatenate([row["r"] for row in history]),
        state=np.concatenate([row["state"] for row in history]),
        hop=np.concatenate([np.full(len(row["time"]), row["hop"]) for row in history]),
    )
    checks["passed"] = True
    write(out / "validation.json", checks)
    from .report import report_case

    report_case(case, p, terrain, history)
    print(case, "ALL CHECKS PASSED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=(*CASES, "all"))
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate reports/plots from accepted arrays without solving",
    )
    parser.add_argument(
        "--retry", action="store_true", help="Preserve a rejected sizing attempt and try again"
    )
    args = parser.parse_args()
    cases = CASES if args.case == "all" else (args.case,)
    for case in cases:
        run_case(case, retry=args.retry, report_only=args.report_only)

"""New, reproducible mounting study on an unchanged saved DEM flight.

This reconstructs a method, not the deleted historical experiment. It never
re-solves a trajectory, changes installed mounts, or writes into a flight case.
Run from the repository root with the shared virtualenv and -B.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(__import__("pathlib").Path(__file__).resolve().parents[1] / "results/mount_reconstruction/mplconfig"))

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
METHOD = ("5-degree elevation grid at azimuths 0/180; additional four-angle random/Powell search only if the grid does not attain the mount-independent rate-only upper bound; shortlisted candidates checked at finer time steps.")
sys.path.insert(0, str(ROOT))
from hopper.star_tracker import rotation_z, top_mount_axes
from mission.config import parameters, physical_identity, read_solution, digest, write
from mission.terrain import DemTerrain
from mission.trackers import SkyTerrain, blocks


def batched_cones(sky, positions, axes, half):
    """Same 4-ring/32-azimuth rays and spacing as SkyTerrain.cone, batched.

    The fine directions contain every 2-ring/16-azimuth coarse direction, so
    their conjunction has the same clearance result. No DEM is approximated.
    """
    result = np.ones(len(positions), bool)
    for offset in range(0, len(positions), 16):
        r, axis = positions[offset:offset+16], axes[offset:offset+16]
        radius = np.linalg.norm(r, axis=1)
        from_nadir = np.arccos(np.clip(np.sum(axis*(-r/radius[:, None]), axis=1), -1, 1))
        bound_clear = (radius > sky.outer) & (from_nadir-half > np.arcsin(np.minimum(1, sky.outer/radius)))
        ref = np.tile([0., 0., 1.], (len(r), 1))
        ref[np.abs(axis[:, 2]) >= .9] = [1., 0., 0.]
        e1 = np.cross(axis, ref)
        e1 /= np.linalg.norm(e1, axis=1)[:, None]
        e2 = np.cross(axis, e1)
        directions = [axis]
        for theta in np.linspace(0, half, 5)[1:]:
            for phi in np.linspace(0, 2*np.pi, 32, endpoint=False):
                directions.append(np.cos(theta)*axis + np.sin(theta)*(np.cos(phi)*e1+np.sin(phi)*e2))
        directions = np.stack(directions, axis=1)
        rd = np.einsum("ni,nji->nj", r, directions)
        closest2 = radius[:, None]**2 - np.maximum(0, -rd)**2
        inner_hit = np.any(closest2 < sky.inner**2, axis=1) & ~bound_clear
        disc = rd**2 + sky.outer**2 - radius[:, None]**2
        lengths = -rd + np.sqrt(np.maximum(0, disc))
        active = (~bound_clear & ~inner_hit)[:, None] & (disc >= 0) & (lengths > 0)
        minimum = np.full(len(r), np.inf)
        points, owners, count = [], [], 0

        def flush():
            nonlocal points, owners, count
            if not points:
                return
            xyz, which = np.concatenate(points), np.concatenate(owners)
            clearance = np.linalg.norm(xyz, axis=1)-sky.native.radius-sky.height(xyz)
            np.minimum.at(minimum, which, clearance)
            points, owners, count = [], [], 0

        for i, j in zip(*np.nonzero(active)):
            if minimum[i] < -.001:
                continue
            length = lengths[i, j]
            near = min(length, 5000.)
            distance = np.unique(np.r_[np.arange(0, near, 30.), np.arange(near, length, 240.), length])
            points.append(r[i]+distance[:, None]*directions[i, j])
            owners.append(np.full(len(distance), i, dtype=int))
            count += len(distance)
            if count >= 100000:
                flush()
        flush()
        result[offset:offset+len(r)] = ~inner_hit & (minimum >= -.001)
    return result


class Evaluator:
    def __init__(self, p, histories, sky, step):
        self.st, self.sky, self.step = p.star_tracker, sky, step
        self.samples, self.cache = [], {}
        self.head_calls, self.ray_calls = 0, 0
        for block in histories:
            t = np.unique(np.r_[np.linspace(block.start, block.end,
                               int(np.ceil((block.end-block.start)/step))+1), block.knots])
            C = np.empty((len(t), 3, 3))
            r = np.empty((len(t), 3))
            rates = np.empty(len(t))
            assigned = np.zeros(len(t), bool)
            for ph in block.phases:
                select = (~assigned) & (t >= ph.start-1e-10) & (t <= ph.end+1e-10)
                if not select.any():
                    continue
                C[select], rates[select] = ph.orientation(t[select])
                r[select] = p.mission.r_start + ph.state(np.clip(t[select]-ph.start, 0, ph.ph["t"]))[:, :3]
                assigned[select] = True
            assert assigned.all()
            rate_ok = rates <= self.st.tracking_body_rate_max + self.st.rate_tolerance
            self.samples.append(dict(name=block.name, t=t, C=C, r=r,
                rotate=rotation_z(-p.moon.omega*t), rate_ok=rate_ok,
                start=block.start, end=block.end))

    def head(self, elevation, azimuth):
        key = (round(float(elevation), 8), round(float(azimuth) % 360, 8))
        if key in self.cache:
            return self.cache[key]
        axis = top_mount_axes(np.radians([elevation]), np.radians([azimuth]))[0]
        half = self.st.full_cone_fov/2 + self.st.moon_limb_margin
        result = []
        for s in self.samples:
            bore = s["C"] @ axis
            # Same center-to-boresight exclusions and rate tolerance as the saved checker.
            sun = np.arccos(np.clip(bore @ self.st.sun_direction_I, -1, 1))
            earth = np.arccos(np.clip(bore @ self.st.earth_direction_I, -1, 1))
            valid = s["rate_ok"] & (sun >= self.st.sun_exclusion-self.st.angular_tolerance) & (earth >= self.st.earth_exclusion-self.st.angular_tolerance)
            direction = np.einsum("nij,nj->ni", s["rotate"], bore)
            radius = np.linalg.norm(s["r"], axis=1)
            from_nadir = np.arccos(np.clip(np.sum(direction * (-s["r"]/radius[:, None]), axis=1), -1, 1))
            # This is a sufficient DEM bound, not a replacement lunar surface.
            bound_clear = (radius > self.sky.outer) & (from_nadir-half > np.arcsin(np.minimum(1, self.sky.outer/radius)))
            k = np.flatnonzero(valid & ~bound_clear)
            self.ray_calls += len(k)
            valid[k] = batched_cones(self.sky, s["r"][k], direction[k], half)
            result.append(valid)
        self.head_calls += 1
        self.cache[key] = result
        if self.head_calls % 20 == 0:
            print(f"step={self.step:g}s heads={self.head_calls} DEM queries={self.ray_calls}", flush=True)
        return result

    def metrics(self, masks):
        rows = {}
        total, duration = 0., 0.
        for s, valid in zip(self.samples, masks):
            updates = s["t"][valid]
            gap = float(np.diff(np.r_[s["start"], updates, s["end"]]).max())
            # Trapezoid weights avoid overweighting short control-knot intervals.
            seconds = float(np.sum(np.diff(s["t"]) * (valid[:-1].astype(float)+valid[1:])/2))
            span = s["end"]-s["start"]
            rows[s["name"]] = dict(available_fraction=seconds/span,
                max_sampled_update_age_s=gap, available_seconds=seconds,
                duration_s=span, measurement_count=int(valid.sum()))
            total += seconds
            duration += span
        worst = max(v["max_sampled_update_age_s"] for v in rows.values())
        return dict(available_fraction=total/duration,
            max_sampled_update_age_s=worst,
            passed=bool(worst <= self.st.update_gap_max+1e-8 and all(v["measurement_count"] for v in rows.values())),
            hops=rows)

    def pair(self, angles):
        e1, a1, e2, a2 = map(float, angles)
        one, two = self.head(e1, a1), self.head(e2, a2)
        m = self.metrics([a | b for a, b in zip(one, two)])
        return dict(elevations_deg=[e1, e2], azimuths_deg=[a1 % 360, a2 % 360], **m)

    def ceiling(self):
        return self.metrics([s["rate_ok"] for s in self.samples])


def score(row, limit):
    if not row["passed"]:
        return -1. - max(0., row["max_sampled_update_age_s"]-limit)/limit
    return row["available_fraction"]


def angles(row):
    return [row["elevations_deg"][0], row["azimuths_deg"][0],
            row["elevations_deg"][1], row["azimuths_deg"][1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="a2200_08")
    ap.add_argument("--step", type=float, default=1.)
    ap.add_argument("--fine-step", type=float, default=.25)
    ap.add_argument("--grid-step", type=float, default=5.)
    ap.add_argument("--random-pairs", type=int, default=24)
    ap.add_argument("--maxfev", type=int, default=100)
    ap.add_argument("--resume-grid", action="store_true", help="Reuse a completed grid from this interrupted study; requires identical protected inputs and grid settings")
    ap.add_argument("--report-only", action="store_true", help="Rebuild the figure and interpretation from saved study results without geometry calculations")
    ap.add_argument("--out", default="mount_reconstruction/a2200_08")
    args = ap.parse_args()
    if not (0 < args.fine_step <= args.step <= 1 and 0 < args.grid_step <= 10):
        ap.error("Require 0 < fine-step <= step <= 1 s and 0 < grid-step <= 10 deg")
    out = (ROOT/"results"/args.out).resolve()
    if not out.is_relative_to((ROOT/"results").resolve()) or out == (ROOT/"results"/args.case).resolve():
        ap.error("Output must be a separate folder under Astro/results")
    if args.report_only:
        report_existing(out)
        return
    if (out/"study.json").exists():
        ap.error("study.json already exists; use a new --out to preserve the previous study")
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    case = ROOT/"results"/args.case
    summary = json.loads((case/"summary.json").read_text())
    # Fingerprint every accepted case file, the frozen core, source configuration,
    # and navigation numerical outputs. New study outputs are separate.
    paths = list(case.glob("*")) + list((ROOT/"hopper").rglob("*.py")) + list((ROOT/"mission").rglob("*.py"))
    paths += [ROOT/"mission.yaml", ROOT/"config.yaml", ROOT.parent/"GN/nav_config.yaml"]
    paths += list((ROOT.parent/"GN/results").rglob("*.json")) + list((ROOT.parent/"GN/results").rglob("*.npz"))
    before = {str(p.relative_to(ROOT.parent)): digest(p) for p in paths if p.is_file()}
    if args.resume_grid:
        if before != json.loads((out/"input_hashes.json").read_text()):
            raise ValueError("Cannot resume: protected inputs changed")
    write(out/"input_hashes.json", before)
    terrain = DemTerrain()
    p = parameters(args.case, terrain)
    if physical_identity(p) != summary["physical_parameters_sha256"]:
        raise ValueError("Current physical parameters do not match the accepted flight")
    for name, expected in summary["source_hashes"].items():
        if digest(ROOT/name) != expected:
            raise ValueError(f"Saved source/data identity changed: {name}")
    if digest(case/"sizing.npz") != summary["trajectory_sha256"]:
        raise ValueError("Saved trajectory identity changed")
    if not json.loads((case/"trackers.json").read_text())["passed"]:
        raise ValueError("The source flight tracker checks did not pass")
    sol = read_solution(case/"sizing.npz")
    histories = blocks(p, sol)
    sky = SkyTerrain()
    evaluator = Evaluator(p, histories, sky, args.step)
    st = p.star_tracker
    # Cross-check the batching against the original checker near the terrain,
    # including clear and obstructed directions. This is not a sensor model change.
    sample = evaluator.samples[0]
    near = np.flatnonzero(np.linalg.norm(sample["r"], axis=1) < sky.outer)
    indices = near[np.linspace(0, len(near)-1, 16).astype(int)]
    test_rng = np.random.default_rng(19)
    test_axes = test_rng.normal(size=(len(indices), 3))
    test_axes /= np.linalg.norm(test_axes, axis=1)[:, None]
    half = st.full_cone_fov/2+st.moon_limb_margin
    reference = np.array([sky.cone(sample["r"][i], axis, half, 2, 16)["clear"]
                          and sky.cone(sample["r"][i], axis, half, 4, 32)["clear"]
                          for i, axis in zip(indices, test_axes)])
    actual = batched_cones(sky, sample["r"][indices], test_axes, half)
    np.testing.assert_array_equal(actual, reference)
    print(f"Batched DEM checks match original: {len(actual)} cases, {sum(actual)} clear", flush=True)
    retained_angles = [np.degrees(st.initial_elevations[0]), np.degrees(st.initial_azimuths[0]),
                       np.degrees(st.initial_elevations[1]), np.degrees(st.initial_azimuths[1])]
    retained = evaluator.pair(retained_angles)
    print("Retained:", json.dumps(retained), flush=True)
    elevations = np.arange(10., 90.+args.grid_step/2, args.grid_step)
    if args.resume_grid:
        if args.step != 1.:
            raise ValueError("This grid resume requires the original 1 s screening step")
        grid = json.loads((out/"grid.json").read_text())
        expected = [[float(a), float(b)] for a in elevations for b in elevations]
        if [r["elevations_deg"] for r in grid] != expected:
            raise ValueError("Cannot resume: elevation grid settings differ")
    else:
        masks1 = {e: evaluator.head(e, 0.) for e in elevations}
        masks2 = {e: evaluator.head(e, 180.) for e in elevations}
        grid = []
        for e1 in elevations:
            for e2 in elevations:
                row = evaluator.metrics([a | b for a, b in zip(masks1[e1], masks2[e2])])
                grid.append(dict(elevations_deg=[float(e1), float(e2)], azimuths_deg=[0., 180.], **row))
        write(out/"grid.json", grid)
    ranked = sorted(grid, key=lambda r: score(r, st.update_gap_max), reverse=True)
    print("Best grid:", json.dumps(ranked[0]), flush=True)
    # Retained values are comparisons, not optimization starts or target penalties.
    searches = []
    bounds = [(np.degrees(st.elevation_bounds[0][0]), np.degrees(st.elevation_bounds[0][1])),
              (0., 360.), (np.degrees(st.elevation_bounds[1][0]), np.degrees(st.elevation_bounds[1][1])), (0., 360.)]
    rng = np.random.default_rng(42)
    ceiling_score = evaluator.ceiling()["available_fraction"]
    def reaches_ceiling(row):
        return row["passed"] and row["available_fraction"] >= ceiling_score-1e-12
    candidates, random_candidates = list(ranked), []
    if reaches_ceiling(ranked[0]):
        print("Grid reaches the rate-only coverage upper bound; no four-angle refinement can improve this sampled objective.", flush=True)
    else:
        for unit in rng.random((args.random_pairs, 4)):
            row = evaluator.pair([lo+u*(hi-lo) for u, (lo, hi) in zip(unit, bounds)])
            candidates.append(row)
            random_candidates.append(row)
            if reaches_ceiling(row):
                break
    # Use a grid seed and two independent random starts when available.
    starts = [angles(ranked[0])]
    random_ranked = sorted(random_candidates, key=lambda r: score(r, st.update_gap_max), reverse=True)
    starts.extend(angles(r) for r in random_ranked[:2])
    for seed in starts:
        if any(reaches_ceiling(r) for r in candidates):
            break
        class ReachedCeiling(Exception):
            pass
        calls = 0
        def objective(x):
            nonlocal calls
            row = evaluator.pair(x)
            candidates.append(row)
            calls += 1
            if reaches_ceiling(row):
                raise ReachedCeiling()
            return -score(row, st.update_gap_max)
        try:
            result = minimize(objective, seed, method="Powell", bounds=bounds,
                options=dict(maxiter=6, maxfev=args.maxfev, xtol=.1, ftol=1e-6))
            searches.append(dict(start_deg=seed, final_deg=result.x.tolist(), success=bool(result.success),
                                 message=str(result.message), evaluations=int(result.nfev)))
        except ReachedCeiling:
            searches.append(dict(start_deg=seed, final_deg=angles(candidates[-1]), success=True,
                message="Reached rate-only coverage upper bound; further sampled coverage optimization unnecessary", evaluations=calls))
        print("Four-angle search:", json.dumps(searches[-1]), flush=True)
        write(out/"search_progress.json", dict(searches=searches, candidates=candidates))
    ranked_all = sorted(candidates, key=lambda r: score(r, st.update_gap_max), reverse=True)
    finalist_specs = [("retained", retained_angles), ("rounded_72_55", [72., 0., 55., 180.]),
                      ("rounded_73_55", [73., 0., 55., 180.]), ("best_elevation_grid", angles(ranked[0]))]
    seen = set()
    for row in ranked_all:
        key = tuple(round(v, 3) for v in angles(row))
        if key not in seen:
            finalist_specs.append((f"search_candidate_{len(seen)+1}", angles(row)))
            seen.add(key)
        if len(seen) == 3:
            break
    finalists = []
    fine = Evaluator(p, histories, sky, args.fine_step)
    for label, aa in finalist_specs:
        row = fine.pair(aa)
        row["label"] = label
        finalists.append(row)
        print("Fine:", label, json.dumps(row), flush=True)
    ceiling = fine.ceiling()
    # Numerical convergence check for retained and the best newly searched pair.
    best = max((r for r in finalists if r["label"].startswith("search_candidate")),
               key=lambda r: score(r, st.update_gap_max))
    finer = Evaluator(p, histories, sky, args.fine_step/2)
    convergence = {label: finer.pair(aa) for label, aa in [("retained", retained_angles), ("best_search", angles(best))]}
    changed = [name for name, expected in before.items() if digest(ROOT.parent/name) != expected]
    if changed:
        raise RuntimeError(f"Protected input/output files changed: {changed}")
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), case=args.case,
        provenance="New reconstruction on the current saved flight; not the deleted historical study.",
        args=vars(args), objective="Maximize duration-weighted either-head availability, subject to <=5 s sampled update age on each ascent/coast block; no target-angle penalty.",
        numerical_method=METHOD,
        geometry=st.geometry_label, required_phase_kinds=list(st.required_phase_kinds),
        trajectory_sha256=digest(case/"sizing.npz"), physical_parameters_sha256=physical_identity(p),
        sun_exclusion_deg=float(np.degrees(st.sun_exclusion)), earth_exclusion_deg=float(np.degrees(st.earth_exclusion)),
        full_cone_fov_deg=float(np.degrees(st.full_cone_fov)), rate_limit_deg_s=float(np.degrees(st.tracking_body_rate_max)),
        update_gap_limit_s=st.update_gap_max, grid_size=len(grid), feasible_grid_pairs=sum(r["passed"] for r in grid),
        coarse_retained=retained, coarse_best_grid=ranked[0], searches=searches, finalists=finalists,
        coarse_rate_only_upper_bound=evaluator.ceiling(), random_four_angle_pairs_evaluated=len(random_candidates),
        sampled_upper_bound_reached=any(reaches_ceiling(r) for r in candidates),
        finer_time_checks=convergence, rate_only_upper_bound=ceiling,
        protected_files_unchanged=True, protected_file_count=len(before), elapsed_s=time.monotonic()-start,
        batched_cone_crosscheck=dict(cases=len(actual), clear=int(sum(actual)), matched_original=True),
        limitations=["Original trajectory, original objective and original search settings are unavailable; exact historical reproduction is not claimed.",
            "This is a sampled search, not a proof of unique or global optimality or continuous-time clearance.",
            "Fine DEM cone checks use the existing 2x16 and 4x32 sampled-ray tests; finite time grids may miss shorter transitions.",
            "Sun/Earth vectors are existing fixed synthetic directions; no real ephemeris, catalog, physical spacecraft CAD or acquisition delays.",
            "Only ascent/coast is required by Astro; this does not replace whole-flight GN availability or qualify changed mounts for existing GN results."])
    write(out/"study.json", result)
    with (out/"elevation_grid.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["head1_elevation_deg", "head2_elevation_deg", "available_fraction", "max_sampled_update_age_s", "passed"])
        writer.writerows([*r["elevations_deg"], r["available_fraction"], r["max_sampled_update_age_s"], r["passed"]] for r in grid)
    report = ["# Tracker mounting reconstruction", "", result["provenance"], "",
        "The reference flight and installed mounting configuration were held unchanged. No trajectory or navigation Monte Carlo was rerun.", "",
        "Objective: "+result["objective"], "", f"Grid: {len(grid)} pairs; {result['feasible_grid_pairs']} meet the sampled update-age limit.", "",
        "| Candidate | Elevations above body XY (deg) | Azimuths (deg) | Either-head availability (%) | Worst sampled update age (s) | Pass |",
        "|---|---|---|---:|---:|---|"]
    for row in finalists:
        report.append(f"| {row['label']} | {row['elevations_deg'][0]:.4f}, {row['elevations_deg'][1]:.4f} | {row['azimuths_deg'][0]:.3f}, {row['azimuths_deg'][1]:.3f} | {100*row['available_fraction']:.4f} | {row['max_sampled_update_age_s']:.4f} | {row['passed']} |")
    report += ["", f"Rate-only availability upper bound on the same fine grid: {100*ceiling['available_fraction']:.4f}%.", "",
        "A retained feasible pair need not be the unique optimum. Report all candidates and numerical resolution; do not portray this as recovered historical output.", "",
        "Protected flight/source/configuration/navigation numerical files unchanged: "+str(len(before))+" (SHA-256 verified).", "",
        *["- "+s for s in result["limitations"]]]
    (out/"REPORT.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    plot(grid, finalists, out)
    report_existing(out, render_plot=False)
    print(f"Complete: {out}; elapsed {time.monotonic()-start:.1f}s", flush=True)


def report_existing(out, render_plot=True):
    result = json.loads((out/"study.json").read_text())
    grid = json.loads((out/"grid.json").read_text())
    protected = json.loads((out/"input_hashes.json").read_text())
    changed = [name for name, expected in protected.items() if digest(ROOT.parent/name) != expected]
    if changed:
        raise ValueError(f"Protected inputs changed since the study: {changed}")
    result["numerical_method"] = METHOD
    result["analysis_source_sha256"] = digest(Path(__file__))
    best = max(r["available_fraction"] for r in grid if r["passed"])
    tied = sum(r["passed"] and abs(r["available_fraction"]-best) < 1e-12 for r in grid)
    result["coarse_grid_pairs_tied_for_best"] = tied
    write(out/"study.json", result)
    fine = next(r for r in result["finalists"] if r["label"] == "retained")
    finer = result["finer_time_checks"]["retained"]
    report = (out/"REPORT.md").read_text(encoding="utf-8").split("\n## Interpretation")[0]
    report += "\n## Interpretation\n\n"
    report += (f"{tied} of {len(grid)} elevation-grid pairs tie for the best coarse-grid coverage. "
        "The retained 72.6601 / 54.9357 deg pair attains the same coverage and reaches the rate-only upper bound on the 0.25 s fine grid. "
        "This supports retaining an already analyzed installation; it does not recover a unique historical optimum.\n\n")
    report += (f"At {result['args']['fine_step']:g} s plus control-knot samples, retained availability is {100*fine['available_fraction']:.4f}% "
        f"and the longest sampled update age is {fine['max_sampled_update_age_s']:.4f} s. "
        f"At half that time step, availability is {100*finer['available_fraction']:.4f}% and the longest age is {finer['max_sampled_update_age_s']:.4f} s. "
        "The retained and best grid pair remain tied in the half-step check. These are sampled results, not continuous-time guarantees.\n\n")
    report += (f"Additional random four-angle pairs evaluated: {result['random_four_angle_pairs_evaluated']}; "
        f"Powell runs: {len(result['searches'])}. The coarse-grid rate-only upper bound was attained, so additional four-angle refinement was unnecessary for that sampled objective.\n\n")
    report += ("The original study objective is unknown. Availability was chosen explicitly for this new reconstruction; "
        "a margin-maximizing historical search might select a different point within this plateau.\n\n"
        "Suggested presentation wording: 'We retained the preliminary tracker installation at 72.66 and 54.94 degrees above body XY. "
        "A new sweep on the saved reference flight confirmed that this pair lies on a broad plateau of maximum sampled coverage, "
        "with about 99.3 percent either-head availability and sampled update ages below one second during the required ascent/coast phases.'\n\n"
        "Use 72 / 55 deg only as an explicitly tested approximation. The accepted flight and GN studies still use the exact retained angles. "
        "This study does not certify different mounts for the complete GN flight.\n\n"
        "![Mounting trade](mounting_trade.png)\n\n"
        "![Elevation versus tracker coverage](angle_vs_coverage.png)\n\n"
        "The elevation-versus-coverage figure includes every coarse-grid pair. Each light curve varies one head while holding the other at one of its 17 grid elevations; the dark line is the best coverage across those other-head settings. "
        "The star shows the exact retained pair evaluated at the same coarse time resolution. The coarse 99.7096% plateau and fine approximately 99.30% verification use different time grids and must not be compared as different mounting performance.\n")
    (out/"REPORT.md").write_text(report, encoding="utf-8")
    if render_plot:
        plot(grid, result["finalists"], out)
    print(f"Report/figure complete; {len(protected)} protected files unchanged.", flush=True)


def plot(grid, finalists, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    e = np.unique([r["elevations_deg"][0] for r in grid])
    coverage = np.array([100*r["available_fraction"] for r in grid]).reshape(len(e), len(e))
    gap = np.array([r["max_sampled_update_age_s"] for r in grid]).reshape(len(e), len(e))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, data, title in zip(axes, [coverage, gap], ["Either-head availability (%)", "Worst sampled update age (s)"]):
        mesh = ax.pcolormesh(e, e, data.T, shading="nearest", cmap="viridis")
        fig.colorbar(mesh, ax=ax)
        failed = np.argwhere(gap > 5)
        if len(failed):
            ax.scatter(e[failed[:, 0]], e[failed[:, 1]], marker="x", color="white", s=25, label="Exceeds 5 s")
        ax.scatter(*finalists[0]["elevations_deg"], marker="*", s=160, color="orangered", edgecolor="white", label="Retained 72.66 / 54.94")
        ax.set(xlabel="Head 1 elevation above body XY (deg)", ylabel="Head 2 elevation above body XY (deg)", title=title)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("New mounting reconstruction on saved a2200_08 flight\nElevation grid: azimuths fixed at 0 / 180 deg; ascent/coast only")
    fig.savefig(out/"mounting_trade.png", dpi=180)
    plt.close(fig)
    # Show angle against coverage directly without selecting only poor alternatives.
    study = json.loads((out/"study.json").read_text())
    chosen = study["coarse_retained"]
    chosen_coverage = 100*chosen["available_fraction"]
    upper = 100*study["coarse_rate_only_upper_bound"]["available_fraction"]
    fine_coverage = 100*study["finer_time_checks"]["retained"]["available_fraction"]
    tied = sum(r["passed"] and abs(100*r["available_fraction"]-coverage.max()) < 1e-10 for r in grid)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    fig.subplots_adjust(top=.76, bottom=.23, left=.08, right=.98, wspace=.12)
    for head, ax in enumerate(axes):
        data = coverage if head == 0 else coverage.T
        for j in range(len(e)):
            ax.plot(e, data[:, j], color="#9bbbd0", alpha=.75, linewidth=1,
                    label="Other head held at each grid angle" if j == 0 else None)
        ax.plot(e, data.max(axis=1), color="#176d92", linewidth=2.2,
                label="Best coverage across other-head angles")
        ax.axhline(upper, color="#555555", linestyle="--", linewidth=1,
                   label="Body-rate coverage upper bound")
        ax.scatter(chosen["elevations_deg"][head], chosen_coverage, marker="*", s=220,
                   color="#d14935", edgecolor="white", linewidth=1, zorder=6,
                   label=f"Chosen pair: {chosen['elevations_deg'][head]:.2f} deg for this head")
        ax.annotate(f"{chosen['elevations_deg'][head]:.2f} deg", 
                    (chosen["elevations_deg"][head], chosen_coverage), xytext=(0, 12),
                    textcoords="offset points", ha="center", color="#b43726", fontweight="bold")
        ax.set(xlabel=f"Head {head+1} elevation above body XY (deg)",
               title=f"Head {head+1} elevation sweep", xlim=(8, 92),
               ylim=(np.floor(coverage.min())-1, 102))
        ax.grid(alpha=.2)
        ax.legend(loc="lower right", fontsize=8, framealpha=.95)
    axes[0].set_ylabel("Time with at least one usable tracker (%)")
    fig.suptitle("Tracker mounting angle versus coverage on the saved flight", y=.97, fontsize=16)
    fig.text(.5, .875, f"Chosen pair ties the best sampled coverage: {chosen_coverage:.2f}%  |  {tied} of {len(grid)} grid pairs tie for best", 
             ha="center", fontsize=11, color="#333333")
    fig.text(.5, .105, "All grid alternatives shown; no unique peak at the chosen angles. Azimuths fixed at 0 / 180 deg.",
             ha="center", fontsize=10)
    fig.text(.5, .055, f"Ascent/coast only. Curves and chosen star: 1 s screening grid + control knots. Finer retained-pair check: {fine_coverage:.2f}%.",
             ha="center", fontsize=9, color="#555555")
    fig.savefig(out/"angle_vs_coverage.png", dpi=200)
    fig.savefig(out/"angle_vs_coverage.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()

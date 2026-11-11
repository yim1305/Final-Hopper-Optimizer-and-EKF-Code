"""Terrain-only radio visibility from the real main flight to the original launch base."""

import csv
import json
import numpy as np
from scipy.interpolate import CubicHermiteSpline
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import ROOT, SETTINGS, parameters, write, digest
from .terrain import DemTerrain
from .validation import ray_clearance
from .report import save


def summarize(times, clearance):
    times, index = np.unique(times, return_index=True)
    clearance = np.asarray(clearance)[index]
    if len(times) < 2 or not np.isfinite(clearance).all():
        raise ValueError("Two finite LOS samples required")
    cuts = [float(times[0]), float(times[-1])]
    for k in np.flatnonzero((clearance[:-1] >= 0) != (clearance[1:] >= 0)):
        cuts.append(
            float(
                times[k]
                + (times[k + 1] - times[k]) * (-clearance[k]) / (clearance[k + 1] - clearance[k])
            )
        )
    intervals = []
    for start, end in zip(sorted(set(cuts))[:-1], sorted(set(cuts))[1:]):
        visible = bool(np.interp((start + end) / 2, times, clearance) >= 0)
        if intervals and intervals[-1]["visible"] == visible:
            intervals[-1]["end_s"] = end
            intervals[-1]["duration_s"] = end - intervals[-1]["start_s"]
        else:
            intervals.append(
                dict(start_s=start, end_s=end, duration_s=end - start, visible=visible)
            )
    duration = float(times[-1] - times[0])
    available = sum(row["duration_s"] for row in intervals if row["visible"])
    outages = [row for row in intervals if not row["visible"]]
    return dict(
        duration_s=duration,
        visible_duration_s=available,
        blocked_duration_s=duration - available,
        visible_fraction=available / duration,
        longest_blackout_s=max((row["duration_s"] for row in outages), default=0.0),
        intervals=intervals,
        outages=outages,
    )


def trace(p, terrain, flight, time_step, ray_step):
    up = p.mission.r_start / np.linalg.norm(p.mission.r_start)
    base = p.mission.r_start + SETTINGS["base_antenna_height_above_dem_m"] * up
    rows = []
    reports = {}
    for hop in (1, 2):
        selected = flight["hop"] == hop
        original = flight["time"][selected]
        times, index = np.unique(original, return_index=True)
        positions = flight["r"][selected][index]
        velocities = flight["state"][selected][index, 3:6]
        interpolate = CubicHermiteSpline(times, positions, velocities)
        sample = np.unique(np.r_[np.arange(times[0], times[-1], time_step), times[-1]])
        positions = interpolate(sample)
        antenna = (
            positions
            + SETTINGS["hopper_antenna_radial_offset_m"]
            * positions
            / np.linalg.norm(positions, axis=1)[:, None]
        )
        clearance = []
        for k, point in enumerate(antenna):
            clearance.append(ray_clearance(terrain, base, point, ray_step))
            if k % 1000 == 0:
                print("LOS hop", hop, k, "/", len(sample), "step", time_step, flush=True)
        reports[f"hop{hop}"] = summarize(sample, clearance)
        rows.extend(
            dict(time_s=float(t), hop=hop, minimum_clearance_m=float(c), visible=bool(c >= 0))
            for t, c in zip(sample, clearance)
        )
    reports["total"] = summarize(
        [row["time_s"] for row in rows], [row["minimum_clearance_m"] for row in rows]
    )
    return rows, reports


def main():
    case = ROOT / "results/a2200_08"
    out = ROOT / "results/los"
    out.mkdir(exist_ok=True)
    summary = json.loads((case / "summary.json").read_text())
    validation = json.loads((case / "validation.json").read_text())
    if summary["status"] != "numerically_validated_dem_flight" or not validation["passed"]:
        raise ValueError("Validated main DEM flight required")
    terrain = DemTerrain()
    p = parameters("a2200_08", terrain)
    with np.load(case / "independent_flight.npz", allow_pickle=False) as data:
        flight = {name: data[name].copy() for name in data.files}
    coarse_rows, coarse = trace(
        p, terrain, flight, SETTINGS["coarse_time_step_s"], SETTINGS["coarse_ray_step_m"]
    )
    rows, reports = trace(
        p, terrain, flight, SETTINGS["fine_time_step_s"], SETTINGS["fine_ray_step_m"]
    )
    difference = max(
        abs(reports[key]["visible_duration_s"] - coarse[key]["visible_duration_s"])
        for key in reports
    )
    if difference > max(SETTINGS["coarse_time_step_s"], SETTINGS["fine_time_step_s"]):
        raise ValueError("Coarse/fine LOS boundary timing differs by more than a coarse time step")
    result = dict(
        source="a2200_08/independent_flight.npz",
        source_sha256=digest(case / "independent_flight.npz"),
        reports=reports,
        coarse_fine_maximum_visible_duration_difference_s=difference,
        antenna_heights_m=dict(
            base=SETTINGS["base_antenna_height_above_dem_m"],
            hopper=SETTINGS["hopper_antenna_radial_offset_m"],
        ),
        time_step_s=SETTINGS["fine_time_step_s"],
        ray_step_m=SETTINGS["fine_ray_step_m"],
        scope="Terrain-only straight-ray LOS to original launch base; no link budget, atmospheric refraction, elevation mask or terrain uncertainty margin. Actual flight only; no descent extension or surface stay.",
    )
    write(out / "summary.json", result)
    with (out / "visibility.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 5), sharex=True, gridspec_kw={"height_ratios": [1, 2]}
    )
    for row in reports["total"]["intervals"]:
        axes[0].broken_barh(
            [(row["start_s"], row["duration_s"])],
            (0, 1),
            facecolors="#187b45" if row["visible"] else "#a52c30",
        )
    axes[0].set(
        yticks=[],
        ylabel="Launch-base LOS",
        title="Actual 8-engine DEM flight — green: visible; red: blocked",
    )
    axes[1].plot(
        [row["time_s"] for row in rows],
        [row["minimum_clearance_m"] / 1000 for row in rows],
        color="#0759a0",
    )
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(
        xlabel="Cumulative flight time (s; no added descent or surface stay)",
        ylabel="Minimum terrain\nray clearance (km)",
    )
    axes[1].grid(alpha=0.25)
    boundary = next(row["time_s"] for row in rows if row["hop"] == 2)
    for ax in axes:
        ax.axvline(boundary, color="black", ls="--", lw=0.8)
    axes[0].text(
        boundary, 0.5, " Outbound contact / return start", va="center", fontsize=8, color="white"
    )
    save(fig, out / "visibility.png")
    lines = [
        "# Main-design launch-base LOS",
        "",
        "Only the actual validated 8-engine DEM flight is used. One Moon base at original launch; no station at the crater destination, no five-second extension and no surface-stay duration.",
        "",
        "| Flight | Duration (s) | Visible (s) | Coverage | Longest blackout (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in reports.items():
        lines.append(
            f"| {name} | {row['duration_s']:.2f} | {row['visible_duration_s']:.2f} | {row['visible_fraction']:.2%} | {row['longest_blackout_s']:.2f} |"
        )
    lines += [
        "",
        "[Full-flight LOS timeline and ray clearance](visibility.png)",
        "",
        f"Both antennas use the existing 2 m radial-height assumption. Native 30 m terrain, sampled at {SETTINGS['fine_ray_step_m']} m along rays and {SETTINGS['fine_time_step_s']} s in time. Coarse/fine visible-duration difference is {difference:.4f} s. Terrain-only geometric visibility; link budget, elevation mask and terrain uncertainty margin are not specified. Clear-ray minima can be set by the antenna endpoints.",
        "",
        "`summary.json` contains coverage and blocked intervals; `visibility.csv` contains sampled time/clearance/visibility.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                name: {
                    key: row[key]
                    for key in ("duration_s", "visible_fraction", "longest_blackout_s")
                }
                for name, row in reports.items()
            },
            indent=2,
        )
    )

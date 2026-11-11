"""Short flight reports and standard plots from accepted numerical trajectories."""

import csv
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hopper.sensor import image_angles
from hopper.postprocess import phase_mode
from .config import ROOT, CASES, read_solution, physical_identity, digest
from .validation import integrate


def save(fig, path):
    path = Path(path)
    result = path.parent / "summary.json"
    if result.exists() and json.loads(result.read_text()).get("status") == "validation_failed":
        fig.text(
            0.5, 0.01, "SIZING ONLY — NOT VALIDATED", ha="center", color="#a52c30", weight="bold"
        )
    fig.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(fig)


def trajectory_plot(p, sol, terrain, path):
    origin = p.mission.r_start
    up = origin / np.linalg.norm(origin)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    basis = np.column_stack([north, east, up])
    r = np.concatenate([(origin[:, None] + ph["X"][:3]).T for ph in sol.phases.values()])
    x, y = terrain.projected(tuple(r.T))
    pad = 5000.0
    gx = np.linspace(
        max(min(x) - pad, terrain.xgrid[0] + 1.0), min(max(x) + pad, terrain.xgrid[-1] - 1.0), 220
    )
    gy = np.linspace(
        max(min(y) - pad, terrain.ygrid[0] + 1.0), min(max(y) + pad, terrain.ygrid[-1] - 1.0), 120
    )
    a, b = np.meshgrid(gx, gy)
    den = a * a + b * b + 4 * terrain.radius**2
    unit = np.stack(
        [
            4 * terrain.radius * b / den,
            4 * terrain.radius * a / den,
            (a * a + b * b - 4 * terrain.radius**2) / den,
        ],
        axis=-1,
    )
    ground = unit * (terrain.radius + terrain.height_positions(unit * terrain.radius))[..., None]
    surface = (ground - origin) @ basis / 1000.0
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(
        surface[..., 0],
        surface[..., 1],
        surface[..., 2],
        cmap="Greys",
        alpha=0.78,
        linewidth=0,
        rstride=2,
        cstride=2,
    )
    for hop, color in ((1, "#d77800"), (2, "#0759a0")):
        points = np.concatenate(
            [(origin[:, None] + ph["X"][:3]).T for ph in sol.phases.values() if ph["hop"] == hop]
        )
        xyz = (points - origin) @ basis / 1000.0
        ax.plot(*xyz.T, color=color, lw=2.3, label="Outbound" if hop == 1 else "Return")
    for point, label in ((origin, "Launch base"), (p.mission.r_target, "Drygalski target")):
        xyz = (point - origin) @ basis / 1000.0
        ax.scatter(*xyz, color="black", s=25)
        ax.text(*xyz, " " + label)
    ax.set(
        xlabel="North from launch (km)",
        ylabel="East from launch (km)",
        zlabel="Local up from launch (km)",
        title="Reference trajectory and mapped lunar surface",
    )
    ax.set_box_aspect((2.3, 1, 0.8))
    ax.view_init(25, -65)
    ax.legend()
    save(fig, path)


def report_case(case, p, terrain, history=None):
    out = ROOT / "results" / case
    summary = json.loads((out / "summary.json").read_text())
    validation = json.loads((out / "validation.json").read_text())
    trackers = json.loads((out / "trackers.json").read_text())
    if summary["physical_parameters_sha256"] != physical_identity(p) or summary[
        "trajectory_sha256"
    ] != digest(out / "sizing.npz"):
        raise ValueError(
            "Report inputs changed; regenerate validation for the current trajectory and settings"
        )
    diagnostic = summary["status"] == "validation_failed"
    if not diagnostic and (
        summary["status"] != "numerically_validated_dem_flight"
        or not validation["passed"]
        or not trackers["passed"]
    ):
        raise ValueError("Only accepted flights may be reported")
    sol = read_solution(out / "trajectory.npz")
    if history is None:
        _, history, _ = integrate(p, sol, terrain)
    trajectory_plot(p, sol, terrain, out / "trajectory_3d.png")
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for item in history:
        state = item["state"]
        r = item["r"]
        up = r / np.linalg.norm(r, axis=1)[:, None]
        vertical = np.einsum("ij,ij->i", state[:, 3:6], up)
        color = "#d77800" if item["hop"] == 1 else "#0759a0"
        axes[0].plot(item["time"], item["AGL"] / 1000, color=color)
        axes[1].plot(item["time"], np.linalg.norm(state[:, 3:6], axis=1), color=color)
        axes[2].plot(item["time"], item["U"][:, 3] / 1000, color=color)
        axes[3].plot(item["time"], state[:, 6], color=color)
    for ax, label in zip(
        axes, ("Height above DEM (km)", "Speed (m/s)", "Thrust (kN)", "Mass (kg)")
    ):
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Cumulative flight time (s; surface stay excluded)")
    axes[0].set_title(case + " — flight profiles")
    save(fig, out / "flight_profiles.png")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    for ax, item in zip(axes, [row for row in history if row["kind"] == "observation"]):
        points = [
            p.mission.r_target if item["hop"] == 1 else p.mission.r_start,
            p.mission.r_start
            + sol.phases["terminal_uprange" if item["hop"] == 1 else "terminal2_uprange"]["X"][
                :3, -1
            ],
        ]
        angles = np.array(
            [
                [image_angles(u[:3], p.mission.downrange, point - r)[:2] for point in points]
                for r, u in zip(item["r"], item["U"])
            ]
        )
        ax.plot(
            item["time"],
            np.max(np.abs(angles[:, :, 0]), axis=1),
            label="Maximum horizontal look angle",
        )
        ax.plot(
            item["time"],
            np.max(np.abs(angles[:, :, 1]), axis=1),
            label="Maximum vertical look angle",
        )
        ax.axhline(np.degrees(p.sensor.fov[0]) / 2, color="red", ls="--", label="Usable half FOV")
        result = validation["camera"][item["name"]]
        ax.set_title(
            ("Outbound" if item["hop"] == 1 else "Return")
            + f" camera observation — visibility {'PASS' if result['passed'] else 'FAIL'}; terrain rays checked"
        )
        ax.set(ylabel="Absolute look angle (deg)", xlabel="Flight time (s)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    save(fig, out / "camera_visibility.png")
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    for ax, (name, block) in zip(axes, trackers["reports"].items()):
        events = block["events"]
        times = np.array([e["time_s"] for e in events])
        heads = np.array([e["tracker"] for e in events])
        for head, marker in ((1, "o"), (2, "^")):
            chosen = heads == head
            ax.scatter(times[chosen], heads[chosen], s=9, marker=marker, label=f"Tracker {head}")
        ax.set(
            yticks=[1, 2],
            ylim=(0.5, 2.5),
            xlabel="Flight time (s)",
            ylabel="Accepted head",
            title=f"{name}: ascent + coast — {len(events)} valid updates, maximum gap {block['max_update_gap_s']:.3f} s / {p.star_tracker.update_gap_max:.1f} s limit",
        )
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    save(fig, out / "tracker_coverage.png")
    table = summary["table"]
    with (out / "phases.csv").open("w", newline="", encoding="utf-8") as f:
        fields = (
            "phase",
            "t_start",
            "t_end",
            "duration",
            "engine_counts",
            "thrust_range_n",
            "propellant",
            "dv",
            "start_mass_kg",
            "end_mass_kg",
        )
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)
    total = summary["summary"]["total"]
    hops = summary["summary"]
    lines = [
        f"# {case} — " + ("unvalidated sizing result" if diagnostic else "validated DEM flight"),
        "",
        f"Wet mass **{sol.m_wet:.2f} kg**; dry mass **{p.vehicle.m_dry:.0f} kg**. Loaded / consumed / remaining propellant: **{total['propellant_loaded']:.2f} / {total['propellant_used']:.2f} / {total['margin']:.2f} kg**.",
        "",
        f"Outbound / return time: **{hops['hop1']['duration']:.2f} / {hops['hop2']['duration']:.2f} s**. Total powered time: **{hops['hop1']['burn_time']+hops['hop2']['burn_time']:.2f} s**. Integrated thrust delta-v: **{total['finite_burn_dv']:.2f} m/s**.",
        "",
        "Same DEM mission, 2,000 kg dry, 150 m uprange divert, 25 m landing region, and 1.5/0.5 m/s downward/horizontal touchdown limits. Ground is assumed flat at DEM-derived elevation; vertical/tilt use radial local up. Surface stay is excluded.",
        "",
        "| Contact | Downward speed (m/s) | Horizontal speed (m/s) | Target offset (m) |",
        "|---|---:|---:|---:|",
    ]
    for name, row in validation["contacts"].items():
        lines.append(
            f"| {name} | {-row['vertical_speed_m_s']:.6f} | {row['horizontal_speed_m_s']:.6f} | {row['target_error_m']:.3f} |"
        )
    if diagnostic:
        lines += [
            "",
            "**No fully validated solution. Best independently checked return horizontal touchdown speed: 0.5015 m/s, against the 0.5000 m/s limit.** The sizing solver converged, but independent contact validation failed. All masses, times and plots in this report are diagnostic sizing results. Refinement was stopped at the user’s request.",
            "",
        ]
    lines += [
        "",
        "| Phase | Start (s) | Duration (s) | Active engines | Propellant (kg) | Delta-v (m/s) |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for row in table:
        counts = ", ".join(f"{n} {name}" for name, n in row["engine_counts"].items() if n) or "Off"
        lines.append(
            f"| {row['phase']} | {row['t_start']:.2f} | {row['duration']:.3f} | {counts} | {row['propellant']:.3f} | {row['dv']:.3f} |"
        )
    lines += [
        "",
        (
            "Independent contact validation failed; there is no final flight acceptance or fixed-mass acceptance claim. Camera and tracker plots show checks on the diagnostic source trajectory."
            if diagnostic
            else "Independent first-DEM-contact propagation, clearance, camera, engine checks, tracker update schedules and fixed-mass reserve reproduction pass."
        )
        + " Tracker availability is required during ascent/coast, with a 2 s maximum update gap. Camera geometry is evaluated during observation.",
        "",
        "[3D trajectory and lunar surface](trajectory_3d.png) · [Flight profiles](flight_profiles.png) · [Camera visibility](camera_visibility.png) · [Tracker updates](tracker_coverage.png)",
        "",
        "`trajectory.npz`: complete reference states/controls/times. `parameters.json`: effective inputs. `summary.json` / `phases.csv`: flight bookkeeping. `validation.json` / `trackers.json`: acceptance evidence. `sizing.npz` / `duals.npz`: restart inputs.",
        "",
        "These are nominal local optimization results. Synthetic Sun/Earth directions, prescribed attitude and sampled terrain-ray/cone checks are retained. No global-optimality, continuous terrain-uncertainty or GNC claim is made.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    comparison_report()


def comparison_report():
    lines = [
        "# Retained Astro cases",
        "",
        "Eight A2200 remains the selected design. All five tested cases use the common DEM mission, 2,000 kg dry mass and 1.5/0.5 m/s touchdown limits. Failed or unvalidated cases are explicitly labeled.",
        "",
        "| Case | Wet mass (kg) | Used propellant (kg) | Powered time (s) | Flight time (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in CASES:
        path = ROOT / "results" / case / "summary.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        if data["status"] == "not_converged":
            lines.append(
                f"| [{case}]({case}/REPORT.md) — did not converge after {data['iterations']:,} iterations | — | — | — | — |"
            )
            continue
        s = data["summary"]
        t = s["total"]
        label = case + (
            " (sizing only; not validated)" if data["status"] == "validation_failed" else ""
        )
        lines.append(
            f"| [{label}]({case}/REPORT.md) | {t['m_start']:.2f} | {t['propellant_used']:.2f} | {s['hop1']['burn_time']+s['hop2']['burn_time']:.2f} | {s['hop1']['duration']+s['hop2']['duration']:.2f} |"
        )
    lines += [
        "",
        "14-engine note: no fully validated solution; best independently checked horizontal touchdown speed 0.5015 m/s versus the 0.5000 m/s limit. Its table values are provisional sizing diagnostics. Mixed-engine case: no accepted result after 1,000 IPOPT iterations.",
        "",
        "Engine/feed hardware mass is included in the common dry-mass assumption, without an architecture-dependent mass adjustment. Each architecture retains its configured engine shutdown sequence and hardware envelope.",
        "",
        "[Main-design launch-base LOS](los/REPORT.md).",
        "",
    ]
    (ROOT / "results/REPORT.md").write_text("\n".join(lines), encoding="utf-8")

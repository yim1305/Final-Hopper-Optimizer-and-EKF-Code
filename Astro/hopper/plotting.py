"""Matplotlib views of a solved round trip; all plotted samples come from the optimizer."""
from pathlib import Path

import numpy as np

from .postprocess import chain, phase_mode

OUTBOUND = "#2369a8"
RETURN = "#d6732b"


def plot_round_trip(p, sol, branch, summary, output_dir="results", *, show=True):
    """Save a 3D trajectory and flight profiles, then optionally open the three figure windows."""
    if not sol.success:
        raise ValueError("Plotting requires a converged trajectory")
    if not show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # A true Cartesian frame at launch: downrange, crossrange, local up.
    start = p.mission.r_start
    up = start / np.linalg.norm(start)
    along = p.mission.r_target - start
    along = along - (along @ up) * up
    along /= np.linalg.norm(along)
    frame = np.vstack((along, np.cross(up, along), up))

    segments, elapsed = [], 0.0
    for name in chain(branch, p):
        ph = sol.phases[name]
        X, U = ph["X"], ph["U"]
        r_abs = start[:, None] + X[0:3]
        radius = np.linalg.norm(r_abs, axis=0)
        tilt = np.full(X.shape[1], np.nan)
        if ph["powered"]:
            cosine = np.sum(U[0:3] * (r_abs / radius), axis=0) / np.linalg.norm(U[0:3], axis=0)
            tilt = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        returning = "2_" in name
        segments.append(dict(name=name, phase=ph, xyz=frame @ X[0:3] / 1000,
                             time=(elapsed + np.linspace(0, ph["t"], X.shape[1])) / 60,
                             altitude=(radius - p.moon.radius) / 1000,
                             speed=np.linalg.norm(X[3:6], axis=0), tilt=tilt,
                             color=RETURN if returning else OUTBOUND,
                             style="-" if ph["powered"] else "--",
                             label=("Return" if returning else "Outbound") if ph["kind"] == "rise" else None))
        elapsed += ph["t"]

    fig3d = plt.figure("Lunar hopper - 3D trajectory", figsize=(11, 7), clear=True)
    ax3d = fig3d.add_subplot(projection="3d")
    fig3d.subplots_adjust(left=0.03, right=0.90, bottom=0.15, top=0.88)
    for seg in segments:
        ax3d.plot(*seg["xyz"], color=seg["color"], linestyle=seg["style"],
                  linewidth=2, label=seg["label"])

    # A small spherical surface patch makes lunar curvature visible without plotting the whole Moon.
    xyz = np.hstack([seg["xyz"] for seg in segments])
    pad = max(np.ptp(xyz[0]) * 0.03, 1.0)
    xx, yy = np.meshgrid(np.linspace(xyz[0].min() - pad, xyz[0].max() + pad, 45),
                         np.linspace(xyz[1].min() - 0.25, xyz[1].max() + 0.25, 7))
    zz = np.sqrt(np.maximum((p.moon.radius / 1000) ** 2 - xx ** 2 - yy ** 2, 0))
    zz -= np.linalg.norm(start) / 1000
    ax3d.plot_wireframe(xx, yy, zz, color="0.65", linewidth=0.5, alpha=0.45)

    for site, marker, color, label in ((start, "o", "#27804a", "Launch / return site"),
                                     (p.mission.r_target, "+", "0.4", "Original target"),
                                     (start + sol.phases[f"terminal_{branch}"]["X"][:3, -1], "v", "#a43636", "Actual divert landing")):
        point = frame @ (site - start) / 1000
        ax3d.scatter(*point, marker=marker, color=color, s=55, label=label, depthshade=False)
    for i, name in enumerate(("observation", f"observation2_{branch}")):
        point = frame @ sol.phases[name]["X"][0:3, -1] / 1000
        ax3d.scatter(*point, marker="*", color="#7045a0", s=80,
                     label="Moving gates" if i == 0 else None, depthshade=False)
    handles, labels = ax3d.get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.4", linestyle="--"))
    ax3d.legend(handles, labels + ["Coast: engines off"], loc="upper left", fontsize=9)
    ax3d.set(xlabel="Downrange from launch (km)", ylabel="Crossrange (km)",
             zlabel="Height above launch tangent plane (km)")
    ax3d.set_box_aspect((2.4, 1, 1.1))
    ax3d.view_init(elev=23, azim=-65)
    fig3d.suptitle(f"Lunar hopper round trip | {p.mission.divert_radius:g} m {branch} divert", fontsize=14)
    total = summary["total"]
    fig3d.text(0.05, 0.07,
               f"Wet mass {sol.m_wet:,.1f} kg   |   Propellant used {total['propellant_used']:,.1f} kg   |   "
               f"Delta-v {total['finite_burn_dv']:,.0f} m/s   |   Flight time {elapsed / 60:.1f} min", fontsize=10)
    fig3d.text(0.05, 0.035, "Local Cartesian coordinates; lunar surface shown in grey. Axis scales differ.",
               fontsize=9, color="0.4")

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout="constrained",
                             num="Lunar hopper - Flight profiles", clear=True)
    altitude_ax, speed_ax, thrust_ax, mass_ax, tilt_ax, divert_ax = axes.flat
    time_axes = (altitude_ax, speed_ax, thrust_ax, mass_ax, tilt_ax)
    for seg in segments:
        ph, t = seg["phase"], seg["time"]
        values = (seg["altitude"], seg["speed"], ph["U"][3] / 1000, ph["X"][6], seg["tilt"])
        for ax, value in zip(time_axes, values):
            ax.plot(t, value, color=seg["color"], linestyle=seg["style"],
                    linewidth=1.8, label=seg["label"])
            if not ph["powered"]:
                ax.axvspan(t[0], t[-1], color="0.5", alpha=0.08)
        if ph["kind"] == "observation":
            for ax in time_axes:
                ax.axvspan(t[0], t[-1], color="#45a17e", alpha=0.25)
            for ax in time_axes:
                ax.axvline(t[-1], color="#7045a0", linestyle=":", linewidth=0.8)
        if ph["kind"] == "terminal":
            tilt_ax.scatter(t[-1], seg["tilt"][-1], color=seg["color"], s=25, zorder=5)

    # Fixed engines switch at phase boundaries; the pulsing remainder is a
    # time-averaged continuous command. Draw the actual interpolation model.
    fixed_time, fixed_thrust = [], []
    for seg in segments:
        mode = phase_mode(p, seg["phase"])
        fixed_time.extend([seg["time"][0], seg["time"][-1]])
        fixed_thrust.extend([mode.thrust_fixed / 1000 if mode else 0] * 2)
    thrust_ax.plot(fixed_time, fixed_thrust, color="#7045a0", drawstyle="steps-post",
                   linewidth=1.2, label="Large engines (fixed steps)")
    thrust_ax.set_title("Total mean thrust and fixed-engine steps")

    for ax, title, ylabel in zip(time_axes,
                                 ("Altitude", "Moon-fixed speed", "Mean thrust / large-engine steps", "Vehicle mass", "Thrust / body tilt"),
                                 ("Altitude above lunar surface (km)", "Speed (m/s)", "Thrust (kN)", "Mass (kg)", "Angle to local vertical (deg)")):
        ax.set(title=title, xlabel="Time since launch (min)", ylabel=ylabel)
        ax.set_xlim(0, elapsed / 60)
        ax.grid(alpha=0.25)
    altitude_ax.legend(loc="upper right", fontsize=9)
    altitude_ax.text(0.02, 0.95, "Grey: coast | Green: sensing | Dotted: gate", transform=altitude_ax.transAxes,
                     va="top", fontsize=8, color="0.4")
    thrust_ax.axhline(p.engine.T_max / 1000, color="0.4", linestyle=":", linewidth=1, label="Maximum")
    thrust_ax.legend(fontsize=8)
    mass_ax.axhline(p.vehicle.m_dry, color="0.4", linestyle=":", label="Dry mass")
    mass_ax.legend(fontsize=8)
    tilt_ax.axhline(np.degrees(p.limits.td_tilt_max), color="0.4", linestyle=":",
                    label="Touchdown limit (endpoints only)")
    tilt_ax.legend(fontsize=8)

    # Terminal close-up uses the same local height/range definition as the glide-slope constraint.
    ph = sol.phases[f"terminal_{branch}"]
    site = start + ph["X"][0:3, -1]
    site_up = site / np.linalg.norm(site)
    offset = start[:, None] + ph["X"][0:3] - site[:, None]
    height = site_up @ offset
    horizontal = np.linalg.norm(offset - site_up[:, None] * height, axis=0)
    divert_ax.plot(horizontal, height, color=OUTBOUND, linewidth=2, label="Outbound terminal descent")
    divert_ax.scatter(horizontal[0], height[0], marker="*", color="#7045a0", s=80, label="Moving gate")
    divert_ax.scatter(horizontal[-1], height[-1], marker="v", color="#a43636", s=45, label="Landing")
    distance = np.linspace(0, max(horizontal.max(), 1.0) * 1.05, 50)
    divert_ax.plot(distance, np.tan(p.limits.glide_slope) * distance, color="0.4", linestyle=":",
                    label=f"{np.degrees(p.limits.glide_slope):g} deg glide-slope boundary")
    divert_ax.set(title=f"Outbound divert close-up ({branch}, {p.mission.divert_radius:g} m)",
                   xlabel="Horizontal distance to divert site (m)", ylabel="Height above landing-site plane (m)")
    divert_ax.set_ylim(bottom=0)
    divert_ax.grid(alpha=0.25)
    divert_ax.legend(fontsize=8)
    fig.suptitle("Flight profiles | Blue: outbound | Orange: return | Surface stay time ignored", fontsize=13)

    # Camera angles during both optimized observation windows, for both required points.
    from .sensor import image_angles
    sensor_fig, sensor_axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained",
                                           num="Lunar hopper - Hazard camera", clear=True)
    for row, (name, target, terminal, color) in enumerate((
            ("observation", p.mission.r_target, f"terminal_{branch}", OUTBOUND),
            (f"observation2_{branch}", start, f"terminal2_{branch}", RETURN))):
        ph = sol.phases[name]
        times = np.linspace(0, ph["t"], ph["X"].shape[1])
        for point, style, label in ((target, "--", "Original target"),
                                   (start + sol.phases[terminal]["X"][:3, -1], "-", "Actual touchdown")):
            angles = np.array([image_angles(T, p.mission.downrange, point - start - r)[:2]
                               for T, r in zip(ph["U"][:3].T, ph["X"][:3].T)])
            for col in range(2):
                ax = sensor_axes[row, col]
                ax.plot(times, angles[:, col], style, color=color, label=label)
                limit = np.degrees(p.sensor.fov[col]) / 2
                ax.axhline(limit, color="0.5", linestyle=":")
                ax.axhline(-limit, color="0.5", linestyle=":")
                ax.set(xlabel="Time within sensing window (s)", ylabel="Image angle (deg)",
                       title=("Outbound" if row == 0 else "Return") + (" horizontal FOV" if col == 0 else " vertical FOV"))
                ax.grid(alpha=0.25)
        sensor_axes[row, 0].legend(fontsize=8)
    sensor_fig.suptitle(f"Bottom camera | {np.degrees(p.sensor.fov[0]):g} x {np.degrees(p.sensor.fov[1]):g} deg usable FOV | "
                       + ("Pointing enforced" if p.sensor.enforce_pointing else "Pointing disabled: diagnostic angles"))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (output_dir / f"trajectory_3d_{branch}.png", output_dir / f"flight_profiles_{branch}.png",
             output_dir / f"sensor_pointing_{branch}.png")
    for figure, path in zip((fig3d, fig, sensor_fig), paths):
        figure.savefig(path, dpi=160, bbox_inches="tight")
        print(f"Plot saved: {path}")
    if show:
        print("Opening Matplotlib windows; close the figure windows to finish the run.")
        plt.show(block=True)
    else:
        plt.close(fig3d)
        plt.close(fig)
        plt.close(sensor_fig)
    return paths

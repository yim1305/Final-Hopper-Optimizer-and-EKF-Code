"""Static PNG figures for the navigation results (matplotlib, light theme).

Colors follow the reference data-viz palette: categorical slots in fixed order (blue, orange, aqua),
one sequential hue (blue) for the heat map, text in neutral inks, recessive grid.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
REQ = "#52514e"
ARCSEC = np.pi / 180 / 3600

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2.0,
})


def _phase_shading(ax, truth):
    t0 = truth.t[0]
    for name, kind, s0, s1 in truth.phase_bounds:
        if kind == "coast":
            ax.axvspan(s0 - t0, s1 - t0, color="#f0efec", lw=0, zorder=0)


def _checkpoint_lines(ax, truth, label=True):
    t0 = truth.t[0]
    names = {"end_of_ascent_burn": "burnout", "end_of_coast": "coast end", "touchdown": "touchdown"}
    for key, k in truth.checkpoints.items():
        if key == "launch":
            continue
        short = names.get(key, key.replace("descent_", "").replace("m", " m"))
        ax.axvline(truth.t[k] - t0, color=MUTED, lw=0.8, ls=":", zorder=1)
        if label:
            ax.text(truth.t[k] - t0 - 3, 0.97, short, transform=ax.get_xaxis_transform(), rotation=90,
                    ha="right", va="top", fontsize=7.5, color=MUTED)


def error_history(res, truth, path):
    """3 x 3 grid: rows position / velocity / attitude, columns local axes; sample runs + 3-sigma bounds."""
    H = res["hist"]
    t = res["t"]
    rows = [("Position error (m)", "e_r", "P_r", 1.0, ["downrange", "crossrange", "vertical"]),
            ("Velocity error (m/s)", "e_v", "P_v", 1.0, ["downrange", "crossrange", "vertical"]),
            ("Attitude error (arcsec)", "e_th", "P_th", ARCSEC, ["body x", "body y", "body z"])]
    fig, axes = plt.subplots(3, 3, figsize=(13, 8.5), sharex=True)
    for r, (title, ek, pk, scale, axis_names) in enumerate(rows):
        e = H[ek] / scale
        sig = 3 * np.sqrt(np.mean(np.diagonal(H[pk], axis1=2, axis2=3), axis=1)) / scale
        for c in range(3):
            ax = axes[r, c]
            _phase_shading(ax, truth)
            for j in range(min(5, e.shape[1])):
                ax.plot(t, e[:, j, c], color="#b9b8b2", lw=0.8, zorder=2)
            ax.plot(t, sig[:, c], color=SERIES[0], lw=1.5, zorder=3)
            ax.plot(t, -sig[:, c], color=SERIES[0], lw=1.5, zorder=3)
            ax.set_title(f"{title.split(' (')[0]}: {axis_names[c]}", loc="left", fontsize=10)
            if c == 0:
                ax.set_ylabel(title.split("(")[-1].rstrip(")"))
            _checkpoint_lines(ax, truth, label=(r == 0))
    for ax in axes[-1]:
        ax.set_xlabel("Time from launch (s)")
    fig.suptitle(f"Hop {res['hop']} — case {res['case_id']}: {res['label']}", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.text(0.01, 0.945, "Blue: filter ±3σ.  Gray: 5 sample Monte Carlo runs.  Shaded: coast.",
             ha="left", color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def comparison(results_by_hop, truths, requirement_m, path):
    """Horizontal position 3-sigma (Monte Carlo) vs time for cases A, B, C, log scale, both hops."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (hop, results) in zip(axes, sorted(results_by_hop.items())):
        truth = truths[hop]
        _phase_shading(ax, truth)
        finals = {}
        for color, cid in zip(SERIES, ("A", "B", "C")):
            res = results[cid]
            e = res["hist"]["e_r"]
            horiz = 3 * np.sqrt(np.mean(e[:, :, 0]**2 + e[:, :, 1]**2, axis=1))
            ax.plot(res["t"], np.maximum(horiz, 1e-3), color=color, lw=2, label=f"{cid}: {res['label']}")
            finals[cid] = horiz[-1]
        tend = res["t"][-1]
        ax.annotate(f"A {finals['A']:,.0f} m\nB {finals['B']:,.0f} m", (tend, np.sqrt(finals["A"] * finals["B"])),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
        ax.annotate(f"C {finals['C']:,.0f} m", (tend, finals["C"]), xytext=(6, -10), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
        ax.axhline(requirement_m, color=REQ, lw=1.2, ls="--")
        ax.text(tend * 0.5, requirement_m * 0.62, f"landing requirement {requirement_m:.0f} m", color=INK2,
                fontsize=8, ha="center")
        ax.set_yscale("log")
        ax.set_ylim(1e-1, 1e4)
        ax.set_title(f"Hop {hop}", loc="left")
        ax.set_xlabel("Time from launch (s)")
        _checkpoint_lines(ax, truth)
        ax.set_xlim(0, res["t"][-1] * 1.12)
    axes[0].set_ylabel("Horizontal position error, 3σ (m)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Navigation error by sensor set (Monte Carlo 3σ, horizontal)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _worst_axis_3sigma(res, key):
    e = res["hist"][key]
    return 3 * np.sqrt(np.max(np.mean(e**2, axis=1), axis=1))


def star_tracker_value(results_by_hop, truths, requirement_m, path):
    """With (C) vs without (E) star trackers: attitude error and horizontal position error, 3-sigma."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex="col")
    pairs = (("C", SERIES[2]), ("E", SERIES[3]))
    for col, (hop, results) in enumerate(sorted(results_by_hop.items())):
        truth = truths[hop]
        ax = axes[0, col]
        _phase_shading(ax, truth)
        for cid, color in pairs:
            res = results[cid]
            att = _worst_axis_3sigma(res, "e_th") / ARCSEC
            ax.plot(res["t"], att, color=color, lw=2, label=f"{cid}: {res['label']}")
            ax.annotate(f"{cid}  {att[-1]:,.0f}″", (res["t"][-1], att[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9)
        ax.set_yscale("log")
        ax.set_ylim(0.5, 1e4)
        ax.set_title(f"Hop {hop}: attitude error, worst axis (3σ)", loc="left")
        ax.set_ylabel("arcsec")
        _checkpoint_lines(ax, truth)
        ax = axes[1, col]
        _phase_shading(ax, truth)
        for cid, color in pairs:
            res = results[cid]
            e = res["hist"]["e_r"]
            horiz = 3 * np.sqrt(np.mean(e[:, :, 0]**2 + e[:, :, 1]**2, axis=1))
            ax.plot(res["t"], horiz, color=color, lw=2)
            ax.annotate(f"{cid}  {horiz[-1]:,.0f} m", (res["t"][-1], horiz[-1]),
                        xytext=(6, 8 if cid == "E" else -8), textcoords="offset points", va="center", fontsize=9)
        ax.axhline(requirement_m, color=REQ, lw=1.2, ls="--")
        ax.text(res["t"][-1] * 0.5, requirement_m * 1.04, f"landing requirement {requirement_m:.0f} m",
                color=INK2, fontsize=8, ha="center", va="bottom")
        ax.set_title(f"Hop {hop}: horizontal position error (3σ)", loc="left")
        ax.set_ylabel("m")
        ax.set_xlabel("Time from launch (s)")
        ax.set_xlim(0, res["t"][-1] * 1.12)
        _checkpoint_lines(ax, truth, label=False)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("What the star trackers buy: attitude knowledge (Monte Carlo 3σ)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _ellipse(ax, P2, color, scale=3.0):
    vals, vecs = np.linalg.eigh(P2)
    th = np.linspace(0, 2 * np.pi, 200)
    pts = vecs @ (scale * np.sqrt(np.maximum(vals, 0))[:, None] * np.vstack((np.cos(th), np.sin(th))))
    ax.plot(pts[0], pts[1], color=color, lw=1.5, ls="--")


def touchdown_distribution(td, requirement_m, path):
    """td[hop][case] = dict(e=(n,3) local errors at touchdown, P=(n,3,3) filter covariance)."""
    hops = sorted(td)
    fig, axes = plt.subplots(len(hops), 3, figsize=(15, 4.9 * len(hops)))
    circle = np.linspace(0, 2 * np.pi, 300)
    for row, hop in enumerate(hops):
        cases = td[hop]
        # (1) precise cases: scatter + 50 m circle + filter 3-sigma ellipse
        ax = axes[row, 0]
        for cid, color in (("E", SERIES[3]), ("C", SERIES[2])):
            if cid not in cases:
                continue
            e = cases[cid]["e"]
            ax.scatter(e[:, 0], e[:, 1], s=16, color=color, edgecolor=SURFACE, linewidth=0.6, zorder=3,
                       label=f"{cid} ({len(e)} runs)")
            _ellipse(ax, np.mean(cases[cid]["P"][:, :2, :2], axis=0), color)
        ax.plot(requirement_m * np.cos(circle), requirement_m * np.sin(circle), color=REQ, lw=1.2)
        ax.text(0, requirement_m * 1.05, f"{requirement_m:.0f} m requirement", ha="center", va="bottom",
                fontsize=8, color=INK2)
        lim = requirement_m * 2.0
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("Downrange error (m)")
        ax.set_ylabel("Crossrange error (m)")
        ax.set_title(f"Hop {hop}: touchdown nav error, each dot = 1 run", loc="left")
        ax.legend(loc="lower left", fontsize=8, title="dashed: filter 3σ", title_fontsize=8)
        # (2) histogram of horizontal error, case C
        ax = axes[row, 1]
        e = cases["C"]["e"]
        horiz = np.hypot(e[:, 0], e[:, 1])
        bins = np.linspace(0, requirement_m * 1.2, 25)
        ax.hist(horiz, bins=bins, color=SERIES[2], edgecolor=SURFACE, linewidth=1.0)
        ax.axvline(requirement_m, color=REQ, lw=1.2, ls="--")
        p95, mx = np.percentile(horiz, 95), horiz.max()
        ax.text(0.55, 0.95, f"median {np.median(horiz):.1f} m\n95th pct {p95:.1f} m\nworst run {mx:.1f} m\n"
                            f"runs over {requirement_m:.0f} m: {int(np.sum(horiz > requirement_m))}/{len(horiz)}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9, color=INK,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2))
        ax.set_xlabel("Horizontal touchdown nav error (m)")
        ax.set_ylabel("Runs")
        ax.set_title(f"Hop {hop}: case C distribution", loc="left")
        # (3) dead reckoning / INS scatter in km
        ax = axes[row, 2]
        for cid, color in (("A", SERIES[0]), ("B", SERIES[1])):
            e = cases[cid]["e"] / 1000
            ax.scatter(e[:, 0], e[:, 1], s=16, color=color, edgecolor=SURFACE, linewidth=0.6, label=cid)
        r = requirement_m / 1000
        ax.plot(r * np.cos(circle), r * np.sin(circle), color=REQ, lw=1.2)
        ax.set_aspect("equal")
        ax.set_xlabel("Downrange error (km)")
        ax.set_ylabel("Crossrange error (km)")
        ax.set_title(f"Hop {hop}: without TRN (A, B); small circle = 50 m", loc="left")
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("Touchdown navigation error distributions (Monte Carlo)", x=0.01, ha="left", fontsize=13,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def tracker_timeline(truth, avail_info, max_rate_deg_s, path):
    t = truth.t - truth.t[0]
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    ax = axes[0]
    _phase_shading(ax, truth)
    ax.plot(t, np.degrees(truth.body_rate), color=SERIES[0], lw=1.5)
    ax.axhline(max_rate_deg_s, color=REQ, ls="--", lw=1.2)
    ax.text(t[-1] * 0.01, max_rate_deg_s * 1.08, f"tracker limit {max_rate_deg_s:.0f} °/s", color=INK2, fontsize=8)
    ax.set_ylabel("Body rate (°/s)")
    ax.set_title(f"Hop {truth.hop}: star tracker availability", loc="left")
    _checkpoint_lines(ax, truth)
    ax = axes[1]
    for j, color in enumerate(SERIES[:2]):
        ok = avail_info["geometry_ok"][:, j] & avail_info["rate_ok"]
        ax.fill_between(t, j + 0.1, j + 0.9, where=ok, color=color, lw=0, step="post")
    ax.set_yticks([0.5, 1.5], ["head 1", "head 2"])
    ax.set_ylim(0, 2)
    ax.grid(False)
    ax.set_xlabel("Time from launch (s)")
    fig.text(0.01, 0.01, "Bars: head can measure (Sun/Earth/Moon clear and body rate within limit). Shaded: coast.",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sweep_heatmap(grid, sigmas, rates, requirement_m, path):
    """grid[hop] = (len(sigmas), len(rates)) touchdown horizontal 3-sigma (m)."""
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, axes = plt.subplots(1, len(grid), figsize=(6.2 * len(grid), 4.6), gridspec_kw=dict(wspace=0.2))
    axes = np.atleast_1d(axes)
    vmax = max(float(np.max(g)) for g in grid.values())
    for ax, (hop, g) in zip(axes, sorted(grid.items())):
        im = ax.imshow(g, cmap=cmap, origin="lower", aspect="auto", vmin=0, vmax=vmax)
        for i in range(len(sigmas)):
            for j in range(len(rates)):
                ok = g[i, j] <= requirement_m
                ax.text(j, i, f"{g[i, j]:.0f} m\n{'pass' if ok else 'fail'}", ha="center", va="center",
                        fontsize=9, color="#ffffff" if g[i, j] > 0.55 * vmax else INK,
                        fontweight="bold" if ok else "normal")
        ax.set_xticks(range(len(rates)), [f"{r:g} Hz" for r in rates])
        ax.set_yticks(range(len(sigmas)), [f"{s:g} m" for s in sigmas])
        ax.set_xlabel("SPLICE TRN update rate")
        if ax is axes[0]:
            ax.set_ylabel("SPLICE TRN accuracy (1σ per axis)")
        ax.set_title(f"Hop {hop}: touchdown horizontal error, 3σ", loc="left")
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.85, label="m")
    fig.suptitle(f"Derived requirement on SPLICE TRN data (pass = within {requirement_m:.0f} m)", x=0.01,
                 ha="left", fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

"""Render a fixed-camera MP4 of the saved DEM 8-A2200 round trip with launch and landing close-ups.

Run from the repository root:
    .\\.venv\\Scripts\\python.exe -B Animation/animate_hop.py            # full MP4
    .\\.venv\\Scripts\\python.exe -B Animation/animate_hop.py --preview  # PNG stills only
Every plotted sample comes from the saved optimizer trajectory; nothing is re-solved.
Astro is only read: its saved case in Astro/results/ and its mission package.
Output stays local in Animation/results/a2200_08/.
"""

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "Astro"
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from mpl_toolkits.mplot3d import proj3d

from mission.config import parameters
from mission.terrain import DemTerrain

CASE = "a2200_08"
CASE_DIR = ROOT / "results" / CASE
OUT_DIR = HERE / "results" / CASE

# Playback: the whole flight fills FLIGHT_SECONDS of video. Kinds in SCREEN_SECONDS get that
# many seconds per hop (braking includes its engine step-downs); every other phase shares
# the remaining screen time at one rate. Extend playback to give observation time to read.
FLIGHT_SECONDS = 48.0
SCREEN_SECONDS = {"rise": 3.0, "braking": 3.0, "observation": 4.0, "terminal": 4.0}
HOLD_SECONDS = 2.0
FPS = 30
SIZE = (1920, 1080)

# Display choices only, not physics.
MAIN_VERTICAL_EXAGGERATION = 2.0
MAIN_TERRAIN_STRIDE = 8
INSET_GRID = 60
SUN_AZIMUTH_DEG, SUN_ELEVATION_DEG = 300.0, 25.0  # illustrative hillshade, not the mission Sun
MAIN_VIEW = dict(elev=16, azim=-74)
INSET_VIEW = dict(elev=22, azim=-62)

BG = "#07090d"
PANEL = "#10141b"
TEXT = "#e8ecf2"
MUTED = "#8b95a7"
PLUME_GLOW, PLUME_CORE = "#ff8c32", "#fff1c1"
PHASE_COLORS = {
    "rise": "#ffd166",
    "ascent": "#ff9f1c",
    "coast": "#9aa5b8",
    "braking": "#ef476f",
    "observation": "#4cc9f0",
    "terminal": "#06d6a0",
}
PHASE_LABELS = {
    "rise": "Vertical rise",
    "ascent": "Ascent burn",
    "coast": "Ballistic coast",
    "braking": "Braking burn",
    "observation": "Hazard observation",
    "terminal": "Terminal descent",
}
REGOLITH = LinearSegmentedColormap.from_list("regolith", ["#1b1c20", "#5d5c5a", "#b9b5ac"])


class Arrow3D(FancyArrowPatch):
    """Arrow with a 2D head whose endpoints are projected from 3D data each draw."""

    def __init__(self, **kwargs):
        super().__init__((0, 0), (0, 0), **kwargs)
        self.xyz = np.zeros((2, 3))

    def _project(self):
        xs, ys, zs = proj3d.proj_transform(*self.xyz.T, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

    def draw(self, renderer):
        self._project()
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        return self._project()


def load_flight():
    data = np.load(CASE_DIR / "trajectory.npz")
    meta = json.loads(str(data["metadata"]))
    with open(CASE_DIR / "phases.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    phases = []
    for row in rows:
        name = row["phase"]
        info = meta["phase_info"][name]
        X, U = data[f"{name}_X"], data[f"{name}_U"]
        t0, t1 = float(row["t_start"]), float(row["t_end"])
        phases.append(dict(name=name, kind=info["kind"], hop=info["hop"], stage=info["stage"],
                           engines=ast.literal_eval(row["engine_counts"]).get("a2200", 0),
                           t0=t0, t1=t1, t=np.linspace(t0, t1, X.shape[1]), X=X, U=U))
    return phases, meta["branch"]


def playback(phases):
    real = np.array([ph["t1"] - ph["t0"] for ph in phases])
    rates = np.full(len(phases), np.nan)
    for i, ph in enumerate(phases):
        if ph["kind"] in SCREEN_SECONDS:
            group = sum(r for r, q in zip(real, phases) if (q["hop"], q["kind"]) == (ph["hop"], ph["kind"]))
            rates[i] = group / SCREEN_SECONDS[ph["kind"]]
    fixed = ~np.isnan(rates)
    remaining = FLIGHT_SECONDS - (real[fixed] / rates[fixed]).sum()
    if remaining <= 0:
        raise ValueError("SCREEN_SECONDS leaves no time for the other phases")
    fast_rate = real[~fixed].sum() / remaining
    rates[~fixed] = fast_rate
    uniform = real.sum() / FLIGHT_SECONDS
    return rates, uniform, fast_rate


def frame_times(phases, rates):
    """Mission time and phase index of every video frame, ending with a held final frame."""
    edges = np.concatenate(([0.0], np.cumsum([(ph["t1"] - ph["t0"]) / r for ph, r in zip(phases, rates)])))
    screen = np.arange(int(round(FLIGHT_SECONDS * FPS)) + 1) / FPS
    k = np.clip(np.searchsorted(edges, screen, side="right") - 1, 0, len(phases) - 1)
    t1 = np.array([phases[i]["t1"] for i in k])
    t = np.minimum(np.array([phases[i]["t0"] for i in k]) + (screen - edges[k]) * rates[k], t1)
    hold = int(round(HOLD_SECONDS * FPS))
    return np.concatenate((t, np.full(hold, t[-1]))), np.concatenate((k, np.full(hold, k[-1])))


def state_at(ph, t):
    return (np.array([np.interp(t, ph["t"], row) for row in ph["X"]]),
            np.array([np.interp(t, ph["t"], row) for row in ph["U"]]))


def dem_to_moon(terrain, rows, cols):
    """Moon-fixed positions of DEM pixel centres; the inverse of DemTerrain.projected."""
    R, tr = terrain.radius, terrain.transform
    east = tr.c + (cols + 0.5) * tr.a
    north = tr.f + (rows + 0.5) * tr.e
    rho2 = east ** 2 + north ** 2
    z = R * (rho2 - 4 * R * R) / (rho2 + 4 * R * R)
    f = 2 * R / (R - z)
    unit = np.stack((north / f, east / f, z), axis=-1) / R
    return unit * (R + terrain.fraction * terrain.dem[rows, cols])[..., None]


def local_frame(origin, along):
    up = origin / np.linalg.norm(origin)
    a = along - (along @ up) * up
    a /= np.linalg.norm(a)
    return np.vstack((a, np.cross(up, a), up))


def shaded(dem, spacing, vert_exag=1.0):
    ls = LightSource(azdeg=SUN_AZIMUTH_DEG, altdeg=SUN_ELEVATION_DEG)
    return ls.shade(dem, cmap=REGOLITH, blend_mode="soft", vert_exag=vert_exag, dx=spacing, dy=spacing)


def style_3d(ax):
    ax.set_facecolor(BG)
    ax.set_axis_off()
    ax.computed_zorder = False


def set_points(line, pts):
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    line.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])


class Inset:
    """Close-up of one site in its own local tangent frame, in metres and at true scale."""

    def __init__(self, fig, rect, title, terrain, center, along, flight, half_width, top):
        self.center, self.F = center, local_frame(center, along)
        self.H, self.top = half_width, top
        self.ax = fig.add_axes(rect, projection="3d")
        style_3d(self.ax)
        # Surface sampled through DemTerrain's bilinear interpolation, the optimizer's terrain model.
        step = 2 * half_width / INSET_GRID
        g = np.linspace(-half_width - step, half_width + step, INSET_GRID + 3)
        gx, gy = np.meshgrid(g, g, indexing="ij")
        flat = center + np.stack((gx, gy, np.zeros_like(gx)), axis=-1) @ self.F
        unit = flat / np.linalg.norm(flat, axis=-1, keepdims=True)
        height = terrain.height_positions(unit * terrain.radius)
        xyz = self.local(unit * (terrain.radius + height)[..., None])
        self.ground = float(self.local(center[None])[0, 2])
        self.ax.plot_surface(*xyz.transpose(2, 0, 1), facecolors=shaded(height, step),
                             rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
        bottom = xyz[..., 2].min()
        self.ax.set(xlim=(-half_width, half_width), ylim=(-half_width, half_width), zlim=(bottom, top))
        self.ax.set_box_aspect((2 * half_width, 2 * half_width, top - bottom), zoom=1.25)
        self.ax.view_init(**INSET_VIEW)
        for seg in flight:
            p = self.local(seg["r"])
            self.ax.plot(*p.T, color=TEXT, alpha=0.18, linewidth=1.0, axlim_clip=True,
                         linestyle="--" if seg["kind"] == "coast" else "-")
        # Scale bar along the near edge.
        bar = 100.0
        y0 = -0.85 * half_width
        self.ax.plot([-0.85 * half_width, -0.85 * half_width + bar], [y0, y0], [self.ground] * 2,
                     color=TEXT, linewidth=2)
        self.ax.text(-0.85 * half_width + bar / 2, y0, self.ground + 0.04 * (top - bottom), "100 m",
                     color=TEXT, fontsize=10, ha="center")
        x0, y0, w, h = rect
        self.frame = Rectangle((x0, y0), w, h, transform=fig.transFigure, fill=False,
                               linewidth=1.5, edgecolor="#2a3140", animated=True)
        fig.add_artist(self.frame)
        fig.add_artist(Rectangle((x0, y0), w, h, transform=fig.transFigure, fill=False,
                                 linewidth=1.0, edgecolor="#2a3140"))
        self.title = fig.text(x0 + 0.012, y0 + h - 0.035, title, color=TEXT, fontsize=15,
                              fontweight="bold", animated=True)
        fig.text(x0 + 0.012, y0 + h - 0.06, f"True scale  |  box ±{half_width:.0f} m", color=MUTED, fontsize=10)
        self.phase = fig.text(x0 + 0.012, y0 + h - 0.085, "", fontsize=11,
                              fontweight="bold", animated=True)

    def local(self, r):
        return (np.asarray(r) - self.center) @ self.F.T

    def contains(self, r):
        p = self.local(r[None])[0]
        return abs(p[0]) <= self.H and abs(p[1]) <= self.H and p[2] <= self.top

    def mark(self, r, marker, color, label, size=11, side=0):
        """Site marker; side -1/0/+1 puts the label left, centred or right of it."""
        p = self.local(r[None])[0]
        self.ax.plot(*[[v] for v in p], marker=marker, color=color, markersize=size,
                     markeredgecolor=BG, linestyle="none", zorder=5)
        shift = 0.12 * self.H * side
        self.ax.text(p[0] + shift, p[1] - shift, p[2] + 0.08 * (self.top - self.ground), label, color=color,
                     fontsize=10, ha=("right", "center", "left")[side + 1], fontweight="bold", zorder=6)

    def tolerance_ring(self, r, radius):
        p = self.local(r[None])[0]
        a = np.linspace(0, 2 * np.pi, 120)
        self.ax.plot(p[0] + radius * np.cos(a), p[1] + radius * np.sin(a), np.full(a.size, p[2] + 0.5),
                     color=PHASE_COLORS["terminal"], linewidth=1.6, zorder=4)


def build(phases, branch):
    terrain = DemTerrain()
    p = parameters(CASE, terrain)
    start = p.mission.r_start
    along_vec = p.mission.r_target - start
    F = local_frame(start, along_vec)
    landing_site = p.mission.sites[branch]
    touchdown_out = start + next(ph for ph in phases if ph["kind"] == "terminal" and ph["hop"] == 1)["X"][:3, -1]
    touchdown_ret = start + phases[-1]["X"][:3, -1]
    rated = p.engine.classes["a2200"].thrust

    rates, uniform, fast_rate = playback(phases)
    t_frames, k_frames = frame_times(phases, rates)
    states = [state_at(phases[k], t) for t, k in zip(t_frames, k_frames)]
    X = np.array([s[0] for s in states])
    U = np.array([s[1] for s in states])
    r_abs = start + X[:, :3]
    agl = terrain.altitude(r_abs)
    flight = [dict(kind=ph["kind"], r=start + ph["X"][:3].T) for ph in phases]

    fig = plt.figure(figsize=(SIZE[0] / 100, SIZE[1] / 100), dpi=100, facecolor=BG)
    plt.rcParams["font.family"] = "DejaVu Sans"

    # ---- main fixed view, km in the launch tangent frame ----
    ax = fig.add_axes([-0.03, 0.03, 0.78, 0.86], projection="3d")
    style_3d(ax)
    S = MAIN_TERRAIN_STRIDE
    rows, cols = np.meshgrid(np.arange(0, terrain.dem.shape[0], S), np.arange(0, terrain.dem.shape[1], S),
                             indexing="ij")
    km = lambda r: (np.asarray(r) - start) @ F.T / 1000.0
    T = km(dem_to_moon(terrain, rows, cols))
    ax.plot_surface(*T.transpose(2, 0, 1),
                    facecolors=shaded(terrain.dem[rows, cols], 30.0 * S, MAIN_VERTICAL_EXAGGERATION),
                    rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    path = km(np.vstack([seg["r"] for seg in flight]))
    lo = np.minimum(T.reshape(-1, 3).min(0), path.min(0))
    hi = np.maximum(T.reshape(-1, 3).max(0), path.max(0))
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2] * 1.05))
    ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], (hi[2] * 1.05 - lo[2]) * MAIN_VERTICAL_EXAGGERATION),
                      zoom=1.55)
    ax.view_init(**MAIN_VIEW)
    for seg in flight:
        ax.plot(*km(seg["r"]).T, color=TEXT, alpha=0.16, linewidth=1.0,
                linestyle="--" if seg["kind"] == "coast" else "-")
    for r, label, color, above in ((start, "Launch / return base", TEXT, True),
                                   (touchdown_out, "Drygalski landing site", PHASE_COLORS["terminal"], False)):
        q = km(r[None])[0]
        ax.plot([q[0]], [q[1]], [q[2]], marker="o", markersize=9, color=color, markeredgecolor=BG,
                linestyle="none")
        if above:
            ax.plot([q[0], q[0]], [q[1], q[1]], [q[2], q[2] + 6], color=color, linewidth=1, alpha=0.7)
        ax.text(q[0], q[1], q[2] + (7 if above else -2.5), label, color=color, fontsize=13,
                ha="center" if above else "right", va="bottom" if above else "top", fontweight="bold")
    bar_y = lo[1] + 0.1 * (hi[1] - lo[1])
    bar_z = km(start[None])[0, 2]
    ax.plot([20, 70], [bar_y, bar_y], [bar_z, bar_z], color=TEXT, linewidth=2)
    ax.text(45, bar_y, bar_z + 2.5, "50 km", color=TEXT, fontsize=11, ha="center")

    # ---- site insets ----
    def box_for(hop, site):
        pts = [start + ph["X"][:3, 0] for ph in phases if ph["kind"] == "terminal" and ph["hop"] == hop]
        Fs = local_frame(site, along_vec)
        loc = (np.array(pts) - site) @ Fs.T
        H = max(250.0, np.ceil(1.15 * np.abs(loc[:, :2]).max() / 50) * 50)
        top = max(150.0, np.ceil(1.25 * loc[:, 2].max() / 50) * 50)
        # Keep the existing zoom when it fits; also include every observation/descent sample.
        nearby = np.vstack([start + ph["X"][:3].T for ph in phases
                            if ph["hop"] == hop and ph["kind"] in ("observation", "terminal")])
        loc = (nearby - site) @ Fs.T
        H = max(H, np.ceil(np.abs(loc[:, :2]).max() / 50) * 50)
        top = max(top, np.ceil(loc[:, 2].max() / 50) * 50)
        return H, top

    H_launch, top_launch = box_for(2, start)
    H_land, top_land = box_for(1, landing_site)
    inset_land = Inset(fig, [0.715, 0.515, 0.27, 0.42], "Drygalski landing site", terrain, landing_site,
                       along_vec, flight, H_land, top_land)
    inset_launch = Inset(fig, [0.715, 0.075, 0.27, 0.42], "Launch / return base", terrain, start,
                         along_vec, flight, H_launch, top_launch)
    inset_land.tolerance_ring(landing_site, p.mission.landing_tolerance)
    inset_land.mark(p.mission.r_target, "x", MUTED, "Original target")
    inset_land.mark(touchdown_out, "v", PHASE_COLORS["terminal"], "Touchdown", side=-1)
    inset_launch.tolerance_ring(start, p.mission.landing_tolerance)
    inset_launch.mark(start, "^", TEXT, "Launch pad", side=-1)
    inset_launch.mark(touchdown_ret, "v", PHASE_COLORS["terminal"], "Return touchdown", side=1)
    fig.text(0.72, 0.045, f"Green ring: {p.mission.landing_tolerance:g} m landing tolerance",
                   color=MUTED, fontsize=10)

    # ---- static text, legend and timeline ----
    fig.text(0.02, 0.945, "Hopper1  |  DEM case: 8 × A2200 round trip", color=TEXT, fontsize=24, fontweight="bold")
    fig.text(0.02, 0.915, f"Optimizer trajectory over the LOLA 30 m DEM  |  fixed camera  |  "
                          f"vertical scale ×{MAIN_VERTICAL_EXAGGERATION:g} in main view",
             color=MUTED, fontsize=12)
    handles = [Patch(color=c, label=PHASE_LABELS[k]) for k, c in PHASE_COLORS.items()]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.075), ncol=6, frameon=False,
               labelcolor=TEXT, fontsize=11, handlelength=1.2, columnspacing=1.4)

    tl = fig.add_axes([0.03, 0.035, 0.66, 0.022])
    tl.set_facecolor(PANEL)
    for ph in phases:
        tl.axvspan(ph["t0"], ph["t1"], color=PHASE_COLORS[ph["kind"]], alpha=0.85 if ph["kind"] != "coast" else 0.35,
                   linewidth=0)
    tl.set_xlim(0, phases[-1]["t1"])
    tl.set_yticks([])
    tl.tick_params(colors=MUTED, labelsize=9, length=2)
    tl.set_xticks(np.arange(0, phases[-1]["t1"], 120), [f"{int(m)} min" for m in np.arange(0, phases[-1]["t1"], 120) / 60])
    for s in tl.spines.values():
        s.set_visible(False)
    hop2 = next(ph["t0"] for ph in phases if ph["hop"] == 2)
    tl.text(hop2 / 2, 1.25, "OUTBOUND", transform=tl.get_xaxis_transform(), color=MUTED, fontsize=10, ha="center")
    tl.text((hop2 + phases[-1]["t1"]) / 2, 1.25, "RETURN", transform=tl.get_xaxis_transform(), color=MUTED,
            fontsize=10, ha="center")
    tl.axvline(hop2, color=TEXT, linewidth=1, alpha=0.6)
    cursor = tl.axvline(0, color=TEXT, linewidth=2.5, animated=True)

    fig.add_artist(FancyBboxPatch((0.02, 0.62), 0.268, 0.27, boxstyle="round,pad=0.006,rounding_size=0.01",
                                  transform=fig.transFigure, facecolor=PANEL, edgecolor="#2a3140", alpha=0.92))
    clock = fig.text(0.032, 0.85, "", color=TEXT, fontsize=26, fontweight="bold", family="DejaVu Sans Mono",
                     animated=True)
    phase_txt = fig.text(0.032, 0.815, "", fontsize=14, fontweight="bold", animated=True)
    stage_txt = fig.text(0.032, 0.792, "", color=MUTED, fontsize=11, animated=True)
    metrics = fig.text(0.032, 0.69, "", color=TEXT, fontsize=12, family="DejaVu Sans Mono", linespacing=1.5,
                       animated=True)
    fig.text(0.032, 0.655, "Engines lit", color=MUTED, fontsize=11)
    dots = [Circle((0.098 + 0.0125 * i, 0.661), 0.0045, transform=fig.transFigure, animated=True)
            for i in range(8)]
    for d in dots:
        fig.add_artist(d)
    speed_txt = fig.text(0.032, 0.628, "", fontsize=11, fontweight="bold", animated=True)

    # ---- dynamic 3D artists ----
    views = [(ax, km, 9000.0, False), (inset_land.ax, inset_land.local, 0.22 * H_land, True),
             (inset_launch.ax, inset_launch.local, 0.22 * H_launch, True)]
    dyn = []
    for a, to, length, clip in views:
        trails = [a.plot([], [], [], color=PHASE_COLORS[ph["kind"]], linewidth=3.0 if not clip else 2.6,
                         linestyle=(0, (3, 2)) if ph["kind"] == "coast" else "-", animated=True,
                         axlim_clip=clip)[0] for ph in phases]
        glow = a.plot([], [], [], color=PLUME_GLOW, linewidth=10, alpha=0.45, solid_capstyle="round",
                      animated=True, axlim_clip=clip)[0]
        core = a.plot([], [], [], color=PLUME_CORE, linewidth=4, solid_capstyle="round", animated=True,
                      axlim_clip=clip)[0]
        arrow = Arrow3D(arrowstyle="-|>", mutation_scale=18, color=TEXT, linewidth=2, animated=True)
        a.add_artist(arrow)
        vehicle = a.plot([], [], [], marker="o", markersize=9, color="white", markeredgecolor=BG,
                         linestyle="none", animated=True, axlim_clip=clip)[0]
        dyn.append(dict(ax=a, to=to, length=length, clip=clip, trails=trails, glow=glow, core=core,
                        arrow=arrow, vehicle=vehicle, local=[to(seg["r"]) for seg in flight]))
    # Matplotlib shrinks 3D axes to a square box; let the wide main scene draw past it.
    for artist in ax.get_children():
        artist.set_clip_on(False)

    n_flight = len(t_frames) - int(round(HOLD_SECONDS * FPS))

    def update(i):
        k, t = k_frames[i], t_frames[i]
        ph = phases[k]
        r, u = r_abs[i], U[i]
        landed = i >= n_flight
        thrust = np.linalg.norm(u[:3]) if ph["engines"] and not landed else 0.0
        height = max(agl[i], 0.0)
        direction = u[:3] / thrust if thrust > 0 else np.zeros(3)
        throttle = thrust / (ph["engines"] * rated) if ph["engines"] else 0.0
        for d in dyn:
            for j, line in enumerate(d["trails"]):
                if j < k:
                    pts = d["local"][j]
                elif j == k:
                    keep = phases[j]["t"] < t
                    pts = np.vstack((d["local"][j][keep], d["to"](r[None])))
                else:
                    pts = np.empty((0, 3))
                set_points(line, pts)
            p0 = d["to"](r[None])[0]
            inside = (not d["clip"]) or (d is dyn[1] and inset_land.contains(r)) or \
                     (d is dyn[2] and inset_launch.contains(r))
            set_points(d["vehicle"], p0)
            firing = thrust > 0 and inside
            if firing:
                # Arrow along thrust, plume opposite; both scale with throttle (lengths in metres)
                # and the plume stops at the ground.
                tip = d["to"]((r + direction * d["length"] * throttle)[None])[0]
                plume = min(0.6 * d["length"] * throttle, 0.9 * height)
                tail = d["to"]((r - direction * plume)[None])[0]
                d["arrow"].xyz = np.vstack((p0, tip))
                set_points(d["glow"], np.vstack((p0, tail)))
                set_points(d["core"], np.vstack((p0, p0 + 0.6 * (tail - p0))))
            else:
                set_points(d["glow"], np.empty((0, 3)))
                set_points(d["core"], np.empty((0, 3)))
            d["arrow"].set_visible(firing)
        for hop, inset in ((1, inset_land), (2, inset_launch)):
            active = inset.contains(r)
            inset.frame.set_edgecolor(PHASE_COLORS[ph["kind"]] if active else "#2a3140")
            inset.frame.set_linewidth(3.0 if active else 1.5)
            inset.title.set_color(PHASE_COLORS[ph["kind"]] if active else TEXT)
            show_phase = active and ph["hop"] == hop and ph["kind"] in ("observation", "terminal")
            inset.phase.set_text(PHASE_LABELS[ph["kind"]] if show_phase else "")
            inset.phase.set_color(PHASE_COLORS[ph["kind"]])
        minutes, seconds = divmod(t, 60)
        clock.set_text(f"T+ {int(minutes):02d}:{seconds:04.1f}")
        phase_txt.set_text(f"{'OUTBOUND' if ph['hop'] == 1 else 'RETURN'}  ·  {PHASE_LABELS[ph['kind']].upper()}")
        phase_txt.set_color(PHASE_COLORS[ph["kind"]])
        stage_txt.set_text(f"Engine step-down to {ph['engines']} engines" if ph["stage"] else "")
        m0 = phases[0]["X"][6, 0]
        altitude = f"{height / 1000:8.2f} km" if height >= 1000 else f"{height:8.0f} m"
        metrics.set_text(f"Altitude AGL {altitude}\n"
                         f"Speed        {np.linalg.norm(X[i, 3:6]):8.0f} m/s\n"
                         f"Thrust       {thrust / 1000:8.2f} kN ({throttle * 100:3.0f}%)\n"
                         f"Prop. used   {m0 - X[i, 6]:8.0f} kg")
        for j, dot in enumerate(dots):
            dot.set_facecolor(PLUME_GLOW if j < ph["engines"] and thrust > 0 else "#2a3140")
        rate = rates[k]
        if landed:
            speed_txt.set_text("Flight complete")
            speed_txt.set_color(TEXT)
        elif rate < uniform:
            speed_txt.set_text(f"▶ {rate:.{1 if rate < 10 else 0}f}× real time  ·  "
                               f"slowed {100 * (1 - rate / uniform):.0f}% for detail")
            speed_txt.set_color(PHASE_COLORS["rise"])
        else:
            speed_txt.set_text(f"▶ {rate:.0f}× real time")
            speed_txt.set_color(MUTED)
        cursor.set_xdata([t, t])

    animated = [clock, phase_txt, stage_txt, metrics, speed_txt, cursor, *dots,
                inset_land.frame, inset_land.title, inset_land.phase,
                inset_launch.frame, inset_launch.title, inset_launch.phase]
    for d in dyn:
        animated += [*d["trails"], d["glow"], d["core"], d["arrow"], d["vehicle"]]

    summary = dict(flight_seconds=FLIGHT_SECONDS, hold_seconds=HOLD_SECONDS, fps=FPS,
                   mission_seconds=phases[-1]["t1"], uniform_rate=uniform,
                   screen_seconds_per_hop=SCREEN_SECONDS,
                   fill_rate=fast_rate,
                   phases=[dict(name=ph["name"], hop=ph["hop"], kind=ph["kind"], real_s=ph["t1"] - ph["t0"],
                                rate=float(r), screen_s=(ph["t1"] - ph["t0"]) / r,
                                percent_vs_uniform=100 * (r / uniform - 1))
                           for ph, r in zip(phases, rates)])
    return fig, update, animated, len(t_frames), k_frames, summary


def render(preview):
    phases, branch = load_flight()
    fig, update, animated, n, k_frames, summary = build(phases, branch)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "playback.json").write_text(json.dumps(summary, indent=2))
    print(f"Mission {summary['mission_seconds']:.1f} s -> {FLIGHT_SECONDS:g} s video "
          f"(uniform would be {summary['uniform_rate']:.1f}x real time)")
    groups = {}
    for row in summary["phases"]:
        g = groups.setdefault((row["hop"], row["kind"]), [0.0, 0.0])
        g[0] += row["real_s"]
        g[1] += row["screen_s"]
    print(f"{'hop':<9}{'section':<13}{'real s':>8}{'screen s':>10}{'rate':>8}{'vs uniform':>12}")
    for (hop, kind), (real_s, screen_s) in groups.items():
        rate = real_s / screen_s
        print(f"{'outbound' if hop == 1 else 'return':<9}{kind:<13}{real_s:8.1f}{screen_s:10.2f}"
              f"{rate:7.1f}x{100 * (rate / summary['uniform_rate'] - 1):+11.0f}%")

    canvas = fig.canvas
    canvas.draw()
    background = canvas.copy_from_bbox(fig.bbox)

    def frame(i):
        canvas.restore_region(background)
        update(i)
        for artist in animated:
            if artist.get_visible():
                fig.draw_artist(artist)
        return np.asarray(canvas.buffer_rgba())[:, :, :3]

    if preview:
        picks = {}
        for i, k in enumerate(k_frames):
            picks.setdefault(phases[k]["name"], []).append(i)
        wanted = ["rise", "ascent", "coast", "braking", "observation", "terminal_uprange",
                  "ascent2_uprange", "observation2_uprange", "terminal2_uprange"]
        for name in wanted:
            idx = picks[name][len(picks[name]) // 2] if name != "terminal2_uprange" else picks[name][-1]
            plt.imsave(OUT_DIR / f"preview_{name}.png", frame(idx))
            print(f"Preview saved: {OUT_DIR / f'preview_{name}.png'}")
        plt.imsave(OUT_DIR / "poster.png", frame(n - 1))
        return

    import imageio_ffmpeg
    path = OUT_DIR / "hop_animation.mp4"
    writer = imageio_ffmpeg.write_frames(str(path), SIZE, fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
                                         quality=9, macro_block_size=8)
    writer.send(None)
    for i in range(n):
        writer.send(np.ascontiguousarray(frame(i)))
        if i % (5 * FPS) == 0:
            print(f"  frame {i}/{n}")
    writer.close()
    plt.imsave(OUT_DIR / "poster.png", frame(n - 1))
    print(f"Video saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preview", action="store_true", help="save PNG stills instead of the MP4")
    render(parser.parse_args().preview)

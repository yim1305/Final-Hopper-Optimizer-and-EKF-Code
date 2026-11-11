"""Outbound hop of the saved 8-A2200 DEM flight with a moving camera (zoom in at launch, out for the
coast, in again at the landing site). Only the phase name is shown. Same look as animate_hop.py.

Run from the repository root:
    .\\.venv\\Scripts\\python.exe -B Animation/animate_phases.py [--preview]
Output stays local in Animation/results/a2200_08_phases/.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import animate_hop as A  # reuse loaders, colours and terrain helpers
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FPS, SIZE = 30, (1920, 1080)
SCREEN = {"rise": 3.0, "ascent": 5.5, "coast": 5.0, "braking": 4.5, "observation": 3.5, "terminal": 4.5}
HOLD = 2.0
EXAG = A.MAIN_VERTICAL_EXAGGERATION
VIEW = dict(elev=16, azim=-74)
S_IN = 0.007           # zoomed-in scale (fraction of the full scene box)
ORDER = ["rise", "ascent", "coast", "braking", "observation", "terminal"]


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def main(preview):
    phases, branch = A.load_flight()
    hop1 = [ph for ph in phases if ph["hop"] == 1]
    terrain = A.DemTerrain()
    p = A.parameters(A.CASE, terrain)
    start = p.mission.r_start
    along = p.mission.r_target - start
    F = A.local_frame(start, along)
    km = lambda r: (np.asarray(r) - start) @ F.T / 1000.0
    site = p.mission.sites[branch]

    # ---------------- frame schedule (outbound hop only)
    kinds = [ph["kind"] for ph in hop1]
    real_kind = {k: sum(ph["t1"] - ph["t0"] for ph in hop1 if ph["kind"] == k) for k in ORDER}
    t_list, k_list, tau = [], [], []
    clock = 0.0
    for i, ph in enumerate(hop1):
        n = (ph["t1"] - ph["t0"]) / real_kind[ph["kind"]] * SCREEN[ph["kind"]] * FPS
        m = max(1, int(round(n)))
        for j in range(m):
            t_list.append(ph["t0"] + (j + 1) / m * (ph["t1"] - ph["t0"])); k_list.append(i)
    n_fly = len(t_list)
    t_list += [t_list[-1]] * int(HOLD * FPS); k_list += [k_list[-1]] * int(HOLD * FPS)
    states = [A.state_at(hop1[k], t) for t, k in zip(t_list, k_list)]
    R = np.array([start + s[0][:3] for s in states]); Uc = np.array([s[1] for s in states])
    P = km(R)

    # ---------------- figure and terrain (coarse corridor + native patches at both sites)
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(SIZE[0] / 100, SIZE[1] / 100), dpi=100, facecolor=A.BG)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d", proj_type="ortho")
    A.style_3d(ax)

    def surface(r0, r1, c0, c1, stride, lift=0.0):
        rows, cols = np.meshgrid(np.arange(r0, r1, stride), np.arange(c0, c1, stride), indexing="ij")
        T = km(A.dem_to_moon(terrain, rows, cols)); T[..., 2] += lift
        return ax.plot_surface(*T.transpose(2, 0, 1), facecolors=A.shaded(terrain.dem[rows, cols], 30.0 * stride, EXAG),
                               rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    nr, nc = terrain.dem.shape
    surface(0, nr, 0, nc, A.MAIN_TERRAIN_STRIDE)
    tr = terrain.transform

    def rc_of(r):  # DEM row/col of a Moon-fixed point (inverse of dem_to_moon, stereographic)
        u = r / np.linalg.norm(r); Rm = terrain.radius
        f = 2 * Rm / (Rm - u[2] * Rm)
        north, east = u[0] * Rm * f, u[1] * Rm * f
        return int((north - tr.f) / tr.e - 0.5), int((east - tr.c) / tr.a - 0.5)
    for r in (start, site):
        rr, cc = rc_of(r)
        h = 110  # +-3.3 km at native 30 m
        surface(max(rr - h, 0), min(rr + h, nr), max(cc - h, 0), min(cc + h, nc), 1, lift=0.002)

    full = km(np.vstack([A.dem_to_moon(terrain, *np.meshgrid(np.arange(0, nr, 40), np.arange(0, nc, 40), indexing="ij")).reshape(-1, 3),
                         start + np.hstack([ph["X"][:3] for ph in hop1]).T]))
    lo, hi = full.min(0), full.max(0); hi[2] *= 1.05
    span = hi - lo; mid = (lo + hi) / 2
    ax.set_box_aspect((span[0], span[1], span[2] * EXAG), zoom=1.5)
    ax.view_init(**VIEW)

    # static: faint full path, site markers
    for ph in hop1:
        ax.plot(*km(start + ph["X"][:3].T).T, color=A.TEXT, alpha=0.15, lw=1.0, ls="--" if ph["kind"] == "coast" else "-")
    for r, c in ((start, A.TEXT), (site, A.PHASE_COLORS["terminal"])):
        q = km(r[None])[0]
        ax.plot([q[0]], [q[1]], [q[2]], marker="o", ms=8, color=c, mec=A.BG, ls="none")
    trails = [ax.plot([], [], [], color=A.PHASE_COLORS[ph["kind"]], lw=3.2, ls=(0, (3, 2)) if ph["kind"] == "coast" else "-")[0]
              for ph in hop1]
    glow = ax.plot([], [], [], color=A.PLUME_GLOW, lw=10, alpha=0.45, solid_capstyle="round")[0]
    core = ax.plot([], [], [], color=A.PLUME_CORE, lw=4, solid_capstyle="round")[0]
    veh = ax.plot([], [], [], marker="o", ms=10, color="white", mec=A.BG, ls="none")[0]
    for a in ax.get_children():
        a.set_clip_on(False)

    # phase label + phase strip (only information shown)
    label = fig.text(0.04, 0.88, "", fontsize=44, fontweight="bold")
    chips = []
    for j, k in enumerate(ORDER):
        x0 = 0.04 + j * 0.155
        box = FancyBboxPatch((x0, 0.035), 0.145, 0.055, boxstyle="round,pad=0.004,rounding_size=0.01",
                             transform=fig.transFigure, fc=A.PANEL, ec=A.PHASE_COLORS[k], lw=1.5)
        fig.add_artist(box)
        txt = fig.text(x0 + 0.0725, 0.0625, A.PHASE_LABELS[k], ha="center", va="center", fontsize=16,
                       fontweight="bold", color=A.MUTED)
        chips.append((box, txt, k))

    # camera: log-zoom keyframes in screen time, centre blends from vehicle to scene middle
    edges = np.cumsum([0] + [SCREEN[k] for k in ORDER])
    E = dict(zip(ORDER, edges[:-1]))
    keys = [(0.0, S_IN), (E["ascent"] + 0.4, S_IN), (E["coast"] + 0.6, 1.0), (E["braking"], 1.0),
            (E["observation"] - 0.4, 0.06), (E["observation"] + 1.0, S_IN), (edges[-1] + HOLD, S_IN * 0.9)]
    kt = np.array([k[0] for k in keys]); ks = np.log(np.array([k[1] for k in keys]))

    def camera(tau):
        j = np.clip(np.searchsorted(kt, tau, side="right") - 1, 0, len(kt) - 2)
        w = smooth((tau - kt[j]) / (kt[j + 1] - kt[j]))
        return float(np.exp(ks[j] + (ks[j + 1] - ks[j]) * w))

    def draw(i):
        tau = i / FPS
        s = camera(tau)
        a = np.clip((s - S_IN) / (1 - S_IN), 0, 1) ** 1.3
        c = (1 - a) * P[i] + a * mid
        half = span * s / 2
        ax.set(xlim=(c[0] - half[0], c[0] + half[0]), ylim=(c[1] - half[1], c[1] + half[1]),
               zlim=(c[2] - half[2], c[2] + half[2]))
        k = k_list[i]; ph = hop1[k]; t = t_list[i]
        for j, line in enumerate(trails):
            if j < k:
                pts = km(start + hop1[j]["X"][:3].T)
            elif j == k:
                keep = hop1[j]["t"] < t
                pts = np.vstack((km(start + hop1[j]["X"][:3, keep].T), P[i][None]))
            else:
                pts = np.empty((0, 3))
            line.set_data_3d(*pts.T)
        veh.set_data_3d([P[i][0]], [P[i][1]], [P[i][2]])
        u = Uc[i][:3]; th = np.linalg.norm(u)
        landed = i >= n_fly
        if th > 0 and ph["engines"] and not landed:
            d = (u / th) @ F.T
            L = 0.035 * span[0] * s
            tail = P[i] - d * L
            glow.set_data_3d(*np.vstack((P[i], tail)).T); core.set_data_3d(*np.vstack((P[i], P[i] + 0.6 * (tail - P[i]))).T)
        else:
            glow.set_data_3d([], [], []); core.set_data_3d([], [], [])
        kind = ph["kind"] if not landed else "terminal"
        label.set_text("Touchdown" if landed else A.PHASE_LABELS[kind]); label.set_color(A.PHASE_COLORS[kind])
        for box, txt, kk in chips:
            on = kk == kind
            box.set_facecolor(A.PHASE_COLORS[kk] if on else A.PANEL)
            txt.set_color(A.BG if on else (A.TEXT if ORDER.index(kk) < ORDER.index(kind) else A.MUTED))
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3]

    out = HERE / "results" / "a2200_08_phases"; out.mkdir(parents=True, exist_ok=True)
    n = len(t_list)
    if preview:
        picks = [0, int(E["ascent"] * FPS) + 30, int(E["coast"] * FPS) + 60, int(E["braking"] * FPS) + 60,
                 int(E["observation"] * FPS) + 50, int(E["terminal"] * FPS) + 60, n - 1]
        for q in picks:
            plt.imsave(out / f"preview_{q:04d}.png", draw(min(q, n - 1)))
        return
    import imageio_ffmpeg
    w = imageio_ffmpeg.write_frames(str(out / "tmp_hop_phases.mp4"), SIZE, fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
                                    quality=9, macro_block_size=8)
    w.send(None)
    for i in range(n):
        w.send(np.ascontiguousarray(draw(i)))
        if i % 60 == 0: print("frame", i, "/", n, flush=True)
    w.close()
    (out / "tmp_hop_phases.mp4").rename(out / "hop_phases.mp4")
    plt.imsave(out / "poster.png", draw(n - 1))
    print("done", n, "frames")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--preview", action="store_true")
    main(ap.parse_args().preview)

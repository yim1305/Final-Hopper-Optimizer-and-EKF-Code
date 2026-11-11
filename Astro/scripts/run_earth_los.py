"""Direct-to-Earth line of sight along the saved flight and at both landed sites, over the libration range.

    .\\.venv\\Scripts\\python.exe -B Astro/scripts/run_earth_los.py [case]

Run from the repository root after run_case.py has accepted the case (reads results/<case>/parameters.json and
independent_flight.npz).

Earth direction: sub-Earth point at the optical-libration corners (lon -7.9/0/+7.9 deg x lat -6.83/0/+6.83 deg, the same
9 cases as the tracker Sun/Earth sweep), Earth at 384,400 km (parallax included). Terrain-only ray from the hopper
(2 m antenna height on the surface) toward Earth through the native 30 m DEM and the global LOLA grid
(mission.trackers.SkyTerrain). No link budget, antenna pattern or elevation mask. Writes results/earth_los/.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mission.trackers import SkyTerrain

case = sys.argv[1] if len(sys.argv) > 1 else "a2200_08"
src = ROOT / "results" / case
out = ROOT / "results" / "earth_los"; out.mkdir(parents=True, exist_ok=True)
DIST, STEP, ANT = 384_400e3, 1.0, 2.0


def unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


f = np.load(src / "independent_flight.npz")
t_all, r_all, hop_all = f["time"], f["r"], f["hop"]
p = json.loads((src / "parameters.json").read_text())
sky = SkyTerrain()
# cumulative flight time (independent flight restarts its clock each hop)
dur1 = float(t_all[hop_all == 1].max())
tc = t_all  # independent flight time is already cumulative across hops
grid = np.arange(0, tc.max(), STEP)
R = np.array([np.interp(grid, tc, r_all[:, k]) for k in range(3)]).T
up = R / np.linalg.norm(R, axis=1)[:, None]
agl = np.linalg.norm(R, axis=1) - sky.native.radius - sky.height(R)
R = R + up * np.maximum(ANT - agl, 0.0)[:, None]          # keep the antenna 2 m above ground on the pad / at touchdown
sites = {"launch base": np.array(p["mission"]["r_start"]), "Drygalski site": np.array(p["mission"]["sites"]["uprange"])}
cases = [(la, lo) for la in (-6.83, 0.0, 6.83) for lo in (-7.9, 0.0, 7.9)]
rows, series = [], {}
for la, lo in cases:
    E = DIST * unit(la, lo)
    vis = np.empty(len(R), bool); elev = np.empty(len(R))
    for i, r in enumerate(R):
        e = E - r; e /= np.linalg.norm(e)
        elev[i] = np.degrees(np.arcsin(np.clip(e @ up[i], -1, 1)))
        vis[i] = sky.cone(r, e, 0.0, 0, 1)["clear"]
    site = {}
    for nm, rs_ in sites.items():
        u = rs_ / np.linalg.norm(rs_); rr = u * (sky.native.radius + sky.height(rs_[None])[0] + ANT)
        e = E - rr; e /= np.linalg.norm(e)
        site[nm] = dict(visible=bool(sky.cone(rr, e, 0.0, 0, 1)["clear"]), earth_elevation_deg=float(np.degrees(np.arcsin(e @ u))))
    blk = ~vis; longest = cur = 0
    for b in blk:
        cur = cur + 1 if b else 0; longest = max(longest, cur)
    h1 = grid < dur1
    rows.append(dict(earth_lat=la, earth_lon=lo, visible_fraction=float(vis.mean()), visible_fraction_hop1=float(vis[h1].mean()),
                     visible_fraction_hop2=float(vis[~h1].mean()), longest_blackout_s=float(longest * STEP), sites=site))
    series[f"{la:+.2f}_{lo:+.1f}"] = dict(vis=vis, elev=elev)
    print(la, lo, round(vis.mean() * 100, 1), longest * STEP, {k: v["visible"] for k, v in site.items()}, flush=True)
np.savez_compressed(out / "series.npz", time=grid, **{k + "_vis": v["vis"] for k, v in series.items()}, **{k + "_elev": v["elev"] for k, v in series.items()})
summary = dict(case=case, step_s=STEP, antenna_m=ANT, earth_distance_m=DIST, hop1_duration_s=dur1, cases=rows,
               scope="Terrain-only straight ray to Earth (native 30 m DEM + global LOLA 16 ppd); libration corners, no ephemeris date, "
                     "no link budget, antenna pattern or elevation mask.")
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print("worst flight visibility", min(r["visible_fraction"] for r in rows), "best", max(r["visible_fraction"] for r in rows))

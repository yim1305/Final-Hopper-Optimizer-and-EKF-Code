r"""Star-tracker availability on the 8-A2200 reference flight over a sweep of Sun and Earth directions.

Post-check only: the solved trajectory and the fixed mounts are unchanged. Terrain-cone clearance and body
rate do not depend on Sun/Earth, so they are evaluated once per head and time sample; Sun/Earth exclusions are
then applied for every geometry.

Sun direction (Moon-fixed axes at launch, frozen inertially): subsolar longitude lambda swept 0-360 deg,
subsolar latitude delta in {-1.5427, 0, +1.5427} deg (lunar equator tilt to the ecliptic).
Earth direction: sub-Earth point at the optical-libration extremes, longitude in {-7.9, 0, +7.9} deg and
latitude in {-6.83, 0, +6.83} deg (amplitudes 7 deg 54' and 6 deg 50').

Run from the repository root:
    .\.venv\Scripts\python.exe -B Astro/scripts/run_tracker_sun_sweep.py [--report-only]
Reads results/a2200_08/sizing.npz (a stored restart array) and writes results/tracker_sun_sweep/a2200_08/.
"""

import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hopper.star_tracker import rotation_z, top_mount_axes
from mission.config import ROOT, parameters, read_solution, write
from mission.terrain import DemTerrain
from mission.trackers import SkyTerrain, blocks

CASE = "a2200_08"
OUT = ROOT / "results/tracker_sun_sweep" / CASE
STEP_S = 0.25
SUN_LON = np.arange(0.0, 360.0, 5.0)
SUN_LAT = (-1.5427, 0.0, 1.5427)
EARTH_LON = (-7.9, 0.0, 7.9)
EARTH_LAT = (-6.83, 0.0, 6.83)


def unit(lat_deg, lon_deg):
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def geometry_samples(p, sol, sky):
    """Per block: times, boresights in inertial axes (n x 2 x 3), and terrain/rate validity (n x 2)."""
    st = p.star_tracker
    mounts = top_mount_axes(st.initial_elevations, st.initial_azimuths)
    half = st.full_cone_fov / 2 + st.moon_limb_margin
    out = {}
    for block in blocks(p, sol):
        times = np.linspace(block.start, block.end, int(np.ceil((block.end - block.start) / STEP_S)) + 1)
        bore = np.zeros((len(times), 2, 3))
        ok = np.zeros((len(times), 2), bool)
        rates = np.zeros(len(times))
        for k, t in enumerate(times):
            ph = next((ph for ph in block.phases if t <= ph.end + 1e-10), block.phases[-1])
            C, rate = ph.orientation(np.array([t]))
            rates[k] = rate[0]
            r = p.mission.r_start + ph.state(np.array([np.clip(t - ph.start, 0, ph.ph["t"])]))[0, :3]
            rate_ok = (not st.enforce_body_rate) or rate[0] <= st.tracking_body_rate_max + st.rate_tolerance
            for j, axis in enumerate(mounts):
                b = C[0] @ axis
                bore[k, j] = b
                if rate_ok:
                    d = rotation_z(-p.moon.omega * t) @ b
                    ok[k, j] = sky.cone(r, d, half, 2, 16)["clear"] and sky.cone(r, d, half, 4, 32)["clear"]
        out[block.name] = dict(times=times, bore=bore, ok=ok, rates=rates, start=block.start, end=block.end)
        print(block.name, "samples", len(times), "terrain/rate-valid fraction per head",
              ok.mean(axis=0).round(3).tolist(), flush=True)
    return out


def evaluate(samples, st, sun, earth):
    """Availability and worst estimate age for one Sun/Earth geometry."""
    result = {}
    for name, s in samples.items():
        sun_ok = np.degrees(np.arccos(np.clip(s["bore"] @ sun, -1, 1))) >= np.degrees(st.sun_exclusion)
        earth_ok = np.degrees(np.arccos(np.clip(s["bore"] @ earth, -1, 1))) >= np.degrees(st.earth_exclusion)
        valid = (s["ok"] & sun_ok & earth_ok).any(axis=1)
        t = s["times"][valid]
        ages = np.diff(np.r_[s["start"], t, s["end"]]) if len(t) else np.array([s["end"] - s["start"]])
        sun_blocked = (~sun_ok & s["ok"]).all(axis=1) & s["ok"].any(axis=1)
        result[name] = dict(available_fraction=float(valid.mean()), max_age_s=float(ages.max()),
                            sun_blocked_s=float(sun_blocked.sum() * STEP_S),
                            duration_s=float(s["end"] - s["start"]))
    return result


def main():
    terrain = DemTerrain()
    p = parameters(CASE, terrain)
    sol = read_solution(ROOT / "results" / CASE / "sizing.npz")
    st = p.star_tracker
    sky = SkyTerrain()
    samples = geometry_samples(p, sol, sky)
    rows = []
    for e_lat in EARTH_LAT:
        for e_lon in EARTH_LON:
            earth = unit(e_lat, e_lon)
            for s_lat in SUN_LAT:
                for s_lon in SUN_LON:
                    res = evaluate(samples, st, unit(s_lat, s_lon), earth)
                    rows.append(dict(sun_lat=s_lat, sun_lon=float(s_lon), earth_lat=e_lat, earth_lon=e_lon,
                                     **{f"{b}_{k}": v for b, r in res.items() for k, v in r.items()}))
    OUT.mkdir(parents=True, exist_ok=True)
    nominal_lon = float(np.degrees(np.arctan2(*st.sun_direction_I[1::-1])))
    write(OUT / "sweep.json", dict(case=CASE, step_s=STEP_S, update_gap_max_s=st.update_gap_max,
                                   sun_exclusion_deg=float(np.degrees(st.sun_exclusion)),
                                   earth_exclusion_deg=float(np.degrees(st.earth_exclusion)),
                                   nominal_sun_lon_deg=nominal_lon, rows=rows,
                                   head_terrain_rate_valid_fraction={b: s["ok"].mean(axis=0).tolist()
                                                                     for b, s in samples.items()}))
    np.savez_compressed(OUT / "samples.npz", **{f"{b}_{k}": v for b, s in samples.items()
                                                 for k, v in s.items() if isinstance(v, np.ndarray)})
    report(OUT)


def report(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.loads((out / "sweep.json").read_text())
    rows, gap = d["rows"], d["update_gap_max_s"]
    age = np.array([max(r["hop1_max_age_s"], r["hop2_max_age_s"]) for r in rows])
    avail = np.array([min(r["hop1_available_fraction"], r["hop2_available_fraction"]) for r in rows])
    worst = rows[int(np.argmax(age - 1e-6 * avail))]
    nominal_earth = [r for r in rows if r["earth_lat"] == 0 and r["earth_lon"] == 0]
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for s_lat, style in zip(sorted({r["sun_lat"] for r in rows}), ("--", "-", ":")):
        sel = [r for r in nominal_earth if r["sun_lat"] == s_lat]
        lon = [r["sun_lon"] for r in sel]
        for hop, color in (("hop1", "#1f4e9c"), ("hop2", "#c0504d")):
            ax[0].plot(lon, [100 * r[f"{hop}_available_fraction"] for r in sel], style, color=color,
                       label=f"{'Outbound' if hop == 'hop1' else 'Return'}, Sun lat {s_lat:+.2f}°")
            ax[1].plot(lon, [r[f"{hop}_max_age_s"] for r in sel], style, color=color)
    ax[1].axhline(gap, color="k", lw=1)
    ax[1].text(2, gap + 0.05, f"limit {gap:g} s", fontsize=9)
    for a in ax:
        a.axvline(d["nominal_sun_lon_deg"] % 360, color="gray", lw=0.8)
        a.grid(alpha=0.3)
    ax[0].set_ylabel("Time with a valid head (%)")
    ax[1].set_ylabel("Longest estimate age (s)")
    ax[1].set_xlabel("Subsolar longitude (deg); grey line = nominal assumed Sun")
    ax[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "sun_sweep.png", dpi=150)
    n_fail = int((age > gap + 1e-8).sum())
    lines = [f"# Star-tracker Sun/Earth sweep: {d['case']}", "",
             "Post-check on the unchanged reference flight and fixed mounts. Sample step "
             f"{d['step_s']} s; valid head = terrain-clear cone, body rate <= limit, Sun and Earth exclusions "
             f"{d['sun_exclusion_deg']:g} deg.", "",
             f"Geometries: {len(rows)} (Sun longitude 0-355 deg every 5 deg x latitude -1.54/0/+1.54 deg x "
             "Earth libration corners lon -7.9/0/+7.9 x lat -6.83/0/+6.83 deg).", "",
             f"- Geometries exceeding the {gap:g} s estimate-age limit: {n_fail} of {len(rows)}",
             f"- Worst longest estimate age: {age.max():.2f} s (Sun lon {worst['sun_lon']:g}, lat {worst['sun_lat']:+g}; "
             f"Earth lon {worst['earth_lon']:+g}, lat {worst['earth_lat']:+g})",
             f"- Lowest time with a valid head (worse hop): {100 * avail.min():.1f} %",
             f"- Median time with a valid head (worse hop): {100 * np.median(avail):.1f} %",
             f"- Worst case per hop: outbound max age {worst['hop1_max_age_s']:.2f} s, Sun-blocked "
             f"{worst['hop1_sun_blocked_s']:.1f} of {worst['hop1_duration_s']:.1f} s; return max age "
             f"{worst['hop2_max_age_s']:.2f} s, Sun-blocked {worst['hop2_sun_blocked_s']:.1f} of {worst['hop2_duration_s']:.1f} s",
             "", "Terrain/rate-valid fraction of samples per head (independent of Sun/Earth): " +
             json.dumps({k: [round(x, 3) for x in v] for k, v in d["head_terrain_rate_valid_fraction"].items()}),
             "", "![sweep](sun_sweep.png)", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        report(OUT)
    else:
        main()

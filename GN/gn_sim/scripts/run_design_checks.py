"""Case C sensitivity studies on the accepted flight (design-verification support).

Each study changes one input from nav_config.yaml (all others unchanged) and reruns case C on both hops:

  trn_ceiling : SPLICE TRN available only below an altitude ceiling (is high-altitude TRN needed?)
  init_pos    : initial position 1-sigma at launch (surface-initialization uncertainty, e.g. hop-1 landing error)
  imu_vrw     : accelerometer velocity random walk (the PLACEHOLDER value in nav_config.yaml)

Run from the repo root:  .\\.venv\\Scripts\\python.exe -B GN/scripts/run_design_checks.py [--runs N]
Output: <outputs.directory>/design_checks.json and DESIGN_CHECKS.md. This script does not create the output folder,
so run GN/scripts/run_nav.py (or run_trn_sweep.py) first, or the script stops with a file-not-found error at the end.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from gn_sim.config import ROOT, load_config
from gn_sim.montecarlo import run_case
from gn_sim.sensors import tracker_availability
from gn_sim.truth import build_hop, load_case

STUDIES = {
    "trn_ceiling": dict(label="TRN altitude ceiling (m)", key=("trn", "altitude_band_m"),
                        values=[float("inf"), 10000.0, 5000.0, 2000.0, 1000.0],
                        make=lambda v: [0.0, v]),
    "init_pos": dict(label="Initial position 1σ per axis (m)", key=("initial", "position_sigma_m"),
                     values=[0.001, 10.0, 30.0, 100.0], make=lambda v: v),
    "imu_vrw": dict(label="Accelerometer VRW (m/s/√hr)", key=("imu", "accel_vrw_m_s_rthr"),
                    values=[0.01, 0.03, 0.1, 0.3], make=lambda v: v),
}


def override(key, value):
    d = value
    for k in reversed(key):
        d = {k: d}
    return d


def main(runs=50):
    base = load_config()
    out_dir = ROOT / base.raw.get("outputs", {}).get("directory", "GN/results")
    case = load_case(base)
    req = base.requirements["landing_horizontal_3sigma_m"]
    tick = time.perf_counter()
    truths, avails = {}, {}
    for hop in (1, 2):
        truths[hop] = build_hop(case, base, hop)
        avails[hop], _ = tracker_availability(truths[hop], case, base)
    results = {}
    for name, st in STUDIES.items():
        rows = []
        for v in st["values"]:
            cfg = load_config(overrides=override(st["key"], st["make"](v)))
            row = dict(value=v)
            for hop in (1, 2):
                res = run_case(truths[hop], case, cfg, cfg.cases["C"], avails[hop], runs=runs, store=False,
                               case_id="C")
                cp = res["checkpoints"]
                td = cp["touchdown"]
                row[f"hop{hop}"] = dict(
                    touchdown_horizontal_3sigma_mc_m=td["pos_horizontal_3sigma_mc_m"],
                    touchdown_horizontal_3sigma_filter_m=td["pos_horizontal_3sigma_filter_m"],
                    touchdown_vertical_3sigma_mc_m=td["pos_3sigma_mc_m"][2],
                    descent_1000m_horizontal_3sigma_mc_m=cp["descent_1000m"]["pos_horizontal_3sigma_mc_m"],
                    trn_updates=res["updates"]["trn"])
            row["worst_hop_m"] = max(max(row[h]["touchdown_horizontal_3sigma_mc_m"],
                                         row[h]["touchdown_horizontal_3sigma_filter_m"]) for h in ("hop1", "hop2"))
            row["meets_requirement"] = bool(row["worst_hop_m"] <= req)
            rows.append(row)
            print(f"{name:12s} {v:>10g}: worst hop {row['worst_hop_m']:7.1f} m  "
                  f"(out {row['hop1']['touchdown_horizontal_3sigma_mc_m']:.1f} / back "
                  f"{row['hop2']['touchdown_horizontal_3sigma_mc_m']:.1f} m MC)", flush=True)
        results[name] = dict(label=st["label"], key=".".join(st["key"]), rows=rows)
    out = dict(description="Case C one-at-a-time sensitivity studies on the accepted flight",
               trajectory_case=str(base.case_dir.relative_to(ROOT)), runs_per_value_per_hop=runs, requirement_m=req,
               nominal=dict(trn_altitude_band_m=list(base.trn.altitude_band), initial_position_sigma_m=base.position_sigma0,
                            accel_vrw_m_s_rthr=base.raw["imu"]["accel_vrw_m_s_rthr"]),
               studies=results, runtime_s=time.perf_counter() - tick)
    (out_dir / "design_checks.json").write_text(json.dumps(out, indent=1, default=lambda x: None if x == float("inf") else x).replace("Infinity", "null"))
    L = ["# G&N design checks (case C sensitivities)", "",
         f"{runs} Monte Carlo runs per value per hop. Value = touchdown horizontal 3σ, worse of Monte Carlo and filter, "
         f"worse hop. Pass = within {req:g} m. One input changed at a time; all others as in nav_config.yaml.", ""]
    for name, r in results.items():
        L += [f"## {r['label']}", "", "| Value | Outbound MC (m) | Return MC (m) | Worst (m) | ≤ 50 m |", "|---:|---:|---:|---:|---|"]
        for row in r["rows"]:
            L.append(f"| {row['value']:g} | {row['hop1']['touchdown_horizontal_3sigma_mc_m']:.1f} | "
                     f"{row['hop2']['touchdown_horizontal_3sigma_mc_m']:.1f} | {row['worst_hop_m']:.1f} | "
                     f"{'yes' if row['meets_requirement'] else 'no'} |")
        L.append("")
    (out_dir / "DESIGN_CHECKS.md").write_text("\n".join(L), encoding="utf-8")
    print(f"done in {time.perf_counter() - tick:.0f} s")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    main(ap.parse_args().runs)

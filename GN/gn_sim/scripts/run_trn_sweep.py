"""Sweep SPLICE TRN accuracy and update rate (case C) to derive the requirement on TRN data.

Run from the repository root with the launcher (it sets up the import paths):
    .\\.venv\\Scripts\\python.exe -B GN/scripts/run_trn_sweep.py [--runs N]
Needs the generated Astro flight (Astro/scripts/run_case.py a2200_08). Writes to outputs.directory of GN/nav_config.yaml.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from gn_sim import plots
from gn_sim.config import GN_DIR, ROOT, load_config
from gn_sim.montecarlo import run_case
from gn_sim.sensors import tracker_availability
from gn_sim.truth import build_hop, load_case

OUT = GN_DIR / "results"


def main(runs=None):
    global OUT
    cfg = load_config()
    OUT=ROOT/cfg.raw.get('outputs',{}).get('directory','GN/results')
    case = load_case(cfg)
    sig = [float(s) for s in cfg.sweep["trn_sigma_m"]]
    rates = [float(r) for r in cfg.sweep["trn_rate_hz"]]
    runs = runs or int(cfg.sweep["runs"])
    req = cfg.requirements["landing_horizontal_3sigma_m"]
    OUT.mkdir(parents=True, exist_ok=True)
    grid, grid_filter, rows = {}, {}, []
    tick = time.perf_counter()
    for hop in (1, 2):
        truth = build_hop(case, cfg, hop)
        avail, _ = tracker_availability(truth, case, cfg)
        g = np.zeros((len(sig), len(rates)))
        gf = np.zeros_like(g)
        for i, s in enumerate(sig):
            for j, r in enumerate(rates):
                res = run_case(truth, case, cfg, cfg.cases["C"], avail, runs=runs, trn_sigma=s, trn_rate=r,
                               store=False, case_id=f"S{i}{j}")
                td = res["checkpoints"]["touchdown"]
                g[i, j] = td["pos_horizontal_3sigma_mc_m"]
                gf[i, j] = td["pos_horizontal_3sigma_filter_m"]
                rows.append(dict(hop=hop, trn_sigma_m=s, trn_rate_hz=r, touchdown_horizontal_3sigma_mc_m=g[i, j],
                                 touchdown_horizontal_3sigma_filter_m=gf[i, j],
                                 touchdown_vertical_3sigma_mc_m=td["pos_3sigma_mc_m"][2],
                                 touchdown_velocity_3sigma_mc_m_s=td["vel_3sigma_mc_m_s"]))
                print(f"hop {hop}  sigma {s:5g} m  rate {r:5g} Hz  ->  {g[i, j]:7.1f} m (filter {gf[i, j]:7.1f} m)",
                      flush=True)
        grid[hop], grid_filter[hop] = g, gf
    # report the worse of Monte Carlo and filter, worse hop
    worst = np.maximum(np.maximum(grid[1], grid_filter[1]), np.maximum(grid[2], grid_filter[2]))
    passing = [dict(trn_sigma_m=sig[i], trn_rate_hz=rates[j], worst_hop_horizontal_3sigma_m=float(worst[i, j]))
               for i in range(len(sig)) for j in range(len(rates)) if worst[i, j] <= req]
    plots.sweep_heatmap({h: np.maximum(grid[h], grid_filter[h]) for h in grid}, sig, rates, req,
                        OUT / "trn_sweep.png")
    out = dict(description="SPLICE TRN accuracy x rate sweep, case C, touchdown horizontal 3-sigma",
               trajectory_case=str(cfg.case_dir.relative_to(ROOT)),case_notes=case.notes,
               runs=runs, requirement_m=req, sigmas_m=sig, rates_hz=rates, rows=rows,
               passing_combinations=passing, runtime_s=time.perf_counter() - tick)
    (OUT / "trn_sweep.json").write_text(json.dumps(out, indent=1))
    L = ["# Derived requirement on SPLICE TRN data", "",
         f"Case C (IMU + 2 star trackers + SPLICE TRN), {runs} Monte Carlo runs per cell. Value = touchdown horizontal "
         f"3σ, the worse of Monte Carlo and filter, worse of the two hops. Pass = within {req:g} m.", "",
         "| TRN accuracy (1σ) \\ rate | " + " | ".join(f"{r:g} Hz" for r in rates) + " |",
         "|---|" + "---:|" * len(rates)]
    for i, s in enumerate(sig):
        L.append(f"| {s:g} m | " + " | ".join(
            f"{worst[i, j]:.0f} m {'✓' if worst[i, j] <= req else '✗'}" for j in range(len(rates))) + " |")
    L += ["", "Per-hop values are in `trn_sweep.json`; figure: `trn_sweep.png`.", ""]
    (OUT / "TRN_SWEEP.md").write_text("\n".join(L), encoding="utf-8")
    print(f"done in {time.perf_counter() - tick:.0f} s")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=None)
    main(ap.parse_args().runs)

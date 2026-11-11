"""Run all navigation cases (A, B, C, D1, D2, E) on both hops; write figures, summary.json and RESULTS.md.

Run from the repository root with the launcher (it sets up the import paths):
    .\\.venv\\Scripts\\python.exe -B GN/scripts/run_nav.py [--runs N]
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
from gn_sim.sensors import imu_range_check, tracker_availability, update_gaps
from gn_sim.truth import build_hop, load_case

OUT = GN_DIR / "results"


def window_gaps(truth, avail_any, kinds=("ascent", "coast")):
    starts = [b[2] for b in truth.phase_bounds if b[1] in kinds]
    ends = [b[3] for b in truth.phase_bounds if b[1] in kinds]
    lo, hi = min(starts), max(ends)
    sel = (truth.t >= lo) & (truth.t <= hi)
    times = truth.t[sel & avail_any]
    edges = np.r_[lo, times, hi]
    return dict(window=f"{'+'.join(kinds)} ({lo - truth.t[0]:.1f}-{hi - truth.t[0]:.1f} s)",
                max_gap_s=float(np.diff(edges).max()))


def main(runs=None):
    global OUT
    cfg = load_config()
    OUT=ROOT/cfg.raw.get('outputs',{}).get('directory','GN/results')
    case = load_case(cfg)
    runs = runs or cfg.runs
    OUT.mkdir(parents=True, exist_ok=True)
    req = cfg.requirements["landing_horizontal_3sigma_m"]
    truths, results, avails, summary = {}, {1: {}, 2: {}}, {}, {}
    hops = {}
    tick = time.perf_counter()
    for hop in (1, 2):
        truth = build_hop(case, cfg, hop)
        avail, info = tracker_availability(truth, case, cfg)
        truths[hop], avails[hop] = truth, avail
        gaps_all = update_gaps(truth, avail.any(axis=1))
        hops[hop] = dict(
            duration_s=float(truth.t[-1] - truth.t[0]), nav_epochs=len(truth.t),
            max_altitude_m=float(truth.altitude.max()),
            phases=[dict(name=b[0], kind=b[1], start_s=b[2] - truth.t[0], end_s=b[3] - truth.t[0])
                    for b in truth.phase_bounds],
            checkpoints={k: dict(time_s=float(truth.t[v] - truth.t[0]), altitude_m=float(truth.altitude[v]))
                         for k, v in truth.checkpoints.items()},
            attitude_jumps_smoothed=truth.attitude_jumps, imu_range=imu_range_check(truth, cfg),
            star_tracker=dict(
                fraction_of_time_available=[float(x) for x in (info["geometry_ok"] & info["rate_ok"][:, None]).mean(0)],
                fraction_geometry_ok=[float(x) for x in info["geometry_ok"].mean(0)],
                fraction_body_rate_ok=float(info["rate_ok"].mean()),
                either_head_whole_flight=gaps_all,
                either_head_ascent_coast=window_gaps(truth, avail.any(axis=1)),
                head1_only_whole_flight=dict(max_gap_s=update_gaps(truth, avail[:, 0])["max_gap_s"]),
                head2_only_whole_flight=dict(max_gap_s=update_gaps(truth, avail[:, 1])["max_gap_s"])))
        plots.tracker_timeline(truth, info, np.degrees(cfg.tracker.max_body_rate), OUT / f"star_tracker_hop{hop}.png")
        for cid, cdef in cfg.cases.items():
            t0 = time.perf_counter()
            res = run_case(truth, case, cfg, cdef, avail, runs=runs, case_id=cid)
            results[hop][cid] = res
            plots.error_history(res, truth, OUT / f"errors_{cid}_hop{hop}.png")
            td = res["checkpoints"]["touchdown"]
            print(f"hop {hop} case {cid:2s}: touchdown horizontal 3σ {td['pos_horizontal_3sigma_mc_m']:9.1f} m "
                  f"(filter {td['pos_horizontal_3sigma_filter_m']:9.1f} m)  [{time.perf_counter() - t0:.1f} s]",
                  flush=True)
    plots.comparison(results, truths, req, OUT / "comparison_ABC.png")
    plots.star_tracker_value(results, truths, req, OUT / "star_tracker_value.png")
    tdr = {hop: {} for hop in (1, 2)}
    for hop in (1, 2):
        for cid, res in results[hop].items():
            i = list(res["hist"]["idx"]).index(truths[hop].checkpoints["touchdown"])
            tdr[hop][cid] = dict(e=res["hist"]["e_r"][i], P=res["hist"]["P_r"][i],
                                ev=res["hist"]["e_v"][i], eth=res["hist"]["e_th"][i])
    plots.touchdown_distribution(tdr, req, OUT / "touchdown_distribution.png")
    np.savez_compressed(OUT / "touchdown_runs.npz", **{
        f"hop{hop}_{cid}_{k}": v for hop in tdr for cid, d in tdr[hop].items() for k, v in d.items()})

    # ---- summary -----------------------------------------------------------------------------------
    case_rows = {}
    for hop in (1, 2):
        for cid, res in results[hop].items():
            td = res["checkpoints"]["touchdown"]
            att = 3 * np.sqrt(np.max(np.mean(res["hist"]["e_th"]**2, axis=1), axis=1)) * 180 / np.pi * 3600
            horiz = np.hypot(tdr[hop][cid]["e"][:, 0], tdr[hop][cid]["e"][:, 1])
            case_rows.setdefault(cid, dict(label=res["label"]))[f"hop{hop}"] = dict(
                checkpoints=res["checkpoints"], consistency=res["consistency"], updates=res["updates"],
                attitude_worst_axis_3sigma_arcsec=dict(max_in_flight=float(att.max()), touchdown=float(att[-1])),
                touchdown_horizontal_per_run_m=dict(median=float(np.median(horiz)),
                                                    p95=float(np.percentile(horiz, 95)), worst=float(horiz.max()),
                                                    runs_over_requirement=int(np.sum(horiz > req))),
                touchdown_meets_landing_requirement=bool(td["pos_horizontal_3sigma_mc_m"] <= req and
                                                         td["pos_horizontal_3sigma_filter_m"] <= req))
    summary = dict(
        description="G&N navigation Monte Carlo, launch to touchdown, both hops (gn_sim)",
        trajectory_case=str(cfg.case_dir.relative_to(ROOT)), case_notes=case.notes,
        star_tracker_mounts_body=case.mounts.tolist(), runs_per_case=runs, seed=cfg.seed,
        settings=cfg.raw, hops=hops, cases=case_rows,
        requirement=dict(landing_horizontal_3sigma_m=req,
                         note="Pass if both Monte Carlo and filter horizontal 3σ at touchdown are within the requirement"),
        runtime_s=time.perf_counter() - tick)
    (OUT / "summary.json").write_text(json.dumps(_json_safe(summary), indent=1, allow_nan=False))
    np.savez_compressed(OUT / "histories.npz", **{
        f"hop{hop}_{cid}_{key}": arr.astype(np.float32)
        for hop in (1, 2) for cid, res in results[hop].items()
        for key, arr in _compact(res).items()})
    write_results_md(summary, cfg)
    print(f"done in {time.perf_counter() - tick:.0f} s -> {OUT}")
    return summary


def _json_safe(x):
    """Plain JSON: numpy scalars to Python, infinities to the string 'inf'."""
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return _json_safe(x.tolist())
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else ("inf" if x > 0 else "-inf")
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _compact(res):
    H = res["hist"]
    out = dict(t=res["t"])
    for k in ("e_r", "e_v", "e_th"):
        out[f"{k}_mc3sigma"] = 3 * np.sqrt(np.mean(H[k]**2, axis=1))
        out[f"{k}_samples"] = H[k][:, :5]
    for k in ("P_r", "P_v", "P_th"):
        out[f"{k}_filter3sigma"] = 3 * np.sqrt(np.mean(np.diagonal(H[k], axis1=2, axis2=3), axis=1))
    return out


def write_results_md(s, cfg):
    req = s["requirement"]["landing_horizontal_3sigma_m"]
    L = ["# G&N navigation results", "",
         f"Trajectory: `{s['trajectory_case']}` (8 × A2200). {s['runs_per_case']} Monte Carlo runs per case per hop. "
         "Generated by `gn_sim/scripts/run_nav.py`; numbers come from `summary.json`.", "",
         f"SPLICE TRN placeholder: {cfg.trn.sigma:g} m (1σ per axis) at {cfg.trn.rate:g} Hz. " +
         ("See `TRN_SWEEP.md` for the derived requirement." if (OUT/'TRN_SWEEP.md').exists()
          else "The sensitivity grid was not rerun for this flight. The previous `../TRN_SWEEP.md` is historical."), "",
         "## Touchdown navigation error (3σ, Monte Carlo)", "",
         "| Case | Sensors | Hop 1 horizontal | Hop 1 vertical | Hop 2 horizontal | Hop 2 vertical | Meets 50 m? |",
         "|---|---|---:|---:|---:|---:|---|"]
    for cid, row in s["cases"].items():
        cells = []
        ok = True
        for hop in (1, 2):
            td = row[f"hop{hop}"]["checkpoints"]["touchdown"]
            cells += [f"{td['pos_horizontal_3sigma_mc_m']:,.1f} m", f"{td['pos_3sigma_mc_m'][2]:,.1f} m"]
            ok &= row[f"hop{hop}"]["touchdown_meets_landing_requirement"]
        L.append(f"| {cid} | {row['label']} | " + " | ".join(cells) + f" | {'yes' if ok else 'no'} |")
    L += ["", f"Requirement: {req:g} m horizontal (3σ navigation knowledge at touchdown).", "",
          "## Per-run touchdown distribution and attitude (Monte Carlo)", "",
          "| Case | Hop | Median horizontal | 95th pct | Worst run | Runs over 50 m | Attitude 3σ, worst in flight | Attitude 3σ at touchdown |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for cid, row in s["cases"].items():
        for hop in (1, 2):
            d, a = row[f"hop{hop}"]["touchdown_horizontal_per_run_m"], row[f"hop{hop}"]["attitude_worst_axis_3sigma_arcsec"]
            L.append(f"| {cid} | {hop} | {d['median']:,.1f} m | {d['p95']:,.1f} m | {d['worst']:,.1f} m | "
                     f"{d['runs_over_requirement']}/{s['runs_per_case']} | {a['max_in_flight']:,.0f}″ | {a['touchdown']:,.0f}″ |")
    L += ["",
          "## Case C at each checkpoint (3σ, Monte Carlo)", "",
          "| Hop | Checkpoint | Time (s) | Altitude (m) | Horizontal pos | Vertical pos | Horizontal vel | Vertical vel | Attitude (max axis) |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for hop in (1, 2):
        for name, c in s["cases"]["C"][f"hop{hop}"]["checkpoints"].items():
            L.append(f"| {hop} | {name} | {c['time_s']:.1f} | {c['altitude_m']:,.0f} | "
                     f"{c['pos_horizontal_3sigma_mc_m']:.1f} m | {c['pos_3sigma_mc_m'][2]:.1f} m | "
                     f"{c['vel_horizontal_3sigma_mc_m_s']:.3f} m/s | {c['vel_3sigma_mc_m_s'][2]:.3f} m/s | "
                     f"{max(c['att_3sigma_mc_arcsec']):.1f}″ |")
    L += ["", "## Filter consistency (case C)", "",
          "| Hop | Block | Within ±3σ | Mean NEES (expect 3) |", "|---|---|---:|---:|"]
    for hop in (1, 2):
        for blk, c in s["cases"]["C"][f"hop{hop}"]["consistency"].items():
            L.append(f"| {hop} | {blk} | {100 * c['fraction_within_3sigma']:.2f}% | {c['mean_nees']:.2f} |")
    L += ["", "## Star tracker availability (6 °/s hardware limit)", "",
          "| Hop | Max gap, whole flight | Max gap, ascent+coast | Gaps over 2 s |", "|---|---:|---:|---|"]
    for hop in (1, 2):
        st = s["hops"][hop]["star_tracker"]
        g = st["either_head_whole_flight"]
        long = "; ".join(f"{x['length_s']:.1f} s at t={x['start_s']:.0f} s" for x in g["gaps_over_2s"]) or "none"
        L.append(f"| {hop} | {g['max_gap_s']:.2f} s | {st['either_head_ascent_coast']['max_gap_s']:.2f} s | {long} |")
    L += ["", "## Attitude steps in the optimizer trajectory (smoothed in the truth model)", ""]
    for hop in (1, 2):
        for j in s["hops"][hop]["attitude_jumps_smoothed"]:
            if j["angle_deg"] > 0.1:
                L.append(f"- Hop {hop}, {j['at']}: {j['angle_deg']:.1f}° step, slewed over {j['smoothed_over_s']:.2f} s "
                         f"at {j['slew_rate_deg_s']:.0f} °/s")
    L += ["", "## Figures", "", "- `comparison_ABC.png`: cases A/B/C, horizontal 3σ vs time, both hops",
          "- `errors_<case>_hop<n>.png`: error and filter ±3σ for position, velocity, attitude",
          "- `star_tracker_hop<n>.png`: body rate vs the 6 °/s limit and head availability",
          "- `star_tracker_value.png`: case C (with trackers) vs E (without): attitude and position 3σ",
          "- `touchdown_distribution.png`: per-run touchdown errors (scatter, histogram) for C, E, A, B",
          "- `touchdown_runs.npz`: per-run touchdown errors and filter covariances", ""]
    (OUT / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=None)
    main(ap.parse_args().runs)

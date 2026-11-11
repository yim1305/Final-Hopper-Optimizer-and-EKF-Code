"""Walk one case from its saved (old) solution to the mission.yaml descent settings by continuation.

A direct jump to the 2224 N floor and touchdown T/W <= 1 can give a junk iterate, so each step is seeded from the
previous good one: floor only, T/W 2.0 1.7 1.4 1.2 1.1 1.0 (12 observation nodes), then 36 nodes, then the solve-side
pointing margin. When a step does not converge, an intermediate step is inserted between the last good step and the
failed one (up to --depth halvings) and the walk continues. Each step runs scripts/run_descent_variant.py in its own
process. When the last step converges the full acceptance check (check_descent_variant.py) is run on it.

    .\\.venv\\Scripts\\python.exe -B Astro/scripts/descent_ladder.py a2200_12

Run from the repository root. The warm start is Astro/results/<case>/sizing.npz (or --seed), so only a2200_08 and
a2200_12 can be walked from the stored restart arrays; a2200_10 and a2200_14 have no stored arrays.
Writes only Astro/results/ladder_<case>/ (one subfolder per step, ladder.json). The case folder is not touched.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mission.config import CONFIG

PY = sys.executable
OLD_FLOOR = 4448.0  # the floor the saved solutions were solved with (4 engines at 50 %)
TW_NONE = 3.0  # "no limit" when halving a T/W gap
MARGIN_NONE = 6.0  # the real pointing limit, "no margin" when halving a margin gap
OLD_NODES = 12


def interpolate(good, step):
    """Midpoint between two step settings on every field where they differ."""
    mid = dict(step)
    for key, none_value in (("floor", None), ("tw", TW_NONE), ("margin", MARGIN_NONE), ("nodes", None)):
        g, s = good.get(key), step.get(key)
        g = none_value if g is None else g
        s = none_value if s is None else s
        if g is not None and s is not None and g != s:
            mid[key] = round((g + s) / 2) if key == "nodes" else round((g + s) / 2, 3)
    return mid


def too_close(a, b):
    """True when the two steps are indistinguishable at the halving resolution."""
    tol = dict(floor=20.0, tw=0.02, margin=0.05, nodes=3)
    for key, t in tol.items():
        x, y = a.get(key), b.get(key)
        if x is None and y is None:
            continue
        if (x is None) != (y is None):
            return False
        if abs(x - y) > t:
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("case", choices=tuple(CONFIG["cases"]))
    ap.add_argument("--seed", help="results folder holding the warm-start sizing.npz (default: the case folder)")
    ap.add_argument("--distance-km", type=float, help="hop-distance study: target at this distance (use a2200_08 and --seed hop_distance/dNNN)")
    ap.add_argument("--skip-floor", action="store_true", help="resume: --seed already holds the floor-only solution")
    ap.add_argument(
        "--accept-near",
        action="store_true",
        help="count a converged IPOPT status with primal infeasibility <= 1e-5 as good for continuation "
        "(the strict residual test is only a diagnostic seed requirement)",
    )
    ap.add_argument("--depth", type=int, default=3, help="maximum halvings of a failed step")
    ap.add_argument("--wall", type=float, default=900.0, help="IPOPT wall-time cap per step [s]")
    ap.add_argument("--tw", type=float, nargs="+", default=[2.0, 1.7, 1.4, 1.2, 1.1, 1.0])
    a = ap.parse_args()

    dv = CONFIG["descent_variant"]
    floor, tw_final, margin_final = dv["thrust_floor_n"], dv["touchdown_thrust_to_weight_max"], dv["solve_pointing_rate_deg_s"]
    nodes_final = CONFIG["common"]["solver"]["nodes"]["observation"]
    dist = [] if a.distance_km is None else ["--distance-km", str(a.distance_km)]
    base = f"ladder_{a.case}" if a.distance_km is None else f"ladder_d{round(a.distance_km):03d}"
    (ROOT / "results" / base).mkdir(parents=True, exist_ok=True)

    steps = [dict(name="floor", floor=floor, tw=None, nodes=OLD_NODES, margin=None)]
    if a.skip_floor:
        steps = []
    steps += [dict(name=f"tw{t}", floor=floor, tw=t, nodes=OLD_NODES, margin=None) for t in a.tw]
    steps += [dict(name=f"n{nodes_final}", floor=floor, tw=tw_final, nodes=nodes_final, margin=None)]
    steps += [dict(name="margin", floor=floor, tw=tw_final, nodes=nodes_final, margin=margin_final)]

    good = dict(floor=floor if a.skip_floor else OLD_FLOOR, tw=None, nodes=OLD_NODES, margin=None)
    if a.skip_floor:
        base += "_resume"
        (ROOT / "results" / base).mkdir(parents=True, exist_ok=True)
    good_tag = a.seed or a.case
    log, queue, counter = [], [dict(s, depth=0) for s in steps], 0
    failed = None
    while queue:
        step = queue.pop(0)
        counter += 1
        tag = f"{base}/{counter:02d}_{step['name']}"
        cmd = [PY, "-B", str(ROOT / "scripts" / "run_descent_variant.py"), tag, "--case", a.case, "--seed", good_tag,
               "--wall", str(a.wall), "--yaml-off", *dist]
        if step["floor"] is not None:
            cmd += ["--floor", str(step["floor"])]
        if step["tw"] is not None:
            cmd += ["--tw", str(step["tw"])]
        if step["margin"] is not None:
            cmd += ["--margin", str(step["margin"])]
        cmd += ["--obs-nodes", str(step["nodes"])]
        print(f"[{time.strftime('%H:%M:%S')}] {a.case} step {tag}: floor={step['floor']} tw={step['tw']} "
              f"nodes={step['nodes']} margin={step['margin']} seed={good_tag}", flush=True)
        tick = time.monotonic()
        proc = subprocess.run(cmd, cwd=ROOT.parent, capture_output=True, text=True)
        diag_path = ROOT / "results" / tag / "diag.json"
        diag = json.loads(diag_path.read_text()) if diag_path.exists() else {}
        ok = bool(diag.get("success")) and proc.returncode == 0
        if (not ok and a.accept_near and proc.returncode == 0
                and diag.get("status") in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
                and diag.get("primal", 1.0) <= 1e-5):
            ok = True
            print("    (accepted as near-converged: primal %.1e)" % diag["primal"], flush=True)
        entry = dict(tag=tag, step={k: step[k] for k in ("floor", "tw", "nodes", "margin")}, seed=good_tag, ok=ok,
                     status=diag.get("status"), iterations=diag.get("iterations"), m_wet_kg=diag.get("m_wet_kg"),
                     wall_s=round(time.monotonic() - tick))
        if proc.returncode != 0:
            entry["stderr_tail"] = proc.stderr[-600:]
        log.append(entry)
        (ROOT / "results" / base / "ladder.json").write_text(json.dumps(log, indent=1))
        print(f"    -> {'OK' if ok else 'FAILED'} status={entry['status']} iters={entry['iterations']} "
              f"m_wet={entry['m_wet_kg']} ({entry['wall_s']} s)", flush=True)
        if ok:
            good, good_tag = step, tag
            continue
        mid = interpolate(good, step)
        if step["depth"] >= a.depth or too_close(mid, step) or too_close(mid, good):
            failed = step
            break
        mid.update(name=f"{step['name']}_mid{step['depth'] + 1}", depth=step["depth"] + 1)
        step["depth"] += 1
        print(f"    relaxing: inserting {mid['name']} floor={mid['floor']} tw={mid['tw']} nodes={mid['nodes']} "
              f"margin={mid['margin']}", flush=True)
        queue = [mid, step] + queue

    if failed:
        print(f"{a.case}: STOPPED at {failed['name']} after halving {a.depth} times; last good step: {good_tag}", flush=True)
        (ROOT / "results" / base / "ladder.json").write_text(json.dumps(dict(log=log, stopped_at=failed, last_good=good_tag), indent=1))
        sys.exit(2)

    print(f"{a.case}: all steps converged; last good step {good_tag}. Running acceptance check.", flush=True)
    cmd = [PY, "-B", str(ROOT / "scripts" / "check_descent_variant.py"), good_tag, "--case", a.case,
           "--floor", str(good["floor"]), "--tw", str(good["tw"]), "--obs-nodes", str(good["nodes"]), *dist]
    if good.get("margin") is not None:
        cmd += ["--solve-pointing-rate", str(good["margin"])]
    sys.exit(subprocess.run(cmd, cwd=ROOT.parent).returncode)


if __name__ == "__main__":
    main()

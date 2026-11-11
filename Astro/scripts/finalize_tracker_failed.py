"""Write summary.json for a case whose flight passed every check but whose tracker schedule did not.

Same summary/phase-table content as mission/run.py; status marks the tracker failure so it is never mistaken
for an accepted flight.
Usage (repository root): .\\.venv\\Scripts\\python.exe -B Astro/scripts/finalize_tracker_failed.py <case> [...]
Needs results/<case>/ with sizing.npz, validation.json and trackers.json from an earlier run_case.py attempt.
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hopper.lambert import min_dv_hop
from hopper.postprocess import round_trip_summary, phase_table, phase_mode, propellant_by_class
from mission.config import parameters, read_solution, write
from mission.terrain import DemTerrain
from mission.validation import integrate
from mission.config import physical_identity
from hopper.tracker_cache import save_trajectory
import numpy as np

terrain = DemTerrain()
for case in sys.argv[1:]:
    out = ROOT / "results" / case
    p = parameters(case, terrain)
    sol = read_solution(out / "sizing.npz")
    v = json.loads((out / "validation.json").read_text())
    tr = json.loads((out / "trackers.json").read_text())
    lam = (min_dv_hop(p).dv, min_dv_hop(p, r1=p.mission.sites["uprange"], r2=p.mission.r_start).dv)
    table = phase_table(p, sol, "uprange")
    for row, ph in zip(table, sol.phases.values()):
        mode = phase_mode(p, ph)
        row["engine_counts"] = dict(mode.fixed + mode.throttled) if mode else {}
        row["propellant_by_class"] = propellant_by_class(p, ph)
        row["thrust_range_n"] = [float(ph["U"][3].min()), float(ph["U"][3].max())]
    write(out / "summary.json", dict(
        status="flight_validated_tracker_gap_exceeded", case=case,
        summary=round_trip_summary(p, sol, "uprange", *lam), table=table, contacts=v["contacts"],
        flight_checks_passed=bool(all(r["passed"] for r in v["propagation"]) and all(c["passed"] for c in v["contacts"].values())),
        fixed_mass_passed=bool(v.get("fixed_mass", {}).get("passed")),
        tracker_summary={n: {k: r[k] for k in ("passed", "event_count", "max_update_gap_s")} for n, r in tr["reports"].items()},
        scope="Flight validated; star-tracker update gap exceeds the configured limit, so not an accepted flight"))
    # diagnostic trajectory + independent flight (never marked accepted)
    save_trajectory(out / "trajectory.npz", p, sol, "uprange", physical_identity(p),
                    dict(passed=False, tracker_validation="failed", diagnostic_only=True))
    rows, history, contacts = integrate(p, sol, terrain)
    np.savez_compressed(out / "independent_flight.npz",
                        time=np.concatenate([h["time"] for h in history]), r=np.concatenate([h["r"] for h in history]),
                        state=np.concatenate([h["state"] for h in history]),
                        hop=np.concatenate([np.full(len(h["time"]), h["hop"]) for h in history]))
    print(case, "summary written", flush=True)

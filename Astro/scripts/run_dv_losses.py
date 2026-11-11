"""Split each hop's finite-burn delta-v excess over Lambert into gravity loss, steering loss and the rest.

    .\\.venv\\Scripts\\python.exe -B Astro/scripts/run_dv_losses.py [case]

Run from the repository root after run_case.py has accepted the case (reads results/<case>/parameters.json,
trajectory.npz and summary.json).

Gravity loss: -s * integral(g . v_hat) dt; steering loss: integral (T/m)(1 - s cos(T, v)) dt, with s = +1 for the
ascent burn (rise, ascent stages) and s = -1 for the descent burn (braking, observation, terminal). g is central
gravity plus the centrifugal term of the Moon-fixed frame (Coriolis is perpendicular to v and does no work).
The remainder (finite-burn minus Lambert minus both losses) is the trajectory-shape / constraint cost.
Integrated over the saved collocation nodes (trapezoid). Writes results/<case>/dv_losses.json.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
case = sys.argv[1] if len(sys.argv) > 1 else "a2200_08"
out = ROOT / "results" / case
p = json.loads((out / "parameters.json").read_text())
mu, w = p["moon"]["mu"], p["moon"]["omega"]
rs = np.array(p["mission"]["r_start"])
d = np.load(out / "trajectory.npz")
meta = json.loads(str(d["metadata"].item()))
summ = json.loads((out / "summary.json").read_text())["summary"]
res = {}
for name, info in meta["phase_info"].items():
    kind = info["kind"]
    if kind == "coast":
        continue
    hop = info.get("hop", 2 if "2_" in name else 1)
    s = 1.0 if kind in ("rise", "ascent") else -1.0
    X, U, T = d[name + "_X"], d[name + "_U"], float(d[name + "_t"])
    t = np.linspace(0, T, X.shape[1])
    r = X[:3].T + rs; v = X[3:6].T; m = X[6]
    rn = np.linalg.norm(r, axis=1)
    g = -mu * r / rn[:, None] ** 3 + np.c_[w * w * r[:, 0], w * w * r[:, 1], np.zeros(len(r))]
    sp = np.linalg.norm(v, axis=1)
    vh = np.where(sp[:, None] > 1e-6, v / np.maximum(sp, 1e-12)[:, None], 0.0)
    th = U[3]; tv = U[:3].T
    cosa = np.einsum("ij,ij->i", tv / np.maximum(th, 1e-9)[:, None], vh)
    grav = -s * np.einsum("ij,ij->i", g, vh)
    steer = th / m * (1 - s * cosa)
    acc = th / m
    b = res.setdefault(hop, dict(thrust_dv=0.0, gravity=0.0, steering=0.0))
    b["thrust_dv"] += float(np.trapezoid(acc, t)); b["gravity"] += float(np.trapezoid(grav, t)); b["steering"] += float(np.trapezoid(steer, t))
outd = {}
for hop, b in sorted(res.items()):
    h = summ[f"hop{hop}"]
    lam, fin = h["lambert_dv"], h["finite_burn_dv"]
    outd[f"hop{hop}"] = dict(lambert=lam, finite=fin, excess=fin - lam, gravity_loss=b["gravity"], steering_loss=b["steering"],
                             other=fin - lam - b["gravity"] - b["steering"], integrated_thrust_dv=b["thrust_dv"])
(out / "dv_losses.json").write_text(json.dumps(outd, indent=1))
for k, v in outd.items():
    print(k, {a: round(x, 1) for a, x in v.items()})

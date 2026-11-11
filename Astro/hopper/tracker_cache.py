"""Versioned numeric-only trajectory cache; summaries are not sufficient input."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .postprocess import chain
from .phases import stage_nodes
from .problem import Solution


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def baseline_identity(p, branch, source_root):
    """Hash effective physical/numerical settings and the baseline implementation.

    Optical settings are excluded. Enabled stage-junction attitude continuity is
    included because it changes the flight constraints.
    """
    params = asdict(p)
    params.pop("star_tracker", None)
    st = p.star_tracker
    if st is not None and st.enabled and (
            any(p.engine.stage_count(k, hop=h) > 1 for h in (1, 2) for k in st.required_phase_kinds) or
            {"rise", "ascent"}.issubset(st.required_phase_kinds)):
        params["tracker_attitude_continuity"] = list(st.required_phase_kinds)
    source_root = Path(source_root)
    files = [path for path in sorted((source_root / "hopper").rglob("*.py"))
             if not path.name.startswith(("star_tracker", "tracker_"))]
    files.append(source_root / "scripts" / "run_roundtrip.py")
    source = hashlib.sha256()
    for path in files:
        source.update(path.relative_to(source_root).as_posix().encode())
        source.update(path.read_bytes().replace(b"\r\n", b"\n"))
    identity = dict(branch=branch, params=params, baseline_source_sha256=source.hexdigest())
    return hashlib.sha256(json.dumps(identity, default=json_value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_trajectory(path, p, sol, branch, identity, validation):
    if list(sol.phases) != chain(branch, p):
        raise ValueError("Trajectory phase/stage order does not match engine configuration")
    metadata = dict(version=2, branch=branch, baseline_identity=identity, validation=validation,
                    status=sol.status, success=sol.success, m_wet=sol.m_wet, z=sol.z,
                    iterations=sol.iterations, solve_time=sol.solve_time,
                    phase_info={name: {k: ph.get(k, 0 if k == "stage" else False) for k in
                                       ("kind", "powered", "branch", "stage", "relaxed")}
                                for name, ph in sol.phases.items()})
    for name, info in metadata["phase_info"].items():
        info["hop"] = sol.phases[name].get("hop", 2 if "2_" in name else 1)
    arrays = {"metadata": np.array(json.dumps(metadata, default=json_value))}
    for name in chain(branch, p):
        ph = sol.phases[name]
        for key in ("X", "U"):
            arrays[f"{name}_{key}"] = np.asarray(ph[key], dtype=float)
        arrays[f"{name}_t"] = np.array(ph["t"], dtype=float)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_trajectory(path, p, branch, identity):
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if (metadata.get("version") not in (1, 2) or metadata.get("baseline_identity") != identity or
                    metadata.get("branch") != branch):
                raise ValueError("Stale/incompatible trajectory cache; rerun with --refresh-baseline")
            if not metadata.get("success") or not metadata.get("validation", {}).get("passed"):
                raise ValueError("Trajectory cache is not a validated successful solution")
            phases = {}
            if metadata["version"] == 2 and list(metadata["phase_info"]) != chain(branch, p):
                raise ValueError("Stale/incompatible trajectory stage order")
            for name in chain(branch, p):
                info = metadata.get("phase_info", {}).get(name, {})
                kind = info.get("kind", name.split("2_")[0] if "2_" in name else name.split("_")[0])
                X, U, duration = data[f"{name}_X"], data[f"{name}_U"], float(data[f"{name}_t"])
                hop = int(info.get("hop", 2 if "2_" in name else 1))
                n = stage_nodes(p.solver.nodes[kind], p.engine.stage_count(kind, hop=hop)) + 1
                if (X.shape != (7, n) or U.shape != (4, n) or not np.isfinite(X).all() or
                        not np.isfinite(U).all() or not np.isfinite(duration) or duration <= 0):
                    raise ValueError("Invalid trajectory cache array shapes/values")
                phases[name] = dict(X=X.copy(), U=U.copy(), t=duration, powered=kind != "coast",
                                    kind=kind, branch=info.get("branch", branch if kind == "terminal" else None),
                                    stage=int(info.get("stage", 0)), relaxed=bool(info.get("relaxed", False)), hop=hop)
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Full versioned NPZ trajectory required; summary JSON is insufficient") from exc
    return Solution(phases, metadata["z"], metadata["status"], True, metadata["iterations"],
                    metadata["solve_time"], metadata["m_wet"]), metadata["validation"]

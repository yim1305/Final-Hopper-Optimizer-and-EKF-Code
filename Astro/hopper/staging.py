"""Relax, stage, refine: solve the round trip with fixed-thrust (non-pulsing) engines.

1. Relaxed solve: one stage per phase, every engine of the pool acts as one continuous throttle at the pool's
   average Isp. A smooth, well-conditioned problem that serves as the seed; its propellant use is not a bound.
2. Stage: where the relaxed thrust passes the fixed-thrust engines' bands, cut each multi-stage burn into the stages
   of the engine config (number of fixed-thrust engines firing, non-increasing along the burn) and resample the
   relaxed trajectory as the guess.
3. Refine: solve the staged problem from that guess. The stage durations are decision variables, so the optimizer
   moves the engine shutdown times; a stage it has no use for collapses to `limits.stage_duration_min`.
"""
from __future__ import annotations

import numpy as np
import time

from .params import POWERED_KINDS, Params
from .phases import stage_name, stage_nodes
from .problem import Problem


def _snap_counts(p: Params, ph: dict, counts: list):
    """Best non-increasing engine schedule for the whole relaxed thrust history.

    A running minimum incorrectly shuts every large engine off for the entire
    burn when its first relaxed node has low thrust (common after a coast).
    Dynamic programming instead fits the complete history to the allowed bands.
    """
    mode_r = p.engine.relaxed[ph["kind"]]
    name = mode_r.fixed[0][0]
    per_engine = p.engine.classes[name].thrust
    gamma = ph["U"][3]
    lower = np.array(counts)[:, None] * per_engine
    upper = lower + mode_r.pulse_capacity
    cost = ((gamma - np.clip(gamma, lower, upper)) / per_engine) ** 2
    cost[:, [0, -1]] *= 0.5
    best, parents = cost[:, 0].copy(), np.zeros(cost.shape, dtype=int)
    for i in range(1, len(gamma)):
        previous = best.copy()
        for k in range(len(counts)):
            parents[k, i] = np.argmin(previous[:k + 1])
            best[k] = cost[k, i] + previous[parents[k, i]]
    schedule = np.empty(len(gamma), dtype=int)
    k = int(np.argmin(best))
    for i in range(len(gamma) - 1, -1, -1):
        schedule[i] = counts[k]
        k = parents[k, i]
    return schedule


def _stage_edges(p: Params, ph: dict, counts: list) -> np.ndarray:
    """Stage boundary times (K + 1 values from 0 to the burn duration) from the snapped relaxed counts."""
    K, t_tot = len(counts), ph["t"]
    snapped = _snap_counts(p, ph, counts)
    tn = np.linspace(0.0, t_tot, len(snapped))
    # stage index of each node: the first stage whose count is not above the node's snapped count
    index = np.array([next(k for k, c in enumerate(counts) if c <= s) for s in snapped])
    # the boundary after stage k is the time of the first node that belongs to a later stage
    inner = [tn[np.argmax(index > k)] if np.any(index > k) else t_tot for k in range(K - 1)]
    edges = np.maximum.accumulate(np.array([0.0] + inner + [t_tot]))
    # every stage gets a little time: later stages at least twice their minimum, the first its phase minimum
    widths = np.diff(edges)
    floor = np.array([p.limits.phase_duration_bounds[ph["kind"]][0] * 1.05] +
                     [2.0 * p.limits.stage_duration_min] * (K - 1))
    widths = np.maximum(widths, floor)
    widths *= t_tot / widths.sum()
    return np.concatenate([[0.0], np.cumsum(widths)])


def _resample(ph: dict, times: np.ndarray):
    tn = np.linspace(0.0, ph["t"], ph["X"].shape[1])
    X = np.array([np.interp(times, tn, row) for row in ph["X"]])
    U = np.array([np.interp(times, tn, row) for row in ph["U"]])
    return X, U


def _clip_thrust(U: np.ndarray, mode) -> np.ndarray:
    """Hold the thrust magnitude inside the stage's band, keeping the thrust direction."""
    U = U.copy()
    G = np.clip(U[3], mode.gamma_lo, mode.gamma_hi)
    norm = np.linalg.norm(U[0:3], axis=0)
    U[0:3] = U[0:3] / np.where(norm > 1e-9, norm, 1.0) * G
    U[3] = G
    return U


def stage_guess(p: Params, relaxed_phases: dict, nodes: dict | None = None) -> dict:
    """Guess for the staged problem from the phases of a relaxed solution (`Solution.phases`)."""
    nodes = nodes or p.solver.nodes
    eng = p.engine
    guess = {}
    for name, ph in relaxed_phases.items():
        kind = ph["kind"]
        hop = ph.get("hop", 2 if "2_" in name else 1)
        if not ph["powered"]:
            guess[name] = dict(X=ph["X"], U=ph["U"], t=ph["t"])
            continue
        K = eng.stage_count(kind, hop=hop)
        if K == 1:
            guess[name] = dict(X=ph["X"], U=_clip_thrust(ph["U"], eng.mode(kind, hop=hop)), t=ph["t"])
            continue
        counts = [dict(m.fixed)[eng.relaxed[kind].fixed[0][0]] for m in eng.modes_for(hop)[kind]]
        edges = _stage_edges(p, ph, counts)
        n = stage_nodes(nodes[kind], K)
        for k in range(K):
            X, U = _resample(ph, np.linspace(edges[k], edges[k + 1], n + 1))
            guess[stage_name(name, k)] = dict(X=X, U=_clip_thrust(U, eng.mode(kind, k, hop=hop)), t=edges[k + 1] - edges[k])
    return guess


def solve_round_trip(p: Params, branch: str = "uprange", nodes: dict | None = None, mode: str | None = None,
                     print_level: int = 0, guess: dict | None = None, m_wet0: float | None = None,
                     progress=None):
    """Solve the round trip. Returns (solution, relaxed solution or None, problem).

    A staged config uses a relaxed seed and fixed-engine solve on a smaller mesh,
    followed by refinement on the requested mesh. A supplied guess skips seeding.
    Failed stages return immediately; they never seed another optimization.
    """
    target = dict(nodes or p.solver.nodes)
    seed = {k: min(v, (p.solver.seed_nodes or target)[k]) for k, v in target.items()}
    history = []

    def run(label, mesh, relaxed=False, initial=None, mass=None):
        if progress:
            progress(f"  {label}: starting", flush=True)
        start = time.perf_counter()
        pr = Problem(p, branch, nodes=mesh, mode=mode, relaxed=relaxed)
        pr.set_guess(initial, m_wet0=mass)
        sol = pr.solve(print_level)
        history.append(dict(stage=label, status=sol.status, success=sol.success,
                            iterations=sol.iterations, wall_s=time.perf_counter() - start,
                            m_wet=sol.m_wet, primal_infeasibility=sol.primal_infeasibility,
                            dual_infeasibility=sol.dual_infeasibility))
        pr.solve_stages = history.copy()
        if progress:
            progress(f"  {label}: {sol.status}, {sol.iterations} iterations, {history[-1]['wall_s']:.2f} s", flush=True)
        return sol, pr

    relaxed_sol = None
    if guess is None and any(p.engine.stage_count(k, hop=h) > 1 for h in (1, 2) for k in POWERED_KINDS):
        relaxed_sol, pr = run("relaxed seed", seed, relaxed=True)
        if not relaxed_sol.success:
            return relaxed_sol, relaxed_sol, pr
        guess = stage_guess(p, relaxed_sol.phases, seed)
        sol, pr = run("fixed-engine seed", seed, initial=guess, mass=relaxed_sol.m_wet)
        if not sol.success or seed == target:
            return sol, relaxed_sol, pr
        guess, m_wet0 = Problem.guess_from_solution(sol), sol.m_wet
    sol, pr = run("full mesh", target, initial=guess, mass=m_wet0)
    return sol, relaxed_sol, pr

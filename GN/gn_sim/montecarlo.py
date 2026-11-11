"""Run one sensor case on one hop for N Monte Carlo runs (vectorized) and summarize."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from .config import NavConfig
from .ekf import BA, BG, R, TH, V, BatchEKF
from .rotations import exp_so3
from .sensors import (draw_imu_biases, imu_measure, tracker_covariance, tracker_frames, tracker_measure,
                      trn_availability, trn_measure)

ARCSEC = np.pi / 180 / 3600


def initial_covariance(cfg: NavConfig, att_cov):
    P0 = np.zeros((15, 15))
    P0[R, R] = cfg.position_sigma0**2 * np.eye(3)
    P0[V, V] = cfg.velocity_sigma0**2 * np.eye(3)
    P0[TH, TH] = att_cov
    P0[BA, BA] = cfg.imu.accel_bias**2 * np.eye(3)
    P0[BG, BG] = cfg.imu.gyro_bias**2 * np.eye(3)
    return P0


def run_case(truth, case, cfg: NavConfig, case_def: dict, tracker_avail, runs=None, seed=None,
             trn_sigma=None, trn_rate=None, store=True, case_id="C"):
    n = runs or cfg.runs
    seed = cfg.seed if seed is None else seed
    trn_sigma = cfg.trn.sigma if trn_sigma is None else trn_sigma
    trn_rate = cfg.trn.rate if trn_rate is None else trn_rate
    rng = np.random.default_rng(np.random.SeedSequence([seed, truth.hop, sum(map(ord, case_id))]))

    frames = tracker_frames(case.mounts)
    covs = [tracker_covariance(f, cfg) for f in frames]
    heads = list(case_def.get("trackers", []))
    use_trn = bool(case_def.get("trn", False))
    trn_ok = trn_availability(truth, cfg, trn_rate) if use_trn else np.zeros(len(truth.t), bool)
    R_trn = trn_sigma**2 * np.eye(3)

    # truth biases and initial estimate errors drawn from the filter's own initial covariance
    bg, ba = draw_imu_biases(rng, n, cfg)
    P0 = initial_covariance(cfg, covs[0])
    L = np.linalg.cholesky(P0[:9, :9] + 1e-30 * np.eye(9))
    e0 = rng.normal(size=(n, 9)) @ L.T
    r_hat = truth.r_I[0] - e0[:, R]
    v_hat = truth.v_I[0] - e0[:, V]
    C_hat = truth.C_IB[0] @ exp_so3(-e0[:, TH])          # C_true = C_hat exp(dtheta)
    ekf = BatchEKF(r_hat, v_hat, C_hat, P0, case.p.moon.mu, cfg.imu.gyro_arw, cfg.imu.accel_vrw)

    store_every = cfg.steps_per(cfg.store_rate)
    cps = truth.checkpoints
    keep = set(range(0, len(truth.t), store_every)) | set(cps.values()) if store else set(cps.values())
    hist = {k: [] for k in ("idx", "e_r", "e_v", "e_th", "P_r", "P_v", "P_th", "e_ba", "e_bg")}
    n_updates = dict(tracker=[0 for _ in heads], trn=0)

    def record(k):
        er, ev, eth, eba, ebg = ekf.errors(truth.r_I[k], truth.v_I[k], truth.C_IB[k], ba, bg)
        A = truth.local_basis_I[k]
        hist["idx"].append(k)
        hist["e_r"].append(er @ A)
        hist["e_v"].append(ev @ A)
        hist["e_th"].append(eth)
        hist["e_ba"].append(eba)
        hist["e_bg"].append(ebg)
        hist["P_r"].append(A.T @ ekf.P[:, R, R] @ A)
        hist["P_v"].append(A.T @ ekf.P[:, V, V] @ A)
        hist["P_th"].append(ekf.P[:, TH, TH].copy())

    record(0)
    for k in range(len(truth.dt)):
        dth_m, dv_m = imu_measure(rng, truth, k, bg, ba, cfg)
        ekf.propagate(dth_m, dv_m, truth.dt[k])
        kk = k + 1
        for slot, j in enumerate(heads):
            if tracker_avail[kk, j]:
                ekf.update_attitude(tracker_measure(rng, truth.C_IB[kk], frames[j], cfg, n), covs[j])
                n_updates["tracker"][slot] += 1
        if trn_ok[kk]:
            ekf.update_position(trn_measure(rng, truth.r_I[kk], trn_sigma, n), R_trn)
            n_updates["trn"] += 1
        if kk in keep:
            record(kk)

    H = {k: np.asarray(v) for k, v in hist.items()}
    return dict(case_id=case_id, label=case_def.get("label", case_id), hop=truth.hop, runs=n,
                trn_sigma_m=trn_sigma, trn_rate_hz=trn_rate, heads=heads, use_trn=use_trn,
                updates=n_updates, hist=H, t=truth.t[H["idx"]] - truth.t[0],
                checkpoints=summarize_checkpoints(truth, H), consistency=consistency(H))


def _block_stats(e, P):
    """e: (n,3) errors, P: (n,3,3) covariances in the same frame."""
    filt = 3 * np.sqrt(np.mean(np.diagonal(P, axis1=1, axis2=2), axis=0))
    mc = 3 * np.sqrt(np.mean(e**2, axis=0))
    return filt, mc


def summarize_checkpoints(truth, H):
    out = {}
    pos = {int(k): i for i, k in enumerate(H["idx"])}
    for name, k in truth.checkpoints.items():
        i = pos[k]
        fr, mr = _block_stats(H["e_r"][i], H["P_r"][i])
        fv, mv = _block_stats(H["e_v"][i], H["P_v"][i])
        fa, ma = _block_stats(H["e_th"][i], H["P_th"][i])
        out[name] = dict(
            time_s=float(truth.t[k] - truth.t[0]), altitude_m=float(truth.altitude[k]),
            pos_3sigma_filter_m=fr.tolist(), pos_3sigma_mc_m=mr.tolist(),
            pos_horizontal_3sigma_filter_m=float(np.hypot(fr[0], fr[1])),
            pos_horizontal_3sigma_mc_m=float(np.hypot(mr[0], mr[1])),
            vel_3sigma_filter_m_s=fv.tolist(), vel_3sigma_mc_m_s=mv.tolist(),
            vel_horizontal_3sigma_mc_m_s=float(np.hypot(mv[0], mv[1])),
            att_3sigma_filter_arcsec=(fa / ARCSEC).tolist(), att_3sigma_mc_arcsec=(ma / ARCSEC).tolist())
    return out


def consistency(H):
    """Fraction of (epoch, run, axis) samples with |error| <= 3 sigma, and run-averaged NEES per block."""
    out = {}
    n = H["e_r"].shape[1]
    lo, hi = chi2.ppf([0.025, 0.975], 3 * n) / n
    for key, eb, Pb in (("position", "e_r", "P_r"), ("velocity", "e_v", "P_v"), ("attitude", "e_th", "P_th")):
        e, P = H[eb][1:], H[Pb][1:]          # skip the initialization epoch
        sig = np.sqrt(np.diagonal(P, axis1=2, axis2=3))
        inside = np.abs(e) <= 3 * sig
        nees = np.einsum("tni,tnij,tnj->tn", e, np.linalg.inv(P), e).mean(axis=1)
        out[key] = dict(fraction_within_3sigma=float(inside.mean()),
                        mean_nees=float(nees.mean()), expected_nees=3.0,
                        fraction_epochs_nees_in_95pct_band=float(np.mean((nees >= lo) & (nees <= hi))))
    return out

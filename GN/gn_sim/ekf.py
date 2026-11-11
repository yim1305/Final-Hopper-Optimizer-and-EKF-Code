"""15-state error-state (multiplicative) EKF, vectorized over Monte Carlo runs.

Error state: [dr(3), dv(3), dtheta(3), dba(3), dbg(3)], with r, v in the inertial frame I,
C_true = C_hat exp([dtheta x]) (body-frame attitude error), d(.) = true - estimate.

The filter has no rotational dynamics: attitude is propagated by integrating gyro delta-theta
(strapdown) and corrected by star tracker updates.
"""
from __future__ import annotations

import numpy as np

from .rotations import exp_so3, log_so3, skew

R, V, TH, BA, BG = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15)
I3 = np.eye(3)


def gravity(r, mu):
    rn = np.linalg.norm(r, axis=-1, keepdims=True)
    return -mu * r / rn**3


def gravity_gradient(r, mu):
    rn = np.linalg.norm(r, axis=-1)
    u = r / rn[:, None]
    return (mu / rn**3)[:, None, None] * (3 * np.einsum("ni,nj->nij", u, u) - I3)


class BatchEKF:
    def __init__(self, r, v, C, P0, mu, gyro_arw, accel_vrw):
        n = r.shape[0]
        self.r, self.v, self.C = r.copy(), v.copy(), C.copy()
        self.ba, self.bg = np.zeros((n, 3)), np.zeros((n, 3))
        self.P = np.broadcast_to(P0, (n, 15, 15)).copy()
        self.mu, self.arw2, self.vrw2 = mu, gyro_arw**2, accel_vrw**2
        self.n = n

    # ------------------------------------------------------------ propagation
    def propagate(self, dth_m, dv_m, h):
        dth = dth_m - self.bg * h
        dv = dv_m - self.ba * h
        C_mid = self.C @ exp_so3(dth / 2)
        dvI = np.einsum("nij,nj->ni", C_mid, dv)
        r0, v0 = self.r, self.v
        g0 = gravity(r0, self.mu)
        r_pred = r0 + v0 * h + 0.5 * (dvI + g0 * h) * h
        g1 = gravity(r_pred, self.mu)
        v1 = v0 + dvI + 0.5 * (g0 + g1) * h
        self.r = r0 + 0.5 * (v0 + v1) * h
        self.v = v1
        self.C = self.C @ exp_so3(dth)

        F = np.zeros((self.n, 15, 15))
        F[:, R, V] = I3
        F[:, V, R] = gravity_gradient(r0, self.mu)
        F[:, V, TH] = -C_mid @ skew(dv / h)
        F[:, V, BA] = -C_mid
        F[:, TH, TH] = -skew(dth / h)
        F[:, TH, BG] = -I3
        Fd = F * h
        Phi = np.eye(15) + Fd + 0.5 * Fd @ Fd
        P = Phi @ self.P @ np.swapaxes(Phi, 1, 2)
        P[:, V, V] += self.vrw2 * h * I3
        P[:, TH, TH] += self.arw2 * h * I3
        self.P = 0.5 * (P + np.swapaxes(P, 1, 2))

    # ------------------------------------------------------------ updates (H selects one 3-state block)
    def _update(self, block, z, Rm):
        PHt = self.P[:, :, block]                       # (n, 15, 3)
        S = self.P[:, block, block] + Rm                # (n, 3, 3)
        K = PHt @ np.linalg.inv(S)                      # (n, 15, 3)
        dx = np.einsum("nij,nj->ni", K, z)
        KH_P = K @ self.P[:, block, :]
        P = self.P - KH_P - np.swapaxes(KH_P, 1, 2) + K @ S @ np.swapaxes(K, 1, 2)   # Joseph-equivalent
        self.P = 0.5 * (P + np.swapaxes(P, 1, 2))
        self._inject(dx)
        return z, S

    def _inject(self, dx):
        self.r = self.r + dx[:, R]
        self.v = self.v + dx[:, V]
        self.C = self.C @ exp_so3(dx[:, TH])
        self.ba = self.ba + dx[:, BA]
        self.bg = self.bg + dx[:, BG]

    def update_attitude(self, C_meas, Rm):
        z = log_so3(np.swapaxes(self.C, 1, 2) @ C_meas)
        return self._update(TH, z, Rm)

    def update_position(self, r_meas, Rm):
        return self._update(R, r_meas - self.r, Rm)

    # ------------------------------------------------------------ diagnostics
    def errors(self, r_true, v_true, C_true, ba_true, bg_true):
        return (r_true - self.r, v_true - self.v, log_so3(np.swapaxes(self.C, 1, 2) @ C_true),
                ba_true - self.ba, bg_true - self.bg)

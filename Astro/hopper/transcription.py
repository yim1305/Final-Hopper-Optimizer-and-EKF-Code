"""Direct transcription. Swappable: any class with `defects(f, X, U, tdur, n)` can replace this."""
from __future__ import annotations

import casadi as ca


class Transcription:
    def defects(self, f: ca.Function, X, U, tdur, n: int):
        raise NotImplementedError


class HermiteSimpson(Transcription):
    """Separated Hermite-Simpson on normalized time tau in [0, 1] with n intervals.

    dx/dtau = tdur * f(x, u). Controls are linear between nodes, so u_mid is the node average.
    X: 7 x (n+1), U: 4 x (n+1), tdur: nondimensional duration (scalar, may be a decision variable).
    Returns a 7 x n matrix of defects that must be zero.
    """

    def defects(self, f, X, U, tdur, n):
        h = tdur / n
        F = f.map(n + 1)(X, U)
        Xk, Xk1 = X[:, 0:n], X[:, 1:n + 1]
        Fk, Fk1 = F[:, 0:n], F[:, 1:n + 1]
        Xm = 0.5 * (Xk + Xk1) + (h / 8.0) * (Fk - Fk1)
        Um = 0.5 * (U[:, 0:n] + U[:, 1:n + 1])
        Fm = f.map(n)(Xm, Um)
        return Xk1 - Xk - (h / 6.0) * (Fk + 4.0 * Fm + Fk1)

"""Gravity models. `accel(r_abs)` works on CasADi symbols and numpy arrays alike."""


class Gravity:
    def accel(self, r_abs):
        raise NotImplementedError


class CentralGravity(Gravity):
    def __init__(self, mu: float):
        self.mu = mu

    def accel(self, r_abs):
        x, y, z = r_abs[0], r_abs[1], r_abs[2]
        r3 = (x * x + y * y + z * z) ** 1.5
        return [-self.mu * x / r3, -self.mu * y / r3, -self.mu * z / r3]

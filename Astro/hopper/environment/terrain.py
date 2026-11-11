"""Terrain model and altitude helper."""


class Terrain:
    def height(self, r_abs):
        """Terrain height above the reference sphere beneath r_abs [m]."""
        raise NotImplementedError


class FlatTerrain(Terrain):
    """PDR terrain model: reference sphere, zero height everywhere."""

    def height(self, r_abs):
        return 0.0


def altitude(r_start, radius, r, terrain):
    """h = |r_abs| - R - terrain.height(r_abs), with r_abs = r_start + r.

    Written as (|r_abs|^2 - R^2)/(|r_abs| + R) to avoid cancelling two Moon-radius numbers.
    Works for numpy arrays and CasADi expressions (r indexable by [0], [1], [2]).
    """
    components = [r[i, :] for i in range(3)] if len(getattr(r, "shape", ())) == 2 else r
    ax, ay, az = (r_start[i] + components[i] for i in range(3))
    n2 = ax * ax + ay * ay + az * az
    n = n2 ** 0.5
    h = (n2 - radius ** 2) / (n + radius)
    return h - terrain.height((ax, ay, az))

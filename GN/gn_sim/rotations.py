"""Batched rotation helpers (numpy). Vectors are (..., 3); matrices (..., 3, 3)."""
import numpy as np
from scipy.spatial.transform import Rotation


def skew(v):
    v = np.asarray(v, float)
    out = np.zeros(v.shape[:-1] + (3, 3))
    out[..., 0, 1], out[..., 0, 2] = -v[..., 2], v[..., 1]
    out[..., 1, 0], out[..., 1, 2] = v[..., 2], -v[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -v[..., 1], v[..., 0]
    return out


def exp_so3(phi):
    """Rotation matrix for rotation vector phi (Rodrigues), batched."""
    phi = np.asarray(phi, float)
    angle = np.linalg.norm(phi, axis=-1)[..., None, None]
    K = skew(phi)
    small = angle < 1e-8
    safe = np.where(small, 1.0, angle)
    a = np.where(small, 1.0 - angle**2 / 6, np.sin(safe) / safe)
    b = np.where(small, 0.5 - angle**2 / 24, (1 - np.cos(safe)) / safe**2)
    return np.eye(3) + a * K + b * (K @ K)


def log_so3(C):
    """Rotation vector of rotation matrix C, batched."""
    C = np.asarray(C, float)
    shape = C.shape[:-2]
    return Rotation.from_matrix(C.reshape(-1, 3, 3)).as_rotvec().reshape(shape + (3,))


def rot_z(angle):
    angle = np.asarray(angle, float)
    c, s = np.cos(angle), np.sin(angle)
    out = np.zeros(angle.shape + (3, 3))
    out[..., 0, 0], out[..., 0, 1], out[..., 1, 0], out[..., 1, 1] = c, -s, s, c
    out[..., 2, 2] = 1.0
    return out


def frame_from_boresight(b):
    """Orthonormal tracker frame (columns x, y, z) with z along boresight b (body)."""
    z = np.asarray(b, float) / np.linalg.norm(b)
    helper = np.array([0., 1., 0.]) if abs(z[1]) < 0.9 else np.array([1., 0., 0.])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))

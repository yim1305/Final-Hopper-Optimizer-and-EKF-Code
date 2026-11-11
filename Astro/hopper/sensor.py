"""Square-FOV camera geometry; thrust is the body-up proxy and roll follows a fixed heading."""
import casadi as ca
import numpy as np


def camera_axes(thrust, heading):
    """Camera x, y and boresight. One image axis follows the projected Moon-fixed downrange heading."""
    symbolic = isinstance(thrust, (ca.MX, ca.SX, ca.DM))
    norm = ca.norm_2 if symbolic else np.linalg.norm
    dot = ca.dot if symbolic else np.dot
    cross = ca.cross if symbolic else np.cross
    boresight = -thrust / norm(thrust)
    image_x = heading - dot(heading, boresight) * boresight
    image_x = image_x / norm(image_x)
    image_y = cross(boresight, image_x)
    return image_x, image_y, boresight


def image_angles(thrust, heading, line_of_sight):
    """Signed horizontal/vertical look angles and forward depth, for numeric validation and plots."""
    image_x, image_y, boresight = camera_axes(thrust, heading)
    depth = float(boresight @ line_of_sight)
    return (float(np.degrees(np.arctan2(image_x @ line_of_sight, depth))),
            float(np.degrees(np.arctan2(image_y @ line_of_sight, depth))), depth)

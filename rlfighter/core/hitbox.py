import math

import numpy as np


def _facing_vector(facing: float) -> np.ndarray:
    return np.array([math.cos(facing), math.sin(facing)], dtype=np.float32)


def _perp(fv: np.ndarray) -> np.ndarray:
    return np.array([-fv[1], fv[0]], dtype=np.float32)


def point_in_sector(
    point: np.ndarray,
    origin: np.ndarray,
    facing: float,
    radius: float,
    arc_deg: float,
) -> bool:
    """Check if point lies inside a circular sector."""
    diff = point - origin
    dist_sq = float(np.dot(diff, diff))
    if dist_sq > radius * radius:
        return False
    angle = math.atan2(diff[1], diff[0])
    half_arc = math.radians(arc_deg / 2.0)
    delta = _angle_diff(angle, facing)
    return abs(delta) <= half_arc


def point_in_oriented_rect(
    point: np.ndarray,
    origin: np.ndarray,
    facing: float,
    width: float,
    length: float,
) -> bool:
    """Check if point lies inside an oriented rectangle extending forward from origin."""
    fv = _facing_vector(facing)
    pv = _perp(fv)
    diff = point - origin
    along = float(np.dot(diff, fv))
    across = float(np.dot(diff, pv))
    return 0.0 <= along <= length and abs(across) <= width / 2.0


def _angle_diff(a: float, b: float) -> float:
    """Signed smallest angle difference a - b, wrapped to [-pi, pi]."""
    diff = a - b
    while diff > math.pi:
        diff -= 2.0 * math.pi
    while diff < -math.pi:
        diff += 2.0 * math.pi
    return diff

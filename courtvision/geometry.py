"""2-D projective geometry helpers (homogeneous lines and points)."""
import numpy as np


def line_through(p, q) -> np.ndarray:
    """Homogeneous line through two image points."""
    return np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])


def line_from_fit(origin, direction) -> np.ndarray:
    """Homogeneous line from a point and a direction vector (cv2.fitLine output)."""
    (x0, y0), (vx, vy) = origin, direction
    return np.array([-vy, vx, vy * x0 - vx * y0], dtype=np.float64)


def intersect(l1, l2, eps: float = 1e-8) -> np.ndarray:
    """Intersection of two homogeneous lines. Raises ValueError if (nearly) parallel."""
    p = np.cross(l1, l2)
    if abs(p[2]) < eps:
        raise ValueError("Lines are nearly parallel")
    return p[:2] / p[2]


def point_line_distance(pt, p, q) -> float:
    """Perpendicular distance from `pt` to the infinite line through p and q."""
    d = np.asarray(q, np.float64) - np.asarray(p, np.float64)
    n = float(np.hypot(d[0], d[1]))
    if n < 1e-8:
        return float("inf")
    return abs(d[0] * (pt[1] - p[1]) - d[1] * (pt[0] - p[0])) / n


def line_angle(p1, p2) -> float:
    return float(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))


def angle_difference(a: float, b: float) -> float:
    """Unsigned difference between two undirected line angles, in [0, pi/2]."""
    return abs((a - b + np.pi / 2) % np.pi - np.pi / 2)


def corner_errors(a, b):
    """Per-corner Euclidean distance plus (mean, max)."""
    e = np.linalg.norm(np.asarray(a, np.float64) - np.asarray(b, np.float64), axis=1)
    return e, float(e.mean()), float(e.max())

"""Seeded court-line detection and lane-corner reconstruction (original cells 34–40).

Lines are found by keeping Hough segments close (angle + distance) to a *seed* line
and fitting a robust line through their endpoints. Seeds come either from manual
clicks (`annotations.seed_lines`) or from predicted lane corners (`seed_lines_from_corners`).
This is therefore refinement around a prior, not prior-free detection.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import angle_difference, intersect, line_angle, line_from_fit, point_line_distance

LINE_NAMES = ("baseline", "upper_lane", "lower_lane", "free_throw")

# (max_angle_deg, max_distance_px) per line when seeded from manual clicks (cells 35 & 37).
SEED_TOLERANCES = {"baseline": (8, 14), "upper_lane": (8, 14), "lower_lane": (8, 14), "free_throw": (12, 18)}
# Tolerances used when seeded from a propagated prediction (cell 39).
PRIOR_TOLERANCES = {name: (10, 20) for name in LINE_NAMES}


@dataclass(frozen=True)
class HoughParams:
    blur_ksize: int = 5
    blur_sigma: float = 1.2
    canny_low: int = 50
    canny_high: int = 150
    threshold: int = 45
    min_line_length: int = 35
    max_line_gap: int = 25


@dataclass
class LaneResult:
    corners: np.ndarray | None          # 4x2 in LANE_CORNER_ORDER, or None
    status: str
    counts: dict = field(default_factory=dict)  # segments supporting each line
    lines: dict = field(default_factory=dict)   # name -> homogeneous line


def detect_segments(frame, mask=None, params: HoughParams = HoughParams()):
    """Canny + probabilistic Hough. Returns an (N, 4) int array or None.

    `mask` (uint8, 255 = keep) optionally restricts edges to e.g. the court region.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (params.blur_ksize, params.blur_ksize), params.blur_sigma)
    edges = cv2.Canny(blurred, params.canny_low, params.canny_high)
    if mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=mask)
    seg = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=params.threshold,
                          minLineLength=params.min_line_length, maxLineGap=params.max_line_gap)
    return None if seg is None else seg.reshape(-1, 4)


def fit_seeded_line(seed, segments, max_angle_deg=8, max_distance=14, min_segment_length=25):
    """Fit a line to segments near `seed` (2x2 points). Returns ((origin, direction), n_segments) or (None, 0).

    Behaviour-identical to the original `detect_reference_line` (cell 35).
    """
    p1, p2 = np.asarray(seed, np.float64)
    ref_angle = line_angle(p1, p2)
    accepted = []
    for x1, y1, x2, y2 in segments:
        a, b = np.array([x1, y1], np.float64), np.array([x2, y2], np.float64)
        if np.linalg.norm(b - a) < min_segment_length:
            continue
        if np.degrees(angle_difference(line_angle(a, b), ref_angle)) > max_angle_deg:
            continue
        if point_line_distance((a + b) / 2, p1, p2) > max_distance:
            continue
        accepted.extend([a, b])
    if len(accepted) < 4:
        return None, 0
    pts = np.asarray(accepted, np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    return (np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)), len(pts) // 2


def seed_lines_from_corners(corners) -> dict:
    """Seed lines from 4 lane corners in LANE_CORNER_ORDER (cell 39)."""
    c = np.asarray(corners, np.float64)
    return {"baseline": c[[0, 1]], "upper_lane": c[[1, 2]], "lower_lane": c[[0, 3]], "free_throw": c[[2, 3]]}


def reconstruct_lane(segments, seeds: dict, tolerances: dict = SEED_TOLERANCES) -> LaneResult:
    """Fit all four lane lines and intersect them into lane corners (cells 35–38)."""
    if segments is None or len(segments) == 0:
        return LaneResult(None, "no_segments")
    counts, lines = {}, {}
    for name in LINE_NAMES:
        ang, dist = tolerances[name]
        fitted, n = fit_seeded_line(seeds[name], segments, ang, dist)
        counts[name] = n
        if fitted is None:
            return LaneResult(None, f"{name}_failed", counts, lines)
        lines[name] = line_from_fit(*fitted)
    try:
        corners = np.array([
            intersect(lines["baseline"], lines["lower_lane"]),
            intersect(lines["baseline"], lines["upper_lane"]),
            intersect(lines["upper_lane"], lines["free_throw"]),
            intersect(lines["lower_lane"], lines["free_throw"]),
        ], dtype=np.float64)
    except ValueError:
        return LaneResult(None, "intersection_failed", counts, lines)
    if not np.isfinite(corners).all():
        return LaneResult(None, "invalid_corners", counts, lines)
    return LaneResult(corners, "ok", counts, lines)


def refine_lane_from_prior(frame, prior_corners, max_correction_px=100.0,
                           tolerances: dict = PRIOR_TOLERANCES, mask=None) -> LaneResult:
    """Re-detect lane lines around a predicted lane (original cell 39 `refine_frame_calibration`).

    Rejects the result if any corner moves more than `max_correction_px` from the prior.
    """
    res = reconstruct_lane(detect_segments(frame, mask), seed_lines_from_corners(prior_corners), tolerances)
    if res.corners is None:
        return res
    corr = np.linalg.norm(res.corners - np.asarray(prior_corners, np.float64), axis=1)
    res.counts["mean_correction_px"] = float(corr.mean())
    res.counts["max_correction_px"] = float(corr.max())
    if corr.max() > max_correction_px:
        return LaneResult(None, "correction_too_large", res.counts, res.lines)
    return res

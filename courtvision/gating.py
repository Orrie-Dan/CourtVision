"""Acceptance gates for per-frame calibration (Stage 2).

A propagation step is accepted only if (a) the frame-to-frame motion estimate looks like
court-plane camera motion and (b) the lane it predicts is a plausible continuation of the
last accepted lane. Thresholds are PROVISIONAL: they are set from the Stage 1 diagnostics
(frames 101-115 only) and must be re-tuned once more reference frames are annotated.
"""
from dataclasses import dataclass, replace

import cv2
import numpy as np


@dataclass(frozen=True)
class GateParams:
    # Motion gates (on top of estimate_motion's own status gating).
    # Stage 1 pinned full-frame fits: 0.47-0.71 static inliers. Stage 2 court-only slow pans (~1.4 px/frame,
    # visually on-lane): up to 0.32 static. 0.20 falsely rejected 5 such frames; 0.40 separates the two.
    max_static_inlier_share: float = 0.40
    static_check_min_shift_px: float = 2.0  # ...while the fit still moved test points 5-21 px
    min_inlier_hull_frac: float = 0.03     # inlier convex hull / frame area; rejects clustered fits
    # Lane geometry gates, relative to the last accepted lane.
    max_lane_step_px: float = 40.0         # per frame of gap; Stage 1 lane moved ~10 px/frame
    min_area_ratio: float = 0.80
    max_area_ratio: float = 1.25
    frame_margin_frac: float = 0.25        # corners may leave the frame by this fraction of its size
    # Recovery.
    max_refine_correction_px: float = 25.0  # refinement may move a corner at most this far
    max_consecutive_rejections: int = 5     # then the track is "lost" and propagation stops


@dataclass(frozen=True)
class ReanchorParams:
    """Stage 3: periodic bounded refinement of a propagated lane (re-anchoring).

    Stage 2 measured ~0.5 px/frame of drift at the free-throw end after frame 115, so a fix every
    10 frames should find the painted lines within a few px of the prediction. The tight correction
    limit is what keeps this from snapping to a different line; it is still seeded refinement.
    PROVISIONAL: compare every_n = 5 / 10 / 15 against the manual references on Kaggle.
    """
    every_n: int = 10                   # attempt once this many frames have passed since the last fix
    max_correction_px: float = 10.0     # refinement may move a corner at most this far from the prediction
    min_line_segments: int = 3          # Hough segments supporting EACH of the 4 lane lines
    min_area_ratio: float = 0.93        # refined / predicted lane area: scale must roughly agree
    max_area_ratio: float = 1.07

    def refine_gate(self, gate: "GateParams") -> "GateParams":
        """`gate` with the refinement correction limit tightened for re-anchoring."""
        return replace(gate, max_refine_correction_px=self.max_correction_px)


def reanchor_failures(refined, predicted, counts: dict, params: ReanchorParams = ReanchorParams()):
    """Reasons a refined lane must NOT replace the propagated one ([] = accept). Returns (failures, metrics)."""
    r = np.asarray(refined, np.float64)
    p = np.asarray(predicted, np.float64)
    if not is_convex_quad(r):
        return ["reanchor_not_convex"], {}
    corr = np.linalg.norm(r - p, axis=1)
    area_ratio = abs(cv2.contourArea(r.astype(np.float32))) / abs(cv2.contourArea(p.astype(np.float32)))
    support = min((counts.get(name, 0) for name in ("baseline", "upper_lane", "lower_lane", "free_throw")))
    met = {"reanchor_max_correction_px": round(float(corr.max()), 2),
           "reanchor_area_ratio": round(area_ratio, 3), "reanchor_min_segments": int(support)}
    fails = []
    if corr.max() > params.max_correction_px:
        fails.append("reanchor_correction_too_large")
    if support < params.min_line_segments:
        fails.append("reanchor_weak_support")
    if not params.min_area_ratio <= area_ratio <= params.max_area_ratio:
        fails.append("reanchor_scale_disagrees")
    return fails, met


def motion_gate_failures(metrics: dict, gate: GateParams = GateParams()) -> list[str]:
    """Reasons an `estimate_motion` result fails the Stage 2 motion gates ([] = pass)."""
    if metrics.get("status") != "ok":
        return [f"motion_{metrics.get('status', 'unknown')}"]
    fails = []
    # Mostly-static inliers are only suspicious if the fitted motion nevertheless moves the image
    # (the Stage 1 signature: a fit pinned to a static graphic). A genuinely stationary camera has
    # static inliers AND ~0 shift, and must pass. Metrics alone cannot tell a large static overlay
    # from a stationary camera; overlays are handled by the feature mask, not here.
    if (metrics.get("static_inlier_share", 0.0) > gate.max_static_inlier_share
            and metrics.get("max_test_point_shift_px", 0.0) > gate.static_check_min_shift_px):
        fails.append("static_inliers")
    if metrics.get("inlier_hull_frac", 0.0) < gate.min_inlier_hull_frac:
        fails.append("clustered_inliers")
    return fails


def is_convex_quad(corners, min_area: float = 1.0) -> bool:
    """True if the 4 points, in order, form a non-degenerate convex quadrilateral."""
    c = np.asarray(corners, np.float64)
    if c.shape != (4, 2) or not np.isfinite(c).all():
        return False
    edges = np.roll(c, -1, axis=0) - c
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if not ((cross > 0).all() or (cross < 0).all()):
        return False
    return abs(cv2.contourArea(c.astype(np.float32))) >= min_area


def lane_gate(corners, prev_corners, image_size, gap: int = 1, gate: GateParams = GateParams()):
    """Check a candidate lane against the last accepted one.

    image_size = (W, H). gap = frames since the last accepted lane (the step limit scales with it).
    Returns (failures list, metrics dict).
    """
    c = np.asarray(corners, np.float64)
    p = np.asarray(prev_corners, np.float64)
    if not is_convex_quad(c):
        return ["lane_not_convex"], {}
    w, h = image_size
    mx, my = gate.frame_margin_frac * w, gate.frame_margin_frac * h
    step = float(np.linalg.norm(c - p, axis=1).max())
    area_ratio = abs(cv2.contourArea(c.astype(np.float32))) / abs(cv2.contourArea(p.astype(np.float32)))
    met = {"lane_step_px": round(step, 2), "lane_area_ratio": round(area_ratio, 3)}
    fails = []
    if not ((c[:, 0] >= -mx) & (c[:, 0] <= w + mx) & (c[:, 1] >= -my) & (c[:, 1] <= h + my)).all():
        fails.append("lane_outside_frame")
    if step > gate.max_lane_step_px * gap:
        fails.append("lane_step_too_large")
    if not gate.min_area_ratio <= area_ratio <= gate.max_area_ratio:
        fails.append("lane_area_change")
    return fails, met

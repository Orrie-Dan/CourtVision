"""Frame-to-frame camera motion via ORB + RANSAC homography (original cells 18, 20, 24, 28).

A high inlier ratio only says the matched points agree on *some* homography; it does
not prove that homography is the court-plane motion. The extra diagnostics returned
here (static / bottom-band / off-court inlier shares) exist to expose that.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionParams:
    n_features: int = 3000
    ratio_test: float = 0.75
    ransac_thresh_px: float = 3.0
    min_matches: int = 30
    min_inliers: int = 25
    min_inlier_ratio: float = 0.55
    max_test_point_shift_px: float = 150.0  # cell 20 "excessive_motion" gate
    static_motion_px: float = 0.5           # inlier counted as static below this displacement
    bottom_band_frac: float = 0.15          # broadcast-graphics band for diagnostics


def estimate_motion(img_a, img_b, mask_a=None, mask_b=None, diag_mask=None,
                    params: MotionParams = MotionParams(), orb=None, matcher=None):
    """Homography M mapping image A pixels to image B pixels.

    mask_a / mask_b restrict where features are detected (None = full frame).
    diag_mask (e.g. court mask of A) is only used to report `offcourt_inlier_share`.
    Returns (M or None, metrics dict with a `status` key). Gating matches cell 20.
    """
    orb = orb or cv2.ORB_create(nfeatures=params.n_features)
    matcher = matcher or cv2.BFMatcher(cv2.NORM_HAMMING)
    ga = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    kpa, da = orb.detectAndCompute(ga, mask_a)
    kpb, db = orb.detectAndCompute(gb, mask_b)
    if da is None or db is None:
        return None, {"status": "no_features"}

    good = [p[0] for p in matcher.knnMatch(da, db, k=2)
            if len(p) == 2 and p[0].distance < params.ratio_test * p[1].distance]
    if len(good) < params.min_matches:
        return None, {"status": "insufficient_matches", "matches": len(good)}

    pa = np.float32([kpa[m.queryIdx].pt for m in good])
    pb = np.float32([kpb[m.trainIdx].pt for m in good])
    M, inl = cv2.findHomography(pa, pb, cv2.RANSAC, params.ransac_thresh_px)
    if M is None or inl is None:
        return None, {"status": "ransac_failed", "matches": len(good)}

    inl = inl.ravel().astype(bool)
    h, w = img_a.shape[:2]
    disp = np.linalg.norm(pb[inl] - pa[inl], axis=1)
    met = {
        "matches": len(good),
        "inliers": int(inl.sum()),
        "inlier_ratio": round(float(inl.mean()), 4),
        "median_inlier_motion_px": round(float(np.median(disp)), 2) if len(disp) else float("nan"),
        "static_inlier_share": round(float((disp < params.static_motion_px).mean()), 3) if len(disp) else float("nan"),
        "bottom_band_inlier_share": round(float((pa[inl, 1] > (1 - params.bottom_band_frac) * h).mean()), 3),
    }
    if diag_mask is not None and inl.any():
        xi = pa[inl, 0].astype(int).clip(0, w - 1)
        yi = pa[inl, 1].astype(int).clip(0, h - 1)
        met["offcourt_inlier_share"] = round(float((diag_mask[yi, xi] == 0).mean()), 3)

    if met["inliers"] < params.min_inliers or met["inlier_ratio"] < params.min_inlier_ratio:
        return None, {**met, "status": "low_confidence"}

    test = np.float32([[w * .25, h * .25], [w * .75, h * .25], [w * .75, h * .75], [w * .25, h * .75]])
    moved = cv2.perspectiveTransform(test.reshape(-1, 1, 2), M).reshape(-1, 2)
    if not np.isfinite(moved).all():
        return None, {**met, "status": "invalid_geometry"}
    shift = float(np.linalg.norm(moved - test, axis=1).max())
    met["max_test_point_shift_px"] = round(shift, 2)
    if shift > params.max_test_point_shift_px:
        return None, {**met, "status": "excessive_motion"}
    return M, {**met, "status": "ok"}

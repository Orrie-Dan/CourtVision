"""Image <-> court homographies for the free-throw lane.

Court coordinate convention (lane-relative, metres) — as in the original notebook:
    origin at baseline-upper corner, +x along the lane away from the baseline,
    +y from the upper lane line toward the lower lane line.
This is NOT full-court coordinates; converting to a court-origin frame needs the
lane's offset on the baseline, which must be verified before use.
"""
import cv2
import numpy as np

from .config import LANE_LENGTH_M, LANE_WIDTH_M


def lane_court_points(length_m: float = LANE_LENGTH_M, width_m: float = LANE_WIDTH_M) -> np.ndarray:
    """Lane corners in court metres, in LANE_CORNER_ORDER."""
    return np.array([[0, width_m], [0, 0], [length_m, 0], [length_m, width_m]], dtype=np.float32)


def validate_correspondences(image_pts, court_pts, image_size=None, min_image_area=500.0, min_court_area=0.5):
    """Basic sanity checks from original cell 43. Raises AssertionError on failure."""
    src = np.asarray(image_pts, np.float32)
    dst = np.asarray(court_pts, np.float32)
    assert src.shape == dst.shape == (4, 2), "Exactly four (x, y) correspondences required"
    assert np.isfinite(src).all() and np.isfinite(dst).all(), "Non-finite correspondence"
    if image_size is not None:
        w, h = image_size
        assert ((src[:, 0] >= 0) & (src[:, 0] < w) & (src[:, 1] >= 0) & (src[:, 1] < h)).all(), \
            "Image point outside the frame"
    assert cv2.contourArea(cv2.convexHull(src)) > min_image_area, "Image landmarks too close / collinear"
    assert cv2.contourArea(cv2.convexHull(dst)) > min_court_area, "Court landmarks too close / collinear"


def image_to_court(image_corners, court_pts=None) -> np.ndarray:
    """3x3 homography mapping image pixels -> court metres from 4 lane corners."""
    court_pts = lane_court_points() if court_pts is None else court_pts
    H = cv2.getPerspectiveTransform(np.asarray(image_corners, np.float32), np.asarray(court_pts, np.float32))
    assert np.isfinite(H).all() and abs(np.linalg.det(H)) > 1e-10, "Degenerate homography"
    return H


def project_court_to_image(H_img_to_court, court_pts=None) -> np.ndarray:
    """Project court-metre points back into the image with inv(H)."""
    court_pts = lane_court_points() if court_pts is None else court_pts
    pts = np.asarray(court_pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.linalg.inv(H_img_to_court)).reshape(-1, 2)


def map_points(H, pts) -> np.ndarray:
    """Apply homography H to an (N, 2) array of points."""
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)

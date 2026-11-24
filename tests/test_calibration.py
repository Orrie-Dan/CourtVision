"""Unit tests for the Phase 3 calibration package.

Frame-dependent results (real Hough output on SportsMOT frames) cannot be tested
here — those are reproduced in the notebook's Stage 1 audit on Kaggle.
"""
import cv2
import numpy as np
import pytest

from courtvision import annotations as ann
from courtvision.court_mask import court_mask
from courtvision.geometry import corner_errors
from courtvision.homography import image_to_court, lane_court_points, map_points, project_court_to_image
from courtvision.lane import detect_segments, reconstruct_lane, refine_lane_from_prior, seed_lines_from_corners
from courtvision.motion import estimate_motion
from courtvision.propagation import chain_propagate
from courtvision.viz import SELECTOR_LABELS, write_selector_html

HANDOVER_AUTO_115 = np.array([[340.44, 540.66], [435.97, 437.65], [861.26, 482.80], [797.80, 592.70]])


# ---------------------------------------------------------------- annotations
def test_manual_lane_115_matches_handover_comparison():
    """The 4.10 / 6.86 px handover figures are relative to the cell-31 reconstructed lane."""
    manual = ann.lane_corners(ann.ANNOTATIONS[115])
    np.testing.assert_allclose(manual[2:], [[861, 486], [799, 592]])
    per, mean, mx = corner_errors(HANDOVER_AUTO_115, manual)
    np.testing.assert_allclose(per, [6.86, 4.95, 3.21, 1.39], atol=0.01)
    assert round(mean, 2) == 4.10 and round(mx, 2) == 6.86


@pytest.mark.parametrize("fid", [105, 115, 150, 100, 275, 299])
def test_ft_order_is_geometrically_consistent(fid):
    assert ann.check_ft_order(ann.ANNOTATIONS[fid])["ok"]


@pytest.mark.parametrize("fid, off_px", [(175, 9.1), (199, 33.9), (250, 59.0)])
def test_partial_references_flag_offscreen_ft_lower(fid, off_px):
    a = ann.ANNOTATIONS[fid]
    d = ann.check_ft_order(a)
    assert not d["ok"] and round(d["ft_lower_to_lower"], 1) == off_px and d["ft_upper_to_upper"] < 5
    np.testing.assert_array_equal(ann.corner_validity(a), [True, True, True, False])


def test_frame299_uses_ft_corners_only():
    """Baseline clicks 59 px apart at the frame edge: the extrapolated baseline corners lie outside the image."""
    a = ann.ANNOTATIONS[299]
    np.testing.assert_array_equal(ann.corner_validity(a), [False, False, True, True])
    assert (ann.lane_corners(a)[:2, 0] < 0).all()


def test_full_references_have_all_corners_valid():
    for fid, a in ann.ANNOTATIONS.items():
        if a.status == "reference":
            assert ann.corner_validity(a).all(), fid


def test_partial_reference_requires_unreliable_points():
    with pytest.raises(AssertionError):
        ann.LineAnnotation(1, ((0, 0),) * 8, "lower_upper", "partial_reference")


def test_rejected_frame200_fails_order_check():
    d = ann.check_ft_order(ann.ANNOTATIONS[200])
    assert not d["ok"] and d["ft_lower_to_lower"] > 30


def test_swapped_order_is_detected():
    a = ann.ANNOTATIONS[115]
    swapped = ann.LineAnnotation(a.frame, a.points, "upper_lower", a.status)
    assert not ann.check_ft_order(swapped)["ok"]


def test_annotation_coordinates_are_verbatim():
    assert ann.ANNOTATIONS[105].points[0] == (280, 471) and ann.ANNOTATIONS[105].points[7] == (793, 411)
    assert ann.ANNOTATIONS[115].points[0] == (365, 522) and ann.ANNOTATIONS[115].points[7] == (861, 486)


# ----------------------------------------------------------------- homography
def test_homography_roundtrip():
    corners = ann.lane_corners(ann.ANNOTATIONS[115])
    H = image_to_court(corners)
    np.testing.assert_allclose(map_points(H, corners), lane_court_points(), atol=1e-4)
    np.testing.assert_allclose(project_court_to_image(H), corners, atol=1e-3)


# ---------------------------------------------------------------- lane lines
def _synthetic_lane_image(corners, size=(720, 1280)):
    img = np.full((*size, 3), (60, 110, 170), np.uint8)  # wood-ish background (BGR)
    c = np.round(corners).astype(int)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        a, b = c[i], c[j]
        # extend lines beyond the corners, like real court markings
        d = (b - a) / np.linalg.norm(b - a)
        cv2.line(img, tuple((a - 60 * d).astype(int)), tuple((b + 60 * d).astype(int)), (245, 245, 245), 4)
    return img


def test_reconstruct_lane_recovers_corners_from_offset_seeds():
    truth = ann.lane_corners(ann.ANNOTATIONS[115])
    img = _synthetic_lane_image(truth)
    seeds = seed_lines_from_corners(truth + np.array([4.0, -5.0]))  # imperfect prior
    res = reconstruct_lane(detect_segments(img), seeds)
    assert res.status == "ok", res.status
    _, mean, mx = corner_errors(res.corners, truth)
    assert mx < 4.0, (mean, mx)


def test_reconstruct_lane_fails_when_seed_far_from_lines():
    truth = ann.lane_corners(ann.ANNOTATIONS[115])
    img = _synthetic_lane_image(truth)
    res = reconstruct_lane(detect_segments(img), seed_lines_from_corners(truth + 80.0))
    assert res.corners is None and res.status.endswith("_failed")


def test_refine_rejects_large_corrections():
    truth = ann.lane_corners(ann.ANNOTATIONS[115])
    img = _synthetic_lane_image(truth)
    res = refine_lane_from_prior(img, truth + np.array([0.0, 8.0]), max_correction_px=2.0)
    assert res.corners is None and res.status == "correction_too_large"


# -------------------------------------------------------------------- motion
def _textured(seed=0, size=(720, 1280)):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (size[0] // 8, size[1] // 8, 3), dtype=np.uint8)
    return cv2.GaussianBlur(cv2.resize(small, size[::-1], interpolation=cv2.INTER_NEAREST), (3, 3), 0)


def _shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, img.shape[1::-1], borderMode=cv2.BORDER_REFLECT)


def test_estimate_motion_recovers_translation():
    a = _textured()
    b = _shift(a, 6, -3)
    cv2.setRNGSeed(0)
    M, met = estimate_motion(a, b)
    assert met["status"] == "ok"
    p = cv2.perspectiveTransform(np.float32([[[640, 360]]]), M)[0, 0]
    np.testing.assert_allclose(p, [646, 357], atol=0.5)
    assert met["static_inlier_share"] < 0.05


def test_small_static_overlay_is_rejected_as_outliers():
    """A small fixed graphic is outvoted by the dominant (panning) motion: its points are
    RANSAC outliers, so bottom-band/static inlier shares stay ~0 and motion stays correct.
    (Whether the real scoreboard behaves like this is what the Stage 1 audit measures.)"""
    a = _textured()
    b = _shift(a, 6, 0)
    overlay = _textured(seed=1)[620:, :]
    a[620:], b[620:] = overlay, overlay
    cv2.setRNGSeed(0)
    M, met = estimate_motion(a, b)
    assert met["status"] == "ok"
    assert met["bottom_band_inlier_share"] < 0.02 and met["static_inlier_share"] < 0.02
    p = cv2.perspectiveTransform(np.float32([[[640, 300]]]), M)[0, 0]
    np.testing.assert_allclose(p, [646, 300], atol=0.5)


def test_static_scene_reports_static_inliers():
    a = _textured()
    cv2.setRNGSeed(0)
    _, met = estimate_motion(a, a.copy())
    assert met["static_inlier_share"] > 0.95


# --------------------------------------------------------------- propagation
def test_chain_propagation_tracks_known_pan():
    base = _textured(size=(720, 1400))
    frames = {i: np.ascontiguousarray(base[:, i * 5: i * 5 + 1280]) for i in range(6)}  # camera pans right 5 px/frame
    corners0 = np.array([[300, 500], [400, 400], [800, 450], [750, 560]], np.float64)
    homs, recs = chain_propagate(frames.__getitem__, image_to_court(corners0), 0, 6)
    assert len(homs) == 6 and all(r["status"] == "ok" for r in recs)
    np.testing.assert_allclose(project_court_to_image(homs[5]), corners0 - [25, 0], atol=1.5)


def test_chain_propagation_stops_on_failure():
    frames = {0: _textured(), 1: np.zeros((720, 1280, 3), np.uint8), 2: _textured()}
    homs, recs = chain_propagate(frames.__getitem__, np.eye(3), 0, 3)
    assert list(homs) == [0] and len(recs) == 1 and recs[0]["status"] == "no_features"


# --------------------------------------------------------------- mask / viz
def test_court_mask_excludes_bottom_band():
    img = np.full((720, 1280, 3), (60, 110, 170), np.uint8)
    m = court_mask(img)
    assert m[:600].mean() > 250 and m[int(720 * 0.85):].max() == 0


def test_selector_page(tmp_path):
    out = write_selector_html(np.zeros((72, 128, 3), np.uint8), 115, tmp_path / "s.html")
    html = out.read_text(encoding="utf-8")
    assert "LOWER free-throw corner" in html and "lower_upper" in html and len(SELECTOR_LABELS) == 8
    assert html.index("LOWER free-throw") < html.index("UPPER free-throw")

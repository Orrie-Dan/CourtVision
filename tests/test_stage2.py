"""Stage 2: court-plane feature masks, acceptance gates and gated propagation (synthetic data).

Real-frame behaviour (and threshold tuning) is evaluated on Kaggle in the notebook's Stage 2 section.
"""
import cv2
import numpy as np
import pandas as pd
import pytest

from courtvision.court_mask import exclude_boxes
from courtvision.gating import GateParams, is_convex_quad, lane_gate, motion_gate_failures
from courtvision.homography import image_to_court, project_court_to_image
from courtvision.lane import LaneResult
from courtvision.motion import estimate_motion
from courtvision.propagation import FRAME_STATUSES, chain_propagate, gated_propagate
from courtvision.tracking import boxes_for_frame

LANE0 = np.array([[300, 500], [400, 400], [800, 450], [750, 560]], np.float64)
SIZE = (1280, 720)


def _textured(seed=0, size=(720, 1280)):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (size[0] // 8, size[1] // 8, 3), dtype=np.uint8)
    return cv2.GaussianBlur(cv2.resize(small, size[::-1], interpolation=cv2.INTER_NEAREST), (3, 3), 0)


def _pan_frames(n, px_per_frame=5):
    base = _textured(size=(720, 1280 + n * px_per_frame))
    return {i: np.ascontiguousarray(base[:, i * px_per_frame: i * px_per_frame + 1280]) for i in range(n)}


def _no_refine(frame, prior, gate):
    return LaneResult(None, "disabled")


# ------------------------------------------------------------ feature masks
def test_exclude_boxes_zeroes_padded_boxes_and_clips():
    m = np.full((100, 200), 255, np.uint8)
    out = exclude_boxes(m, [[50, 20, 10, 30], [190, 90, 40, 40]], pad_px=2)
    assert out[18:52, 48:62].max() == 0                      # box + padding cleared
    assert out[17, 55] == 255 and out[30, 63] == 255         # just outside stays
    assert out[88:, 188:].max() == 0                         # clipped at image border
    assert m.min() == 255                                    # input not modified
    np.testing.assert_array_equal(exclude_boxes(m, np.empty((0, 4))), m)


def test_boxes_for_frame_uses_file_stem_ids():
    tracks = pd.DataFrame({"frame": [101, 101, 102], "track_id": [1, 2, 1],
                           "x": [1.0, 2, 3], "y": [4.0, 5, 6], "w": [7.0, 8, 9], "h": [10.0, 11, 12]})
    np.testing.assert_array_equal(boxes_for_frame(tracks, 101), [[1, 4, 7, 10], [2, 5, 8, 11]])
    assert boxes_for_frame(tracks, 999).shape == (0, 4)


# ------------------------------------------------------------- motion gates
def test_inlier_hull_frac_reports_spatial_spread():
    a = _textured()
    cv2.setRNGSeed(0)
    _, spread = estimate_motion(a, np.roll(a, 4, axis=1))
    assert spread["status"] == "ok" and spread["inlier_hull_frac"] > 0.5

    flat = np.full_like(a, 120)
    flat[300:420, 600:720] = a[300:420, 600:720]              # texture only in one small patch
    moved = np.roll(flat, 4, axis=1)
    cv2.setRNGSeed(0)
    _, clustered = estimate_motion(flat, moved)
    assert clustered["status"] == "ok", clustered
    assert clustered["inlier_hull_frac"] < GateParams().min_inlier_hull_frac
    assert motion_gate_failures(clustered) == ["clustered_inliers"]


# Real full-frame rows from results/phase3/stage1/stage1_propagation_diagnostics.csv.
@pytest.mark.parametrize("frame, static_share, shift, flagged", [
    (101, 0.473, 3.74, True),
    (105, 0.601, 12.06, True),
    (115, 0.674, 21.32, True),
    (101, 0.089, 5.96, False),   # court_only row: moving inliers, genuine pan
])
def test_motion_gate_flags_stage1_pinned_fits(frame, static_share, shift, flagged):
    met = {"status": "ok", "static_inlier_share": static_share, "max_test_point_shift_px": shift,
           "inlier_hull_frac": 0.5}
    assert ("static_inliers" in motion_gate_failures(met)) is flagged


def test_motion_gate_passes_stationary_camera():
    a = _textured()
    cv2.setRNGSeed(0)
    _, met = estimate_motion(a, a.copy())
    assert met["static_inlier_share"] > 0.95 and motion_gate_failures(met) == []


def test_motion_gate_passes_through_motion_failures():
    assert motion_gate_failures({"status": "low_confidence"}) == ["motion_low_confidence"]


# --------------------------------------------------------------- lane gates
def test_is_convex_quad():
    assert is_convex_quad(LANE0)
    assert is_convex_quad(LANE0[::-1])                        # orientation-independent
    assert not is_convex_quad(LANE0[[0, 2, 1, 3]])            # bow-tie
    assert not is_convex_quad([[0, 0], [1, 0], [2, 0], [3, 0]])  # degenerate
    assert not is_convex_quad(np.full((4, 2), np.nan))


@pytest.mark.parametrize("candidate, gap, expected", [
    (LANE0 + [10, 0], 1, []),
    (LANE0 + [60, 0], 1, ["lane_step_too_large"]),
    (LANE0 + [60, 0], 2, []),                                 # limit scales with the gap
    (LANE0[[0, 2, 1, 3]], 1, ["lane_not_convex"]),
    (LANE0 + [1200, 0], 100, ["lane_outside_frame"]),
])
def test_lane_gate(candidate, gap, expected):
    fails, _ = lane_gate(candidate, LANE0, SIZE, gap)
    assert fails == expected


def test_lane_gate_area_change():
    centre = LANE0.mean(axis=0)
    grown = centre + (LANE0 - centre) * 1.2                   # area x1.44; far corners move ~50 px,
    fails, met = lane_gate(grown, LANE0, SIZE, gap=2)         # so gap=2 keeps the step gate out of it
    assert fails == ["lane_area_change"] and met["lane_area_ratio"] == pytest.approx(1.44, abs=0.01)


# ------------------------------------------------------- gated propagation
def test_gated_matches_chain_on_clean_pan():
    frames = _pan_frames(6)
    H0 = image_to_court(LANE0)
    homs, recs = gated_propagate(frames.__getitem__, H0, 0, 6)
    chain_homs, _ = chain_propagate(frames.__getitem__, H0, 0, 6)
    assert [r["status"] for r in recs] == ["reference"] + ["propagated"] * 5
    assert {r["status"] for r in recs} <= set(FRAME_STATUSES)
    np.testing.assert_allclose(project_court_to_image(homs[5]), LANE0 - [25, 0], atol=1.5)
    np.testing.assert_allclose(project_court_to_image(homs[5]), project_court_to_image(chain_homs[5]), atol=1e-6)


def test_rejected_frame_does_not_break_chain():
    frames = _pan_frames(6)
    frames[3] = np.zeros_like(frames[3])                      # e.g. a cut / blur frame
    homs, recs = gated_propagate(frames.__getitem__, image_to_court(LANE0), 0, 6)
    by_frame = {r["frame"]: r for r in recs}
    assert by_frame[3]["status"] == "rejected" and 3 not in homs
    assert by_frame[4]["status"] == "propagated" and by_frame[4]["anchor"] == 2   # matched across the gap
    np.testing.assert_allclose(project_court_to_image(homs[5]), LANE0 - [25, 0], atol=1.5)


def test_lost_after_max_consecutive_rejections():
    frames = _pan_frames(1)
    blank = np.zeros_like(frames[0])
    load = lambda i: frames[0] if i == 0 else blank
    homs, recs = gated_propagate(load, image_to_court(LANE0), 0, 20,
                                 gate=GateParams(max_consecutive_rejections=3))
    assert [r["status"] for r in recs] == ["reference", "rejected", "rejected", "lost"]
    assert list(homs) == [0]


def test_stationary_camera_keeps_propagating():
    """Static inliers with ~0 fitted shift is a camera holding still, not a pinned fit."""
    still = _textured()
    homs, recs = gated_propagate(lambda i: still.copy(), image_to_court(LANE0), 0, 8)
    assert [r["status"] for r in recs] == ["reference"] + ["propagated"] * 7
    np.testing.assert_allclose(project_court_to_image(homs[7]), LANE0, atol=0.5)


def _failing_step_frames():
    """Frame 1 is unusable for motion (no features), forcing the fallback path."""
    frames = _pan_frames(2, px_per_frame=6)
    frames[1] = np.zeros_like(frames[1])
    return frames


def test_failed_step_is_rejected_without_refinement():
    homs, recs = gated_propagate(_failing_step_frames().__getitem__, image_to_court(LANE0), 0, 2,
                                 refine_fn=_no_refine)
    assert recs[1]["status"] == "rejected" and recs[1]["gate_failures"] == "motion_no_features"
    assert 1 not in homs


def test_failed_step_recovered_by_refinement():
    frames = _failing_step_frames()
    truth = LANE0 - [6, 0]
    seen = {}

    def refine(frame, prior, gate):
        seen["prior"] = prior
        return LaneResult(truth.copy(), "ok")

    homs, recs = gated_propagate(frames.__getitem__, image_to_court(LANE0), 0, 2, refine_fn=refine)
    assert recs[1]["status"] == "reconstructed"
    np.testing.assert_allclose(project_court_to_image(homs[1]), truth, atol=1e-3)
    np.testing.assert_allclose(seen["prior"], LANE0, atol=1e-3)   # no prediction -> last accepted lane


def test_refinement_result_must_pass_lane_gate():
    frames = _failing_step_frames()
    refine = lambda frame, prior, gate: LaneResult(LANE0 + [300, 0], "ok")   # implausible jump
    homs, recs = gated_propagate(frames.__getitem__, image_to_court(LANE0), 0, 2, refine_fn=refine)
    assert recs[1]["status"] == "rejected" and recs[1]["refine_status"].startswith("gate_failed")


def test_mask_fn_receives_frame_index():
    frames = _pan_frames(3)
    calls = []

    def mask_fn(idx, frame):
        calls.append(idx)
        return None

    gated_propagate(frames.__getitem__, image_to_court(LANE0), 0, 3, mask_fn=mask_fn)
    assert calls == [0, 1, 2]                                 # each frame's mask computed once


def test_default_refine_reconstructs_lane_on_line_image():
    """Every step must fall back to the real `default_refine` (seeded Hough line fit) and
    recover the drawn lane. Line-only images already fail motion (low_confidence: few ORB
    features); the impossible hull gate guarantees the fallback even if ORB did better."""
    def lane_image(corners):
        img = np.full((720, 1280, 3), (60, 110, 170), np.uint8)
        c = np.round(corners).astype(int)
        for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            d = (c[j] - c[i]) / np.linalg.norm(c[j] - c[i])
            cv2.line(img, tuple((c[i] - 60 * d).astype(int)), tuple((c[j] + 60 * d).astype(int)), (245, 245, 245), 4)
        return img

    lanes = {i: LANE0 + [3 * i, 0] for i in range(4)}
    frames = {i: lane_image(lanes[i]) for i in lanes}
    homs, recs = gated_propagate(frames.__getitem__, image_to_court(lanes[0]), 0, 4,
                                 gate=GateParams(min_inlier_hull_frac=2.0))
    assert [r["status"] for r in recs] == ["reference"] + ["reconstructed"] * 3, recs
    assert all(r["gate_failures"] and r["refine_status"] == "ok" for r in recs[1:])
    for i in (1, 2, 3):  # same < 4 px bound as test_reconstruct_lane_recovers_corners_from_offset_seeds
        err = np.linalg.norm(project_court_to_image(homs[i]) - lanes[i], axis=1)
        assert err.max() < 4.0, (i, err)

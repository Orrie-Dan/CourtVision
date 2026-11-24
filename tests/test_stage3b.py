"""Stage 3b: keyframe-anchored gated propagation (synthetic data).

Real-frame behaviour is evaluated on Kaggle (notebook section 2d) against the manual references.
"""
import cv2
import numpy as np

from courtvision.gating import KeyframeParams, ReanchorParams
from courtvision.homography import image_to_court, project_court_to_image
from courtvision.lane import LaneResult
from courtvision.propagation import gated_propagate, gated_propagate_order

LANE0 = np.array([[300, 500], [400, 400], [800, 450], [750, 560]], np.float64)
NEVER_REFRESH = dict(refresh_min_inliers=0, refresh_max_shift_px=1e9)


def _pan_frames(n, px):
    w = 1280 + n * px
    small = np.random.default_rng(0).integers(0, 255, (90, w // 8, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(cv2.resize(small, (w, 720), interpolation=cv2.INTER_NEAREST), (3, 3), 0)
    return {i: np.ascontiguousarray(base[:, i * px: i * px + 1280]) for i in range(n)}


def _no_refine(frame, prior, gate):
    return LaneResult(None, "disabled")


def _err(homs, i, truth):
    return np.linalg.norm(project_court_to_image(homs[i]) - truth, axis=1).max()


def test_needs_refresh():
    kf = KeyframeParams(max_gap=10, refresh_min_inliers=100, refresh_max_shift_px=80)
    ok = {"inliers": 500, "max_test_point_shift_px": 20}
    assert not kf.needs_refresh(9, ok)
    assert kf.needs_refresh(10, ok)
    assert kf.needs_refresh(1, {**ok, "inliers": 99})
    assert kf.needs_refresh(1, {**ok, "max_test_point_shift_px": 81})


def test_keyframe_held_until_max_gap():
    frames = _pan_frames(9, 5)
    homs, recs = gated_propagate(frames.__getitem__, image_to_court(LANE0), 0, 9, refine_fn=_no_refine,
                                 keyframe=KeyframeParams(max_gap=4, **NEVER_REFRESH))
    assert [r["status"] for r in recs] == ["reference"] + ["propagated"] * 8
    assert [r["anchor"] for r in recs[1:]] == [0, 0, 0, 0, 4, 4, 4, 4]
    assert [r["frame"] for r in recs if r.get("new_keyframe")] == [4, 8]
    assert all(r["anchor_mode"] == "keyframe" for r in recs[1:])
    for i in range(9):
        np.testing.assert_allclose(project_court_to_image(homs[i]), LANE0 - [5 * i, 0], atol=1.5)


def test_lane_gate_uses_last_accepted_lane_not_keyframe():
    """35 px/frame: 4 frames from the keyframe the lane has moved 140 px; the 40 px/frame step gate
    must still pass because it compares against the previous accepted frame (gap 1)."""
    frames = _pan_frames(5, 35)
    _, recs = gated_propagate(frames.__getitem__, image_to_court(LANE0), 0, 5, refine_fn=_no_refine,
                              keyframe=KeyframeParams(max_gap=100, **NEVER_REFRESH))
    assert [r["status"] for r in recs] == ["reference"] + ["propagated"] * 4
    assert [r["anchor"] for r in recs[1:]] == [0, 0, 0, 0]
    assert recs[4]["lane_step_px"] < 40 and recs[4]["max_test_point_shift_px"] > 130


def test_fallback_to_last_accepted_frame_becomes_keyframe():
    """At 35 px/frame the match to keyframe 0 exceeds the 150 px excessive-motion limit at gap 5
    (175 px); that frame falls back to frame 4 and becomes the next keyframe."""
    frames = _pan_frames(8, 35)
    homs, recs = gated_propagate(frames.__getitem__, image_to_court(LANE0), 0, 8, refine_fn=_no_refine,
                                 keyframe=KeyframeParams(max_gap=100, **NEVER_REFRESH))
    by = {r["frame"]: r for r in recs}
    assert [by[i]["anchor"] for i in range(1, 5)] == [0] * 4
    assert by[5]["status"] == "propagated" and by[5]["anchor_mode"] == "fallback" and by[5]["anchor"] == 4
    assert by[5]["keyframe_gate_failures"] == "motion_excessive_motion" and by[5]["new_keyframe"]
    assert by[6]["anchor"] == 5 and by[6]["anchor_mode"] == "keyframe"
    np.testing.assert_allclose(project_court_to_image(homs[7]), LANE0 - [245, 0], atol=2.0)


def test_weak_match_refreshes_every_frame_like_chain():
    frames = _pan_frames(6, 5)
    _, recs = gated_propagate(frames.__getitem__, image_to_court(LANE0), 0, 6, refine_fn=_no_refine,
                              keyframe=KeyframeParams(refresh_min_inliers=10 ** 6))
    assert [r["anchor"] for r in recs[1:]] == [0, 1, 2, 3, 4]


def test_reconstructed_frame_becomes_keyframe_after_both_matches_fail():
    frames = _pan_frames(6, 5)
    frames[3] = np.zeros_like(frames[3])
    truth3 = LANE0 - [15, 0]
    refine = lambda frame, prior, gate: (LaneResult(truth3.copy(), "ok") if not frame.any()
                                         else LaneResult(None, "disabled"))
    homs, recs = gated_propagate(frames.__getitem__, image_to_court(LANE0), 0, 6, refine_fn=refine,
                                 keyframe=KeyframeParams(max_gap=100, **NEVER_REFRESH))
    by = {r["frame"]: r for r in recs}
    assert by[3]["status"] == "reconstructed" and by[3]["anchor_mode"] == "fallback"
    assert by[3]["keyframe_gate_failures"] == "motion_no_features" and by[3]["new_keyframe"]
    assert by[4]["anchor"] == 3
    np.testing.assert_allclose(project_court_to_image(homs[3]), truth3, atol=1e-3)


def test_keyframe_rejection_does_not_move_keyframe():
    frames = _pan_frames(6, 5)
    frames[3] = np.zeros_like(frames[3])
    _, recs = gated_propagate(frames.__getitem__, image_to_court(LANE0), 0, 6, refine_fn=_no_refine,
                              keyframe=KeyframeParams(max_gap=100, **NEVER_REFRESH))
    by = {r["frame"]: r for r in recs}
    assert by[3]["status"] == "rejected"
    assert by[3]["anchor_mode"] == "fallback" and by[3]["anchor"] == 2   # key 0 failed, then last (2) tried
    assert by[3]["keyframe_gate_failures"] == "motion_no_features"
    assert by[4]["anchor"] == 0 and by[4]["anchor_mode"] == "keyframe"
    # `last` did not advance to the rejected frame: the lane step is measured over 2 frames (10 px)
    assert 8.5 < by[4]["lane_step_px"] < 11.5


def test_keyframe_mode_calls_mask_fn_once_per_frame():
    frames = _pan_frames(6, 5)
    frames[3] = np.zeros_like(frames[3])                      # forces a fallback attempt on frame 3
    calls = []

    def mask_fn(idx, frame):
        calls.append(idx)
        return None

    gated_propagate(frames.__getitem__, image_to_court(LANE0), 0, 6, mask_fn=mask_fn, refine_fn=_no_refine,
                    keyframe=KeyframeParams(max_gap=100, **NEVER_REFRESH))
    assert calls == [0, 1, 2, 3, 4, 5]


def test_independent_frame_becomes_keyframe_with_reanchor():
    frames = _pan_frames(7, 5)
    counts = {"baseline": 5, "upper_lane": 5, "lower_lane": 5, "free_throw": 5}
    refine = lambda frame, prior, gate: LaneResult(np.asarray(prior) + [1, 0], "ok", dict(counts))
    homs, recs = gated_propagate(frames.__getitem__, image_to_court(LANE0), 0, 7, refine_fn=refine,
                                 reanchor=ReanchorParams(every_n=3),
                                 keyframe=KeyframeParams(max_gap=100, **NEVER_REFRESH))
    by = {r["frame"]: r for r in recs}
    assert [by[i]["status"] for i in range(1, 7)] == ["propagated", "propagated", "independent",
                                                     "propagated", "propagated", "independent"]
    assert by[3]["new_keyframe"] and by[6]["new_keyframe"] and not by[4].get("new_keyframe")
    assert [by[i]["anchor"] for i in range(1, 7)] == [0, 0, 0, 3, 3, 3]
    np.testing.assert_allclose(project_court_to_image(homs[3]), LANE0 - [15, 0] + [1, 0], atol=1.5)


def _zoom_sequence(n, zoom=1.01):
    """Camera zooming 1 %/frame about the centre while panning 2 px/frame, with pixel noise.
    Returns frames and the true lane per frame."""
    small = np.random.default_rng(0).integers(0, 255, (180, 320, 3), dtype=np.uint8)
    world = cv2.GaussianBlur(cv2.resize(small, (2560, 1440), interpolation=cv2.INTER_LINEAR), (5, 5), 0)
    noise = np.random.default_rng(1)

    def T(i):
        s = zoom ** i
        return np.array([[s, 0, 640 - s * (1280 - 2 * i)], [0, s, 360 - s * 720], [0, 0, 1]])

    frames, truth = {}, {}
    W0 = np.linalg.inv(T(0))
    for i in range(n):
        f = cv2.warpPerspective(world, T(i), (1280, 720), flags=cv2.INTER_LINEAR)
        frames[i] = np.clip(f.astype(np.int16) + noise.integers(-6, 7, f.shape), 0, 255).astype(np.uint8)
        truth[i] = cv2.perspectiveTransform(LANE0.reshape(-1, 1, 2), T(i) @ W0).reshape(-1, 2)
    return frames, truth


def test_keyframes_reduce_drift_under_zoom():
    """The Stage 2 failure mode: per-step errors compounding while the camera zooms. Holding the
    anchor fixed should leave clearly less error at the end of the sequence than chaining."""
    n = 40
    frames, truth = _zoom_sequence(n)
    H0 = image_to_court(LANE0)
    chain, chain_recs = gated_propagate(frames.__getitem__, H0, 0, n, refine_fn=_no_refine)
    kf, kf_recs = gated_propagate(frames.__getitem__, H0, 0, n, refine_fn=_no_refine,
                                  keyframe=KeyframeParams(max_gap=10))
    assert all(r["status"] in ("reference", "propagated") for r in chain_recs + kf_recs)
    assert sum(r["anchor"] < r["frame"] - 1 for r in kf_recs[1:]) > n // 2   # keyframes really were held
    chain_err = [_err(chain, i, truth[i]) for i in range(n)]
    kf_err = [_err(kf, i, truth[i]) for i in range(n)]
    assert chain_err[-1] > 1.5                                 # the chain does drift here
    assert kf_err[-1] < 0.75 * chain_err[-1], (kf_err[-1], chain_err[-1])
    assert np.mean(kf_err) < 0.75 * np.mean(chain_err)


def test_propagate_order_backward_from_anchor():
    """Anchor at frame 6, propagate backward to 0: real frame ids in records and homographies."""
    frames = _pan_frames(7, 5)
    anchor_lane = LANE0 - [30, 0]                              # true lane in frame 6
    seen = []

    def mask_fn(idx, img):
        seen.append(idx)
        return None

    homs, recs = gated_propagate_order(frames.__getitem__, image_to_court(anchor_lane), range(6, -1, -1),
                                       mask_fn=mask_fn, refine_fn=_no_refine, keyframe=KeyframeParams())
    assert [r["frame"] for r in recs] == [6, 5, 4, 3, 2, 1, 0] and seen == [6, 5, 4, 3, 2, 1, 0]
    assert recs[0]["status"] == "reference" and all(r["status"] == "propagated" for r in recs[1:])
    assert recs[1]["anchor"] == 6
    for i in range(7):
        np.testing.assert_allclose(project_court_to_image(homs[i]), LANE0 - [5 * i, 0], atol=1.5)

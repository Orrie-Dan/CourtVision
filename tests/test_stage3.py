"""Stage 3: re-anchoring of gated propagation by periodic bounded refinement (synthetic data).

Real-frame behaviour (and every_n / guard tuning) is evaluated on Kaggle in the notebook's Stage 3 section
against the manual references at frames 150 / 175 / 199.
"""
import cv2
import numpy as np
import pytest

from courtvision.gating import GateParams, ReanchorParams, reanchor_failures
from courtvision.homography import image_to_court, project_court_to_image
from courtvision.lane import LaneResult
from courtvision.propagation import FRAME_STATUSES, gated_propagate

LANE0 = np.array([[300, 500], [400, 400], [800, 450], [750, 560]], np.float64)
GOOD_COUNTS = {"baseline": 5, "upper_lane": 5, "lower_lane": 5, "free_throw": 5}
PX = 4  # pan per frame


def _lane_pan_frames(n, amp=0.3):
    """Pan over a low-contrast textured floor (ORB features) with the lane painted on it (Hough lines).
    True lane in frame i is LANE0 - [PX * i, 0]."""
    w = 1280 + n * PX
    small = np.random.default_rng(0).integers(0, 255, (90, w // 8, 3), dtype=np.uint8)
    tex = cv2.GaussianBlur(cv2.resize(small, (w, 720), interpolation=cv2.INTER_NEAREST), (3, 3), 0)
    base = (110 + (tex.astype(np.float32) - 128) * amp).clip(0, 255).astype(np.uint8)
    c = np.round(LANE0).astype(int)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        cv2.line(base, tuple(c[i]), tuple(c[j]), (245, 245, 245), 4)
    return {i: np.ascontiguousarray(base[:, i * PX: i * PX + 1280]) for i in range(n)}


def _truth(i):
    return LANE0 - [PX * i, 0]


def _drifted_reference():
    lane = LANE0.copy()
    lane[2:] += [-7, 0]  # free-throw end 7 px short: the Stage 2 failure mode (scale under-estimate)
    return lane


def _stub_refine(frames, results):
    """refine_fn returning results[frame index] (a LaneResult, or a callable of the prior)."""
    index_of = {id(f): i for i, f in frames.items()}
    calls = []

    def refine(frame, prior, gate):
        i = index_of[id(frame)]
        calls.append((i, np.asarray(prior).copy(), gate))
        r = results.get(i, LaneResult(None, "lower_lane_failed"))
        return r(prior) if callable(r) else r
    return refine, calls


def _err(homs, i):
    return np.linalg.norm(project_court_to_image(homs[i]) - _truth(i), axis=1).max()


# ------------------------------------------------------------ guards (pure)
@pytest.mark.parametrize("shift, counts, scale, expected", [
    ([3, 0], GOOD_COUNTS, 1.0, []),
    ([15, 0], GOOD_COUNTS, 1.0, ["reanchor_correction_too_large"]),
    ([0, 0], {**GOOD_COUNTS, "free_throw": 2}, 1.0, ["reanchor_weak_support"]),
    ([0, 0], {"baseline": 5}, 1.0, ["reanchor_weak_support"]),          # missing lines count as 0
    ([0, 0], GOOD_COUNTS, 1.012, ["reanchor_scale_disagrees"]),         # scale guard on its own
])
def test_reanchor_failures(shift, counts, scale, expected):
    centre = LANE0.mean(axis=0)
    refined = centre + (LANE0 - centre) * scale + shift
    params = ReanchorParams(min_area_ratio=0.99, max_area_ratio=1.01) if scale != 1.0 else ReanchorParams()
    fails, met = reanchor_failures(refined, LANE0, counts, params)
    assert fails == expected
    assert set(met) == {"reanchor_max_correction_px", "reanchor_area_ratio", "reanchor_min_segments"}


def test_reanchor_failures_rejects_bow_tie():
    assert reanchor_failures(LANE0[[0, 2, 1, 3]], LANE0, GOOD_COUNTS) == (["reanchor_not_convex"], {})


def test_default_scale_band_rejects_ten_percent_shrink():
    centre = LANE0.mean(axis=0)
    shrunk = centre + (LANE0 - centre) * 0.95                 # area x0.90 (corners move ~13 px)
    fails, met = reanchor_failures(shrunk, LANE0, GOOD_COUNTS, ReanchorParams(max_correction_px=50))
    assert met["reanchor_area_ratio"] == pytest.approx(0.9025, abs=0.002)
    assert fails == ["reanchor_scale_disagrees"]               # default 0.93-1.07 band on its own


# ------------------------------------------------- gated propagation + re-anchor
def test_reanchor_disabled_by_default():
    frames = _lane_pan_frames(6)
    refine, calls = _stub_refine(frames, {})
    _, recs = gated_propagate(frames.__getitem__, image_to_court(_drifted_reference()), 0, 6, refine_fn=refine)
    assert [r["status"] for r in recs] == ["reference"] + ["propagated"] * 5
    assert calls == [] and not any("reanchor_status" in r for r in recs)


def test_reanchor_schedule_and_tight_correction_limit():
    frames = _lane_pan_frames(9)
    refine, calls = _stub_refine(frames, {i: LaneResult(_truth(i), "ok", dict(GOOD_COUNTS)) for i in frames})
    homs, recs = gated_propagate(frames.__getitem__, image_to_court(_drifted_reference()), 0, 9,
                                 refine_fn=refine, reanchor=ReanchorParams(every_n=3))
    assert [r["status"] for r in recs] == ["reference", "propagated", "propagated", "independent",
                                           "propagated", "propagated", "independent", "propagated", "propagated"]
    assert {r["status"] for r in recs} <= set(FRAME_STATUSES)
    assert [c[0] for c in calls] == [3, 6]                    # only attempted when due
    assert all(c[2].max_refine_correction_px == 10.0 for c in calls)
    assert calls[0][2].max_static_inlier_share == GateParams().max_static_inlier_share  # other gates untouched
    np.testing.assert_allclose(calls[0][1], _drifted_reference() - [3 * PX, 0], atol=1.5)  # seeded by prediction
    assert _err(homs, 2) > 6.0                                # drift carried until the first fix...
    assert _err(homs, 3) < 1e-3 and _err(homs, 8) < 1.5       # ...then the chain continues from the refined lane
    assert recs[4]["anchor"] == 3


def test_failed_reanchor_keeps_propagated_and_retries_next_frame():
    frames = _lane_pan_frames(9)
    results = {i: LaneResult(_truth(i), "ok", dict(GOOD_COUNTS)) for i in (4, 5, 6, 7, 8)}
    refine, calls = _stub_refine(frames, results)            # frame 3: refinement fails (e.g. line occluded)
    homs, recs = gated_propagate(frames.__getitem__, image_to_court(_drifted_reference()), 0, 9,
                                 refine_fn=refine, reanchor=ReanchorParams(every_n=3))
    by = {r["frame"]: r for r in recs}
    assert by[3]["status"] == "propagated" and by[3]["reanchor_status"] == "lower_lane_failed" and 3 in homs
    assert by[4]["status"] == "independent"
    assert by[7]["status"] == "independent"                   # schedule restarts from the last fix (4)
    assert [c[0] for c in calls] == [3, 4, 7]


def test_guard_failure_keeps_prediction():
    frames = _lane_pan_frames(5)
    refine, _ = _stub_refine(frames, {3: lambda prior: LaneResult(np.asarray(prior) + [30, 0], "ok", dict(GOOD_COUNTS)),
                                      4: LaneResult(_truth(4), "ok", {**GOOD_COUNTS, "free_throw": 1})})
    homs, recs = gated_propagate(frames.__getitem__, image_to_court(_drifted_reference()), 0, 5,
                                 refine_fn=refine, reanchor=ReanchorParams(every_n=3))
    assert recs[3]["status"] == "propagated" and recs[3]["reanchor_status"] == "reanchor_correction_too_large"
    assert recs[4]["status"] == "propagated" and recs[4]["reanchor_status"] == "reanchor_weak_support"
    assert _err(homs, 4) > 6.0                                # the rejected refinement was not applied


def test_reanchor_only_on_propagated_frames():
    frames = _lane_pan_frames(6)
    frames[3] = np.zeros_like(frames[3])                      # motion fails; Stage 2 fallback refines instead
    refine, calls = _stub_refine(frames, {i: LaneResult(_truth(i), "ok", dict(GOOD_COUNTS)) for i in (4, 5)})
    _, recs = gated_propagate(frames.__getitem__, image_to_court(_drifted_reference()), 0, 6,
                              refine_fn=refine, reanchor=ReanchorParams(every_n=3))
    by = {r["frame"]: r for r in recs}
    assert by[3]["status"] == "rejected" and "reanchor_status" not in by[3]
    assert by[4]["status"] == "independent" and by[4]["anchor"] == 2


def test_real_refinement_corrects_drift():
    """End to end with real ORB motion and the real `default_refine` (seeded Hough): a lane whose
    free-throw end starts 7 px short is pulled back to the painted lines at the first fix."""
    frames = _lane_pan_frames(12)
    homs, recs = gated_propagate(frames.__getitem__, image_to_court(_drifted_reference()), 0, 12,
                                 reanchor=ReanchorParams(every_n=4))
    statuses = [r["status"] for r in recs]
    assert "rejected" not in statuses and "lost" not in statuses, recs
    first_fix = statuses.index("independent")
    assert 4 <= first_fix <= 6, statuses                      # due at 4; may retry on weak support
    assert all(_err(homs, i) > 6.0 for i in range(first_fix))
    assert all(_err(homs, i) < 4.0 for i in range(first_fix, 12)), [round(_err(homs, i), 2) for i in range(12)]


def test_degenerate_refined_homography_keeps_prediction_and_retries(monkeypatch):
    import courtvision.propagation as prop

    def degenerate(_corners, court_pts=None):
        raise AssertionError("Degenerate homography")

    monkeypatch.setattr(prop, "image_to_court", degenerate)   # only the re-anchor path calls it here
    frames = _lane_pan_frames(6)
    refine, calls = _stub_refine(frames, {i: LaneResult(_truth(i), "ok", dict(GOOD_COUNTS)) for i in frames})
    homs, recs = gated_propagate(frames.__getitem__, image_to_court(_drifted_reference()), 0, 6,
                                 refine_fn=refine, reanchor=ReanchorParams(every_n=3))
    assert [r["status"] for r in recs] == ["reference"] + ["propagated"] * 5
    assert all(recs[i]["reanchor_status"] == "degenerate_homography" for i in (3, 4, 5))
    assert [c[0] for c in calls] == [3, 4, 5]                 # not counted as a fix: retried every frame
    assert _err(homs, 5) > 6.0 and recs[5]["anchor"] == 4     # prediction kept as the anchor


def test_schedule_starts_at_reference_frame():
    frames = _lane_pan_frames(7)
    offset = 5                                                # reference at index 5, not 0
    refine, calls = _stub_refine(frames, {i: LaneResult(_truth(i), "ok", dict(GOOD_COUNTS)) for i in frames})
    _, recs = gated_propagate(lambda i: frames[i - offset], image_to_court(_drifted_reference()), offset,
                              offset + 7, refine_fn=refine, reanchor=ReanchorParams(every_n=3))
    assert [c[0] + offset for c in calls] == [8, 11]          # first attempt at ref_idx + every_n
    assert [r["frame"] for r in recs if r["status"] == "independent"] == [8, 11]

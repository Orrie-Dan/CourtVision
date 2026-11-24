"""Acceptance rule for frames used in court mapping (`gating.accept_segment`)."""
import numpy as np

from courtvision.gating import accept_segment

HOMS = {i: np.eye(3) * (i + 1) for i in range(100, 200)}


def test_accepts_whole_segment_when_all_refs_within_threshold():
    acc, why = accept_segment(HOMS, {105: 6.0, 150: 7.9, 199: 9.6}, 100, 199)
    assert sorted(acc) == list(range(100, 200)) and why.startswith("accepted 100 frames in [100, 199]")
    assert acc[150] is HOMS[150]


def test_one_reference_over_threshold_rejects_segment():
    acc, why = accept_segment(HOMS, {105: 6.0, 150: 7.9, 199: 10.01}, 100, 199)
    assert acc == {} and "199: 10.01 px > 10.0" in why


def test_unmeasured_or_uncalibrated_reference_rejects_segment():
    assert accept_segment(HOMS, {150: float("nan")}, 100, 199)[0] == {}
    homs = {k: v for k, v in HOMS.items() if k != 150}
    acc, why = accept_segment(homs, {150: 3.0}, 100, 199)
    assert acc == {} and "150: not calibrated" in why


def test_needs_a_reference_inside_segment_other_than_start():
    acc, why = accept_segment(HOMS, {100: 0.0, 250: 1.0}, 100, 199)
    assert acc == {} and why.startswith("no reference frame")


def test_anchor_at_segment_end_is_not_a_check():
    """Backward run anchored on a manual reference at 199: 199 itself must not validate the segment."""
    assert accept_segment(HOMS, {199: 0.0}, 100, 199, anchor=199)[0] == {}
    acc, why = accept_segment(HOMS, {150: 4.0, 199: 0.0}, 100, 199, anchor=199)
    assert sorted(acc) == list(range(150, 200)) and "refs [150]" in why   # bracketed by 150 and the anchor
    assert accept_segment(HOMS, {100: 4.0, 199: 0.0}, 100, 199, anchor=199)[0]   # start counts if not the anchor


def test_frames_without_homography_and_outside_segment_excluded():
    homs = {k: v for k, v in HOMS.items() if k != 120}
    homs[250] = np.eye(3)
    acc, _ = accept_segment(homs, {150: 5.0, 199: 5.0}, 100, 199)
    assert 120 not in acc and 250 not in acc and len(acc) == 99


def test_only_frames_bracketed_by_known_lanes():
    """Anchor 275, refs 250 and 299: backward frames before 250 are extrapolation (run C: 242-249)."""
    homs = {i: np.eye(3) for i in range(242, 300)}
    acc, why = accept_segment(homs, {250: 9.56, 275: 0.0, 299: 6.45}, 242, 299, anchor=275)
    assert sorted(acc) == list(range(250, 300)) and "[250, 299]" in why


def test_unchecked_refined_frames_dropped_but_measured_reference_kept():
    homs = {i: np.eye(3) for i in range(242, 300)}
    statuses = {i: "propagated" for i in homs}
    statuses.update({245: "reconstructed", 250: "reconstructed", 253: "reconstructed", 254: "independent"})
    acc, why = accept_segment(homs, {250: 9.56, 299: 6.45}, 242, 299, anchor=275, statuses=statuses)
    assert 250 in acc                                          # measured reference: its own check stands
    assert 253 not in acc and 254 not in acc and 245 not in acc
    assert len(acc) == 48 and "dropped unchecked refined frames [253, 254]" in why


def test_refined_reference_over_threshold_still_rejects():
    homs = {i: np.eye(3) for i in range(242, 300)}
    acc, _ = accept_segment(homs, {250: 10.2, 299: 6.45}, 242, 299, anchor=275, statuses={250: "reconstructed"})
    assert acc == {}

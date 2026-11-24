"""Acceptance rule for frames used in court mapping (`gating.accept_segment`)."""
import numpy as np

from courtvision.gating import accept_segment

HOMS = {i: np.eye(3) * (i + 1) for i in range(100, 200)}


def test_accepts_whole_segment_when_all_refs_within_threshold():
    acc, why = accept_segment(HOMS, {105: 6.0, 150: 7.9, 199: 9.6}, 100, 199)
    assert sorted(acc) == list(range(100, 200)) and why.startswith("accepted 100 frames")
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
    assert len(acc) == 100 and "refs [150]" in why
    assert accept_segment(HOMS, {100: 4.0, 199: 0.0}, 100, 199, anchor=199)[0]   # start counts if not the anchor


def test_frames_without_homography_and_outside_segment_excluded():
    homs = {k: v for k, v in HOMS.items() if k != 120}
    homs[250] = np.eye(3)
    acc, _ = accept_segment(homs, {150: 5.0}, 100, 199)
    assert 120 not in acc and 250 not in acc and len(acc) == 99

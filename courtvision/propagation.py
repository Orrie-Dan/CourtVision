"""Calibration propagation.

`chain_propagate` (original cell 20) is kept as the BASELINE. Known weakness: errors
accumulate along the chain and nothing checks the propagated court geometry. Frame 115
drifted 62.92 px from its independent annotation under this scheme (handover, reproduced
in Stage 1).

`gated_propagate` (Stage 2) accepts a frame only when the motion estimate and the lane it
predicts pass `gating` checks; otherwise it tries lane refinement, and if that fails it
rejects the frame and matches the next one against the last accepted frame.
"""
from typing import Callable

import cv2
import numpy as np

from .gating import GateParams, is_convex_quad, lane_gate, motion_gate_failures
from .homography import image_to_court, project_court_to_image
from .motion import MotionParams, estimate_motion

# Per-frame status values produced by gated_propagate.
FRAME_STATUSES = ("reference", "propagated", "reconstructed", "rejected", "lost")


def chain_propagate(load_frame: Callable, H_ref: np.ndarray, ref_idx: int, stop: int, step: int = 1,
                    mask_fn: Callable | None = None, diag_mask_fn: Callable | None = None,
                    params: MotionParams = MotionParams(), seed: int | None = 0):
    """Propagate an image->court homography from `ref_idx` toward `stop` (exclusive).

    load_frame(idx) -> BGR image. mask_fn(frame) -> feature mask or None (None = full frame,
    as in the original). Stops at the first failed step, since every later frame depends on it.
    Returns (homographies {idx: H}, records [dict per attempted frame]).
    """
    if seed is not None:
        cv2.setRNGSeed(seed)
    orb = cv2.ORB_create(nfeatures=params.n_features)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    homs = {ref_idx: np.asarray(H_ref, np.float64).copy()}
    records = []
    prev_img = load_frame(ref_idx)
    for idx in range(ref_idx + step, stop, step):
        img = load_frame(idx)
        ma = mask_fn(prev_img) if mask_fn else None
        mb = mask_fn(img) if mask_fn else None
        dm = diag_mask_fn(prev_img) if diag_mask_fn else None
        M, met = estimate_motion(prev_img, img, ma, mb, dm, params, orb, matcher)
        rec = {"frame": idx, **met}
        records.append(rec)
        if M is None:
            break
        try:
            H = homs[idx - step] @ np.linalg.inv(M)  # current image -> previous image -> court
        except np.linalg.LinAlgError:
            rec["status"] = "singular_motion"
            break
        if not np.isfinite(H).all() or abs(H[2, 2]) < 1e-10:
            rec["status"] = "invalid_homography"
            break
        homs[idx] = H / H[2, 2]
        prev_img = img
    return homs, records


def default_refine(frame, prior_corners, gate: GateParams):
    """Lane refinement around a prior (original cell 39) with the Stage 2 correction limit."""
    from .lane import refine_lane_from_prior
    return refine_lane_from_prior(frame, prior_corners, max_correction_px=gate.max_refine_correction_px)


def gated_propagate(load_frame: Callable, H_ref: np.ndarray, ref_idx: int, stop: int,
                    mask_fn: Callable | None = None, diag_mask_fn: Callable | None = None,
                    motion_params: MotionParams = MotionParams(), gate: GateParams = GateParams(),
                    refine_fn: Callable | None = default_refine, seed: int | None = 0):
    """Confidence-gated forward propagation from `ref_idx` toward `stop` (exclusive).

    load_frame(idx) -> BGR image.
    mask_fn(idx, frame) -> feature mask or None. Takes the index so per-frame player boxes
        can be excluded (see `court_mask.exclude_boxes`).
    diag_mask_fn(frame) -> mask used only for the off-court inlier diagnostic.
    refine_fn(frame, prior_corners, gate) -> lane.LaneResult; None disables reconstruction.

    Each frame is matched against the last ACCEPTED frame (the anchor), so a rejected frame
    does not break the chain. Statuses: propagated (motion + lane gates pass), reconstructed
    (gates failed, lane refinement around the prediction succeeded and passed the lane gate),
    rejected (neither), lost (`gate.max_consecutive_rejections` reached; propagation stops).

    Returns (homographies {idx: image->court H} for accepted frames, records [dict per frame]).
    """
    if seed is not None:
        cv2.setRNGSeed(seed)
    orb = cv2.ORB_create(nfeatures=motion_params.n_features)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    H_ref = np.asarray(H_ref, np.float64)
    homs = {ref_idx: H_ref / H_ref[2, 2]}
    records = [{"frame": ref_idx, "status": "reference"}]
    anchor_idx, anchor_img = ref_idx, load_frame(ref_idx)
    anchor_lane = project_court_to_image(homs[ref_idx])
    anchor_mask = mask_fn(ref_idx, anchor_img) if mask_fn else None
    anchor_diag = diag_mask_fn(anchor_img) if diag_mask_fn else None
    image_size = anchor_img.shape[1::-1]
    rejections = 0

    for idx in range(ref_idx + 1, stop):
        img = load_frame(idx)
        gap = idx - anchor_idx
        mask = mask_fn(idx, img) if mask_fn else None
        M, met = estimate_motion(anchor_img, img, anchor_mask, mask, anchor_diag, motion_params, orb, matcher)
        rec = {"frame": idx, "anchor": anchor_idx, **met, "motion_status": met["status"]}
        fails = motion_gate_failures(met, gate)

        H = pred = None
        if M is not None:
            try:
                H = homs[anchor_idx] @ np.linalg.inv(M)  # current image -> anchor image -> court
                H = H / H[2, 2] if np.isfinite(H).all() and abs(H[2, 2]) > 1e-10 else None
            except np.linalg.LinAlgError:
                H = None
            if H is None:
                fails.append("invalid_homography")
            else:
                pred = project_court_to_image(H)
                lane_fails, lane_met = lane_gate(pred, anchor_lane, image_size, gap, gate)
                rec.update(lane_met)
                fails += lane_fails
        rec["gate_failures"] = ";".join(fails)

        status, lane = None, None
        if not fails:
            status, lane = "propagated", pred
        elif refine_fn is not None:
            prior = pred if pred is not None and is_convex_quad(pred) else anchor_lane
            res = refine_fn(img, prior, gate)
            rec["refine_status"] = res.status
            if res.corners is not None:
                refine_fails, refine_met = lane_gate(res.corners, anchor_lane, image_size, gap, gate)
                rec.update({f"refined_{k}": v for k, v in refine_met.items()})
                if refine_fails:
                    rec["refine_status"] = "gate_failed:" + ";".join(refine_fails)
                else:
                    try:
                        status, lane, H = "reconstructed", res.corners, image_to_court(res.corners)
                    except AssertionError:
                        status, rec["refine_status"] = None, "degenerate_homography"

        if status is None:
            rejections += 1
            rec["status"] = "lost" if rejections >= gate.max_consecutive_rejections else "rejected"
            records.append(rec)
            if rec["status"] == "lost":
                break
            continue

        rec["status"] = status
        records.append(rec)
        homs[idx] = H / H[2, 2]
        anchor_idx, anchor_img, anchor_lane, anchor_mask = idx, img, np.asarray(lane, np.float64), mask
        anchor_diag = diag_mask_fn(img) if diag_mask_fn else None
        rejections = 0
    return homs, records

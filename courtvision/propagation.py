"""Calibration propagation.

`chain_propagate` (original cell 20) is kept as the BASELINE. Known weakness: errors
accumulate along the chain and nothing checks the propagated court geometry. Frame 115
drifted 62.92 px from its independent annotation under this scheme (handover, reproduced
in Stage 1).

`gated_propagate` (Stage 2) accepts a frame only when the motion estimate and the lane it
predicts pass `gating` checks; otherwise it tries lane refinement, and if that fails it
rejects the frame and matches the next one against the last accepted frame.

Stage 3 (optional `reanchor`): every `reanchor.every_n` frames a propagated lane is refined
against the painted lines with a tight correction limit; if the guards pass the frame becomes
`independent` and the chain continues from the refined lane, so drift cannot compound.
"""
from typing import Callable

import cv2
import numpy as np

from .gating import GateParams, KeyframeParams, ReanchorParams, is_convex_quad, lane_gate, motion_gate_failures, reanchor_failures
from .homography import image_to_court, project_court_to_image
from .motion import MotionParams, estimate_motion

# Per-frame status values produced by gated_propagate.
FRAME_STATUSES = ("reference", "independent", "propagated", "reconstructed", "rejected", "lost")


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


_UNSET = object()


def _anchor(idx, img, mask_fn, diag_mask_fn, mask=_UNSET):
    """Frame state used as a motion reference: index, image, feature mask, diagnostic mask."""
    if mask is _UNSET:
        mask = mask_fn(idx, img) if mask_fn else None
    return {"idx": idx, "img": img, "mask": mask, "diag": diag_mask_fn(img) if diag_mask_fn else None}


def gated_propagate(load_frame: Callable, H_ref: np.ndarray, ref_idx: int, stop: int,
                    mask_fn: Callable | None = None, diag_mask_fn: Callable | None = None,
                    motion_params: MotionParams = MotionParams(), gate: GateParams = GateParams(),
                    refine_fn: Callable | None = default_refine, seed: int | None = 0,
                    reanchor: ReanchorParams | None = None, keyframe: KeyframeParams | None = None):
    """Confidence-gated forward propagation from `ref_idx` toward `stop` (exclusive).

    load_frame(idx) -> BGR image.
    mask_fn(idx, frame) -> feature mask or None. Takes the index so per-frame player boxes
        can be excluded (see `court_mask.exclude_boxes`).
    diag_mask_fn(frame) -> mask used only for the off-court inlier diagnostic.
    refine_fn(frame, prior_corners, gate) -> lane.LaneResult; None disables reconstruction.

    Each frame is matched against an anchor frame; by default (keyframe=None) that is the last
    ACCEPTED frame, so a rejected frame does not break the chain. The lane gate always compares
    against the last accepted lane. Statuses: propagated (motion + lane gates pass), reconstructed
    (gates failed, lane refinement around the prediction succeeded and passed the lane gate),
    rejected (neither), lost (`gate.max_consecutive_rejections` reached; propagation stops).

    reanchor: None keeps Stage 2 behaviour. Otherwise, once `reanchor.every_n` frames have passed
    since the reference / last independent frame, a `propagated` frame is refined around its
    prediction (refine_fn with `reanchor.max_correction_px`). If `gating.reanchor_failures` passes,
    the frame is `independent` and its refined lane becomes the anchor; if not, it stays
    `propagated` and the next frame tries again. Requires refine_fn.

    keyframe (Stage 3b): None keeps Stage 2 behaviour. Otherwise the anchor is a KEYFRAME that
    stays fixed across accepted frames, so errors compound once per keyframe instead of once per
    frame. If matching to the keyframe fails the gates, the frame is matched to the last accepted
    frame instead (`anchor_mode` = "fallback"). An accepted frame becomes the new keyframe when it
    used the fallback, was reconstructed / independent, or `keyframe.needs_refresh` (gap reached
    `max_gap`, or the match to the keyframe is getting weak).

    Returns (homographies {idx: image->court H} for accepted frames, records [dict per frame]).
    """
    if seed is not None:
        cv2.setRNGSeed(seed)
    orb = cv2.ORB_create(nfeatures=motion_params.n_features)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    H_ref = np.asarray(H_ref, np.float64)
    homs = {ref_idx: H_ref / H_ref[2, 2]}
    records = [{"frame": ref_idx, "status": "reference"}]
    key = last = _anchor(ref_idx, load_frame(ref_idx), mask_fn, diag_mask_fn)
    last_lane = project_court_to_image(homs[ref_idx])
    image_size = key["img"].shape[1::-1]
    rejections = 0
    last_fix_idx = ref_idx  # reference or last independent frame

    def attempt(anchor, idx, img, mask):
        """Match `img` to `anchor`, predict the lane, run the motion and lane gates."""
        M, met = estimate_motion(anchor["img"], img, anchor["mask"], mask, anchor["diag"],
                                 motion_params, orb, matcher)
        rec = {"frame": idx, "anchor": anchor["idx"], **met, "motion_status": met["status"]}
        fails = motion_gate_failures(met, gate)
        H = pred = None
        if M is not None:
            try:
                H = homs[anchor["idx"]] @ np.linalg.inv(M)  # current image -> anchor image -> court
                H = H / H[2, 2] if np.isfinite(H).all() and abs(H[2, 2]) > 1e-10 else None
            except np.linalg.LinAlgError:
                H = None
            if H is None:
                fails.append("invalid_homography")
            else:
                pred = project_court_to_image(H)
                lane_fails, lane_met = lane_gate(pred, last_lane, image_size, idx - last["idx"], gate)
                rec.update(lane_met)
                fails += lane_fails
        rec["gate_failures"] = ";".join(fails)
        return rec, fails, H, pred, met

    for idx in range(ref_idx + 1, stop):
        img = load_frame(idx)
        gap = idx - last["idx"]
        mask = mask_fn(idx, img) if mask_fn else None
        rec, fails, H, pred, met = attempt(key, idx, img, mask)
        fallback = False
        if fails and keyframe is not None and key["idx"] != last["idx"]:
            key_fails = rec["gate_failures"]
            rec, fails, H, pred, met = attempt(last, idx, img, mask)
            rec["keyframe_gate_failures"] = key_fails
            fallback = True
        if keyframe is not None:
            rec["anchor_mode"] = "fallback" if fallback else "keyframe"

        status, lane = None, None
        if not fails:
            status, lane = "propagated", pred
        elif refine_fn is not None:
            prior = pred if pred is not None and is_convex_quad(pred) else last_lane
            res = refine_fn(img, prior, gate)
            rec["refine_status"] = res.status
            if res.corners is not None:
                refine_fails, refine_met = lane_gate(res.corners, last_lane, image_size, gap, gate)
                rec.update({f"refined_{k}": v for k, v in refine_met.items()})
                if refine_fails:
                    rec["refine_status"] = "gate_failed:" + ";".join(refine_fails)
                else:
                    try:
                        status, lane, H = "reconstructed", res.corners, image_to_court(res.corners)
                    except AssertionError:
                        status, rec["refine_status"] = None, "degenerate_homography"

        if (status == "propagated" and reanchor is not None and refine_fn is not None
                and idx - last_fix_idx >= reanchor.every_n):
            res = refine_fn(img, pred, reanchor.refine_gate(gate))
            rec["reanchor_status"] = res.status
            if res.corners is not None:
                ra_fails, ra_met = reanchor_failures(res.corners, pred, res.counts, reanchor)
                rec.update(ra_met)
                if ra_fails:
                    rec["reanchor_status"] = ";".join(ra_fails)
                else:
                    try:
                        H_fix = image_to_court(res.corners)
                    except AssertionError:
                        rec["reanchor_status"] = "degenerate_homography"
                    else:
                        status, lane, H = "independent", res.corners, H_fix
                        last_fix_idx = idx

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
        last, last_lane = _anchor(idx, img, mask_fn, diag_mask_fn, mask), np.asarray(lane, np.float64)
        # needs_refresh only runs when no fallback happened, so `met` is the keyframe match's metrics.
        if (keyframe is None or fallback or status != "propagated"
                or keyframe.needs_refresh(idx - key["idx"], met)):
            key = last
            if keyframe is not None:
                rec["new_keyframe"] = True
        rejections = 0
    return homs, records

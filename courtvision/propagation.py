"""Chained calibration propagation (original cell 20) — kept as the BASELINE.

Known weakness: errors accumulate along the chain and nothing checks the propagated
court geometry. Frame 115 drifted 62.92 px from its independent annotation under
this scheme (handover). Use only for comparison until confidence checks and
re-anchoring are added.
"""
from typing import Callable

import cv2
import numpy as np

from .motion import MotionParams, estimate_motion


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

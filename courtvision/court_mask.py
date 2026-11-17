"""Colour-based court-floor mask (original cells 23–24). Not a semantic court segmentation."""
import cv2
import numpy as np

WOOD_HSV_LOWER = np.array([5, 25, 65], dtype=np.uint8)
WOOD_HSV_UPPER = np.array([35, 255, 255], dtype=np.uint8)


def court_mask(frame, min_area: int = 10000, exclude_bottom_frac: float = 0.15) -> np.ndarray:
    """uint8 mask (255 = court-coloured) of large warm-wood regions.

    `exclude_bottom_frac` blanks the bottom of the frame where broadcast graphics sit
    (cell 24 used 0.15). Pass 0 to disable.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, WOOD_HSV_LOWER, WOOD_HSV_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > min_area]
    out = np.isin(labels, keep).astype(np.uint8) * 255
    if exclude_bottom_frac > 0:
        out[int(out.shape[0] * (1 - exclude_bottom_frac)):, :] = 0
    return out


def exclude_boxes(mask: np.ndarray, boxes_xywh, pad_px: int = 8) -> np.ndarray:
    """Copy of `mask` with each (x, y, w, h) box (plus `pad_px`) zeroed.

    Used to drop player features: players are inside the court mask but move independently
    of the camera, so their ORB matches are not court-plane motion.
    """
    out = mask.copy()
    h, w = out.shape[:2]
    for x, y, bw, bh in np.asarray(boxes_xywh, np.float64).reshape(-1, 4):
        x0, y0 = max(int(np.floor(x - pad_px)), 0), max(int(np.floor(y - pad_px)), 0)
        x1, y1 = min(int(np.ceil(x + bw + pad_px)), w), min(int(np.ceil(y + bh + pad_px)), h)
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = 0
    return out

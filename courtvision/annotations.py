"""Manual court-line annotations — the single source of truth.

Coordinates are copied verbatim from the original notebook / handover. Do not edit
them; add new annotations as new entries.

Each annotation has 8 clicked image points (pixels, FRAME_PATHS index convention):

    P1, P2  two points on the baseline
    P3, P4  two points on the upper lane line   (far side, smaller image y)
    P5, P6  two points on the lower lane line   (near side, larger image y)
    P7, P8  the two free-throw corners, in the order given by `ft_order`

The baseline/lane corners are usually hidden by the scoreboard, so they are
reconstructed as line intersections instead of being clicked.

Lane-corner order used everywhere in this package (LANE_CORNER_ORDER):
    0 baseline-lower, 1 baseline-upper, 2 free-throw-upper, 3 free-throw-lower
"""
from dataclasses import dataclass

import numpy as np

from .geometry import intersect, line_through, point_line_distance

LANE_CORNER_ORDER = ("baseline_lower", "baseline_upper", "ft_upper", "ft_lower")


@dataclass(frozen=True)
class LineAnnotation:
    frame: int                 # FRAME_PATHS index (not the file stem)
    points: tuple              # 8 (x, y) tuples, order as clicked
    ft_order: str              # "lower_upper": P7 = FT-lower, P8 = FT-upper; "upper_lower": the reverse
    status: str                # "reference" | "partial_reference" | "historical" | "rejected"
    note: str = ""
    unreliable_points: tuple = ()  # indices (0-7) of clicks known to be inaccurate, e.g. clamped at the frame edge

    def __post_init__(self):
        assert self.status != "partial_reference" or self.unreliable_points, \
            "partial_reference requires unreliable_points"

    @property
    def array(self) -> np.ndarray:
        return np.asarray(self.points, dtype=np.float64)


ANNOTATIONS = {
    105: LineAnnotation(
        105,
        ((280, 471), (323, 422), (372, 370), (622, 394), (373, 482), (543, 499), (732, 518), (793, 411)),
        ft_order="lower_upper",
        status="reference",
        note="User-identified clear reference frame. P7/P8 order verified geometrically (each <0.6 px from its lane line).",
    ),
    115: LineAnnotation(
        115,
        ((365, 522), (416, 466), (531, 449), (681, 466), (520, 561), (759, 588), (799, 592), (861, 486)),
        ft_order="lower_upper",
        status="reference",
        note="User-identified clear reference frame; strongest reconstruction result (4.10 px mean).",
    ),
    175: LineAnnotation(
        175,
        ((197, 687), (449, 418), (386, 489), (822, 537), (419, 666), (763, 707), (949, 720), (1038, 563)),
        ft_order="lower_upper",
        status="partial_reference",
        unreliable_points=(6,),
        note="Stage 2 validation (2026-11-17). FT-lower corner is below the frame; P7 was clicked at the bottom "
             "edge as a best guess (9.1 px off the lower lane line). Use only corners 0-2.",
    ),
    199: LineAnnotation(
        199,
        ((253, 684), (527, 392), (504, 490), (1092, 560), (437, 675), (644, 700), (1092, 720), (1182, 572)),
        ft_order="lower_upper",
        status="partial_reference",
        unreliable_points=(6,),
        note="Stage 2 validation (2026-11-17). FT-lower corner is below the frame; P7 was clicked at the bottom "
             "edge as a best guess (33.9 px off the lower lane line). Use only corners 0-2.",
    ),
    150: LineAnnotation(
        150,
        ((224, 703), (474, 438), (475, 482), (931, 532), (339, 610), (789, 665), (906, 678), (987, 541)),
        ft_order="lower_upper",
        status="reference",
        note="Stage 2 validation reference (2026-11-17, new selector). FT corners 2.84 / 1.29 px from their lane lines.",
    ),
    100: LineAnnotation(
        100,
        ((261, 455), (287, 427), (356, 350), (486, 362), (375, 467), (683, 499), (789, 393), (728, 500)),
        ft_order="upper_lower",
        status="historical",
        note="Original cell 14 (drawn on FRAME_PATHS[100]). It is unverified whether this lane or the cell-15 "
             "'Frame 75' lane built the propagation reference HOMOGRAPHY.",
    ),
    200: LineAnnotation(
        200,
        ((302, 648), (457, 480), (601, 496), (1101, 557), (471, 675), (810, 716), (1185, 570), (1098, 716)),
        ft_order="upper_lower",
        status="rejected",
        note="Older experiment set (cell 26). P8 lies 34.6 px from its own lower-lane line (near the image "
             "bottom/scoreboard). Do not use as an anchor; do not mix with 105/115.",
    ),
}

# Original cell 15: four lane corners labelled "from Frame 75". Historical only.
FRAME75_LANE_CORNERS = np.array([[63, 468], [174, 341], [682, 386], [612, 508]], dtype=np.float64)


def ft_indices(ann: LineAnnotation):
    """(index of FT-upper, index of FT-lower) within the 8 points."""
    if ann.ft_order == "lower_upper":
        return 7, 6
    if ann.ft_order == "upper_lower":
        return 6, 7
    raise ValueError(f"Unknown ft_order {ann.ft_order!r}")


def seed_lines(ann: LineAnnotation) -> dict:
    """Clicked point pairs per court line, used to seed line detection (original cells 35/37)."""
    p = ann.array
    up, lo = ft_indices(ann)
    return {
        "baseline": p[0:2],
        "upper_lane": p[2:4],
        "lower_lane": p[4:6],
        "free_throw": np.array([p[up], p[lo]]),
    }


def lane_corners(ann: LineAnnotation) -> np.ndarray:
    """Lane corners in LANE_CORNER_ORDER (original cell 31 `reconstruct_lane`)."""
    p = ann.array
    base, upper, lower = line_through(p[0], p[1]), line_through(p[2], p[3]), line_through(p[4], p[5])
    up, lo = ft_indices(ann)
    return np.array([intersect(base, lower), intersect(base, upper), p[up], p[lo]], dtype=np.float64)


REFERENCE_STATUSES = ("reference", "partial_reference")


def corner_validity(ann: LineAnnotation) -> np.ndarray:
    """Boolean mask over LANE_CORNER_ORDER: True where the corner depends only on reliable clicks.

    baseline_lower = baseline x lower lane (P1, P2, P5, P6); baseline_upper = baseline x upper lane
    (P1-P4); ft_upper / ft_lower are the clicked FT corners themselves.
    """
    bad = set(ann.unreliable_points)
    up, lo = ft_indices(ann)
    needs = ({0, 1, 4, 5}, {0, 1, 2, 3}, {up}, {lo})
    return np.array([not (n & bad) for n in needs])


def check_ft_order(ann: LineAnnotation, tol_px: float = 5.0) -> dict:
    """Verify from geometry that the FT corners lie on the lane lines `ft_order` claims.

    Returns the four distances and `ok`. FT-upper must lie on the upper lane line and
    FT-lower on the lower lane line, each within `tol_px` and closer than to the other line.
    """
    p = ann.array
    up, lo = ft_indices(ann)
    d = {
        "ft_upper_to_upper": point_line_distance(p[up], p[2], p[3]),
        "ft_upper_to_lower": point_line_distance(p[up], p[4], p[5]),
        "ft_lower_to_lower": point_line_distance(p[lo], p[4], p[5]),
        "ft_lower_to_upper": point_line_distance(p[lo], p[2], p[3]),
    }
    d["ok"] = (
        d["ft_upper_to_upper"] < tol_px
        and d["ft_lower_to_lower"] < tol_px
        and d["ft_upper_to_upper"] < d["ft_upper_to_lower"]
        and d["ft_lower_to_lower"] < d["ft_lower_to_upper"]
    )
    return d

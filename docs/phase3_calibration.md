# Phase 3 — Court Calibration: status, conventions, findings

## Conventions

| Item | Convention |
|---|---|
| Calibration frame number | `FRAME_PATHS` index (0-based). |
| Tracking `frame` column | `int(path.stem)`, i.e. the SportsMOT file name. If files start at `000001.jpg`, calibration index 115 = tracking frame 116. Use `data.frame_id_of` / `data.index_of`. |
| Lane corner order | 0 baseline-lower, 1 baseline-upper, 2 FT-upper, 3 FT-lower (`annotations.LANE_CORNER_ORDER`). "Upper" = far side (smaller image y). |
| 8-point annotation | P1–P2 baseline, P3–P4 upper lane line, P5–P6 lower lane line, P7–P8 FT corners in `ft_order`. |
| Court coordinates | Lane-relative metres: origin at baseline-upper corner, +x along the lane, +y toward the lower lane line. NCAA 19 ft × 12 ft lane (cell 16). **Not verified for this court; not full-court coordinates.** |

## Annotations (`courtvision/annotations.py`)

| Frame | Status | ft_order | Notes |
|---|---|---|---|
| 105 | reference | lower_upper | Each FT corner < 0.6 px from its lane line. |
| 115 | reference | lower_upper | As above. |
| 100 | historical | upper_lower | Original cell 14. Unverified whether this, or the cell-15 "Frame 75" lane, built the propagation reference. |
| 200 | rejected | upper_lower | P8 is 34.6 px off its own lower-lane line. |

The original selector page labelled P7 = upper and P8 = lower, but frames 105/115 were clicked lower-then-upper
(confirmed geometrically). The new selector (`viz.write_selector_html`) uses the lower-then-upper order.

## Reported results (from the handover; not yet re-run in this repo)

- Static homography invalid: frame 100 vs 200 anchors disagree by 229.28 px mean (frame 150: 306.01 px).
- Court-only ORB, frames 100→101: 1,182 matches, 876 inliers (0.7411).
- Independent vs chained propagation: frame 105 mean 9.4 / max 13.41 px; frame 115 mean 62.92 / max 79.82 px.
- Frame 115 lane reconstruction: mean 4.10 / max 6.86 px vs the manual lane.

The 4.10 / 6.86 px arithmetic **is verified** (`tests/test_calibration.py`): the reference is the cell-31 lane
(baseline corners are intersections of the clicked lines: `[347.25, 541.48]`, `[440.79, 438.78]`).

## Findings from the code audit

1. **Frame-115 "automatic" result is seeded by the manual clicks.** Only segments within 14–18 px / 8–12° of the
   manual lines are used. It shows agreement with the manual lines, not prior-free detection. The manual baseline
   corners are themselves extrapolated, not observed.
2. **Propagation used full-frame ORB.** The court mask was only tested on 100→101. A synthetic test shows a small
   static overlay is normally rejected by RANSAC, so the scoreboard is a hypothesis for the drift, not a
   conclusion. Stage 1 logs static, bottom-band and off-court inlier shares per step on real frames.
3. **Reference homography source is ambiguous.** Cell 15 overwrote the cell-14 lane (differences of 107–198 px)
   before cell 16 built `HOMOGRAPHY`. The 9.4 px agreement at frame 105 suggests the cell-14 lane was used
   (inference). Stage 1 checks saved artifacts.
4. **Refinement (cell 39) is seeded by the propagated lane with a 20 px band.** With 60+ px drift the seed misses
   the true lines, so refinement inherits propagation errors.
5. **Mapping (old cell 56) applied one static homography and tested `inside_court` with full-court dimensions on
   lane-relative coordinates.** It is now gated on per-frame accepted calibrations.

## What was removed from the original notebook and why

| Original cells | Fate |
|---|---|
| 7 (full-frame Canny/Hough diagnostic) | Removed: false segments from crowd, players and graphics (handover). |
| 9–13, 29, 46–48 (frame viewers / grids / ad-hoc PNG saves) | Removed: exploratory only. |
| 13, 30, 41, 49 (four selector variants) | Merged into `viz.write_selector_html`, with corrected labels. |
| 14, 15, 26, 31, 50 (annotation + lane reconstruction copies) | Merged into `annotations.py` (data verbatim) and `annotations.lane_corners`. |
| 16, 43, 44, 45 (four-point homography + zero-by-construction checks) | Merged into `homography.py`; reprojection-of-fitting-points checks dropped. |
| 18, 19, 24, 25 (single-pair ORB experiments) | Merged into `motion.estimate_motion` (full-frame or masked). |
| 20 (chained propagation) | Merged into `propagation.chain_propagate`, kept as the baseline. |
| 21, 33, 40 (overlay checks), 27–28 (frame-200 anchor, frame-150 comparison) | Removed: results recorded above; frame-200 anchor rejected. |
| 22 (hand-drawn polygon mask) | Removed: superseded by the HSV court mask (`court_mask.py`). |
| 23 (HSV court mask) | Merged into `court_mask.py`. |
| 32 (drift table 106–115) | Superseded by the Stage 1 per-step diagnostics. |
| 34–39 (Hough, seeded line fit, intersections, refinement) | Merged into `lane.py` with identical parameters. |
| 52 (independent validation landmarks) | Superseded by multi-frame validation in later stages. |
| 54 (YOLO + ByteTrack) | Merged into `tracking.py` with logic and parameters unchanged. |
| 56, 58, 60 (static mapping, top-down plot, sample frame) | Replaced by gated per-frame mapping; plots return once calibration is validated. |
| 62 (export) | Kept and simplified. |

## Next stages

1. **Stage 1 (now):** run notebook section 2 on Kaggle and confirm the reproduction numbers.
2. **Stage 2:** confidence-gated local propagation. Use court-only features, check inlier spatial spread and
   convex/plausible lane polygons, and compare independent vs propagated estimates at the reference frames.
3. **Stage 3:** automatic re-anchoring and rejection. Frame status is one of
   `independent | propagated | reconstructed | rejected`.
4. **Stage 4:** validate on other camera-motion segments, then do per-frame footpoint mapping.

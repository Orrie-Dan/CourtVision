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
| 150 | reference | lower_upper | Added 2026-11-17. FT corners 2.84 / 1.29 px from their lane lines. |
| 175 | partial_reference | lower_upper | FT-lower corner below the frame; P7 clicked at the edge (9.1 px off). Corners 0–2 only. |
| 199 | partial_reference | lower_upper | As 175 (33.9 px off). Corners 0–2 only. |
| 100 | historical | upper_lower | Original cell 14. Unverified whether this, or the cell-15 "Frame 75" lane, built the propagation reference. |
| 200 | rejected | upper_lower | P8 is 34.6 px off its own lower-lane line. |

The original selector page labelled P7 = upper and P8 = lower, but frames 105/115 were clicked lower-then-upper
(confirmed geometrically). The new selector (`viz.write_selector_html`) uses the lower-then-upper order.

## Reported results (from the handover)

- Static homography invalid: frame 100 vs 200 anchors disagree by 229.28 px mean (frame 150: 306.01 px).
  *Not re-run.*
- Court-only ORB, frames 100→101: 1,182 matches, 876 inliers (0.7411). **Reproduced in Stage 1** (0.741).
- Independent vs chained propagation: frame 105 mean 9.4 / max 13.41 px; frame 115 mean 62.92 / max 79.82 px.
  **Reproduced in Stage 1** (full-frame mode: 9.400 / 13.408 and 62.923 / 79.822).
- Frame 115 lane reconstruction: mean 4.10 / max 6.86 px vs the manual lane.

The 4.10 / 6.86 px arithmetic **is verified** (`tests/test_calibration.py`): the reference is the cell-31 lane
(baseline corners are intersections of the clicked lines: `[347.25, 541.48]`, `[440.79, 438.78]`).

## Stage 1 results (Kaggle run, 2026-11-17)

Artifacts: `results/phase3/stage1/` — `stage1_propagation_diagnostics.csv` (per-step diagnostics, frames
101–115, `full_frame` and `court_only` modes) and `experiment_config.json` are tracked; the overlay PNGs
(`stage1_overlays_105_115.png`, `stage1_frame115_lane.png`) are kept locally and git-ignored for size.

Chained propagation error vs the manual lane:

| Mode | Frame 105 mean / max (px) | Frame 115 mean / max (px) |
|---|---|---|
| full_frame | 9.400 / 13.408 | 62.923 / 79.822 |
| court_only | 5.845 / 10.528 | — (step 114→115 `low_confidence`: 229 / 423 inliers, 0.541) |

Per-step diagnostics:

| | full_frame | court_only |
|---|---|---|
| Median inlier motion, steps 101–103 | 1.44–2.07 px | 2.44–3.58 px |
| Median inlier motion, steps 104–115 | **0.0 px** | 5.83–17.60 px |
| Static inlier share | 0.47–0.48 → 0.51–0.71 from step 104 | 0.00–0.09 |
| Off-court inlier share | 0.865–0.999 | 0 (by mask) |

Interpretation:

- **Full-frame drift is explained by static features dominating the fit.** From step 104 the median full-frame
  inlier does not move while court-only features show 6–17 px of camera motion per step, so the chained
  homography under-estimates camera motion and the lane lags. Inliers are overwhelmingly off-court and
  increasingly static. The static share tracks the bottom-band share almost exactly (e.g. 0.601 / 0.609 at
  105, 0.710 / 0.713 at 108), pointing at the bottom score bar; this is a correlation of aggregate shares,
  not a per-point check. A homography can hold that strip fixed while still moving the rest of the image,
  which is how the median inlier motion is 0 px while test points shift 5–21 px. This upgrades finding 2
  from hypothesis to supported.
- **Court-only is more accurate but not robust on its own.** Steps 110 (558 matches, 17.6 px median motion) and
  115 are weak; full-frame match counts also dip on those steps, suggesting blur or fast motion. This motivates
  the Stage 2 confidence gating rather than a straight swap to court-only.
- **Reference source (finding 3) is still unconfirmed.** No saved frame-100 lane artifact existed on Kaggle;
  the config records `"ASSUMED: cell-14 lane"`. The exact reproduction of 9.4 px is consistent with that, but
  is not proof.
- `accepted_calibrated_frames` is empty: no frame is accepted for mapping yet.

## Stage 2 — gated propagation

Code: `courtvision/gating.py`, `propagation.gated_propagate`, `court_mask.exclude_boxes`,
`tracking.boxes_for_frame`; notebook section 2b. `chain_propagate` is unchanged and remains the baseline.

- **Features:** court mask (bottom 15 % already excluded) minus padded player boxes from the detector, so
  neither the score bar, the stands (off-plane, parallax) nor moving players drive the court homography.
- **Anchor:** every frame is matched against the last *accepted* frame, so a bad frame is skipped rather
  than ending the chain (the step limit scales with the gap).
- **Per-frame status:** `reference | propagated | reconstructed | rejected | lost`.
  - `propagated`: `estimate_motion` status ok, not "pinned" (static-inlier share > 0.40 *while* the fit
    moves test points > 2 px — the Stage 1 signature; a stationary camera has ~0 shift and passes; a large
    static overlay is indistinguishable from a stationary camera by these metrics, so overlays must be
    removed by the feature mask), inlier hull ≥ 3 % of the frame
    (new `inlier_hull_frac` metric), and the predicted lane is convex, within the frame ± 25 %, moved
    ≤ 40 px per frame of gap, with area ratio in [0.80, 1.25] vs the last accepted lane.
  - `reconstructed`: gates failed, but `refine_lane_from_prior` around the prediction (or the last accepted
    lane if there is no usable prediction) moved no corner more than 25 px and its lane passes the same lane
    gate. This is still seeded refinement (finding 4) — bounded, not prior-free.
  - `rejected`: neither; `lost` after 5 consecutive rejections (propagation stops).
- **Thresholds are provisional** (`GateParams`), chosen from the frames 101–115 diagnostics only. Tune them
  once more reference frames are annotated; until then gated frames are not added to `ACCEPTED_HOMOGRAPHIES`.
- **Verified:** synthetic unit tests (`tests/test_stage2.py`) cover each gate, recovery across a rejected
  frame, `lost`, static-dominated fits being rejected, refinement fallback (stubbed and real Hough), and
  per-frame player masks.

### Stage 2 results (Kaggle run, 2026-11-17, frames 100–199, static threshold 0.20)

Artifacts: `results/phase3/stage2/stage2_gated_propagation.csv` (tracked); `stage2_overlays_105_115.png`,
`stage2_grid_120_190.png` (local, git-ignored). Detector produced 712 player boxes for the window (~7/frame,
so some people were probably missed: masking is partial).

| Frame | Stage 1 full-frame | Stage 1 court-only | Stage 2 gated |
|---|---|---|---|
| 105 (mean / max px vs manual) | 9.40 / 13.41 | 5.85 / 10.53 | **1.79 / 2.81** |
| 115 | 62.92 / 79.82 | failed (low_confidence) | **3.76 / 6.27** |

- Statuses: 92 propagated, 2 reconstructed (162, 163), 5 rejected (152, 154, 156, 158, 188), none lost.
- Static-inlier share 0.000 for frames 104–121; off-court share ≤ 0.002; bottom-band share 0.
- Frame-115 error (3.76 px) is about the size of the manual-click vs fitted-line disagreement (4.10 px),
  i.e. at the resolution the current references can measure.
- **Frames 120–190 (visual check only, half-resolution grid):** no gross failure — the lane stays near the
  painted lane and players grow with it, so the camera really zooms in (2.77× lane area over 100–199). The
  grid is too coarse to see 20–50 px errors; the measured check below **supersedes** the earlier reading of
  this grid as "no drift".
- **Rejections were false:** all five were slow pans (median motion ~1.4 px) with static share
  0.22–0.32 and the next frame accepted across the gap. Accepted frames reached 0.295; Stage 1 pinned fits
  started at 0.47. The threshold is now **0.40** (tuned on this run; recheck on new segments).
- **Reconstructions at 162–163** moved the lane 13–17 px from the motion prediction (~3 px). Frame 170
  still looks aligned, so this did not derail the chain, but whether it corrected drift or snapped to the
  wrong line is unresolved. Refinement is still seeded (finding 4).

### Stage 2 re-run (threshold 0.40) and measured error at the new references

All 99 frames 101–199 `propagated` (the 5 false rejections and the 162–163 reconstructions are gone; metrics
for 101–151 identical to the first run). Error of the gated lane vs the manual lane, per corner
(baseline-lower, baseline-upper, FT-upper, FT-lower; partial references omit FT-lower):

| Frame | Corners | Per corner (px) | Mean / max (px) |
|---|---|---|---|
| 105 | 4 | 2.8, 0.4, 1.6, 2.4 | 1.79 / 2.81 |
| 115 | 4 | 6.0, 6.3, 0.2, 2.5 | 3.76 / 6.27 |
| 150 | 4 | 7.3, 4.6, 19.6, 19.4 | 12.72 / 19.63 |
| 175 | 3 | 10.7, 4.8, 20.1 | 11.89 / 20.13 |
| 199 | 3 | 15.1, 5.6, 48.2 | 22.96 / 48.15 |

- **Gated propagation drifts after ~115.** Baseline-upper stays ~5 px, but the free-throw end drifts to
  ~20 px by 150–175 and 48 px by 199; at 199 the gated lane is visibly shorter than the manual one.
  Pattern = scale under-estimation compounding over ~100 chained steps while the camera zooms in.
- Consequence: chained motion alone is good for roughly 15 frames from an anchor. Frames after ~120 are
  **not** accurate enough to accept; `ACCEPTED_HOMOGRAPHIES` stays empty. Stage 3 (re-anchoring) is needed.
- Manual-reference precision is a few px (105/115 FT-corner checks); the 20–48 px errors are well above it.
- Image: `results/phase3/stage2/stage2_reference_check.png` (local, git-ignored).

## Stage 3 — re-anchoring by bounded refinement (run 2026-11-24: does not fix drift)

Code: `gating.ReanchorParams`, `gating.reanchor_failures`, `gated_propagate(..., reanchor=...)`; notebook
section 2c. `reanchor=None` (default) is exactly Stage 2.

- Once `every_n` frames (default 10) have passed since the reference or the last fix, a `propagated` frame is
  refined around its prediction with `refine_fn` and a **10 px** correction limit (Stage 2 fallback: 25 px).
- Guards: refined lane convex, every corner ≤ 10 px from the prediction, ≥ 3 Hough segments on each of the
  4 lines, refined/predicted area in [0.93, 1.07]. Pass → status `independent`, the refined lane becomes the
  anchor. Fail → stays `propagated` (`reanchor_status` records why) and the next frame retries.
- Rationale: Stage 2 drift after 115 is ~0.5 px/frame at the FT end, so after 10 frames the painted lines
  should be within a few px of the prediction — inside the refinement band, unlike the 60+ px case of finding 4.
- Still seeded refinement (finding 4). A snap to the wrong line within 10 px would persist; the measured
  error at 150 / 175 / 199 is the check.
- Thresholds are provisional; 2c.1 compares `every_n` = 5 / 10 / 15.
- **Verified:** synthetic tests (`tests/test_stage3.py`): guards, schedule, retry after a failed fix, guard
  rejection keeps the prediction, no re-anchor on rejected frames, and an end-to-end run with real ORB and
  real Hough refinement that pulls a lane whose FT end starts 7 px short back to < 4 px.

### Stage 3 results (Kaggle run, 2026-11-24, frames 100–199)

Artifacts: `results/phase3/stage3/stage3_reanchor_every{5,10,15}.csv` (tracked); `stage3_reference_check.png`
(local, git-ignored). Stage 2 metrics in the same run are identical to the committed Stage 2 re-run.

Mean error vs the manual lane (px):

| Frame | Stage 2 | every_5 | every_10 | every_15 |
|---|---|---|---|---|
| 105 | 1.79 | 4.39 | 1.79 | 1.79 |
| 115 | 3.76 | 6.91 | 6.56 | 6.07 |
| 150 | 12.72 | 12.59 | 12.47 | 15.06 |
| 175 | 11.89 | 15.82 | 11.44 | 13.04 |
| 199 | 22.96 | 21.76 | 22.47 | 25.04 |

Re-anchor outcomes: every_5 — 10 ok (105–150), 30 `correction_too_large`, 15 `free_throw_failed`;
every_10 — 2 ok (110, 120), 40 / 30; every_15 — 1 ok (115), 36 / 34. No frame rejected or lost.

- **Refinement is less accurate than short-range propagation.** Accepted fixes moved the lane 3–10 px and made
  105 / 115 worse (1.79 → 4.39, 3.76 → 6.91 px). The seeded Hough fit is biased by ~5–8 px vs the manual
  lines (cf. the 4.10 px click-vs-fit disagreement), so it cannot correct errors of that size.
- **It follows the seed.** every_5 re-anchored *at* frame 150 (7.3 px correction) and still has 12.6 px error:
  the fit settled near the drifted prediction, not on the painted line (finding 4 confirmed on real frames).
- **Drift exceeds the 10 px band from ~130**, so the guard blocks the corrections that matter; widening it
  would also admit wrong snaps.
- **The free-throw line is not found from ~150** (not the lower lane line, as predicted): players stand on it,
  the blue key boundary is adjacent, and its lower end runs under the score bar.
- **Conclusion:** bounded seeded refinement does not fix the drift. The code stays (`reanchor=None` default).
  Next: reduce the number of compounding motion steps (keyframe matching), see Stage 3b.

## Stage 3b — keyframe matching (implemented, not yet run on Kaggle)

Code: `gating.KeyframeParams`, `gated_propagate(..., keyframe=...)`; notebook section 2d. `keyframe=None`
(default) is exactly Stage 2.

- Each frame is matched to a **keyframe** held fixed across accepted frames instead of to the previous frame,
  so the per-step scale error compounds once per keyframe rather than ~100 times over 100–199.
- The lane gate still compares with the last accepted lane (gap 1), so its 40 px/frame step limit is unchanged.
- Keyframe refresh: after `max_gap` frames (default 25), when the keyframe match weakens (inliers < 150, or
  test-point motion > 100 px, i.e. before the 150 px `excessive_motion` limit), after a reconstructed /
  independent frame, or when the keyframe match fails the gates and the frame falls back to the last accepted
  frame (`anchor_mode = fallback`; the fallback frame becomes the keyframe).
- No line refinement is used for correction; Stage 2's refinement fallback for failed frames is unchanged.
- Synthetic check (`tests/test_stage3b.py`): with 1 %/frame zoom + pan + pixel noise over 40 frames, final
  error is 2.2 px chained vs ~1.0–1.2 px with keyframes. Real-frame size of the gain is unknown: 2d.1 compares
  `max_gap` = 10 / 25 / 50 against the references.
- Risk: matching across a larger gap means larger appearance change (zoom 2.77× over the window); ORB's
  scale pyramid should cope with the ≤ ~1.3× change inside 25 frames, and failures fall back to the chain.

## Findings from the code audit

1. **Frame-115 "automatic" result is seeded by the manual clicks.** Only segments within 14–18 px / 8–12° of the
   manual lines are used. It shows agreement with the manual lines, not prior-free detection. The manual baseline
   corners are themselves extrapolated, not observed.
2. **Propagation used full-frame ORB.** The court mask was only tested on 100→101. A synthetic test shows a small
   static overlay is normally rejected by RANSAC, so the scoreboard is a hypothesis for the drift, not a
   conclusion. Stage 1 logs static, bottom-band and off-court inlier shares per step on real frames; the
   results support the static-feature explanation (see Stage 1 results).
3. **Reference homography source is ambiguous.** Cell 15 overwrote the cell-14 lane (differences of 107–198 px)
   before cell 16 built `HOMOGRAPHY`. The 9.4 px agreement at frame 105 suggests the cell-14 lane was used
   (inference). Stage 1 found no saved artifact; still unconfirmed (see Stage 1 results).
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

1. **Stage 1 (done, 2026-11-17):** reproduction numbers confirmed; see Stage 1 results.
2. **Stage 2 (done: accurate to ~6 px for ~15 frames, then drifts to 20–48 px by 150–199):** confidence-gated local propagation. Use court-only features, check inlier spatial spread and
   convex/plausible lane polygons, and compare independent vs propagated estimates at the reference frames.
3. **Stage 3 (re-anchoring by refinement: run, does not fix drift; 3b keyframe matching implemented, awaiting run):** automatic re-anchoring and rejection. Frame status is one of
   `reference | independent | propagated | reconstructed | rejected | lost`.
4. **Stage 4:** validate on other camera-motion segments, then do per-frame footpoint mapping.

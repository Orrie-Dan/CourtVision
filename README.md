# CourtVision

Basketball broadcast analytics on SportsMOT sequence `v_00HRwkvvjtQ_c007` (300 frames, 1280×720):
player detection (YOLO26n) → tracking (optimized ByteTrack) → court calibration → top-down mapping.

## Layout

```
courtvision/            reusable package (imported by the notebook)
  config.py             sequence, detector/tracker parameters, court dimensions
  data.py               frame/checkpoint discovery, frame index <-> file id
  annotations.py        manual lane annotations (verbatim) + lane reconstruction + order check
  geometry.py           homogeneous lines / intersections / distances
  lane.py               seeded Hough line fitting -> lane corners; refinement from a prior
  court_mask.py         HSV court-floor mask
  motion.py             ORB + RANSAC frame-to-frame motion with diagnostics
  homography.py         image <-> lane-relative court homographies
  propagation.py        chained propagation (baseline)
  tracking.py           Phase 1-2 YOLO + ByteTrack (unchanged logic)
  viz.py                overlays and the landmark-selector page
notebooks/
  courtvision_phase3.ipynb   current notebook (run on Kaggle)
  archive/                   original Phase 3 notebook + handover, unmodified
docs/phase3_calibration.md   conventions, findings, removed-code map, next stages
tests/                       unit tests (no video frames needed)
```

## Running on Kaggle

The notebook imports `courtvision` from whichever of these it finds first:

1. a `git clone` of this repo at `/kaggle/working/Courtvision`, or
2. this repo uploaded as a Kaggle dataset and attached as input.

Also attach the dataset containing `best.pt`. SportsMOT frames are downloaded via `trackers` if not attached.

## Local tests

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
.venv/Scripts/python -m pytest tests
```

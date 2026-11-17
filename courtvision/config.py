"""Project-wide constants. Values are carried over from the original Phase 1–3 notebook."""
from pathlib import Path

# --- Data ---------------------------------------------------------------------
SEQUENCE_NAME = "v_00HRwkvvjtQ_c007"  # SportsMOT val sequence used since Phase 2
N_FRAMES = 300
FPS = 25.0
FRAME_SIZE = (1280, 720)  # (W, H) of this sequence

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

# --- Phase 1: detection (YOLO26n fine-tuned) ------------------------------------
IMGSZ = 1280
CONF = 0.25
IOU = 0.70
DEVICE = 0

# --- Phase 2: optimized ByteTrack (IDF1 47.89 %, MOTA 60.54 %, 76 ID switches) ----
BYTETRACK_OVERRIDES = {
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.10,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.90,
    "fuse_score": True,
}

# --- Phase 3: court geometry ----------------------------------------------------
# NCAA dimensions as used in the original notebook (cell 16).
# UNVERIFIED for this specific court: confirm before reporting metric quantities.
COURT_LENGTH_M = 28.6512  # 94 ft
COURT_WIDTH_M = 15.24     # 50 ft
LANE_LENGTH_M = 5.7912    # 19 ft, baseline to far edge of free-throw line
LANE_WIDTH_M = 3.6576     # 12 ft

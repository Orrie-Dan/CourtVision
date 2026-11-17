"""Locating SportsMOT frames and the detector checkpoint (original cell 5).

Frame indexing: calibration code uses `frame_paths[index]` (0-based); the tracking
CSV uses `int(path.stem)` (SportsMOT file names, typically 1-based). Use
`frame_id_of` / `index_of` to convert — never assume they are equal.
"""
import subprocess
from pathlib import Path

from .config import KAGGLE_INPUT, KAGGLE_WORKING, N_FRAMES, SEQUENCE_NAME

DATA_ROOT = KAGGLE_WORKING / "courtvision_data"


def find_sequence(name: str = SEQUENCE_NAME, roots=(DATA_ROOT, KAGGLE_INPUT)):
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob(name):
            if (p / "img1").is_dir():
                return p
    return None


def get_frame_paths(name: str = SEQUENCE_NAME, n_frames: int = N_FRAMES, download: bool = True):
    """Sorted frame paths of the sequence, downloading SportsMOT val via `trackers` if missing."""
    seq = find_sequence(name)
    if seq is None and download:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        subprocess.run(["trackers", "download", "sportsmot", "--split", "val",
                        "--asset", "frames,annotations", "-o", str(DATA_ROOT)], check=True)
        seq = find_sequence(name)
    if seq is None:
        raise FileNotFoundError(f"SportsMOT sequence {name} not found")
    paths = sorted((seq / "img1").glob("*.jpg"))[:n_frames]
    if not paths:
        raise FileNotFoundError(f"No frames in {seq / 'img1'}")
    return seq, paths


def find_checkpoint(root: Path = KAGGLE_INPUT):
    """Prefer a best.pt whose path mentions 'courtvision' (Phase 1 detector)."""
    pts = sorted(Path(root).rglob("best.pt")) if Path(root).exists() else []
    return next((p for p in pts if "courtvision" in str(p).lower()), pts[0] if pts else None)


def frame_id_of(frame_paths, index: int) -> int:
    """Tracking frame id (file stem) for a FRAME_PATHS index."""
    return int(frame_paths[index].stem)


def index_of(frame_paths, frame_id: int) -> int:
    """FRAME_PATHS index for a tracking frame id (file stem)."""
    for i, p in enumerate(frame_paths):
        if int(p.stem) == frame_id:
            return i
    raise KeyError(frame_id)

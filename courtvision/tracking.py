"""Phase 1 + 2: YOLO26n detection with optimized ByteTrack (original cell 54, logic unchanged)."""
from pathlib import Path

import cv2
import pandas as pd

from .config import BYTETRACK_OVERRIDES, CONF, DEVICE, IMGSZ, IOU

TRACK_COLUMNS = ["frame", "track_id", "x", "y", "w", "h"]


def write_tracker_yaml(out_path: Path, overrides: dict = BYTETRACK_OVERRIDES) -> Path:
    import ultralytics
    import yaml
    src = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml"
    cfg = yaml.safe_load(src.read_text())
    cfg.update(overrides)
    out_path = Path(out_path)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_path


def run_tracking(checkpoint, frame_paths, tracker_yaml, imgsz=IMGSZ, conf=CONF, iou=IOU,
                 device=DEVICE, log_every=100) -> pd.DataFrame:
    """Track all frames. `frame` column = int(path.stem) (file id, not FRAME_PATHS index)."""
    from ultralytics import YOLO
    model = YOLO(str(checkpoint))
    rows = []
    for i, path in enumerate(frame_paths):
        frame = cv2.imread(str(path))
        result = model.track(frame, persist=True, tracker=str(tracker_yaml), imgsz=imgsz, conf=conf,
                             iou=iou, device=device, verbose=False)[0]
        if result.boxes is not None and result.boxes.id is not None:
            for tid, box in zip(result.boxes.id.int().cpu().tolist(), result.boxes.xyxy.cpu().numpy()):
                x1, y1, x2, y2 = map(float, box)
                rows.append({"frame": int(path.stem), "track_id": int(tid),
                             "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1})
        if log_every and (i + 1) % log_every == 0:
            print("Tracked", i + 1, "/", len(frame_paths))
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


def load_tracks(csv_path) -> pd.DataFrame:
    tracks = pd.read_csv(csv_path)
    missing = set(TRACK_COLUMNS) - set(tracks.columns)
    assert not missing, f"Track CSV missing columns: {missing}"
    return tracks


def boxes_for_frame(tracks: pd.DataFrame, frame_id: int):
    """(N, 4) array of (x, y, w, h) boxes for a tracking frame id (file stem, not FRAME_PATHS index)."""
    return tracks.loc[tracks["frame"] == frame_id, ["x", "y", "w", "h"]].to_numpy(dtype=float)


def add_footpoints(tracks: pd.DataFrame) -> pd.DataFrame:
    """Bottom-centre of each box (pixel footpoint)."""
    tracks = tracks.copy()
    tracks["foot_x_px"] = tracks["x"] + tracks["w"] / 2
    tracks["foot_y_px"] = tracks["y"] + tracks["h"]
    return tracks

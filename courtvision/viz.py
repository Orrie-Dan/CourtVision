"""Drawing helpers and the standalone landmark-selector page (original cells 13/30/41/49)."""
import base64
from pathlib import Path

import cv2
import numpy as np

GREEN, RED, BLUE, YELLOW = (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)


def draw_polygon(img, pts, color=GREEN, thickness=2, label_corners=False):
    """Draw a closed polygon in place (BGR). Returns img."""
    pts = np.asarray(pts, np.float64)
    if not np.isfinite(pts).all():
        return img
    cv2.polylines(img, [np.round(pts).astype(np.int32)], True, color, thickness)
    if label_corners:
        for i, (x, y) in enumerate(pts, 1):
            cv2.circle(img, (int(round(x)), int(round(y))), 5, color, -1)
            cv2.putText(img, f"P{i}", (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def draw_infinite_line(img, hline, color=YELLOW, thickness=2):
    """Draw a homogeneous line a*x + b*y + c = 0 across the image."""
    a, b, c = hline
    h, w = img.shape[:2]
    if abs(b) > abs(a):
        p1, p2 = (0, -c / b), (w, -(c + a * w) / b)
    else:
        p1, p2 = (-c / a, 0), (-(c + b * h) / a, h)
    cv2.line(img, tuple(int(round(v)) for v in p1), tuple(int(round(v)) for v in p2), color, thickness)
    return img


# Labels match the convention of the 105/115 reference annotations (P7 = FT-lower,
# P8 = FT-upper). The original selector said the reverse, which did not match how
# those frames were actually clicked. Record new annotations with ft_order="lower_upper".
SELECTOR_LABELS = [
    "Baseline point A", "Baseline point B",
    "Upper lane line point A", "Upper lane line point B",
    "Lower lane line point A", "Lower lane line point B",
    "LOWER free-throw corner (near side)", "UPPER free-throw corner (far side)",
]

_SELECTOR_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>CourtVision Landmark Selector</title>
<style>
body{font-family:Arial;margin:20px;background:#f5f5f5}
#c{position:relative;width:__W__px;max-width:100%}
#img{width:100%;display:block;cursor:crosshair}
#ov{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
pre{background:#222;color:#90ee90;padding:15px}button{padding:10px 16px;margin-right:8px}
</style></head><body>
<h2>CourtVision — __TITLE__</h2><p id="ins"></p>
<div id="c"><img id="img" src="data:image/png;base64,__IMG__"><canvas id="ov" width="__W__" height="__H__"></canvas></div>
<p><button onclick="undo()">Undo</button><button onclick="reset()">Reset</button>
<button onclick="copyPts()">Copy</button></p><pre id="out"></pre>
<script>
const labels=__LABELS__;let pts=[];
const img=document.getElementById("img"),ctx=document.getElementById("ov").getContext("2d");
img.addEventListener("click",e=>{if(pts.length>=labels.length)return;const r=img.getBoundingClientRect();
pts.push([Math.round((e.clientX-r.left)*img.naturalWidth/r.width),Math.round((e.clientY-r.top)*img.naturalHeight/r.height)]);draw();});
function draw(){ctx.clearRect(0,0,__W__,__H__);pts.forEach(([x,y],i)=>{ctx.beginPath();ctx.arc(x,y,6,0,7);ctx.fillStyle="red";ctx.fill();
ctx.font="bold 18px Arial";ctx.fillStyle="yellow";ctx.fillText("P"+(i+1),x+10,y-10);});
document.getElementById("ins").textContent="Next click: "+(pts.length<labels.length?labels[pts.length]:"done — copy the coordinates");
document.getElementById("out").textContent="frame=__FRAME__  ft_order='lower_upper'\\nLINE_POINTS = "+JSON.stringify(pts);}
function undo(){pts.pop();draw();}function reset(){pts=[];draw();}
function copyPts(){navigator.clipboard.writeText(JSON.stringify(pts));}
draw();
</script></body></html>"""


def write_selector_html(frame, frame_index: int, out_path) -> Path:
    """Standalone page for clicking the 8 lane landmarks on one frame."""
    ok, png = cv2.imencode(".png", frame)
    assert ok, "PNG encoding failed"
    h, w = frame.shape[:2]
    labels = "[" + ",".join(f'"{s}"' for s in SELECTOR_LABELS) + "]"
    html = (_SELECTOR_TEMPLATE
            .replace("__IMG__", base64.b64encode(png.tobytes()).decode("ascii"))
            .replace("__W__", str(w)).replace("__H__", str(h))
            .replace("__TITLE__", f"FRAME_PATHS[{frame_index}] landmark selector")
            .replace("__FRAME__", str(frame_index))
            .replace("__LABELS__", labels))
    out_path = Path(out_path)
    out_path.write_text(html, encoding="utf-8")
    return out_path

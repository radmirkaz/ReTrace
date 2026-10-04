"""Re-cut the four RoundaboutHD feeds (Cambie corridor) at one common, synchronised window.

The window is chosen from RoundaboutHD's multi-camera ground truth to contain the most
vehicles seen by two or more cameras, so the same car can be followed between feeds.

Run on a CUDA machine from the repository root, after export_live.py:
    python pipeline/export_cambie_sync.py
"""
import json
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

from export_live import DUR, OUT, RHD, export
from make_web_clips import WEIGHTS

CAMS = {"cambie-king-edward-n": 1, "cambie-king-edward-e": 2, "cambie-king-edward-s": 3, "cambie-king-edward-w": 4}
FPS = 15  # RoundaboutHD native frame rate


def best_start():
    seen = defaultdict(lambda: defaultdict(set))  # frame bucket (s) -> id -> cams
    for line in (RHD / "Multi_CAM_Ground_Turth.txt").read_text().splitlines():
        cam, vid, frame = line.split()[:3]
        seen[int(frame) // FPS][int(vid)].add(int(cam))
    last = max(seen)
    best, best_n = 0, -1
    for s in range(0, last - int(DUR), 5):
        cams_by_id = defaultdict(set)
        for sec in range(s, s + int(DUR)):
            for vid, cams in seen.get(sec, {}).items():
                cams_by_id[vid] |= cams
        n = sum(len(c) >= 2 for c in cams_by_id.values())
        if n > best_n:
            best, best_n = s, n
    print(f"common window starts at {best}s with {best_n} vehicles seen by >=2 cameras")
    return float(best)


def main():
    start = best_start()
    model = YOLO(str(WEIGHTS))
    for cid, cam in CAMS.items():
        export(model, cid, RHD / f"imagesc00{cam}" / "video.mp4", start, DUR)
    spec_path = OUT / "live_spec.json"
    spec = json.loads(spec_path.read_text())
    for c in spec["cameras"]:
        if c["id"] in CAMS:
            c["start"] = start
            c["gt"] = {"dataset": "roundabouthd", "cam": CAMS[c["id"]], "scale": 0.5, "fps": FPS}
    spec_path.write_text(json.dumps(spec, indent=1))


if __name__ == "__main__":
    main()

"""Re-export the night and snow scenes from hand-checked windows (see pipeline/pick_weather.py).

Story clips (3 s, scroll-scrub) and live feeds (45 s) are both refreshed and their specs updated.
"""
import json
from pathlib import Path

from ultralytics import YOLO

import export_cameras
import export_live
from make_web_clips import WEIGHTS

ROOT = Path(__file__).resolve().parents[1]
AAU = ROOT / "demo_footage" / "AAU_RainSnow" / "aaurainsnow"
# camera id -> (sequence, window start s from pick_weather, condition)
PICKS = {
    "georgia-denman": ("Ostre/Ostre-4", 210.0, "night"),
    "hastings-commercial": ("Hjorringvej/Hjorringvej-4", 228.0, "night"),
    "marine-main": ("Egensevej/Egensevej-4", 0.0, "snow"),
}
STORY = {"georgia-denman": 3.0, "marine-main": 3.0}  # story scenes (offset into the window, s)


def patch(spec_path, cid, seq, start, cond):
    spec = json.loads(spec_path.read_text())
    for c in spec["cameras"]:
        if c["id"] == cid:
            c["source"] = f"AAU RainSnow · {seq.split('/')[1]} (Denmark)"
            c["condition"] = cond
            c["start"] = start
    spec_path.write_text(json.dumps(spec, indent=1))


def main():
    model = YOLO(str(WEIGHTS))
    for cid, (seq, start, cond) in PICKS.items():
        mkv = AAU / seq / "cam1.mkv"
        live_start = max(0.0, start - 10.0)
        export_live.export(model, cid, mkv, live_start, export_live.DUR)
        patch(export_live.OUT / "live_spec.json", cid, seq, live_start, cond)
        if cid in STORY:
            export_cameras.export(model, cid, mkv, start + STORY[cid], 3.0, fps=20)
            patch(export_cameras.OUT / "cameras_spec.json", cid, seq, start + STORY[cid], cond)


if __name__ == "__main__":
    main()

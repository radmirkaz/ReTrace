"""Cut short demo clips, track vehicles with the ReTrace detector, and export
scroll-scrubbable MP4s plus per-frame box JSON for the website.

Run on the home PC in the `canai` env from C:\\Users\\Radmir\\Desktop\\stormhacks:
    python pipeline\\make_web_clips.py
Outputs go to demo_footage\\web\\: <name>.mp4 (every frame a keyframe, 960 px wide),
<name>.json (per-frame tracks) and <name>.jpg (poster frame).
"""
import json
import subprocess
from pathlib import Path

import cv2
import supervision as sv
from ultralytics import YOLO

ROOT = Path(r"C:\Users\Radmir\Desktop\stormhacks")
CANAI = Path(r"C:\Users\Radmir\Desktop\canai25")  # baseline project: source videos and detector weights, read-only
OUT = ROOT / "demo_footage" / "web"
WEIGHTS = CANAI / "submission" / "streamlit" / "yolo" / "best_v10.pt"
WIDTH = 960
CONF = 0.25

# name, source video, start second, duration seconds
CLIPS = [
    ("overpass", CANAI / "yolo" / "videos" / "vehicles.mp4", 2.0, 3.0),
    ("pole", CANAI / "yolo" / "videos" / "sherbrooke_video.mp4", 20.0, 3.0),
    ("dusk", CANAI / "yolo" / "videos" / "traffic_3rd.mp4", 5.0, 3.0),
    ("longrange", CANAI / "yolo" / "videos" / "traffic_35min.mp4", 60.0, 3.0),
]


def cut(src: Path, start: float, dur: float, dst: Path, fps: float = 25, width: int = WIDTH) -> None:
    """Re-encode a clip so every frame is a keyframe; this is what makes scroll scrubbing smooth."""
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(dur), "-i", str(src),
        "-vf", f"scale='min({width},iw)':-2,fps={fps}", "-an",
        "-c:v", "libx264", "-preset", "slow", "-crf", "26", "-pix_fmt", "yuv420p",
        "-g", "1", "-keyint_min", "1", "-movflags", "+faststart", str(dst),
    ], check=True)


def track(model: YOLO, clip: Path) -> dict:
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = sv.ByteTrack(frame_rate=int(round(fps)))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        det = sv.Detections.from_ultralytics(model(frame, conf=CONF, verbose=False)[0])
        det = tracker.update_with_detections(det)
        boxes = []
        for xyxy, tid, conf, cls in zip(det.xyxy.tolist(), det.tracker_id.tolist(), det.confidence.tolist(), det.class_id.tolist()):
            boxes.append([int(tid)] + [round(v, 1) for v in xyxy] + [round(float(conf), 3), int(cls)])
        frames.append(boxes)
    cap.release()
    return {"w": w, "h": h, "fps": fps, "n": len(frames), "names": model.names, "frames": frames}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, src, start, dur in CLIPS:
        mp4 = OUT / f"{name}.mp4"
        cut(src, start, dur, mp4)
        model = YOLO(str(WEIGHTS))
        data = track(model, mp4)
        (OUT / f"{name}.json").write_text(json.dumps(data, separators=(",", ":")))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-frames:v", "1", str(OUT / f"{name}.jpg")], check=True)
        ids = {b[0] for f in data["frames"] for b in f}
        print(f"{name}: {data['n']} frames, {len(ids)} tracks, {mp4.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

"""Export the live-map feeds: 45 s high-quality windows, tracked at native resolution.

Each window is picked automatically (vehicle count, vehicle size, detector confidence),
then cut at <=1920 px for tracking/crops and 960 px for playback, and looped on the map.

Run on the GPU PC from the repository root (canai env):
    python pipeline/export_live.py
Writes demo_footage/live/<camera>.{mp4,json,jpg}, <camera>_hi.{mp4,json} and live_spec.json.
"""
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from make_web_clips import WEIGHTS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "demo_footage"
OUT = DATA / "live"
CANAI_VIDEOS = Path(r"C:\Users\Radmir\Desktop\canai25\yolo\videos")
AAU = DATA / "AAU_RainSnow" / "aaurainsnow"
RHD = DATA / "RoundaboutHD" / "RoundaboutHD"
CF = DATA / "CityFlowV2"
DUR = 45.0
CONF = 0.25

# id, name, lat, lon, video, view, condition, source label, fixed start (None = auto-pick)
FEEDS = [
    ("cambie-king-edward-n", "Cambie St & King Edward · N", 49.2490, -123.1150, RHD / "imagesc001" / "video.mp4", "roundabout", "day", "RoundaboutHD · cam 1 (4K)", None),
    ("cambie-king-edward-e", "Cambie St & King Edward · E", 49.2487, -123.1140, RHD / "imagesc002" / "video.mp4", "roundabout", "day", "RoundaboutHD · cam 2 (4K)", None),
    ("cambie-king-edward-s", "Cambie St & King Edward · S", 49.2480, -123.1147, RHD / "imagesc003" / "video.mp4", "roundabout", "day", "RoundaboutHD · cam 3 (4K)", None),
    ("cambie-king-edward-w", "Cambie St & King Edward · W", 49.2484, -123.1158, RHD / "imagesc004" / "video.mp4", "roundabout", "day", "RoundaboutHD · cam 4 (4K)", None),
    ("broadway-arbutus", "Broadway & Arbutus St", 49.2638, -123.1530, CF / "validation" / "S05" / "c033" / "vdo.avi", "pole", "day", "CityFlowV2 · S05/c033 (Iowa)", None),
    ("broadway-oak", "Broadway & Oak St", 49.2632, -123.1270, CF / "validation" / "S05" / "c034" / "vdo.avi", "pole", "day", "CityFlowV2 · S05/c034 (Iowa)", None),
    ("broadway-commercial", "Broadway & Commercial Dr", 49.2622, -123.0697, CF / "validation" / "S05" / "c026" / "vdo.avi", "pole", "day", "CityFlowV2 · S05/c026 (Iowa)", None),
    ("granville-broadway", "Granville St & Broadway", 49.2633, -123.1386, AAU / "Ostre" / "Ostre-2" / "cam1.mkv", "pole", "rain", "AAU RainSnow · Ostre-2 (Denmark)", None),
    ("georgia-denman", "Georgia St & Denman St", 49.2925, -123.1350, AAU / "Hobrovej" / "Hobrovej-1" / "cam1.mkv", "pole", "night", "AAU RainSnow · Hobrovej-1 (Denmark)", None),
    ("hastings-commercial", "Hastings St & Commercial Dr", 49.2810, -123.0700, AAU / "Hjorringvej" / "Hjorringvej-3" / "cam1.mkv", "pole", "night", "AAU RainSnow · Hjorringvej-3 (Denmark)", None),
    ("marine-main", "SE Marine Dr & Main St", 49.2110, -123.1010, AAU / "Egensevej" / "Egensevej-4" / "cam1.mkv", "pole", "snow", "AAU RainSnow · Egensevej-4 (Denmark)", None),
    ("hwy1-boundary", "Hwy 1 at Boundary Rd", 49.2590, -123.0235, CANAI_VIDEOS / "test2_4k.mp4", "overpass", "day", "Highway overpass (4K)", None),
    ("knight-bridge", "Knight St Bridge", 49.2050, -123.0770, CANAI_VIDEOS / "traffic_3rd.mp4", "overpass", "dusk", "Motorway at dusk (1440p)", 0.0),
]
KINGSWAY = [("kingsway-fraser", "Kingsway & Fraser St", 49.2496, -123.0890),
            ("kingsway-knight", "Kingsway & Knight St", 49.2445, -123.0770),
            ("kingsway-victoria", "Kingsway & Victoria Dr", 49.2415, -123.0660),
            ("kingsway-nanaimo", "Kingsway & Nanaimo St", 49.2360, -123.0565),
            ("kingsway-joyce", "Kingsway & Joyce St", 49.2295, -123.0390)]


def probe(path):
    cap = cv2.VideoCapture(str(path))
    info = (cap.get(cv2.CAP_PROP_FPS) or 25, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


def pick_window(model, path, dur=DUR):
    """Seek-sample one frame every 2 s and choose the window with the most, biggest, surest vehicles."""
    fps, n = probe(path)
    total = n / fps
    if total <= dur:
        return 0.0
    cap = cv2.VideoCapture(str(path))
    times, scores = [], []
    for t in np.arange(0, total, 2.0):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (1280, int(1280 * frame.shape[0] / frame.shape[1])))
        r = model(small, conf=0.3, verbose=False)[0]
        if r.boxes is None or not len(r.boxes):
            s = 0.0
        else:
            xyxy = r.boxes.xyxy.cpu().numpy()
            area = ((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])) / (small.shape[0] * small.shape[1])
            s = min(len(xyxy), 12) / 12 + min(float(np.sort(area)[-3:].mean()) / 0.02, 1) + float(r.boxes.conf.mean())
        times.append(t)
        scores.append(s)
    cap.release()
    k = int(dur / 2)
    if len(scores) <= k:
        return 0.0
    rolling = np.convolve(scores, np.ones(k) / k, mode="valid")
    return float(times[int(np.argmax(rolling))])


def cut(src, start, dur, dst, width, fps, all_intra=False):
    gop = ["-g", "1", "-keyint_min", "1"] if all_intra else ["-g", str(int(fps * 2))]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(dur), "-i", str(src),
                    "-vf", f"scale='min({width},iw)':-2,fps={fps}", "-an", "-c:v", "libx264", "-preset", "medium",
                    "-crf", "20" if width > 960 else "26", "-pix_fmt", "yuv420p", *gop, "-movflags", "+faststart", str(dst)], check=True)


def track(model, path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = sv.ByteTrack(frame_rate=int(round(fps)))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        det = tracker.update_with_detections(sv.Detections.from_ultralytics(model(frame, conf=CONF, verbose=False)[0]))
        frames.append([[int(t)] + [round(v, 1) for v in b] + [round(float(c), 3), int(k)]
                       for b, t, c, k in zip(det.xyxy.tolist(), det.tracker_id.tolist(), det.confidence.tolist(), det.class_id.tolist())])
    cap.release()
    return {"w": w, "h": h, "fps": fps, "n": len(frames), "names": model.names, "frames": frames}


def export(model, cid, src, start, dur):
    src_fps, _ = probe(src)
    fps = min(15, round(src_fps))
    hi = OUT / f"{cid}_hi.mp4"
    cut(src, start, dur, hi, 1920, fps)
    data = track(model, hi)
    (OUT / f"{cid}_hi.json").write_text(json.dumps(data, separators=(",", ":")))
    web = OUT / f"{cid}.mp4"
    cut(hi, 0, dur, web, 960, fps)
    cap = cv2.VideoCapture(str(web))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    sx, sy = w / data["w"], h / data["h"]
    scaled = {**data, "w": w, "h": h, "frames": [[[d[0], round(d[1] * sx, 1), round(d[2] * sy, 1), round(d[3] * sx, 1), round(d[4] * sy, 1), d[5], d[6]] for d in f] for f in data["frames"]]}
    (OUT / f"{cid}.json").write_text(json.dumps(scaled, separators=(",", ":")))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1", "-i", str(web), "-frames:v", "1", str(OUT / f"{cid}.jpg")], check=True)
    print(f"{cid}: start {start:.0f}s, {data['n']} frames @ {fps} fps, {len({b[0] for f in data['frames'] for b in f})} tracks, {data['w']}x{data['h']}, {web.stat().st_size / 1e6:.1f} MB", flush=True)


def corridor_target():
    base = CF / "train" / "S01"
    cams = sorted(base.glob("c*"))
    seen = {}
    for cam in cams:
        for line in (cam / "gt" / "gt.txt").read_text().splitlines():
            f, vid, x, y, w, h = map(float, line.split(",")[:6])
            seen.setdefault(int(vid), {}).setdefault(cam.name, []).append((int(f), x, y, w, h))
    full = {v: c for v, c in seen.items() if len(c) == len(cams)}
    vid = max(full, key=lambda v: np.mean([w * h for rows in full[v].values() for *_, w, h in rows]))
    return base, vid, full[vid]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    spec, trace = [], {}
    for cid, name, lat, lon, src, view, cond, label, start in FEEDS:
        if not Path(src).exists():
            print("missing", src)
            continue
        s = pick_window(model, src) if start is None else start
        export(model, cid, src, s, DUR)
        spec.append(dict(id=cid, name=name, lat=lat, lon=lon, view=view, condition=cond, source=label))

    base, vid, rows = corridor_target()
    for (cid, name, lat, lon), cam in zip(KINGSWAY, sorted(rows)):
        frames = sorted(rows[cam])
        fps_src, n = probe(base / cam / "vdo.avi")
        mid = frames[len(frames) // 2]
        start = max(0.0, min(mid[0] / fps_src - DUR / 2, n / fps_src - DUR))
        export(model, cid, base / cam / "vdo.avi", start, DUR)
        trace[cid] = {"gt_vehicle": vid, "t": round(mid[0] / fps_src - start, 2), "box": mid[1:]}
        spec.append(dict(id=cid, name=name, lat=lat, lon=lon, view="pole", condition="day", source=f"CityFlowV2 · S01/{cam} (Iowa)"))

    (OUT / "live_spec.json").write_text(json.dumps({"cameras": spec, "trace": trace}, indent=1))
    print(f"{len(spec)} live feeds; corridor target {vid}")


if __name__ == "__main__":
    main()

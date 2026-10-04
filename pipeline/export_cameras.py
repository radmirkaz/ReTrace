"""Export every demo camera for the live map: pick a good window, cut a scrub-ready clip, track it.

Run on the GPU PC from the repository root (canai env):
    python pipeline/export_cameras.py
Writes demo_footage/web/<camera>.{mp4,json,jpg} and demo_footage/web/cameras_spec.json,
which pipeline/build_cache.py turns into the website cache.

Cameras are real public-dataset feeds placed at Vancouver locations for the demo; `source`
records where each one really comes from and is shown in the UI.
"""
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from make_web_clips import WEIGHTS, cut, track
from scan_candidates import scan, score_windows

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "demo_footage"
OUT = DATA / "web"
CANAI_VIDEOS = Path(r"C:\Users\Radmir\Desktop\canai25\yolo\videos")
AAU = DATA / "AAU_RainSnow" / "aaurainsnow"
CITYFLOW = DATA / "CityFlowV2" / "train" / "S01"

# Kingsway corridor, west to east: CityFlow S01 cameras c001..c005 (Dubuque, Iowa).
KINGSWAY = [("kingsway-fraser", "Kingsway & Fraser St", 49.2496, -123.0890),
            ("kingsway-knight", "Kingsway & Knight St", 49.2445, -123.0770),
            ("kingsway-victoria", "Kingsway & Victoria Dr", 49.2415, -123.0660),
            ("kingsway-nanaimo", "Kingsway & Nanaimo St", 49.2360, -123.0565),
            ("kingsway-joyce", "Kingsway & Joyce St", 49.2295, -123.0390)]

FIXED = [  # id, name, lat, lon, source video, start s, duration s, view, condition, source label
    ("hwy1-boundary", "Hwy 1 at Boundary Rd", 49.2590, -123.0235, CANAI_VIDEOS / "vehicles.mp4", 2.0, 3.0, "overpass", "day", "Highway overpass clip (EU)"),
    ("main-broadway", "Main St & Broadway", 49.2627, -123.1007, CANAI_VIDEOS / "sherbrooke_video.mp4", 20.0, 3.0, "pole", "day", "Urban Tracker · Sherbrooke, QC"),
    ("knight-bridge", "Knight St Bridge", 49.2050, -123.0770, CANAI_VIDEOS / "traffic_3rd.mp4", 5.0, 3.0, "overpass", "dusk", "Motorway at dusk (UK)"),
    ("oak-bridge", "Oak St Bridge", 49.2040, -123.1270, CANAI_VIDEOS / "traffic_35min.mp4", 60.0, 3.0, "overpass", "day", "Motorway, 720p (UK)"),
]

WEATHER = [  # id, name, lat, lon, AAU sequence, view, condition
    ("granville-broadway", "Granville St & Broadway", 49.2633, -123.1386, "Ostre/Ostre-2", "pole", "rain"),
    ("cambie-41st", "Cambie St & 41st Ave", 49.2335, -123.1166, "Hadsundvej/Hadsundvej-1", "pole", "rain"),
    ("georgia-denman", "Georgia St & Denman St", 49.2925, -123.1350, "Hobrovej/Hobrovej-1", "pole", "night"),
    ("hastings-commercial", "Hastings St & Commercial Dr", 49.2810, -123.0700, "Hjorringvej/Hjorringvej-3", "pole", "night"),
    ("marine-main", "SE Marine Dr & Main St", 49.2110, -123.1010, "Egensevej/Egensevej-4", "pole", "snow"),
    ("lions-gate", "Lions Gate Bridge approach", 49.3120, -123.1390, "Egensevej/Egensevej-2", "pole", "rain"),
]


def best_window(model, video: Path, dur: float) -> float:
    picks = score_windows(scan(model, video))
    return picks[0]["start"] if picks else 0.0


def cityflow_target():
    """Vehicle ID seen by all S01 cameras with the largest average box, our 'trace one car'."""
    seen = {}
    for cam in sorted(CITYFLOW.glob("c*")):
        for line in (cam / "gt" / "gt.txt").read_text().splitlines():
            f, vid, x, y, w, h = map(float, line.split(",")[:6])
            seen.setdefault(int(vid), {}).setdefault(cam.name, []).append((int(f), x, y, w, h))
    full = {vid: c for vid, c in seen.items() if len(c) == len(list(CITYFLOW.glob("c*")))}
    vid = max(full, key=lambda v: np.mean([w * h for rows in full[v].values() for *_, w, h in rows]))
    return vid, full[vid]


def export(model, cam_id, src, start, dur, fps=25):
    """Track at native resolution (<=1920 px) for sharp crops; ship a 960 px clip with scaled boxes."""
    hi = OUT / f"{cam_id}_hi.mp4"
    cut(src, start, dur, hi, fps=fps, width=1920)
    data = track(model, hi)
    (OUT / f"{cam_id}_hi.json").write_text(json.dumps(data, separators=(",", ":")))
    mp4 = OUT / f"{cam_id}.mp4"
    cut(hi, 0, dur, mp4, fps=fps)
    import cv2
    cap = cv2.VideoCapture(str(mp4))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    sx, sy = w / data["w"], h / data["h"]
    web = {**data, "w": w, "h": h, "frames": [[[d[0], round(d[1] * sx, 1), round(d[2] * sy, 1), round(d[3] * sx, 1), round(d[4] * sy, 1), d[5], d[6]] for d in f] for f in data["frames"]]}
    (OUT / f"{cam_id}.json").write_text(json.dumps(web, separators=(",", ":")))
    import subprocess
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-frames:v", "1", str(OUT / f"{cam_id}.jpg")], check=True)
    ids = {b[0] for f in data["frames"] for b in f}
    print(f"{cam_id}: {data['n']} frames, {len(ids)} tracks, native {data['w']}x{data['h']}")
    return data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    spec, trace = [], {}

    for cid, name, lat, lon, src, start, dur, view, cond, label in FIXED:
        export(model, cid, src, start, dur)
        spec.append(dict(id=cid, name=name, lat=lat, lon=lon, view=view, condition=cond, source=label))

    for cid, name, lat, lon, seq, view, cond in WEATHER:
        mkv = AAU / seq / "cam1.mkv"
        start = best_window(model, mkv, 3.0)
        export(model, cid, mkv, start, 3.0, fps=20)
        spec.append(dict(id=cid, name=name, lat=lat, lon=lon, view=view, condition=cond, source=f"AAU RainSnow · {seq.split('/')[1]} (Denmark)"))

    vid, rows = cityflow_target()
    for (cid, name, lat, lon), cam in zip(KINGSWAY, sorted(rows)):
        frames = sorted(rows[cam])
        f0, f1 = frames[0][0], frames[-1][0]
        start = max(0.0, (f0 - 15) / 10.0)
        dur = min(6.0, max(3.0, (f1 - f0 + 30) / 10.0))
        mid = frames[len(frames) // 2]
        export(model, cid, CITYFLOW / cam / "vdo.avi", start, dur, fps=10)
        # Ground-truth box of the target at its middle appearance, in clip time.
        trace[cid] = {"gt_vehicle": vid, "t": round(mid[0] / 10.0 - start, 2), "box": mid[1:]}
        spec.append(dict(id=cid, name=name, lat=lat, lon=lon, view="pole", condition="day", source=f"CityFlowV2 · S01/{cam} (Iowa, USA)"))

    (OUT / "cameras_spec.json").write_text(json.dumps({"cameras": spec, "trace": trace}, indent=1))
    print(f"{len(spec)} cameras; CityFlow target vehicle {vid}")


if __name__ == "__main__":
    main()

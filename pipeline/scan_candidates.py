"""Shortlist the best 3-second windows in every demo video for the ReTrace website.

Run on a CUDA machine from the repository root:
    python pipeline\\scan_candidates.py
Writes demo_footage\\candidates\\: one contact-sheet JPG per candidate window
(3 frames with detections drawn), sheet_<video>.jpg overviews, and candidates.json.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from paths import DETECTOR_WEIGHTS, ROOT, SOURCE_VIDEOS

OUT = ROOT / "demo_footage" / "candidates"
WEIGHTS = DETECTOR_WEIGHTS
SOURCES = [SOURCE_VIDEOS, ROOT / "demo_footage"]
SKIP = ("processed", "_logs", "candidates", "web", "sources")
SAMPLE_FPS = 2
WINDOW_S = 3
TOP_PER_VIDEO = 4
CONF = 0.3


def videos():
    seen = set()
    for base in SOURCES:
        for p in sorted(base.rglob("*")):
            if p.suffix.lower() not in (".mp4", ".avi", ".mov", ".mkv") or any(s in str(p) for s in SKIP):
                continue
            if p.stat().st_size in seen:  # skip byte-identical duplicates
                continue
            seen.add(p.stat().st_size)
            yield p


def scan(model, path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(fps / SAMPLE_FPS))
    samples = []
    for idx in range(n):
        if idx % step:
            if not cap.grab():  # decode-skip is far faster than seeking long H.264 files
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        r = model(frame, conf=CONF, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (w * h)) if len(boxes) else np.zeros(0)
        samples.append({"t": idx / fps, "n": int(len(boxes)), "big": float(areas.max()) if len(areas) else 0.0,
                        "conf": float(confs.mean()) if len(confs) else 0.0,
                        "gray": float(cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY).mean())})
    cap.release()
    return {"w": w, "h": h, "fps": fps, "dur": n / fps, "samples": samples}


def score_windows(info):
    s = info["samples"]
    per = WINDOW_S * SAMPLE_FPS
    wins = []
    for i in range(0, max(1, len(s) - per + 1)):
        win = s[i:i + per]
        n = np.median([x["n"] for x in win])
        big = max(x["big"] for x in win)
        conf = np.mean([x["conf"] for x in win])
        # 4-12 vehicles reads well on screen; one large car is needed for the pull-out crop.
        busy = min(n, 12) / 12 - max(0, n - 16) / 20
        score = 0.45 * busy + 0.35 * min(big / 0.04, 1) + 0.2 * conf
        wins.append({"start": win[0]["t"], "score": round(float(score), 3), "n": float(n),
                     "big": round(big, 4), "conf": round(float(conf), 3),
                     "brightness": round(float(np.mean([x["gray"] for x in win])), 1)})
    picked = []
    for wnd in sorted(wins, key=lambda x: -x["score"]):
        if all(abs(wnd["start"] - p["start"]) >= WINDOW_S * 2 for p in picked):
            picked.append(wnd)
        if len(picked) == TOP_PER_VIDEO:
            break
    return picked


def contact(model, path, wnd, dst):
    cap = cv2.VideoCapture(str(path))
    tiles = []
    for k in (0.1, 0.5, 0.9):
        cap.set(cv2.CAP_PROP_POS_MSEC, (wnd["start"] + k * WINDOW_S) * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        r = model(frame, conf=CONF, verbose=False)[0]
        tiles.append(cv2.resize(r.plot(line_width=2, font_size=0.6), (640, int(640 * frame.shape[0] / frame.shape[1]))))
    cap.release()
    if tiles:
        cv2.imwrite(str(dst), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 82])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    report = []
    for path in videos():
        info = scan(model, path)
        picks = score_windows(info)
        for j, wnd in enumerate(picks):
            cid = f"{path.stem[:40]}__{j}"
            wnd.update({"id": cid, "video": str(path), "res": f"{info['w']}x{info['h']}", "fps": round(info["fps"], 2)})
            contact(model, path, wnd, OUT / f"{cid}.jpg")
            report.append(wnd)
        print(f"{path.name}: {info['dur']:.0f}s, best score {picks[0]['score'] if picks else '-'}")
    (OUT / "candidates.json").write_text(json.dumps(report, indent=1))
    print(f"{len(report)} candidates -> {OUT}")


if __name__ == "__main__":
    main()

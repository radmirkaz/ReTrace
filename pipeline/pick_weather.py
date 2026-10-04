"""Find the best night/snow windows in AAU RainSnow: small, distant vehicles, no huge boxes.

On these pole cameras, very large detections are almost always lens drops or headlight
glare, so windows are scored for several small-to-medium boxes and penalised for big ones.
Writes demo_footage/survey/weather_<sequence>_<rank>.jpg contact sheets and weather_picks.json.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from make_web_clips import WEIGHTS

ROOT = Path(__file__).resolve().parents[1]
AAU = ROOT / "demo_footage" / "AAU_RainSnow" / "aaurainsnow"
OUT = ROOT / "demo_footage" / "survey"
SEQS = {"night": ["Hobrovej/Hobrovej-1", "Hjorringvej/Hjorringvej-3", "Hjorringvej/Hjorringvej-4", "Ringvej/Ringvej-1", "Ringvej/Ringvej-2", "Ostre/Ostre-4"],
        "snow": ["Egensevej/Egensevej-1", "Egensevej/Egensevej-3", "Egensevej/Egensevej-4", "Egensevej/Egensevej-5"]}
WIN = 6  # samples per window (every 2 s -> 12 s window)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    picks = []
    for cond, seqs in SEQS.items():
        for seq in seqs:
            path = AAU / seq / "cam1.mkv"
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 20
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
            rows = []
            for t in np.arange(0, total, 2.0):
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, frame = cap.read()
                if not ok:
                    break
                r = model(frame, conf=0.3, verbose=False)[0]
                area = ((r.boxes.xywh[:, 2] * r.boxes.xywh[:, 3]) / (frame.shape[0] * frame.shape[1])).cpu().numpy() if len(r.boxes) else np.zeros(0)
                small = int(((area > 0.0008) & (area < 0.03)).sum())
                big = int((area >= 0.05).sum())
                rows.append((t, min(small, 8) - 3 * big))
            cap.release()
            if len(rows) <= WIN:
                continue
            scores = np.convolve([s for _, s in rows], np.ones(WIN) / WIN, mode="valid")
            for rank, i in enumerate(np.argsort(-scores)[:2]):
                start = rows[i][0]
                cap = cv2.VideoCapture(str(path))
                tiles = []
                for k in (0.2, 0.5, 0.8):
                    cap.set(cv2.CAP_PROP_POS_MSEC, (start + k * WIN * 2) * 1000)
                    ok, frame = cap.read()
                    if ok:
                        tiles.append(cv2.resize(model(frame, conf=0.3, verbose=False)[0].plot(line_width=2, font_size=0.5), (480, 360)))
                cap.release()
                name = f"weather_{seq.split('/')[1]}_{rank}"
                cv2.putText(tiles[0], f"{name} score {scores[i]:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imwrite(str(OUT / f"{name}.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 80])
                picks.append({"name": name, "condition": cond, "seq": seq, "start": float(start), "score": round(float(scores[i]), 2)})
            print(seq, "best", round(float(scores.max()), 2), flush=True)
    (OUT / "weather_picks.json").write_text(json.dumps(picks, indent=1))


if __name__ == "__main__":
    main()

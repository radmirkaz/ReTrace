"""Quick survey: AAU RainSnow frame sheet (to label conditions) and CityFlowV2 multi-camera vehicles."""
import collections
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "demo_footage"
OUT = ROOT / "survey"
OUT.mkdir(exist_ok=True)

tiles = []
for mkv in sorted((ROOT / "AAU_RainSnow" / "aaurainsnow").rglob("cam1.mkv")):
    cap = cv2.VideoCapture(str(mkv))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.5))
    ok, f = cap.read()
    cap.release()
    if not ok:
        continue
    f = cv2.resize(f, (320, 240))
    cv2.putText(f, mkv.parent.name, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    tiles.append(f)
while len(tiles) % 6:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
cv2.imwrite(str(OUT / "aau_sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])

for scen in ["train/S01", "train/S03", "train/S04", "validation/S02", "validation/S05"]:
    base = ROOT / "CityFlowV2" / scen
    cams = collections.defaultdict(set)
    for gt in sorted(base.glob("c*/gt/gt.txt")):
        for line in gt.read_text().splitlines():
            parts = line.split(",")
            cams[int(parts[1])].add(gt.parent.parent.name)
    multi = sorted(((len(c), vid) for vid, c in cams.items()), reverse=True)
    print(scen, "cams:", len(list(base.glob("c*"))), "ids:", len(cams), "top multi-cam ids:", multi[:6])
print("sheet:", OUT / "aau_sheet.jpg")

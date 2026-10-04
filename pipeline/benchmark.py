"""Measure throughput of the ReTrace pipeline on this GPU (for the cost & speed section).

    python pipeline/benchmark.py   # -> web/public/cache/benchmark.json
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO

from make_web_clips import WEIGHTS
from paths import CLASSIFIER_WEIGHTS

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "demo_footage" / "live"
import src.models.classifier  # noqa: E402


def frames(path, n):
    cap = cv2.VideoCapture(str(path))
    out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


def main():
    dev = "cuda"
    res = {"gpu": torch.cuda.get_device_name(0)}
    model = YOLO(str(WEIGHTS))
    for name, clip in (("1080p", "kingsway-knight_hi.mp4"), ("4K-source downscaled 1080p", "cambie-king-edward-e_hi.mp4")):
        fr = frames(LIVE / clip, 300)
        for f in fr[:10]:
            model(f, conf=0.25, verbose=False)
        tracker = sv.ByteTrack()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for f in fr:
            tracker.update_with_detections(sv.Detections.from_ultralytics(model(f, conf=0.25, verbose=False)[0]))
        torch.cuda.synchronize()
        res[f"detect_track_fps_{name}"] = round(len(fr) / (time.perf_counter() - t), 1)

    reid = src.models.classifier.EffNetv2(class_num=9630, features_dim=2048, model_name="m", mix_prec=True)
    ck = torch.load(CLASSIFIER_WEIGHTS, map_location=dev)
    reid.load_state_dict(ck.get("model_state_dict", ck))
    reid.to(dev).eval()
    x = torch.randn(32, 3, 300, 300, device=dev)
    with torch.no_grad():
        for _ in range(3):
            reid(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            reid(x)
        torch.cuda.synchronize()
    res["reid_crops_per_s"] = round(320 / (time.perf_counter() - t), 1)

    emb = np.load(ROOT / "demo_footage" / "reid_embeddings.npy")
    big = np.tile(emb, (max(1, 1_000_000 // len(emb)), 1)).astype(np.float32)
    q = torch.from_numpy(big[:1]).to(dev)
    g = torch.from_numpy(big).to(dev).half()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        torch.topk((g @ q.half().T).squeeze(1), 10)
    torch.cuda.synchronize()
    res["match_ms_vs_1M_vehicles"] = round((time.perf_counter() - t) / 20 * 1000, 2)
    res["index_size_for_1M_vehicles_GB"] = round(big.shape[0] * 2048 * 2 / 1e9, 2)
    (ROOT / "web" / "public" / "cache" / "benchmark.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

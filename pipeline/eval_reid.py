"""Compare re-identification checkpoints on real cross-camera ground truth.

Uses the Kingsway feeds (CityFlowV2 S01, exported by export_live.py): every tracked vehicle
is matched to its CityFlow ground-truth ID, cropped with the same quality selection as the
website, embedded by each checkpoint, then queried against the other cameras.

Run on a CUDA machine from the repository root:
    python pipeline/eval_reid.py
Prints rank-1 / rank-5 / mAP per checkpoint and writes demo_footage/eval_reid.json.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from crops import select_crops
from paths import CLASSIFIER_WEIGHTS, EMBEDDINGS_WEIGHTS

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "demo_footage" / "live"
S01 = ROOT / "demo_footage" / "CityFlowV2" / "train" / "S01"
CHECKPOINTS = {
    "classifier (9,630 classes)": (CLASSIFIER_WEIGHTS, 9630),
    "embeddings model (5,445 classes)": (EMBEDDINGS_WEIGHTS, 5445),
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import src.models.classifier  # noqa: E402


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def load(path, n_cls):
    m = src.models.classifier.EffNetv2(class_num=n_cls, features_dim=2048, model_name="m", mix_prec=True)
    sd = torch.load(path, map_location=DEVICE)
    m.load_state_dict(sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd)
    return m.to(DEVICE).eval()


def collect():
    """[(camera, gt_id, [crops])] for every clean track that matches a ground-truth vehicle."""
    spec = json.loads((LIVE / "live_spec.json").read_text())
    items = []
    for cam in spec["cameras"]:
        if not cam["id"].startswith("kingsway-"):
            continue
        src_cam = cam["source"].split("S01/")[1].split(" ")[0]
        gt = {}
        for line in (S01 / src_cam / "gt" / "gt.txt").read_text().splitlines():
            f, vid, x, y, w, h = map(float, line.split(",")[:6])
            gt.setdefault(int(f) - 1, []).append((int(vid), (x, y, x + w, y + h)))  # clip starts at source frame 1
        hi = json.loads((LIVE / f"{cam['id']}_hi.json").read_text())
        for tid, pick in select_crops(str(LIVE / f"{cam['id']}_hi.mp4"), hi, k=4).items():
            if not pick["clean"]:
                continue
            fi = pick["frames"][0]
            cands = [(iou(pick["box"], b), vid) for vid, b in gt.get(fi, [])]
            if cands and max(cands)[0] > 0.5:
                items.append((cam["id"], max(cands)[1], pick["crops"]))
    return items


def evaluate(emb, cams, ids):
    S = emb @ emb.T
    r1, r5, aps, n = 0, 0, [], 0
    for i in range(len(ids)):
        mask = cams != cams[i]
        pos = (ids == ids[i]) & mask
        if not pos.any():
            continue
        order = np.argsort(-S[i][mask])
        hits = pos[mask][order]
        n += 1
        r1 += hits[0]
        r5 += hits[:5].any()
        ranks = np.where(hits)[0] + 1
        aps.append(np.mean([(k + 1) / r for k, r in enumerate(ranks)]))
    return {"queries": n, "rank1": round(r1 / n, 4), "rank5": round(r5 / n, 4), "mAP": round(float(np.mean(aps)), 4)}


def main():
    items = collect()
    cams = np.array([c for c, _, _ in items])
    ids = np.array([v for _, v, _ in items])
    print(f"{len(items)} matched vehicles, {len(set(ids))} identities, {len(set(cams))} cameras")
    tf = T.Compose([T.Resize((300, 300)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    results, embs = {}, {}
    for name, (path, n_cls) in CHECKPOINTS.items():
        model = load(path, n_cls)
        rows = []
        with torch.no_grad():
            for _, _, crops in items:
                x = torch.stack([tf(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops]).to(DEVICE)
                _, feats = model(x)
                rows.append(F.normalize(F.normalize(feats.float(), dim=1).mean(0, keepdim=True), dim=1)[0].cpu().numpy())
        embs[name] = np.stack(rows)
        results[name] = evaluate(embs[name], cams, ids)
        print(name, results[name])
        del model
        torch.cuda.empty_cache()
    ens = sum(embs.values())
    ens /= np.linalg.norm(ens, axis=1, keepdims=True)
    results["ensemble (mean of both)"] = evaluate(ens, cams, ids)
    print("ensemble", results["ensemble (mean of both)"])
    (ROOT / "demo_footage" / "eval_reid.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

"""Search by photo or video: find the vehicle in an upload and rank every indexed vehicle by re-ID similarity.

Uses the same models and steps as indexing (pipeline/build_cache.py): detector, best crops,
embeddings model averaged over crops, and the body/colour gate used for journey matching.
"""
import base64
import json
import math
import sys
import tempfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
import build_cache as B  # noqa: E402  (model loading and the body/colour vocabulary shared with indexing)
from crops import EDGE_MARGIN, pad_box, sharpness  # noqa: E402
from paths import DETECTOR_WEIGHTS  # noqa: E402

INDEX = ROOT / "demo_footage"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}
MAX_FRAMES = 120  # frames sampled from an uploaded clip
MAX_SIDE = 1920  # larger frames are downscaled before detection
CONF = 0.25
BODY_MIN, COLOUR_MIN = 0.15, 0.10  # same gate as journey matching (web/public/cache/metrics.json)
MIN_SIM, MIN_RESULTS = 0.25, 4  # weaker matches are noise; always show the best few


@lru_cache
def models():
    from ultralytics import YOLO
    M = B.Models()

    def txt(prompts):
        t = M.clip.get_text_features(**M.proc(text=prompts, return_tensors="pt", padding=True).to(B.DEVICE))
        return F.normalize(t.float(), dim=1)

    with torch.no_grad():
        body_t = txt([f"a photo of a {b}" for b in B.BODIES])
        colour_t = txt([f"a photo of a {c} vehicle" for c in B.COLOURS])
    allowed = torch.from_numpy(M.allowed).to(B.DEVICE)
    return M, YOLO(str(DETECTOR_WEIGHTS)), body_t, colour_t, allowed


@lru_cache
def index():
    E = np.load(INDEX / "reid_embeddings.npy")
    keys = json.loads((INDEX / "embedding_keys.json").read_text())
    attrs = json.loads((INDEX / "attributes.json").read_text())
    return E, keys, attrs


def _fit(frame):
    h, w = frame.shape[:2]
    s = MAX_SIDE / max(h, w)
    return cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame


def _crop(frame, box):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = pad_box(box, W, H)
    return frame[y1:y2, x1:x2].copy()


def _quality(frame, box, conf):
    """Same idea as pipeline/crops.py: big, confident, sharp and not cut off by the frame edge."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    edge = x1 < EDGE_MARGIN or y1 < EDGE_MARGIN or x2 > W - EDGE_MARGIN or y2 > H - EDGE_MARGIN
    return math.sqrt(max(1.0, (x2 - x1) * (y2 - y1))) * float(conf) * (0.4 if edge else 1.0)


def crops_from_image(img):
    """The most prominent vehicle (size x confidence, favouring the centre); the whole image if none is found."""
    _, det, *_ = models()
    img = _fit(img)
    H, W = img.shape[:2]
    r = det(img, conf=CONF, verbose=False)[0]
    boxes, confs = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
    if not len(boxes):
        return [img], False

    def score(i):
        x1, y1, x2, y2 = boxes[i]
        off = math.hypot((x1 + x2) / 2 / W - 0.5, (y1 + y2) / 2 / H - 0.5)
        return (x2 - x1) * (y2 - y1) * confs[i] * (1 - off)

    return [_crop(img, boxes[max(range(len(boxes)), key=score)])], True


def crops_from_video(path, k=4):
    """Track every vehicle; keep the best k crops of the one seen biggest and longest."""
    import supervision as sv
    _, det, *_ = models()
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or MAX_FRAMES
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    stride = max(1, math.ceil(n / MAX_FRAMES))
    tracker = sv.ByteTrack(frame_rate=max(1, round(fps / stride)))
    seen = defaultdict(list)  # track id -> [(quality, frame index, crop)]
    used = i = 0
    while used < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if (i - 1) % stride:
            continue
        used += 1
        frame = _fit(frame)
        d = tracker.update_with_detections(sv.Detections.from_ultralytics(det(frame, conf=CONF, verbose=False)[0]))
        for box, tid, conf in zip(d.xyxy, d.tracker_id, d.confidence):
            seen[int(tid)].append((_quality(frame, box, conf), used, _crop(frame, box)))
    cap.release()
    if not seen:
        return [], used
    main = max(seen, key=lambda t: sum(q for q, *_ in seen[t]))
    picks, cands = [], sorted(seen[main], key=lambda c: c[0], reverse=True)[:12]
    cands.sort(key=lambda c: c[0] * min(1.0, sharpness(c[2]) / 150) ** 0.5, reverse=True)
    for q, fi, crop in cands:  # spread the chosen crops over time
        if all(abs(fi - p[1]) >= 3 for p in picks):
            picks.append((q, fi, crop))
        if len(picks) == k:
            break
    return [c for *_, c in picks], used


def describe(crops):
    """Re-ID embedding, make/model probabilities and CLIP body/colour for a set of crops of one vehicle."""
    M, _, body_t, colour_t, allowed = models()
    rgbs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
    with torch.no_grad():
        batch = torch.stack([M.tf(r) for r in rgbs]).to(B.DEVICE)
        _, emb = M.embedder(batch)
        q = F.normalize(F.normalize(emb.float(), dim=1).mean(0, keepdim=True), dim=1)[0].cpu().numpy()
        logits, _ = M.reid(batch)
        p = F.softmax(logits.float(), dim=1) * M.priors * allowed
        p = (p / p.sum(dim=1, keepdim=True)).mean(0)
        ci = F.normalize(M.clip.get_image_features(**M.proc(images=rgbs, return_tensors="pt").to(B.DEVICE)).float(), dim=1)
        ci = F.normalize(ci.mean(0, keepdim=True), dim=1)
        body_p = F.softmax((ci @ body_t.T)[0] * 100, dim=0).cpu().numpy()
        colour_p = F.softmax((ci @ colour_t.T)[0] * 100, dim=0).cpu().numpy()
    return q, p.cpu().numpy(), body_p, colour_p


def _gate(attr, body_f, colour_f):
    groups = sorted(set(B.BODY_GROUP.values()))
    fams = sorted({f for fs in B.COLOUR_FAMILY.values() for f in fs})
    if sum(body_f[i] * attr["body"].get(g, 0) for i, g in enumerate(groups)) < BODY_MIN:
        return False
    if attr["night"]:
        return True  # colour is unreliable under streetlights
    return sum(colour_f[i] * attr["colour"].get(f, 0) for i, f in enumerate(fams)) >= COLOUR_MIN


def _thumb(crop):
    h, w = crop.shape[:2]
    s = 360 / max(h, w)
    small = cv2.resize(crop, (max(1, round(w * s)), max(1, round(h * s)))) if s < 1 else crop
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 88])[1]).decode()


def search_upload(data: bytes, filename: str, content_type: str, k: int = 12):
    is_video = content_type.startswith("video/") or Path(filename).suffix.lower() in VIDEO_EXT
    if is_video:
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix or ".mp4", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            crops, frames = crops_from_video(path)
        finally:
            Path(path).unlink(missing_ok=True)
        if not crops:
            raise ValueError("No vehicle found in the clip.")
        found = True
    else:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Unsupported image. Use JPG, PNG or WebP.")
        crops, found = crops_from_image(img)
        frames = 1

    q, p, body_p, colour_p = describe(crops)
    M = models()[0]
    top = int(p.argmax())
    E, keys, attrs = index()
    sims = E @ q
    body_f = B._family_probs(body_p, B.BODIES, B.BODY_GROUP)
    colour_f = B._family_probs(colour_p, B.COLOURS, B.COLOUR_FAMILY)
    fits = [_gate(attrs[g], body_f, colour_f) if g in attrs else True for g in keys]
    order = sorted(range(len(keys)), key=lambda i: (not fits[i], -sims[i]))[:k]
    order = [i for n, i in enumerate(order) if n < MIN_RESULTS or sims[i] >= MIN_SIM]
    vehicles = json.loads((B.CACHE / "vehicles.json").read_text())
    results = []
    for i in order:
        gid = keys[i]
        cam, track = gid.split(":")
        results.append({"gid": gid, "cam": cam, "track": int(track), "crop": vehicles[gid]["crop"],
                        "score": round(float(sims[i]), 4), "match": bool(fits[i])})
    return {
        "query": {
            "crop": _thumb(crops[0]), "source": "video" if is_video else "photo", "found": found, "frames": frames, "crops": len(crops),
            "body": B.BODIES[int(body_p.argmax())], "colour": B.COLOURS[int(colour_p.argmax())],
            "guess": {"name": M.names[top], "p": round(float(p[top]), 3)},
        },
        "results": results,
    }

"""Choose the crops of each tracked vehicle that are actually usable for re-identification.

The largest box of a track is often the worst one: the car is entering or leaving the
frame (truncated), overlapped by a neighbour, or motion-blurred. Every detection is
scored instead, and the best few (spread over time) are kept so their embeddings
can be averaged.
"""
import math
from collections import defaultdict

import cv2
import numpy as np

EDGE_MARGIN = 6  # px; a box this close to the border is treated as cut off
MIN_SIDE = 40  # px; smaller crops carry too little detail to identify
SIZE_REF = 160  # px; sqrt(area) at which size stops adding to the score
MIN_GAP_S = 0.3  # keep chosen crops at least this far apart in time
SHARP_CANDIDATES = 10  # how many top geometric candidates get a sharpness check


def _overlap_ratio(box, others):
    """Largest share of `box` covered by any other box (intersection over this box's area)."""
    x1, y1, x2, y2 = box
    area = max(1e-6, (x2 - x1) * (y2 - y1))
    best = 0.0
    for ox1, oy1, ox2, oy2 in others:
        iw = min(x2, ox2) - max(x1, ox1)
        ih = min(y2, oy2) - max(y1, oy1)
        if iw > 0 and ih > 0:
            best = max(best, iw * ih / area)
    return best


def geometric_candidates(tracks: dict) -> dict:
    """track id -> list of candidate dicts with a geometry-only quality score."""
    W, H = tracks["w"], tracks["h"]
    by_track = defaultdict(list)
    for fi, frame in enumerate(tracks["frames"]):
        boxes = {d[0]: d[1:5] for d in frame}
        for tid, x1, y1, x2, y2, conf, _cls in frame:
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            truncated = x1 <= EDGE_MARGIN or y1 <= EDGE_MARGIN or x2 >= W - EDGE_MARGIN or y2 >= H - EDGE_MARGIN
            occl = _overlap_ratio((x1, y1, x2, y2), [b for t, b in boxes.items() if t != tid])
            by_track[tid].append({"frame": fi, "box": (x1, y1, x2, y2), "conf": conf, "w": w, "h": h,
                                  "truncated": truncated, "occlusion": occl})
    for tid, cands in by_track.items():
        med_ar = float(np.median([c["w"] / c["h"] for c in cands]))
        for c in cands:
            size = min(1.0, math.sqrt(c["w"] * c["h"]) / SIZE_REF)
            aspect = math.exp(-2.0 * abs(math.log((c["w"] / c["h"]) / med_ar)))
            occl = max(0.0, 1.0 - 1.6 * c["occlusion"])
            edge = 0.1 if c["truncated"] else 1.0
            small = 0.2 if min(c["w"], c["h"]) < MIN_SIDE else 1.0
            c["geo"] = c["conf"] * size * aspect * occl * edge * small
    return by_track


def sharpness(crop: np.ndarray) -> float:
    gray = cv2.cvtColor(cv2.resize(crop, (128, 96)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pad_box(box, W, H, pad=0.06):
    x1, y1, x2, y2 = box
    pw, ph = pad * (x2 - x1), pad * (y2 - y1)
    return int(max(0, x1 - pw)), int(max(0, y1 - ph)), int(min(W, x2 + pw)), int(min(H, y2 + ph))


def select_crops(video_path: str, tracks: dict, k: int = 4) -> dict:
    """track id -> {"crops": [BGR arrays, best first], "frames": [...], "quality": float, "clean": bool}.

    `clean` is False when every candidate is cut off, occluded or tiny; such tracks are
    shown as detections only and are not identified.
    """
    W, H, fps = tracks["w"], tracks["h"], tracks["fps"]
    cands = geometric_candidates(tracks)
    wanted = defaultdict(list)  # frame -> [(tid, cand)]
    shortlist = {}
    for tid, cs in cands.items():
        top = sorted(cs, key=lambda c: -c["geo"])[:SHARP_CANDIDATES]
        shortlist[tid] = top
        for c in top:
            wanted[c["frame"]].append((tid, c))

    cap = cv2.VideoCapture(video_path)
    fi = 0
    while wanted:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, c in wanted.pop(fi, []):
            x1, y1, x2, y2 = pad_box(c["box"], W, H)
            c["crop"] = frame[y1:y2, x1:x2].copy()
            c["sharp"] = sharpness(c["crop"]) if c["crop"].size else 0.0
        fi += 1
    cap.release()

    out = {}
    for tid, top in shortlist.items():
        top = [c for c in top if c.get("crop") is not None and c["crop"].size]
        if not top:
            continue
        max_sharp = max(c["sharp"] for c in top) or 1.0
        for c in top:
            c["score"] = c["geo"] * (0.5 + 0.5 * min(1.0, c["sharp"] / max_sharp))
        chosen = []
        for c in sorted(top, key=lambda c: -c["score"]):
            if all(abs(c["frame"] - o["frame"]) / fps >= MIN_GAP_S for o in chosen):
                chosen.append(c)
            if len(chosen) == k:
                break
        best = chosen[0]
        clean = (not best["truncated"]) and best["occlusion"] < 0.35 and min(best["w"], best["h"]) >= MIN_SIDE
        out[tid] = {"crops": [c["crop"] for c in chosen], "frames": [c["frame"] for c in chosen],
                    "box": best["box"], "quality": round(best["score"], 3), "clean": clean,
                    "truncated": best["truncated"], "occlusion": round(best["occlusion"], 2)}
    return out

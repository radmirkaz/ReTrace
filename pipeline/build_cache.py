"""Build the website's demo cache from exported clips.

For every camera clip in demo_footage/web this:
  * crops each tracked vehicle at its largest appearance,
  * runs the re-identification model for make/model/generation (top-5) and a
    2048-D appearance embedding,
  * links vehicles across cameras by embedding similarity,
  * ranks crops with CLIP for the landing page's witness query.

Run on the GPU PC from the repository root (canai env):
    python pipeline/build_cache.py
Writes web/public/cache/{cameras,vehicles,landing}.json plus clips/, tracks/, crops/.
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CANAI = Path(r"C:\Users\Radmir\Desktop\canai25")  # baseline project: re-ID weights and class metadata, read-only
CLIPS = ROOT / "demo_footage" / "web"
CACHE = ROOT / "web" / "public" / "cache"
REID_WEIGHTS = CANAI / "submission" / "reid" / "checkpoints" / "ep6_v8.pt"
CLASS_META = CANAI / "submission" / "notebooks" / "llm_data_9630_classes.json"
CLIP_MODEL = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.append(str(CANAI / "submission" / "reid"))
import src.models.classifier  # noqa: E402  (baseline model definition)

# Demo feeds placed at real Vancouver locations. `source` is always shown in the UI.
CAMERAS = [
    dict(id="hwy1-boundary", name="Hwy 1 at Boundary Rd", lat=49.2590, lon=-123.0235, clip="overpass", view="overpass", condition="day", source="Highway overpass clip (EU)"),
    dict(id="main-kingsway", name="Main St & Kingsway", lat=49.2590, lon=-123.1003, clip="pole", view="pole", condition="day", source="Urban Tracker · Sherbrooke, QC"),
    dict(id="knight-bridge", name="Knight St Bridge", lat=49.2050, lon=-123.0770, clip="dusk", view="overpass", condition="dusk", source="Motorway at dusk (UK)"),
    dict(id="oak-bridge", name="Oak St Bridge", lat=49.2040, lon=-123.1270, clip="longrange", view="overpass", condition="day", source="Motorway, 720p (UK)"),
]

LANDING_SCENES = [
    ("overpass", "Overpass", "Looking down the lanes from ~8 m."),
    ("pole", "Pole camera", "An old city CCTV corner cam — 800×600."),
    ("dusk", "Dusk, dense traffic", "Low sun, glare and heavy occlusion."),
    ("longrange", "Long range, low-res", "Small, distant vehicles at 720p."),
]
TRACE_CLIP = "dusk"
WITNESS = {
    "transcript": "It was a white sedan, maybe a BMW. It hit the cyclist and kept going east on Kingsway, around nine-thirty.",
    "query": "a white sedan car",
}


def pretty_class(path: str) -> str:
    parts = [p for p in path.strip("/").split("/") if p]
    brand = parts[0].replace("_", " ").replace("mercedesbenz", "mercedes-benz").title() if parts else "?"
    model = parts[1].replace("_", " ").title() if len(parts) > 1 else ""
    gen = parts[2].replace("_", " ") if len(parts) > 2 and parts[2] not in ("1",) else ""
    return " ".join(x for x in (brand, model, f"({gen})" if gen else "") if x)


def prior_weights(meta, camera_region="USA"):
    """Same regional prior idea as the baseline: down-weight models not sold in North America."""
    w = np.ones(len(meta), dtype=np.float32)
    for item in meta:
        prior = 1.0
        if str(item.get("sold_in_US_or_Canada", "no")).lower() == "no":
            prior *= 0.2
        region = item.get("region_of_origin", "Other")
        if region in {"France", "Italy", "Sweden", "Czech Republic", "Spain"}:
            prior *= 0.7
        elif region not in {camera_region, "Japan", "Germany", "Korea", "South Korea", "UK"}:
            prior *= 0.6
        if item.get("main_market") == "domestic" and region != camera_region:
            prior *= 0.7
        w[int(item["class_id"])] = max(prior, 0.01)
    return w


def colour_name(crop_bgr: np.ndarray) -> str:
    h, w = crop_bgr.shape[:2]
    core = crop_bgr[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hue, sat, val = np.median(hsv[:, 0]), np.median(hsv[:, 1]), np.median(hsv[:, 2])
    if val < 60:
        return "black"
    if sat < 45:
        return "white" if val > 165 else "grey"
    for limit, name in ((8, "red"), (22, "orange"), (34, "yellow"), (85, "green"), (130, "blue"), (160, "purple")):
        if hue < limit:
            return name
    return "red"


def best_frames(tracks: dict) -> dict:
    """track id -> (frame index, box) where the vehicle is largest on screen."""
    best = {}
    for i, frame in enumerate(tracks["frames"]):
        for tid, x1, y1, x2, y2, _conf, _cls in frame:
            area = (x2 - x1) * (y2 - y1)
            if tid not in best or area > best[tid][2]:
                best[tid] = (i, (x1, y1, x2, y2), area)
    return best


def main():
    for sub in ("clips", "tracks", "crops"):
        (CACHE / sub).mkdir(parents=True, exist_ok=True)
    meta = json.loads(CLASS_META.read_text())
    names = {int(m["class_id"]): pretty_class(m["class"]) for m in meta}
    priors = torch.from_numpy(prior_weights(meta)).to(DEVICE)

    reid = src.models.classifier.EffNetv2(class_num=len(meta), features_dim=2048, model_name="m", mix_prec=True)
    ckpt = torch.load(REID_WEIGHTS, map_location=DEVICE)
    reid.load_state_dict(ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt)
    reid.to(DEVICE).eval()
    tf = T.Compose([T.Resize((300, 300)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE).eval()
    clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL)

    cameras, vehicles, emb_rows, clip_rows, keys = [], {}, [], [], []
    for cam in CAMERAS:
        mp4, tj = CLIPS / f"{cam['clip']}.mp4", CLIPS / f"{cam['clip']}.json"
        if not mp4.exists():
            print("skip (no clip):", cam["id"])
            continue
        tracks = json.loads(tj.read_text())
        shutil.copy(mp4, CACHE / "clips" / f"{cam['id']}.mp4")
        shutil.copy(tj, CACHE / "tracks" / f"{cam['id']}.json")
        shutil.copy(CLIPS / f"{cam['clip']}.jpg", CACHE / "clips" / f"{cam['id']}.jpg")
        cameras.append({**{k: v for k, v in cam.items() if k != "clip"},
                        "res": f"{tracks['w']}x{tracks['h']}", "clip": f"clips/{cam['id']}.mp4",
                        "tracks": f"tracks/{cam['id']}.json", "poster": f"clips/{cam['id']}.jpg"})

        cap = cv2.VideoCapture(str(mp4))
        for tid, (fi, (x1, y1, x2, y2), _area) in best_frames(tracks).items():
            if (x2 - x1) < 24 or (y2 - y1) < 18:
                continue  # too small to identify honestly
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            pad = 0.06
            bx1, by1 = max(0, int(x1 - pad * (x2 - x1))), max(0, int(y1 - pad * (y2 - y1)))
            bx2, by2 = min(frame.shape[1], int(x2 + pad * (x2 - x1))), min(frame.shape[0], int(y2 + pad * (y2 - y1)))
            crop = frame[by1:by2, bx1:bx2]
            gid = f"{cam['id']}:{tid}"
            crop_rel = f"crops/{cam['id']}_{tid}.jpg"
            cv2.imwrite(str(CACHE / crop_rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            rgb = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                logits, feats = reid(tf(rgb).unsqueeze(0).to(DEVICE))
                probs = F.softmax(logits.float(), dim=1)[0] * priors
                probs = probs / probs.sum()
                top = torch.topk(probs, 5)
                ci = clip_model.get_image_features(**clip_proc(images=rgb, return_tensors="pt").to(DEVICE))
            emb_rows.append(F.normalize(feats.float(), dim=1)[0].cpu().numpy())
            clip_rows.append(F.normalize(ci.float(), dim=1)[0].cpu().numpy())
            keys.append(gid)
            vehicles[gid] = {
                "gid": gid, "crop": crop_rel, "color": colour_name(crop),
                "top5": [{"name": names[int(i)], "p": round(float(p), 4)} for p, i in zip(top.values, top.indices)],
                "sightings": [{"cam": cam["id"], "track": int(tid), "t": round(fi / tracks["fps"], 2), "crop": crop_rel, "sim": 1.0}],
            }
        cap.release()
        print(cam["id"], "vehicles so far:", len(vehicles))

    # Cross-camera links: nearest appearance embeddings on *other* cameras.
    E = np.stack(emb_rows)
    S = E @ E.T
    for i, gid in enumerate(keys):
        cam_i = gid.split(":")[0]
        for j in np.argsort(-S[i]):
            if j == i or keys[j].split(":")[0] == cam_i:
                continue
            v = vehicles[keys[j]]
            vehicles[gid]["sightings"].append({**v["sightings"][0], "sim": round(float(S[i, j]), 3)})
            if len(vehicles[gid]["sightings"]) >= 4:
                break
    np.save(ROOT / "demo_footage" / "reid_embeddings.npy", E)

    # Witness query ranked by CLIP over every crop.
    with torch.no_grad():
        q = clip_model.get_text_features(**clip_proc(text=[WITNESS["query"]], return_tensors="pt", padding=True).to(DEVICE))
    qv = F.normalize(q.float(), dim=1)[0].cpu().numpy()
    C = np.stack(clip_rows)
    scores = C @ qv
    order = np.argsort(-scores)
    hits = [{"gid": keys[i], "cam": keys[i].split(":")[0], "track": int(keys[i].split(":")[1]),
             "crop": vehicles[keys[i]]["crop"], "score": round(float(scores[i]), 4)} for i in order[:8]]
    np.save(ROOT / "demo_footage" / "clip_embeddings.npy", C)
    (ROOT / "demo_footage" / "embedding_keys.json").write_text(json.dumps(keys))

    trace_cam = next(c for c in CAMERAS if c["clip"] == TRACE_CLIP)
    trace_target = hits[0] if hits[0]["cam"] == trace_cam["id"] else max(
        (v for v in vehicles.values() if v["sightings"][0]["cam"] == trace_cam["id"]),
        key=lambda v: v["top5"][0]["p"], default=None)
    t_gid = trace_target["gid"]
    landing = {
        "scenes": [{"cam": c["id"], "title": t, "sub": s, "clip": f"clips/{c['id']}.mp4", "tracks": f"tracks/{c['id']}.json",
                    "poster": f"clips/{c['id']}.jpg", "res": next(x["res"] for x in cameras if x["id"] == c["id"])}
                   for clip, t, s in LANDING_SCENES for c in CAMERAS if c["clip"] == clip and any(x["id"] == c["id"] for x in cameras)],
        "trace": {"cam": trace_cam["id"], "clip": f"clips/{trace_cam['id']}.mp4", "tracks": f"tracks/{trace_cam['id']}.json",
                  "poster": f"clips/{trace_cam['id']}.jpg", "track": int(t_gid.split(":")[1]),
                  "t": vehicles[t_gid]["sightings"][0]["t"], "vehicle": t_gid},
        "witness": {**WITNESS, "hits": hits},
    }
    (CACHE / "cameras.json").write_text(json.dumps(cameras, indent=1))
    (CACHE / "vehicles.json").write_text(json.dumps(vehicles))
    (CACHE / "landing.json").write_text(json.dumps(landing, indent=1))
    print(f"cache: {len(cameras)} cameras, {len(vehicles)} vehicles, witness top hit {hits[0]['gid']} {hits[0]['score']}")


if __name__ == "__main__":
    main()

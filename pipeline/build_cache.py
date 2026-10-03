"""Build the website's data from exported footage.

Two sets are indexed the same way:
  * story scenes (pipeline/export_cameras.py → demo_footage/web) — short scrub clips for the landing page,
  * live feeds (pipeline/export_live.py → demo_footage/live) — 45 s looping feeds for the city map.

For every camera this:
  * picks each vehicle's best crops (pipeline/crops.py: not cut off, not occluded, sharp),
  * runs the re-identification model on up to 4 crops and averages class probabilities
    and the 2048-D appearance embedding,
  * links vehicles across cameras by embedding similarity; the CityFlow corridor target
    is additionally matched to ground truth so the landing page's trace is verified,
  * ranks crops with CLIP for the landing page's witness statement.

Run on the GPU PC from the repository root (canai env):
    python pipeline/build_cache.py
Writes web/public/cache/{cameras,vehicles}.json + clips/, tracks/, crops/ for the map and
web/public/cache/story/ + landing.json for the landing page.
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

from crops import select_crops

ROOT = Path(__file__).resolve().parents[1]
CANAI = Path(r"C:\Users\Radmir\Desktop\canai25")  # baseline project: re-ID weights and class metadata, read-only
WEB = ROOT / "demo_footage" / "web"
LIVE = ROOT / "demo_footage" / "live"
CACHE = ROOT / "web" / "public" / "cache"
REID_WEIGHTS = CANAI / "submission" / "reid" / "checkpoints" / "ep6_v8.pt"  # make/model names (9,630 classes)
REID_EMBED_WEIGHTS = CANAI / "submission" / "reid" / "pretrain" / "v24.pt.sd"  # image-to-image re-ID (5,445-class head unused)
# "v24" = re-ID embedding from v24 only; "ensemble" = mean of v24 and ep6_v8 embeddings
# (pipeline/eval_reid.py on CityFlowV2: v24 mAP 67.4%, ensemble 73.3%).
REID_EMBED = "v24"
CLASS_META = CANAI / "submission" / "notebooks" / "llm_data_9630_classes.json"
CLIP_MODEL = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BODIES = ["sedan", "SUV", "pickup truck", "hatchback", "minivan", "van", "coupe", "station wagon", "bus", "box truck", "semi truck", "motorcycle"]
COLOURS = ["white", "black", "grey", "silver", "red", "blue", "green", "yellow", "orange", "brown", "beige"]


def reliable(consistency, margin, p1, min_side, contrast, n_crops):
    """Make/model is only shown when independent signals agree — softmax confidence alone is
    not trusted (night/snow crops can be confidently wrong)."""
    return n_crops >= 2 and consistency >= 0.75 and p1 >= 0.25 and margin >= 0.08 and min_side >= 72 and contrast >= 28


LOCATIONS = json.loads((Path(__file__).parent / "map" / "camera_locations.json").read_text())

sys.path.append(str(CANAI / "submission" / "reid"))
import src.models.classifier  # noqa: E402  (baseline model definition)

LANDING_SCENES = [
    ("hwy1-boundary", "Overpass", "Looking down the lanes from ~8 m."),
    ("main-broadway", "Pole camera", "An old city CCTV corner cam — 800×600."),
    ("granville-broadway", "Rain", "Wet road, spray and reflections."),
    ("georgia-denman", "Night", "Streetlights, headlights and glare."),
    ("marine-main", "Snow", "Snow on the road at night."),
    ("oak-bridge", "Long range, low-res", "Small, distant vehicles at 720p."),
]
WITNESS = {
    "transcript": "It was a white pickup truck, kind of big. It hit the cyclist and kept going east on Kingsway, around nine-thirty.",
    "query": "a white pickup truck",
}


def pretty_class(path: str) -> str:
    parts = [p for p in path.strip("/").split("/") if p]
    brand = parts[0].replace("_", " ").replace("mercedesbenz", "mercedes-benz").title() if parts else "?"
    model = parts[1].replace("_", " ").title() if len(parts) > 1 else ""
    gen = parts[2].replace("_", " ") if len(parts) > 2 and parts[2] not in ("1",) else ""
    return " ".join(x for x in (brand, model, f"({gen})" if gen else "") if x)


def prior_weights(meta, camera_region="USA"):
    """Regional prior from the baseline: down-weight models not sold in North America."""
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


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


CF_ROOT = ROOT / "demo_footage" / "CityFlowV2"
RHD_GT = ROOT / "demo_footage" / "RoundaboutHD" / "RoundaboutHD" / "Multi_CAM_Ground_Turth.txt"


def _load_cityflow(src_cam: str):
    scen = "train" if (CF_ROOT / "train" / src_cam.split("/")[0]).exists() else "validation"
    frames = {}
    for line in (CF_ROOT / scen / src_cam / "gt" / "gt.txt").read_text().splitlines():
        f, vid, x, y, w, h = map(float, line.split(",")[:6])
        frames.setdefault(int(f), []).append((int(vid), (x, y, x + w, y + h)))
    return frames


_RHD = None


def _load_rhd(cam: int):
    global _RHD
    if _RHD is None:
        _RHD = {}
        for line in RHD_GT.read_text().splitlines():
            c, vid, f, x, y, w, h = line.split()[:7]
            _RHD.setdefault(int(c), {}).setdefault(int(f), []).append((int(vid), (float(x), float(y), float(x) + float(w), float(y) + float(h))))
    return _RHD.get(cam, {})


def assign_ground_truth(spec, vehicles):
    """gid -> (dataset, ground-truth id) by matching each vehicle's best box to the labelled box."""
    out = {}
    for cam in spec["cameras"]:
        cid, src = cam["id"], cam.get("source", "")
        if "CityFlowV2" in src and cid in spec.get("trace", {}):
            src_cam = src.split("· ")[1].split(" ")[0]  # e.g. S01/c003
            frames = _load_cityflow(src_cam)
            t = spec["trace"][cid]
            tx, ty, tw, th = t["box"]
            f_target = next((f for f, rows in frames.items() for vid, b in rows
                             if vid == t["gt_vehicle"] and abs(b[0] - tx) < 1 and abs(b[1] - ty) < 1), None)
            if f_target is None:
                continue
            base, scale, key = f_target - round(t["t"] * 10), 1.0, "cityflow"
        elif cam.get("gt", {}).get("dataset") == "roundabouthd":
            g = cam["gt"]
            frames = _load_rhd(g["cam"])
            base, scale, key = round(cam["start"] * g["fps"]), g["scale"], "roundabouthd"
        else:
            continue
        for gid, v in vehicles.items():
            if not gid.startswith(cid + ":"):
                continue
            best = (0.0, None)
            for off in (-1, 0, 1):
                for vid, b in frames.get(base + v["_fi"] + off, []):
                    o = iou(v["_box"], tuple(x * scale for x in b))
                    if o > best[0]:
                        best = (o, vid)
            if best[0] > 0.5:
                out[gid] = (key, best[1])
    return out


class Models:
    def __init__(self):
        meta = json.loads(CLASS_META.read_text())
        self.names = {int(m["class_id"]): pretty_class(m["class"]) for m in meta}
        self.priors = torch.from_numpy(prior_weights(meta)).to(DEVICE)
        self.reid = src.models.classifier.EffNetv2(class_num=len(meta), features_dim=2048, model_name="m", mix_prec=True)
        ckpt = torch.load(REID_WEIGHTS, map_location=DEVICE)
        self.reid.load_state_dict(ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt)
        self.reid.to(DEVICE).eval()
        self.embedder = src.models.classifier.EffNetv2(class_num=5445, features_dim=2048, model_name="m", mix_prec=True)
        sd = torch.load(REID_EMBED_WEIGHTS, map_location=DEVICE)
        self.embedder.load_state_dict(sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd)
        self.embedder.to(DEVICE).eval()
        self.tf = T.Compose([T.Resize((300, 300)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        from transformers import CLIPModel, CLIPProcessor
        self.clip = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_MODEL)


def index(M, src_dir, spec_name, out):
    """Crop, identify and link every vehicle of one footage set; assets go under `out`."""
    for sub in ("clips", "tracks", "crops"):
        d = out / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    rel = out.relative_to(CACHE).as_posix()
    rel = "" if rel == "." else rel + "/"
    spec = json.loads((src_dir / spec_name).read_text())
    names, priors, reid, tf, clip_model, clip_proc = M.names, M.priors, M.reid, M.tf, M.clip, M.proc
    with torch.no_grad():
        txt = lambda xs: F.normalize(clip_model.get_text_features(**clip_proc(text=xs, return_tensors="pt", padding=True).to(DEVICE)).float(), dim=1)
        body_t = txt([f"a photo of a {b}" for b in BODIES])
        colour_t = txt([f"a photo of a {c} vehicle" for c in COLOURS])
    WEB_ = src_dir
    cameras, vehicles, emb, clip_emb, keys = [], {}, [], [], []
    stats = {"tracks": 0, "identified": 0, "skipped_unclean": 0}
    for cam in spec["cameras"]:
        cid = cam["id"]
        web_tracks = json.loads((WEB_ / f"{cid}.json").read_text())
        hi_tracks = json.loads((WEB_ / f"{cid}_hi.json").read_text())
        shutil.copy(WEB_ / f"{cid}.mp4", out / "clips" / f"{cid}.mp4")
        shutil.copy(WEB_ / f"{cid}.json", out / "tracks" / f"{cid}.json")
        shutil.copy(WEB_ / f"{cid}.jpg", out / "clips" / f"{cid}.jpg")
        if cid in LOCATIONS:  # real intersection from OSM (pipeline/map/locate_cameras.py)
            loc = LOCATIONS[cid]
            cam = {**cam, "lat": loc[0], "lon": loc[1], **({"name": loc[2]} if len(loc) > 2 else {})}
        cameras.append({**cam, "res": f"{hi_tracks['w']}x{hi_tracks['h']}", "clip": f"{rel}clips/{cid}.mp4",
                        "tracks": f"{rel}tracks/{cid}.json", "poster": f"{rel}clips/{cid}.jpg"})

        picks = select_crops(str(WEB_ / f"{cid}_hi.mp4"), hi_tracks, k=4)
        stats["tracks"] += len(picks)
        for tid, pick in picks.items():
            if not pick["clean"]:
                stats["skipped_unclean"] += 1
                continue
            rgbs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in pick["crops"]]
            with torch.no_grad():
                batch = torch.stack([tf(r) for r in rgbs]).to(DEVICE)
                logits, feats = reid(batch)
                per_crop = F.softmax(logits.float(), dim=1) * priors
                per_crop = per_crop / per_crop.sum(dim=1, keepdim=True)
                probs = per_crop.mean(0)
                _, v24_feats = M.embedder(batch)
                feat = F.normalize(F.normalize(v24_feats.float(), dim=1).mean(0, keepdim=True), dim=1)[0]
                if REID_EMBED == "ensemble":
                    own = F.normalize(F.normalize(feats.float(), dim=1).mean(0, keepdim=True), dim=1)[0]
                    feat = F.normalize(feat + own, dim=0)
                ci = F.normalize(clip_model.get_image_features(**clip_proc(images=rgbs, return_tensors="pt").to(DEVICE)).float(), dim=1)
                ci = F.normalize(ci.mean(0, keepdim=True), dim=1)
                body = BODIES[int((ci @ body_t.T).argmax())]
                colour = COLOURS[int((ci @ colour_t.T).argmax())]
            top = torch.topk(probs, 5)
            top1 = int(top.indices[0])
            consistency = float((per_crop.argmax(dim=1) == top1).float().mean())
            margin = float(top.values[0] - top.values[1])
            best = pick["crops"][0]
            min_side = min(best.shape[:2])
            contrast = float(cv2.cvtColor(best, cv2.COLOR_BGR2GRAY).std())
            ok = reliable(consistency, margin, float(top.values[0]), min_side, contrast, len(pick["crops"]))
            stats["reliable"] = stats.get("reliable", 0) + int(ok)
            gid = f"{cid}:{tid}"
            crop_rel = f"{rel}crops/{cid}_{tid}.jpg"
            cv2.imwrite(str(CACHE / crop_rel), pick["crops"][0], [cv2.IMWRITE_JPEG_QUALITY, 92])
            emb.append(feat.cpu().numpy())
            clip_emb.append(ci[0].cpu().numpy())
            keys.append(gid)
            stats["identified"] += 1
            vehicles[gid] = {
                "gid": gid, "crop": crop_rel, "color": colour, "body": body, "quality": pick["quality"], "reliable": ok,
                "evidence": {"consistency": round(consistency, 2), "margin": round(margin, 3), "min_side": int(min_side), "contrast": round(contrast, 1), "crops": len(pick["crops"])},
                "top5": [{"name": names[int(i)], "p": round(float(p), 4)} for p, i in zip(top.values, top.indices)],
                "sightings": [{"cam": cid, "track": int(tid), "t": round(pick["frames"][0] / hi_tracks["fps"], 2), "crop": crop_rel, "sim": 1.0}],
                "_fi": pick["frames"][0], "_box": pick["box"],
            }
        mine = [v for k, v in vehicles.items() if k.startswith(cid + ":")]
        print(f"{cid} [{cam.get('condition')}]: {len(picks)} tracks, {len(mine)} identified, {sum(v['reliable'] for v in mine)} with a reliable make/model")

    E = np.stack(emb)
    S = E @ E.T
    index = {k: i for i, k in enumerate(keys)}

    # Ground-truth identities (CityFlowV2 for Kingsway, RoundaboutHD for Cambie) for every vehicle we can match.
    gt_of = assign_ground_truth(spec, vehicles)
    order = {c["id"]: i for i, c in enumerate(spec["cameras"])}
    groups = {}
    for gid, key in gt_of.items():
        groups.setdefault(key, []).append(gid)
    target = {}
    for cid, t in spec.get("trace", {}).items():
        for gid, key in gt_of.items():
            if gid.startswith(cid + ":") and key == ("cityflow", t["gt_vehicle"]):
                target[cid] = gid
    t_keys = list(target.values())

    for gid in keys:
        i = index[gid]
        cam_i = gid.split(":")[0]
        me = vehicles[gid]
        members = sorted(groups.get(gt_of.get(gid), [gid]), key=lambda g: order[g.split(":")[0]])
        # One appearance per camera: tracker ID switches can split a car into several tracks.
        per_cam = {}
        for g in members:
            c = g.split(":")[0]
            if c == cam_i and g != gid:
                continue
            if g == gid or c not in per_cam or (per_cam[c] != gid and vehicles[g]["quality"] > vehicles[per_cam[c]]["quality"]):
                per_cam[c] = g
        members = sorted(per_cam.values(), key=lambda g: order[g.split(":")[0]])
        me["journey"] = [{**vehicles[g]["sightings"][0], "sim": round(float(S[i, index[g]]), 3), "verified": len(members) > 1, **({"self": True} if g == gid else {})}
                         for g in members]
        me["similar"] = []
        for j in np.argsort(-S[i]):
            k = keys[j]
            if k in members or k.split(":")[0] == cam_i:
                continue
            me["similar"].append({**vehicles[k]["sightings"][0], "sim": round(float(S[i, j]), 3)})
            if len(me["similar"]) >= 4:
                break
        me["sightings"] = [me["sightings"][0]] + [x for x in me["journey"] if not x.get("self")] + me["similar"]
    for v in vehicles.values():
        v.pop("_fi", None)
        v.pop("_box", None)
    stats["journeys"] = sum(1 for g in groups.values() if len({x.split(":")[0] for x in g}) > 1)

    print(json.dumps({**stats, "set": spec_name, "cameras": len(cameras), "corridor_target": t_keys}))
    return {"cameras": cameras, "vehicles": vehicles, "keys": keys, "E": E, "C": np.stack(clip_emb), "target": t_keys}


def main():
    M = Models()
    live = index(M, LIVE, "live_spec.json", CACHE)
    (CACHE / "cameras.json").write_text(json.dumps(live["cameras"], indent=1))
    (CACHE / "vehicles.json").write_text(json.dumps(live["vehicles"]))
    np.save(ROOT / "demo_footage" / "reid_embeddings.npy", live["E"])
    np.save(ROOT / "demo_footage" / "clip_embeddings.npy", live["C"])
    (ROOT / "demo_footage" / "embedding_keys.json").write_text(json.dumps(live["keys"]))

    story = index(M, WEB, "cameras_spec.json", CACHE / "story")
    cameras, vehicles, keys, t_keys = story["cameras"], story["vehicles"], story["keys"], story["target"]
    clip_model, clip_proc = M.clip, M.proc
    with torch.no_grad():
        q = clip_model.get_text_features(**clip_proc(text=[WITNESS["query"]], return_tensors="pt", padding=True).to(DEVICE))
    qv = F.normalize(q.float(), dim=1)[0].cpu().numpy()
    C = story["C"]
    scores = C @ qv
    order = np.argsort(-scores)
    hits = [{"gid": keys[i], "cam": keys[i].split(":")[0], "track": int(keys[i].split(":")[1]),
             "crop": vehicles[keys[i]]["crop"], "score": round(float(scores[i]), 4)} for i in order[:8]]

    # Landing trace: the corridor target on the camera where its best crop is cleanest.
    if t_keys:
        t_gid = max(t_keys, key=lambda k: vehicles[k]["quality"])
    else:
        t_gid = max(vehicles, key=lambda k: vehicles[k]["top5"][0]["p"])
    t_cam = t_gid.split(":")[0]
    by_id = {c["id"]: c for c in cameras}
    landing = {
        "scenes": [{"cam": c, "title": t, "sub": s, "clip": by_id[c]["clip"], "tracks": by_id[c]["tracks"],
                    "poster": by_id[c]["poster"], "res": by_id[c]["res"]} for c, t, s in LANDING_SCENES if c in by_id],
        "trace": {"cam": t_cam, "clip": by_id[t_cam]["clip"], "tracks": by_id[t_cam]["tracks"], "poster": by_id[t_cam]["poster"],
                  "track": int(t_gid.split(":")[1]), "t": vehicles[t_gid]["sightings"][0]["t"], "vehicle": t_gid},
        "witness": {**WITNESS, "hits": hits},
        "vehicles": vehicles,
    }
    (CACHE / "landing.json").write_text(json.dumps(landing))
    ev = ROOT / "demo_footage" / "eval_reid.json"
    if ev.exists():
        (CACHE / "metrics.json").write_text(json.dumps({"reid_cityflow_s01": json.loads(ev.read_text()), "reid_embedding_in_use": REID_EMBED}, indent=1))
    print(json.dumps({"witness_top": hits[:3], "trace": landing["trace"]["vehicle"]}, indent=1))


if __name__ == "__main__":
    main()

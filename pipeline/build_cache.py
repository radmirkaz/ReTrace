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


BODY_GROUP = {"sedan": "car", "hatchback": "car", "coupe": "car", "station wagon": "car", "SUV": "suv", "minivan": "suv",
              "pickup truck": "pickup", "van": "van", "bus": "heavy", "box truck": "heavy", "semi truck": "heavy", "motorcycle": "moto"}
# Colour families overlap on purpose: CLIP confuses white/silver, silver/grey and brown/black on real footage.
COLOUR_FAMILY = {"white": ["light"], "silver": ["light", "grey"], "beige": ["light", "warm"], "grey": ["grey"], "black": ["dark"],
                 "brown": ["dark", "warm"], "red": ["red"], "orange": ["red", "warm"], "yellow": ["warm"], "blue": ["blue"], "green": ["green"]}


def _family_probs(p, names, mapping):
    fams = sorted({f for fs in mapping.values() for f in (fs if isinstance(fs, list) else [fs])})
    out = np.zeros(len(fams))
    for prob, n in zip(p, names):
        fs = mapping[n] if isinstance(mapping[n], list) else [mapping[n]]
        for f in fs:
            out[fams.index(f)] += prob / len(fs)
    return out / out.sum()


def compatible(a, b, body_min, colour_min):
    """Post-processing gate for matches: same body group and colour family (soft, via CLIP probabilities)."""
    body = float(_family_probs(a["_body"], BODIES, BODY_GROUP) @ _family_probs(b["_body"], BODIES, BODY_GROUP))
    if body < body_min:
        return False
    if a["_night"] or b["_night"]:
        return True  # colour is unreliable under streetlights
    colour = float(_family_probs(a["_colour"], COLOURS, COLOUR_FAMILY) @ _family_probs(b["_colour"], COLOURS, COLOUR_FAMILY))
    return colour >= colour_min


QUERY_BODY = {"sedan": "car", "car": "car", "hatchback": "car", "coupe": "car", "wagon": "car", "suv": "suv", "minivan": "suv",
              "pickup": "pickup", "truck": "pickup", "van": "van", "bus": "heavy", "lorry": "heavy", "semi": "heavy", "motorcycle": "moto", "motorbike": "moto"}
QUERY_COLOUR = {c: COLOUR_FAMILY[c] for c in COLOURS} | {"gray": ["grey"], "dark": ["dark"], "light": ["light"]}


MAKES = {"toyota", "ford", "honda", "chevrolet", "chevy", "dodge", "ram", "gmc", "nissan", "hyundai", "kia", "bmw", "audi", "mercedes",
         "volkswagen", "subaru", "mazda", "jeep", "tesla", "lexus", "acura", "volvo"}


def query_attributes(q: str):
    """Colour families and body group named in a text query (e.g. 'blue pickup truck')."""
    words = [w.strip(".,!?").lower() for w in q.split()]
    colours = sorted({f for w in words if w in QUERY_COLOUR for f in QUERY_COLOUR[w]})
    body = next((QUERY_BODY[w] for w in words if w in QUERY_BODY and not (w == "truck" and "pickup" not in words and any(x in words for x in ("box", "semi", "dump")))), None)
    return colours, body


def attributes(v):
    """Per-vehicle body-group and colour-family probabilities (saved for search filtering)."""
    groups = sorted(set(BODY_GROUP.values()))
    fams = sorted({f for fs in COLOUR_FAMILY.values() for f in fs})
    bg = _family_probs(v["_body"], BODIES, BODY_GROUP)
    cf = _family_probs(v["_colour"], COLOURS, COLOUR_FAMILY)
    return {"body": {g: round(float(p), 3) for g, p in zip(groups, bg)}, "colour": {f: round(float(p), 3) for f, p in zip(fams, cf)}, "night": bool(v["_night"])}


def matches_query(attr, colours, body, min_p=0.25):
    if body and attr["body"].get(body, 0) < min_p:
        return False
    if colours and not attr["night"] and sum(attr["colour"].get(f, 0) for f in colours) < min_p:
        return False
    return True


def reliable(consistency, margin, p1, min_side, contrast, n_crops, condition="day"):
    """Show a make/model when the crop has enough detail and either most crops agree or one clean
    prediction is strong. Rain/snow/night — where confident-but-wrong labels happened — need the
    crops to agree strongly."""
    if min_side < 48 or contrast < 20:
        return False
    if condition in ("night", "rain", "snow"):
        return n_crops >= 2 and consistency >= 0.75 and p1 >= 0.25 and margin >= 0.08
    return (consistency >= 0.5 and p1 >= 0.2 and margin >= 0.05) or p1 >= 0.6


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
    "transcript": "It was a blue pickup truck, maybe a Toyota. It hit the cyclist and kept going east on Kingsway, around nine-thirty.",
    "query": "a blue pickup truck",
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
                body_p = F.softmax((ci @ body_t.T)[0] * 100, dim=0).cpu().numpy()
                colour_p = F.softmax((ci @ colour_t.T)[0] * 100, dim=0).cpu().numpy()
                body = BODIES[int(body_p.argmax())]
                colour = COLOURS[int(colour_p.argmax())]
            top = torch.topk(probs, 5)
            top1 = int(top.indices[0])
            consistency = float((per_crop.argmax(dim=1) == top1).float().mean())
            margin = float(top.values[0] - top.values[1])
            best = pick["crops"][0]
            min_side = min(best.shape[:2])
            contrast = float(cv2.cvtColor(best, cv2.COLOR_BGR2GRAY).std())
            ok = reliable(consistency, margin, float(top.values[0]), min_side, contrast, len(pick["crops"]), cam.get("condition", "day"))
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
                "_fi": pick["frames"][0], "_box": pick["box"], "_body": body_p, "_colour": colour_p, "_night": cam.get("condition") == "night",
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

    # Corridors: cameras on one street, where the same traffic passes (camera graph of the city).
    corridors = {}
    for c in spec["cameras"]:
        for prefix in ("kingsway-", "cambie-king-edward-"):
            if c["id"].startswith(prefix):
                corridors.setdefault(prefix, []).append(c["id"])
    corridor_of = {cid: ids for ids in corridors.values() for cid in ids}
    by_cam = {}
    for g in keys:
        by_cam.setdefault(g.split(":")[0], []).append(g)

    def best_on(gid, cam, filt):
        cands = [k for k in by_cam.get(cam, []) if not filt or compatible(vehicles[gid], vehicles[k], *filt)]
        return max(cands, key=lambda k: S[index[gid], index[k]]) if cands else None

    def propose(gid, tau, filt, mutual=True):
        """The model's journey: best match on each other corridor camera, above `tau`; with `mutual`,
        that match must also pick this vehicle as its best on our camera (mutual nearest neighbours)."""
        i, cam_i = index[gid], gid.split(":")[0]
        steps = []
        for c in corridor_of.get(cam_i, []):
            if c == cam_i:
                continue
            k = best_on(gid, c, filt)
            if k is None or S[i, index[k]] < tau:
                continue
            if mutual and best_on(k, cam_i, filt) != gid:
                continue
            steps.append(k)
        return steps

    def score(tau, filt, mutual=True):
        tp = fp = pos = 0
        for gid in keys:
            if gid not in gt_of or gid.split(":")[0] not in corridor_of:
                continue
            cam_i = gid.split(":")[0]
            truth = {k.split(":")[0] for k in groups[gt_of[gid]] if k.split(":")[0] != cam_i}
            pos += len(truth)
            for k in propose(gid, tau, filt, mutual):
                if gt_of.get(k) == gt_of[gid]:
                    tp += 1
                else:
                    fp += 1
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, pos)
        return {"tau": round(tau, 2), "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(2 * prec * rec / max(1e-9, prec + rec), 3)}

    # Filter strength: the loosest-to-strictest setting that still keeps >=95% of true same-car pairs.
    true_pairs = [(a, b) for g in groups.values() for a in g for b in g if a < b and a.split(":")[0] != b.split(":")[0]]
    filt = (0.15, 0.1)
    for f in [(0.4, 0.3), (0.3, 0.25), (0.25, 0.2), (0.2, 0.15), (0.15, 0.1)]:
        kept = sum(compatible(vehicles[a], vehicles[b], *f) for a, b in true_pairs) / max(1, len(true_pairs))
        if kept >= 0.95:
            filt = f
            break
    kept = sum(compatible(vehicles[a], vehicles[b], *filt) for a, b in true_pairs) / max(1, len(true_pairs))
    grid = [0.3 + 0.02 * k for k in range(31)]

    def operating_point(rows):
        """Best F1: with the filter and mutual check this finds ~2/3 of real sightings at ~2/3 precision."""
        return max(rows, key=lambda r: r["f1"])

    if true_pairs:
        variants = {
            "model only": operating_point([score(t, None, False) for t in grid]),
            "+ body/colour filter": operating_point([score(t, filt, False) for t in grid]),
            "+ filter + mutual best match": operating_point([score(t, filt, True) for t in grid]),
        }
    else:
        variants = {"+ filter + mutual best match": {"tau": 0.6}}
    curve = [score(t, filt, True) for t in grid] if true_pairs else []
    tau = variants["+ filter + mutual best match"]["tau"]
    postproc = {"filter": {"body_min": filt[0], "colour_min": filt[1], "true_pairs_kept": round(kept, 3)},
                "variants": variants, "curve_filter_mutual": curve, "gt_pairs": len(true_pairs), "in_use": "+ filter + mutual best match"}

    for gid in keys:
        i = index[gid]
        cam_i = gid.split(":")[0]
        me = vehicles[gid]
        steps = propose(gid, tau, filt)
        members = sorted([gid] + steps, key=lambda g: order[g.split(":")[0]])

        def correctness(k):
            if k == gid or gid not in gt_of or k not in gt_of:
                return None
            return gt_of[k] == gt_of[gid]

        me["journey"] = [{**vehicles[g]["sightings"][0], "sim": round(float(S[i, index[g]]), 3),
                          **({"self": True} if g == gid else {"correct": correctness(g), "verified": correctness(g) is True})}
                         for g in members]
        me["similar"] = []
        for j in np.argsort(-S[i]):
            k = keys[j]
            if k in members or k.split(":")[0] == cam_i or not compatible(me, vehicles[k], *filt):
                continue
            me["similar"].append({**vehicles[k]["sightings"][0], "sim": round(float(S[i, j]), 3)})
            if len(me["similar"]) >= 4:
                break
        me["sightings"] = [me["sightings"][0]] + [x for x in me["journey"] if not x.get("self")] + me["similar"]
    stats["postproc"] = postproc
    attrs = {g: attributes(v) for g, v in vehicles.items()}
    for v in vehicles.values():
        v.pop("_fi", None)
        v.pop("_box", None)
        for k in ("_body", "_colour", "_night"):
            v.pop(k, None)
    stats["journeys"] = sum(1 for v in vehicles.values() if len(v["journey"]) > 1)

    print(json.dumps({**stats, "set": spec_name, "cameras": len(cameras), "corridor_target": t_keys}))
    return {"cameras": cameras, "vehicles": vehicles, "keys": keys, "E": E, "C": np.stack(clip_emb), "target": t_keys, "postproc": postproc, "attrs": attrs}


def main():
    M = Models()
    live = index(M, LIVE, "live_spec.json", CACHE)
    (CACHE / "cameras.json").write_text(json.dumps(live["cameras"], indent=1))
    (CACHE / "vehicles.json").write_text(json.dumps(live["vehicles"]))
    np.save(ROOT / "demo_footage" / "reid_embeddings.npy", live["E"])
    np.save(ROOT / "demo_footage" / "clip_embeddings.npy", live["C"])
    (ROOT / "demo_footage" / "embedding_keys.json").write_text(json.dumps(live["keys"]))

    (ROOT / "demo_footage" / "attributes.json").write_text(json.dumps(live["attrs"]))

    story = index(M, WEB, "cameras_spec.json", CACHE / "story")
    cameras = story["cameras"]
    lv, lkeys = live["vehicles"], live["keys"]

    # Witness search over the live feeds, ranked by CLIP and gated by the colour/body words in the query.
    clip_model, clip_proc = M.clip, M.proc
    with torch.no_grad():
        q = clip_model.get_text_features(**clip_proc(text=[WITNESS["query"]], return_tensors="pt", padding=True).to(DEVICE))
    qv = F.normalize(q.float(), dim=1)[0].cpu().numpy()
    scores = live["C"] @ qv
    colours, body = query_attributes(WITNESS["query"])
    # A make the witness mentions ("maybe a Toyota") lifts vehicles whose reliable make/model agrees.
    makes = [w for w in (x.strip(".,!?").lower() for x in WITNESS["transcript"].split()) if w in MAKES]
    def make_hit(g):
        v = lv[g]
        return bool(makes) and v["reliable"] and any(m in v["top5"][0]["name"].lower() for m in makes)
    ranked = sorted(range(len(lkeys)), key=lambda i: (not matches_query(live["attrs"][lkeys[i]], colours, body), not make_hit(lkeys[i]), -scores[i]))
    hits = [{"gid": lkeys[i], "cam": lkeys[i].split(":")[0], "track": int(lkeys[i].split(":")[1]),
             "crop": lv[lkeys[i]]["crop"], "score": round(float(scores[i]), 4)} for i in ranked[:8]]

    # Trace: the blue Toyota Tacoma on the Kingsway corridor (identified with a reliable make/model).
    def tacoma_rank(v):
        return ("tacoma" in v["top5"][0]["name"].lower(), v["reliable"], v["color"] == "blue", len(v["journey"]), v["quality"])
    t_gid = max((g for g in lv if g.startswith("kingsway-")), key=lambda g: tacoma_rank(lv[g]))
    t_cam, t_track = t_gid.split(":")[0], int(t_gid.split(":")[1])
    live_tracks = json.loads((LIVE / f"{t_cam}.json").read_text())
    fps = live_tracks["fps"]
    t_mid = lv[t_gid]["sightings"][0]["t"]
    # 5 s around the vehicle's best (clean, sharp) frame, so the pull-out starts from a good view.
    clip_start = max(0.0, t_mid - 3.0)
    clip_dur = 5.0
    import subprocess
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(clip_start), "-t", str(clip_dur), "-i", str(LIVE / f"{t_cam}.mp4"), "-an",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "24", "-pix_fmt", "yuv420p", "-g", "1", "-keyint_min", "1",
                    "-movflags", "+faststart", str(CACHE / "story" / "clips" / "trace.mp4")], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(CACHE / "story" / "clips" / "trace.mp4"), "-frames:v", "1",
                    str(CACHE / "story" / "clips" / "trace.jpg")], check=True)
    f0 = int(round(clip_start * fps))
    sliced = {**live_tracks, "frames": live_tracks["frames"][f0:f0 + int(round(clip_dur * fps))]}
    sliced["n"] = len(sliced["frames"])
    (CACHE / "story" / "tracks" / "trace.json").write_text(json.dumps(sliced, separators=(",", ":")))

    landing_vehicles = dict(story["vehicles"])
    for g in {t_gid, *(h["gid"] for h in hits), *(s["cam"] + ":" + str(s["track"]) for s in lv[t_gid]["sightings"])}:
        landing_vehicles[g] = lv[g]
    by_id = {c["id"]: c for c in cameras}
    landing = {
        "scenes": [{"cam": c, "title": t, "sub": s, "clip": by_id[c]["clip"], "tracks": by_id[c]["tracks"],
                    "poster": by_id[c]["poster"], "res": by_id[c]["res"]} for c, t, s in LANDING_SCENES if c in by_id],
        "trace": {"cam": t_cam, "clip": "story/clips/trace.mp4", "tracks": "story/tracks/trace.json", "poster": "story/clips/trace.jpg",
                  "track": t_track, "t": round(t_mid - clip_start, 2),
                  "vehicle": t_gid},
        "witness": {**WITNESS, "hits": hits},
        "vehicles": landing_vehicles,
    }
    (CACHE / "landing.json").write_text(json.dumps(landing))
    ev = ROOT / "demo_footage" / "eval_reid.json"
    if ev.exists():
        (CACHE / "metrics.json").write_text(json.dumps({"reid_cityflow_s01": json.loads(ev.read_text()), "reid_embedding_in_use": REID_EMBED,
                                                         "journey_matching_live_feeds": live["postproc"]}, indent=1))
    print(json.dumps({"witness_top": hits[:4], "trace": landing["trace"], "trace_vehicle": {k: lv[t_gid][k] for k in ("color", "body", "reliable")} | {"top1": lv[t_gid]["top5"][0], "journey": [(x["cam"], x["sim"], x.get("correct")) for x in lv[t_gid]["journey"]]}}, indent=1))


if __name__ == "__main__":
    main()

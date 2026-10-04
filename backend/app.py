"""ReTrace API: serves the indexed feeds and runs text, voice and photo/video search on the GPU.

Run on a CUDA machine from the repository root:
    uvicorn backend.app:app --host 0.0.0.0 --port 8000
Then start the web app with VITE_API_URL=http://<pc-address>:8000.
"""
import json
import tempfile
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "web" / "public" / "cache"
INDEX = ROOT / "demo_footage"
CLIP_MODEL = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="ReTrace")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def read(name: str):
    return json.loads((CACHE / name).read_text())


@lru_cache
def clip():
    from transformers import CLIPModel, CLIPProcessor
    return CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE).eval(), CLIPProcessor.from_pretrained(CLIP_MODEL)


@lru_cache
def whisper_model():
    import whisper
    return whisper.load_model("base", device=DEVICE)


# Query words -> body group / colour families (same vocabulary as pipeline/build_cache.py).
QUERY_BODY = {"sedan": "car", "car": "car", "hatchback": "car", "coupe": "car", "wagon": "car", "suv": "suv", "minivan": "suv",
              "pickup": "pickup", "truck": "pickup", "van": "van", "bus": "heavy", "lorry": "heavy", "semi": "heavy", "motorcycle": "moto"}
QUERY_COLOUR = {"white": ["light"], "silver": ["light", "grey"], "beige": ["light", "warm"], "grey": ["grey"], "gray": ["grey"],
                "black": ["dark"], "brown": ["dark", "warm"], "dark": ["dark"], "light": ["light"], "red": ["red"], "orange": ["red", "warm"],
                "yellow": ["warm"], "blue": ["blue"], "green": ["green"]}


MAKES = {"toyota", "ford", "honda", "chevrolet", "chevy", "dodge", "ram", "gmc", "nissan", "hyundai", "kia", "bmw", "audi", "mercedes",
         "volkswagen", "subaru", "mazda", "jeep", "tesla", "lexus", "acura", "volvo"}


@lru_cache
def attributes():
    path = INDEX / "attributes.json"
    return json.loads(path.read_text()) if path.exists() else {}


def matches(attr, colours, body, min_p=0.25):
    """Post-processing gate: keep results whose body group and colour family fit the query."""
    if not attr:
        return True
    if body and attr["body"].get(body, 0) < min_p:
        return False
    if colours and not attr["night"] and sum(attr["colour"].get(f, 0) for f in colours) < min_p:
        return False
    return True


@lru_cache
def crop_index():
    return np.load(INDEX / "clip_embeddings.npy"), json.loads((INDEX / "embedding_keys.json").read_text())


@app.on_event("startup")
def warm():
    clip()
    crop_index()
    attributes()
    threading.Thread(target=_warm_photo, daemon=True).start()  # detector + re-ID models load in the background


def _warm_photo():
    from . import photo
    photo.models()
    photo.index()


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE}


@app.get("/cameras")
def cameras():
    return read("cameras.json")


@app.get("/cameras/{cam_id}/tracks")
def tracks(cam_id: str):
    path = CACHE / "tracks" / f"{cam_id}.json"
    if not path.exists():
        raise HTTPException(404, "unknown camera")
    return json.loads(path.read_text())


@app.get("/vehicles")
def vehicles():
    return read("vehicles.json")


@app.get("/landing")
def landing():
    return read("landing.json")


@app.get("/files/{path:path}")
def files(path: str):
    target = (CACHE / path).resolve()
    if CACHE.resolve() not in target.parents or not target.is_file():
        raise HTTPException(404)
    return FileResponse(target)


@app.get("/search")
def search(q: str, k: int = 12):
    model, proc = clip()
    emb, keys = crop_index()
    with torch.no_grad():
        t = model.get_text_features(**proc(text=[q], return_tensors="pt", padding=True).to(DEVICE))
    scores = emb @ F.normalize(t.float(), dim=1)[0].cpu().numpy()
    words = [w.strip(".,!?").lower() for w in q.split()]
    colours = sorted({f for w in words for f in QUERY_COLOUR.get(w, [])})
    body = next((QUERY_BODY[w] for w in words if w in QUERY_BODY), None)
    attrs = attributes()
    v = read("vehicles.json")
    makes = [w for w in words if w in MAKES]

    def make_hit(gid):
        x = v.get(gid, {})
        return bool(makes) and x.get("reliable") and any(m in x["top5"][0]["name"].lower() for m in makes)

    # Matching colour/body first, then a spoken/typed make, then CLIP similarity.
    order = sorted(range(len(keys)), key=lambda i: (not matches(attrs.get(keys[i]), colours, body), not make_hit(keys[i]), -scores[i]))
    out = []
    for i in order[:k]:
        gid = keys[i]
        cam, track = gid.split(":")
        out.append({"gid": gid, "cam": cam, "track": int(track), "crop": v[gid]["crop"], "score": round(float(scores[i]), 4)})
    return out


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=Path(audio.filename or "a.webm").suffix, delete=False) as f:
        f.write(await audio.read())
        path = f.name
    try:
        text = whisper_model().transcribe(path, language="en", fp16=DEVICE == "cuda")["text"].strip()
    finally:
        Path(path).unlink(missing_ok=True)
    return {"text": text}


MAX_UPLOAD = 200 * 1024 * 1024


@app.post("/search/photo")
def search_photo(file: UploadFile = File(...), k: int = 12):
    """Find the vehicle in an uploaded photo or clip and rank every indexed vehicle by re-ID similarity."""
    from . import photo
    data = file.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "File too large (200 MB max).")
    try:
        return photo.search_upload(data, file.filename or "", file.content_type or "", k)
    except ValueError as e:
        raise HTTPException(422, str(e))

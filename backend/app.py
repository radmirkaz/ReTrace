"""ReTrace API — serves the indexed feeds and runs text/voice search on the GPU.

Run on the GPU PC from the repository root (canai env):
    uvicorn backend.app:app --host 0.0.0.0 --port 8000
Then start the web app with VITE_API_URL=http://<pc-address>:8000.
"""
import json
import tempfile
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


@lru_cache
def crop_index():
    return np.load(INDEX / "clip_embeddings.npy"), json.loads((INDEX / "embedding_keys.json").read_text())


@app.on_event("startup")
def warm():
    clip()
    crop_index()


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
    v = read("vehicles.json")
    out = []
    for i in np.argsort(-scores)[:k]:
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

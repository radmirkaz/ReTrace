# ReTrace

Vehicle re-identification across city cameras. ReTrace detects and tracks every
vehicle, recognises its make, model and generation, and finds a specific car by
appearance, by a text description or by a witness's spoken statement, without
relying on licence plates.

Built at StormHacks 2026 (SFU Burnaby, October 3–4).

## Run it

The website works on any machine with Node 20+; the feed data ships in `web/public/cache`.

```bash
cd web
npm install
npm run dev        # http://localhost:5173
```

- `#/` is the story: the problem, detection on real footage in any condition, a witness's
  voice statement turned into a search, and one car traced across cameras.
- `#/live` is the city map: click any camera to watch its feed with tracks, inspect a
  vehicle, follow it to other cameras, or search by text, voice, or a photo or clip of a car
  (button next to the mic, or drag a file onto the map).
- Press `~` (or **Under the hood**) for models, datasets and metrics.

### GPU backend (optional)

Text search with CLIP, voice transcription with Whisper, and photo/video search with the detector and
embeddings model run on a CUDA machine:

```bash
git lfs pull   # model weights in training/
pip install fastapi uvicorn python-multipart torch torchvision transformers openai-whisper ultralytics supervision opencv-python pillow
uvicorn backend.app:app --host 0.0.0.0 --port 8000
cd web && VITE_API_URL=http://<gpu-host>:8000 npm run dev
```

Without the backend, search matches the indexed labels, voice uses the browser's speech
recognition, and photo search is unavailable.

To share the site through a tunnel (one public address for the site and the backend):

```bash
cd web && VITE_API_URL=/api npm run build
API_PROXY=http://<gpu-host>:8000 npx vite preview --port 4173
cloudflared tunnel --url http://localhost:4173
```

## Layout

| Path | Contents |
| --- | --- |
| `web/` | Website (React + Vite) |
| `backend/` | API: feeds, vehicles, CLIP search, Whisper transcription, photo/video search |
| `pipeline/` | Footage selection, clip export, tracking, crop selection, indexing, map build |
| `training/` | Model training code, logs and weights: detector, classifier, embeddings model ([how to retrain](training/README.md)) |
| `demo_footage/` | Source datasets and intermediate exports (not committed) |

## Pipeline

Run on the GPU machine from the repository root. The pipeline loads its models from `training/`
(`git lfs pull` first) and reads raw videos from `demo_footage/sources/`.

| Model | Weights | Role |
| --- | --- | --- |
| Detector | `training/yolo/runs/detect/train1/weights/best.pt` | YOLOv12n, one `car` class, tracked with ByteTrack |
| Classifier | `training/reid/models/classifier.pt` | Make, model and generation (9,630 classes) |
| Embeddings model | `training/reid/models/embeddings_model.pt` | 2048-D appearance embedding for re-identification |

The classifier and the embeddings model share the same architecture and training code; the embeddings model uses
less data, focused on US and Canadian cars, with parameters tuned for embedding quality.


| Step | Script | Output |
| --- | --- | --- |
| Shortlist footage | `pipeline/scan_candidates.py` | Scored 3-second windows with contact sheets |
| Story clips | `pipeline/export_cameras.py` | Short scroll-scrub clips + per-frame tracks |
| Live feeds | `pipeline/export_live.py` | 45 s feeds picked by quality, tracked at native resolution |
| Index | `pipeline/build_cache.py` | Best crops per vehicle, identification, cross-camera links, CLIP index |
| Map | `pipeline/map/build_map.py` | Simplified Vancouver geometry from OpenStreetMap |

Crop selection (`pipeline/crops.py`) scores every detection of a track for truncation at
the frame edge, overlap with other vehicles, aspect-ratio consistency, size, confidence
and sharpness, then averages identification over the best few crops of each vehicle.

## Data

Camera locations on the map are illustrative; the footage comes from public datasets.

| Dataset | Use | Licence |
| --- | --- | --- |
| CityFlowV2 (AI City Challenge 2022) | Multi-camera feeds with cross-camera ground truth (US) | AIC 2022 data licence |
| RoundaboutHD | 4K multi-camera feeds | See dataset page |
| AAU RainSnow | Rain, snow and night feeds (Denmark) | CC BY 4.0 |
| MIO-TCD Localization | Camera-quality and angle evaluation | CC BY-NC-SA 4.0 |
| OpenStreetMap | Vancouver map geometry | ODbL, © OpenStreetMap contributors |

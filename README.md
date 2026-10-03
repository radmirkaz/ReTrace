# ReTrace

Vehicle re-identification across traffic cameras: detect and track every
vehicle, recognise make, model and generation, and find a specific car by
appearance, text or a witness's voice, without relying on licence plates.

Built at StormHacks 2026 (SFU Burnaby, October 3–4).

## Layout

| Path | Contents |
| --- | --- |
| `web/` | ReTrace website (React) |
| `backend/` | API that serves runs, detections, crops and search |
| `pipeline/` | Footage shortlisting, clip export and robustness evaluation |
| `demo_footage/` | Evaluation datasets and exported web clips (not committed) |

## Pipeline

Run on the GPU machine from the repository root:

```bash
python pipeline/scan_candidates.py   # shortlist 3-second windows in every video
python pipeline/make_web_clips.py    # export scroll-scrubbable clips + per-frame tracks
```

## Datasets

| Dataset | Use | Licence |
| --- | --- | --- |
| CityFlowV2 (AI City Challenge 2022) | Multi-camera tracking, US fleet | AIC 2022 data licence |
| RoundaboutHD | Cross-camera re-identification, 4K | See dataset page |
| MIO-TCD Localization | Detection across Canadian/US traffic cameras | CC BY-NC-SA 4.0 |
| AAU RainSnow | Rain, snow and night robustness | CC BY 4.0 |

"""Serve the built website and the API from one address, for sharing the GPU machine through a tunnel.

Build the site with the API under /api, then run from the repository root:
    cd web && VITE_API_URL=/api npm run build
    uvicorn backend.serve:app --host 0.0.0.0 --port 8000
The API is then at /api (e.g. /api/search); the site at /.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .app import app as api, warm

DIST = Path(__file__).resolve().parents[1] / "web" / "dist"

app = FastAPI(title="ReTrace site")
app.add_event_handler("startup", warm)  # mounted apps do not run their own startup hooks
app.mount("/api", api)
app.mount("/", StaticFiles(directory=DIST, html=True), name="site")

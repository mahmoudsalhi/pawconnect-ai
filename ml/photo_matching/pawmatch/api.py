"""FastAPI service for the PawConnect backend.

    CHECKPOINT=checkpoints/m2.pt uvicorn pawmatch.api:app --port 8002

Endpoints
  POST /reports         register a lost/found report (photos + metadata)
                        -> returns report_id and likely matches to alert on
  POST /match           match photos without storing anything
  DELETE /reports/{id}  remove a report (pet reunited / closed)
  GET  /health
"""
from __future__ import annotations

import io
import os
import uuid
from datetime import datetime

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from .dataset import eval_transform
from .matcher import Matcher, Report
from .model import EmbeddingNet, load_checkpoint

app = FastAPI(title="PawConnect M2 - Photo Matching")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = os.environ.get("CHECKPOINT")
THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.5"))
MAX_PHOTOS = 6

if CKPT and os.path.exists(CKPT):
    _model = load_checkpoint(CKPT, DEVICE)
    _size = torch.load(CKPT, map_location="cpu").get("img_size", 224)
else:  # untrained fallback so the API can be developed against before training ends
    _model = EmbeddingNet(pretrained=False).to(DEVICE).eval()
    _size = 224
_tf = eval_transform(_size)
matcher = Matcher(min_score=THRESHOLD)


def _strip_exif(img: Image.Image) -> Image.Image:
    """Re-encode pixels only, dropping EXIF/GPS (privacy requirement)."""
    return Image.fromarray(np.asarray(img))


@torch.no_grad()
def _embed(files: list[UploadFile]) -> np.ndarray:
    if not files:
        raise HTTPException(400, "At least one photo is required")
    if len(files) > MAX_PHOTOS:
        raise HTTPException(400, f"At most {MAX_PHOTOS} photos")
    tensors = []
    for f in files:
        try:
            img = Image.open(io.BytesIO(f.file.read())).convert("RGB")
        except Exception:
            raise HTTPException(400, f"Cannot read image {f.filename}")
        tensors.append(_tf(_strip_exif(img)))
    return _model(torch.stack(tensors).to(DEVICE)).cpu().numpy()


def _build_report(report_id, kind, species, embs, colors, when, lat, lon) -> Report:
    if kind not in ("lost", "found"):
        raise HTTPException(400, "kind must be 'lost' or 'found'")
    return Report(
        report_id=report_id,
        kind=kind,
        species=species.lower(),
        embeddings=embs,
        colors={c.strip().lower() for c in colors.split(",") if c.strip()},
        when=datetime.fromisoformat(when) if when else None,
        lat=lat,
        lon=lon,
    )


def _serialize(matches):
    return [
        {"report_id": m.report_id, "score": round(m.score, 4),
         "visual": round(m.visual, 4), "metadata": round(m.meta, 4)}
        for m in matches
    ]


@app.get("/health")
def health():
    return {"status": "ok", "open_reports": len(matcher.reports), "device": DEVICE,
            "trained": bool(CKPT and os.path.exists(CKPT))}


@app.post("/embed")
def embed_photo(photo: UploadFile = File(...)):
    """Visual signature of one photo: a 256-number, L2-normalised vector.

    This is what the PawConnect backend uses (``app/ai/models.py``). It keeps its own report
    database and match cache, so this service stays stateless for that purpose.
    """
    vec = _embed([photo])[0]
    return {"dim": int(vec.shape[0]), "embedding": [round(float(x), 6) for x in vec]}


@app.post("/reports")
def create_report(
    kind: str = Form(...),
    species: str = Form(...),
    colors: str = Form(""),
    when: str | None = Form(None),
    lat: float | None = Form(None),
    lon: float | None = Form(None),
    top_k: int = Form(5),
    photos: list[UploadFile] = File(...),
):
    report_id = str(uuid.uuid4())
    report = _build_report(report_id, kind, species, _embed(photos), colors, when, lat, lon)
    matches = matcher.find(report, top_k)  # search BEFORE adding so it can't match itself
    matcher.add(report)
    return {"report_id": report_id, "matches": _serialize(matches)}


@app.post("/match")
def match_only(
    kind: str = Form(...),
    species: str = Form(...),
    colors: str = Form(""),
    when: str | None = Form(None),
    lat: float | None = Form(None),
    lon: float | None = Form(None),
    top_k: int = Form(5),
    photos: list[UploadFile] = File(...),
):
    report = _build_report("query", kind, species, _embed(photos), colors, when, lat, lon)
    return {"matches": _serialize(matcher.find(report, top_k))}


@app.delete("/reports/{report_id}")
def delete_report(report_id: str):
    if report_id not in matcher.reports:
        raise HTTPException(404, "Unknown report")
    matcher.remove(report_id)
    return {"deleted": report_id}

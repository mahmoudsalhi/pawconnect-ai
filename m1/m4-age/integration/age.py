"""M4 · Age prediction at inference time. Copy to backend/app/ai/age.py.

Needs only: onnxruntime, numpy, pillow (no PyTorch on the server).

    model = get_age_model()                    # loaded once, cached
    result = model.predict(list_of_photo_bytes, species="dog")
    # -> AgeResult(group="adult"|"young"|"senior"|None, confidence=82, probabilities={...})
    # group None means "Not sure" (confidence below the threshold chosen at training time).
    # get_age_model() returns None when the model files are missing, so the caller can
    # fall back to the demo placeholder.

Config (.env):  M4_AGE_MODEL_PATH=/path/to/m4_age.onnx   (m4_age.json must sit next to it)
"""
from __future__ import annotations

import io
import json
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

DEFAULT_PATH = "models/m4_age.onnx"
MAX_PHOTOS = 8  # a report has a handful of photos; more adds cost, not accuracy


@dataclass
class AgeResult:
    group: str | None  # "young" | "adult" | "senior" | None (= not sure)
    confidence: int  # 0-100, of the best group after averaging all photos
    probabilities: dict[str, float] = field(default_factory=dict)


def preprocess(image: Image.Image, meta: dict) -> np.ndarray:
    """Resize to size x size, scale to 0-1, normalize with the training mean/std. -> CHW float32."""
    size = meta["size"]
    arr = np.asarray(image.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    arr = (arr - np.array(meta["mean"], dtype=np.float32)) / np.array(meta["std"], dtype=np.float32)
    return arr.transpose(2, 0, 1).astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class AgeModel:
    def __init__(self, onnx_path: str | Path):
        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        self.meta = json.loads(onnx_path.with_suffix(".json").read_text())
        self.groups: list[str] = self.meta["groups"]
        self.threshold: float = float(self.meta.get("threshold", 0.6))
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, images: list[bytes | None], species: str | None = None) -> AgeResult | None:
        """Average the probabilities of every readable photo. Returns None if none is usable.

        `species` is accepted so the call matches the provider interface; the shared
        model is trained on dogs and cats together and does not need it.
        """
        batch = []
        for raw in images:
            if not raw:  # remote seed photos come back as None
                continue
            try:
                img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
                batch.append(preprocess(img, self.meta))
            except Exception:  # corrupt or unsupported file: skip it
                log.warning("M4 age: skipped an unreadable photo")
            if len(batch) == MAX_PHOTOS:
                break
        if not batch:
            return None

        logits = self.session.run(None, {self.input_name: np.stack(batch)})[0]
        probs = _softmax(logits).mean(axis=0)
        best = int(probs.argmax())
        confidence = int(round(float(probs[best]) * 100))
        group = self.groups[best] if probs[best] >= self.threshold else None
        return AgeResult(group=group, confidence=confidence,
                         probabilities={g: round(float(p), 4) for g, p in zip(self.groups, probs)})


@lru_cache(maxsize=1)
def get_age_model() -> AgeModel | None:
    path = Path(os.environ.get("M4_AGE_MODEL_PATH", DEFAULT_PATH))
    if not path.exists() or not path.with_suffix(".json").exists():
        log.warning("M4 age model not found at %s; using the placeholder.", path)
        return None
    return AgeModel(path)

"""M5 · Dog skin problem detection at inference time (copied from m1/m5-skin/integration/health.py).

Needs only: onnxruntime, numpy, pillow (no PyTorch on the server).

    model = get_health_model()                   # loaded once, cached; None if files are missing
    res = model.predict(list_of_photo_bytes, species="dog")
    # -> HealthResult(status="healthy"|"skin_problem"|"unsure", condition="ringworm"|None,
    #                 confidence=87, probabilities={...})
    # species == "cat" -> None (the model was trained on dogs only).

This is an indication to see a vet, never a diagnosis.

Config (backend/.env):  M5_SKIN_MODEL_PATH=/path/to/m5_skin.onnx   (m5_skin.json must sit next to it)
Read through app.config.Settings; a relative path starts at the project root.
"""
from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ..config import BASE_DIR, get_settings

log = logging.getLogger(__name__)

MAX_PHOTOS = 8
HEALTHY = 0


@dataclass
class HealthResult:
    status: str                      # "healthy" | "skin_problem" | "unsure"
    condition: str | None            # most likely skin problem when status == "skin_problem"
    confidence: int                  # 0-100
    probabilities: dict[str, float] = field(default_factory=dict)


def preprocess(image: Image.Image, meta: dict) -> np.ndarray:
    size = meta["size"]
    arr = np.asarray(image.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    arr = (arr - np.array(meta["mean"], dtype=np.float32)) / np.array(meta["std"], dtype=np.float32)
    return arr.transpose(2, 0, 1).astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class HealthModel:
    def __init__(self, onnx_path: str | Path):
        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        self.meta = json.loads(onnx_path.with_suffix(".json").read_text())
        self.classes: list[str] = self.meta["classes"]
        self.threshold = float(self.meta.get("threshold", 0.5))
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, images: list[bytes | None], species: str | None = None) -> HealthResult | None:
        if species and species.lower() not in ("dog", "dogs"):
            return None  # trained on dogs only
        batch = []
        for raw in images:
            if not raw:
                continue
            try:
                img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
                batch.append(preprocess(img, self.meta))
            except Exception:
                log.warning("M5 skin: skipped an unreadable photo")
            if len(batch) == MAX_PHOTOS:
                break
        if not batch:
            return None

        probs = _softmax(self.session.run(None, {self.input_name: np.stack(batch)})[0]).mean(axis=0)
        p_healthy = float(probs[HEALTHY])
        p_problem = 1.0 - p_healthy
        t = self.threshold
        condition = None
        if p_healthy >= t and p_healthy >= p_problem:
            status, conf = "healthy", p_healthy
        elif p_problem >= t:
            status, conf = "skin_problem", p_problem
            condition = self.classes[int(np.argmax(probs[1:])) + 1]
        else:
            status, conf = "unsure", max(p_healthy, p_problem)
        return HealthResult(status=status, condition=condition, confidence=int(round(conf * 100)),
                            probabilities={c: round(float(p), 4) for c, p in zip(self.classes, probs)})


def model_path() -> Path:
    path = Path(get_settings().m5_skin_model_path)
    return path if path.is_absolute() else BASE_DIR.parent / path


@lru_cache(maxsize=1)
def get_health_model() -> HealthModel | None:
    path = model_path()
    if not path.exists() or not path.with_suffix(".json").exists():
        log.warning("M5 skin model not found at %s; feature disabled.", path)
        return None
    try:
        return HealthModel(path)
    except Exception:  # broken file or onnxruntime missing: the feature is just off
        log.exception("M5 skin model at %s could not be loaded; feature disabled.", path)
        return None

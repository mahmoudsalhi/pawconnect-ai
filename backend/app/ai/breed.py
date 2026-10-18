"""M1 breed identification: 37 cat and dog breeds from photos.

A ConvNeXt-Tiny network pretrained on ImageNet turns each photo into 768 numbers (frozen, not retrained); a
logistic-regression classifier trained by ``backend/ml/m1_breed/train.py`` on the Oxford-IIIT Pet dataset maps
those to breeds. Weights live in ``weights/m1_breed.npz``.
"""

from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .base import BreedPrediction, ImageInput

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
MODEL_PATH = WEIGHTS_DIR / "m1_breed.npz"
META_PATH = WEIGHTS_DIR / "m1_breed.meta.json"


@lru_cache
def backbone():
    import torch
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    net = convnext_tiny(weights=weights)
    net.classifier[-1] = torch.nn.Identity()  # keep the 768 features, drop the ImageNet classes
    return net.eval(), weights.transforms()


def image_features(images: list) -> np.ndarray:
    """768 features per PIL image, averaged with its mirror image. L2-normalised."""
    import torch

    net, prep = backbone()
    batch = torch.stack([prep(im.convert("RGB")) for im in images])
    with torch.inference_mode():
        f = (net(batch) + net(torch.flip(batch, dims=[3]))) / 2
    f = f.numpy()
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def load_image(image: ImageInput):
    from PIL import Image

    data = image.read()
    if data is None:
        import httpx

        resp = httpx.get(image.url, timeout=10, follow_redirects=True)
        resp.raise_for_status()
        data = resp.content
    return Image.open(io.BytesIO(data))


class BreedModel:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} is missing. Train it with: python -m ml.m1_breed.train")
        w = np.load(MODEL_PATH, allow_pickle=False)
        self.coef, self.intercept = w["coef"], w["intercept"]
        self.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        self.labels: list[str] = self.meta["labels"]
        self.species: list[str] = self.meta["species"]

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        logits = features @ self.coef.T + self.intercept
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def _load(images: list[ImageInput]) -> list:
        loaded = []
        for image in images:
            try:
                loaded.append(load_image(image))
            except Exception:  # unreadable upload or unreachable URL: skip that photo
                continue
        return loaded

    def signature(self, images: list[ImageInput]) -> list[float] | None:
        """The 768 backbone features the classifier reads, averaged over the photos: a description of how the
        animal looks (coat, colour, face shape) that can be compared with cosine similarity."""
        loaded = self._load(images)
        if not loaded:
            return None
        f = image_features(loaded).mean(axis=0)
        return (f / np.linalg.norm(f)).tolist()

    def predict(self, images: list[ImageInput], species: str, top: int = 3) -> list[BreedPrediction]:
        loaded = self._load(images)
        if not loaded:
            return []
        p = self.probabilities(image_features(loaded)).mean(axis=0)
        if species in ("Dog", "Cat"):  # only breeds of the species the reporter chose
            p = np.where(np.array(self.species) == species, p, 0)
            p = p / p.sum()
        best = np.argsort(-p)[:top]
        return [BreedPrediction(self.labels[i], round(float(p[i]) * 100)) for i in best]


@lru_cache
def get_model() -> BreedModel:
    return BreedModel()

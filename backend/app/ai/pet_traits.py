"""M9 trait model: reads a pet's description and predicts six 1-5 ratings that compatibility depends on.

Trained by ``backend/ml/m9_compatibility/train.py`` on 618 dog breeds (descriptions + kennel-club trait ratings,
Kaggle "Dog Breeds: Traits, Origins & Provenance", CC BY 4.0). Text goes through a pretrained sentence-embedding
model; one ridge regression per trait sits on top. Weights live in ``weights/m9_traits.npz``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
MODEL_PATH = WEIGHTS_DIR / "m9_traits.npz"
META_PATH = WEIGHTS_DIR / "m9_traits.meta.json"
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"

TRAITS = ["energy", "children_friendly", "apartment_friendly", "novice_owner_friendly", "good_with_other_dogs", "grooming"]
SIZES = {"Small": 0, "Medium": 1, "Large": 2}
FUR = {"Short": 2.0, "Medium": 3.0, "Long": 4.0}  # on the dataset's 1-5 coat-length scale


@lru_cache
def embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDER, device="cpu")


def embed(texts: list[str]) -> np.ndarray:
    return np.asarray(embedder().encode(texts, normalize_embeddings=True, batch_size=32))


def structured(size: str, fur: str) -> list[float]:
    """Size one-hot plus coat length; unknown fur counts as medium."""
    s = SIZES.get(size, 1)
    return [float(s == 0), float(s == 1), float(s == 2), FUR.get(fur, 3.0)]


class TraitModel:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} is missing. Train it with: python -m ml.m9_compatibility.train")
        w = np.load(MODEL_PATH, allow_pickle=False)
        self.coef, self.intercept = w["coef"], w["intercept"]
        self.meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    def predict(self, texts: list[str], sizes: list[str], furs: list[str]) -> list[dict[str, float]]:
        """Predicted 1-5 rating for each trait, one dict per pet."""
        X = np.hstack([embed(texts), np.array([structured(s, f) for s, f in zip(sizes, furs)])])
        Y = np.clip(X @ self.coef.T + self.intercept, 1, 5)
        return [dict(zip(TRAITS, (round(float(v), 2) for v in row))) for row in Y]


@lru_cache
def get_model() -> TraitModel:
    return TraitModel()

"""M3 adoption speed: XGBoost classifier trained on the PetFinder.my Adoption Prediction dataset.

The features are computed from ``ListingFeatures`` only, so training (``backend/ml/m3_adoption_speed/train.py``)
and the running app build them with the same function. Weights live in ``weights/``.
"""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from .base import AdoptionSpeed, ListingFeatures

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
MODEL_PATH = WEIGHTS_DIR / "m3_adoption_speed.json"
META_PATH = WEIGHTS_DIR / "m3_adoption_speed.meta.json"

FEATURES = [
    "is_cat", "age_months", "size", "is_female", "is_mixed", "is_domestic", "photo_count",
    "desc_chars", "desc_words", "health_words", "fee_tnd", "is_free",
    "vaccinated", "dewormed", "sterilized", "fur_length",
]
FUR = {"Short": 0, "Medium": 1, "Long": 2}
AGE_GROUP_MONTHS = {"young": 6, "adult": 36, "senior": 96}  # when the age text has no number
SIZES = {"Small": 0, "Medium": 1, "Large": 2}
HEALTH_RX = re.compile(r"vaccin|neuter|spay|steril|deworm", re.I)
AGE_RX = re.compile(r"(\d+(?:\.\d+)?)\s*(week|month|year|yr|mo)", re.I)


def age_months(label: str, group: str) -> float:
    m = AGE_RX.search(label or "")
    if not m:
        return AGE_GROUP_MONTHS.get(group, 36)
    n, unit = float(m.group(1)), m.group(2).lower()
    return n * 12 if unit.startswith("y") else n / 4.3 if unit.startswith("w") else n


def parse_fee(fee: str | None) -> float:
    """Fee in TND; 0 for free listings, NaN when unknown (XGBoost handles missing values)."""
    if fee is None or not fee.strip():
        return math.nan
    if "free" in fee.lower():
        return 0.0
    digits = re.sub(r"[^\d.]", "", fee)
    return float(digits) if digits else math.nan


def features(f: ListingFeatures) -> list[float]:
    breed = (f.breed or "").lower()
    text = f.description or ""
    fee = parse_fee(f.fee)
    return [
        1.0 if f.species == "Cat" else 0.0,
        age_months(f.age_label, f.age_group),
        SIZES.get(f.size, 1),
        1.0 if f.sex == "Female" else 0.0,
        1.0 if re.search(r"mix|unknown", breed) else 0.0,
        1.0 if "domestic" in breed else 0.0,
        min(f.photo_count, 30),
        len(text),
        len(text.split()),
        len(HEALTH_RX.findall(text)),
        fee,
        1.0 if fee == 0 else 0.0,
        _yes_no(f.vaccinated),
        _yes_no(f.dewormed),
        _yes_no(f.sterilized),
        FUR.get(f.fur_length, math.nan),
    ]


def _yes_no(value: bool | None) -> float:
    return math.nan if value is None else float(value)


class AdoptionSpeedModel:
    def __init__(self) -> None:
        import xgboost as xgb

        from ..config import get_settings

        name = get_settings().m3_weights
        model_path, meta_path = WEIGHTS_DIR / f"{name}.json", WEIGHTS_DIR / f"{name}.meta.json"
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} is missing. Train it with: python -m ml.m3_adoption_speed.train")
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))

    def _expected(self, rows: list[list[float]]) -> np.ndarray:
        """Expected speed class (probability-weighted average of 0-4), lower is faster."""
        proba = self.model.predict_proba(np.array(rows, dtype=float))
        return proba @ np.arange(proba.shape[1])

    def predict(self, f: ListingFeatures) -> AdoptionSpeed:
        x = features(f)
        i = {name: n for n, name in enumerate(FEATURES)}

        # What-if tips: change one thing the shelter controls and keep the changes the model says help.
        candidates: list[tuple[str, list[float]]] = []
        if x[i["photo_count"]] < 5:
            more = 5 - int(x[i["photo_count"]])
            candidates.append((f"Add {more} more photo{'s' if more > 1 else ''}", _with(x, i, photo_count=5)))
        if x[i["desc_words"]] < 70:
            candidates.append(("Write a fuller description: personality, habits, who they suit",
                               _with(x, i, desc_words=70, desc_chars=max(x[i["desc_chars"]], 400))))
        if x[i["health_words"]] == 0:
            candidates.append(("Mention vaccination and spay/neuter status", _with(x, i, health_words=2)))
        if f.vaccinated is not True or f.dewormed is not True:
            tip = "Vaccinate and deworm before listing" if False in (f.vaccinated, f.dewormed) \
                else "Add vaccination and deworming details"
            candidates.append((tip, _with(x, i, vaccinated=1.0, dewormed=1.0)))
        if not math.isnan(x[i["fee_tnd"]]) and x[i["fee_tnd"]] > 0:
            candidates.append(("Waive or lower the fee", _with(x, i, fee_tnd=0.0, is_free=1.0)))

        scores = self._expected([x] + [row for _, row in candidates])
        base = scores[0]
        speed_class = to_class(base, self.meta["cutoffs"])
        gains = sorted(((base - s, tip) for (tip, _), s in zip(candidates, scores[1:])), reverse=True)
        tips = [tip for gain, tip in gains if gain >= 0.05]
        return AdoptionSpeed(speed_class, tips or ["Listing is in good shape"])


def to_class(expected: float | np.ndarray, cutoffs: list[float]) -> int | np.ndarray:
    """Expected class -> speed class 0-4, using the four cut-offs tuned during training."""
    out = np.digitize(expected, cutoffs)
    return int(out) if np.ndim(out) == 0 else out


def _with(x: list[float], index: dict[str, int], **changes: float) -> list[float]:
    row = list(x)
    for name, value in changes.items():
        row[index[name]] = value
    return row


@lru_cache
def get_model() -> AdoptionSpeedModel:
    return AdoptionSpeedModel()

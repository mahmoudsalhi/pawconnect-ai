"""The team's real models. Enable with ``AI_PROVIDER=app.ai.models:PawModels`` in backend/.env.

Each model overrides only its own method; everything else keeps the DemoProvider placeholder.
``name`` stays "demo" until the matcher (embed / similarity) is replaced, because the Match page
shows "DEMO MODE" based on it.
"""

from __future__ import annotations

import logging

from .age import get_age_model
from .base import AgePrediction, HealthPrediction, ImageInput
from .demo import DemoProvider
from .health import get_health_model

log = logging.getLogger(__name__)


class PawModels(DemoProvider):
    def predict_age(self, images: list[ImageInput], species: str) -> AgePrediction | None:
        model = get_age_model()
        if model is None:  # model files missing or not loadable
            return super().predict_age(images, species)
        try:
            result = model.predict([img.read() for img in images], species)
        except Exception:
            log.exception("M4 age prediction failed; using the placeholder.")
            return super().predict_age(images, species)
        if result is None:  # no readable photo (remote seed photos read as None)
            return super().predict_age(images, species)
        return AgePrediction(group=result.group, confidence=result.confidence)

    def predict_health(self, images: list[ImageInput], species: str) -> HealthPrediction | None:
        if (species or "").lower() != "dog":  # trained on dogs only; skip before reading any photo
            return None
        model = get_health_model()
        if model is None:  # model files missing or not loadable
            return None
        try:
            result = model.predict([img.read() for img in images], species)
        except Exception:
            log.exception("M5 skin prediction failed; no health result.")
            return None
        if result is None:  # no readable photo
            return None
        return HealthPrediction(status=result.status, condition=result.condition, confidence=result.confidence)

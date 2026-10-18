"""PawConnect's trained models. Methods not overridden here keep the DemoProvider placeholders.

Enable with ``AI_PROVIDER=app.ai.models:PawModels`` in ``backend/.env``.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from . import compatibility
from .adoption_speed import get_model as adoption_speed_model
from .age import get_age_model
from .base import AdoptionSpeed, AgePrediction, BreedPrediction, CompatMatch, CompatPet, ImageInput, ListingFeatures, Questionnaire
from .breed import get_model as breed_model
from .demo import DemoProvider
from .pet_traits import embed, get_model as trait_model

log = logging.getLogger(__name__)


class PawModels(DemoProvider):
    name = "pawmodels-v1"

    def predict_breed(self, images: list[ImageInput], species: str, top: int = 3) -> list[BreedPrediction]:
        """M1: ConvNeXt-Tiny features + logistic regression trained on Oxford-IIIT Pet (37 breeds)."""
        return breed_model().predict(images, species, top)

    def predict_age(self, images: list[ImageInput], species: str) -> AgePrediction | None:
        """M4: age group from the report's photos (ONNX)."""
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

    def adoption_speed(self, features: ListingFeatures) -> AdoptionSpeed:
        """M3: XGBoost trained on PetFinder.my adoption listings."""
        return adoption_speed_model().predict(features)

    def compatibility(self, answers: Questionnaire, pets: list[CompatPet]) -> list[CompatMatch]:
        """M9: trait model trained on dog-breed data, plus sentence similarity for the free-text wish."""
        if not pets:
            return []
        model = trait_model()
        ranges = model.meta["label_range"]
        raw = model.predict([p.text for p in pets], [p.size for p in pets], [p.fur_length for p in pets])
        # Read each prediction relative to the spread of real breeds (10th percentile -> 0, 90th -> 1).
        traits = [{k: float(np.clip((v - ranges[k][0]) / (ranges[k][1] - ranges[k][0]), 0, 1)) for k, v in t.items()}
                  for t in raw]
        ideal = _ideal_similarity(answers.ideal, [p.text for p in pets]) if answers.ideal.strip() else None
        return compatibility.score(answers, pets, traits, ideal)


def _ideal_similarity(wish: str, texts: list[str]) -> list[tuple[float, str]]:
    """Best-matching sentence of each pet's text for the adopter's own description, as a 0-1 score."""
    sentences = [[s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if len(s.split()) >= 3] or [t] for t in texts]
    flat = [s for group in sentences for s in group]
    vectors = embed([wish] + flat)
    sims = vectors[1:] @ vectors[0]
    out, i = [], 0
    for group in sentences:
        best = int(np.argmax(sims[i:i + len(group)]))
        out.append((float(np.clip((sims[i + best] - 0.15) / 0.45, 0, 1)), group[best]))
        i += len(group)
    return out

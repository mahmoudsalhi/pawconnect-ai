"""PawConnect's trained models. Methods not overridden here keep the DemoProvider placeholders.

- M1 breed identification, M3 adoption speed, M4 age, M9 compatibility: loaded in-process.
- M2 lost & found photo matching: runs as a small separate service (``ml/photo_matching``) because
  PyTorch is heavy; this class talks to it over HTTP.

Enable in ``backend/.env``::

    AI_PROVIDER=app.ai.models:PawModels
    M2_URL=http://127.0.0.1:8002

How an M2 match score is produced
---------------------------------
1. Each photo becomes a 256-number vector (``POST {M2_URL}/embed``).
2. The similarity of two animals is the best cosine between any photo of one and any photo of the
   other (one good angle is enough; a blurry photo doesn't drag the score down).
3. That cosine is converted to the 0-100 percentage the UI shows with ``CALIBRATION``.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict

import httpx
import numpy as np

from ..config import get_settings
from . import compatibility
from .adoption_speed import get_model as adoption_speed_model
from .age import get_age_model
from .base import (AdoptionSpeed, AgePrediction, BreedPrediction, CompatMatch, CompatPet, ImageInput, ListingFeatures,
                   MatchSubject, ModelUnavailable, Questionnaire)
from .breed import get_model as breed_model
from .demo import DemoProvider
from .pet_traits import embed, get_model as trait_model

log = logging.getLogger(__name__)

# (cosine similarity, percentage shown to users). Measured on held-out DogFaceNet dogs the model
# had never seen: photos of different dogs averaged 0.03 (max 0.41), photos of the same dog 0.54.
# A cosine of ~0.33 gave about 90 % precision, so it maps to 50 % (the "possible match" line used by
# ``match_new_report``). Re-run ``ml/photo_matching/scripts/check_model.py`` after retraining and
# adjust these points.
CALIBRATION: list[tuple[float, int]] = [(-1.0, 0), (0.0, 5), (0.33, 50), (0.55, 72), (0.85, 95), (1.0, 99)]

MAX_PHOTOS_PER_SIDE = 3        # keeps a comparison fast: at most 3 x 3 photo pairs
MAX_REMOTE_BYTES = 10 * 1024 * 1024
CACHE_SIZE = 2048              # embeddings kept in memory (a photo never changes under its URL)


def calibrate(cosine: float) -> int:
    """Piecewise-linear conversion from cosine similarity to a 0-100 percentage."""
    if cosine <= CALIBRATION[0][0]:
        return CALIBRATION[0][1]
    for (x0, y0), (x1, y1) in zip(CALIBRATION, CALIBRATION[1:]):
        if cosine <= x1:
            return round(y0 + (y1 - y0) * (cosine - x0) / (x1 - x0))
    return CALIBRATION[-1][1]


class PawModels(DemoProvider):
    #: Stored next to cached match scores. Change it when you retrain M2 so old scores are recomputed.
    name = "pawmatch-m2-v1"

    def __init__(self, client: httpx.Client | None = None) -> None:
        s = get_settings()
        self._client = client or httpx.Client(base_url=s.m2_url, timeout=s.m2_timeout)
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()

    # -- helpers ------------------------------------------------------------------------------

    def _image_bytes(self, image: ImageInput) -> bytes | None:
        """Bytes of a photo: read from our own storage, or downloaded for remote (seed) photos."""
        data = image.read()
        if data is not None:
            return data
        if not image.url.startswith(("http://", "https://")):
            return None
        try:
            r = httpx.get(image.url, timeout=8, follow_redirects=True)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("Could not download %s: %s", image.url, exc)
            return None
        return r.content if len(r.content) <= MAX_REMOTE_BYTES else None

    def _vector(self, image: ImageInput) -> list[float] | None:
        """256-number embedding of a photo, or None if this particular photo can't be used.

        Raises ``ModelUnavailable`` if the M2 service itself can't be reached.
        """
        with self._lock:
            if image.url in self._cache:
                self._cache.move_to_end(image.url)
                return self._cache[image.url]
        data = self._image_bytes(image)
        if data is None:
            return None
        try:
            r = self._client.post("/embed", files={"photo": ("photo.jpg", data, "image/jpeg")})
        except httpx.HTTPError as exc:
            raise ModelUnavailable(f"M2 service unreachable: {exc}") from exc
        if r.status_code >= 500:
            raise ModelUnavailable(f"M2 service error {r.status_code}")
        if r.status_code != 200:  # e.g. 400: not a readable image
            log.warning("M2 rejected %s (%s)", image.url, r.status_code)
            return None
        vec = r.json()["embedding"]
        with self._lock:
            self._cache[image.url] = vec
            while len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
        return vec

    def _vectors(self, subject: MatchSubject) -> list[list[float]]:
        vecs = (self._vector(img) for img in subject.images[:MAX_PHOTOS_PER_SIDE])
        return [v for v in vecs if v is not None]

    # -- AIProvider ---------------------------------------------------------------------------

    def embed(self, image: ImageInput) -> list[float]:
        """Visual signature for the UI (it draws bars, so values are rescaled from [-1, 1] to [0, 1])."""
        try:
            vec = self._vector(image)
        except ModelUnavailable:
            vec = None
        if vec is None:
            return super().embed(image)  # display only; scores never use this fallback
        return [min(1.0, max(0.0, (x + 1) / 2)) for x in vec]

    def similarity(self, query: MatchSubject, candidate: MatchSubject) -> int:
        if query.species != candidate.species:
            return 0
        q, c = self._vectors(query), self._vectors(candidate)
        if not q or not c:
            return 0  # no usable photo on one side: no visual evidence
        best = max(sum(a * b for a, b in zip(u, v)) for u in q for v in c)
        return max(0, min(100, calibrate(best)))

    # -- Other trained models ------------------------------------------------------------------
    def predict_breed(self, images: list[ImageInput], species: str, top: int = 3) -> list[BreedPrediction]:
        """M1: ConvNeXt-Tiny features + logistic regression trained on Oxford-IIIT Pet (37 breeds)."""
        return breed_model().predict(images, species, top)

    def look_signature(self, images: list[ImageInput]) -> list[float] | None:
        """M1 look-alikes: the ConvNeXt features the breed classifier reads."""
        return breed_model().signature(images)

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

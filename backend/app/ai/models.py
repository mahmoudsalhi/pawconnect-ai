"""PawConnect AI models that are really trained. Today: M2, lost & found photo matching.

``PawModels`` inherits ``DemoProvider``, so every model that isn't overridden here (M1, M3-M6)
keeps its placeholder behaviour. M2 runs as a small separate service (``ml/photo_matching``)
because PyTorch is heavy; this class talks to it over HTTP.

Enable it in ``backend/.env``::

    AI_PROVIDER=app.ai.models:PawModels
    M2_URL=http://127.0.0.1:8002

How a score is produced
-----------------------
1. Each photo becomes a 256-number vector (``POST {M2_URL}/embed``).
2. The similarity of two animals is the best cosine between any photo of one and any photo of the
   other (one good angle is enough; a blurry photo doesn't drag the score down).
3. That cosine is converted to the 0-100 percentage the UI shows with ``CALIBRATION``.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict

import httpx

from ..config import get_settings
from .base import ImageInput, MatchSubject, ModelUnavailable
from .demo import DemoProvider

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
    #: Stored next to cached match scores. Change it when you retrain so old scores are recomputed.
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

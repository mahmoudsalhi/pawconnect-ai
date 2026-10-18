"""AI face login rules on top of face_engine: enrolment, liveness (head turn) and the 1:1 face match.

Thresholds come from models/face/face_config.json, produced by ml/face/calibrate.py on LFW.
If the models are missing, face login reports itself unavailable instead of failing the app.
"""

import json
import logging
from functools import lru_cache

import numpy as np

from .face_engine import MODEL_DIR, FaceEngine, FaceError, FaceResult, cosine, decode_image

log = logging.getLogger(__name__)

MODEL_VERSION = "yunet-2023mar+sface-2021dec"
DEFAULTS = {"match_threshold": 0.363, "frontal_max_yaw": 0.15, "turn_min_yaw": 0.25, "same_session_min_similarity": 0.25}
TURN_FROM_FRONT = 0.2  # the turned photo must also differ from the facing photo by this much


class FaceUnavailable(Exception):
    pass


@lru_cache
def get_engine() -> FaceEngine | None:
    try:
        engine = FaceEngine()
        log.info("Face engine loaded (%s)", MODEL_VERSION)
        return engine
    except Exception as exc:  # models not downloaded, OpenCV missing...
        log.warning("Face login unavailable: %s (run: python app/ai/face_engine.py --download)", exc)
        return None


@lru_cache
def get_config() -> dict:
    try:
        return {**DEFAULTS, **json.loads((MODEL_DIR / "face_config.json").read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return dict(DEFAULTS)


def analyze_image(data: str) -> FaceResult:
    engine = get_engine()
    if engine is None:
        raise FaceUnavailable()
    return engine.analyze(decode_image(data))


def enrolment_embedding(images: list[str]) -> np.ndarray:
    """Several photos facing the camera -> one averaged embedding. Raises FaceError if unusable."""
    cfg = get_config()
    results = [analyze_image(img) for img in images]
    if any(abs(r.yaw) > cfg["frontal_max_yaw"] * 1.5 for r in results):
        raise FaceError("not_frontal", "Look straight at the camera while the photos are taken.")
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            if cosine(results[i].embedding, results[j].embedding) < cfg["match_threshold"]:
                raise FaceError("not_consistent", "The photos didn’t look like the same face. Hold still and try again.")
    mean = np.mean([r.embedding for r in results], axis=0)
    return (mean / np.linalg.norm(mean)).astype(np.float32)


def check_liveness(front: str, turned: list[str], direction: str) -> FaceResult:
    """Photo 1 faces the camera; one of the next photos is clearly turned the asked way, same person.

    A printed photo or a still image on a screen cannot turn: tilting it squeezes the eyes and the nose
    together, so the head-turn measure barely changes. Returns the facing photo's analysis.
    """
    cfg = get_config()
    sign = 1 if direction == "left" else -1  # yaw > 0 = turned to the person's own left
    f = analyze_image(front)
    if abs(f.yaw) > cfg["frontal_max_yaw"]:
        raise FaceError("not_frontal", "For the first photo, look straight at the camera.")
    turns = []
    for img in turned:
        try:
            turns.append(analyze_image(img))
        except FaceError:
            continue  # a blurry frame mid-turn is fine as long as another frame is usable
    best = max(turns, key=lambda r: sign * r.yaw, default=None)
    if best is None or sign * best.yaw < cfg["turn_min_yaw"] or sign * (best.yaw - f.yaw) < TURN_FROM_FRONT:
        raise FaceError("no_turn", f"We couldn’t see your head turn to the {direction}. Try again and turn a bit more.")
    if cosine(f.embedding, best.embedding) < cfg["same_session_min_similarity"]:
        raise FaceError("not_same_person", "The photos didn’t show the same person. Try again.")
    return f


def matches(probe: np.ndarray, enrolled: bytes) -> tuple[bool, float]:
    score = cosine(probe, np.frombuffer(enrolled, dtype=np.float32))
    return score >= get_config()["match_threshold"], score

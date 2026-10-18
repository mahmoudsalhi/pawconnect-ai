"""M6 scam detector: loads the model trained in ml/m6 (train_final.py) and scores message text.

The model file is loaded once, on first use. If it is missing or cannot be loaded, every message
scores level "none" and the reason is logged: a broken detector must never stop people messaging.
"""

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from .base import ScamAssessment
from .scam_text import prepare

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "m6_scam.joblib"

REASON_TEXT = {
    "upfront_payment": "Asks for money before the pet is returned or handed over.",
    "fake_fee": "Asks you to pay a fee in advance (adoption, vaccine, transport, deposit…).",
    "off_platform": "Pushes you to leave PawConnect and talk only somewhere else.",
    "phishing_link": "Asks for a code, password, card or bank details, or to log in through a link.",
    "impersonation": "Claims to be a vet, shelter, official or PawConnect staff to get money or data.",
    "pressure": "Uses urgency or threats to rush your decision.",
}


@lru_cache
def _load() -> dict | None:
    try:
        import joblib

        model = joblib.load(MODEL_PATH)
        log.info("Scam detector %s loaded (%s rows)", model["version"], model["rows"])
        return model
    except Exception as exc:  # missing file, missing scikit-learn, version mismatch...
        log.warning("Scam detector unavailable, messages will not be checked: %s", exc)
        return None


def assess(text: str) -> ScamAssessment:
    model = _load()
    if model is None or not (text or "").strip():
        return ScamAssessment(score=0.0, level="none", model=model["version"] if model else "unavailable")

    prepared = prepare(text)
    score = float(model["scam_model"].predict_proba([prepared])[0, 1])
    thresholds = model["thresholds"]
    level = "high" if score >= thresholds["high"] else "caution" if score >= thresholds["caution"] else "none"
    if level == "none":
        return ScamAssessment(score=round(score, 3), level=level, model=model["version"])

    probs = model["reason_model"].predict_proba([prepared])[0]
    order = np.argsort(-probs)
    labels = [model["reasons"][i] for i in order if probs[i] >= 0.5] or [model["reasons"][order[0]]]
    return ScamAssessment(
        score=round(score, 3),
        level=level,
        labels=labels,
        reasons=[REASON_TEXT[label] for label in labels],
        model=model["version"],
    )

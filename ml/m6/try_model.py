"""Try the saved M6 model on your own messages, without the app.

    ml\\.venv\\Scripts\\python ml\\m6\\try_model.py                      # type messages one by one
    ml\\.venv\\Scripts\\python ml\\m6\\try_model.py "Ab3ath 50 dt 9bal"   # or pass them as arguments

Uses the exact model file and text preparation the app uses (backend/app/ai/).
"""

import sys
from pathlib import Path

import joblib
import numpy as np

from textnorm import prepare

MODEL = Path(__file__).resolve().parents[2] / "backend" / "app" / "ai" / "models" / "m6_scam.joblib"
REASON_TEXT = {
    "upfront_payment": "asks for money before the pet is returned",
    "fake_fee": "asks for a fee in advance",
    "off_platform": "pushes you to talk only outside PawConnect",
    "phishing_link": "asks for a code, password, card/bank details or a login link",
    "impersonation": "claims to be a vet, shelter, official or PawConnect staff",
    "pressure": "uses urgency or threats",
}

m = joblib.load(MODEL)
high, caution = m["thresholds"]["high"], m["thresholds"]["caution"]


def check(text: str) -> None:
    prepared = prepare(text)
    score = m["scam_model"].predict_proba([prepared])[0, 1]
    if score >= high:
        verdict = "LIKELY SCAM"
    elif score >= caution:
        verdict = "BE CAREFUL"
    else:
        verdict = "OK, no warning"
    print(f"\n  score {score:.2f}  ->  {verdict}")
    if score >= caution:
        probs = m["reason_model"].predict_proba([prepared])[0]
        order = np.argsort(-probs)
        labels = [m["reasons"][i] for i in order if probs[i] >= 0.5] or [m["reasons"][order[0]]]
        for label in labels:
            print(f"    - {REASON_TEXT[label]}")
    print(f"  (model saw: {prepared!r})\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"M6 model {m['version']} · trained on {m['rows']} messages · warns from {caution:.2f}, strong warning from {high:.2f}")
    if len(sys.argv) > 1:
        for text in sys.argv[1:]:
            print(f"\n> {text}")
            check(text)
    else:
        print("Type a message and press Enter (empty line to quit).")
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                break
            if not text:
                break
            check(text)

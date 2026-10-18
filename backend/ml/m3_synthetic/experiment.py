"""EXPERIMENT: train M3 on a SIMULATED dataset and compare it with the real PetFinder model.

The listings and their adoption speeds below are generated from assumptions written in this file
(ASSUMPTIONS and NOISE). They are not real adoptions. The app keeps using the real model in
app/ai/weights/; this script never touches it.

Run from ``backend/``:  .venv\\Scripts\\python -m ml.m3_synthetic.experiment
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from app.ai.adoption_speed import FEATURES, WEIGHTS_DIR, features, to_class
from app.ai.base import ListingFeatures
from ml.m3_adoption_speed.train import DATA_DIR as REAL_DATA_DIR, fit, qwk, to_listing

OUT_DIR = __import__("pathlib").Path(__file__).resolve().parent / "data"
N_LISTINGS = 12000
RNG_SEED = 7

# How much each property changes the "slowness" score (negative = adopted faster). These are guesses.
ASSUMPTIONS = {
    "young (under 1 year)": -0.8,
    "senior (7+ years)": +0.6,
    "mixed breed": +0.4,
    "per photo, up to 5": -0.12,
    "per 10 description words, up to 80": -0.06,
    "fee per 50 TND": +0.15,
    "vaccinated": -0.25,
    "sterilized": -0.10,
    "long fur": -0.20,
    "large size": +0.20,
}
# Random variation for everything a listing doesn't show (luck, timing, who visits). Higher = harder to predict.
NOISE = 0.6


def generate(n: int, noise: float, rng: np.random.Generator) -> tuple[list[ListingFeatures], np.ndarray]:
    a = ASSUMPTIONS
    listings, scores = [], []
    for _ in range(n):
        species = rng.choice(["Dog", "Cat"])
        age = float(rng.choice([2, 3, 4, 6, 9, 12, 18, 24, 36, 48, 60, 84, 96, 120]))
        breed_kind = rng.choice(["purebred", "mixed", "domestic"] if species == "Cat" else ["purebred", "mixed"])
        breed = {"purebred": "Persian" if species == "Cat" else "Beagle", "mixed": "Mixed breed",
                 "domestic": "Domestic shorthair"}[breed_kind]
        photos = int(rng.integers(0, 9))
        words = int(rng.integers(5, 150))
        fee = float(rng.choice([0, 0, 0, 20, 50, 100]))
        vaccinated, sterilized = bool(rng.random() < 0.6), bool(rng.random() < 0.4)
        fur = str(rng.choice(["Short", "Medium", "Long"]))
        size = str(rng.choice(["Small", "Medium", "Large"]))

        s = 0.0
        s += a["young (under 1 year)"] * (age < 12) + a["senior (7+ years)"] * (age >= 84)
        s += a["mixed breed"] * (breed_kind == "mixed")
        s += a["per photo, up to 5"] * min(photos, 5)
        s += a["per 10 description words, up to 80"] * min(words, 80) / 10
        s += a["fee per 50 TND"] * fee / 50
        s += a["vaccinated"] * vaccinated + a["sterilized"] * sterilized
        s += a["long fur"] * (fur == "Long") + a["large size"] * (size == "Large")
        s += rng.normal(0, noise)

        listings.append(ListingFeatures(
            species=species, age_group="young" if age < 12 else "senior" if age >= 84 else "adult", breed=breed,
            photo_count=photos, description=" ".join(["word"] * words), fee="Free" if fee == 0 else f"{int(fee)} TND",
            size=size, age_label=f"{int(age)} months", sex=str(rng.choice(["Male", "Female"])),
            vaccinated=vaccinated, dewormed=vaccinated, sterilized=sterilized, fur_length=fur,
        ))
        scores.append(s)
    # Balanced classes: each of the 5 speed classes gets 20% of the listings.
    labels = np.digitize(scores, np.quantile(scores, [0.2, 0.4, 0.6, 0.8]))
    return listings, labels


def run(noise: float, real_X: np.ndarray, real_y: np.ndarray, save: bool = False) -> dict:
    listings, y = generate(N_LISTINGS, noise, np.random.default_rng(RNG_SEED))
    X = np.array([features(f) for f in listings], dtype=float)
    if save:
        OUT_DIR.mkdir(exist_ok=True)
        pd.DataFrame([{**f.__dict__, "description_words": len(f.description.split()), "description": None,
                       "AdoptionSpeed": int(c)} for f, c in zip(listings, y)]
                     ).drop(columns="description").to_csv(OUT_DIR / "synthetic_listings.csv", index=False)
    model, cutoffs, _, test_idx = fit(X, y, np.arange(len(y)), seed=42)
    syn_pred = to_class(model.predict_proba(X[test_idx]) @ np.arange(5), cutoffs)
    real_pred = to_class(model.predict_proba(real_X) @ np.arange(5), cutoffs)
    r = {
        "syn_acc": accuracy_score(y[test_idx], syn_pred), "syn_qwk": qwk(y[test_idx], syn_pred),
        "real_acc": accuracy_score(real_y, real_pred), "real_qwk": qwk(real_y, real_pred),
    }
    if save:
        # Loadable by the app with M3_WEIGHTS=m3_synthetic in backend/.env (the real model stays in place).
        model.save_model(WEIGHTS_DIR / "m3_synthetic.json")
        (WEIGHTS_DIR / "m3_synthetic.meta.json").write_text(json.dumps({
            "model": "XGBoost multi-class (multi:softprob)",
            "dataset": "SIMULATED listings from ml/m3_synthetic/experiment.py (not real adoptions)",
            "assumptions": ASSUMPTIONS, "noise": noise, "features": FEATURES, "cutoffs": cutoffs,
            "metrics": {k: round(float(v), 4) for k, v in r.items()},
        }, indent=2), encoding="utf-8")
    return r


def main() -> None:
    real = pd.read_csv(REAL_DATA_DIR / "train.csv")
    real = real[real.Quantity == 1].reset_index(drop=True)
    names = pd.read_csv(REAL_DATA_DIR / "breed_labels.csv")
    breeds = {**dict(zip(names.BreedID, names.BreedName)), 307: "Mixed breed"}
    real_X = np.array([features(to_listing(r, breeds)) for r in real.itertuples()], dtype=float)
    real_y = real.AdoptionSpeed.to_numpy()

    r = run(NOISE, real_X, real_y, save=True)
    print(f"SIMULATED dataset: {N_LISTINGS} listings, balanced (20% per class), noise {NOISE}")
    print(f"Saved to {OUT_DIR / 'synthetic_listings.csv'}\n")
    print("Model trained on simulated data:")
    print(f"  tested on simulated data:      accuracy {r['syn_acc']:.1%}   QWK {r['syn_qwk']:.3f}")
    print(f"  tested on REAL PetFinder data: accuracy {r['real_acc']:.1%}   QWK {r['real_qwk']:.3f}")
    print("Real PetFinder model (for comparison): accuracy ~33%, QWK 0.337 (average of 5 splits)\n")

    print("Same experiment with different amounts of noise (the score is whatever the noise setting allows):")
    for noise in (0.1, 0.3, 0.6, 1.0, 2.0):
        r = run(noise, real_X, real_y)
        print(f"  noise {noise:<4}  simulated: accuracy {r['syn_acc']:.1%}, QWK {r['syn_qwk']:.3f}   |   "
              f"real data: accuracy {r['real_acc']:.1%}, QWK {r['real_qwk']:.3f}")


if __name__ == "__main__":
    main()

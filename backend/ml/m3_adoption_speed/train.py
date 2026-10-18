"""Train M3 (adoption speed) on the PetFinder.my Adoption Prediction dataset.

Run from ``backend/``:  .venv\\Scripts\\python -m ml.m3_adoption_speed.train

Needs a Kaggle API token in ~/.kaggle/kaggle.json and the competition rules accepted on kaggle.com.
Writes the model to app/ai/weights/ and prints the evaluation.
"""

from __future__ import annotations

import json
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

from app.ai.adoption_speed import FEATURES, META_PATH, MODEL_PATH, features, to_class
from app.ai.base import ListingFeatures

COMPETITION = "petfinder-adoption-prediction"
DATA_DIR = Path(__file__).resolve().parent / "data"
MYR_TO_TND = 0.70  # PetFinder fees are in Malaysian ringgit; approximate conversion
SIZES = {1: "Small", 2: "Medium", 3: "Large", 4: "Large"}
SEXES = {1: "Male", 2: "Female"}
YES_NO = {1: True, 2: False}  # 3 = not sure
FUR = {1: "Short", 2: "Medium", 3: "Long"}
SPEED = ["Same day", "Within a week", "Within a month", "1-3 months", "100+ days"]
SEED = 42
CV_SPLITS = 5


def download() -> None:
    if (DATA_DIR / "train.csv").exists() and (DATA_DIR / "breed_labels.csv").exists():
        return
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("train/train.csv", "breed_labels.csv"):
        api.competition_download_file(COMPETITION, name, path=DATA_DIR, quiet=False)
    for z in DATA_DIR.glob("*.zip"):
        with zipfile.ZipFile(z) as archive:
            archive.extractall(DATA_DIR)
        z.unlink()


def age_group(months: int) -> str:
    # Same cut-offs as the app's listings: up to a year is young, 7 years and up is senior.
    return "young" if months <= 12 else "senior" if months >= 84 else "adult"


def qwk(y_true, y_pred) -> float:
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def tune_cutoffs(expected: np.ndarray, y: np.ndarray) -> list[float]:
    """Cut-offs on the expected class that maximise QWK: start from the true class shares, then nudge each one."""
    shares = np.cumsum(np.bincount(y, minlength=5))[:-1] / len(y)
    cutoffs = list(np.quantile(expected, shares))
    for _ in range(3):
        for k in range(4):
            lo = cutoffs[k - 1] if k else expected.min()
            hi = cutoffs[k + 1] if k < 3 else expected.max()
            grid = np.linspace(lo, hi, 60)
            best = max(grid, key=lambda c: qwk(y, to_class(expected, cutoffs[:k] + [c] + cutoffs[k + 1:])))
            cutoffs[k] = float(best)
    return [round(c, 4) for c in cutoffs]


def to_listing(row: pd.Series, breeds: dict[int, str]) -> ListingFeatures:
    breed = breeds.get(row.Breed1, "Unknown")
    if row.Breed2 and row.Breed2 != row.Breed1:
        breed += " mix"
    return ListingFeatures(
        species="Dog" if row.Type == 1 else "Cat",
        age_group=age_group(row.Age),
        breed=breed,
        photo_count=int(row.PhotoAmt),
        description=row.Description if isinstance(row.Description, str) else "",
        fee="Free" if row.Fee == 0 else f"{round(row.Fee * MYR_TO_TND)} TND",
        size=SIZES.get(row.MaturitySize, "Medium"),
        age_label=f"{row.Age} months",
        sex=SEXES.get(row.Gender, "Unknown"),
        vaccinated=YES_NO.get(row.Vaccinated),
        dewormed=YES_NO.get(row.Dewormed),
        sterilized=YES_NO.get(row.Sterilized),
        fur_length=FUR.get(row.FurLength, ""),
    )


def fit(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int):
    """Train on one rescuer-grouped split. Returns the model, its tuned cut-offs and the held-out test rows."""
    # Split by rescuer so the same rescuer's listings never appear in both train and test.
    train_idx, test_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(X, y, groups))
    fit_rel, val_rel = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed)
                            .split(X[train_idx], y[train_idx], groups[train_idx]))
    fit_idx, val_idx = train_idx[fit_rel], train_idx[val_rel]
    model = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, objective="multi:softprob", eval_metric="mlogloss", early_stopping_rounds=100,
        random_state=SEED,
    )
    model.fit(X[fit_idx], y[fit_idx], eval_set=[(X[val_idx], y[val_idx])], verbose=False)
    cutoffs = tune_cutoffs(model.predict_proba(X[val_idx]) @ np.arange(5), y[val_idx])
    return model, cutoffs, train_idx, test_idx


def main() -> None:
    download()
    df = pd.read_csv(DATA_DIR / "train.csv")
    df = df[df.Quantity == 1].reset_index(drop=True)  # PawConnect listings are always one animal
    breed_names = pd.read_csv(DATA_DIR / "breed_labels.csv")
    breeds = dict(zip(breed_names.BreedID, breed_names.BreedName))
    breeds[307] = "Mixed breed"

    X = np.array([features(to_listing(r, breeds)) for r in df.itertuples()], dtype=float)
    y = df.AdoptionSpeed.to_numpy()
    groups = df.RescuerID.to_numpy()
    print(f"{len(df)} listings, {X.shape[1]} features. Class counts: {np.bincount(y).tolist()}")

    # One split is noisy (about +/-0.02 QWK), so also report the average over several rescuer-grouped splits.
    cv = []
    for seed in range(CV_SPLITS):
        m, cut, _, te = fit(X, y, groups, seed)
        cv.append(qwk(y[te], to_class(m.predict_proba(X[te]) @ np.arange(5), cut)))

    model, cutoffs, train_idx, test_idx = fit(X, y, groups, SEED)
    proba = model.predict_proba(X[test_idx])
    pred = to_class(proba @ np.arange(5), cutoffs)  # same rule as AdoptionSpeedModel.predict
    argmax = proba.argmax(1)
    y_test = y[test_idx]
    majority = np.bincount(y[train_idx]).argmax()
    metrics = {
        "test_listings": int(len(test_idx)),
        "qwk": round(qwk(y_test, pred), 4),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "argmax_qwk": round(qwk(y_test, argmax), 4),
        "argmax_accuracy": round(accuracy_score(y_test, argmax), 4),
        "baseline_accuracy": round(float((y_test == majority).mean()), 4),
        "within_one_class": round(float((np.abs(pred - y_test) <= 1).mean()), 4),
        "qwk_cv_mean": round(float(np.mean(cv)), 4),
        "qwk_cv_std": round(float(np.std(cv)), 4),
        "cv_splits": CV_SPLITS,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    importance = dict(zip(FEATURES, (round(float(v), 4) for v in model.feature_importances_)))
    META_PATH.write_text(json.dumps({
        "model": "XGBoost multi-class (multi:softprob)",
        "dataset": f"Kaggle {COMPETITION} train.csv",
        "trained_on": date.today().isoformat(),
        "training_listings": int(len(train_idx)),
        "best_iteration": int(model.best_iteration),
        "features": FEATURES,
        "cutoffs": cutoffs,
        "myr_to_tnd": MYR_TO_TND,
        "metrics": metrics,
        "feature_importance": dict(sorted(importance.items(), key=lambda kv: -kv[1])),
    }, indent=2), encoding="utf-8")

    print(f"\nAverage over {CV_SPLITS} splits: QWK {metrics['qwk_cv_mean']} +/- {metrics['qwk_cv_std']}")
    print(f"\nShipped model's test set: {metrics['test_listings']} listings from rescuers not seen in training")
    print(f"  Quadratic weighted kappa: {metrics['qwk']}  (the Kaggle competition metric)")
    print(f"  Accuracy:                 {metrics['accuracy']}  (always guessing '{SPEED[majority]}': "
          f"{metrics['baseline_accuracy']})")
    print(f"  Within one class:         {metrics['within_one_class']}")
    print(f"  Argmax rule instead:      QWK {metrics['argmax_qwk']}, accuracy {metrics['argmax_accuracy']}")
    print("\nConfusion matrix (rows = true class, columns = predicted):")
    print(confusion_matrix(y_test, pred, labels=range(5)))
    print("\nFeature importance:", json.dumps(importance, indent=2))
    print(f"\nSaved {MODEL_PATH}")


if __name__ == "__main__":
    main()

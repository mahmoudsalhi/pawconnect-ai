"""Train the M9 trait model on real dog-breed data.

Dataset: Kaggle "jesusic/dog-breeds-traits-origins-provenance" (CC BY 4.0): 629 breeds with a short description,
temperament words and 1-5 ratings reviewed against kennel-club sources (AKC, FCI, ...).

Run from ``backend/``:  .venv\\Scripts\\python -m ml.m9_compatibility.train
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold

from app.ai.pet_traits import EMBEDDER, META_PATH, MODEL_PATH, TRAITS, embed, structured

DATASET = "jesusic/dog-breeds-traits-origins-provenance"
DATA_DIR = Path(__file__).resolve().parent / "data"
ALPHAS = [0.3, 1, 3, 10, 30]
SIZE = {"toy": "Small", "small": "Small", "medium": "Medium", "large": "Large", "giant": "Large"}
SEED = 42


def download() -> None:
    if (DATA_DIR / "dog_breeds.csv").exists():
        return
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file(DATASET, "dog_breeds.csv", path=DATA_DIR)


def fur(coat_length: float) -> str:
    return "Short" if coat_length <= 2 else "Medium" if coat_length == 3 else "Long"


def main() -> None:
    download()
    df = pd.read_csv(DATA_DIR / "dog_breeds.csv")
    df = df[df.review_status == "reviewed"].dropna(subset=TRAITS).reset_index(drop=True)
    texts = (df.description.fillna("") + " Temperament: " + df.temperament.fillna("") + ".").tolist()
    print(f"{len(df)} reviewed breeds. Embedding descriptions with {EMBEDDER}...")
    X = np.hstack([embed(texts), np.array([structured(SIZE.get(s, "Medium"), fur(c))
                                           for s, c in zip(df.size_category, df.coat_length)])])
    Y = df[TRAITS].to_numpy(float)

    # 5-fold cross-validation over breeds: every breed is predicted by a model that never saw it.
    pred, base = np.zeros_like(Y), np.zeros_like(Y)
    for train, test in KFold(5, shuffle=True, random_state=SEED).split(X):
        alpha = RidgeCV(alphas=ALPHAS).fit(X[train], Y[train]).alpha_
        pred[test] = np.clip(Ridge(alpha=alpha).fit(X[train], Y[train]).predict(X[test]), 1, 5)
        base[test] = Y[train].mean(0)  # baseline: always predict the average rating

    metrics = {}
    print(f"\n5-fold cross-validation ({len(df)} breeds, each tested by a model that never saw it)")
    print(f"{'trait':24} {'MAE':>6} {'baseline':>9} {'exact':>7} {'baseline':>9} {'within 1':>9} {'spearman':>9}")
    for k, t in enumerate(TRAITS):
        y, p, b = Y[:, k], pred[:, k], base[:, k]
        m = {
            "mae": round(float(np.abs(p - y).mean()), 3),
            "baseline_mae": round(float(np.abs(b - y).mean()), 3),
            "exact_accuracy": round(float((np.rint(p) == y).mean()), 3),
            "baseline_exact_accuracy": round(float((np.rint(b) == y).mean()), 3),
            "within_one": round(float((np.abs(np.rint(p) - y) <= 1).mean()), 3),
            "spearman": round(float(spearmanr(p, y).statistic), 3),
        }
        metrics[t] = m
        print(f"{t:24} {m['mae']:6.2f} {m['baseline_mae']:9.2f} {m['exact_accuracy']:7.0%} "
              f"{m['baseline_exact_accuracy']:9.0%} {m['within_one']:9.0%} "
              f"{m['spearman']:9.2f}")
    overall = {
        "mae": round(float(np.mean([m["mae"] for m in metrics.values()])), 3),
        "baseline_mae": round(float(np.mean([m["baseline_mae"] for m in metrics.values()])), 3),
        "exact_accuracy": round(float(np.mean([m["exact_accuracy"] for m in metrics.values()])), 3),
        "baseline_exact_accuracy": round(float(np.mean([m["baseline_exact_accuracy"] for m in metrics.values()])), 3),
        "within_one": round(float(np.mean([m["within_one"] for m in metrics.values()])), 3),
    }
    print(f"{'AVERAGE':24} {overall['mae']:6.2f} {overall['baseline_mae']:9.2f} {overall['exact_accuracy']:7.0%} "
          f"{overall['baseline_exact_accuracy']:9.0%} {overall['within_one']:9.0%}")

    # Final model on all breeds.
    alpha = RidgeCV(alphas=ALPHAS).fit(X, Y).alpha_
    final = Ridge(alpha=alpha).fit(X, Y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(MODEL_PATH, coef=final.coef_, intercept=final.intercept_)
    META_PATH.write_text(json.dumps({
        "model": f"Ridge regression (alpha={alpha}) on {EMBEDDER} sentence embeddings + size + coat length",
        "dataset": f"Kaggle {DATASET} (CC BY 4.0), reviewed breeds",
        "trained_on": date.today().isoformat(),
        "training_breeds": len(df),
        "traits": TRAITS,
        "cv_folds": 5,
        "metrics": metrics,
        "overall": overall,
        # 10th and 90th percentile of the real ratings: the matcher reads predictions relative to this spread.
        "label_range": {t: [float(np.percentile(Y[:, k], 10)), float(np.percentile(Y[:, k], 90))]
                        for k, t in enumerate(TRAITS)},
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {MODEL_PATH}")


if __name__ == "__main__":
    main()

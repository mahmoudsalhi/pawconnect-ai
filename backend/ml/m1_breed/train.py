"""Train M1 (breed identification) on the Oxford-IIIT Pet dataset.

Dataset: Oxford-IIIT Pet (Parkhi et al., 2012, CC BY-SA 4.0): 7,349 photos of 37 breeds (25 dogs, 12 cats) with an
official trainval/test split. Download images.tar.gz and annotations.tar.gz from
https://www.robots.ox.ac.uk/~vgg/data/pets/ and extract them into ml/m1_breed/data/.

Run from ``backend/``:  .venv\\Scripts\\python -m ml.m1_breed.train
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, top_k_accuracy_score
from sklearn.model_selection import cross_val_score

from app.ai.breed import META_PATH, MODEL_PATH, image_features

DATA_DIR = Path(__file__).resolve().parent / "data"
FEATURES = DATA_DIR / "features.npz"  # cache: extracting features is the slow part
BATCH = 32
C_VALUES = [0.5, 1, 2, 5, 10, 20]


def label_of(image_name: str) -> str:
    """'german_shorthaired_123' -> 'German shorthaired'; cat breeds keep their capitals ('British_Shorthair')."""
    breed = image_name.rsplit("_", 1)[0].replace("_", " ")
    return breed[0].upper() + breed[1:]


def read_split(name: str) -> list[tuple[str, str, str]]:
    rows = []
    for line in (DATA_DIR / "annotations" / f"{name}.txt").read_text().splitlines():
        image, _, species_id, _ = line.split()
        rows.append((image, label_of(image), "Cat" if species_id == "1" else "Dog"))
    return rows


def extract(rows: list[tuple[str, str, str]], what: str) -> np.ndarray:
    out, start = [], time.time()
    for i in range(0, len(rows), BATCH):
        images = [Image.open(DATA_DIR / "images" / f"{image}.jpg") for image, _, _ in rows[i:i + BATCH]]
        out.append(image_features(images))
        done = min(i + BATCH, len(rows))
        if done % (BATCH * 20) == 0 or done == len(rows):
            print(f"  {what}: {done}/{len(rows)} photos ({time.time() - start:.0f}s)")
    return np.vstack(out)


def main() -> None:
    train_rows, test_rows = read_split("trainval"), read_split("test")
    labels = sorted({label for _, label, _ in train_rows})
    species = [next(s for _, l, s in train_rows if l == label) for label in labels]
    y_train = np.array([labels.index(l) for _, l, _ in train_rows])
    y_test = np.array([labels.index(l) for _, l, _ in test_rows])
    print(f"{len(train_rows)} training photos, {len(test_rows)} test photos, {len(labels)} breeds "
          f"({species.count('Dog')} dogs, {species.count('Cat')} cats)")

    if FEATURES.exists():
        cached = np.load(FEATURES)
        X_train, X_test = cached["train"], cached["test"]
    else:
        print("Extracting ConvNeXt-Tiny features (pretrained on ImageNet, frozen)...")
        X_train, X_test = extract(train_rows, "train"), extract(test_rows, "test")
        np.savez(FEATURES, train=X_train, test=X_test)

    # Choose regularisation with 5-fold cross-validation on the training photos only.
    scores = {c: cross_val_score(LogisticRegression(C=c, max_iter=3000), X_train, y_train, cv=5).mean() for c in C_VALUES}
    best_c = max(scores, key=scores.get)
    print("Cross-validation accuracy by C:", {c: round(s, 4) for c, s in scores.items()}, "-> C =", best_c)

    start = time.time()
    clf = LogisticRegression(C=best_c, max_iter=3000).fit(X_train, y_train)
    train_seconds = time.time() - start
    proba = clf.predict_proba(X_test)
    pred = proba.argmax(1)
    cm = confusion_matrix(y_test, pred)
    np.fill_diagonal(cm, 0)
    confusions = sorted(((int(cm[i, j]), labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels))
                         if cm[i, j]), reverse=True)[:5]
    per_breed = {labels[k]: round(float((pred[y_test == k] == k).mean()), 3) for k in range(len(labels))}
    metrics = {
        "test_photos": len(test_rows),
        "top1_accuracy": round(accuracy_score(y_test, pred), 4),
        "top3_accuracy": round(top_k_accuracy_score(y_test, proba, k=3), 4),
        "species_accuracy": round(float((np.array(species)[pred] == np.array(species)[y_test]).mean()), 4),
        "baseline_accuracy": round(float((y_test == np.bincount(y_train).argmax()).mean()), 4),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(MODEL_PATH, coef=clf.coef_, intercept=clf.intercept_)
    META_PATH.write_text(json.dumps({
        "model": f"ConvNeXt-Tiny (ImageNet, frozen) features + logistic regression (C={best_c})",
        "dataset": "Oxford-IIIT Pet (official trainval/test split), CC BY-SA 4.0",
        "trained_on": date.today().isoformat(),
        "training_photos": len(train_rows),
        "classifier_training_seconds": round(train_seconds, 1),
        "labels": labels,
        "species": species,
        "metrics": metrics,
        "most_confused": [{"count": n, "true": a, "predicted": b} for n, a, b in confusions],
        "per_breed_accuracy": per_breed,
    }, indent=2), encoding="utf-8")

    print(f"\nOfficial test set: {metrics['test_photos']} photos the model never saw")
    print(f"  Top-1 accuracy (first guess right):      {metrics['top1_accuracy']:.1%}")
    print(f"  Top-3 accuracy (right breed in top 3):   {metrics['top3_accuracy']:.1%}")
    print(f"  Cat vs dog correct:                      {metrics['species_accuracy']:.1%}")
    print(f"  Baseline (always the most common breed): {metrics['baseline_accuracy']:.1%}")
    print("  Most confused:", "; ".join(f"{a} -> {b} ({n})" for n, a, b in confusions))
    worst = sorted(per_breed.items(), key=lambda kv: kv[1])[:3]
    print("  Hardest breeds:", ", ".join(f"{b} {a:.0%}" for b, a in worst))
    print(f"\nSaved {MODEL_PATH}")


if __name__ == "__main__":
    main()

"""Shared constants and helpers for M5 · Dog skin problem detection."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

# Fixed class order: the model output index = position in this list.
CLASSES = ["healthy", "dermatitis", "fungal_infections", "hypersensitivity", "demodicosis", "ringworm"]
HEALTHY = 0
SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEFAULT_THRESHOLD = 0.5
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SPLIT_ALIASES = {"train": "train", "valid": "valid", "val": "valid", "validation": "valid", "test": "test"}


def normalize_class(folder_name: str) -> str | None:
    """'Fungal_infections' / 'fungal infections' / 'Demodicosis' -> canonical class, else None."""
    key = folder_name.strip().lower().replace(" ", "_").replace("-", "_")
    for c in CLASSES:
        if key == c or key.rstrip("s") == c.rstrip("s"):
            return c
    return None


def file_md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decide(probs: np.ndarray, threshold: float) -> tuple[str, int | None, float]:
    """Turn 6 class probabilities into the app answer.

    Returns (status, condition_index, confidence) where
      status = "healthy" | "skin_problem" | "unsure"
      condition_index = most likely disease index (None when healthy / unsure)
      confidence = probability of the chosen status (0-1)
    """
    p_healthy = float(probs[HEALTHY])
    p_problem = 1.0 - p_healthy
    if p_healthy >= threshold and p_healthy >= p_problem:
        return "healthy", None, p_healthy
    if p_problem >= threshold:
        cond = int(np.argmax(probs[1:]) + 1)
        return "skin_problem", cond, p_problem
    return "unsure", None, max(p_healthy, p_problem)


def evaluate(probs: np.ndarray, labels: np.ndarray, majority_class: int) -> dict:
    """Honest metrics: 6-class macro-F1 vs always-majority baseline, plus healthy-vs-problem."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

    preds = probs.argmax(1)
    idx = list(range(len(CLASSES)))
    base = np.full_like(labels, majority_class)
    is_prob_true = labels != HEALTHY
    p_problem = 1.0 - probs[:, HEALTHY]
    is_prob_pred = p_problem >= 0.5
    prec, rec, f1, _ = precision_recall_fscore_support(is_prob_true, is_prob_pred, average="binary", zero_division=0)
    per_class = precision_recall_fscore_support(labels, preds, labels=idx, zero_division=0)
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=idx, average="macro", zero_division=0)),
        "baseline_always_majority_macro_f1": float(f1_score(labels, base, labels=idx, average="macro", zero_division=0)),
        "per_class_f1": {c: round(float(v), 4) for c, v in zip(CLASSES, per_class[2])},
        "per_class_recall": {c: round(float(v), 4) for c, v in zip(CLASSES, per_class[1])},
        "healthy_vs_problem": {"precision": float(prec), "recall": float(rec), "f1": float(f1)},
        "confusion_matrix": confusion_matrix(labels, preds, labels=idx).tolist(),
        "classes": CLASSES,
    }


def choose_threshold(probs: np.ndarray, labels: np.ndarray, target_acc: float = 0.85,
                     min_coverage: float = 0.6) -> float:
    """Smallest threshold whose answered cases (status != unsure) reach target_acc on the
    healthy-vs-problem decision while still answering at least min_coverage of cases."""
    truth_problem = labels != HEALTHY
    for t in np.arange(0.5, 0.96, 0.05):
        answered, correct = 0, 0
        for p, tp in zip(probs, truth_problem):
            status, _, _ = decide(p, float(t))
            if status == "unsure":
                continue
            answered += 1
            correct += int((status == "skin_problem") == bool(tp))
        if answered / max(len(labels), 1) >= min_coverage and correct / max(answered, 1) >= target_acc:
            return round(float(t), 2)
    return DEFAULT_THRESHOLD

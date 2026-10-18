"""Shared helpers for M4 · Age prediction (no torch needed, so they are easy to test)."""
from __future__ import annotations

import numpy as np
import pandas as pd

GROUPS = ["young", "adult", "senior"]  # same values as backend/app/routers/pets.py
SIZE = 224
MEAN = [0.485, 0.456, 0.406]  # ImageNet
STD = [0.229, 0.224, 0.225]
DEFAULT_THRESHOLD = 0.6


def group_from_months(months: float) -> str:
    """Puppy/kitten < 1 year, Adult 1-7 years, Senior 8+ years (same as the Report form)."""
    if months < 12:
        return "young"
    if months < 96:
        return "adult"
    return "senior"


def aggregate_by_pet(df: pd.DataFrame, probs: np.ndarray):
    """Average photo probabilities per pet. df needs pet_id, species, group columns.

    Returns (pet_df, pet_probs, pet_labels) with one row per pet.
    """
    frame = df[["pet_id", "species", "group"]].reset_index(drop=True).copy()
    for i, g in enumerate(GROUPS):
        frame[f"p_{g}"] = probs[:, i]
    pets = frame.groupby("pet_id", sort=True).agg(
        species=("species", "first"), group=("group", "first"),
        **{f"p_{g}": (f"p_{g}", "mean") for g in GROUPS})
    pet_probs = pets[[f"p_{g}" for g in GROUPS]].to_numpy()
    pet_labels = pets["group"].map({g: i for i, g in enumerate(GROUPS)}).to_numpy()
    return pets.reset_index(), pet_probs, pet_labels


def choose_threshold(probs: np.ndarray, labels: np.ndarray, target_acc: float = 0.75,
                     min_coverage: float = 0.5, default: float = DEFAULT_THRESHOLD) -> float:
    """Smallest confidence threshold whose answered predictions reach target_acc,
    while still answering at least min_coverage of the pets. Falls back to default."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    for t in np.arange(0.40, 0.951, 0.05):
        keep = conf >= t
        if keep.mean() < min_coverage:
            break
        if (pred[keep] == labels[keep]).mean() >= target_acc:
            return round(float(t), 2)
    return default


def evaluate(probs: np.ndarray, labels: np.ndarray, species: np.ndarray,
             majority_class: int) -> dict:
    """macro-F1, confusion matrix, per-species macro-F1 and the majority-class baseline."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    pred = probs.argmax(axis=1)
    idx = list(range(len(GROUPS)))
    out = {
        "n": int(len(labels)),
        "accuracy": round(float(accuracy_score(labels, pred)), 4),
        "macro_f1": round(float(f1_score(labels, pred, average="macro", labels=idx, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(labels, pred, labels=idx).tolist(),
        "confusion_rows_are_true_cols_are_pred": GROUPS,
        "baseline_always_majority_macro_f1": round(float(f1_score(
            labels, np.full_like(labels, majority_class), average="macro", labels=idx,
            zero_division=0)), 4),
    }
    out["per_species_macro_f1"] = {}
    for sp in sorted(set(species)):
        m = species == sp
        out["per_species_macro_f1"][str(sp)] = round(float(f1_score(
            labels[m], pred[m], average="macro", labels=idx, zero_division=0)), 4)
    return out

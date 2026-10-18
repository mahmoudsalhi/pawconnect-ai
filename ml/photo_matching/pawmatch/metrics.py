"""Retrieval metrics (numpy only, no torch needed)."""
from __future__ import annotations

import numpy as np


def retrieval_metrics(embs: np.ndarray, labels: np.ndarray, ks=(1, 5, 10)) -> dict:
    """Recall@k and mAP.

    Every image is used as a query ("found" photo) against all other images
    (the "lost" gallery); a hit is any other image of the same animal.
    """
    sims = embs @ embs.T
    np.fill_diagonal(sims, -np.inf)  # a query must not match itself
    order = np.argsort(-sims, axis=1)
    matches = labels[order] == labels[:, None]
    matches = matches[:, :-1]  # drop the self column (ranked last at -inf)

    recalls = {f"recall@{k}": float(matches[:, :k].any(axis=1).mean()) for k in ks}

    aps = []
    for row in matches:
        hits = np.flatnonzero(row)
        if len(hits) == 0:
            continue
        precisions = (np.arange(len(hits)) + 1) / (hits + 1)
        aps.append(precisions.mean())
    return {**recalls, "mAP": float(np.mean(aps)) if aps else 0.0}

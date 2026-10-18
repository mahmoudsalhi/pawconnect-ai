"""Train the final M6 scam detector on every row and save it for the app.

    ml/.venv/Scripts/python ml/m6/build_dataset.py   # whenever raw/ changes
    ml/.venv/Scripts/python ml/m6/train_final.py

1. Scam model (model C: TF-IDF char + word phrases, logistic regression).
2. Warning thresholds chosen from out-of-fold scores:
     caution = highest score that still catches >= 95% of scams      ("be careful" banner)
     high    = lowest score where >= 97% of flagged messages are scams ("likely scam" banner)
3. Reasons model: one classifier per scam type, trained on scam rows, so the app can say *why*.
4. Everything is saved to backend/app/ai/models/m6_scam.joblib (+ a readable .json summary),
   and the cross-validated numbers to reports/final_model.md.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MultiLabelBinarizer

from textnorm import prepare

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "m6_dataset.csv"
OUT = HERE.parents[1] / "backend" / "app" / "ai" / "models" / "m6_scam.joblib"
REPORT = HERE / "reports" / "final_model.md"
VERSION = "m6-tfidf-lr-v1"
REASONS = ["upfront_payment", "fake_fee", "off_platform", "phishing_link", "impersonation", "pressure"]
RECALL_TARGET, PRECISION_TARGET = 0.95, 0.97


def features():
    # Text is prepared once with prepare() (already lower-cased), so the vectorisers get no preprocessor.
    return make_union(
        TfidfVectorizer(analyzer="char", ngram_range=(2, 6), min_df=2, sublinear_tf=True, lowercase=False),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=False,
                        token_pattern=r"(?u)\[?\w+\]?"),
    )


def scam_model():
    return make_pipeline(features(), LogisticRegression(C=4.0, class_weight="balanced", max_iter=5000))


def reason_model():
    return make_pipeline(features(), OneVsRestClassifier(LogisticRegression(C=4.0, class_weight="balanced", max_iter=5000)))


def pick_reasons(probs: np.ndarray) -> np.ndarray:
    """Every reason at 0.5 or more; at least the single most likely one."""
    chosen = probs >= 0.5
    chosen[np.arange(len(probs)), probs.argmax(axis=1)] = True
    return chosen


def choose_thresholds(y: np.ndarray, score: np.ndarray) -> tuple[float, float, pd.DataFrame]:
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        p = score >= t
        rows.append({"threshold": t, "precision": precision_score(y, p, zero_division=1), "recall": recall_score(y, p),
                     "legit_flagged": p[y == 0].mean()})
    table = pd.DataFrame(rows)
    caution = table.loc[table["recall"] >= RECALL_TARGET, "threshold"].max()
    high = table.loc[table["precision"] >= PRECISION_TARGET, "threshold"].min()
    if np.isnan(caution):
        caution = table["threshold"].min()
    if np.isnan(high) or high < caution:
        high = max(caution, 0.5)
    return float(caution), float(high), table


def main() -> None:
    df = pd.read_csv(DATA, keep_default_na=False, dtype={"is_scam": int, "fold": int})
    X = df["text"].map(prepare)
    y = df["is_scam"].to_numpy()
    folds = sorted(df["fold"].unique())

    # ---- 1+2. Out-of-fold scores -> thresholds ----
    oof = np.zeros(len(df))
    for k in folds:
        tr, te = df["fold"] != k, df["fold"] == k
        oof[te.to_numpy()] = scam_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    caution, high, table = choose_thresholds(y, oof)

    def at(t):
        p = oof >= t
        return {"threshold": t, "precision": round(precision_score(y, p), 3), "recall": round(recall_score(y, p), 3),
                "f1": round(f1_score(y, p), 3), "legit_flagged": f"{int(p[y == 0].sum())}/{int((y == 0).sum())}",
                "scams_caught": f"{int(p[y == 1].sum())}/{int((y == 1).sum())}"}

    levels = {"caution": at(caution), "high": at(high)}
    pr_auc = average_precision_score(y, oof)

    # ---- 3. Reasons, evaluated on scam rows only ----
    scam = df[df["is_scam"] == 1].reset_index(drop=True)
    Xs = scam["text"].map(prepare)
    mlb = MultiLabelBinarizer(classes=REASONS)
    Ys = mlb.fit_transform(scam["labels"].str.split("|"))
    pred = np.zeros_like(Ys)
    probs = np.zeros(Ys.shape)
    for k in folds:
        tr, te = scam["fold"] != k, scam["fold"] == k
        model = reason_model().fit(Xs[tr], Ys[tr.to_numpy()])
        probs[te.to_numpy()] = model.predict_proba(Xs[te])
        pred[te.to_numpy()] = pick_reasons(probs[te.to_numpy()])
    p, r, f, support = precision_recall_fscore_support(Ys, pred, zero_division=0)
    reason_cv = {lab: {"precision": round(float(p[i]), 2), "recall": round(float(r[i]), 2), "f1": round(float(f[i]), 2),
                       "rows": int(support[i])} for i, lab in enumerate(REASONS)}
    micro_f1 = f1_score(Ys, pred, average="micro")
    # The app lists reasons most-likely first: is that first one a true reason?
    main_reason_right = float(np.mean(Ys[np.arange(len(Ys)), probs.argmax(axis=1)] == 1))

    # ---- 4. Final fit on everything, save ----
    final_scam = scam_model().fit(X, y)
    final_reason = reason_model().fit(Xs, Ys)
    meta = {
        "version": VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(df)), "scams": int(y.sum()), "groups": int(df["group"].nunique()),
        "batches": sorted(df["batch"].unique().tolist()),
        "sklearn_version": sklearn.__version__,
        "reasons": REASONS,
        "thresholds": {"caution": caution, "high": high},
        "cv": {"pr_auc": round(float(pr_auc), 3), "levels": levels, "reasons": reason_cv,
               "reasons_micro_f1": round(float(micro_f1), 3)},
        "note": "Synthetic, LLM-generated training data: cross-validated numbers are optimistic until "
                "confirmed on the human-written test set.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({**meta, "scam_model": final_scam, "reason_model": final_reason}, OUT, compress=3)
    OUT.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- Report ----
    print(f"{len(df)} rows · PR-AUC {pr_auc:.3f} (out-of-fold)")
    print(f"thresholds -> caution {caution:.2f} | high {high:.2f}")
    for name, v in levels.items():
        print(f"  {name:8} >= {v['threshold']:.2f}: precision {v['precision']:.2f}  recall {v['recall']:.2f}  "
              f"scams caught {v['scams_caught']}  legit flagged {v['legit_flagged']}")
    print(f"\nreasons (scam rows, out-of-fold): micro-F1 {micro_f1:.2f} · first reason shown is correct {main_reason_right:.0%}")
    for lab, v in reason_cv.items():
        print(f"  {lab:16} P {v['precision']:.2f}  R {v['recall']:.2f}  F1 {v['f1']:.2f}  ({v['rows']} rows)")
    size_kb = OUT.stat().st_size / 1024
    print(f"\nsaved {OUT.relative_to(HERE.parents[1])} ({size_kb:.0f} KB) + .json · scikit-learn {sklearn.__version__}")

    md = [
        "# M6 final model", "",
        f"Version `{VERSION}`, trained on {len(df)} rows ({int(y.sum())} scams, {df['group'].nunique()} groups, "
        f"batches {', '.join(meta['batches'])}). Numbers below are out-of-fold (grouped 5-fold CV) and optimistic: "
        "synthetic data, and thresholds were picked on the same folds.", "",
        f"PR-AUC: **{pr_auc:.3f}**", "",
        "## Warning levels", "",
        "| Level | Score | Precision | Recall | Scams caught | Legit flagged |", "|---|---|---|---|---|---|",
        *[f"| {n} | ≥ {v['threshold']:.2f} | {v['precision']:.2f} | {v['recall']:.2f} | {v['scams_caught']} | {v['legit_flagged']} |"
          for n, v in levels.items()], "",
        f"- **caution**: highest threshold that still catches ≥ {RECALL_TARGET:.0%} of scams → soft \"be careful\" banner.",
        f"- **high**: lowest threshold where ≥ {PRECISION_TARGET:.0%} of flagged messages are scams → \"likely scam\" banner.", "",
        "## Reasons (scam rows only)", "",
        f"Micro-F1 {micro_f1:.2f}; the first reason shown is a correct one for {main_reason_right:.0%} of scams.", "",
        "| Reason | Precision | Recall | F1 | Rows |", "|---|---|---|---|---|",
        *[f"| {lab} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} | {v['rows']} |" for lab, v in reason_cv.items()], "",
    ]
    REPORT.write_text("\n".join(md), encoding="utf-8")
    print(f"report -> {REPORT.relative_to(HERE)}")


if __name__ == "__main__":
    main()

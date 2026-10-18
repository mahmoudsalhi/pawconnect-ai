"""M6 baselines: keyword rules vs TF-IDF + logistic regression, scored with grouped 5-fold CV.

    ml/.venv/Scripts/python ml/m6/build_dataset.py   # first, whenever raw/ changes
    ml/.venv/Scripts/python ml/m6/baselines.py

Every row is predicted exactly once, by a model that never saw its group (out-of-fold).
Writes reports/baseline_results.md and reports/errors_<model>.csv.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline, make_union

from textnorm import normalize, prepare

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "m6_dataset.csv"
REPORTS = HERE / "reports"
THRESHOLD = 0.5
REWARD_FAMILY = r"REWARD|PAID-AFTER|RIB-TO-PAY"

# ---------- Model A: keyword rules (run on normalised text) ----------

MONEY = r"(\bdt\b|dinar|دينار|\bmille\b|ألف|flous|فلوس|argent|cash|\bfees?\b|frais|acompte|3arboun|عربون|caution|recharge|ريشارج|d17|flouci|e-?dinar|mandat|مندا|حوالة|virement|transfer|western union|paypal|تحويل)"
BEFORE = r"(9bal|avant|d'abord|d'avance|à l'avance|first|upfront|in advance|before|قبل|مسبقا|أولا|اولا)"
PAY = r"(ab3ath|khalles|tkhalles|envoy|payez|payer|réglez|régler|versez|send|\bpay\b|ابعث|خلص|ادفع|أرسل|حوّل|تخلص|تدفع|يدفع)"
NEGATED = r"(ma n?khallas|ma tab3ath|ma tkhallsou|ne (verse|paie|paierai)|je ne verse|rien à l'avance|won't (pay|send)|nothing upfront|anything upfront|لا أدفع|ما نخلص|ما تبعثو|ما تخلصو)"
CHANNEL = r"(whatsapp|watsap|wtsap|telegram|insta|instagram|viber|signal|messenger|facebook|tiktok|email|\[email\]|واتساب|تيليغرام|تيليڤرام|فيسبوك|التيليغرام|الواتساب)"
ONLY = r"(barka|\bonly\b|uniquement|seulement|exclusivement|حصرا|كان على|ma (t)?jawebnich hna|ma tektebch hna|ma tkallemnich hna|ne réponds pas|répond(ez)? pas ici|don't (message|reply|write) here|ما تجاوبنيش|ما تكلمنيش|ما نجاوبش)"
SECRET = r"(\bcode\b|\botp\b|\bcvv\b|mot de passe|password|\brib\b|carte bancaire|bank|bancaire|\bcard\b|carte d'identit|\bcin\b|كود|الكارطة|بطاقة|كلمة السر|بيانات|حساب)"
ASK = r"(ab3ath|envoi|envoy|send|donne|communiqu|saisi|enter|d5el|7ot|ابعث|أرسل|أدخل|ادخل|عمّر|املأ|log ?in|connect|سجّل)"
LINK_LURE = r"(log ?in|connect|d5el|confirm|verif|vérif|activ|password|mot de passe|bank|banc|\bcard\b|carte|بيانات|سجّل|أدخل|بطاق)"


def rules(text: str) -> list[str]:
    t = normalize(text)
    fired = []
    if re.search(MONEY, t) and re.search(BEFORE, t) and re.search(PAY, t) and not re.search(NEGATED, t):
        fired.append("pay_before")
    if re.search(CHANNEL, t) and re.search(ONLY, t):
        fired.append("off_platform")
    if re.search(SECRET, t) and re.search(ASK, t):
        fired.append("secret_request")
    if "[url]" in t and re.search(LINK_LURE, t):
        fired.append("link_lure")
    return fired


# ---------- Model B: TF-IDF character n-grams + logistic regression ----------

def tfidf_lr(preprocess):
    """B: character n-grams inside words only (the first baseline)."""
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, preprocessor=preprocess),
        LogisticRegression(C=4.0, class_weight="balanced", max_iter=5000),
    )


def tfidf_lr_phrases():
    """C: character n-grams across spaces + words and 2-word phrases.

    char_wb never crosses a space, so it cannot see the phrases that separate an off-platform scam
    from a legit message ("WhatsApp only" vs "WhatsApp wala hna / ou ici / or here").
    """
    return make_pipeline(
        make_union(
            TfidfVectorizer(analyzer="char", ngram_range=(2, 6), min_df=2, sublinear_tf=True, preprocessor=prepare),
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, preprocessor=prepare,
                            token_pattern=r"(?u)\[?\w+\]?"),
        ),
        LogisticRegression(C=4.0, class_weight="balanced", max_iter=5000),
    )


def out_of_fold(df: pd.DataFrame, make_model) -> np.ndarray:
    proba = np.zeros(len(df))
    for k in sorted(df["fold"].unique()):
        train, test = df["fold"] != k, df["fold"] == k
        model = make_model()
        model.fit(df.loc[train, "text"], df.loc[train, "is_scam"])
        proba[test.to_numpy()] = model.predict_proba(df.loc[test, "text"])[:, 1]
    return proba


# ---------- Evaluation ----------

def rate(mask: pd.Series, flagged: pd.Series) -> str:
    n = int(mask.sum())
    return f"{int(flagged[mask].sum())}/{n}" if n else "-"


def summarise(name: str, df: pd.DataFrame, pred: np.ndarray, score: np.ndarray | None) -> dict:
    y = df["is_scam"].to_numpy()
    p = pd.Series(pred, index=df.index).astype(bool)
    scam, legit = df["is_scam"] == 1, df["is_scam"] == 0
    hard_neg = legit & (df["difficulty"] == "hard_negative")
    reward = legit & df["template_id"].str.contains(REWARD_FAMILY)
    obf = scam & (df["difficulty"] == "obfuscated")
    fold_f1 = [f1_score(y[df["fold"] == k], pred[df["fold"] == k], zero_division=0) for k in sorted(df["fold"].unique())]
    return {
        "model": name,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "f1_fold_spread": f"{min(fold_f1):.2f}–{max(fold_f1):.2f}",
        "pr_auc": average_precision_score(y, score) if score is not None else None,
        "scams_caught": rate(scam, p),
        "legit_flagged": rate(legit, p),
        "hard_neg_flagged": rate(hard_neg, p),
        "reward_posts_flagged": rate(reward, p),
        "obfuscated_caught": rate(obf, p),
        "_pred": p,
    }


def per_language(df: pd.DataFrame, results: list[dict]) -> pd.DataFrame:
    rows = []
    for lang in sorted(df["language"].unique()):
        m = df["language"] == lang
        row = {"language": lang, "rows": int(m.sum())}
        for r in results:
            p = r["_pred"]
            row[f"{r['model']}: scams caught"] = rate(m & (df["is_scam"] == 1), p)
            row[f"{r['model']}: legit flagged"] = rate(m & (df["is_scam"] == 0), p)
        rows.append(row)
    return pd.DataFrame(rows)


def to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else (f"{v:.2f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(DATA, dtype={"is_scam": int, "fold": int}, keep_default_na=False)
    REPORTS.mkdir(exist_ok=True)

    rule_hits = df["text"].map(rules)
    rule_pred = rule_hits.map(bool).to_numpy().astype(int)

    raw_score = out_of_fold(df, lambda: tfidf_lr(str.lower))
    norm_score = out_of_fold(df, lambda: tfidf_lr(normalize))
    phrase_score = out_of_fold(df, tfidf_lr_phrases)

    results = [
        summarise("A. rules", df, rule_pred, None),
        summarise("B1. char-in-words (raw)", df, (raw_score >= THRESHOLD).astype(int), raw_score),
        summarise("B2. char-in-words (normalised)", df, (norm_score >= THRESHOLD).astype(int), norm_score),
        summarise("C. char + words/phrases", df, (phrase_score >= THRESHOLD).astype(int), phrase_score),
    ]

    table = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
    langs = per_language(df, [results[0], results[3]])

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(f"\n{len(df)} rows · {df['group'].nunique()} groups · grouped 5-fold CV (out-of-fold) · threshold {THRESHOLD}\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nPer language (A = rules, C = char + words/phrases):")
    print(langs.to_string(index=False))

    # Errors for review.
    scores = {"rules": (rule_pred, None), "tfidf_lr": (results[3]["_pred"].to_numpy().astype(int), phrase_score)}
    for name, (pred, score) in scores.items():
        wrong = df[pred != df["is_scam"]].copy()
        wrong["error"] = np.where(wrong["is_scam"] == 1, "missed scam", "false alarm")
        if score is not None:
            wrong["scam_probability"] = score[wrong.index].round(3)
        else:
            wrong["rules_fired"] = rule_hits[wrong.index].map(lambda h: "|".join(h))
        keep = ["error", "id", "language", "difficulty", "labels", "template_id"] + \
               (["scam_probability"] if score is not None else ["rules_fired"]) + ["text"]
        wrong.sort_values(["error", "language"])[keep].to_csv(REPORTS / f"errors_{name}.csv", index=False, encoding="utf-8-sig")
        print(f"\n{name}: {len(wrong)} errors -> reports/errors_{name}.csv")

    md = [
        "# M6 baseline results",
        "",
        f"Dataset: {len(df)} rows, {df['group'].nunique()} groups (batches {', '.join(sorted(df['batch'].unique()))}). "
        f"Grouped 5-fold cross-validation: each row is predicted once by a model that never saw its group. "
        f"Threshold {THRESHOLD}. Synthetic data only, so these numbers are optimistic; "
        "the human-written test set gives the real score.",
        "",
        "## Overall",
        "",
        to_md(table),
        "",
        "- **hard_neg_flagged**: legit messages that look scammy, wrongly flagged. Lower is better.",
        "- **reward_posts_flagged**: real owners offering a reward paid on return, wrongly flagged. Should be 0.",
        "- **f1_fold_spread**: lowest and highest F1 across the 5 folds; a wide spread means the score is noisy.",
        "",
        "## Per language",
        "",
        to_md(langs),
        "",
    ]
    (REPORTS / "baseline_results.md").write_text("\n".join(md), encoding="utf-8")
    print("Report -> reports/baseline_results.md")


if __name__ == "__main__":
    main()

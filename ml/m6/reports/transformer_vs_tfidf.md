# M6: XLM-RoBERTa vs TF-IDF (model C)

Same 999 rows, same grouped 5 folds (`fold` column), out-of-fold predictions, threshold 0.5.
XLM-R: `xlm-roberta-base`, 5 epochs, lr 2e-5, batch 16, max length 128, balanced class weights,
fixed in advance (no tuning), one seed, trained on a Colab T4 (`colab/m6_transformer_colab.ipynb`).
Row-level predictions: `compare_c_vs_xlmr.csv`.

| Model | Precision | Recall | F1 | F1 across folds | PR-AUC | Legit wrongly flagged | Reward posts wrongly flagged | New scam types (B09–B10) missed |
|---|---|---|---|---|---|---|---|---|
| **C. TF-IDF char + word phrases + LR** | **0.93** | **0.95** | **0.94** | 0.93–0.97 | **0.98** | **29/608** | **5/47** | 7/76 |
| XLM-R base (fine-tuned) | 0.86 | 0.88 | 0.87 | 0.78–0.93 | 0.91 | 54/608 | 22/47 | 8/76 |
| Average of both | 0.88 | 0.90 | 0.89 | – | 0.97 | 48/608 | 20/47 | 7/76 |

Errors: 28 made only by C, 82 only by XLM-R, 20 by both. 51 of XLM-R's 102 errors were made with more
than 95% confidence.

## Findings

1. **The small model wins on every measure**, including on scam types it never saw (B09–B10).
2. **XLM-R flags real reward posts** (22 of 47): it tends to treat any amount of money as a scam,
   which is exactly the failure M6 must avoid ("warn, never block" a genuine owner).
3. **XLM-R is unstable and over-confident**: its F1 swings from 0.78 to 0.93 across folds, and half of its
   mistakes are made with >95% confidence, so its score cannot be trusted as a warning level.
4. **Averaging the two does not help**: the transformer's errors drag the combination down.

## Likely reasons

- **Too little data for a 278M-parameter model**: about 800 training rows per fold.
- **Tunisian Arabizi** ("ab3athli", "9bal", "wala hna") is rare in XLM-R's pre-training text, so its
  multilingual knowledge transfers poorly to the most common language in the dataset.
- **No tuning, single seed**: settings were fixed in advance to keep the comparison fair; a tuned run
  could do somewhat better, but the gap (F1 0.87 vs 0.94, reward false alarms 22 vs 5) is large.

## Decision

Model C is the production candidate: more accurate, more stable, explainable (its top-weighted words
and phrases can be shown as reasons), trains in seconds and runs on CPU. The transformer can be
re-tested once the dataset passes ~1,500 rows, with a few seeds and a small learning-rate search.

# M6 final model

Version `m6-tfidf-lr-v1`, trained on 999 rows (391 scams, 616 groups, batches B01, B02, B03, B04, B05, B06, B07, B08, B09, B10). Numbers below are out-of-fold (grouped 5-fold CV) and optimistic: synthetic data, and thresholds were picked on the same folds.

PR-AUC: **0.982**

## Warning levels

| Level | Score | Precision | Recall | Scams caught | Legit flagged |
|---|---|---|---|---|---|
| caution | ≥ 0.50 | 0.93 | 0.95 | 372/391 | 29/608 |
| high | ≥ 0.64 | 0.97 | 0.84 | 327/391 | 10/608 |

- **caution**: highest threshold that still catches ≥ 95% of scams → soft "be careful" banner.
- **high**: lowest threshold where ≥ 97% of flagged messages are scams → "likely scam" banner.

## Reasons (scam rows only)

Micro-F1 0.84; the first reason shown is a correct one for 92% of scams.

| Reason | Precision | Recall | F1 | Rows |
|---|---|---|---|---|
| upfront_payment | 0.79 | 0.75 | 0.77 | 106 |
| fake_fee | 0.84 | 0.89 | 0.86 | 122 |
| off_platform | 1.00 | 0.97 | 0.99 | 102 |
| phishing_link | 0.95 | 0.96 | 0.95 | 76 |
| impersonation | 0.73 | 0.64 | 0.69 | 76 |
| pressure | 0.89 | 0.64 | 0.74 | 80 |

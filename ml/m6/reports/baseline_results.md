# M6 baseline results

Dataset: 999 rows, 616 groups (batches B01, B02, B03, B04, B05, B06, B07, B08, B09, B10). Grouped 5-fold cross-validation: each row is predicted once by a model that never saw its group. Threshold 0.5. Synthetic data only, so these numbers are optimistic; the human-written test set gives the real score.

## Overall

| model | precision | recall | f1 | f1_fold_spread | pr_auc | scams_caught | legit_flagged | hard_neg_flagged | reward_posts_flagged | obfuscated_caught |
|---|---|---|---|---|---|---|---|---|---|---|
| A. rules | 0.98 | 0.48 | 0.64 | 0.58–0.71 |  | 187/391 | 4/608 | 4/290 | 2/47 | 53/97 |
| B1. char-in-words (raw) | 0.90 | 0.92 | 0.91 | 0.90–0.92 | 0.97 | 358/391 | 38/608 | 32/290 | 5/47 | 93/97 |
| B2. char-in-words (normalised) | 0.90 | 0.92 | 0.91 | 0.90–0.92 | 0.97 | 361/391 | 39/608 | 34/290 | 5/47 | 93/97 |
| C. char + words/phrases | 0.93 | 0.95 | 0.94 | 0.93–0.97 | 0.98 | 372/391 | 29/608 | 27/290 | 5/47 | 90/97 |

- **hard_neg_flagged**: legit messages that look scammy, wrongly flagged. Lower is better.
- **reward_posts_flagged**: real owners offering a reward paid on return, wrongly flagged. Should be 0.
- **f1_fold_spread**: lowest and highest F1 across the 5 folds; a wide spread means the score is noisy.

## Per language

| language | rows | A. rules: scams caught | A. rules: legit flagged | C. char + words/phrases: scams caught | C. char + words/phrases: legit flagged |
|---|---|---|---|---|---|
| darija_arabic | 136 | 26/57 | 0/79 | 54/57 | 7/79 |
| darija_latin | 317 | 53/125 | 2/192 | 118/125 | 9/192 |
| english | 105 | 19/32 | 0/73 | 31/32 | 1/73 |
| french | 188 | 39/77 | 2/111 | 71/77 | 4/111 |
| mixed | 151 | 33/58 | 0/93 | 57/58 | 5/93 |
| msa_arabic | 102 | 17/42 | 0/60 | 41/42 | 3/60 |

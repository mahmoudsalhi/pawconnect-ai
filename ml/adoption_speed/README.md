# M3 adoption speed: training dataset

`build_dataset.py` builds one clean table, `data/processed/adoption_speed.csv`. Each row is one adoptable stay of a dog or cat, labelled with the PawConnect adoption speed class used by `AIProvider.adoption_speed` (`backend/app/ai/base.py`).

| Class | Meaning | Rows | Share |
|---|---|---|---|
| 0 | adopted the same day | 540 | 0.9 % |
| 1 | adopted within 1–7 days | 24,188 | 41.3 % |
| 2 | adopted within 8–30 days | 18,916 | 32.3 % |
| 3 | adopted within 31–99 days | 10,199 | 17.4 % |
| 4 | not adopted after 100+ days | 4,789 | 8.2 % |

That's 58,632 rows from Austin Animal Center: 39,894 dogs and 18,738 cats taken in between Oct 2013 and Jan 2025. The exact filter counts are written to `data/processed/adoption_speed_stats.json` on every build.

## Rebuild it

```powershell
cd ml
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt

# Austin Animal Center open data (public, no account needed, ~47 MB)
mkdir adoption_speed\data\raw
curl.exe -L -o adoption_speed\data\raw\austin_intakes.csv  "https://data.austintexas.gov/api/views/wter-evkm/rows.csv?accessType=DOWNLOAD"
curl.exe -L -o adoption_speed\data\raw\austin_outcomes.csv "https://data.austintexas.gov/api/views/9t4d-g238/rows.csv?accessType=DOWNLOAD"

.\.venv\Scripts\python adoption_speed\build_dataset.py
```

### Optional: add PetFinder.my (strongly recommended)

The Kaggle [PetFinder.my Adoption Prediction](https://www.kaggle.com/c/petfinder-adoption-prediction) data is the only public source with the listing fields PawConnect shows: **photo count, description and fee**. The Austin data doesn't have them. Accept the competition rules on Kaggle, create an API token (`%USERPROFILE%\.kaggle\kaggle.json`), then run:

```powershell
.\.venv\Scripts\python -m pip install kaggle
.\.venv\Scripts\kaggle competitions download -c petfinder-adoption-prediction -f train/train.csv -p adoption_speed\data\raw\petfinder
.\.venv\Scripts\kaggle competitions download -c petfinder-adoption-prediction -f breed_labels.csv -p adoption_speed\data\raw\petfinder
.\.venv\Scripts\kaggle competitions download -c petfinder-adoption-prediction -f color_labels.csv -p adoption_speed\data\raw\petfinder
# unzip any .zip files in that folder, then rebuild
```

On the next build, the script adds about 11,500 single-animal PetFinder listings with `source = petfinder`. Check the competition's licence terms before using a model trained on PetFinder data commercially.

## Columns

| Column | Notes |
|---|---|
| `record_id`, `source`, `animal_id` | `source` is `austin` or `petfinder` |
| `intake_date`, `split` | `split` is `train` / `val` / `test` (see below) |
| **`species`** | `Dog` / `Cat` |
| **`age_group`** | `young` (< 1 yr), `adult`, `senior` (dogs ≥ 8 yrs, cats ≥ 10 yrs) |
| **`breed`** | in app style, e.g. `Labrador retriever mix`, `Domestic shorthair` |
| **`size`** | `Small` / `Medium` / `Large`. Austin dogs get it from a breed lookup (99.8 % covered); cats default to `Medium` unless the breed is known to be small or large |
| **`photo_count`, `description`, `fee_amount`, `fee_currency`** | PetFinder rows only; empty for Austin |
| `age_months`, `breed_primary`, `is_mixed`, `size_source` | finer detail behind the fields above |
| `sex`, `sterilized`, `color_primary`, `is_multicolor`, `has_name` | taken at intake |
| `intake_type`, `intake_condition`, `intake_month` | Austin only. For PetFinder, `intake_condition` holds its health field |
| `days_to_outcome`, `outcome_type` | how the stay ended |
| **`adoption_speed`** | the target, 0–4 |

The columns in **bold** are the ones the app passes in `ListingFeatures` today.

## How the labels are made (Austin)

1. Each intake is paired with the first outcome after it for the same animal. If the animal was taken in again before any outcome, the intake is dropped. Returning animals produce several stays.
2. The script keeps only stays that look like an adoption listing:
   - dogs and cats only
   - taken in as stray, owner surrender or abandoned
   - not neonatal, nursing, pregnant or feral
   - at least 8 weeks old
3. These stays are removed, because their time in the shelter doesn't measure adopter demand:
   - reclaimed by the owner
   - transferred, euthanised or died **before** 100 days
   - taken in during the last 100 days of the data, where the result isn't known yet
4. Adoptions are binned by calendar days in care. Any animal still not adopted after 100 days is class 4.

## Splits

- **Austin:** split by time. Animals first seen in the newest 12 months go to `test`, the 12 months before that to `val`, and everything older to `train`. All stays of one animal stay in the same split. Use this split: a random split gives optimistic scores because adoption trends drift (2020 and 2023 were noticeably slower).
- **PetFinder:** has no dates, so it is split 70/15/15 by `RescuerID`. One rescuer's listings never end up in two splits.

## What a model can learn from it

A quick gradient-boosting check, trained on `train` and scored on `test`:

| Inputs | Accuracy | Quadratic weighted kappa |
|---|---|---|
| Always predict the most common class | 0.31 | 0.00 |
| Only the 4 fields the app sends today (`species`, `age_group`, `breed`, `size`) | 0.37 | 0.20 |
| All columns | 0.45 | 0.37 |

For comparison, the PetFinder competition winners reached about 0.45 kappa, using text and images as well. The signal in this data matches what shelters see: small dogs, young animals and Chihuahua or Dachshund mixes go fastest, while seniors, pit-bull types and animals that arrived sick or injured wait longest.

Passing `sex`, `sterilized`, `age_months` and the health condition through `ListingFeatures` should make the model in the app much more accurate.

## Known limitations

- **Region.** Austin is in the USA and PetFinder in Malaysia; neither is Tunisia. Use this data to start, then retrain once PawConnect records its own adoptions. The `Pet` model needs an adoption timestamp for that.
- **Stray hold.** Austin holds strays for about 3 days before they can be adopted, which is why class 0 is rare (about 1 %).
- **No listing-quality fields in Austin.** Tips based on photos, description or fee can only be learned from PetFinder rows.

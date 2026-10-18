# PawConnect AI

AI-assisted lost & found and adoption platform for Tunisia, built from the `PawConnect AI` Claude Design project.

| Part | Stack |
| --- | --- |
| `frontend/` | Vite, React 19, TypeScript, React Router, TanStack Query, Leaflet |
| `backend/` | FastAPI, SQLAlchemy 2, Alembic, PostgreSQL 17, JWT auth |
| `design/` | The original design files (`PawConnect AI.dc.html`, `PetCard.dc.html`, `support.js`), kept for reference |

## Run it

Prerequisites on this machine: Python 3.14, Node 24 (`C:\Program Files\nodejs`), PostgreSQL 17 (Windows service `postgresql-x64-17`).

```powershell
powershell -ExecutionPolicy Bypass -File .\dev.ps1   # starts API + web app in two windows
```

Or by hand:

```powershell
# API  ->  http://localhost:8000  (interactive docs at /docs)
cd backend
.\.venv\Scripts\python -m uvicorn app.main:app --reload --reload-dir app

# Web app  ->  http://localhost:5173  (proxies /api and /uploads to the API)
cd frontend
npm run dev
```

Demo sign-ins (password `demo-password`):

- `yasmine@example.com`: pet owner with Rex's lost report, matches, sightings, an adoption application and saved pets
- `carthage@example.com`: shelter staff for Refuge Les Amis de Carthage (can move incoming applications between stages)

The sign-in page also has a **Use demo account** button.

## First-time setup (already done here)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env                    # then set DATABASE_URL, TEST_DATABASE_URL, JWT_SECRET
.\.venv\Scripts\alembic upgrade head      # create tables
.\.venv\Scripts\python -m app.seed        # load the demo dataset (use --reset to wipe and reload)

cd ..\frontend
npm install
```

## Tests

```powershell
cd backend
.\.venv\Scripts\python -m pytest          # uses TEST_DATABASE_URL, never the dev database
```

```powershell
cd frontend
npm run build                             # type-checks and builds
```

## What's in the app

- **Home**: hero, missing-right-now grid, how-it-works tabs, the six AI models, adoption picks, community
- **Find a pet** (`/pets`): status tabs with counts, search, governorate, sort, species/age/size/breed/date/distance filters, grid, list and map views, pagination. Filters live in the URL.
- **Pet page** (`/pets/:id`): gallery, attributes, AI breed estimate, sightings timeline, approximate-area map, similar pets, contact / sighting / adoption modals, owner actions (mark reunited, delete), printable poster (`/pets/:id/poster`)
- **Report** (`/report`): 6-step wizard with photo upload, AI breed, quality and tag suggestions, map pin, contact details and review. Drafts save locally, and you're asked to sign in before publishing. Publishing runs AI matching against open reports and notifies both owners above their alert threshold.
- **AI Matching** (`/match`): match by an existing report or an uploaded photo; species / status / area / radius / threshold; ranked results, map and side-by-side; owners can dismiss "not a match"
- **Adoption** (`/adopt`): filters, apply flow, your applications, partner shelters, adoption-speed insights
- **Dashboard** (`/dashboard/...`): overview, reports, matches, applications (plus incoming applications for shelter staff), saved pets, notifications, settings

## Plugging in your own AI models

Every AI call goes through one interface: `backend/app/ai/base.py` → `AIProvider`.

| Model | Method | Used by |
| --- | --- | --- |
| M1 breed CNN | `predict_breed(images, species)` | report wizard, new reports |
| M2 photo matcher (ArcFace embeddings) | `embed(image)`, `similarity(query, candidate)` | `/api/ai/match`, auto-matching on publish |
| M3 adoption speed | `adoption_speed(features)` | shelter insights |
| M4 text tagger | `suggest_tags(text)` | report wizard tags |
| M5 photo score | `score_photo(image)` | report wizard, stored per photo |
| M6 segmentation | `remove_background(image)` | report wizard preview (return a cut-out image or `None`) |
| M7 location | `predict_location(query)` | "Where to search" map on lost pets' pages (`/api/ai/location/{id}`) |

The built-in `DemoProvider` (`backend/app/ai/demo.py`) returns deterministic placeholder results. To swap in real models:

1. Create e.g. `backend/app/ai/models.py` with `class PawModels(DemoProvider)` and override the methods you have models for. Anything you don't override keeps its placeholder.
2. Images arrive as `ImageInput(url, path)`. `path` points at the file in `backend/uploads/` for user uploads; seed photos are remote URLs.
3. Set `AI_PROVIDER=app.ai.models:PawModels` in `backend/.env` and set `name = "your-model-id"` on the class (it's stored next to cached match scores).
4. Match scores are cached in the `match_results` table. After changing models, clear it (`TRUNCATE match_results;`) so scores are recomputed.

### Trained models

`app.ai.models:PawModels` holds the models that are trained so far; the rest still use the placeholders.

- **M3 adoption speed** (`app/ai/adoption_speed.py`): XGBoost trained on the Kaggle [PetFinder.my Adoption Prediction](https://www.kaggle.com/competitions/petfinder-adoption-prediction) data (single-animal listings), using only fields a PawConnect listing has: species, age, size, sex, breed (mixed/domestic), fur length, vaccinated / dewormed / spayed-neutered, photo count, description length and health mentions, fee. QWK averaged over 5 rescuer-grouped splits and the other metrics are in `app/ai/weights/m3_adoption_speed.meta.json`. Listing tips are what-if checks: the model is re-run with more photos, a fuller description, health info or no fee, and only changes it predicts would help are shown.

  To retrain, accept the competition rules on Kaggle, put your API token in `~/.kaggle/kaggle.json`, then:

  ```powershell
  cd backend
  .\.venv\Scripts\python -m pip install -r requirements-ml.txt
  .\.venv\Scripts\python -m ml.m3_adoption_speed.train
  ```

- **M9 compatibility matching** (`/find-match` page, `POST /api/ai/compatibility`): an adopter answers ten questions (home, children, other pets, activity, hours alone, experience, grooming, age, size) plus an optional free-text wish. A trait model (`app/ai/pet_traits.py`) reads each adoption profile and predicts six 1-5 ratings: energy, good with children, apartment-friendly, first-time-owner friendly, good with other dogs, grooming. It is a ridge regression on `all-MiniLM-L6-v2` sentence embeddings, trained on 618 reviewed breeds from Kaggle [Dog Breeds: Traits, Origins & Provenance](https://www.kaggle.com/datasets/jesusic/dog-breeds-traits-origins-provenance) (CC BY 4.0); 5-fold cross-validation results are in `app/ai/weights/m9_traits.meta.json`. `app/ai/compatibility.py` turns the answers and traits into a 0-100 score with reasons; hours alone and living with cats use written rules because the dataset has no such ratings. The data covers dogs only, so cat estimates are less certain. Retrain with `.\.venv\Scripts\python -m ml.m9_compatibility.train` (downloads the dataset and the embedding model on first run).

- **M2 photo matching** (`ml/photo_matching/`): see below.

### M2 is already wired in

`backend/app/ai/models.py` (`PawModels`) replaces `embed` and `similarity` with the trained M2 photo matcher; everything else stays a placeholder. The model itself runs as a small service in `ml/photo_matching/` (see its README for training, evaluation and setup).

1. Put the trained model at `ml/photo_matching/checkpoints/m2.pt` and create `ml/photo_matching/.venv` (`pip install -r requirements.txt`).
2. In `backend/.env` set `AI_PROVIDER=app.ai.models:PawModels` (and `M2_URL` if the service isn't on `http://127.0.0.1:8002`).
3. Start everything with `dev.ps1` (it also starts the M2 service when the model file exists), then clear the cached scores: `TRUNCATE match_results;`
4. If the M2 service is down, publishing a report still works (auto-matching is skipped) and `/api/ai/match` answers `503`.

The seed data includes the design's hand-picked match scores (Mishmish ↔ Ginger 87%, Rex ↔ Kahwa 64%, …) so the demo looks like the design until real models produce their own.

## M7: where to search for a lost pet

Lost pets' pages show the chance of being found alive and how long it may take, and search zones (50 / 80 / 95% likely), from the M7 models in `ml/m7_location/` (see its README for the dataset, training and test results). Unlike M1–M6, `AIProvider.predict_location` has a working default: the bundled model.

```powershell
cd backend
.\.venv\Scripts\python -m pip install -r requirements-ml.txt   # numpy, scipy, pandas, lightgbm
```

Without these libraries the endpoint answers 503 and the page simply hides the map. M7 covers Grand Tunis, Sousse, Sfax and Nabeul–Hammamet, for cats and dogs. It is **experimental**: it was trained on simulated cases and should be retrained on real reunions. It loads about 400 MB of map grids into memory on first use.

## Notes for production

- **Map tiles**: the app uses OpenStreetMap's public tile servers, which are fine for development but not for production traffic. Set `VITE_MAP_TILES` (and `VITE_MAP_ATTRIBUTION`) in `frontend/.env.local` to a provider you have an agreement with. (The design's CARTO URL now requires an API key.)
- **Secrets**: generate a new `JWT_SECRET` and database password per environment. `backend/.env` is git-ignored.
- **Uploads**: images are re-encoded as JPEG (which strips EXIF/GPS) and stored in `backend/uploads/`. Move them to object storage before scaling out.
- **Not built yet**: password reset and Google sign-in (both buttons say so), email/SMS delivery for notifications (in-app only), and editing a published report.
- **Locations**: public listings expose coordinates rounded to ~100 m and an approximate-area circle; the exact address is never shown.

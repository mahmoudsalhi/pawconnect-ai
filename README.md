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
| M2 Siamese matcher | `embed(image)`, `similarity(query, candidate)` | `/api/ai/match`, auto-matching on publish |
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

The seed data includes the design's hand-picked match scores (Mishmish ↔ Ginger 87%, Rex ↔ Kahwa 64%, …) so the demo looks like the design until real models produce their own.

## M7: where to search for a lost pet

Lost pets' pages show the chance of being found alive and how long it may take, and search zones (50 / 80 / 95% likely), from the M7 models in `ml/m7_location/` (see its README for the dataset, training and test results). Unlike M1–M6, `AIProvider.predict_location` has a working default: the bundled model.

```powershell
cd backend
.\.venv\Scripts\python -m pip install -r requirements-ml.txt   # numpy, scipy, pandas, lightgbm
```

Without these libraries the endpoint answers 503 and the page simply hides the map. M7 covers Grand Tunis, Sousse, Sfax and Nabeul–Hammamet, for cats and dogs. It is **experimental**: it was trained on simulated cases and should be retrained on real reunions. It loads about 400 MB of map grids into memory on first use.

## AI search assistant (LLM API: Groq or Claude)

On a lost pet's page, the "Where to search" panel can also use an LLM API wrapped around M7 (`GET /api/ai/assistant/{id}`, code in `backend/app/ai/assistant.py`):

1. **Reads the description.** The owner may write in Darija, French, Arabic or English. The LLM extracts what the form doesn't ask: whether a cat goes outside, temperament, how the pet got lost, and collar or tag. These go into M7, so the zones and the chances get sharper. The page shows what was read as chips.
2. **Writes a search plan** from M7's zones and outlook and the real named places inside the zones (from OpenStreetMap, `ml/m7_location/data/grids/*_places.json`): when and where to search, what to do, a poster in French and Arabic, and a post to share.

The LLM only explains and organises. Every number and place comes from M7 and the map data. Phone numbers and emails are stripped from the description before it's sent, and only the approximate public area is shared. Answers are validated against a schema; invalid values become "unknown".

Pick a provider in `backend/.env` (the first key found is used):

```
# Free, no credit card: create a key at https://console.groq.com
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-120b

# Or the Claude API (paid, about $0.002 per lost pet with Haiku 5.5)
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-haiku-5-5
```

Without a key the panel shows M7 alone. Groq's free tier is rate-limited (per minute and per day), which is plenty for development and demos. Results are cached per description, and plans per pet per day.

## Notes for production

- **Map tiles**: the app uses OpenStreetMap's public tile servers, which are fine for development but not for production traffic. Set `VITE_MAP_TILES` (and `VITE_MAP_ATTRIBUTION`) in `frontend/.env.local` to a provider you have an agreement with. (The design's CARTO URL now requires an API key.)
- **Secrets**: generate a new `JWT_SECRET` and database password per environment. `backend/.env` is git-ignored.
- **Uploads**: images are re-encoded as JPEG (which strips EXIF/GPS) and stored in `backend/uploads/`. Move them to object storage before scaling out.
- **Not built yet**: password reset and Google sign-in (both buttons say so), email/SMS delivery for notifications (in-app only), and editing a published report.
- **Locations**: public listings expose coordinates rounded to ~100 m and an approximate-area circle; the exact address is never shown.

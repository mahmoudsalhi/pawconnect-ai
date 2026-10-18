# M7 — Lost Pet Location Prediction: training dataset

A generated dataset of **24,000 lost-pet cases in four Tunisian regions** for training M7, the model that answers *"given what we know now, where is this pet most likely to be?"*

No public dataset pairs a lost pet's last-seen point with where it was found, and none covers Tunisia. So this dataset is **simulated, but anchored to reality in two ways**:

| | What | Where it comes from |
|---|---|---|
| **Real** | Streets, buildings, parks, scrub, farmland, lakes, sea, shops, restaurants, markets, mosques, schools, vets | OpenStreetMap, Geofabrik Tunisia extract (ODbL) |
| **Published** | How far pets travel and how often and how fast they're found | Huang et al. 2018 (1,210 cats), Lord et al. 2007 (187 dogs). The simulator is **tuned automatically** until it reproduces these figures (`calibrate.py`). |
| **Assumed** | How pets behave between those end points (hiding, fleeing, homing, food and people attraction, temperament effects) | Movement-ecology practice and Missing Animal Response field experience. **All in `config.py`**, nothing hidden in code. |

See **[output/validation.md](output/validation.md)** for every target against what was achieved, plus distance and recovery plots and example tracks on the real maps.

## Regions

| Region | Governorates | Cases |
|---|---|---|
| `grand_tunis` | Tunis, Ariana, Ben Arous, Manouba | 9,000 |
| `sousse` | Sousse | 5,000 |
| `sfax` | Sfax | 5,000 |
| `nabeul_hammamet` | Nabeul | 5,000 |

These match the governorates the app offers (`backend/app/geo.py`).

## How a case is generated

1. **Profile.** Species, cat lifestyle (indoor-only or outdoor access), temperament (*curious / cautious / xenophobic* for cats, *gregarious / aloof / xenophobic* for dogs), age, sex, neutered, dog size and breed group (including the local *kelb arbi* type), collar tag and microchip.
2. **Circumstance and time.** Door or window escape, balcony fall, didn't come home, slipped the leash on a walk, wedding or Eid fireworks, moving house, escape during transport, car accident or following other dogs. Each has its own hour-of-day and season pattern (fireworks peak on summer nights).
3. **Place.** The home is drawn from built-up residential cells. The last-seen point is the home for escapes, or somewhere along a walk, near a vet, and so on. After a house move, the pet tries to head for its *old* home. The owner's reported last-seen point gets a realistic error (5 m for "from my door", 60 m for "fireworks somewhere on the street").
4. **Movement (step-selection model, 10-minute steps, up to 60 days).** Each step the pet draws 10 candidate moves and picks one according to how attractive the landing cell is: cover, food, foot traffic (liked or avoided depending on temperament), corridors, major roads (avoided, and risky to cross) and water (impassable). A state machine (*hide, explore, forage, flee, homing*) sets speed and goals. It responds to time of day (cats at dawn and dusk, xenophobic dogs at night), midday summer heat (rest in shade) and hunger (cats come out of hiding after a few days).
5. **Recovery (competing hazards each step).** The pet can come home by itself, be found by the owner searching near the last-seen point, or be found by the public (more likely in busy places, near posters, for approachable or tagged pets). It can be picked up by a vet or the municipality, or taken in by a stranger and driven elsewhere (which explains the long-distance tail). It can be killed on a road (found or not), or stay missing.
6. **Observation noise.** Public sightings come with location errors of 15–250 m and reporting delays. About a fifth are **false sightings** of a similar-looking animal, which is what real sightings look like.

## Files (`output/`)

| File | Rows | What |
|---|---|---|
| `cases.parquet` / `cases.csv` | 1 per case | Profile, when and where lost, outcome, where and when found, place type |
| `sightings.parquet` | 1 per report | Public sightings (true and false) with time, position, accuracy |
| `tracks.parquet` | many per case | **Ground-truth** positions: hourly for 3 days, then every 6 h |
| `training_candidates.parquet` | 32 per query | Ready-to-train cell-ranking table (below) |
| `calibration.json` | | Tuned parameters and the calibration history |
| `validation.md`, `plots/` | | Targets vs achieved, distributions, maps |

### Key columns in `cases`

- **Inputs** (known when the report is filed): `species`, `group`, `temperament`, `age_group`, `sex`, `neutered`, `size`, `breed_group`, `collar_tag`, `microchipped`, `circumstance`, `lost_at`, `hour_of_day`, `month`, `temp_c_at_loss`, `last_seen_lat/lng`, `last_seen_accuracy_m`, `home_lat/lng`, `home_distance_m`, `old_home_lat/lng`
- **Labels:** `outcome`, `found`, `found_alive`, `hours_to_found`, `found_lat/lng`, `found_distance_m`, `found_bearing_deg`, `found_place_type`. Not-found cases are **censored** at 60 days (`followed_up_hours`), so keep them for survival-style models.
- **Ground truth for analysis only. Never use as model inputs:** `true_origin_*`, `found_true_*`, `relocated_by_human`, `died_unfound`, `path_length_m`, `max_distance_m`, and in `sightings`, `is_false`, `true_lat/lng`.
- `split`: `train` / `val` / `test` by **3 km spatial blocks**, so test cases come from neighbourhoods the model never saw.

### `training_candidates`: the ready-to-train table

M7 is framed as **ranking map cells**. Each *query* is one `(case, t)` pair: one positive cell (`label=1`, where the pet actually was) and 31 negatives sampled around the last-seen point (log-uniform distance up to 4 km for cats and 12 km for dogs, any direction).

- `label_source = recovery`: t is the time found, and the positive is the reported found point. This is what the app's real data will look like.
- `label_source = track`: t is 6 h, 24 h, 72 h or 7 days, and the positive is the true simulated position. This adds supervision for "where is it *now*?"
- **Features:** distance, `toward_home_cos`, `dist_home_m`, cell layers (cover, food, people, corridor, green, building, farm, roads, water, …), 200 m context, `major_crossings` and `water_on_line` between the last-seen point and the cell, sightings known *at time t* (`nearest_sighting_m`, `latest_sighting_m`, `sightings_within_300m`), plus the case inputs and `t_hours`.
- `log_q` is the log density of the negative sampler at each candidate. Subtract it from the model's logit in a softmax over each query (sampled softmax). The trained scorer then gives correct relative probabilities when you score **every** cell at inference and draw the heat map.

A good first model is LightGBM with `objective="lambdarank"` (group = `query_id`), or a softmax over each query with the `log_q` correction. Evaluate on `split == "test"` with top-k hit rate (is the true cell among the top 1%, 5% of cells?) and median search area to find the pet.

## The trained model (`output/model/`)

`train.py` trains M7 on this dataset: LightGBM scoring one map cell at a time with an exact sampled-softmax objective (score = tree output + log q). Softmax over an equal-area lattice around the last-seen point gives the heat map: 50 m cells within 4 km for cats, 100 m cells within 12 km for dogs.

| File | What |
|---|---|
| `m7_lgbm.txt` | The model |
| `m7_meta.json` | Features, lattice, test metrics, come-home rates per group (`self_return_rate`) |
| `model_report.md` | Full test results and feature importance |
| `heatmap_examples.png` | Heat maps on held-out test cases |

Use it from Python with `m7_model.M7(...).predict_grid(grid, layers, prepare_layers(layers), case, t_hours, sightings)`. It returns lat, lng and probability per cell. Unknown inputs can be `"unknown"`.

**Test results** (held-out 3 km areas, 120 cases per group, every cell scored):

| | Median area to search: model | Circle search | 80% zone holds the pet |
|---|---|---|---|
| Dogs, full report | **0.43 km²** | 0.70 km² | 81% |
| Outdoor cats, full report | **0.54 km²** | 0.79 km² | 78% |
| Indoor cats, full report | 0.003 km² | 0.007 km² | 82% |
| Dogs, today's app form | 0.65 km² | 0.70 km² | 86% |
| Outdoor cats, today's app form | 0.64 km² | 0.79 km² | 45% |

**Read this before quoting these numbers:** they are measured on *simulated* test cases, so they show the model learned the simulator's structure, not that it works on real Tunisian pets. Most of the gain needs the extra report questions (indoor/outdoor, temperament, how it got lost, home, sighting pins). With today's form the model is only a little better than circle search, and for outdoor cats, without knowing whether the cat goes outside, its zones are too tight.

Pets that came home on their own are left out of the location training, because home isn't a place to search. Show the come-home rate separately instead (in `m7_meta.json`).

```powershell
.\.venv\Scripts\python train.py               # ~2 min training + ~4 min full-map evaluation
.\.venv\Scripts\python train.py --eval-only
```

## Recovery outlook: will it be found, and when?

`recovery.py` trains a second model on `cases.parquet`. It's a LightGBM multiclass over time bands: found alive within 6 h, 24 h, 3 days, 1 week, 2 weeks, 4 weeks or 60 days, or not found alive. Its inputs are the report (species, cat lifestyle, breed type, age, size, sex, temperament, how it got lost, collar, microchip, season, hour) and the surroundings of the last-seen point. The app conditions it on how long the pet has already been missing (`recovery.outlook`).

Test results (held-out areas): well calibrated, with each quintile's predicted chance within about 3 points of the actual one. Discrimination is modest: AUC 0.71 for "found alive" with a full report and 0.62 with today's form. That's expected, since whether a pet is found depends a lot on chance. Overall levels are calibrated to Huang 2018 and Lord 2007. How each detail shifts them is a simulator assumption.

```powershell
.\.venv\Scripts\python recovery.py   # ~1 min -> output/model/m7_recovery_lgbm.txt + m7_recovery_meta.json
```

## Rebuild

```powershell
cd ml\m7_location
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python build_dataset.py              # downloads OSM (~85 MB), calibrates, simulates (~15 min)
.\.venv\Scripts\python build_dataset.py --scale 0.05 # quick run
.\.venv\Scripts\python build_dataset.py --recalibrate
```

Grids are cached in `data/grids/` and calibration in `output/calibration.json`. The shipped calibration also includes one hand correction, recorded under `final_correction`. Outdoor-cat distance reacts very strongly to speed, so its speed was set by interpolating between two full 24,000-case runs instead of using the noisier calibration batches. `--recalibrate` does not redo that step. After changing behaviour in `config.py`, use `--recalibrate`; after changing regions or `raster.py`, use `--rebuild-grids`.

## Limitations: read before trusting the model

- **The targets aren't Tunisian.** Huang (mainly Australia, US, UK) and Lord (Ohio) are the best published figures, but Tunisian cities have more street animals, fewer microchips and shelters, and denser housing. Expect real distances and recovery rates to differ.
- **Behaviour between the targets is assumed.** The model will learn our assumptions, such as cats hiding near cover and xenophobic dogs avoiding crowds. These match field experience but aren't measured here.
- **OSM coverage is uneven.** Building footprints are only partly mapped in Tunisia, so "cover" and "people" are proxies built from roads, land use and points of interest.
- **Use it to pre-train, then replace it.** Train M7 on this, ship it, and log every real reunion: last-seen point, found point, time and pet profile. Once you have a few hundred real cases, fine-tune on them and re-run the calibration with Tunisian targets in `TARGETS`. That needs the app to capture the found location when a report is marked reunited, and coordinates for sightings.

## Licence and attribution

Map-derived layers are built from OpenStreetMap data, © OpenStreetMap contributors, under the **Open Database License (ODbL 1.0)**. If you publish the grids or the dataset, credit OpenStreetMap and share database changes under the ODbL. Generated cases are synthetic and contain no personal data.

References:
- Huang L, Coradini M, Rand J, et al. *Search Methods Used to Locate Missing Cats and Locations Where Missing Cats Are Found.* Animals 2018;8(1):5.
- Lord LK, Wittum TE, Ferketich AK, Funk JA, Rajala-Schultz PJ. *Search and identification methods that owners use to find a lost dog.* JAVMA 2007;230(2):211–216.
- Kays R, et al. *The small home ranges and large local ecological impacts of pet cats.* Animal Conservation 2020. Movebank "Pet Cats" data, CC0 (useful for future validation of movement).

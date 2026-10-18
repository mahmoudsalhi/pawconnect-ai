"""Build the M7 lost-pet location dataset.

    python build_dataset.py                    # full build (download, calibrate, simulate, validate)
    python build_dataset.py --scale 0.05       # quick smoke run with 5% of the cases
    python build_dataset.py --recalibrate      # re-tune even if output/calibration.json exists
    python build_dataset.py --regions sousse   # only some regions

Outputs (output/):
    cases.parquet / cases.csv   one row per lost pet: profile, when/where lost, outcome, where found
    sightings.parquet           public sighting reports (noisy, including false ones)
    tracks.parquet              ground-truth positions over time (hourly for 3 days, then 6-hourly)
    training_candidates.parquet ready-to-train cell-ranking table (see features.py)
    validation.md + plots/      targets vs. achieved, distributions, example maps
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import calibrate
import config as C
import features
import osm
import raster
import validate
from simulate import GROUPS, OUTCOMES, O, STATE_NAMES, sample_cases, simulate

REGION_CODE = {"grand_tunis": "GT", "sousse": "SO", "sfax": "SF", "nabeul_hammamet": "NH"}
GOV_CENTROIDS = {"Tunis": (36.80, 10.18), "Ariana": (36.87, 10.18), "Ben Arous": (36.74, 10.23),
                 "Manouba": (36.81, 10.09), "Sousse": (35.83, 10.61), "Sfax": (34.74, 10.76),
                 "Nabeul": (36.45, 10.73)}
CHUNK = 3000


def _governorate(lat, lng, options):
    best = np.full(lat.shape, options[0], dtype=object)
    best_d = np.full(lat.shape, np.inf)
    for g in options:
        glat, glng = GOV_CENTROIDS[g]
        d = (lat - glat) ** 2 + ((lng - glng) * 0.8) ** 2
        best[d < best_d] = g
        best_d = np.minimum(best_d, d)
    return best


def _bearing(dx, dy_south):
    """Compass bearing in degrees (0 = north, 90 = east) from local dx/dy (y pointing south)."""
    return (np.degrees(np.arctan2(dx, -dy_south)) + 360) % 360


def run_region(region: str, cfg: dict, grid, layers, params, n: int, seed: int):
    rng = np.random.default_rng(seed)
    case_frames, track_frames, sighting_frames = [], [], []
    code = REGION_CODE[region]
    done = 0
    while done < n:
        m = min(CHUNK, n - done)
        cases = sample_cases(m, rng, grid, layers)
        res = simulate(cases, grid, layers, params, rng, record=True, temp_offset=cfg["temp_offset"])
        ids = np.array([f"{code}-{done + i + 1:06d}" for i in range(m)])
        case_frames.append(_cases_frame(region, cfg, grid, layers, cases, res, ids, rng))
        track_frames.append(_tracks_frame(grid, cases, res, ids))
        sighting_frames.append(_sightings_frame(grid, layers, cases, res, ids, rng))
        done += m
        print(f"  [{region}] simulated {done:,}/{n:,}", flush=True)
    df = pd.concat(case_frames, ignore_index=True)
    sightings = pd.concat(sighting_frames, ignore_index=True)
    counts = sightings.groupby("case_id").agg(n_sightings=("is_false", "size"),
                                              n_true_sightings=("is_false", lambda s: int((~s).sum())))
    df = df.join(counts, on="case_id")
    df[["n_sightings", "n_true_sightings"]] = df[["n_sightings", "n_true_sightings"]].fillna(0).astype(int)
    return df, pd.concat(track_frames, ignore_index=True), sightings


def _cases_frame(region, cfg, grid, layers, cases, res, ids, rng):
    n = cases["n"]
    lat_h, lng_h = grid.to_latlng(cases["home_x"], cases["home_y"])
    lat_o, lng_o = grid.to_latlng(cases["origin_x"], cases["origin_y"])
    lat_r, lng_r = grid.to_latlng(cases["reported_x"], cases["reported_y"])
    lat_old, lng_old = grid.to_latlng(cases["old_home_x"], cases["old_home_y"])
    found = ~np.isnan(res["end_x"])
    # Reported found location: people drop a pin near where they picked the pet up.
    fsig = np.where(res["outcome"] == O["returned_home"], 3.0, 10.0)
    frx = res["end_x"] + rng.normal(0, 1, n) * fsig
    fry = res["end_y"] + rng.normal(0, 1, n) * fsig
    lat_f, lng_f = grid.to_latlng(frx, fry)
    lat_ft, lng_ft = grid.to_latlng(res["end_x"], res["end_y"])
    dx, dy = frx - cases["reported_x"], fry - cases["reported_y"]
    lost_at = [datetime(int(y), int(mo), int(d)) + timedelta(hours=float(h))
               for y, mo, d, h in zip(cases["year"], cases["month"], cases["day"], cases["hour"])]
    found_at = [la + timedelta(hours=float(t)) if f else pd.NaT for la, t, f in zip(lost_at, res["t_end"], found)]
    temp = np.array(C.MONTH_TEMP_C)[cases["month"] - 1] + cfg["temp_offset"] + 5 * np.cos(2 * np.pi * (cases["hour"] - 15) / 24)

    r = np.clip((np.nan_to_num(res["end_y"]) // grid.cell).astype(int), 0, grid.H - 1)
    c = np.clip((np.nan_to_num(res["end_x"]) // grid.cell).astype(int), 0, grid.W - 1)
    place = raster.place_type(layers, r, c)
    place[res["outcome"] == O["returned_home"]] = "home"
    place[~found] = None

    outcome = np.array(OUTCOMES, dtype=object)[res["outcome"]]
    return pd.DataFrame({
        "case_id": ids,
        "region": region,
        "governorate": _governorate(lat_o, lng_o, cfg["governorates"]),
        "species": cases["species"], "lifestyle": cases["lifestyle"], "group": np.array(GROUPS)[cases["grp"]],
        "temperament": cases["temperament"], "age_group": cases["age_group"], "sex": cases["sex"],
        "neutered": cases["neutered"], "size": cases["size"], "breed_group": cases["breed_group"],
        "collar_tag": cases["collar_tag"], "microchipped": cases["microchipped"],
        "circumstance": cases["circumstance"],
        "lost_at": pd.to_datetime(lost_at), "month": cases["month"],
        "weekday": [d.strftime("%a") for d in lost_at], "hour_of_day": np.floor(cases["hour"]).astype(int),
        "temp_c_at_loss": np.round(temp, 1),
        "last_seen_lat": lat_r, "last_seen_lng": lng_r, "last_seen_accuracy_m": cases["last_seen_sigma_m"],
        "home_lat": lat_h, "home_lng": lng_h,
        "home_distance_m": np.hypot(cases["home_x"] - cases["reported_x"], cases["home_y"] - cases["reported_y"]).round(1),
        "old_home_lat": lat_old, "old_home_lng": lng_old,
        "outcome": outcome, "found": found,
        "found_alive": found & (res["outcome"] != O["found_dead"]),
        "found_at": pd.to_datetime(found_at), "hours_to_found": np.where(found, res["t_end"].round(2), np.nan),
        "found_lat": lat_f, "found_lng": lng_f,
        "found_distance_m": np.where(found, np.hypot(dx, dy), np.nan).round(1),
        "found_bearing_deg": np.where(found, _bearing(dx, dy), np.nan).round(1),
        "found_place_type": place,
        "followed_up_hours": np.where(found, res["t_end"], C.MAX_DAYS * 24.0),
        # Ground truth the platform would never see; keep for analysis, never as model inputs.
        "true_origin_lat": lat_o, "true_origin_lng": lng_o,
        "found_true_lat": lat_ft, "found_true_lng": lng_ft,
        "relocated_by_human": res["relocated"], "died_unfound": res["died_unfound"],
        "path_length_m": res["path_len"].round(0), "max_distance_m": res["max_d"].round(0),
    })


def _tracks_frame(grid, cases, res, ids):
    t = res["tracks"]
    if not len(t):
        return pd.DataFrame()
    i = t[:, 0].astype(int)
    lat, lng = grid.to_latlng(t[:, 2], t[:, 3])
    return pd.DataFrame({
        "case_id": ids[i], "t_hours": t[:, 1].astype(np.float32),
        "lat": lat, "lng": lng, "state": STATE_NAMES[t[:, 4].astype(int)],
        "distance_m": np.hypot(t[:, 2] - cases["reported_x"][i], t[:, 3] - cases["reported_y"][i]).astype(np.float32),
    })


def _sightings_frame(grid, layers, cases, res, ids, rng):
    s = res["sightings"]
    i = s[:, 0].astype(int)
    acc = rng.choice([15, 40, 100, 250], i.size, p=[0.3, 0.35, 0.25, 0.1]).astype(float)
    rx = s[:, 2] + rng.normal(0, 1, i.size) * acc
    ry = s[:, 3] + rng.normal(0, 1, i.size) * acc
    true = pd.DataFrame({"i": i, "t": s[:, 1], "x": rx, "y": ry, "tx": s[:, 2], "ty": s[:, 3],
                         "acc": acc, "is_false": False})

    # False sightings: a similar-looking animal somewhere busy near the last-seen point.
    days = res["t_end"] / 24
    lam = C.HAZARD["false_sightings_per_day"] * 7 * (1 - np.exp(-days / 7))
    k = rng.poisson(lam)
    fi = np.repeat(np.arange(cases["n"]), k)
    ppl = layers["people"].ravel().astype(np.float64)
    pool = rng.choice(ppl.size, 50_000, p=ppl / ppl.sum())
    cand = pool[rng.integers(0, pool.size, (fi.size, 40))]
    cy = (cand // grid.W + 0.5) * grid.cell
    cx = (cand % grid.W + 0.5) * grid.cell
    d = np.hypot(cx - cases["reported_x"][fi, None], cy - cases["reported_y"][fi, None])
    ok = d < 4000
    j = np.where(ok.any(axis=1), ok.argmax(axis=1), 0)
    ar = np.arange(fi.size)
    fx, fy = cx[ar, j], cy[ar, j]
    false = pd.DataFrame({"i": fi, "t": rng.random(fi.size) * res["t_end"][fi], "x": fx, "y": fy,
                          "tx": np.nan, "ty": np.nan, "acc": rng.choice([40, 100, 250], fi.size).astype(float),
                          "is_false": True})
    df = pd.concat([true, false], ignore_index=True).sort_values(["i", "t"])
    lat, lng = grid.to_latlng(df["x"].to_numpy(), df["y"].to_numpy())
    tlat, tlng = grid.to_latlng(df["tx"].to_numpy(), df["ty"].to_numpy())
    lost_at = pd.to_datetime([datetime(int(y), int(mo), int(dd)) + timedelta(hours=float(h)) for y, mo, dd, h in
                              zip(cases["year"], cases["month"], cases["day"], cases["hour"])])
    seen = lost_at[df["i"].to_numpy()] + pd.to_timedelta(df["t"].to_numpy(), unit="h")
    delay = pd.to_timedelta(rng.exponential(3.0, len(df)), unit="h")
    out = pd.DataFrame({
        "case_id": ids[df["i"].to_numpy()],
        "seen_at": seen, "reported_at": seen + delay,
        "hours_since_lost": df["t"].to_numpy().round(2),
        "lat": lat, "lng": lng, "accuracy_m": df["acc"].to_numpy(),
        "distance_m": np.hypot(df["x"].to_numpy() - cases["reported_x"][df["i"].to_numpy()],
                               df["y"].to_numpy() - cases["reported_y"][df["i"].to_numpy()]).round(1),
        "is_false": df["is_false"].to_numpy(),
        "true_lat": tlat, "true_lng": tlng,
    })
    out.insert(1, "sighting_no", out.groupby("case_id").cumcount() + 1)
    return out


def assign_splits(df: pd.DataFrame, seed: int) -> pd.Series:
    """Spatial-block split (3 km blocks) so test cases come from places the model never saw."""
    lat, lng = df["true_origin_lat"].to_numpy(), df["true_origin_lng"].to_numpy()
    block = (df["region"] + ":" + (lat / 0.027).astype(int).astype(str) + ":" + (lng / 0.033).astype(int).astype(str))
    sizes = block.value_counts()
    order = np.random.default_rng(seed).permutation(sizes.index.to_numpy())
    cum = sizes.loc[order].cumsum() / len(df)
    split = pd.Series(np.where(cum <= 0.70, "train", np.where(cum <= 0.85, "val", "test")), index=order)
    return block.map(split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="*", default=list(C.REGIONS))
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of the configured case counts")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--rebuild-grids", action="store_true")
    ap.add_argument("--no-features", action="store_true")
    args = ap.parse_args()
    C.OUTPUT.mkdir(parents=True, exist_ok=True)

    grids = {}
    for region in C.REGIONS:  # calibration always uses every region
        cfg = C.REGIONS[region]
        loaded = None if args.rebuild_grids else raster.load(region, cfg["bbox"])
        if loaded is None:
            print(f"[{region}] fetching OSM and rasterising", flush=True)
            loaded = raster.build(region, cfg["bbox"], osm.fetch(region, cfg["bbox"]))
        grids[region] = loaded
        g = loaded[0]
        print(f"[{region}] grid {g.H}x{g.W} cells of {g.cell:.0f} m", flush=True)

    params = None if args.recalibrate else calibrate.load()
    if params is None:
        print("[calibration] tuning against published targets", flush=True)
        params, _ = calibrate.calibrate([(*grids[r], C.REGIONS[r]["temp_offset"]) for r in C.REGIONS])
    print("[calibration] " + json.dumps(params.to_dict()), flush=True)

    all_cases, all_tracks, all_sightings = [], [], []
    for k, region in enumerate(args.regions):
        cfg = C.REGIONS[region]
        n = max(50, int(cfg["cases"] * args.scale))
        df, tr, si = run_region(region, cfg, *grids[region], params, n, args.seed + 1000 * k)
        all_cases.append(df), all_tracks.append(tr), all_sightings.append(si)

    cases = pd.concat(all_cases, ignore_index=True)
    cases.insert(2, "split", assign_splits(cases, args.seed).to_numpy())
    tracks = pd.concat(all_tracks, ignore_index=True)
    sightings = pd.concat(all_sightings, ignore_index=True)

    cases.to_parquet(C.OUTPUT / "cases.parquet", index=False)
    cases.to_csv(C.OUTPUT / "cases.csv", index=False)
    tracks.to_parquet(C.OUTPUT / "tracks.parquet", index=False)
    sightings.to_parquet(C.OUTPUT / "sightings.parquet", index=False)
    print(f"[output] cases={len(cases):,} tracks={len(tracks):,} sightings={len(sightings):,}", flush=True)

    if not args.no_features:
        feats = features.build(cases, tracks, sightings, {r: grids[r] for r in args.regions}, seed=args.seed)
        feats.to_parquet(C.OUTPUT / "training_candidates.parquet", index=False)
        print(f"[output] training_candidates={len(feats):,} rows", flush=True)

    validate.report(cases, tracks, sightings, {r: grids[r] for r in args.regions}, params)
    print(f"[done] see {C.OUTPUT / 'validation.md'}", flush=True)


if __name__ == "__main__":
    main()

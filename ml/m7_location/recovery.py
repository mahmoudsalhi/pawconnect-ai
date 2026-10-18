"""M7 recovery outlook: will a lost pet be found alive, and when?

    python recovery.py        # train on output/cases.parquet, evaluate, write output/model/m7_recovery*

Framed as a multiclass problem over 14 time bins (3 h, 6 h, 12 h, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60 days),
plus "not found alive within 60 days" (still missing or died). Fine bins keep the "typical wait" accurate. Every
simulated case is followed for the full 60 days, so there is no censoring to correct for.
The bin probabilities give the whole recovery curve; at inference it is conditioned on how long
the pet has already been missing.

Inputs are what a report knows at filing time plus the surroundings of the last-seen point
(foot traffic, cover, food, roads, water from the OSM grids). As in train.py, masked copies
teach the model to work when temperament, lifestyle, circumstance, collar, neutering are unknown.

How much to trust it: overall recovery levels are calibrated to published studies (Huang 2018,
Lord 2007); how each profile detail shifts them comes from the simulator's assumptions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import config as C
import features
import raster

UNKNOWN = "unknown"
BIN_EDGES_H = [0, 3, 6, 12, 24, 48, 72, 120, 168, 240, 336, 504, 720, 1080, 1440]
N_BINS = len(BIN_EDGES_H) - 1
NOT_FOUND = N_BINS  # class index
MODEL_DIR = C.OUTPUT / "model"

CATEGORICAL = {
    "species": ["cat", "dog"],
    "group": ["cat_indoor", "cat_outdoor", "cat_unknown", "dog"],
    "temperament": ["curious", "cautious", "xenophobic", "gregarious", "aloof", UNKNOWN],
    "age_group": ["kitten", "puppy", "adult", "senior", UNKNOWN],
    "sex": ["male", "female", UNKNOWN],
    "size": ["small", "medium", "large", UNKNOWN],
    "breed_group": ["domestic_cat", "local_mixed", "companion", "terrier", "herding", "guard", "hound_sighthound", UNKNOWN],
    "circumstance": ["door_window_escape", "balcony_fall", "moved_house", "transport_escape", "fireworks_noise",
                     "did_not_return", "escaped_home_yard", "slipped_leash_walk", "car_accident", "followed_dogs", UNKNOWN],
    "collar": ["yes", "no", UNKNOWN],
    "chip": ["yes", "no", UNKNOWN],
    "neutered_s": ["yes", "no", UNKNOWN],
}
ENV = ["people", "people_200m", "cover", "cover_200m", "food_200m", "building_200m", "green_200m", "dist_major_m",
       "water"]
NUMERIC = ["hour_of_day", "month", "temp_c_at_loss", *[f"env_{k}" for k in ENV]]
FEATURES = [*NUMERIC, *CATEGORICAL]


def _yn(v):
    s = str(v).lower()
    return "yes" if s in ("true", "1", "yes") else "no" if s in ("false", "0", "no") else UNKNOWN


def env_at(lay: dict, grid, lat, lng) -> dict:
    """Surroundings of the last-seen point(s)."""
    x, y = grid.to_xy(np.asarray(lat, float), np.asarray(lng, float))
    r = np.clip((y // grid.cell).astype(int), 0, grid.H - 1)
    c = np.clip((x // grid.cell).astype(int), 0, grid.W - 1)
    return {f"env_{k}": lay[k][r, c].astype(np.float32) for k in ENV}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, src in (("collar", "collar_tag"), ("chip", "microchipped"), ("neutered_s", "neutered")):
        if col not in df:
            df[col] = df[src].map(_yn) if src in df else UNKNOWN
    for k, cats in CATEGORICAL.items():
        v = df[k].astype(str).str.lower() if k in df else pd.Series(UNKNOWN, index=df.index)
        df[k] = pd.Categorical(v.where(v.isin(cats), UNKNOWN), categories=cats)
    for k in NUMERIC:
        df[k] = pd.to_numeric(df[k], errors="coerce") if k in df else np.nan
    return df


def labels(cases: pd.DataFrame) -> np.ndarray:
    h = cases["hours_to_found"].to_numpy()
    y = np.full(len(cases), NOT_FOUND)
    alive = cases["found_alive"].to_numpy()
    for b in range(N_BINS):
        y[alive & (h >= BIN_EDGES_H[b]) & (h < BIN_EDGES_H[b + 1])] = b
    y[alive & (h >= BIN_EDGES_H[-1])] = NOT_FOUND
    return y


def mask(df: pd.DataFrame, rng, p: float = 0.3, minimal: bool = False) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    for k in ("temperament", "circumstance", "collar", "neutered_s", "age_group"):
        m = np.ones(n, bool) if minimal and k != "age_group" else rng.random(n) < p
        df.loc[m, k] = UNKNOWN
    m = (np.ones(n, bool) if minimal else rng.random(n) < p) & (df["species"] == "cat").to_numpy()
    df.loc[m, "group"] = "cat_unknown"
    m = rng.random(n) < p
    df.loc[m, "chip"] = UNKNOWN
    return df


def load_frame() -> pd.DataFrame:
    cases = pd.read_parquet(C.OUTPUT / "cases.parquet")
    cases["y"] = labels(cases)
    envs = []
    for region, g in cases.groupby("region"):
        grid, layers = raster.load(region, C.REGIONS[region]["bbox"])
        e = pd.DataFrame(env_at(features.prepare_layers(layers), grid, g["last_seen_lat"], g["last_seen_lng"]), index=g.index)
        envs.append(e)
    cases = cases.join(pd.concat(envs))
    cases["collar"] = cases["collar_tag"].map(_yn)
    cases["chip"] = cases["microchipped"].map(_yn)
    cases["neutered_s"] = cases["neutered"].map(_yn)
    return cases


# --- inference helpers (shared with the backend) --------------------------------------------
def cdf_at(probs: np.ndarray, hours: float) -> float:
    """P(found alive by `hours`), interpolating inside a bin on a log(1 + h) scale."""
    if hours <= 0:
        return 0.0
    cum = np.r_[0.0, np.cumsum(probs[:N_BINS])]
    if hours >= BIN_EDGES_H[-1]:
        return float(cum[-1])
    b = int(np.searchsorted(BIN_EDGES_H, hours, side="right") - 1)
    lo, hi = np.log1p(BIN_EDGES_H[b]), np.log1p(BIN_EDGES_H[b + 1])
    f = (np.log1p(hours) - lo) / (hi - lo)
    return float(cum[b] + f * (cum[b + 1] - cum[b]))


def outlook(probs: np.ndarray, hours_missing: float) -> dict:
    """Chances from *now*, given the pet has not been found in the first `hours_missing` hours."""
    t = float(hours_missing)
    horizon = BIN_EDGES_H[-1]
    f_t = cdf_at(probs, t)
    still = max(1.0 - f_t, 1e-9)
    found_eventually = (cdf_at(probs, horizon) - f_t) / still

    def within(extra):
        return (cdf_at(probs, min(t + extra, horizon)) - f_t) / still

    median = None
    if found_eventually > 0.02 and t < horizon:
        target = f_t + 0.5 * (cdf_at(probs, horizon) - f_t)
        lo, hi = t, float(horizon)
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if cdf_at(probs, mid) < target else (lo, mid)
        median = hi - t
    return {
        "found_alive": round(found_eventually, 3),
        "found_alive_at_start": round(cdf_at(probs, horizon), 3),
        "within_24h": round(within(24), 3),
        "within_7d": round(within(168), 3),
        "median_hours_if_found": None if median is None else round(median, 1),
        "beyond_horizon": t >= horizon,
        "horizon_days": horizon // 24,
    }


class RecoveryModel:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.booster = lgb.Booster(model_file=str(Path(model_dir) / "m7_recovery_lgbm.txt"))

    def probs(self, df: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(prepare(df)[FEATURES])


# --- training ----------------------------------------------------------------------------------
def main(seed: int = 7):
    rng = np.random.default_rng(seed)
    cases = load_frame()
    tr, va, te = (cases[cases["split"] == s] for s in ("train", "val", "test"))
    tr_all = pd.concat([tr, mask(tr, rng), mask(tr, rng, minimal=True)], ignore_index=True)
    Xtr, Xva = prepare(tr_all)[FEATURES], prepare(va)[FEATURES]
    dtr = lgb.Dataset(Xtr, label=tr_all["y"].to_numpy())
    dva = lgb.Dataset(Xva, label=va["y"].to_numpy(), reference=dtr)
    params = {"objective": "multiclass", "num_class": N_BINS + 1, "learning_rate": 0.03, "num_leaves": 31,
              "min_data_in_leaf": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
              "lambda_l2": 2.0, "verbose": -1, "seed": seed}
    booster = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva], valid_names=["val"],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODEL_DIR / "m7_recovery_lgbm.txt"), num_iteration=booster.best_iteration)

    # Evaluation on held-out test areas, with full and with today's minimal report.
    y = te["y"].to_numpy()
    base = np.bincount(tr["y"], minlength=N_BINS + 1) / len(tr)
    rows = []
    for variant, X in (("full", te), ("minimal", mask(te, rng, minimal=True))):
        P = booster.predict(prepare(X)[FEATURES], num_iteration=booster.best_iteration)
        ll = -np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)).mean()
        # Baseline: species-level class frequencies.
        Pb = np.vstack([np.bincount(tr.loc[tr["species"] == s, "y"], minlength=N_BINS + 1) / (tr["species"] == s).sum()
                        for s in te["species"]])
        llb = -np.log(np.clip(Pb[np.arange(len(y)), y], 1e-9, 1)).mean()
        found = (y < NOT_FOUND).astype(float)
        week = (y <= 3).astype(float)
        p_found = P[:, :N_BINS].sum(axis=1)
        p_week = P[:, :4].sum(axis=1)

        def auc(score, target):
            o = np.argsort(score)
            ranks = np.empty(len(o))
            ranks[o] = np.arange(1, len(o) + 1)
            pos = target == 1
            return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))

        dec = pd.qcut(p_found, 5, labels=False, duplicates="drop")
        calib = pd.DataFrame({"predicted": p_found, "actual": found, "q": dec}).groupby("q").mean().round(3)
        rows.append({"variant": variant, "logloss": round(ll, 4), "logloss_species_baseline": round(llb, 4),
                     "auc_found_alive_60d": round(auc(p_found, found), 3), "auc_found_within_7d": round(auc(p_week, week), 3),
                     "brier_found": round(float(((p_found - found) ** 2).mean()), 4),
                     "calibration_by_quintile": calib.to_dict(orient="records")})
    meta = {"model": "M7 recovery outlook", "version": "0.1.0-synthetic", "trained_at": datetime.now(timezone.utc).isoformat(),
            "bins_hours": BIN_EDGES_H, "features": FEATURES, "best_iteration": booster.best_iteration,
            "class_frequencies_train": base.round(4).tolist(), "test_metrics": rows}
    (MODEL_DIR / "m7_recovery_meta.json").write_text(json.dumps(meta, indent=2))
    imp = pd.Series(booster.feature_importance("gain"), index=booster.feature_name()).sort_values(ascending=False)
    print((imp / imp.sum()).head(12).round(3).to_string())
    for r in rows:
        print({k: v for k, v in r.items() if k != "calibration_by_quintile"})
        print("   calibration (predicted vs actual found alive):", r["calibration_by_quintile"])


if __name__ == "__main__":
    main()

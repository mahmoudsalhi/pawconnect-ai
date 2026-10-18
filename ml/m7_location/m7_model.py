"""M7 inference: turn a lost-pet report into a probability map.

The model is a LightGBM booster that scores one map cell at a time; its raw score is
(up to a constant) the log probability density of the pet being in that cell. Scoring
every cell of an equal-area lattice around the last-seen point and taking a softmax gives
the heat map. Used by train.py for evaluation and meant to be used by the backend.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import config as C
import features

UNKNOWN = "unknown"
CATEGORICAL = {
    "species": ["cat", "dog"],
    "group": ["cat_indoor", "cat_outdoor", "cat_unknown", "dog"],
    "temperament": ["curious", "cautious", "xenophobic", "gregarious", "aloof", UNKNOWN],
    "age_group": ["kitten", "puppy", "adult", "senior", UNKNOWN],
    "size": ["small", "medium", "large", UNKNOWN],
    "breed_group": ["domestic_cat", "local_mixed", "companion", "terrier", "herding", "guard", "hound_sighthound", UNKNOWN],
    "circumstance": ["door_window_escape", "balcony_fall", "moved_house", "transport_escape", "fireworks_noise",
                     "did_not_return", "escaped_home_yard", "slipped_leash_walk", "car_accident", "followed_dogs", UNKNOWN],
    "collar": ["yes", "no", UNKNOWN],
}
NUMERIC = [
    "t_hours", "log_t", "dist_m", "log_dist", "dist_per_sqrt_t", "hour_of_day", "month", "temp_c_at_loss",
    "last_seen_accuracy_m", "home_distance_m", "log_dist_home", "toward_home_cos",
    *features.CELL_FEATS, "cover_200m", "people_200m", "food_200m", "building_200m", "green_200m",
    "major_crossings", "water_on_line",
    "nearest_sighting_m", "latest_sighting_m", "sightings_within_300m", "n_sightings_known",
]
FEATURES = [*NUMERIC, *CATEGORICAL]
LATTICE_M = {"cat": 50.0, "dog": 100.0}  # evaluation / inference resolution
RADIUS_M = {"cat": 4000.0, "dog": 12000.0}
DIST_FLOOR_M = 25.0  # distance features are clamped to [DIST_FLOOR_M, R]


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns and fix categorical dtypes. Works on training rows and on lattices."""
    df = df.copy()
    # Negatives are never sampled closer than min_r_m (10 m) to the last-seen point, so raw
    # distances below it only ever belong to positives: a perfect separator the model turns
    # into a spike on the centre cell. Clamping at exactly 10 m doesn't help (positives pile up
    # on 10.0, negatives almost never do), so clamp at a floor that plenty of negatives also
    # fall under (~15% are within 25 m) - then both share the floor value.
    rmin = DIST_FLOOR_M
    R = np.where(df["species"].astype(str).to_numpy() == "dog", RADIUS_M["dog"], RADIUS_M["cat"])
    # Same at the far end: pets beyond R would pile up on R, so cap at 0.8 R (~4% of negatives lie above it).
    df["dist_m"] = df["dist_m"].clip(lower=rmin, upper=0.8 * R)
    df["dist_home_m"] = df["dist_home_m"].clip(lower=rmin)
    t = df["t_hours"].astype(np.float32).clip(lower=0)
    df["log_t"] = np.log1p(t)
    df["log_dist"] = np.log1p(df["dist_m"])
    df["dist_per_sqrt_t"] = df["dist_m"] / np.sqrt(t + 1)
    df["log_dist_home"] = np.log1p(df["dist_home_m"])
    if "collar" not in df:
        ct = df["collar_tag"].astype(str).str.lower()
        df["collar"] = np.where(ct.isin(["true", "1", "yes"]), "yes", np.where(ct.isin(["false", "0", "no"]), "no", UNKNOWN))
    for k, cats in CATEGORICAL.items():
        df[k] = pd.Categorical(df[k].astype(str).where(df[k].astype(str).isin(cats), UNKNOWN), categories=cats)
    return df


class M7:
    def __init__(self, model_dir: Path):
        model_dir = Path(model_dir)
        self.booster = lgb.Booster(model_file=str(model_dir / "m7_lgbm.txt"))
        self.meta = json.loads((model_dir / "m7_meta.json").read_text())

    def score(self, df: pd.DataFrame, cell_m: float | None = None, dx=None, dy=None) -> np.ndarray:
        """Log density (up to a constant): tree output + log of the training proposal q.

        q falls off like 1/distance² near the last-seen point, so when scoring lattice cells
        (cell_m given, dx/dy = cell centre offsets from the last-seen point) q is averaged
        over the cell area for cells near the centre instead of read at the centre point,
        otherwise the centre cell would get a wildly inflated share of the probability.
        """
        raw = self.booster.predict(derive(df)[FEATURES], raw_score=True)
        R = np.where(df["species"].to_numpy() == "cat", RADIUS_M["cat"], RADIUS_M["dog"])
        d = df["dist_m"].to_numpy(np.float64)
        log_q = features.proposal_log_q(d, R)
        if cell_m is not None and dx is not None:
            near = d < 4 * cell_m
            if near.any():
                sub = (np.arange(9) - 4) / 9 * cell_m
                sx, sy = np.meshgrid(sub, sub)
                ds = np.hypot(dx[near, None] + sx.ravel()[None], dy[near, None] + sy.ravel()[None])
                log_q[near] = np.log(np.exp(features.proposal_log_q(ds, R[near, None])).mean(axis=1))
        return raw + log_q

    def predict_grid(self, grid, layers, lay, case: dict, t_hours: float, sightings=None,
                     step_m: float | None = None, radius_m: float | None = None, extra_points=None) -> dict:
        """Probability map for one case.

        case: the CASE_COLS from features.py plus last_seen_lat/lng and home_lat/lng
              (unknown categorical values may be "unknown").
        sightings: (sx, sy, reported_hours) in grid metres, already filtered to t_hours.
        extra_points: optional (x, y) arrays to score as well (e.g. the true location in eval).
        """
        sp = case["species"]
        step = step_m or LATTICE_M[sp]
        R = radius_m or RADIUS_M[sp]
        ox, oy = grid.to_xy(case["last_seen_lat"], case["last_seen_lng"])
        hx, hy = grid.to_xy(case["home_lat"], case["home_lng"])
        ax = np.arange(-R, R + step, step)
        gx, gy = np.meshgrid(ax, ax)
        inside = gx ** 2 + gy ** 2 <= R ** 2
        # Only cells on the mapped region (outside it, features would just repeat the edge).
        inside &= (float(ox) + gx >= 0) & (float(ox) + gx < grid.W * grid.cell)
        inside &= (float(oy) + gy >= 0) & (float(oy) + gy < grid.H * grid.cell)
        cx = (float(ox) + gx[inside])
        cy = (float(oy) + gy[inside])
        n_lat = cx.size
        if extra_points is not None:
            cx = np.r_[cx, extra_points[0]]
            cy = np.r_[cy, extra_points[1]]
        cols = features.cell_features(grid, layers, lay, np.atleast_1d(ox).astype(float), np.atleast_1d(oy).astype(float),
                                      np.atleast_1d(hx).astype(float), np.atleast_1d(hy).astype(float),
                                      cx[None], cy[None], [sightings])
        df = pd.DataFrame({k: v[0] for k, v in cols.items()})
        for k in features.CASE_COLS:
            df[k] = case.get(k, UNKNOWN if k in CATEGORICAL or k == "collar_tag" else np.nan)
        df["t_hours"] = float(t_hours)
        dx = np.r_[gx[inside], np.asarray(cx[n_lat:]) - float(ox)]
        dy = np.r_[gy[inside], np.asarray(cy[n_lat:]) - float(oy)]
        s = self.score(df, cell_m=step, dx=dx, dy=dy)
        lat_s = s[:n_lat]
        p = np.exp(lat_s - lat_s.max())
        p /= p.sum()
        lat, lng = grid.to_latlng(cx[:n_lat], cy[:n_lat])
        return {"lat": lat, "lng": lng, "x": cx[:n_lat], "y": cy[:n_lat], "prob": p, "score": lat_s,
                "extra_score": s[n_lat:], "step_m": step}

"""Features for M7, shared by training-table generation, evaluation and inference.

M7 is framed as: given what the platform knows at time t (pet profile, last-seen point,
home, sightings reported so far), score every map cell by how likely the pet is there.
Each *query* is one (case, t) pair with one positive cell (where the pet actually was) and
N negatives drawn from a fixed proposal around the last-seen point: distance log-uniform
in [min_r, R], direction uniform. `log_q` is the proposal's log density per m² at each
candidate, so a trainer can use sampled-softmax / importance correction and get a model
that scores *all* cells correctly at inference.

Query sources:
  recovery  t = time found, positive = reported found location (what real data will look like)
  track     t = a snapshot time, positive = true simulated position (extra supervision)

Only information available at query time is used as features. Ground-truth columns
(true_*, outcome, died_unfound...) are never copied in.

`cell_features()` is the single implementation used everywhere, so training and inference
cannot drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

import config as C

CELL_FEATS = ["cover", "food", "people", "corridor", "green", "building", "farm", "major", "exposed",
              "water", "wild", "residential", "commercial", "construction", "cemetery", "vet", "dist_major_m"]
CASE_COLS = ["species", "group", "temperament", "age_group", "size", "breed_group", "circumstance",
             "collar_tag", "hour_of_day", "month", "temp_c_at_loss", "last_seen_accuracy_m", "home_distance_m"]
LINE_SAMPLES = 24


def prepare_layers(layers: dict) -> dict:
    """Cell layers plus ~200 m neighbourhood averages of a few key ones."""
    ctx = {f"{k}_200m": ndimage.gaussian_filter(layers[k], 5.0)
           for k in ("cover", "people", "food", "building", "green")}
    return {**{k: layers[k] for k in CELL_FEATS}, **ctx}


def cell_features(grid, layers: dict, lay: dict, ox, oy, hx, hy, cx, cy, sightings: list | None = None) -> dict:
    """Features for candidate cells.

    ox, oy, hx, hy: (nq,) last-seen and home positions in the grid's metric frame.
    cx, cy: (nq, M) candidate positions.
    sightings: per query, None or (sx, sy, reported_hours) of sightings known at query time.
    """
    nq = cx.shape[0]
    W, H = grid.W * grid.cell, grid.H * grid.cell
    cx, cy = np.clip(cx, 0, W - 1), np.clip(cy, 0, H - 1)
    dist = np.hypot(cx - ox[:, None], cy - oy[:, None])
    r = np.clip((cy // grid.cell).astype(int), 0, grid.H - 1)
    c = np.clip((cx // grid.cell).astype(int), 0, grid.W - 1)
    cols = {"dist_m": dist.astype(np.float32)}
    cols.update({k: v[r, c].astype(np.float32) for k, v in lay.items()})

    # Straight-line barriers between the last-seen point and the candidate.
    fr = np.linspace(0, 1, LINE_SAMPLES, dtype=np.float32)[None, None, :]
    crossings = np.zeros(cx.shape, np.float32)
    wet_line = np.zeros(cx.shape, np.float32)
    step = max(1, 64_000 // max(cx.shape[1], 1))
    for a0 in range(0, nq, step):
        sl = slice(a0, a0 + step)
        lx = ox[sl, None, None] + (cx[sl, :, None] - ox[sl, None, None]) * fr
        ly = oy[sl, None, None] + (cy[sl, :, None] - oy[sl, None, None]) * fr
        lr = np.clip((ly // grid.cell).astype(np.int32), 0, grid.H - 1)
        lc = np.clip((lx // grid.cell).astype(np.int32), 0, grid.W - 1)
        maj = layers["major"][lr, lc] > 0
        crossings[sl] = (np.diff(maj.astype(np.int8), axis=2) == 1).sum(axis=2)
        wet_line[sl] = (layers["water"][lr, lc] > 0).mean(axis=2)
    cols["major_crossings"] = crossings
    cols["water_on_line"] = wet_line

    home_d = np.hypot(hx - ox, hy - oy)
    toward = (cx - ox[:, None]) * (hx - ox)[:, None] + (cy - oy[:, None]) * (hy - oy)[:, None]
    cols["toward_home_cos"] = np.where(home_d[:, None] > 1, toward / (np.maximum(dist, 1) * np.maximum(home_d, 1)[:, None]),
                                       0).astype(np.float32)
    cols["dist_home_m"] = np.hypot(cx - hx[:, None], cy - hy[:, None]).astype(np.float32)

    near = np.full(cx.shape, np.nan, np.float32)
    latest = np.full(cx.shape, np.nan, np.float32)
    within = np.zeros(cx.shape, np.float32)
    n_known = np.zeros(nq, np.int16)
    for i, s in enumerate(sightings or []):
        if s is None or not len(s[0]):
            continue
        sx, sy, rep = s
        n_known[i] = len(sx)
        dd = np.hypot(cx[i][:, None] - sx[None], cy[i][:, None] - sy[None])
        near[i] = dd.min(axis=1)
        latest[i] = dd[:, int(np.argmax(rep))]
        within[i] = (dd < 300).sum(axis=1)
    cols["nearest_sighting_m"] = near
    cols["latest_sighting_m"] = latest
    cols["sightings_within_300m"] = within
    cols["n_sightings_known"] = np.repeat(n_known[:, None], cx.shape[1], axis=1)
    return cols


def known_sightings(sg: pd.DataFrame, lost_at: pd.Series, grid, case_ids, t_hours) -> list:
    """For each query (case_id, t), the sightings reported before t, in grid coordinates."""
    if sg.empty:
        return [None] * len(case_ids)
    rep_h = (sg["reported_at"] - lost_at.loc[sg["case_id"]].to_numpy()).dt.total_seconds().to_numpy() / 3600
    sx, sy = grid.to_xy(sg["lat"].to_numpy(), sg["lng"].to_numpy())
    groups = sg.groupby("case_id").indices
    out = []
    for cid, t in zip(case_ids, t_hours):
        idx = groups.get(cid)
        if idx is None:
            out.append(None)
            continue
        idx = idx[rep_h[idx] <= t]
        out.append((sx[idx], sy[idx], rep_h[idx]) if idx.size else None)
    return out


def proposal_log_q(dist, R, rmin=C.FEATURES["min_r_m"]):
    """Log density per m² of the negative sampler at distance `dist`."""
    d = np.maximum(dist, rmin)
    return -np.log(2 * np.pi * d ** 2 * np.log(R / rmin))


def build(cases: pd.DataFrame, tracks: pd.DataFrame, sightings: pd.DataFrame, grids: dict, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    F = C.FEATURES
    frames = []
    for region, (grid, layers) in grids.items():
        cs = cases[cases["region"] == region]
        if cs.empty:
            continue
        lay = prepare_layers(layers)

        # Queries ----------------------------------------------------------------------
        rec = cs[cs["found"]]
        q_rec = pd.DataFrame({"case_id": rec["case_id"], "t_hours": rec["hours_to_found"],
                              "pos_lat": rec["found_lat"], "pos_lng": rec["found_lng"], "label_source": "recovery"})
        tr = tracks[tracks["case_id"].isin(cs["case_id"]) & tracks["t_hours"].isin(F["snapshot_hours"])]
        tr = tr.sample(frac=1.0, random_state=seed).groupby("case_id").head(F["snapshots_per_case"])
        q_tr = pd.DataFrame({"case_id": tr["case_id"], "t_hours": tr["t_hours"].astype(float),
                             "pos_lat": tr["lat"], "pos_lng": tr["lng"], "label_source": "track"})
        q = pd.concat([q_rec, q_tr], ignore_index=True).merge(
            cs[["case_id", "split", "last_seen_lat", "last_seen_lng", "home_lat", "home_lng", *CASE_COLS]], on="case_id")
        nq = len(q)
        if not nq:
            continue
        q["query_id"] = [f"{cid}@{t:.1f}" for cid, t in zip(q["case_id"], q["t_hours"])]

        ox, oy = grid.to_xy(q["last_seen_lat"].to_numpy(), q["last_seen_lng"].to_numpy())
        hx, hy = grid.to_xy(q["home_lat"].to_numpy(), q["home_lng"].to_numpy())
        px, py = grid.to_xy(q["pos_lat"].to_numpy(), q["pos_lng"].to_numpy())

        # Candidates: column 0 = positive, then negatives -------------------------------
        nn = F["negatives"]
        R = np.where(q["species"].to_numpy() == "cat", F["radius_m"]["cat"], F["radius_m"]["dog"]).astype(float)
        d = np.exp(rng.uniform(np.log(F["min_r_m"]), np.log(R)[:, None], (nq, nn)))
        a = rng.uniform(0, 2 * np.pi, (nq, nn))
        cx = np.column_stack([px, ox[:, None] + d * np.cos(a)])
        cy = np.column_stack([py, oy[:, None] + d * np.sin(a)])
        label = np.zeros((nq, nn + 1), np.int8)
        label[:, 0] = 1

        sg = sightings[sightings["case_id"].isin(cs["case_id"])]
        known = known_sightings(sg, cs.set_index("case_id")["lost_at"], grid, q["case_id"].to_numpy(), q["t_hours"].to_numpy())
        cols = cell_features(grid, layers, lay, ox, oy, hx, hy, cx, cy, known)
        log_q = proposal_log_q(cols["dist_m"], R[:, None])

        # Flatten ----------------------------------------------------------------------
        rows = nq * (nn + 1)
        W, H = grid.W * grid.cell, grid.H * grid.cell
        lat, lng = grid.to_latlng(np.clip(cx, 0, W - 1).ravel(), np.clip(cy, 0, H - 1).ravel())
        out = {"query_id": np.repeat(q["query_id"].to_numpy(), nn + 1),
               "case_id": np.repeat(q["case_id"].to_numpy(), nn + 1),
               "split": np.repeat(q["split"].to_numpy(), nn + 1),
               "region": region,
               "label_source": np.repeat(q["label_source"].to_numpy(), nn + 1),
               "t_hours": np.repeat(q["t_hours"].to_numpy(), nn + 1).astype(np.float32),
               "label": label.ravel(),
               "cand_lat": lat, "cand_lng": lng,
               "log_q": log_q.ravel().astype(np.float32)}
        for k in CASE_COLS:
            out[k] = np.repeat(q[k].to_numpy(), nn + 1)
        for k, v in cols.items():
            out[k] = v.ravel()
        frames.append(pd.DataFrame(out))
        assert len(frames[-1]) == rows
        print(f"  [features] {region}: {nq:,} queries -> {rows:,} rows", flush=True)
    return pd.concat(frames, ignore_index=True)

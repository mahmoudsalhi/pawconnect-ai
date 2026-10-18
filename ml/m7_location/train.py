"""Train M7 (lost-pet location model) on output/training_candidates.parquet.

    python train.py               # train, evaluate on full maps, write output/model/
    python train.py --eval-only   # re-evaluate the saved model

Model: LightGBM scoring one cell at a time, trained with an exact sampled-softmax objective.
Every query has 32 candidates (1 true cell + 31 from the proposal q). The model is written as
    score(cell) = f(cell) + log q(cell)          (f = the boosted trees)
so the sampled-softmax logits score - log q are simply f, and training starts from a uniform
guess over the 32 candidates. (Putting -log q into the logits instead starts the true cell at
p ~ 0, its Hessian vanishes and the Newton steps blow up.) score estimates the log density,
so a softmax over *all* cells of an equal-area lattice is the heat map.

Missing inputs: the app's current report form does not ask for temperament, cat lifestyle,
collar, circumstance or home, and sightings have no coordinates. Training includes a masked
copy of every query (those fields set to "unknown", home = last-seen point, no sightings),
so the model works with today's form and gets better as the form collects more.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402
import features  # noqa: E402
import raster  # noqa: E402
from m7_model import FEATURES, M7, UNKNOWN, derive  # noqa: E402

GROUP = C.FEATURES["negatives"] + 1
MODEL_DIR = C.OUTPUT / "model"
APP_FORM_UNKNOWN = ["temperament", "circumstance", "collar"]


# --- data ---------------------------------------------------------------------------------
def load() -> pd.DataFrame:
    """Training table without "came home by itself" recoveries: home is not a place to search.
    How often pets come home on their own is reported separately (meta: self_return_rate)."""
    t = pd.read_parquet(C.OUTPUT / "training_candidates.parquet")
    outcome = pd.read_parquet(C.OUTPUT / "cases.parquet", columns=["case_id", "outcome"]).set_index("case_id")["outcome"]
    home_q = (t["label_source"] == "recovery") & (t["case_id"].map(outcome) == "returned_home").to_numpy()
    t = t[~home_q].reset_index(drop=True)
    lab = t["label"].to_numpy().reshape(-1, GROUP)
    assert (lab[:, 0] == 1).all() and (lab[:, 1:] == 0).all(), "expected 1 positive + negatives per query"
    t["collar"] = np.where(t["collar_tag"].astype(bool), "yes", "no")
    return t


def minimal_report(df: pd.DataFrame) -> pd.DataFrame:
    """What the app knows today: no temperament/lifestyle/collar/circumstance, home unknown, no sighting pins."""
    df = df.copy()
    for k in APP_FORM_UNKNOWN:
        df[k] = UNKNOWN
    df["group"] = np.where(df["species"] == "cat", "cat_unknown", "dog")
    df["home_distance_m"] = 0.0
    df["dist_home_m"] = df["dist_m"]
    df["toward_home_cos"] = 0.0
    df["last_seen_accuracy_m"] = np.nan
    df["nearest_sighting_m"] = np.nan
    df["latest_sighting_m"] = np.nan
    df["sightings_within_300m"] = 0.0
    df["n_sightings_known"] = 0
    return df


def partial_mask(df: pd.DataFrame, rng, p: float = 0.3) -> pd.DataFrame:
    """Hide individual fields at random (per query), so any subset can be missing."""
    df = df.copy()
    nq = len(df) // GROUP
    for k in ["temperament", "circumstance", "collar", "age_group"]:
        m = np.repeat(rng.random(nq) < p, GROUP)
        df.loc[m, k] = UNKNOWN
    m = np.repeat(rng.random(nq) < p, GROUP) & (df["species"] == "cat").to_numpy()
    df.loc[m, "group"] = "cat_unknown"
    return df


# --- objective ----------------------------------------------------------------------------
def make_objective():
    def objective(preds, data):
        s = preds.reshape(-1, GROUP).copy()
        s -= s.max(axis=1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(axis=1, keepdims=True)
        y = data.get_label().reshape(-1, GROUP)
        return (p - y).ravel(), np.maximum(p * (1 - p), 1e-6).ravel()
    return objective


def make_metric(name: str):
    def metric(preds, data):
        s = preds.reshape(-1, GROUP)
        s = s - s.max(axis=1, keepdims=True)
        logp = s[:, 0] - np.log(np.exp(s).sum(axis=1))
        return name, float(-logp.mean()), False
    return metric


def train(t: pd.DataFrame, seed: int):
    rng = np.random.default_rng(seed)
    keep = [c for c in t.columns if c not in ("query_id", "case_id", "region", "label_source", "cand_lat", "cand_lng")]
    tr = t.loc[t["split"] == "train", keep].reset_index(drop=True)
    va = t.loc[t["split"] == "val", keep].reset_index(drop=True)
    tr_all = pd.concat([tr, partial_mask(tr, rng), minimal_report(tr)], ignore_index=True)
    va_min = minimal_report(va)
    Xtr = derive(tr_all)[FEATURES]
    Xva = derive(va)[FEATURES]
    Xvm = derive(va_min)[FEATURES]
    dtr = lgb.Dataset(Xtr, label=tr_all["label"].to_numpy(), free_raw_data=False)
    dva = lgb.Dataset(Xva, label=va["label"].to_numpy(), reference=dtr)
    dvm = lgb.Dataset(Xvm, label=va_min["label"].to_numpy(), reference=dtr)
    params = {
        "objective": make_objective(),
        "learning_rate": 0.08, "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 255,
        "num_threads": 0, "verbose": -1, "seed": seed, "metric": "None",
    }
    metrics = [make_metric("logloss")] * 3

    def feval(preds, data):
        # LightGBM calls feval per dataset; pick the matching metric by length.
        for m, ds in zip(metrics, (dtr, dva, dvm)):
            if data is ds:
                return m(preds, data)
        return metrics[1](preds, data)

    print(f"[train] uniform-guess logloss = {np.log(GROUP):.3f}", flush=True)
    print(f"[train] {len(Xtr):,} rows ({len(Xtr) // GROUP:,} queries incl. masked copies), "
          f"val {len(Xva) // GROUP:,} queries, {len(FEATURES)} features", flush=True)
    t0 = time.time()
    booster = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva, dvm], valid_names=["val", "val_minimal"],
                        feval=feval, callbacks=[lgb.early_stopping(100, first_metric_only=True), lgb.log_evaluation(100)])
    print(f"[train] best iteration {booster.best_iteration} in {time.time() - t0:.0f}s", flush=True)
    return booster


# --- full-map evaluation --------------------------------------------------------------------
def evaluate(model: M7, t: pd.DataFrame, per_cell: int, seed: int) -> tuple[pd.DataFrame, list]:
    """Score every lattice cell around held-out test cases; measure how much area must be searched."""
    cases = pd.read_parquet(C.OUTPUT / "cases.parquet").set_index("case_id")
    sightings = pd.read_parquet(C.OUTPUT / "sightings.parquet")
    pos = t[(t["split"] == "test") & (t["label"] == 1)].copy()
    pos["group"] = cases.loc[pos["case_id"], "group"].to_numpy()
    pos = pos.sample(frac=1.0, random_state=seed)
    sample = pos[pos.groupby(["group", "label_source"]).cumcount() < per_cell]
    grids = {r: raster.load(r, C.REGIONS[r]["bbox"]) for r in sample["region"].unique()}
    lays = {r: features.prepare_layers(g[1]) for r, g in grids.items()}
    rows, examples = [], []
    t0 = time.time()
    for i, (_, q) in enumerate(sample.iterrows()):
        grid, layers = grids[q["region"]]
        case = cases.loc[q["case_id"]]
        base = {k: case[k] for k in features.CASE_COLS}
        base.update(last_seen_lat=case["last_seen_lat"], last_seen_lng=case["last_seen_lng"],
                    home_lat=case["home_lat"], home_lng=case["home_lng"])
        sg = sightings[sightings["case_id"] == q["case_id"]]
        known = features.known_sightings(sg, cases["lost_at"], grid, [q["case_id"]], [q["t_hours"]])[0]
        tx, ty = grid.to_xy(q["cand_lat"], q["cand_lng"])
        ox, oy = grid.to_xy(case["last_seen_lat"], case["last_seen_lng"])
        d_true = float(np.hypot(tx - ox, ty - oy))
        for variant in ("full", "minimal"):
            c = dict(base)
            kn = known
            if variant == "minimal":
                c.update(temperament=UNKNOWN, circumstance=UNKNOWN, collar_tag=UNKNOWN,
                         group="cat_unknown" if c["species"] == "cat" else "dog",
                         home_lat=c["last_seen_lat"], home_lng=c["last_seen_lng"], home_distance_m=0.0,
                         last_seen_accuracy_m=np.nan)
                kn = None
            out = model.predict_grid(grid, layers, lays[q["region"]], c, q["t_hours"], kn,
                                     extra_points=(np.array([tx]), np.array([ty])))
            s_true = float(out["extra_score"][0])
            R = model.meta["radius_m"][c["species"]]
            inside = d_true <= R
            cell_km2 = (out["step_m"] / 1000) ** 2
            order = np.sort(out["prob"])[::-1]
            cum = np.cumsum(order)
            p_true = np.exp(s_true - out["score"].max()) / np.exp(out["score"] - out["score"].max()).sum()
            row = {"query_id": q["query_id"], "group": q["group"], "label_source": q["label_source"],
                   "variant": variant, "t_hours": q["t_hours"], "dist_true_m": d_true, "inside": inside,
                   "area_model_km2": (out["score"] > s_true).sum() * cell_km2 if inside else np.inf,
                   "area_circle_km2": np.pi * (d_true / 1000) ** 2,
                   "top1pct": inside and (out["score"] > s_true).mean() <= 0.01,
                   "top5pct": inside and (out["score"] > s_true).mean() <= 0.05}
            for lvl in (0.5, 0.8, 0.95):
                thr = order[min(np.searchsorted(cum, lvl), order.size - 1)]
                row[f"in_{int(lvl * 100)}pct_region"] = bool(inside and p_true >= thr)
            rows.append(row)
            if variant == "full" and len(examples) < 6 and q["label_source"] == "recovery" \
                    and sum(e["group"] == q["group"] for e in examples) < 2:
                examples.append({"group": q["group"], "out": out, "case": case, "true": (q["cand_lat"], q["cand_lng"]),
                                 "known": known, "grid": grid, "t": q["t_hours"]})
        if (i + 1) % 50 == 0:
            print(f"  [eval] {i + 1}/{len(sample)} cases ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows), examples


def summarize(ev: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        return pd.Series({
            "cases": len(g),
            "median_area_model_km2": g["area_model_km2"].median(),
            "median_area_circle_km2": g["area_circle_km2"].median(),
            "found_in_0.25km2_model": (g["area_model_km2"] <= 0.25).mean(),
            "found_in_0.25km2_circle": (g["area_circle_km2"] <= 0.25).mean(),
            "found_in_1km2_model": (g["area_model_km2"] <= 1).mean(),
            "found_in_1km2_circle": (g["area_circle_km2"] <= 1).mean(),
            "top_5pct_cells": g["top5pct"].mean(),
            "in_50pct_zone": g["in_50pct_region"].mean(),
            "in_80pct_zone": g["in_80pct_region"].mean(),
            "in_95pct_zone": g["in_95pct_region"].mean(),
        })
    return ev.groupby(["variant", "group"]).apply(agg)


def plot_examples(examples, path):
    n = len(examples)
    if not n:
        return
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(6 * ((n + 1) // 2), 11))
    for ax, ex in zip(np.ravel(axes), examples):
        out, case = ex["out"], ex["case"]
        top = out["prob"] >= np.quantile(out["prob"], 0.6)
        ax.scatter(out["lng"][top], out["lat"][top], c=np.log10(out["prob"][top]), s=6, marker="s", cmap="magma_r")
        ax.scatter(case["last_seen_lng"], case["last_seen_lat"], marker="o", s=80, facecolor="white", edgecolor="black",
                   zorder=5, label="last seen")
        if ex["known"] is not None:
            slat, slng = ex["grid"].to_latlng(ex["known"][0], ex["known"][1])
            ax.scatter(slng, slat, marker="^", s=40, color="#2563eb", zorder=5, label="sightings so far")
        ax.scatter(ex["true"][1], ex["true"][0], marker="*", s=220, color="#16a34a", edgecolor="black", zorder=6,
                   label="where found")
        ax.set_title(f"{ex['group']} · {case['circumstance']} · {case['temperament']} · {ex['t'] / 24:.1f} d")
        ax.set_aspect(1 / np.cos(np.radians(case["last_seen_lat"])))
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("M7 heat maps on held-out test cases (darker = more likely)")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def write_report(booster, summary: pd.DataFrame, ev: pd.DataFrame):
    imp = pd.Series(booster.feature_importance("gain"), index=booster.feature_name()).sort_values(ascending=False)
    imp = (imp / imp.sum()).head(20).round(3)
    fmt = summary.copy()
    for c in fmt.columns:
        if c.startswith("median"):
            fmt[c] = fmt[c].map(lambda v: f"{v:.2f}")
        elif c != "cases":
            fmt[c] = fmt[c].map(lambda v: f"{v:.0%}")
    lines = [
        "# M7 model report",
        "",
        f"Trained {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · best iteration {booster.best_iteration} · "
        f"{len(FEATURES)} features",
        "",
        "Evaluated on **held-out test areas** (3 km spatial blocks the model never saw), scoring every "
        "cell of a full lattice around each case (50 m cells for cats within 4 km, 100 m cells for dogs within 12 km).",
        "",
        "- **area_model**: km² you would search, best cells first, before reaching the pet",
        "- **area_circle**: the same with the usual widening-circle search from the last-seen point",
        "- **in_X%_zone**: how often the pet is inside the zone the model says holds X% of the probability "
        "(a well-calibrated model gets ~X%)",
        "- Pets that came home by themselves are excluded (home is not a search area); see `self_return_rate` in m7_meta.json",
        "- **full** = every input known; **minimal** = what today's app form collects "
        "(no temperament / lifestyle / collar / circumstance / home, no sighting pins)",
        "",
        fmt.to_markdown(),
        "",
        "## Most important features (share of gain)",
        "",
        imp.to_frame("gain").to_markdown(),
        "",
        "![examples](heatmap_examples.png)",
    ]
    (MODEL_DIR / "model_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--eval-cases", type=int, default=60, help="test cases per (group, label source)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t = load()

    if not args.eval_only:
        booster = train(t, args.seed)
        booster.save_model(str(MODEL_DIR / "m7_lgbm.txt"), num_iteration=booster.best_iteration)
        calib = json.loads((C.OUTPUT / "calibration.json").read_text())["params"]
        cs = pd.read_parquet(C.OUTPUT / "cases.parquet", columns=["group", "outcome", "hours_to_found"])
        home = cs["outcome"] == "returned_home"
        self_return = {g: {"share": float(home[cs["group"] == g].mean()),
                           "median_hours": float(cs.loc[home & (cs["group"] == g), "hours_to_found"].median())}
                       for g in cs["group"].unique()}
        meta = {"model": "M7 lost-pet location", "version": "0.1.0-synthetic",
                "trained_at": datetime.now(timezone.utc).isoformat(), "best_iteration": booster.best_iteration,
                "features": FEATURES, "lattice_m": {"cat": 50.0, "dog": 100.0},
                "radius_m": {"cat": 4000.0, "dog": 12000.0}, "cell_m": C.CELL_M,
                "training_data": {"rows": int(len(t)), "queries": int(len(t) // GROUP), "calibration": calib},
                "self_return_rate": self_return,
                "notes": "Trained on simulated cases (see README). score = raw tree output + log q(dist); "
                         "softmax of score over an equal-area lattice = probability per cell."}
        (MODEL_DIR / "m7_meta.json").write_text(json.dumps(meta, indent=2))

    model = M7(MODEL_DIR)
    booster = model.booster
    booster.best_iteration = model.meta["best_iteration"]
    print("[eval] scoring full maps for held-out test cases", flush=True)
    ev, examples = evaluate(model, t, args.eval_cases, args.seed)
    ev.to_csv(MODEL_DIR / "eval_cases.csv", index=False)
    summary = summarize(ev)
    meta = model.meta
    meta["test_metrics"] = json.loads(summary.reset_index().to_json(orient="records"))
    (MODEL_DIR / "m7_meta.json").write_text(json.dumps(meta, indent=2))
    plot_examples(examples, MODEL_DIR / "heatmap_examples.png")
    write_report(booster, summary, ev)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(3).to_string())
    print(f"[done] {MODEL_DIR / 'model_report.md'}", flush=True)


if __name__ == "__main__":
    main()

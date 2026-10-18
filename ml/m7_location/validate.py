"""Check the generated dataset against the published targets and draw diagnostic plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402

COLORS = {"cat_indoor": "#c2410c", "cat_outdoor": "#7c3aed", "dog": "#0f766e"}
LABELS = {"cat_indoor": "Indoor-only cats", "cat_outdoor": "Cats with outdoor access", "dog": "Dogs"}


def _metrics(cases: pd.DataFrame) -> list[dict]:
    T, V = C.TARGETS, C.VALIDATION_CHECKS
    by = {g: cases[cases["group"] == g] for g in COLORS}
    cats = cases[cases["species"] == "cat"]
    dogs = by["dog"]

    def q(df, p):
        d = df.loc[df["found"], "found_distance_m"]
        return float(d.quantile(p)) if len(d) else np.nan

    def frac_d(df, f):
        d = df.loc[df["found"], "found_distance_m"]
        return float(f(d).mean()) if len(d) else np.nan

    rows = [
        ("Indoor cats: 75th pct distance (m)", T["cat_indoor"]["dist_m"], q(by["cat_indoor"], 0.75), "rel", "Huang 2018", True),
        ("Outdoor cats: 75th pct distance (m)", T["cat_outdoor"]["dist_m"], q(by["cat_outdoor"], 0.75), "rel", "Huang 2018", True),
        ("Cats found alive within 7 days", T["cat"]["found_alive_7d"],
         float((cats["found_alive"] & (cats["hours_to_found"] <= 168)).mean()), "abs", "Huang 2018", True),
        ("Dogs: share found within 1,609 m", T["dog"]["dist_quantile"], frac_d(dogs, lambda d: d <= 1609), "abs", "Lord 2007", True),
        ("Dogs recovered alive", T["dog"]["recovered"], float(dogs["found_alive"].mean()), "abs", "Lord 2007", True),
        ("All cats: share found within 500 m", V["cat_all_within_500m"], frac_d(cats, lambda d: d <= 500), "abs", "Huang 2018", False),
        ("Dogs: share found beyond 8 km", V["dog_beyond_8km"], frac_d(dogs, lambda d: d > 8047), "abs", "Lord 2007", True),
        ("Dogs: median days to recovery", V["dog_median_days"],
         float(dogs.loc[dogs["found_alive"], "hours_to_found"].median() / 24), "rel", "Lord 2007", True),
        ("Cats found (any outcome, 60 days)", V["cat_found_ever"], float(cats["found"].mean()), "abs",
         "Huang 2018 (1 year)", False),
    ]
    out = []
    for name, target, got, kind, src, tuned in rows:
        err = abs(got - target) / target if kind == "rel" else abs(got - target)
        # Shares: 5 points, but never more than a quarter of the target (small shares like 8%).
        tol = 0.15 if kind == "rel" else min(0.05, 0.25 * target)
        status = "ok" if err <= tol else ("close" if err <= 2 * tol else "off")
        unit = "m" if "(m)" in name else ("days" if "days to" in name else "share")
        out.append({"metric": name, "unit": unit, "target": target, "achieved": got, "status": status, "source": src,
                    "tuned": tuned})
    return out


def _fmt(v, unit):
    if v is None or not np.isfinite(v):
        return "n/a"
    return {"m": f"{v:,.0f} m", "share": f"{v:.0%}", "days": f"{v:.1f} d"}[unit]


def report(cases: pd.DataFrame, tracks: pd.DataFrame, sightings: pd.DataFrame, grids: dict, params):
    plots = C.OUTPUT / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    metrics = _metrics(cases)

    # 1. Distance CDFs
    fig, ax = plt.subplots(figsize=(8, 5))
    for g, col in COLORS.items():
        d = np.sort(cases.loc[(cases["group"] == g) & cases["found"], "found_distance_m"].clip(lower=1).to_numpy())
        if d.size:
            ax.plot(d, np.arange(1, d.size + 1) / d.size, color=col, lw=2, label=LABELS[g])
    ax.scatter([C.TARGETS["cat_indoor"]["dist_m"]], [0.75], color=COLORS["cat_indoor"], marker="D", s=60, zorder=5)
    ax.scatter([C.TARGETS["cat_outdoor"]["dist_m"]], [0.75], color=COLORS["cat_outdoor"], marker="D", s=60, zorder=5)
    ax.scatter([1609], [C.TARGETS["dog"]["dist_quantile"]], color=COLORS["dog"], marker="D", s=60, zorder=5)
    ax.scatter([8047], [1 - C.VALIDATION_CHECKS["dog_beyond_8km"]], color=COLORS["dog"], marker="o",
               facecolor="none", s=60, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("Distance from last-seen point to where found (m)")
    ax.set_ylabel("Share of found pets")
    ax.set_title("Where pets were found (diamonds: tuned targets, circle: check)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "distance_cdf.png", dpi=110)
    plt.close(fig)

    # 2. Recovery over time
    fig, ax = plt.subplots(figsize=(8, 5))
    days = np.linspace(0, C.MAX_DAYS, 241)
    for g, col in COLORS.items():
        df = cases[cases["group"] == g]
        h = df.loc[df["found_alive"], "hours_to_found"].to_numpy() / 24
        ax.plot(days, [(h <= d).sum() / max(len(df), 1) for d in days], color=col, lw=2, label=LABELS[g])
    ax.axvline(7, color="#999", ls=":", lw=1)
    ax.set_xlabel("Days since lost")
    ax.set_ylabel("Share found alive")
    ax.set_title("Recovery over time")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "recovery_curve.png", dpi=110)
    plt.close(fig)

    # 3. Example tracks per region
    for region, (grid, layers) in grids.items():
        cs = cases[(cases["region"] == region) & cases["found"]]
        if cs.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 9 * grid.H / grid.W))
        ext = [grid.w, grid.e, grid.s, grid.n]
        ax.imshow(np.where(layers["water"] > 0, np.nan, layers["people"] + 0.3 * layers["cover"]),
                  cmap="Greys", extent=ext, aspect="auto", vmin=0, vmax=1.5)
        picks = pd.concat([cs[cs["group"] == g].sample(min(6, (cs["group"] == g).sum()), random_state=1) for g in COLORS])
        for _, row in picks.iterrows():
            tr = tracks[tracks["case_id"] == row["case_id"]].sort_values("t_hours")
            col = COLORS[row["group"]]
            ax.plot(np.r_[row["last_seen_lng"], tr["lng"], row["found_lng"]],
                    np.r_[row["last_seen_lat"], tr["lat"], row["found_lat"]], color=col, lw=1.2, alpha=0.9)
            ax.scatter(row["last_seen_lng"], row["last_seen_lat"], color=col, marker="o", s=18, zorder=4)
            ax.scatter(row["found_lng"], row["found_lat"], color=col, marker="x", s=30, zorder=4)
        ax.set_title(f"{region}: example tracks (o last seen, x found; grey = foot traffic + cover)")
        ax.set_xlim(grid.w, grid.e)
        ax.set_ylim(grid.s, grid.n)
        fig.tight_layout()
        fig.savefig(plots / f"tracks_{region}.png", dpi=100)
        plt.close(fig)

    # Tables ----------------------------------------------------------------------------
    outcome_tab = pd.crosstab(cases["group"], cases["outcome"], normalize="index").round(3)
    place_tab = cases.loc[cases["found"]].groupby("group")["found_place_type"].value_counts(normalize=True).round(3)
    by_temp = cases[cases["found"]].groupby(["species", "temperament"])["found_distance_m"].median().round(0)
    by_circ = cases[cases["found"]].groupby(["species", "circumstance"])["found_distance_m"].median().round(0)
    by_split = cases.groupby(["split", "region"]).size().unstack(fill_value=0)

    def md(df):
        return df.to_markdown() if hasattr(df, "to_markdown") else "```\n" + df.to_string() + "\n```"

    try:
        import tabulate  # noqa: F401
    except ImportError:
        def md(df):  # noqa: F811
            return "```\n" + df.to_string() + "\n```"

    lines = [
        "# M7 dataset validation",
        "",
        f"{len(cases):,} cases, {len(tracks):,} track points, {len(sightings):,} sightings "
        f"({sightings['is_false'].mean():.0%} false).",
        "",
        "## Published targets",
        "",
        "| Metric | Target | Achieved | Status | Source | Tuned for |",
        "|---|---|---|---|---|---|",
    ]
    for m in metrics:
        lines.append(f"| {m['metric']} | {_fmt(m['target'], m['unit'])} | {_fmt(m['achieved'], m['unit'])} | {m['status']} | {m['source']} | "
                     f"{'yes' if m['tuned'] else 'no (check)'} |")
    lines += [
        "",
        "Status: ok = within 15% (relative); shares within 5 points and at most a quarter of the target. close = within twice that.",
        "",
        "Calibrated parameters: `" + str({g: {k: round(v, 3) for k, v in d.items()} for g, d in params.to_dict().items()}) + "`",
        "",
        "![distance](plots/distance_cdf.png)",
        "",
        "![recovery](plots/recovery_curve.png)",
        "",
        "## Outcomes by group",
        "",
        md(outcome_tab),
        "",
        "## Where found (place type)",
        "",
        md(place_tab.unstack(fill_value=0).T),
        "",
        "## Median distance by temperament (m)",
        "",
        md(by_temp.unstack(0)),
        "",
        "## Median distance by circumstance (m)",
        "",
        md(by_circ.unstack(0)),
        "",
        "## Split sizes (spatial 3 km blocks)",
        "",
        md(by_split),
        "",
        "## Example tracks",
        "",
    ]
    lines += [f"![{r}](plots/tracks_{r}.png)" for r in grids]
    (C.OUTPUT / "validation.md").write_text("\n".join(lines), encoding="utf-8")
    for m in metrics:
        print(f"  {m['status']:5s} {m['metric']}: target {_fmt(m['target'], m['unit'])}, got {_fmt(m['achieved'], m['unit'])}", flush=True)
    return metrics

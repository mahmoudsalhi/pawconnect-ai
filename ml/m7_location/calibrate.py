"""Tune the simulator until it reproduces the published lost-pet statistics.

Knobs per group (indoor cats, outdoor cats, dogs):
  speed_mult     how far pets travel                  -> distance quantile targets
  detect_mult    how easily pets are found            -> recovery-rate targets
  search_floor   effort left after the first days     -> dog median recovery time
  relocate_mult  strangers taking the pet in by car   -> dog long-distance tail (> 8 km)

Each round simulates a fresh batch spread over every region (so the result is not tuned to
one city), measures the targets and takes damped multiplicative steps. The knobs interact
and each round is a noisy estimate, so the final parameters are the average of the last
half of the rounds rather than the last update.
"""

from __future__ import annotations

import json

import numpy as np

import config as C
from simulate import GROUPS, Params, sample_cases, simulate, summarize

RES_KEYS = ("outcome", "t_end", "end_x", "end_y")


def _hazard(p: float) -> float:
    p = min(max(p, 1e-4), 0.999)
    return -np.log(1 - p)


def _run(params, regions, per_group, seed):
    """Simulate per_group cases of each group spread over the regions; merge the results."""
    parts_c, parts_r = [], []
    for j, (grid, layers, temp_offset) in enumerate(regions):
        rng = np.random.default_rng(seed * 100 + j)
        n = max(20, per_group // len(regions))
        force = np.repeat(np.arange(3), n)
        cases = sample_cases(force.size, rng, grid, layers, force_group=force)
        res = simulate(cases, grid, layers, params, rng, record=False, temp_offset=temp_offset)
        parts_c.append(cases)
        parts_r.append(res)
    # Distances are measured within each region's own frame, so the arrays can be merged.
    cases = {key: np.concatenate([c[key] for c in parts_c]) for key in ("grp", "origin_x", "origin_y")}
    res = {key: np.concatenate([r[key] for r in parts_r]) for key in RES_KEYS}
    return summarize(cases, res)


def calibrate(regions: list, seed: int = 7) -> tuple[Params, list[dict]]:
    """regions: list of (grid, layers, temp_offset)."""
    T, V = C.TARGETS, C.VALIDATION_CHECKS
    params = Params.default()
    iters = C.CALIBRATION["iterations"]
    keep, history = [], []
    for it in range(iters):
        stats = _run(params, regions, C.CALIBRATION["cases_per_group"], seed + it)
        obs = {
            "cat_indoor_q75_m": stats["cat_indoor"]["q75_m"],
            "cat_outdoor_q75_m": stats["cat_outdoor"]["q75_m"],
            "dog_q77_m": stats["dog"]["q77_m"],
            "dog_beyond_8km": stats["dog"]["beyond_8km"],
            "cat_found_alive_7d": stats["cat"]["found_alive_7d"],
            "dog_found_alive": stats["dog"]["found_alive"],
            "dog_median_days": stats["dog"]["median_days"],
        }
        history.append({"iteration": it, "params": params.to_dict(), "observed": obs})
        print(f"  calib {it}: " + "  ".join(f"{k}={v:.3g}" for k, v in obs.items()), flush=True)
        if it >= iters // 2:
            keep.append(params.as_array().copy())

        damp = 0.6
        ratios = [T["cat_indoor"]["dist_m"] / obs["cat_indoor_q75_m"],
                  T["cat_outdoor"]["dist_m"] / obs["cat_outdoor_q75_m"],
                  T["dog"]["dist_m"] / obs["dog_q77_m"]]
        for i, r in enumerate(ratios):
            if np.isfinite(r):
                params.speed_mult[i] *= float(np.clip(r, 1 / 3, 3)) ** damp
        cat_r = _hazard(T["cat"]["found_alive_7d"]) / _hazard(obs["cat_found_alive_7d"])
        dog_r = _hazard(T["dog"]["recovered"]) / _hazard(obs["dog_found_alive"])
        params.detect_mult[:2] *= float(np.clip(cat_r, 1 / 3, 3)) ** damp
        params.detect_mult[2] *= float(np.clip(dog_r, 1 / 3, 3)) ** damp
        day_r = V["dog_median_days"] / obs["dog_median_days"]
        if np.isfinite(day_r):
            params.floor[2] = float(np.clip(params.floor[2] * np.clip(day_r, 1 / 3, 3) ** 1.5, 0.002, 0.5))
        tail_r = V["dog_beyond_8km"] / max(obs["dog_beyond_8km"], 0.005)
        params.relocate_mult[2] = float(np.clip(params.relocate_mult[2] * np.clip(tail_r, 1 / 3, 3) ** damp, 0.1, 20))

    final = Params.from_array(np.mean(keep, axis=0))
    C.OUTPUT.mkdir(parents=True, exist_ok=True)
    (C.OUTPUT / "calibration.json").write_text(json.dumps(
        {"params": final.to_dict(), "averaged_over_iterations": [h["iteration"] for h in history[iters // 2:]],
         "targets": C.TARGETS, "checks": C.VALIDATION_CHECKS, "history": history}, indent=2))
    return final, history


def load() -> Params | None:
    path = C.OUTPUT / "calibration.json"
    if not path.exists():
        return None
    return Params.from_dict(json.loads(path.read_text())["params"])

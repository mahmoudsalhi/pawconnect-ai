"""M7: where a lost pet is likely to be now.

Runs the model trained in ``ml/m7_location`` (see its README): a LightGBM scorer over map
cells, built from OpenStreetMap layers of the covered regions. The model, the map grids and
the libraries (``requirements-ml.txt``) are loaded on first use; if any of them is missing,
``LocationModelUnavailable`` is raised and the API answers 503.

The model was trained on simulated lost-pet cases calibrated to published studies. Treat
the zones as search guidance, not ground truth, until it is retrained on real reunions.
"""

from __future__ import annotations

import dataclasses
import math
import sys
import threading
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from ..config import BASE_DIR, get_settings
from .base import LocationPrediction, LocationQuery, RecoveryOutlook, SearchZone

ZONE_SHARES = (50, 80, 95)
MAX_CELLS_PER_ZONE = 6000
EDGE_MARGIN_DEG = 0.01

_lock = threading.Lock()
_grids: dict[str, tuple] = {}


class LocationModelUnavailable(RuntimeError):
    pass


def _m7_dir() -> Path:
    d = Path(get_settings().m7_dir)
    return d if d.is_absolute() else (BASE_DIR / d).resolve()


@lru_cache
def _runtime() -> SimpleNamespace:
    d = _m7_dir()
    model_dir = d / "output" / "model"
    if not (model_dir / "m7_lgbm.txt").exists():
        raise LocationModelUnavailable(f"M7 model not found in {model_dir}. Train it with ml/m7_location/train.py.")
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    try:
        import numpy as np
        import config as m7_config
        import features
        import m7_model
        import raster
        import recovery
    except ImportError as e:
        raise LocationModelUnavailable(f"M7 libraries missing ({e.name}). Install backend/requirements-ml.txt.") from e
    rec = recovery.RecoveryModel(model_dir) if (model_dir / "m7_recovery_lgbm.txt").exists() else None
    return SimpleNamespace(np=np, cfg=m7_config, features=features, raster=raster, m7=m7_model,
                           model=m7_model.M7(model_dir), recovery=recovery, recovery_model=rec)


def _region_for(rt, lat: float, lng: float) -> str | None:
    for name, cfg in rt.cfg.REGIONS.items():
        s, w, n, e = cfg["bbox"]
        if s + EDGE_MARGIN_DEG <= lat <= n - EDGE_MARGIN_DEG and w + EDGE_MARGIN_DEG <= lng <= e - EDGE_MARGIN_DEG:
            return name
    return None


def _grid(rt, region: str):
    if region not in _grids:
        loaded = rt.raster.load(region, rt.cfg.REGIONS[region]["bbox"])
        if loaded is None:
            raise LocationModelUnavailable(f"Map grid for {region} not found. Run ml/m7_location/build_dataset.py.")
        grid, layers = loaded
        _grids[region] = (grid, layers, rt.features.prepare_layers(layers))
    return _grids[region]


def _breed_group(species: str, breed: str) -> str:
    if species == "cat":
        return "domestic_cat"
    b = breed.lower()
    for words, group in (
        (("beagle", "hound", "greyhound", "sloughi", "whippet", "husky", "pointer", "setter"), "hound_sighthound"),
        (("collie", "shepherd", "malinois", "sheepdog", "cattle", "kelpie", "berger"), "herding"),
        (("terrier", "jack russell", "westie", "schnauzer"), "terrier"),
        (("rottweiler", "doberman", "mastiff", "boxer", "cane corso", "kangal", "pitbull", "akita"), "guard"),
        (("poodle", "chihuahua", "spaniel", "maltese", "shih", "pug", "bichon", "yorkshire", "pomeranian",
          "bulldog", "dachshund", "retriever", "labrador", "golden"), "companion"),
    ):
        if any(w in b for w in words):
            return group
    return "local_mixed"


def _age_group(species: str, age: str | None) -> str:
    a = (age or "").lower()
    if any(w in a for w in ("kitten", "puppy", "young", "baby", "junior")):
        return "kitten" if species == "cat" else "puppy"
    if "senior" in a or "old" in a:
        return "senior"
    return "adult" if "adult" in a else "unknown"


def _missing(q: LocationQuery) -> list[str]:
    out = []
    if q.species == "cat" and q.indoor_only is None:
        out.append("whether the cat goes outside")
    if q.temperament is None:
        out.append("temperament (friendly, shy, fearful)")
    if q.circumstance is None:
        out.append("how it got lost")
    if q.collar_tag is None:
        out.append("collar or tag")
    if q.home_lat is None:
        out.append("home location, if different from where it was last seen")
    out.append("sighting locations pinned on the map")
    return out


def predict(query: LocationQuery) -> LocationPrediction | None:
    # Predictions change slowly with time; cache them per whole hour missing.
    hours = max(1.0, round(query.hours_missing))
    return _predict_cached(dataclasses.replace(query, hours_missing=hours))


@lru_cache(maxsize=256)
def _predict_cached(q: LocationQuery) -> LocationPrediction | None:
    rt = _runtime()
    np = rt.np
    species = q.species.lower()
    if species not in ("cat", "dog"):
        return None
    region = _region_for(rt, q.lat, q.lng)
    if region is None:
        return None
    grid, layers, lay = _grid(rt, region)
    unknown = rt.m7.UNKNOWN

    if species == "cat":
        group = "cat_unknown" if q.indoor_only is None else ("cat_indoor" if q.indoor_only else "cat_outdoor")
    else:
        group = "dog"
    if q.month and q.hour_of_day is not None:
        temp = rt.cfg.MONTH_TEMP_C[q.month - 1] + rt.cfg.REGIONS[region]["temp_offset"] \
            + 5 * math.cos(2 * math.pi * (q.hour_of_day - 15) / 24)
    elif q.month:
        temp = rt.cfg.MONTH_TEMP_C[q.month - 1] + rt.cfg.REGIONS[region]["temp_offset"]
    else:
        temp = math.nan
    home_lat, home_lng = (q.home_lat, q.home_lng) if q.home_lat is not None else (q.lat, q.lng)
    hx, hy = grid.to_xy(home_lat, home_lng)
    ox, oy = grid.to_xy(q.lat, q.lng)
    size = (q.size or "").lower()
    case = {
        "species": species, "group": group,
        "temperament": q.temperament or unknown,
        "age_group": _age_group(species, q.age_group),
        "size": "small" if species == "cat" else (size if size in ("small", "medium", "large") else unknown),
        "breed_group": _breed_group(species, q.breed),
        "circumstance": q.circumstance or unknown,
        "collar_tag": unknown if q.collar_tag is None else q.collar_tag,
        "hour_of_day": math.nan if q.hour_of_day is None else q.hour_of_day,
        "month": math.nan if q.month is None else q.month,
        "temp_c_at_loss": temp,
        "last_seen_accuracy_m": math.nan,
        "home_distance_m": float(np.hypot(hx - ox, hy - oy)),
        "last_seen_lat": q.lat, "last_seen_lng": q.lng, "home_lat": home_lat, "home_lng": home_lng,
    }
    hours = min(q.hours_missing, rt.cfg.MAX_DAYS * 24.0)
    with _lock:
        out = rt.model.predict_grid(grid, layers, lay, case, hours, None)

    p = out["prob"]
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    zones, start = [], 0
    for share in ZONE_SHARES:
        end = min(int(np.searchsorted(cum, share / 100)) + 1, order.size)
        idx = order[start:end][:MAX_CELLS_PER_ZONE]
        zones.append(SearchZone(share=share, cells=[(round(float(out["lat"][i]), 5), round(float(out["lng"][i]), 5))
                                                    for i in idx]))
        start = end
    returns = rt.model.meta.get("self_return_rate", {})
    if group == "cat_unknown":
        shares = [returns[g]["share"] for g in ("cat_indoor", "cat_outdoor") if g in returns]
        self_return = sum(shares) / len(shares) if shares else 0.0
    else:
        self_return = returns.get(group, {}).get("share", 0.0)
    top = order[0]
    outlook = None
    if rt.recovery_model is not None:
        import pandas as pd

        row = {k: case[k] for k in ("species", "group", "temperament", "age_group", "size", "breed_group",
                                    "circumstance", "hour_of_day", "month", "temp_c_at_loss")}
        row.update(sex=(q.sex or unknown).lower(), collar=rt.recovery._yn(q.collar_tag),
                   chip=rt.recovery._yn(q.microchipped), neutered_s=unknown)
        row.update({k: float(v[0]) for k, v in rt.recovery.env_at(lay, grid, [q.lat], [q.lng]).items()})
        rows = [row]
        if group == "cat_unknown":
            rows += [dict(row, group="cat_indoor"), dict(row, group="cat_outdoor")]
        with _lock:
            probs = rt.recovery_model.probs(pd.DataFrame(rows))
        by_lifestyle = None
        if group == "cat_unknown":
            by_lifestyle = {k: rt.recovery.outlook(p, q.hours_missing)["found_alive"]
                            for k, p in (("indoor_only", probs[1]), ("outdoor_access", probs[2]))}
        outlook = RecoveryOutlook(**rt.recovery.outlook(probs[0], q.hours_missing), by_lifestyle=by_lifestyle)
    return LocationPrediction(
        model=f"m7-{rt.model.meta.get('version', 'dev')}",
        cell_m=float(out["step_m"]),
        zones=zones,
        peak=(round(float(out["lat"][top]), 5), round(float(out["lng"][top]), 5)),
        self_return_share=round(float(self_return), 3),
        missing_inputs=_missing(q),
        recovery=outlook,
    )

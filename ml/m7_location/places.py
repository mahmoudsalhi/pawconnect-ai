"""Named places per region, for the AI search assistant's plans ("put posters at the market on ...").

    python places.py      # writes data/grids/<region>_places.json from the cached OSM extracts

Only places that matter when searching for a lost pet or putting up posters are kept: markets,
food shops and restaurants (pets go there for food; owners put posters there), mosques, schools,
bus stops, parks and gardens, car parks (cats hide under cars), cemeteries, vets and shelters.
Data © OpenStreetMap contributors, ODbL 1.0.
"""

from __future__ import annotations

import json

import config as C
import osm

KINDS = {
    "amenity": {
        "marketplace": "market", "place_of_worship": "mosque", "school": "school", "university": "school",
        "veterinary": "vet", "animal_shelter": "animal shelter", "animal_boarding": "animal shelter",
        "restaurant": "restaurant", "fast_food": "restaurant", "cafe": "cafe", "parking": "car park",
        "grave_yard": "cemetery", "pharmacy": "pharmacy", "bus_station": "bus station",
    },
    "shop": {"butcher": "butcher", "seafood": "fish shop", "supermarket": "supermarket", "bakery": "bakery",
             "convenience": "grocery", "pet": "pet shop", "greengrocer": "grocery"},
    "leisure": {"park": "park", "garden": "garden", "playground": "playground"},
    "landuse": {"cemetery": "cemetery"},
    "highway": {"bus_stop": "bus stop"},
}


def _kind(tags: dict) -> str | None:
    for key, table in KINDS.items():
        k = table.get(tags.get(key, ""))
        if k:
            return k
    return None


def _centre(el):
    if "lat" in el:
        return el["lat"], el["lon"]
    if "center" in el:
        return el["center"]["lat"], el["center"]["lon"]
    g = el.get("geometry")
    if g:
        return sum(p["lat"] for p in g) / len(g), sum(p["lon"] for p in g) / len(g)
    return None


def build(region: str) -> list[dict]:
    data = osm.fetch(region)
    seen, out = set(), []
    for el in data["pois"] + data["areas"]:
        tags = el.get("tags", {})
        kind = _kind(tags)
        if not kind:
            continue
        if kind == "mosque" and tags.get("religion") not in (None, "muslim"):
            kind = "place of worship"
        name = tags.get("name:fr") or tags.get("name") or tags.get("name:en") or tags.get("name:ar")
        # Unnamed bus stops, car parks and playgrounds are still useful landmarks; unnamed shops are not.
        if not name and kind not in ("bus stop", "car park", "park", "garden", "playground", "cemetery", "market"):
            continue
        c = _centre(el)
        if not c:
            continue
        key = (kind, name, round(c[0], 4), round(c[1], 4))
        if key in seen:
            continue
        seen.add(key)
        street = tags.get("addr:street")
        out.append({"kind": kind, "name": name or "", "lat": round(c[0], 5), "lng": round(c[1], 5),
                    **({"street": street} if street else {})})
    return out


def main():
    out_dir = C.DATA / "grids"
    for region in C.REGIONS:
        places = build(region)
        (out_dir / f"{region}_places.json").write_text(json.dumps(places, ensure_ascii=False), encoding="utf-8")
        print(f"  [{region}] {len(places):,} places")


if __name__ == "__main__":
    main()

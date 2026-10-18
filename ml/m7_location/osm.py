"""Extract the OpenStreetMap features a lost pet's movement depends on.

Source: Geofabrik's Tunisia extract (one ~85 MB download, parsed locally with pyosmium).
The public Overpass API was too slow for city-sized boxes. Per-region results are cached
as JSON in data/osm/<region>/; delete that folder (or the .pbf) to refresh.

Data © OpenStreetMap contributors, ODbL 1.0.
"""

from __future__ import annotations

import json
import urllib.request

import osmium

from config import DATA, REGIONS

PBF_URL = "https://download.geofabrik.de/africa/tunisia-latest.osm.pbf"
PBF = DATA / "pbf" / "tunisia-latest.osm.pbf"
LAYERS = ("roads", "areas", "pois", "buildings")
AREA_KEYS = ("landuse", "leisure", "natural", "waterway")
AREA_AMENITY = {"parking", "grave_yard", "marketplace"}
PAD = 0.02  # degrees: keep features that start just outside a box


def _download():
    PBF.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {PBF_URL}", flush=True)
    req = urllib.request.Request(PBF_URL, headers={"User-Agent": "PawConnect-M7-dataset/0.1"})
    with urllib.request.urlopen(req) as r, open(PBF, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


class _Extractor(osmium.SimpleHandler):
    def __init__(self, boxes: dict):
        super().__init__()
        self.boxes = {r: (s - PAD, w - PAD, n + PAD, e + PAD) for r, (s, w, n, e) in boxes.items()}
        self.out = {r: {k: [] for k in LAYERS} for r in boxes}

    def _regions(self, lat, lng):
        return [r for r, (s, w, n, e) in self.boxes.items() if s <= lat <= n and w <= lng <= e]

    def node(self, n):
        if not n.tags:
            return
        tags = dict(n.tags)
        if not ("amenity" in tags or "shop" in tags or tags.get("highway") == "bus_stop"):
            return
        loc = n.location
        for r in self._regions(loc.lat, loc.lon):
            self.out[r]["pois"].append({"type": "node", "lat": loc.lat, "lon": loc.lon, "tags": tags})

    def way(self, w):
        tags = dict(w.tags)
        if not tags:
            return
        is_road = "highway" in tags
        is_bldg = "building" in tags
        is_area = any(k in tags for k in AREA_KEYS) or tags.get("amenity") in AREA_AMENITY
        is_poi = "amenity" in tags or "shop" in tags
        if not (is_road or is_bldg or is_area or is_poi):
            return
        try:
            geom = [{"lat": nd.location.lat, "lon": nd.location.lon} for nd in w.nodes]
        except osmium.InvalidLocationError:
            return
        if len(geom) < 2:
            return
        regions = set(self._regions(geom[0]["lat"], geom[0]["lon"])) | set(self._regions(geom[-1]["lat"], geom[-1]["lon"]))
        if not regions:
            return
        el = {"type": "way", "id": w.id, "tags": tags, "geometry": geom}
        for r in regions:
            o = self.out[r]
            if is_road:
                o["roads"].append(el)
            if is_bldg:
                o["buildings"].append(el)
            if is_area:
                o["areas"].append(el)
            if is_poi:
                lat = sum(p["lat"] for p in geom) / len(geom)
                lon = sum(p["lon"] for p in geom) / len(geom)
                o["pois"].append({"type": "way", "center": {"lat": lat, "lon": lon}, "tags": tags})

    def area(self, a):
        # Closed ways are handled in way(); here we only add multipolygon relations.
        if a.from_way():
            return
        tags = dict(a.tags)
        if not (any(k in tags for k in AREA_KEYS) or tags.get("amenity") in AREA_AMENITY):
            return
        members = []
        try:
            for outer in a.outer_rings():
                members.append({"type": "way", "role": "outer",
                                "geometry": [{"lat": nd.lat, "lon": nd.lon} for nd in outer]})
                for inner in a.inner_rings(outer):
                    members.append({"type": "way", "role": "inner",
                                    "geometry": [{"lat": nd.lat, "lon": nd.lon} for nd in inner]})
        except osmium.InvalidLocationError:
            return
        if not members:
            return
        regions = set()
        for m in members:
            if m["role"] == "outer":
                for p in m["geometry"][:: max(1, len(m["geometry"]) // 8)]:
                    regions |= set(self._regions(p["lat"], p["lon"]))
        for r in regions:
            self.out[r]["areas"].append({"type": "relation", "id": a.orig_id(), "tags": tags, "members": members})


def extract_all(regions: dict = REGIONS):
    """Parse the extract once and write every region's cache."""
    if not PBF.exists():
        _download()
    print(f"  parsing {PBF.name} (takes a few minutes)", flush=True)
    h = _Extractor({r: cfg["bbox"] for r, cfg in regions.items()})
    h.apply_file(str(PBF), locations=True, idx="flex_mem")
    for r, layers in h.out.items():
        d = DATA / "osm" / r
        d.mkdir(parents=True, exist_ok=True)
        for k, v in layers.items():
            (d / f"{k}.json").write_text(json.dumps(v), encoding="utf-8")
        print(f"  [{r}] " + ", ".join(f"{k}={len(v):,}" for k, v in layers.items()), flush=True)


def fetch(region: str, bbox=None) -> dict[str, list]:
    d = DATA / "osm" / region
    if not all((d / f"{k}.json").exists() for k in LAYERS):
        extract_all()
    return {k: json.loads((d / f"{k}.json").read_text(encoding="utf-8")) for k in LAYERS}

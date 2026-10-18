"""Turn OSM features into 20 m raster layers the simulator (and later the model) reads.

Coordinates: a local metric frame per region with the origin at the north-west corner,
x pointing east and y pointing south, so cell (row, col) = (y // CELL_M, x // CELL_M).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from config import CELL_M, DATA


@dataclass
class Grid:
    s: float
    w: float
    n: float
    e: float
    cell: float = CELL_M

    def __post_init__(self):
        latc = (self.s + self.n) / 2
        self.mx = 111320.0 * math.cos(math.radians(latc))  # metres per degree of longitude
        self.my = 110574.0                                  # metres per degree of latitude
        self.W = int(math.ceil((self.e - self.w) * self.mx / self.cell))
        self.H = int(math.ceil((self.n - self.s) * self.my / self.cell))

    def to_xy(self, lat, lng):
        return (np.asarray(lng) - self.w) * self.mx, (self.n - np.asarray(lat)) * self.my

    def to_latlng(self, x, y):
        return self.n - np.asarray(y) / self.my, self.w + np.asarray(x) / self.mx

    def meta(self) -> dict:
        return {"s": self.s, "w": self.w, "n": self.n, "e": self.e, "cell": self.cell,
                "H": self.H, "W": self.W, "mx": self.mx, "my": self.my}


# --- tag classification -------------------------------------------------------------------
ROAD_CLASS = {
    **dict.fromkeys(["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"], "major"),
    **dict.fromkeys(["secondary", "secondary_link", "tertiary", "tertiary_link"], "secondary"),
    **dict.fromkeys(["residential", "unclassified", "living_street", "service", "road"], "minor"),
    **dict.fromkeys(["footway", "path", "pedestrian", "track", "steps", "cycleway", "bridleway"], "path"),
}
ROAD_WIDTH_M = {"major": 30, "secondary": 16, "minor": 10, "path": 5}

LANDUSE = {
    "residential": "residential", "commercial": "commercial", "retail": "commercial",
    "industrial": "industrial", "railway": "industrial", "port": "industrial",
    "construction": "construction", "brownfield": "construction", "landfill": "construction",
    "greenfield": "construction",
    "farmland": "farm", "orchard": "farm", "vineyard": "farm", "meadow": "farm", "farmyard": "farm",
    "plant_nursery": "farm", "allotments": "farm",
    "grass": "green", "village_green": "green", "recreation_ground": "green",
    "forest": "wild", "cemetery": "cemetery", "religious": "green",
    "basin": "water", "reservoir": "water", "salt_pond": "water",
}
LEISURE = {
    "park": "green", "garden": "green", "pitch": "green", "playground": "green",
    "golf_course": "green", "recreation_ground": "green", "dog_park": "green",
    "nature_reserve": "wild", "common": "green",
}
NATURAL = {
    "water": "water", "bay": "water", "wood": "wild", "scrub": "wild", "heath": "wild",
    "grassland": "wild", "wetland": "wild", "beach": "open", "sand": "open", "bare_rock": "open",
}
FOOD_AMENITY = {"restaurant", "fast_food", "cafe", "food_court", "marketplace", "ice_cream", "bar",
                "pub", "waste_disposal", "waste_basket", "recycling"}
FOOD_SHOP = {"butcher", "seafood", "supermarket", "convenience", "bakery", "greengrocer", "deli",
             "pastry", "general", "fish"}
CROWD_AMENITY = {"school", "place_of_worship", "marketplace", "university", "college", "hospital",
                 "bus_station"}
VET = {"veterinary", "animal_shelter", "animal_boarding"}


def _area_class(tags: dict) -> str | None:
    if tags.get("waterway") == "riverbank" or tags.get("water"):
        return "water"
    if tags.get("amenity") == "parking":
        return "parking"
    if tags.get("amenity") == "grave_yard":
        return "cemetery"
    if tags.get("amenity") == "marketplace":
        return "commercial"
    for key, table in (("leisure", LEISURE), ("landuse", LANDUSE), ("natural", NATURAL)):
        if tags.get(key) in table:
            return table[tags[key]]
    return None


def _assemble_rings(segments: list[list[tuple]]) -> list[list[tuple]]:
    """Join multipolygon member ways end to end into closed rings."""
    segs = [s for s in segments if len(s) >= 2]
    rings = []
    while segs:
        ring = list(segs.pop())
        changed = True
        while ring[0] != ring[-1] and changed:
            changed = False
            for i, s in enumerate(segs):
                if s[0] == ring[-1]:
                    ring.extend(s[1:])
                elif s[-1] == ring[-1]:
                    ring.extend(reversed(s[:-1]))
                elif s[-1] == ring[0]:
                    ring = list(s[:-1]) + ring
                elif s[0] == ring[0]:
                    ring = list(reversed(s[1:])) + ring
                else:
                    continue
                segs.pop(i)
                changed = True
                break
        if len(ring) >= 3:
            rings.append(ring)
    return rings


class Canvas:
    """One PIL image per layer; draw in pixel coordinates."""

    def __init__(self, grid: Grid, names):
        self.g = grid
        self.img = {n: Image.new("L", (grid.W, grid.H), 0) for n in names}
        self.draw = {n: ImageDraw.Draw(im) for n, im in self.img.items()}

    def px(self, geom):
        x, y = self.g.to_xy([p[0] for p in geom], [p[1] for p in geom])
        return list(zip((x / self.g.cell).tolist(), (y / self.g.cell).tolist()))

    def polygon(self, layer, geom):
        pts = self.px(geom)
        if len(pts) >= 3:
            self.draw[layer].polygon(pts, fill=255)

    def multipolygon(self, layer, outers, inners):
        # Draw in a cropped temp mask so holes don't erase neighbouring features.
        all_pts = [p for r in outers for p in self.px(r)]
        if not all_pts:
            return
        xs, ys = zip(*all_pts)
        x0, y0 = max(int(min(xs)) - 1, 0), max(int(min(ys)) - 1, 0)
        x1, y1 = min(int(max(xs)) + 2, self.g.W), min(int(max(ys)) + 2, self.g.H)
        if x1 <= x0 or y1 <= y0:
            return
        tmp = Image.new("L", (x1 - x0, y1 - y0), 0)
        d = ImageDraw.Draw(tmp)
        for r in outers:
            d.polygon([(px - x0, py - y0) for px, py in self.px(r)], fill=255)
        for r in inners:
            d.polygon([(px - x0, py - y0) for px, py in self.px(r)], fill=0)
        region = self.img[layer].crop((x0, y0, x1, y1))
        self.img[layer].paste(Image.fromarray(np.maximum(np.asarray(region), np.asarray(tmp))), (x0, y0))

    def line(self, layer, geom, width_m):
        pts = self.px(geom)
        if len(pts) >= 2:
            self.draw[layer].line(pts, fill=255, width=max(1, round(width_m / self.g.cell)))

    def arrays(self):
        return {n: (np.asarray(im) > 0).astype(np.float32) for n, im in self.img.items()}


def _norm(a, pct=99.0):
    hi = np.percentile(a[a > 0], pct) if np.any(a > 0) else 1.0
    return np.clip(a / (hi or 1.0), 0, 1).astype(np.float32)


def build(region: str, bbox, osm: dict[str, list]) -> tuple[Grid, dict[str, np.ndarray]]:
    g = Grid(*bbox)
    names = ["major", "secondary", "minor", "path", "building", "residential", "commercial",
             "industrial", "construction", "farm", "green", "wild", "cemetery", "water", "open",
             "parking", "coastline"]
    cv = Canvas(g, names)

    for el in osm["roads"]:
        cls = ROAD_CLASS.get(el.get("tags", {}).get("highway"))
        if cls and "geometry" in el:
            cv.line(cls, [(p["lat"], p["lon"]) for p in el["geometry"]], ROAD_WIDTH_M[cls])

    for el in osm["areas"]:
        tags = el.get("tags", {})
        if tags.get("natural") == "coastline" and "geometry" in el:
            cv.line("coastline", [(p["lat"], p["lon"]) for p in el["geometry"]], CELL_M)
            continue
        if el["type"] == "way" and tags.get("waterway") in ("river", "canal") and "geometry" in el:
            cv.line("water", [(p["lat"], p["lon"]) for p in el["geometry"]], 15)
            continue
        cls = _area_class(tags)
        if not cls:
            continue
        if el["type"] == "way" and "geometry" in el:
            geom = [(p["lat"], p["lon"]) for p in el["geometry"]]
            if geom[0] == geom[-1]:
                cv.polygon(cls, geom)
        elif el["type"] == "relation":
            outer, inner = [], []
            for m in el.get("members", []):
                if m.get("type") == "way" and "geometry" in m:
                    seg = [(round(p["lat"], 7), round(p["lon"], 7)) for p in m["geometry"]]
                    (inner if m.get("role") == "inner" else outer).append(seg)
            cv.multipolygon(cls, _assemble_rings(outer), _assemble_rings(inner))

    for el in osm["buildings"]:
        if "geometry" in el:
            geom = [(p["lat"], p["lon"]) for p in el["geometry"]]
            if geom[0] == geom[-1]:
                cv.polygon("building", geom)

    L = cv.arrays()

    # Points of interest -> count grids.
    food = np.zeros((g.H, g.W), np.float32)
    people = np.zeros_like(food)
    vet = np.zeros_like(food)
    for el in osm["pois"]:
        lat = el.get("lat", el.get("center", {}).get("lat"))
        lng = el.get("lon", el.get("center", {}).get("lon"))
        if lat is None:
            continue
        x, y = g.to_xy(lat, lng)
        c, r = int(x // g.cell), int(y // g.cell)
        if not (0 <= r < g.H and 0 <= c < g.W):
            continue
        tags = el.get("tags", {})
        am, sh = tags.get("amenity"), tags.get("shop")
        if am in FOOD_AMENITY or sh in FOOD_SHOP:
            food[r, c] += 1
        people[r, c] += 3 if am in CROWD_AMENITY else 1
        if am in VET or sh == "pet":
            vet[r, c] += 1

    # Sea: the coastline splits the box into components; the ones with almost no
    # mapped land features are sea.
    land_evidence = np.zeros_like(food, bool)
    for n in ("major", "secondary", "minor", "path", "building", "residential", "commercial",
              "industrial", "farm", "green", "wild", "construction", "cemetery", "open"):
        land_evidence |= L[n] > 0
    land_evidence = ndimage.binary_dilation(land_evidence, iterations=10)
    wall = ndimage.binary_dilation(L["coastline"] > 0, iterations=1)
    labels, k = ndimage.label(~wall)
    sea = np.zeros_like(wall)
    if k:
        sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, k + 1))
        land = ndimage.sum(land_evidence, labels, index=range(1, k + 1))
        for i, (sz, ld) in enumerate(zip(sizes, land), 1):
            if sz > 2000 and ld / sz < 0.03:
                sea |= labels == i
    water = np.clip(L["water"] + sea + (wall & ndimage.binary_dilation(sea, iterations=2)), 0, 1)
    water = water * (1 - np.clip(L["major"] + L["secondary"], 0, 1))  # keep bridges and causeways

    b = L["building"]
    edge = (ndimage.binary_dilation(b > 0, iterations=1) & (b == 0)).astype(np.float32)
    green_s = ndimage.gaussian_filter(L["green"], 1.0)
    parking_s = ndimage.gaussian_filter(L["parking"], 1.0)
    minor_s = ndimage.gaussian_filter(L["minor"], 1.5)

    layers = {
        # Hiding places: beside buildings (courtyards, under parked cars), gardens, scrub,
        # vacant lots, car parks, cemeteries, olive groves.
        "cover": np.clip(0.6 * edge + 0.5 * green_s + 0.9 * L["wild"] + 0.7 * L["construction"]
                         + 0.6 * parking_s + 0.5 * L["cemetery"] + 0.4 * minor_s + 0.3 * L["farm"],
                         0, 1).astype(np.float32),
        "food": _norm(ndimage.gaussian_filter(np.log1p(food) * 10, 3.0)),
        "people": _norm(ndimage.gaussian_filter(people + 0.15 * b + 0.3 * L["secondary"]
                                                + 0.1 * L["minor"] + 0.3 * L["commercial"], 4.0)),
        "corridor": np.clip(np.maximum.reduce([L["minor"], L["path"], 0.7 * L["secondary"],
                                               0.4 * L["green"], 0.5 * L["open"]]), 0, 1),
        "green": np.clip(L["green"] + 0.5 * L["cemetery"], 0, 1).astype(np.float32),
        "building": b,
        "farm": L["farm"],
        "major": L["major"],
        "exposed": np.clip(L["secondary"] + 0.5 * L["open"], 0, 1).astype(np.float32),
        "water": water.astype(np.float32),
        "wild": L["wild"],
        "residential": np.clip(L["residential"] + b, 0, 1).astype(np.float32),
        "commercial": L["commercial"],
        "construction": L["construction"],
        "cemetery": L["cemetery"],
        "minor": L["minor"],
        "vet": ndimage.gaussian_filter(vet, 2.0),
        "dist_major_m": (ndimage.distance_transform_edt(L["major"] == 0) * g.cell).astype(np.float32),
    }

    out = DATA / "grids"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{region}.npz", **{k: v.astype(np.float16 if k != "dist_major_m" else np.float32)
                                                  for k, v in layers.items()})
    (out / f"{region}.json").write_text(json.dumps(g.meta(), indent=2))
    return g, layers


def load(region: str, bbox) -> tuple[Grid, dict[str, np.ndarray]] | None:
    path = DATA / "grids" / f"{region}.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        return Grid(*bbox), {k: z[k].astype(np.float32) for k in z.files}


# Classify the cell where a pet was found, for the found_place_type column.
def place_type(layers: dict, r: np.ndarray, c: np.ndarray) -> np.ndarray:
    def at(name):
        return layers[name][r, c]

    out = np.full(r.shape, "residential_street", dtype=object)
    out[at("minor") == 0] = "open_ground"
    near_building = layers["cover"][r, c] > 0.5
    out[(at("residential") > 0) & near_building] = "near_buildings_courtyard"
    out[at("farm") > 0] = "farmland_olive_grove"
    out[at("wild") > 0] = "scrub_woodland"
    out[at("green") > 0] = "park_garden"
    out[at("cemetery") > 0] = "cemetery"
    out[at("construction") > 0] = "vacant_lot_construction"
    out[(at("food") > 0.55) & (at("commercial") + at("residential") > 0)] = "market_restaurants"
    out[at("building") > 0] = "inside_building_block"
    out[at("major") > 0] = "major_road"
    return out

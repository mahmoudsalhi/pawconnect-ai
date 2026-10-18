"""Matching: compare a new report against open reports of the opposite kind.

Final score = visual similarity (embedding cosine) fused with metadata
agreement (species, colour, time gap, distance). Species is a hard filter;
everything else nudges the ranking.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class Report:
    report_id: str
    kind: str  # "lost" or "found"
    species: str  # "dog" | "cat" | ...
    embeddings: np.ndarray  # (n_photos, dim), L2-normalised
    colors: set[str] = field(default_factory=set)
    when: datetime | None = None
    lat: float | None = None  # approximate coords only (privacy)
    lon: float | None = None


@dataclass
class Match:
    report_id: str
    score: float
    visual: float
    meta: float


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def visual_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Best cosine similarity between any photo of A and any photo of B.

    Max over pairs is deliberate: one good angle of the same face is enough,
    and a bad photo should not drag the score down.
    """
    return float((a @ b.T).max())


def metadata_score(q: Report, c: Report, max_days: float = 30.0, max_km: float = 50.0) -> float:
    """0..1 agreement from colour, time and distance (missing fields are neutral)."""
    parts = []
    if q.colors and c.colors:
        parts.append(len(q.colors & c.colors) / len(q.colors | c.colors))
    if q.when and c.when:
        days = abs((q.when - c.when).total_seconds()) / 86400
        parts.append(max(0.0, 1.0 - days / max_days))
    if None not in (q.lat, q.lon, c.lat, c.lon):
        km = haversine_km(q.lat, q.lon, c.lat, c.lon)
        parts.append(max(0.0, 1.0 - km / max_km))
    return float(np.mean(parts)) if parts else 0.5


class Matcher:
    def __init__(self, w_visual: float = 0.8, w_meta: float = 0.2, min_score: float = 0.0):
        assert abs(w_visual + w_meta - 1.0) < 1e-6
        self.w_visual, self.w_meta, self.min_score = w_visual, w_meta, min_score
        self.reports: dict[str, Report] = {}

    def add(self, report: Report) -> None:
        self.reports[report.report_id] = report

    def remove(self, report_id: str) -> None:
        self.reports.pop(report_id, None)

    def find(self, query: Report, top_k: int = 5) -> list[Match]:
        opposite = "found" if query.kind == "lost" else "lost"
        results = []
        for c in self.reports.values():
            if c.kind != opposite or c.species != query.species or c.report_id == query.report_id:
                continue
            v = visual_similarity(query.embeddings, c.embeddings)
            m = metadata_score(query, c)
            s = self.w_visual * v + self.w_meta * m
            if s >= self.min_score:
                results.append(Match(c.report_id, s, v, m))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

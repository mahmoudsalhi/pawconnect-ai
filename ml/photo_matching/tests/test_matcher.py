from datetime import datetime, timedelta

import numpy as np

from pawmatch.metrics import retrieval_metrics
from pawmatch.matcher import Matcher, Report, haversine_km


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def mk(rid, kind, vec, species="dog", **kw):
    return Report(rid, kind, species, unit([vec]), **kw)


def test_finds_same_animal_first():
    m = Matcher()
    m.add(mk("lost_rex", "lost", [1, 0, 0]))
    m.add(mk("lost_other", "lost", [0, 1, 0]))
    res = m.find(mk("found_1", "found", [0.95, 0.05, 0]))
    assert res[0].report_id == "lost_rex"
    assert res[0].score > res[1].score


def test_only_compares_opposite_kind():
    m = Matcher()
    m.add(mk("found_a", "found", [1, 0, 0]))
    assert m.find(mk("found_b", "found", [1, 0, 0])) == []


def test_species_is_hard_filter():
    m = Matcher()
    m.add(mk("lost_cat", "lost", [1, 0, 0], species="cat"))
    assert m.find(mk("found_dog", "found", [1, 0, 0], species="dog")) == []


def test_multiple_photos_use_best_pair():
    lost = Report("l", "lost", "dog", unit([[0, 1, 0], [1, 0, 0]]))  # one bad, one good photo
    m = Matcher()
    m.add(lost)
    res = m.find(mk("f", "found", [1, 0, 0]))
    assert res[0].visual > 0.99


def test_metadata_breaks_ties():
    now = datetime(2026, 10, 17)
    m = Matcher()
    m.add(mk("near", "lost", [1, 0, 0], colors={"brown"}, when=now, lat=36.80, lon=10.18))
    m.add(mk("far", "lost", [1, 0, 0], colors={"white"}, when=now - timedelta(days=60),
             lat=34.74, lon=10.76))
    q = mk("q", "found", [1, 0, 0], colors={"brown"}, when=now, lat=36.81, lon=10.17)
    res = m.find(q)
    assert [r.report_id for r in res] == ["near", "far"]


def test_min_score_threshold():
    m = Matcher(min_score=0.9)
    m.add(mk("l", "lost", [0, 1, 0]))
    assert m.find(mk("f", "found", [1, 0, 0])) == []


def test_haversine_tunis_sfax():
    d = haversine_km(36.8065, 10.1815, 34.7406, 10.7603)
    assert 220 < d < 250


def test_retrieval_metrics_perfect_and_random():
    rng = np.random.default_rng(0)
    centers = unit(rng.normal(size=(10, 32)))
    labels = np.repeat(np.arange(10), 4)
    embs = unit(centers[labels] + 0.01 * rng.normal(size=(40, 32)))
    perfect = retrieval_metrics(embs, labels)
    assert perfect["recall@1"] == 1.0 and perfect["mAP"] > 0.99
    noise = retrieval_metrics(unit(rng.normal(size=(40, 32))), labels)
    assert noise["recall@1"] < 0.5

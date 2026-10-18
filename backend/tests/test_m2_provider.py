"""Unit tests for the M2 provider. The M2 service is replaced by an in-process fake (no torch needed)."""

import httpx
import pytest

from app.ai.base import ImageInput, MatchSubject, ModelUnavailable
from app.ai.models import CALIBRATION, PawModels, calibrate

# Fake embeddings per photo name: dogA photos are close, dogB is orthogonal.
VECTORS = {
    "a1": [1.0, 0.0, 0.0],
    "a2": [0.9, 0.436, 0.0],   # cosine with a1 = 0.9
    "b1": [0.0, 1.0, 0.0],
}


def fake_service(calls: list[str] | None = None, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        name = next(k for k in VECTORS if f"name-{k}".encode() in body)
        if calls is not None:
            calls.append(name)
        if status != 200:
            return httpx.Response(status)
        return httpx.Response(200, json={"dim": 3, "embedding": VECTORS[name]})

    return httpx.Client(base_url="http://m2.test", transport=httpx.MockTransport(handler))


def img(name: str) -> ImageInput:
    return ImageInput(url=f"/uploads/{name}.jpg", loader=lambda: f"name-{name}".encode())


def subject(*names: str, species: str = "Dog") -> MatchSubject:
    return MatchSubject(images=[img(n) for n in names], species=species)


def test_calibration_is_monotonic_and_hits_its_anchors():
    values = [calibrate(x / 100) for x in range(-100, 101)]
    assert values == sorted(values) and values[0] == 0 and values[-1] == 99
    for cosine, pct in CALIBRATION:
        assert calibrate(cosine) == pct
    assert calibrate(0.33) == 50  # the "possible match" line used by match_new_report


def test_same_dog_scores_higher_than_different_dog():
    m = PawModels(client=fake_service())
    same = m.similarity(subject("a1"), subject("a2"))
    other = m.similarity(subject("a1"), subject("b1"))
    assert same > 70 > 50 > other


def test_best_photo_pair_counts():
    m = PawModels(client=fake_service())
    assert m.similarity(subject("b1", "a1"), subject("a2")) == m.similarity(subject("a1"), subject("a2"))


def test_different_species_is_zero_without_calling_the_model():
    calls: list[str] = []
    m = PawModels(client=fake_service(calls))
    assert m.similarity(subject("a1"), subject("a2", species="Cat")) == 0
    assert calls == []


def test_embeddings_are_cached_per_photo():
    calls: list[str] = []
    m = PawModels(client=fake_service(calls))
    m.similarity(subject("a1"), subject("a2"))
    m.similarity(subject("a1"), subject("b1"))
    assert sorted(calls) == ["a1", "a2", "b1"]


def test_service_down_raises_model_unavailable():
    m = PawModels(client=fake_service(status=503))
    with pytest.raises(ModelUnavailable):
        m.similarity(subject("a1"), subject("a2"))


def test_unreadable_photo_scores_zero():
    m = PawModels(client=fake_service())
    blank = MatchSubject(images=[ImageInput(url="/uploads/missing.jpg", loader=lambda: None)], species="Dog")
    assert m.similarity(subject("a1"), blank) == 0


def test_embed_for_the_ui_is_in_zero_one_range():
    m = PawModels(client=fake_service())
    sig = m.embed(img("a2"))
    assert len(sig) == 3 and all(0.0 <= v <= 1.0 for v in sig)
    down = PawModels(client=fake_service(status=503))
    assert len(down.embed(img("a1"))) == 128  # falls back to the placeholder signature

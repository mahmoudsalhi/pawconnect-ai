"""Dog skin check: dogs only, no result instead of a crash, the three statuses, and /api/ai/analyze.
The real-model test is skipped when backend/models/m5_skin.onnx is not present (it is not in git)."""

import io

import numpy as np
import pytest
from PIL import Image

from app.ai import health as health_mod
from app.ai import models as models_mod
from app.ai.base import HealthPrediction, ImageInput
from app.ai.demo import DemoProvider
from app.ai.health import HealthModel, HealthResult
from app.ai.models import PawModels
from app.config import get_settings

CLASSES = ["healthy", "dermatitis", "fungal_infections", "hypersensitivity", "demodicosis", "ringworm"]
META = {"classes": CLASSES, "size": 8, "mean": [0.5] * 3, "std": [0.5] * 3}


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (180, 120, 60)).save(buf, "JPEG")
    return buf.getvalue()


def _uploaded() -> ImageInput:
    return ImageInput(url="/uploads/test.jpg", loader=_jpeg)


def _remote() -> ImageInput:
    return ImageInput(url="https://images.unsplash.com/photo-1")  # read() -> None


class _Session:
    def __init__(self, logits):
        self.logits = np.array(logits, dtype=np.float32)

    def run(self, _outputs, feeds):
        return [np.tile(self.logits, (len(next(iter(feeds.values()))), 1))]


def _fake_model(logits, threshold=0.5) -> HealthModel:
    m = HealthModel.__new__(HealthModel)  # skip __init__: no ONNX file needed
    m.meta, m.classes, m.threshold = META, CLASSES, threshold
    m.session, m.input_name = _Session(logits), "input"
    return m


def _use(monkeypatch, model):
    monkeypatch.setattr(models_mod, "get_health_model", lambda: model)


@pytest.fixture
def no_model_cache():
    health_mod.get_health_model.cache_clear()
    yield
    health_mod.get_health_model.cache_clear()


def test_demo_provider_has_no_health_result():
    assert DemoProvider().predict_health([_uploaded()], "Dog") is None


@pytest.mark.parametrize("species", ["Cat", "cat", "Other", "", None])
def test_only_dogs_and_photos_are_not_read(monkeypatch, species):
    _use(monkeypatch, _fake_model([5, 0, 0, 0, 0, 0]))
    read = []
    photo = ImageInput(url="/uploads/x.jpg", loader=lambda: read.append(1) or _jpeg())
    assert PawModels().predict_health([photo], species) is None
    assert read == []  # cats never cost a photo read


def test_missing_model_disables_the_feature(monkeypatch, tmp_path, no_model_cache):
    monkeypatch.setattr(get_settings(), "m5_skin_model_path", str(tmp_path / "nope.onnx"))
    assert health_mod.get_health_model() is None
    assert PawModels().predict_health([_uploaded()], "Dog") is None


def test_broken_model_file_disables_the_feature(monkeypatch, tmp_path, no_model_cache):
    (tmp_path / "m.onnx").write_bytes(b"not a model")
    (tmp_path / "m.json").write_text('{"classes": ["healthy", "ringworm"], "size": 224, "mean": [0,0,0], "std": [1,1,1]}')
    monkeypatch.setattr(get_settings(), "m5_skin_model_path", str(tmp_path / "m.onnx"))
    assert health_mod.get_health_model() is None
    assert PawModels().predict_health([_uploaded()], "Dog") is None


def test_relative_and_absolute_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(get_settings(), "m5_skin_model_path", "backend/models/m5_skin.onnx")
    assert health_mod.model_path().parts[-3:] == ("backend", "models", "m5_skin.onnx")
    monkeypatch.setattr(get_settings(), "m5_skin_model_path", str(tmp_path / "m5.onnx"))
    assert health_mod.model_path() == tmp_path / "m5.onnx"


def test_unreadable_photos_give_no_result(monkeypatch):
    model = _fake_model([5, 0, 0, 0, 0, 0])
    _use(monkeypatch, model)
    assert model.predict([None, b"not an image"], "dog") is None
    assert PawModels().predict_health([_remote(), _remote()], "Dog") is None


def test_healthy(monkeypatch):
    _use(monkeypatch, _fake_model([5, 0, 0, 0, 0, 0]))
    assert PawModels().predict_health([_remote(), _uploaded()], "Dog") == HealthPrediction("healthy", None, 97)


def test_skin_problem_names_the_most_likely_condition(monkeypatch):
    _use(monkeypatch, _fake_model([0, 1, 0, 0, 0, 4]))
    pred = PawModels().predict_health([_uploaded()], "Dog")
    assert pred.status == "skin_problem" and pred.condition == "ringworm" and 95 <= pred.confidence <= 100


def test_unsure_below_the_threshold(monkeypatch):
    _use(monkeypatch, _fake_model([np.log(9), 0, 0, 0, 0, np.log(5)], threshold=0.7))  # healthy 9/18 vs problem 9/18
    pred = PawModels().predict_health([_uploaded()], "Dog")
    assert pred.status == "unsure" and pred.condition is None and pred.confidence == 50


def test_model_error_gives_no_result(monkeypatch):
    class Boom:
        def predict(self, *_):
            raise RuntimeError("onnx failure")

    _use(monkeypatch, Boom())
    assert PawModels().predict_health([_uploaded()], "Dog") is None


def test_real_model_if_present(no_model_cache):
    if not health_mod.model_path().exists():
        pytest.skip("m5_skin.onnx is not here")
    model = health_mod.get_health_model()
    assert model is not None, "m5_skin.onnx is present but could not be loaded (see the logged error)"
    assert model.predict([_jpeg()], "cat") is None
    result = model.predict([_jpeg()], "dog")
    assert isinstance(result, HealthResult) and result.status in {"healthy", "skin_problem", "unsure"}
    assert (result.condition is not None) == (result.status == "skin_problem")
    assert 0 <= result.confidence <= 100 and abs(sum(result.probabilities.values()) - 1) < 0.01


@pytest.mark.parametrize("species,expected", [("Dog", {"status": "healthy", "condition": None, "confidence": 97}),
                                              ("Cat", None)])
def test_analyze_returns_health_for_dogs_only(monkeypatch, species, expected):
    from fastapi.testclient import TestClient

    from app.main import app

    _use(monkeypatch, _fake_model([5, 0, 0, 0, 0, 0]))
    monkeypatch.setattr(models_mod, "get_age_model", lambda: None)  # keep M4 out of this test
    monkeypatch.setattr("app.routers.ai.get_provider", lambda: PawModels())
    monkeypatch.setattr("app.routers.ai.image_input", lambda url: _uploaded())
    r = TestClient(app).post("/api/ai/analyze", json={"photos": ["/uploads/test.jpg"], "species": species})
    assert r.status_code == 200, r.text
    assert r.json()["health"] == expected

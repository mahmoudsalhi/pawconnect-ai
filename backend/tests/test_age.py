"""Age prediction: fallbacks to the placeholder, the "not sure" threshold, and the /api/ai/analyze field.
The real-model test is skipped when backend/models/m4_age.onnx is not present (it is not in git)."""

import io

import numpy as np
import pytest
from PIL import Image

from app.ai import age as age_mod
from app.ai import models as models_mod
from app.ai.age import AgeModel, AgeResult
from app.ai.base import AgePrediction, ImageInput
from app.ai.demo import DemoProvider
from app.ai.models import PawModels
from app.config import get_settings

PLACEHOLDER = AgePrediction(group=None, confidence=0)
META = {"groups": ["young", "adult", "senior"], "size": 8, "mean": [0.5] * 3, "std": [0.5] * 3}


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (180, 120, 60)).save(buf, "JPEG")
    return buf.getvalue()


def _uploaded(data: bytes | None = None) -> ImageInput:
    return ImageInput(url="/uploads/test.jpg", loader=lambda: data if data is not None else _jpeg())


def _remote() -> ImageInput:
    return ImageInput(url="https://images.unsplash.com/photo-1")  # read() -> None


class _Session:
    """Stands in for onnxruntime: returns the same logits for every photo."""

    def __init__(self, logits):
        self.logits = np.array(logits, dtype=np.float32)

    def run(self, _outputs, feeds):
        return [np.tile(self.logits, (len(next(iter(feeds.values()))), 1))]


def _fake_model(logits, threshold) -> AgeModel:
    m = AgeModel.__new__(AgeModel)  # skip __init__: no ONNX file needed
    m.meta, m.groups, m.threshold = META, META["groups"], threshold
    m.session, m.input_name = _Session(logits), "input"
    return m


@pytest.fixture
def no_model_cache():
    age_mod.get_age_model.cache_clear()
    yield
    age_mod.get_age_model.cache_clear()


def test_demo_placeholder_is_not_sure():
    assert DemoProvider().predict_age([_uploaded()], "Dog") == PLACEHOLDER


def test_missing_model_falls_back_to_placeholder(monkeypatch, tmp_path, no_model_cache):
    monkeypatch.setattr(get_settings(), "m4_age_model_path", str(tmp_path / "nope.onnx"))
    assert age_mod.get_age_model() is None
    assert PawModels().predict_age([_uploaded()], "Dog") == PLACEHOLDER


def test_broken_model_file_falls_back_to_placeholder(monkeypatch, tmp_path, no_model_cache):
    (tmp_path / "m.onnx").write_bytes(b"not a model")
    (tmp_path / "m.json").write_text('{"groups": ["young", "adult", "senior"], "size": 224, "mean": [0,0,0], "std": [1,1,1]}')
    monkeypatch.setattr(get_settings(), "m4_age_model_path", str(tmp_path / "m.onnx"))
    assert age_mod.get_age_model() is None
    assert PawModels().predict_age([_uploaded()], "Dog") == PLACEHOLDER


def test_relative_path_starts_at_project_root(monkeypatch):
    monkeypatch.setattr(get_settings(), "m4_age_model_path", "backend/models/m4_age.onnx")
    assert age_mod.model_path().parts[-3:] == ("backend", "models", "m4_age.onnx")
    assert age_mod.model_path().parent.parent.name == "backend"


def test_unreadable_photos_fall_back_to_placeholder(monkeypatch):
    model = _fake_model([0.0, 5.0, 0.0], threshold=0.4)
    monkeypatch.setattr(models_mod, "get_age_model", lambda: model)
    assert model.predict([None, None]) is None
    assert model.predict([b"not an image"]) is None
    assert PawModels().predict_age([_remote(), _remote()], "Cat") == PLACEHOLDER


def test_confident_prediction(monkeypatch):
    model = _fake_model([0.0, 5.0, 0.0], threshold=0.4)
    monkeypatch.setattr(models_mod, "get_age_model", lambda: model)
    pred = PawModels().predict_age([_remote(), _uploaded()], "Dog")  # the remote photo is skipped
    assert pred.group == "adult" and 90 <= pred.confidence <= 100


def test_below_threshold_is_not_sure(monkeypatch):
    model = _fake_model([1.0, 1.2, 0.9], threshold=0.6)  # best class ~0.38 < 0.6
    monkeypatch.setattr(models_mod, "get_age_model", lambda: model)
    pred = PawModels().predict_age([_uploaded()], "Dog")
    assert pred.group is None and 0 < pred.confidence < 60


def test_model_error_falls_back_to_placeholder(monkeypatch):
    class Boom:
        def predict(self, *_):
            raise RuntimeError("onnx failure")

    monkeypatch.setattr(models_mod, "get_age_model", lambda: Boom())
    assert PawModels().predict_age([_uploaded()], "Dog") == PLACEHOLDER


def test_real_model_if_present(no_model_cache):
    if not age_mod.model_path().exists():
        pytest.skip("backend/models/m4_age.onnx is not here")
    model = age_mod.get_age_model()
    assert model is not None, "m4_age.onnx is present but could not be loaded (see the logged error)"
    result = model.predict([_jpeg()], "dog")
    assert isinstance(result, AgeResult)
    assert result.group in {"young", "adult", "senior", None} and 0 <= result.confidence <= 100
    assert abs(sum(result.probabilities.values()) - 1) < 0.01


@pytest.mark.parametrize("readable,expected_group", [(True, "young"), (False, None)])
def test_analyze_returns_age(monkeypatch, readable, expected_group):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setattr(models_mod, "get_age_model", lambda: _fake_model([4.0, 0.0, 0.0], threshold=0.4))
    monkeypatch.setattr("app.routers.ai.get_provider", lambda: PawModels())
    monkeypatch.setattr("app.routers.ai.image_input", lambda url: _uploaded() if readable else ImageInput(url=url))
    r = TestClient(app).post("/api/ai/analyze", json={"photos": ["https://images.unsplash.com/photo-1"], "species": "Dog"})
    assert r.status_code == 200, r.text
    age = r.json()["age"]
    assert age["group"] == expected_group
    assert age["confidence"] > 90 if readable else age["confidence"] == 0

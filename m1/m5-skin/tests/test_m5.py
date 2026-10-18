"""Tests that run without torch or a GPU."""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integration"))

import health  # noqa: E402
from common import (CLASSES, MEAN, SIZE, STD, choose_threshold, decide, evaluate,  # noqa: E402
                    normalize_class)
from prepare_data import dedupe, scan  # noqa: E402


@pytest.mark.parametrize("name,expected", [
    ("Healthy", "healthy"), ("Dermatitis", "dermatitis"), ("Fungal_infections", "fungal_infections"),
    ("Hypersensitivity", "hypersensitivity"), ("demodicosis", "demodicosis"), ("ringworm", "ringworm"),
    ("Fungal infections", "fungal_infections"), ("cats", None)])
def test_normalize_class(name, expected):
    assert normalize_class(name) == expected


def _img(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)


def test_scan_and_dedupe_removes_leaks(tmp_path):
    _img(tmp_path / "train" / "Healthy" / "a.jpg", (1, 2, 3))
    _img(tmp_path / "train" / "ringworm" / "b.jpg", (9, 9, 9))
    _img(tmp_path / "valid" / "Healthy" / "c.jpg", (50, 60, 70))
    _img(tmp_path / "test" / "Healthy" / "leak.jpg", (1, 2, 3))   # same bytes as train a.jpg
    _img(tmp_path / "test" / "ringworm" / "d.jpg", (200, 10, 10))
    (tmp_path / "test" / "Other").mkdir()
    df, skipped = scan(tmp_path)
    assert skipped == ["test/Other"]  # unknown folders are reported, not used
    df, info = dedupe(df)
    assert info["leaked_from_train"] == 1
    assert set(df.split) == {"train", "valid", "test"}
    assert "leak.jpg" not in " ".join(df.path)


def test_decide_healthy_problem_unsure():
    healthy = np.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.02])
    problem = np.array([0.1, 0.05, 0.05, 0.05, 0.05, 0.70])
    mixed = np.array([0.45, 0.2, 0.1, 0.1, 0.1, 0.05])
    assert decide(healthy, 0.6)[0] == "healthy"
    status, cond, conf = decide(problem, 0.6)
    assert status == "skin_problem" and CLASSES[cond] == "ringworm" and abs(conf - 0.9) < 1e-9
    assert decide(mixed, 0.6)[0] == "unsure"


def test_evaluate_reports_baseline_and_binary():
    probs = np.eye(6)[[0, 1, 2, 0, 5, 5]]
    labels = np.array([0, 1, 2, 1, 5, 4])
    out = evaluate(probs, labels, majority_class=5)
    assert 0 < out["macro_f1"] < 1
    assert out["baseline_always_majority_macro_f1"] < out["macro_f1"]
    assert set(out["healthy_vs_problem"]) == {"precision", "recall", "f1"}
    assert np.array(out["confusion_matrix"]).shape == (6, 6)


def test_choose_threshold_default_when_impossible():
    probs = np.tile(np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1]), (10, 1))
    labels = np.full(10, 3)
    assert choose_threshold(probs, labels) == 0.5


# ---------- inference with a tiny stand-in ONNX model ----------
@pytest.fixture()
def tiny_model(tmp_path):
    """Channel means pick the class: red -> healthy, green -> dermatitis, blue -> fungal.
    Other classes get logit 0 constant."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    w = np.zeros((3, 6), dtype=np.float32)
    w[0, 0] = w[1, 1] = w[2, 2] = 20
    axes = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name="axes")
    nodes = [helper.make_node("ReduceMean", ["input", "axes"], ["pooled"], keepdims=0),
             helper.make_node("MatMul", ["pooled", "w"], ["logits"])]
    graph = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 6])],
        initializer=[numpy_helper.from_array(w, name="w"), axes])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    path = tmp_path / "m5_skin.onnx"
    onnx.save(model, str(path))
    path.with_suffix(".json").write_text(json.dumps(
        {"classes": CLASSES, "size": SIZE, "mean": MEAN, "std": STD, "threshold": 0.6}))
    return health.HealthModel(path)


def _png(color):
    buf = io.BytesIO()
    Image.new("RGB", (60, 40), color).save(buf, format="PNG")
    return buf.getvalue()


def test_healthy_prediction(tiny_model):
    r = tiny_model.predict([_png((250, 5, 5))], species="dog")
    assert r.status == "healthy" and r.condition is None and r.confidence >= 60


def test_skin_problem_prediction_names_condition(tiny_model):
    r = tiny_model.predict([_png((5, 250, 5))], species="dog")
    assert r.status == "skin_problem" and r.condition == "dermatitis"
    assert abs(sum(r.probabilities.values()) - 1) < 1e-3


def test_unsure_when_mixed(tiny_model):
    mean_pixel = tuple(int(round(m * 255)) for m in MEAN)
    tiny_model.threshold = 0.9   # uniform output: 1/6 healthy, 5/6 problem, neither reaches 0.9
    assert tiny_model.predict([_png(mean_pixel)]).status == "unsure"


def test_cat_is_not_analyzed(tiny_model):
    assert tiny_model.predict([_png((250, 5, 5))], species="cat") is None


def test_skips_bad_photos_and_none_when_unusable(tiny_model):
    assert tiny_model.predict([None, b"junk", _png((250, 5, 5))]).status == "healthy"
    assert tiny_model.predict([None, b"", b"junk"]) is None


def test_missing_files_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv("M5_SKIN_MODEL_PATH", str(tmp_path / "nope.onnx"))
    health.get_health_model.cache_clear()
    assert health.get_health_model() is None
    health.get_health_model.cache_clear()

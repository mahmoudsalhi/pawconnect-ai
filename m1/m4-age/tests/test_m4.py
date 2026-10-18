"""Tests that run without torch or a GPU: data rules, splits, thresholds and inference."""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integration"))

import age  # noqa: E402
from common import (GROUPS, MEAN, SIZE, STD, aggregate_by_pet, choose_threshold,  # noqa: E402
                    evaluate, group_from_months)
from prepare_data import build_manifest, index_images  # noqa: E402


# ---------- age groups ----------
@pytest.mark.parametrize("months,expected", [
    (0, "young"), (11, "young"), (12, "adult"), (95, "adult"), (96, "senior"), (180, "senior")])
def test_group_boundaries(months, expected):
    assert group_from_months(months) == expected


# ---------- data split ----------
def _fake_dataset(tmp_path, n=120):
    img_dir = tmp_path / "imgs"; img_dir.mkdir()
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        pid = f"pet{i:03d}"
        months = int(rng.choice([3, 6, 24, 48, 60, 120, 150], p=[.2, .2, .2, .15, .1, .1, .05]))
        rows.append({"PetID": pid, "Type": int(rng.choice([1, 2])), "Age": months})
        for k in range(1, 4):
            Image.new("RGB", (8, 8)).save(img_dir / f"{pid}-{k}.jpg")
    return pd.DataFrame(rows), img_dir


def test_split_is_by_pet_with_no_leakage(tmp_path):
    df, img_dir = _fake_dataset(tmp_path)
    manifest = build_manifest(df, index_images(str(img_dir), max_per_pet=5), seed=7)
    splits_per_pet = manifest.groupby("pet_id")["split"].nunique()
    assert (splits_per_pet == 1).all(), "a pet appears in more than one split"
    pets = manifest.drop_duplicates("pet_id")
    share = pets["split"].value_counts(normalize=True)
    assert abs(share["train"] - 0.70) < 0.06
    assert set(manifest["group"]) <= set(GROUPS)
    assert set(manifest["species"]) <= {"dog", "cat"}


def test_max_photos_per_pet(tmp_path):
    df, img_dir = _fake_dataset(tmp_path, n=20)
    manifest = build_manifest(df, index_images(str(img_dir), max_per_pet=2))
    assert manifest.groupby("pet_id").size().max() == 2


# ---------- aggregation, threshold, metrics ----------
def test_aggregate_averages_photos_per_pet():
    df = pd.DataFrame({"pet_id": ["a", "a", "b"], "species": ["dog", "dog", "cat"],
                       "group": ["adult", "adult", "young"]})
    probs = np.array([[0.1, 0.8, 0.1], [0.3, 0.6, 0.1], [0.9, 0.05, 0.05]])
    pets, pp, pl = aggregate_by_pet(df, probs)
    assert len(pets) == 2
    assert np.allclose(pp[0], [0.2, 0.7, 0.1])
    assert pl.tolist() == [1, 0]


def test_choose_threshold_prefers_smallest_that_reaches_target():
    # Confident predictions are right, unsure ones are wrong.
    probs = np.array([[0.9, 0.05, 0.05]] * 8 + [[0.45, 0.4, 0.15]] * 4 + [[0.5, 0.3, 0.2]] * 2)
    labels = np.array([0] * 8 + [1] * 4 + [1] * 2)
    t = choose_threshold(probs, labels, target_acc=0.9, min_coverage=0.5)
    assert 0.5 <= t <= 0.9
    keep = probs.max(1) >= t
    assert (probs.argmax(1)[keep] == labels[keep]).mean() >= 0.9


def test_choose_threshold_falls_back_to_default():
    probs = np.array([[0.5, 0.3, 0.2]] * 10)
    labels = np.array([1] * 10)  # always wrong
    assert choose_threshold(probs, labels) == 0.6


def test_evaluate_reports_baseline_and_species():
    probs = np.eye(3)[[0, 1, 2, 1]]
    labels = np.array([0, 1, 2, 2])
    out = evaluate(probs, labels, np.array(["dog", "dog", "cat", "cat"]), majority_class=1)
    assert out["accuracy"] == 0.75
    assert out["baseline_always_majority_macro_f1"] < out["macro_f1"]
    assert set(out["per_species_macro_f1"]) == {"dog", "cat"}


# ---------- inference with a tiny stand-in ONNX model ----------
@pytest.fixture()
def tiny_model(tmp_path):
    """3-class model: mean red / green / blue channel of the photo decides young / adult / senior."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    w = numpy_helper.from_array((np.eye(3, dtype=np.float32) * 20), name="w")
    axes = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name="axes")
    nodes = [helper.make_node("ReduceMean", ["input", "axes"], ["pooled"], keepdims=0),
             helper.make_node("MatMul", ["pooled", "w"], ["logits"])]
    graph = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 3])],
        initializer=[w, axes])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    path = tmp_path / "m4_age.onnx"
    onnx.save(model, str(path))
    path.with_suffix(".json").write_text(json.dumps(
        {"groups": GROUPS, "size": SIZE, "mean": MEAN, "std": STD, "threshold": 0.6}))
    return age.AgeModel(path)


def _png(color):
    buf = io.BytesIO()
    Image.new("RGB", (60, 40), color).save(buf, format="PNG")
    return buf.getvalue()


def test_preprocess_shape_dtype_and_range():
    meta = {"size": SIZE, "mean": MEAN, "std": STD}
    x = age.preprocess(Image.new("RGB", (500, 300), (255, 255, 255)), meta)
    assert x.shape == (3, SIZE, SIZE) and x.dtype == np.float32
    assert np.allclose(x[0], (1 - MEAN[0]) / STD[0], atol=1e-4)


def test_predict_confident_group(tiny_model):
    res = tiny_model.predict([_png((250, 5, 5))], species="dog")
    assert res.group == "young" and res.confidence >= 60
    assert abs(sum(res.probabilities.values()) - 1) < 1e-3


def test_predict_not_sure_below_threshold(tiny_model):
    # A pixel at exactly the ImageNet mean normalizes to 0 in every channel -> 1/3 each.
    mean_pixel = tuple(int(round(m * 255)) for m in MEAN)
    res = tiny_model.predict([_png(mean_pixel)])
    assert res.group is None
    assert res.confidence < 60


def test_predict_averages_several_photos(tiny_model):
    res = tiny_model.predict([_png((250, 5, 5)), _png((5, 250, 5)), _png((5, 250, 5))])
    assert res.group == "adult"


def test_predict_skips_missing_and_corrupt_photos(tiny_model):
    res = tiny_model.predict([None, b"not an image", _png((5, 5, 250))])
    assert res.group == "senior"


def test_predict_returns_none_when_no_usable_photo(tiny_model):
    assert tiny_model.predict([None, b"", b"garbage"]) is None


def test_get_age_model_returns_none_when_files_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("M4_AGE_MODEL_PATH", str(tmp_path / "nope.onnx"))
    age.get_age_model.cache_clear()
    assert age.get_age_model() is None
    age.get_age_model.cache_clear()

"""Export m4_age.pt to ONNX (runs on a plain CPU server with onnxruntime) and verify it.

Usage: python export_onnx.py --ckpt models/m4_age.pt --out models
Writes models/m4_age.onnx and models/m4_age.json (preprocessing + threshold).
Both files go to backend/models/ and are committed with the code (the .onnx is ~16 MB).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from common import GROUPS, MEAN, SIZE, STD
from train import build_model

sys.path.insert(0, str(Path(__file__).parent / "integration"))
from age import preprocess  # noqa: E402  (same code the backend uses at inference)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/m4_age.pt")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / "m4_age.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, SIZE, SIZE), str(onnx_path),
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)

    # Recent torch versions write the weights to a separate m4_age.onnx.data file. Fold them back
    # into m4_age.onnx so the backend only needs one file (well under onnx's 2 GB single-file limit).
    import onnx
    data_file = onnx_path.with_name(onnx_path.name + ".data")
    onnx.save_model(onnx.load(str(onnx_path)), str(onnx_path), save_as_external_data=False)
    if data_file.exists():
        data_file.unlink()

    meta = {"version": "m4-age-1", "groups": GROUPS, "size": SIZE, "mean": MEAN, "std": STD,
            "threshold": ckpt.get("threshold", 0.6)}
    (out / "m4_age.json").write_text(json.dumps(meta, indent=2))

    # Parity check: torch vs onnxruntime on the exact backend preprocessing.
    import onnxruntime as ort
    from PIL import Image
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))
    x = preprocess(img, meta)[None]
    with torch.no_grad():
        ref = torch.softmax(model(torch.from_numpy(x)), dim=1).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": x})[0]
    got = np.exp(got - got.max()) / np.exp(got - got.max()).sum()
    diff = float(np.abs(ref - got).max())
    print(f"max probability difference torch vs onnx: {diff:.2e}")
    assert diff < 1e-3, "ONNX output does not match the PyTorch model"
    print(f"OK -> {onnx_path} and {out / 'm4_age.json'}")


if __name__ == "__main__":
    main()

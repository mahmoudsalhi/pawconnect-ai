"""Export models/m5_skin.pt to a single-file models/m5_skin.onnx + m5_skin.json, with a parity check.

    python export_onnx.py --ckpt models/m5_skin.pt --out models
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

sys.path.insert(0, str(Path(__file__).parent / "integration"))
from common import CLASSES, DEFAULT_THRESHOLD, MEAN, SIZE, STD  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/m5_skin.pt")
    ap.add_argument("--out", default="models")
    a = ap.parse_args()
    out = Path(a.out)

    ck = torch.load(a.ckpt, map_location="cpu")
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    m.load_state_dict(ck["state"])
    m.eval()

    onnx_path = out / "m5_skin.onnx"
    dummy = torch.randn(1, 3, SIZE, SIZE)
    torch.onnx.export(m, dummy, str(onnx_path), input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=18)

    import onnx  # merge weights into ONE file so only m5_skin.onnx has to be copied
    model = onnx.load(str(onnx_path))
    onnx.save_model(model, str(onnx_path), save_as_external_data=False)
    data_file = Path(str(onnx_path) + ".data")
    if data_file.exists():
        data_file.unlink()

    thr = DEFAULT_THRESHOLD
    mpath = out / "metrics.json"
    if mpath.exists():
        thr = json.loads(mpath.read_text()).get("threshold", thr)
    (out / "m5_skin.json").write_text(json.dumps(
        {"classes": CLASSES, "size": SIZE, "mean": MEAN, "std": STD, "threshold": thr}, indent=2))

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.randn(2, 3, SIZE, SIZE).astype(np.float32)
    with torch.no_grad():
        ref = m(torch.from_numpy(x)).numpy()
    got = sess.run(None, {"input": x})[0]
    diff = float(np.abs(ref - got).max())
    print(f"max torch-vs-onnx difference: {diff:.2e}")
    assert diff < 1e-3, "ONNX output differs from PyTorch"
    print("OK:", onnx_path, f"({onnx_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

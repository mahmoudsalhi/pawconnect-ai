# M5 · Dog skin problem detection

Photo of a dog in; answer out: **healthy**, **skin problem** (with the most likely type) or
**unsure**. An indication to see a vet, never a diagnosis. Dogs only (cats return nothing).

Classes (dataset "Dog's skin diseases (Image Dataset)", Kaggle, Youssef Mohamed):
healthy, dermatitis, fungal_infections, hypersensitivity, demodicosis, ringworm.

Model: EfficientNet-B0 pretrained on ImageNet, fine-tuned (head first, then everything),
class-weighted loss, early stopping on validation macro-F1. Inference: onnxruntime on CPU.

| File | Purpose |
|---|---|
| `prepare_data.py` | scan the dataset, write `manifest.csv`, drop test/valid images that also exist in train |
| `train.py` | two-phase training and honest test evaluation |
| `export_onnx.py` | `.pt` to a single `m5_skin.onnx` + `m5_skin.json`, with parity check |
| `integration/health.py` | backend inference, copy to `backend/app/ai/health.py` |
| `integration/PATCHES.md` | edits to base.py, schemas.py, routers/ai.py, Report.tsx |
| `tests/test_m5.py` | 16 tests, `pytest tests` (no GPU) |

Limits to state in the report: dogs only; no wound/injury class; images come from one public
dataset so real phone photos may do worse; the result is not a diagnosis.

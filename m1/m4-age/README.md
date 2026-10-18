# M4 · Age prediction

Photo(s) of a dog or cat in, one of three groups out, matching the Report form:

| Group | Meaning |
|---|---|
| `young` | puppy / kitten, under 1 year |
| `adult` | 1 to 7 years |
| `senior` | 8+ years |

If the averaged confidence is below a threshold chosen on validation data, the answer is
"Not sure" (`group = None`) and the form keeps its "Not sure" option.

Model: EfficientNet-B0 pretrained on ImageNet, fine-tuned on both species together.
Inference runs on CPU with onnxruntime (no PyTorch on the server).

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | PetFinder.my `train.csv` + photos to `manifest.csv`, split by pet 70/15/15 |
| `train.py` | two-phase training, class weights, early stopping, honest evaluation |
| `export_onnx.py` | `.pt` to `.onnx` + `.json` (preprocessing, threshold), with a parity check |
| `integration/age.py` | backend inference, copy to `backend/app/ai/age.py` |
| `integration/PATCHES.md` | the small edits in `base.py`, `demo.py`, `ai.py`, `schemas.py`, `Report.tsx` |
| `tests/test_m4.py` | 19 tests, run with `pytest tests` (no GPU needed) |

## Steps

1. Download "PetFinder.my Adoption Prediction" from Kaggle (needs a Kaggle account).
2. Open Colab or Kaggle with a GPU, upload this folder, then:
   ```
   pip install -r requirements.txt
   python prepare_data.py --csv train.csv --images train_images --out data/manifest.csv
   python train.py --manifest data/manifest.csv --out models
   python export_onnx.py --ckpt models/m4_age.pt --out models
   ```
3. Read `models/metrics.json`. Check that macro-F1 clearly beats
   `baseline_always_majority_macro_f1`, and look at the confusion matrix and the per-species
   scores. Expect young vs adult to work better than adult vs senior.
4. Copy `m4_age.onnx` and `m4_age.json` to the backend machine (Drive or Git LFS, not git),
   set `M4_AGE_MODEL_PATH`, then apply `integration/PATCHES.md`.

## Notes

- Shelter ages are guessed and rounded, so labels are noisy. Do not expect perfect accuracy.
- Seniors are rare. Class weights help, and `metrics.json` shows how well they really do.
- Preprocessing lives in one place for inference (`age.preprocess`); `export_onnx.py` uses the
  same function in its parity check, so training and serving cannot drift apart silently.
- Model weights stay out of git.

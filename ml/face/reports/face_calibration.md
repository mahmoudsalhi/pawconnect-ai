# Face login: calibration on LFW

Pipeline: YuNet face detection → 5-landmark alignment → SFace 128-d embedding → cosine similarity (`backend/app/ai/face_engine.py`). Benchmark: LFW, official 6,000 pairs in 10 folds (7,701 images; no usable face found in 8, counted as non-matches).

## Results

| Measure | Value |
|---|---|
| LFW 10-fold accuracy | **99.22%** ± 0.48% |
| ROC AUC | 0.9962 |
| Right person accepted at FAR 1% | 99.17% |
| Right person accepted at FAR 0.1% | 98.40% |

## Login threshold

| Threshold | Strangers let in (FAR) | Right person refused (FRR) | Accuracy |
|---|---|---|---|
| **0.3518** (chosen: FAR ≤ 0.1%) | 0.10% | 1.60% | 99.15% |
| 0.363 (SFace paper) | 0.10% | 1.73% | 99.08% |

Login is a 1:1 check (the person types their email, the face must match that account), limited to 5 failed attempts per 15 minutes, so the per-attempt FAR above is the relevant number.

## Head turn (liveness)

|yaw| (nose offset from the eye midpoint ÷ eye distance) on LFW photos: {'p50': 0.089, 'p90': 0.227, 'p95': 0.282, 'p99': 0.425}. Mirrored-photo sign check: passed. The app asks for one photo facing the camera (|yaw| ≤ 0.15) and one turned the requested way (|yaw| ≥ 0.25).

## Limits

- LFW photos are celebrities in good conditions; webcam photos (lighting, angle, low resolution) will do worse.
- The head-turn check stops a printed photo or a still image, not a video replay or a 3D mask.
- Face login is a convenience, not stronger security than a password. Recommended next step: ask for the password before changing the account email (PawConnect's Account settings don't ask for it yet).

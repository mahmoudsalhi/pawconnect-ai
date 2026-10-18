# M2: lost & found photo matching

Tells whether two photos show the same animal. A CNN turns each photo into a 256-number vector (an embedding); photos of the same animal land close together. It is trained with ArcFace metric learning on folders of photos with **one folder per individual animal**.

The PawConnect backend talks to this service through `backend/app/ai/models.py` (`PawModels.embed` / `.similarity`).

| Part | File |
| --- | --- |
| Embedding network + ArcFace loss | `pawmatch/model.py` |
| Training | `pawmatch/train.py` |
| Recall@k / mAP evaluation | `pawmatch/evaluate.py`, `pawmatch/metrics.py` |
| HTTP service (`/embed`, `/reports`, `/match`, `/health`) | `pawmatch/api.py` |
| Check the model on your own photos | `scripts/check_model.py` |

## Set up

```powershell
cd ml\photo_matching
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

The trained model (`checkpoints/m2.pt`, ~94 MB) is not in git; share it through Drive and place it in `checkpoints/`.

## Run the service

```powershell
$env:CHECKPOINT = "checkpoints/m2.pt"
.\.venv\Scripts\python -m uvicorn pawmatch.api:app --port 8002
```

`GET http://localhost:8002/health` should answer `"trained": true`. (`dev.ps1` at the project root does this for you.)

## Train (Google Colab or any GPU)

Dataset layout, split by animal (never by photo):

```
data/train/<animal_id>/*.jpg
data/val/<animal_id>/*.jpg
```

```powershell
python scripts\make_splits.py --src raw_dataset --dst data --val-frac 0.15
python -m pawmatch.train --train-dir data/train --val-dir data/val --epochs 20 --out checkpoints/m2.pt
python -m pawmatch.evaluate --checkpoint checkpoints/m2.pt --val-dir data/val
```

Trained so far on **DogFaceNet** (Zenodo, CC BY 4.0): dogs only, faces already cropped and aligned.

## Check the model on your own photos

Put one folder per dog in `mes_chiens/` (at least 2 *different* photos each), then:

```powershell
.\.venv\Scripts\python scripts\check_model.py --checkpoint checkpoints/m2.pt --photos-dir mes_chiens
```

It prints the validation metrics stored in the checkpoint, a same-dog vs different-dog test, and a suggested threshold. After retraining, update `CALIBRATION` in `backend/app/ai/models.py` with the new numbers.

## Results (held-out dogs the model never saw)

| Metric | Value |
| --- | --- |
| Recall@1 | 86.3 % |
| Recall@5 | 94.3 % |
| Recall@10 | 96.3 % |
| mAP | 0.70 |

Same-dog photo pairs scored about 0.54 on average and different-dog pairs about 0.03. Photos of the same dog that look very different can still score below some pairs of different dogs, so treat a score as a ranking aid, not a verdict.

## Known limits

- Trained on dogs only. Cats need their own data (e.g. a cat re-identification dataset).
- Training photos are cropped faces; real uploads are full photos. Adding automatic face cropping before the model should help.
- The checks above use a small sample. Re-run them on real PawConnect lost/found pairs as they accumulate.

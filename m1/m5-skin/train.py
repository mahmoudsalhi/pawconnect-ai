"""Train M5 (dog skin problem classifier): EfficientNet-B0, two phases, class weights.

    python train.py --manifest data/manifest.csv --out models
Writes models/m5_skin.pt and models/metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

from common import CLASSES, MEAN, SIZE, STD, choose_threshold, evaluate

dev = "cuda" if torch.cuda.is_available() else "cpu"

train_tf = T.Compose([
    T.RandomResizedCrop(SIZE, scale=(0.6, 1.0)), T.RandomHorizontalFlip(), T.RandomRotation(15),
    T.ColorJitter(0.2, 0.2, 0.15), T.ToTensor(), T.Normalize(MEAN, STD)])
eval_tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])


class Imgs(Dataset):
    def __init__(self, df, tf):
        self.paths, self.y, self.tf = df["path"].tolist(), df["label_idx"].tolist(), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (SIZE, SIZE))
        return self.tf(img), self.y[i]


def make_model():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    return m.to(dev)


@torch.no_grad()
def predict(model, df):
    model.eval()
    out = []
    for x, _ in DataLoader(Imgs(df, eval_tf), batch_size=64, num_workers=2):
        out.append(torch.softmax(model(x.to(dev)), 1).cpu().numpy())
    return np.concatenate(out)


def run_epochs(model, params, lr, epochs, loaders, val_df, loss_fn, best, patience=3):
    from sklearn.metrics import f1_score
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    bad = 0
    for ep in range(epochs):
        model.train()
        total = 0.0
        for x, y in loaders:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        probs = predict(model, val_df)
        f1 = f1_score(val_df["label_idx"], probs.argmax(1), average="macro", zero_division=0)
        print(f"  epoch {ep + 1}/{epochs}  loss {total / len(loaders.dataset):.3f}  val macro-F1 {f1:.3f}")
        if f1 > best["f1"]:
            best["f1"], best["state"], bad = f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                print("  early stop")
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--head-epochs", type=int, default=3)
    ap.add_argument("--ft-epochs", type=int, default=12)
    a = ap.parse_args()
    torch.manual_seed(0)

    df = pd.read_csv(a.manifest)
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "valid", "test"))
    if va.empty:  # no valid folder: carve 15% out of train
        va = tr.sample(frac=0.15, random_state=0)
        tr = tr.drop(va.index).reset_index(drop=True)
        va = va.reset_index(drop=True)
    print(f"device={dev}  train={len(tr)}  valid={len(va)}  test={len(te)}")

    counts = tr["label_idx"].value_counts().reindex(range(len(CLASSES)), fill_value=1).values.astype(float)
    weights = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32, device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    loader = DataLoader(Imgs(tr, train_tf), batch_size=32, shuffle=True, num_workers=2, drop_last=True)

    model, best = make_model(), {"f1": -1.0, "state": None}
    for p in model.features.parameters():
        p.requires_grad = False
    print("Phase 1: head only")
    run_epochs(model, model.classifier.parameters(), 1e-3, a.head_epochs, loader, va, loss_fn, best)
    for p in model.features.parameters():
        p.requires_grad = True
    print("Phase 2: full fine-tuning")
    run_epochs(model, model.parameters(), 1e-4, a.ft_epochs, loader, va, loss_fn, best)

    model.load_state_dict(best["state"])
    val_probs = predict(model, va)
    threshold = choose_threshold(val_probs, va["label_idx"].values)
    majority = int(tr["label_idx"].value_counts().idxmax())
    metrics = {"val_macro_f1": best["f1"], "threshold": threshold}
    if len(te):
        metrics["test"] = evaluate(predict(model, te), te["label_idx"].values, majority)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best["state"], "classes": CLASSES}, out / "m5_skin.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.get("test", {}).items() if k != "confusion_matrix"}, indent=2))
    print("confusion matrix (rows=true, cols=pred):")
    for c, row in zip(CLASSES, metrics.get("test", {}).get("confusion_matrix", [])):
        print(f"  {c:18s}", row)
    print("threshold:", threshold)


if __name__ == "__main__":
    main()

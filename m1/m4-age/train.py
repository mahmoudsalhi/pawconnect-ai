"""Train M4 · Age prediction: EfficientNet-B0, 3 classes (young / adult / senior).

Run on Colab or Kaggle with a GPU:
    python train.py --manifest data/manifest.csv --out models

Phase 1: backbone frozen, only the new last layer trains (lr 1e-3).
Phase 2: everything trains (lr 1e-4) with early stopping on validation macro-F1.
Outputs: models/m4_age.pt and models/metrics.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from common import (DEFAULT_THRESHOLD, GROUPS, MEAN, SIZE, STD, aggregate_by_pet,
                    choose_threshold, evaluate)

LABEL = {g: i for i, g in enumerate(GROUPS)}


def build_model(pretrained: bool = True) -> nn.Module:
    model = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(GROUPS))
    return model


class PetPhotos(Dataset):
    def __init__(self, df: pd.DataFrame, tfm):
        self.paths = df["path"].tolist()
        self.labels = df["group"].map(LABEL).tolist()
        self.tfm = tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tfm(img), self.labels[i]


def transforms_for(train: bool):
    norm = transforms.Normalize(MEAN, STD)
    if train:
        # Mild colour jitter only: a grey muzzle is a real sign of age.
        return transforms.Compose([
            transforms.RandomResizedCrop(SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(), norm])
    return transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(), norm])


@torch.no_grad()
def predict_probs(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        out.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
    return np.concatenate(out)


def train_epochs(model, loader, val_loader, val_labels, optimizer, loss_fn, device,
                 epochs, patience, tag, best):
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        probs = predict_probs(model, val_loader, device)
        f1 = f1_score(val_labels, probs.argmax(1), average="macro",
                      labels=list(range(len(GROUPS))), zero_division=0)
        print(f"[{tag}] epoch {epoch:02d} loss {total / len(loader.dataset):.4f} val macro-F1 {f1:.4f}")
        if f1 > best["f1"]:
            best["f1"] = f1
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"[{tag}] early stop")
                break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs-head", type=int, default=3)
    ap.add_argument("--epochs-full", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)
    parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}

    def loader(frame, train):
        return DataLoader(PetPhotos(frame, transforms_for(train)), batch_size=args.batch_size,
                          shuffle=train, num_workers=args.workers)

    train_loader = loader(parts["train"], True)
    val_loader = loader(parts["val"], False)
    test_loader = loader(parts["test"], False)
    val_labels = parts["val"]["group"].map(LABEL).to_numpy()

    # Inverse-frequency class weights fight the lack of senior animals.
    counts = parts["train"]["group"].value_counts().reindex(GROUPS).fillna(1).to_numpy()
    weights = torch.tensor(counts.sum() / (len(GROUPS) * counts), dtype=torch.float32)
    print("train photos per class", dict(zip(GROUPS, counts.astype(int))), "weights", weights.tolist())
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    model = build_model().to(device)
    best = {"f1": -1.0, "state": None}

    for p in model.features.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    train_epochs(model, train_loader, val_loader, val_labels, opt, loss_fn, device,
                 args.epochs_head, patience=99, tag="head", best=best)

    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_epochs(model, train_loader, val_loader, val_labels, opt, loss_fn, device,
                 args.epochs_full, args.patience, tag="full", best=best)

    model.load_state_dict(best["state"])

    # Threshold chosen on validation pets (photos of one pet averaged together).
    val_pets, val_pp, val_pl = aggregate_by_pet(parts["val"], predict_probs(model, val_loader, device))
    threshold = choose_threshold(val_pp, val_pl)

    majority = int(np.argmax(counts))
    test_probs = predict_probs(model, test_loader, device)
    test_pets, test_pp, test_pl = aggregate_by_pet(parts["test"], test_probs)
    per_photo = evaluate(test_probs, parts["test"]["group"].map(LABEL).to_numpy(),
                         parts["test"]["species"].to_numpy(), majority)
    per_pet = evaluate(test_pp, test_pl, test_pets["species"].to_numpy(), majority)

    conf = test_pp.max(axis=1)
    keep = conf >= threshold
    answered = {
        "threshold": threshold,
        "coverage": round(float(keep.mean()), 4),
        "accuracy_when_answered": round(float((test_pp.argmax(1)[keep] == test_pl[keep]).mean()), 4)
        if keep.any() else None,
    }
    metrics = {"best_val_macro_f1_photo": round(float(best["f1"]), 4), "test_per_photo": per_photo,
               "test_per_pet": per_pet, "test_with_not_sure_threshold": answered}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save({"state_dict": model.state_dict(), "groups": GROUPS,
                "threshold": threshold or DEFAULT_THRESHOLD}, out_dir / "m4_age.pt")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved {out_dir / 'm4_age.pt'}. Next: python export_onnx.py")


if __name__ == "__main__":
    main()

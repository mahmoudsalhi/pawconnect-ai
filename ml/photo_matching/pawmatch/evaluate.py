"""Retrieval evaluation: Recall@k and mAP on held-out identities.

Protocol: embed every validation image. For each image (the "found" query),
rank all OTHER images (the "lost" gallery) by cosine similarity. A hit is any
image of the same animal.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import IdentityDataset, eval_transform
from .metrics import retrieval_metrics
from .model import load_checkpoint


@torch.no_grad()
def embed_dataset(model, dataset, device="cpu", batch_size=64, num_workers=2):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    embs, labels = [], []
    for x, y in tqdm(loader, desc="embedding"):
        embs.append(model(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(embs).numpy(), torch.cat(labels).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = load_checkpoint(args.checkpoint, args.device)
    ds = IdentityDataset(args.val_dir, eval_transform())
    embs, labels = embed_dataset(model, ds, args.device)
    for k, v in retrieval_metrics(embs, labels).items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()

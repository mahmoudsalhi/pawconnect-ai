"""Train the embedding model with ArcFace on identity folders.

    python -m pawmatch.train --train-dir data/train --val-dir data/val \
        --epochs 20 --out checkpoints/m2.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import IdentityDataset, eval_transform, train_transform
from .evaluate import embed_dataset
from .metrics import retrieval_metrics
from .model import ArcFace, EmbeddingNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--out", default="checkpoints/m2.pt")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--warmup-epochs", type=int, default=2, help="train head only first")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    train_ds = IdentityDataset(args.train_dir, train_transform(args.img_size))
    val_ds = IdentityDataset(args.val_dir, eval_transform(args.img_size))
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True, pin_memory=device == "cuda",
    )
    print(f"train: {len(train_ds)} images / {train_ds.num_classes} animals | "
          f"val: {len(val_ds)} images / {val_ds.num_classes} animals")

    model = EmbeddingNet(args.backbone, args.embed_dim, pretrained=not args.no_pretrained).to(device)
    arc = ArcFace(args.embed_dim, train_ds.num_classes).to(device)

    params = [
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},  # gentle on pretrained
        {"params": model.head.parameters(), "lr": args.lr},
        {"params": arc.parameters(), "lr": args.lr},
    ]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best = -1.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.freeze_backbone(epoch < args.warmup_epochs)
        model.train()
        total = 0.0
        for x, y in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.split(":")[0], enabled=device == "cuda"):
                loss = arc(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
        sched.step()

        model.eval()
        embs, labels = embed_dataset(model, val_ds, device, num_workers=args.workers)
        m = retrieval_metrics(embs, labels)
        print(f"epoch {epoch + 1}: loss={total / len(loader):.3f} " +
              " ".join(f"{k}={v:.3f}" for k, v in m.items()))

        if m["mAP"] > best:
            best = m["mAP"]
            torch.save(
                {"model": model.state_dict(), "backbone": args.backbone,
                 "embed_dim": args.embed_dim, "img_size": args.img_size, "metrics": m},
                args.out,
            )
            print(f"  saved best (mAP={best:.3f}) -> {args.out}")


if __name__ == "__main__":
    main()

"""Split a folder-per-animal dataset into train/val BY ANIMAL.

    python scripts/make_splits.py --src raw_dataset --dst data --val-frac 0.15

Hard-links/copies images so the original stays untouched. Animals with fewer
than 2 photos are skipped (nothing to match against).
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="data")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    animals = [d for d in sorted(src.iterdir()) if d.is_dir()
               and sum(p.suffix.lower() in IMG_EXTS for p in d.iterdir()) >= 2]
    random.Random(args.seed).shuffle(animals)
    n_val = max(1, int(len(animals) * args.val_frac))
    for i, d in enumerate(animals):
        split = "val" if i < n_val else "train"
        out = dst / split / d.name
        out.mkdir(parents=True, exist_ok=True)
        for p in d.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                shutil.copy2(p, out / p.name)
    print(f"{len(animals) - n_val} train animals, {n_val} val animals -> {dst}")


if __name__ == "__main__":
    main()

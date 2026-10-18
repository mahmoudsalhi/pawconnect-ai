"""Scan the Kaggle 'Dog's skin diseases (Image Dataset)' folder and write data/manifest.csv.

Expected layout (anywhere under --root):  train/<class>/*.jpg, valid/<class>/*.jpg, test/<class>/*.jpg
Duplicates are removed: an image of valid/test whose bytes also appear in train is dropped, so the
test score is not inflated by leaked copies.

    python prepare_data.py --root dataset --out data/manifest.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import CLASSES, IMG_EXT, SPLIT_ALIASES, file_md5, normalize_class


def find_split_dirs(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for d in sorted(root.rglob("*")):
        if d.is_dir() and d.name.lower() in SPLIT_ALIASES:
            split = SPLIT_ALIASES[d.name.lower()]
            # keep the shallowest folder for each split
            if split not in found or len(d.parts) < len(found[split].parts):
                found[split] = d
    return found


def scan(root: Path) -> tuple[pd.DataFrame, list[str]]:
    splits = find_split_dirs(root)
    if "train" not in splits:
        raise SystemExit(f"No 'train' folder found under {root}. Check the download.")
    rows, skipped = [], []
    for split, sdir in splits.items():
        for cdir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            cls = normalize_class(cdir.name)
            if cls is None:
                skipped.append(f"{split}/{cdir.name}")
                continue
            for f in sorted(cdir.rglob("*")):
                if f.suffix.lower() in IMG_EXT:
                    rows.append({"path": str(f.resolve()), "split": split, "label": cls,
                                 "label_idx": CLASSES.index(cls)})
    return pd.DataFrame(rows), skipped


def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["md5"] = [file_md5(p) for p in df["path"]]
    n0 = len(df)
    train_hashes = set(df.loc[df.split == "train", "md5"])
    leaked = (df.split != "train") & df["md5"].isin(train_hashes)
    df = df[~leaked]
    # exact duplicates inside the same split
    df = df.drop_duplicates(subset=["split", "md5"])
    return df.reset_index(drop=True), {"before": n0, "after": len(df), "leaked_from_train": int(leaked.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder where the dataset was extracted")
    ap.add_argument("--out", default="data/manifest.csv")
    a = ap.parse_args()

    df, skipped = scan(Path(a.root))
    if skipped:
        print("Ignored folders (unknown class):", skipped)
    df, info = dedupe(df)
    print("Duplicates:", info)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.pivot_table(index="label", columns="split", values="path", aggfunc="count", fill_value=0))
    missing = [c for c in CLASSES if c not in set(df.label)]
    if missing:
        print("WARNING: classes with no image:", missing)


if __name__ == "__main__":
    main()

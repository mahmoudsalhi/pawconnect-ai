"""Build the training manifest from the PetFinder.my Adoption Prediction data (Kaggle).

Usage:
    python prepare_data.py --csv path/to/train.csv --images path/to/train_images --out data/manifest.csv

train.csv needs: PetID, Type (1 = dog, 2 = cat), Age (months).
Photos are named {PetID}-{n}.jpg.

The split is done by PetID (70/15/15) so photos of one pet never sit in two splits.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from common import group_from_months

SPECIES = {1: "dog", 2: "cat"}


def index_images(image_dir: str, max_per_pet: int) -> dict[str, list[str]]:
    """Map PetID -> sorted photo paths (at most max_per_pet)."""
    found: dict[str, list[str]] = {}
    for name in sorted(os.listdir(image_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in {".jpg", ".jpeg", ".png"} or "-" not in stem:
            continue
        pet_id = stem.rsplit("-", 1)[0]
        found.setdefault(pet_id, []).append(os.path.join(image_dir, name))
    return {k: v[:max_per_pet] for k, v in found.items()}


def build_manifest(df: pd.DataFrame, photos: dict[str, list[str]], seed: int = 42) -> pd.DataFrame:
    pets = df[df["Type"].isin(SPECIES)].copy()
    pets = pets[pets["PetID"].isin(photos)]
    pets = pets.dropna(subset=["Age"])
    pets["species"] = pets["Type"].map(SPECIES)
    pets["group"] = pets["Age"].map(group_from_months)
    pets = pets.rename(columns={"PetID": "pet_id"})[["pet_id", "species", "group", "Age"]]
    pets = pets.drop_duplicates("pet_id").reset_index(drop=True)

    def can_stratify(frame: pd.DataFrame) -> bool:
        return frame["group"].value_counts().min() >= 2

    train_pets, rest = train_test_split(
        pets, test_size=0.30, random_state=seed,
        stratify=pets["group"] if can_stratify(pets) else None)
    val_pets, test_pets = train_test_split(
        rest, test_size=0.50, random_state=seed,
        stratify=rest["group"] if can_stratify(rest) else None)

    split_of = {**{p: "train" for p in train_pets["pet_id"]},
                **{p: "val" for p in val_pets["pet_id"]},
                **{p: "test" for p in test_pets["pet_id"]}}
    rows = []
    for rec in pets.itertuples(index=False):
        for path in photos[rec.pet_id]:
            rows.append({"path": path, "pet_id": rec.pet_id, "species": rec.species,
                         "group": rec.group, "age_months": rec.Age,
                         "split": split_of[rec.pet_id]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="data/manifest.csv")
    ap.add_argument("--max-per-pet", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    manifest = build_manifest(df, index_images(args.images, args.max_per_pet), args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, index=False)

    print(f"{len(manifest)} photos, {manifest['pet_id'].nunique()} pets -> {args.out}")
    print("\nPets per split and group:")
    pets = manifest.drop_duplicates("pet_id")
    print(pd.crosstab(pets["split"], pets["group"]))
    print("\nPets per species and group:")
    print(pd.crosstab(pets["species"], pets["group"]))


if __name__ == "__main__":
    main()

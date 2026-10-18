"""Dataset loading.

Expected layout (one folder per individual animal):

    data/
      train/
        animal_0001/ img1.jpg img2.jpg ...
        animal_0002/ ...
      val/
        animal_9001/ ...

Split by IDENTITY, never by image: the validation animals must be animals the
model has never seen, because in production every lost pet is new.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def train_transform(size: int = 224) -> T.Compose:
    # Augmentations mimic real lost/found photo differences: angle, light,
    # blur, partial crops.
    return T.Compose(
        [
            T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(0.3, 0.3, 0.3, 0.05),
            T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.3),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
            T.RandomErasing(p=0.2),
        ]
    )


def eval_transform(size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ]
    )


def scan_identities(root: str | Path, min_images: int = 2) -> list[tuple[str, list[Path]]]:
    root = Path(root)
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
        if len(imgs) >= min_images:
            out.append((d.name, imgs))
    return out


class IdentityDataset(Dataset):
    def __init__(self, root: str | Path, transform=None, min_images: int = 2):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self.class_names: list[str] = []
        for label, (name, imgs) in enumerate(scan_identities(root, min_images)):
            self.class_names.append(name)
            self.samples.extend((p, label) for p in imgs)
        if not self.samples:
            raise ValueError(f"No identities with >= {min_images} images found in {root}")

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

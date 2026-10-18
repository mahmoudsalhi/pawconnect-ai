"""Vérifie le modèle M2 sur TES photos et affiche les chiffres du rapport.

Usage (depuis le dossier du projet, venv actif) :

    python scripts/check_model.py --checkpoint checkpoints/m2.pt --photos-dir mes_chiens

Structure de mes_chiens : un dossier par chien, au moins 2 photos DIFFÉRENTES
du même chien dans chaque dossier.

    mes_chiens/
      rex/    photo1.jpg photo2.jpg
      luna/   photo1.jpg photo2.jpg
      max/    photo1.jpg photo2.jpg

Le script affiche : (1) les métriques de validation enregistrées pendant
l'entraînement (recall@k, mAP), (2) le test « même chien » sur tes photos,
(3) un seuil conseillé pour MATCH_THRESHOLD.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pawmatch.dataset import IMG_EXTS, eval_transform  # noqa: E402
from pawmatch.model import load_checkpoint  # noqa: E402

try:  # lecture des .avif si le plugin est installé
    import pillow_avif  # noqa: F401
    IMG_EXTS = IMG_EXTS | {".avif"}
except ImportError:
    pass

W_VISUAL, W_META, META_NEUTRAL = 0.8, 0.2, 0.5  # mêmes valeurs que matcher.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--photos-dir", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    print("=== 1. Métriques de validation enregistrées pendant l'entraînement ===")
    m = ckpt.get("metrics")
    if m:
        for k, v in m.items():
            print(f"  {k}: {v:.4f}")
        print("  (animaux de validation jamais vus à l'entraînement)")
    else:
        print("  Aucune métrique dans ce checkpoint.")

    model = load_checkpoint(args.checkpoint, "cpu")
    tf = eval_transform(ckpt.get("img_size", 224))

    names, labels, embs = [], [], []
    root = Path(args.photos_dir)
    dogs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    for li, d in enumerate(dogs):
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
        if len(files) < 2:
            print(f"  ignoré : {d.name} (moins de 2 photos)")
            continue
        for p in files:
            img = Image.open(p).convert("RGB")
            with torch.no_grad():
                e = model(tf(img).unsqueeze(0)).squeeze(0).numpy()
            names.append(f"{d.name}/{p.name}")
            labels.append(li)
            embs.append(e)
    if len(set(labels)) < 2:
        sys.exit("Il faut au moins 2 chiens avec 2 photos chacun.")

    E, L = np.array(embs), np.array(labels)
    S = E @ E.T
    iu = np.triu_indices(len(L), k=1)
    sims, same = S[iu], (L[:, None] == L[None, :])[iu]
    pos, neg = sims[same], sims[~same]

    print("\n=== 2. Test « même chien » sur tes photos ===")
    print(f"  {len(set(labels))} chiens, {len(L)} photos, {len(pos)} paires même chien, {len(neg)} paires chiens différents")
    print(f"  Même chien        : moyenne={pos.mean():.3f}  min={pos.min():.3f}  max={pos.max():.3f}")
    print(f"  Chiens différents : moyenne={neg.mean():.3f}  min={neg.min():.3f}  max={neg.max():.3f}")

    # Pour chaque photo, la photo la plus proche (hors elle-même) est-elle le même chien ?
    S2 = S.copy()
    np.fill_diagonal(S2, -np.inf)
    nn = S2.argmax(axis=1)
    top1 = float((L[nn] == L).mean())
    print(f"  Photo la plus proche = même chien : {top1:.0%} des photos")

    print("\n=== 3. Seuil conseillé ===")
    gap = pos.min() - neg.max()
    if gap > 0:
        t = (pos.min() + neg.max()) / 2
        print(f"  Les deux groupes sont séparés (écart {gap:.3f}). Seuil visuel ≈ {t:.2f}")
    else:
        t = float(np.median(np.concatenate([pos, neg])))
        for cand in np.linspace(0, 1, 201):
            tp, fp = (pos >= cand).sum(), (neg >= cand).sum()
            if tp + fp and tp / (tp + fp) >= 0.9:
                t = float(cand)
                break
        print("  Les groupes se chevauchent : seuil visuel donnant ~90 % de précision "
              f"sur tes photos ≈ {t:.2f} (échantillon petit : à confirmer).")
    print(f"  MATCH_THRESHOLD (score fusionné, métadonnées neutres) ≈ {W_VISUAL * t + W_META * META_NEUTRAL:.2f}")

    print("\nDétail des paires (les plus proches d'abord) :")
    order = np.argsort(-sims)
    ia, ib = iu
    for k in order[:15]:
        tag = "MÊME" if same[k] else "diff"
        print(f"  {sims[k]:.3f}  {tag}  {names[ia[k]]}  <->  {names[ib[k]]}")


if __name__ == "__main__":
    main()

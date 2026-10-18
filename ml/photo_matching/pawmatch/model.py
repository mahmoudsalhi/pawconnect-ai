"""Embedding network: image -> L2-normalised vector.

Two photos of the same animal should land close together (high cosine
similarity), photos of different animals far apart.
"""
from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        embed_dim: int = 256,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return F.normalize(self.head(feats), dim=-1)

    def freeze_backbone(self, freeze: bool = True) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = not freeze


class ArcFace(nn.Module):
    """Additive angular margin loss (Deng et al., 2019).

    Used only for training: it turns "which animal is this?" into a
    classification task over training identities, with a margin that forces
    tight, well-separated clusters. It is discarded at inference.
    """

    def __init__(self, embed_dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.4):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cos = F.linear(emb, F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        sin = torch.sqrt(1.0 - cos * cos)
        phi = cos * self.cos_m - sin * self.sin_m
        phi = torch.where(cos > self.th, phi, cos - self.mm)
        one_hot = F.one_hot(labels, cos.size(1)).float()
        logits = self.scale * (one_hot * phi + (1.0 - one_hot) * cos)
        return F.cross_entropy(logits, labels)


def load_checkpoint(path: str, device: str = "cpu") -> EmbeddingNet:
    ckpt = torch.load(path, map_location=device)
    model = EmbeddingNet(
        backbone=ckpt["backbone"], embed_dim=ckpt["embed_dim"], pretrained=False
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()

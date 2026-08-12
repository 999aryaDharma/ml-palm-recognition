"""
ArcFace Loss untuk PalmNet-Lite metric learning.

Referensi: Deng et al. 2019, "ArcFace: Additive Angular Margin Loss"

Digunakan pada Phase 2 training. Backbone Phase 2 diinisialisasi dari
checkpoint Phase 1 (own checkpoint — bukan external pretrained).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """ArcFace dengan additive angular margin.

    Penggunaan:
        loss_fn = ArcFaceLoss(in_features=128, num_classes=400,
                              margin=0.30, scale=32.0)
        emb = backbone(images)          # [B, 128] raw (belum normalized)
        loss = loss_fn(emb, labels)     # ArcFace handles normalization
        loss.backward()

    ArcFace weight diinisialisasi Xavier (terpisah dari backbone init).

    Args:
        in_features:  embedding dimension (128)
        num_classes:  jumlah kelas training (jumlah identity palm training)
        margin:       angular margin m dalam radian (baseline: 0.30)
        scale:        feature scale s (baseline: 32.0)
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        margin: float = 0.30,
        scale: float = 32.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # Learnable class prototypes — L2-normalized saat forward
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self._update_margin_constants()

    def _update_margin_constants(self) -> None:
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        # Threshold untuk tail handling
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def set_margin(self, margin: float) -> None:
        """Update margin (untuk margin warm-up)."""
        self.margin = margin
        self._update_margin_constants()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, in_features] raw embedding dari backbone
            labels:     [B,] long tensor class indices

        Returns:
            Scalar cross-entropy loss dengan angular margin.
        """
        emb_norm = F.normalize(embeddings, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(emb_norm, w_norm).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=0.0))

        phi = cosine * self.cos_m - sine * self.sin_m
        # Tail linear fallback untuk angle > pi - margin
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits = logits * self.scale

        return F.cross_entropy(logits, labels)


class LinearMarginWarmup:
    """Linear margin warmup: dari 0 → target_margin selama warmup_epochs.

    Penggunaan:
        warmup = LinearMarginWarmup(arcface_loss, target=0.30, warmup_epochs=8)
        for epoch in range(num_epochs):
            current_margin = warmup.step(epoch)
            # train...

    Args:
        arcface_loss:   ArcFaceLoss instance
        target_margin:  target margin akhir
        warmup_epochs:  jumlah epoch untuk ramp-up
    """

    def __init__(
        self,
        arcface_loss: ArcFaceLoss,
        target_margin: float,
        warmup_epochs: int,
    ):
        self.loss_fn = arcface_loss
        self.target = target_margin
        self.warmup_epochs = warmup_epochs

    def step(self, epoch: int) -> float:
        """Update margin untuk epoch ini. Return margin yang digunakan."""
        if epoch >= self.warmup_epochs:
            current = self.target
        else:
            current = self.target * (epoch + 1) / self.warmup_epochs
        self.loss_fn.set_margin(current)
        return current

"""
ArcFace Loss — Additive Angular Margin untuk metric learning.

Referensi:
- Deng et al. 2019, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"

Intuisi singkat:
- Embedding di-L2-normalize, weights di-L2-normalize → cos(theta) = dot product
- Untuk kelas correct, tambahkan margin angular m: cos(theta + m) < cos(theta)
- Hasilnya: kelas correct dipaksa "lebih sulit" → embedding antar-kelas
  terpisah lebih jauh
- Scale s mengurangi softness softmax (sebanding dengan temperature 1/s)

Best practice untuk training:
- Linear warm-up margin dari 0 → m selama 5 epoch (stabilitas awal)
- Gradient clipping (max_norm=5.0) untuk hindari ledakan loss
- Xavier init untuk weight matrix (kelas prototype)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """
    ArcFace loss dengan additive angular margin.

    Penggunaan:
        loss_fn = ArcFaceLoss(in_features=128, num_classes=400, margin=0.5, scale=64)
        embeddings = model(images)              # (B, 128) — raw, BELUM dinormalisasi
        loss = loss_fn(embeddings, labels)      # ArcFace handle normalisasi internal
        loss.backward()

    Catatan:
    - Embedding input ke loss ini SEHARUSNYA belum dinormalisasi
      (loss yang akan normalisasi). Ini penting untuk gradient flow.
    - Untuk inference (di luar loss), L2-normalize manual:
        emb = F.normalize(model(x), dim=1)
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # Learnable kelas prototype (akan di-L2-normalize saat forward)
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute trigonometric constants (re-compute kalau margin diubah)
        self._update_margin_constants()

    def _update_margin_constants(self):
        """Re-compute kalau margin di-update (saat warm-up)."""
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        # Threshold: untuk theta > pi - margin, phi function jadi non-monotonic
        # → fallback ke cosine - mm (linear approximation untuk angle besar)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def set_margin(self, margin: float):
        """Untuk margin warm-up (linear 0 → target selama N epoch)."""
        self.margin = margin
        self._update_margin_constants()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, in_features) — RAW embedding dari backbone
            labels: (B,) long tensor — class indices

        Returns:
            Scalar loss (cross-entropy dengan margin).
        """
        # L2-normalize embedding dan weights
        emb_norm = F.normalize(embeddings, dim=1)
        w_norm = F.normalize(self.weight, dim=1)

        # Cosine similarity: (B, num_classes)
        cosine = F.linear(emb_norm, w_norm).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # sin = sqrt(1 - cos^2) — pakai clamp untuk numerical stability
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=0.0))

        # phi = cos(theta + margin) = cos*cos_m - sin*sin_m
        phi = cosine * self.cos_m - sine * self.sin_m

        # Untuk theta > pi - margin (cosine < th), pakai linear fallback:
        # phi = cosine - mm
        # (mencegah non-monotonic behavior di tail)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot mask: hanya kelas correct yang dapat margin
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        # Logit: kelas correct → phi (cosine + margin penalty)
        #        kelas lain     → cosine biasa
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits = logits * self.scale

        return F.cross_entropy(logits, labels)


class LinearMarginScheduler:
    """
    Linear warm-up margin: dari 0 → target_margin selama warmup_epochs.
    Setelah itu margin stay di target_margin.

    Penggunaan di training loop:
        scheduler = LinearMarginScheduler(arcface_loss, target=0.5, warmup_epochs=5)
        for epoch in range(num_epochs):
            scheduler.step(epoch)  # update margin
            for batch in loader:
                ...
    """

    def __init__(self, arcface_loss: ArcFaceLoss, target_margin: float, warmup_epochs: int):
        self.loss = arcface_loss
        self.target = target_margin
        self.warmup_epochs = warmup_epochs

    def step(self, epoch: int):
        if epoch >= self.warmup_epochs:
            current = self.target
        else:
            current = self.target * (epoch + 1) / self.warmup_epochs
        self.loss.set_margin(current)
        return current


if __name__ == "__main__":
    # Smoke test
    loss_fn = ArcFaceLoss(in_features=128, num_classes=400, margin=0.5, scale=64.0)
    emb = torch.randn(8, 128, requires_grad=True)
    labels = torch.randint(0, 400, (8,))

    loss = loss_fn(emb, labels)
    print(f"Loss: {loss.item():.4f}")
    loss.backward()
    print(f"Gradient norm: {emb.grad.norm().item():.4f}")

    # Test margin scheduler
    sched = LinearMarginScheduler(loss_fn, target_margin=0.5, warmup_epochs=5)
    for ep in range(7):
        m = sched.step(ep)
        print(f"Epoch {ep}: margin = {m:.3f}")

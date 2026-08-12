"""
PalmNet-Lite v1 — Lightweight CNN for Palm Biometric Embedding.

Canonical architecture sesuai docs/02-model-architecture.md.

model_id             : palmnet-lite-scratch
architecture_name    : PalmNetLite
architecture_version : v1
training_mode        : scratch

Input:  [B, 3, 112, 112] float32 RGB, normalized mean=0.5 std=0.5
Output: [B, 128]  raw embedding (belum L2-normalized)
        Untuk inference: gunakan PalmNetLiteInference atau
        F.normalize(backbone(x), p=2, dim=1)

Stage layout (sesuai docs):
    Stem           : Conv 3x3 s2 -> BN -> PReLU      : 3 -> 32, 112 -> 56
    DW Stem        : DWConv 3x3 s1 -> BN -> PReLU    : 32 -> 32, 56
    Stage 1        : 2x InvRes(32->32, s1)            : 56
    Stage 2        : InvRes(32->64,s2) + 2x(64->64,s1): 28
    Stage 3        : InvRes(64->96,s2) + 3x(96->96,s1): 14
    Stage 4        : InvRes(96->128,s2) + (128->128,s1): 7
    Final Proj     : Conv 1x1 -> BN -> PReLU          : 128->256, 7
    GDConv         : DWConv 7x7 -> BN                 : 256, 1x1
    Embedding      : Flatten -> Linear 256->128 -> BN1d: 128-D

Expected param count: ~393k (sanity range 350k-450k)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from palm_recognition.models.blocks import (
    ConvBnPrelu,
    InvertedResidual,
    GlobalDepthwiseConv,
)


def _make_stage(
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    first_stride: int,
    expansion: int = 2,
    prelu_init: float = 0.25,
) -> nn.Sequential:
    """Build one stage of InvertedResidual blocks.

    First block may have stride=2 (downsampling) and channel change -> no residual.
    Subsequent blocks have stride=1 and in==out -> residual.
    """
    blocks = []
    for i in range(num_blocks):
        stride = first_stride if i == 0 else 1
        in_c = in_channels if i == 0 else out_channels
        blocks.append(
            InvertedResidual(
                in_channels=in_c,
                out_channels=out_channels,
                stride=stride,
                expansion=expansion,
                prelu_init=prelu_init,
            )
        )
    return nn.Sequential(*blocks)


class PalmNetLite(nn.Module):
    """PalmNet-Lite v1 Backbone.

    Seluruh weight dimulai dari random initialization.
    Tidak ada pretrained weights, tidak ada external checkpoint.

    Gunakan initialize_scratch_weights(model) setelah konstruksi.

    forward() mengembalikan raw embedding [B, 128].
    Untuk inference L2-normalized, gunakan PalmNetLiteInference.
    """

    ARCHITECTURE_VERSION = "v1"

    def __init__(
        self,
        input_channels: int = 3,
        stem_channels: int = 32,
        stage_channels: tuple[int, ...] = (32, 64, 96, 128),
        stage_repeats: tuple[int, ...] = (2, 3, 4, 2),
        projection_channels: int = 256,
        embedding_dim: int = 128,
        expansion: int = 2,
        prelu_init: float = 0.25,
    ):
        super().__init__()

        assert len(stage_channels) == 4, "Harus ada 4 stage"
        assert len(stage_repeats) == 4, "Harus ada 4 stage repeat count"

        self.embedding_dim = embedding_dim
        self.architecture_version = self.ARCHITECTURE_VERSION

        # ── Stem: 3x112x112 → 32x56x56 ──────────────────────────────────────
        self.stem = ConvBnPrelu(
            in_channels=input_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            prelu_init=prelu_init,
        )

        # ── Depthwise Stem: 32x56x56 → 32x56x56 ─────────────────────────────
        self.dw_stem = ConvBnPrelu(
            in_channels=stem_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=stem_channels,
            prelu_init=prelu_init,
        )

        # ── Stage 1: 32x56x56 → 32x56x56 (2 blocks, stride=1) ────────────────
        self.stage1 = _make_stage(
            in_channels=stem_channels,
            out_channels=stage_channels[0],
            num_blocks=stage_repeats[0],
            first_stride=1,
            expansion=expansion,
            prelu_init=prelu_init,
        )

        # ── Stage 2: 32x56x56 → 64x28x28 (3 blocks, first stride=2) ──────────
        self.stage2 = _make_stage(
            in_channels=stage_channels[0],
            out_channels=stage_channels[1],
            num_blocks=stage_repeats[1],
            first_stride=2,
            expansion=expansion,
            prelu_init=prelu_init,
        )

        # ── Stage 3: 64x28x28 → 96x14x14 (4 blocks, first stride=2) ──────────
        self.stage3 = _make_stage(
            in_channels=stage_channels[1],
            out_channels=stage_channels[2],
            num_blocks=stage_repeats[2],
            first_stride=2,
            expansion=expansion,
            prelu_init=prelu_init,
        )

        # ── Stage 4: 96x14x14 → 128x7x7 (2 blocks, first stride=2) ───────────
        self.stage4 = _make_stage(
            in_channels=stage_channels[2],
            out_channels=stage_channels[3],
            num_blocks=stage_repeats[3],
            first_stride=2,
            expansion=expansion,
            prelu_init=prelu_init,
        )

        # ── Final Projection: 128x7x7 → 256x7x7 ─────────────────────────────
        self.final_proj = ConvBnPrelu(
            in_channels=stage_channels[3],
            out_channels=projection_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            prelu_init=prelu_init,
        )

        # ── GDConv: 256x7x7 → 256x1x1 ───────────────────────────────────────
        self.gdconv = GlobalDepthwiseConv(
            channels=projection_channels,
            kernel_size=7,
        )

        # ── Embedding Head: 256 → 128 ─────────────────────────────────────────
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(projection_channels, embedding_dim, bias=False)
        self.bn_embed = nn.BatchNorm1d(embedding_dim, eps=1e-5, momentum=0.1)
        # No activation after embedding BN per spec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 112, 112] float32 RGB, normalized (mean=0.5, std=0.5)

        Returns:
            [B, 128] raw embedding (NOT L2-normalized).
            Call F.normalize(out, p=2, dim=1) for cosine-ready embedding.
        """
        x = self.stem(x)          # [B, 32, 56, 56]
        x = self.dw_stem(x)       # [B, 32, 56, 56]
        x = self.stage1(x)        # [B, 32, 56, 56]
        x = self.stage2(x)        # [B, 64, 28, 28]
        x = self.stage3(x)        # [B, 96, 14, 14]
        x = self.stage4(x)        # [B, 128, 7, 7]
        x = self.final_proj(x)    # [B, 256, 7, 7]
        x = self.gdconv(x)        # [B, 256, 1, 1]
        x = self.flatten(x)       # [B, 256]
        x = self.fc(x)            # [B, 128]
        x = self.bn_embed(x)      # [B, 128]
        return x

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


class PalmNetLiteInference(nn.Module):
    """Inference wrapper: backbone + L2 normalization.

    Digunakan untuk TorchScript export dan runtime inference.
    Input/Output contract identik dengan MobileFaceNet artifact:
        Input:  [B, 3, 112, 112] float32 RGB normalized
        Output: [B, 128] L2-normalized embedding
    """

    def __init__(self, backbone: PalmNetLite):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)
        return torch.nn.functional.normalize(emb, p=2, dim=1)


def build_palmnet_lite(cfg: dict | None = None) -> PalmNetLite:
    """Build PalmNetLite dari config dict.

    Default menggunakan spec v1 dari docs/02-model-architecture.md.
    """
    defaults = {
        "input_channels": 3,
        "stem_channels": 32,
        "stage_channels": (32, 64, 96, 128),
        "stage_repeats": (2, 3, 4, 2),
        "projection_channels": 256,
        "embedding_dim": 128,
        "expansion": 2,
        "prelu_init": 0.25,
    }
    if cfg:
        defaults.update(cfg)
    return PalmNetLite(**defaults)

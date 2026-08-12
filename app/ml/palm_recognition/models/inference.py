"""
Inference wrapper utilities untuk PalmNet-Lite.

Menyediakan:
1. PalmNetLiteInferenceWrapper - inference dengan L2 normalize
2. load_inference_model() - load backbone dari checkpoint untuk inference
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PalmNetLiteInferenceWrapper(nn.Module):
    """TorchScript-compatible inference wrapper.

    Input:  [B, 3, 112, 112] float32 RGB normalized (mean=0.5, std=0.5)
    Output: [B, 128] L2-normalized embedding
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)
        return F.normalize(emb, p=2, dim=1)


def load_inference_model(
    backbone_class,
    checkpoint_path: str,
    device: str = "cpu",
) -> PalmNetLiteInferenceWrapper:
    """Load backbone dari checkpoint Phase 2 untuk inference.

    Hanya load backbone_state_dict.
    Tidak load classifier, arcface head, optimizer, atau scheduler.

    Args:
        backbone_class: PalmNetLite class
        checkpoint_path: path ke checkpoint .pth
        device: target device

    Returns:
        PalmNetLiteInferenceWrapper dalam eval mode
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Support berbagai format checkpoint
    if "backbone_state_dict" in ckpt:
        state_dict = ckpt["backbone_state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    backbone = backbone_class()
    backbone.load_state_dict(state_dict, strict=True)
    backbone.eval()

    wrapper = PalmNetLiteInferenceWrapper(backbone)
    wrapper.eval()
    return wrapper

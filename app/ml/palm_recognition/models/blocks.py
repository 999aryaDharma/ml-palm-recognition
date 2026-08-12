"""
PalmNet-Lite Building Blocks.

Reusable primitives untuk PalmNet-Lite v1 architecture.
Semua block menggunakan PyTorch primitives — tidak ada pretrained weight.

Spec: docs/02-model-architecture.md
"""
import torch
import torch.nn as nn


class ConvBnPrelu(nn.Module):
    """Conv2d -> BatchNorm2d -> PReLU.

    Dipakai pada stem, depthwise stem, dan final projection.
    bias=False karena langsung diikuti BatchNorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        prelu_init: float = 0.25,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.act = nn.PReLU(num_parameters=out_channels, init=prelu_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    """Inverted Residual Block (MobileNetV2-style, adapted for PalmNet-Lite).

    Structure:
        Input
        -> PW Expansion Conv 1x1 -> BN -> PReLU
        -> DWConv 3x3            -> BN -> PReLU
        -> PW Linear Proj 1x1   -> BN
        -> (+ residual if stride==1 and in_channels==out_channels)

    Tidak ada activation setelah linear projection (linear bottleneck).
    Tidak ada projection shortcut untuk channel/stride mismatch.

    Args:
        in_channels:  input channel count
        out_channels: output channel count
        stride:       stride untuk depthwise conv (1 atau 2)
        expansion:    expansion ratio t; hidden = in_channels * t
        prelu_init:   initial slope PReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 2,
        prelu_init: float = 0.25,
    ):
        super().__init__()
        hidden = in_channels * expansion
        self.use_residual = (stride == 1 and in_channels == out_channels)

        # Pointwise Expansion
        self.pw_expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden, eps=1e-5, momentum=0.1),
            nn.PReLU(num_parameters=hidden, init=prelu_init),
        )

        # Depthwise Conv
        self.dw = nn.Sequential(
            nn.Conv2d(
                hidden, hidden,
                kernel_size=3, stride=stride, padding=1,
                groups=hidden, bias=False,
            ),
            nn.BatchNorm2d(hidden, eps=1e-5, momentum=0.1),
            nn.PReLU(num_parameters=hidden, init=prelu_init),
        )

        # Pointwise Linear Projection — NO activation
        self.pw_proj = nn.Sequential(
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pw_expand(x)
        out = self.dw(out)
        out = self.pw_proj(out)
        if self.use_residual:
            out = out + x
        return out


class GlobalDepthwiseConv(nn.Module):
    """Global Depthwise Convolution (GDConv).

    Menggantikan Global Average Pooling.
    Kernel size = input spatial size (7x7 setelah stage 4).
    Menggunakan learnable spatial weights per channel.
    Tidak ada activation setelah GDConv (linear aggregation).

    Input:  [B, C, H, H] di mana H == kernel_size
    Output: [B, C, 1, 1]
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.gdconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            groups=channels,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)
        # No activation after GDConv per spec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.gdconv(x))

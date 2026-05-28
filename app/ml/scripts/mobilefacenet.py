"""
MobileFaceNet architecture untuk palm biometric.

Referensi:
- Chen et al. 2018, "MobileFaceNets: Efficient CNNs for Accurate Real-time
  Face Verification on Mobile Devices"

Arsitektur:
- Input: 3 × 112 × 112
- Output: 128-dim embedding (L2-normalized di inference time)
- Backbone: Depthwise separable convolutions, mirip MobileNetV2 tapi
  dengan modifikasi khusus untuk face/palm recognition:
  - Tidak ada global average pooling (pakai GDC: Global Depthwise Conv)
  - Embedding dihasilkan via 1×1 conv + flatten

Implementasi di sini meniru struktur paper original. Kalau Anda punya
checkpoint pretrained (mis. dari face.evoLVe), state_dict bisa di-load
selama nama layer cocok.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EMBEDDING_DIM, INPUT_CHANNELS


class ConvBlock(nn.Module):
    """Conv2d + BN + PReLU (atau Linear di output)."""
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1, groups=1, linear=False):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.Identity() if linear else nn.PReLU(out_c)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparable(nn.Module):
    """Depthwise conv 3×3 + pointwise conv 1×1."""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.dw = ConvBlock(in_c, in_c, kernel=3, stride=stride, padding=1, groups=in_c)
        self.pw = ConvBlock(in_c, out_c, kernel=1, stride=1, padding=0, linear=True)

    def forward(self, x):
        return self.pw(self.dw(x))


class Bottleneck(nn.Module):
    """
    Inverted residual bottleneck (mirip MobileNetV2):
    1×1 expand → 3×3 depthwise → 1×1 project (linear).
    Residual hanya kalau stride=1 dan in_c == out_c.
    """
    def __init__(self, in_c, out_c, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden = in_c * expand_ratio
        self.use_residual = (stride == 1 and in_c == out_c)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBlock(in_c, hidden, kernel=1, padding=0))
        # depthwise
        layers.append(ConvBlock(hidden, hidden, kernel=3, stride=stride, padding=1, groups=hidden))
        # pointwise linear
        layers.append(ConvBlock(hidden, out_c, kernel=1, padding=0, linear=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class MobileFaceNet(nn.Module):
    """
    Backbone-only MobileFaceNet untuk embedding extraction.

    Output: 128-dim feature vector (BELUM dinormalisasi — normalisasi
    dilakukan di loss function saat training, dan di inference wrapper
    saat export).

    Untuk training Phase 1 (softmax), tambahkan classifier head terpisah.
    Untuk training Phase 2 (ArcFace), pakai ArcFaceLoss yang sudah
    handle normalisasi.
    """

    # Konfigurasi bottleneck: (expand_ratio, out_channels, num_blocks, stride)
    BOTTLENECK_CFG = [
        (2, 64, 5, 2),
        (4, 128, 1, 2),
        (2, 128, 6, 1),
        (4, 128, 1, 2),
        (2, 128, 2, 1),
    ]

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, input_channels: int = INPUT_CHANNELS):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Stem: 3×112×112 → 64×56×56
        self.stem = ConvBlock(input_channels, 64, kernel=3, stride=2, padding=1)

        # Depthwise stem
        self.dw_stem = ConvBlock(64, 64, kernel=3, stride=1, padding=1, groups=64)

        # Bottleneck stages
        self.bottlenecks = self._make_bottlenecks(in_c=64)

        # Final conv: 128 → 512 channels
        self.conv_last = ConvBlock(128, 512, kernel=1, padding=0)

        # Global Depthwise Conv (GDC): output spatial dimensions ke 1×1
        # Input feature map size setelah bottleneck: 7×7
        self.gdc = ConvBlock(512, 512, kernel=7, stride=1, padding=0, groups=512, linear=True)

        # Embedding layer: 512 → embedding_dim
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

    def _make_bottlenecks(self, in_c: int):
        layers = []
        for expand, out_c, num_blocks, stride in self.BOTTLENECK_CFG:
            for i in range(num_blocks):
                s = stride if i == 0 else 1
                layers.append(Bottleneck(in_c, out_c, s, expand))
                in_c = out_c
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 112, 112) RGB, normalized dengan mean=0.5, std=0.5

        Returns:
            (B, embedding_dim) — RAW embedding (belum L2-normalized).
            Normalisasi dilakukan di loss/inference wrapper.
        """
        x = self.stem(x)
        x = self.dw_stem(x)
        x = self.bottlenecks(x)
        x = self.conv_last(x)
        x = self.gdc(x)
        x = self.embedding(x)
        return x


def load_mobilefacenet(checkpoint_path: str | None = None, strict: bool = False) -> MobileFaceNet:
    """
    Helper untuk load MobileFaceNet, optional dengan checkpoint.

    Args:
        checkpoint_path: path ke .pth file. Kalau None, return random-init.
        strict: kalau True, semua key di state_dict harus cocok. False
                untuk loading pretrained yang punya classifier head
                dengan dimensi berbeda (mis. MS-Celeb-1M).
    """
    model = MobileFaceNet()
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        # Support beberapa format checkpoint
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if missing:
            print(f"[load_mobilefacenet] Missing keys: {len(missing)} (first 5: {missing[:5]})")
        if unexpected:
            print(f"[load_mobilefacenet] Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
    return model


if __name__ == "__main__":
    # Smoke test arsitektur
    model = MobileFaceNet()
    model.eval()
    dummy = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        out = model(dummy)
    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Output dtype: {out.dtype}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Model size (MB):  {total_params * 4 / 1024 / 1024:.2f}")  # FP32

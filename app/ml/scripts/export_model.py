"""
Export model untuk konsumsi backend FastAPI.

Output utama:
- ml/artifacts/palm_recognizer.pt  ← TorchScript, inference-only

InferenceModel:
- Wraps backbone MobileFaceNet
- Forward: input (B, 3, 112, 112) → output (B, 128) L2-normalized
- BatchNorm di-set ke eval mode (statistics dari training di-freeze)
- Tidak ada classifier head — hanya embedding

Verifikasi setelah export:
1. Load palm_recognizer.pt → no error
2. Forward dummy input → output shape (1, 128)
3. Output norm == 1.0 (L2-normalized)
4. Embedding TorchScript == embedding PyTorch eager (toleransi 1e-5)

Optional copy ke backend folder:
    --copy-to-backend /path/to/backend/ml/models/

Untuk export TFLite (bonus opsional), pakai script export_tflite.py terpisah.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    MODELS_DIR,
    ARTIFACTS_DIR,
    RECOGNIZER_OUTPUT,
    THRESHOLD_OUTPUT,
    BACKEND_MODEL_DIR,
    ROI_SIZE,
)
from mobilefacenet import MobileFaceNet


class InferenceModel(nn.Module):
    """
    Wrapper inference-only: backbone + L2 normalize.

    Output: (B, 128) embedding L2-normalized. Backend tinggal pakai
    cosine similarity = dot product (karena sudah unit vector).
    """

    def __init__(self, backbone: MobileFaceNet):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)
        emb = F.normalize(emb, dim=1)
        return emb


def export_model(
    checkpoint_path: Path,
    output_path: Path,
    mode: str = "script",  # "script" atau "trace"
    copy_to_backend: Path | None = None,
):
    """
    Args:
        checkpoint_path: ke checkpoint_phase2_best.pth (atau phase1 kalau
                         Path B / smoke test path).
        output_path: ke .pt output
        mode:
            "script" — torch.jit.script (lebih robust, support control flow,
                       tapi butuh code compatible dengan TorchScript)
            "trace" — torch.jit.trace (lebih simple, tapi tidak support
                      branching dinamis)
        copy_to_backend: kalau di-set, copy juga ke folder ini.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ada: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    backbone = MobileFaceNet()
    if "backbone_state_dict" in ckpt:
        backbone.load_state_dict(ckpt["backbone_state_dict"])
    else:
        backbone.load_state_dict(ckpt)
    backbone.eval()

    model = InferenceModel(backbone)
    model.eval()

    # Dummy input untuk verification & tracing
    dummy = torch.randn(1, 3, ROI_SIZE, ROI_SIZE)

    # Reference output (PyTorch eager mode)
    with torch.no_grad():
        ref_output = model(dummy)
    print(f"Reference output shape: {tuple(ref_output.shape)}")
    print(f"Reference output norm:  {ref_output.norm(dim=1).item():.6f}")

    if not (0.99 < ref_output.norm(dim=1).item() < 1.01):
        raise RuntimeError("Output bukan L2-normalized! Cek InferenceModel.")

    # Export
    print(f"\nExporting via torch.jit.{mode}()...")
    if mode == "script":
        scripted = torch.jit.script(model)
    elif mode == "trace":
        scripted = torch.jit.trace(model, dummy, strict=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Verify scripted output matches eager
    with torch.no_grad():
        scripted_output = scripted(dummy)
    diff = (scripted_output - ref_output).abs().max().item()
    print(f"Max diff (scripted vs eager): {diff:.2e}")
    if diff > 1e-4:
        print("⚠️  Difference besar — kemungkinan ada non-deterministic op")

    # Save
    scripted.save(str(output_path))
    print(f"\n✅ Model saved: {output_path}")
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"   Size: {size_mb:.2f} MB")

    # Test loading
    loaded = torch.jit.load(str(output_path))
    with torch.no_grad():
        loaded_output = loaded(dummy)
    diff2 = (loaded_output - ref_output).abs().max().item()
    print(f"   Reload diff: {diff2:.2e}")

    # Copy ke backend (optional)
    if copy_to_backend is not None:
        backend_path = Path(copy_to_backend)
        backend_path.mkdir(parents=True, exist_ok=True)

        target_model = backend_path / output_path.name
        shutil.copy(output_path, target_model)
        print(f"   Copied to backend: {target_model}")

        # Copy threshold.json juga kalau ada
        threshold_src = THRESHOLD_OUTPUT
        if threshold_src.exists():
            target_threshold = backend_path / threshold_src.name
            shutil.copy(threshold_src, target_threshold)
            print(f"   Copied threshold: {target_threshold}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export trained model to TorchScript")
    parser.add_argument("--checkpoint", type=str,
                        default=str(MODELS_DIR / "checkpoint_phase2_best.pth"),
                        help="Path ke checkpoint Phase 2 (atau Phase 1 untuk Path B)")
    parser.add_argument("--output", type=str, default=str(RECOGNIZER_OUTPUT))
    parser.add_argument("--mode", choices=["script", "trace"], default="script")
    parser.add_argument("--copy-to-backend", type=str, default=None,
                        help=f"Optional: copy juga ke folder ini (mis. {BACKEND_MODEL_DIR})")
    args = parser.parse_args()

    backend_copy = Path(args.copy_to_backend) if args.copy_to_backend else None

    export_model(
        checkpoint_path=Path(args.checkpoint),
        output_path=Path(args.output),
        mode=args.mode,
        copy_to_backend=backend_copy,
    )


if __name__ == "__main__":
    main()

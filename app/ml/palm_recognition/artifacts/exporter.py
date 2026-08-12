"""
TorchScript Exporter untuk PalmNet-Lite Artifact.

Export flow:
    1. Load backbone dari best_phase2.pth (backbone_state_dict only)
    2. Buat PalmNetLiteInferenceWrapper (backbone + L2 normalize)
    3. model.eval()
    4. torch.jit.script()
    5. Verifikasi output: shape, finite, L2 norm ~1
    6. Simpan model.pt, manifest.json, threshold.json, metrics.json

Output artifact:
    backend/ml/models/palmnet-lite-scratch/1.0.0/
        model.pt
        manifest.json
        threshold.json
        metrics.json
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def export_palmnet_lite(
    backbone_class,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    version: str = "1.0.0",
    threshold_data: Optional[dict] = None,
    metrics_data: Optional[dict] = None,
    param_count: Optional[int] = None,
) -> Path:
    """Export PalmNet-Lite backbone ke TorchScript artifact.

    Args:
        backbone_class:   PalmNetLite class
        checkpoint_path:  path ke checkpoint_phase2_best.pth
        output_dir:       target artifact directory
        version:          artifact version string
        threshold_data:   dict untuk threshold.json (dari calibration)
        metrics_data:     dict untuk metrics.json (dari evaluation)
        param_count:      trainable parameter count (jika sudah diketahui)

    Returns:
        Path ke model.pt
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load backbone ─────────────────────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_key = "backbone_state_dict" if "backbone_state_dict" in ckpt else "model_state_dict"
    state_dict = ckpt.get(state_key, ckpt)

    backbone = backbone_class()
    backbone.load_state_dict(state_dict, strict=True)
    backbone.eval()

    # ── Inference wrapper ─────────────────────────────────────────────────────
    class _InferenceWrapper(nn.Module):
        def __init__(self, backbone: nn.Module):
            super().__init__()
            self.backbone = backbone

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            emb = self.backbone(x)
            return F.normalize(emb, p=2, dim=1)

    wrapper = _InferenceWrapper(backbone)
    wrapper.eval()

    # ── TorchScript export ────────────────────────────────────────────────────
    dummy = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        eager_out = wrapper(dummy)

    try:
        scripted = torch.jit.script(wrapper)
    except Exception as e:
        print(f"[Export] torch.jit.script gagal: {e}, fallback ke trace")
        scripted = torch.jit.trace(wrapper, dummy)

    # ── Verifikasi output ─────────────────────────────────────────────────────
    scripted.eval()
    with torch.no_grad():
        scripted_out = scripted(dummy)

    assert scripted_out.shape == (1, 128), f"Shape salah: {scripted_out.shape}"
    assert torch.isfinite(scripted_out).all(), "Output mengandung NaN/Inf"

    l2_norm = scripted_out.norm(p=2, dim=1)
    assert torch.allclose(l2_norm, torch.ones(1), atol=1e-3), \
        f"L2 norm bukan ~1: {l2_norm.item()}"

    max_diff = (eager_out - scripted_out).abs().max().item()
    print(f"[Export] eager vs scripted max diff: {max_diff:.6f}")
    if max_diff > 1e-3:
        print(f"  WARNING: max diff {max_diff:.6f} > 1e-3")

    # ── Simpan model.pt ───────────────────────────────────────────────────────
    model_pt_path = output_dir / "model.pt"
    scripted.save(str(model_pt_path))
    model_size_mb = model_pt_path.stat().st_size / 1024 / 1024
    print(f"[Export] Saved: {model_pt_path} ({model_size_mb:.2f} MB)")

    # ── manifest.json ─────────────────────────────────────────────────────────
    manifest = {
        "model_id": "palmnet-lite-scratch",
        "name": "PalmNet-Lite (Scratch)",
        "version": version,
        "architecture": "PalmNetLite",
        "architecture_version": "v1",
        "training_mode": "scratch",
        "source": "project_training",
        "input_shape": [3, 112, 112],
        "output_dim": 128,
        "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "output_normalized": True,
        "parameter_count": param_count,
        "model_size_mb": round(model_size_mb, 3),
        "exported_at": datetime.now().isoformat(),
        "checkpoint_source": str(checkpoint_path),
        "export_verification": {
            "output_shape": list(scripted_out.shape),
            "is_finite": True,
            "l2_norm": round(float(l2_norm.item()), 6),
            "eager_vs_scripted_max_diff": round(max_diff, 8),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)

    # ── threshold.json ────────────────────────────────────────────────────────
    if threshold_data:
        _write_json(output_dir / "threshold.json", threshold_data)
    else:
        default_threshold = {
            "model_id": "palmnet-lite-scratch",
            "version": version,
            "metric": "cosine_similarity",
            "threshold": 0.50,
            "calibration_split": "not_calibrated",
            "selection": "default",
            "note": "Default threshold — run calibration untuk threshold aktual",
        }
        _write_json(output_dir / "threshold.json", default_threshold)

    # ── metrics.json ──────────────────────────────────────────────────────────
    if metrics_data:
        clean = {k: v for k, v in metrics_data.items() if not k.startswith("_")}
        _write_json(output_dir / "metrics.json", clean)

    print(f"[Export] Artifact selesai di: {output_dir}")
    return model_pt_path


def verify_artifact(artifact_dir: str | Path) -> dict:
    """Verifikasi artifact yang sudah diexport.

    Load model.pt dan jalankan forward pass.

    Returns:
        dict hasil verifikasi
    """
    artifact_dir = Path(artifact_dir)
    model_pt = artifact_dir / "model.pt"
    manifest_path = artifact_dir / "manifest.json"

    if not model_pt.exists():
        return {"ok": False, "error": f"model.pt tidak ditemukan di {artifact_dir}"}

    try:
        model = torch.jit.load(str(model_pt), map_location="cpu")
        model.eval()

        dummy = torch.randn(1, 3, 112, 112)
        with torch.no_grad():
            out = model(dummy)

        assert out.shape == (1, 128)
        assert torch.isfinite(out).all()
        l2 = out.norm(p=2, dim=1).item()
        assert abs(l2 - 1.0) < 1e-2, f"L2 norm {l2:.4f}"

        manifest = {}
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)

        return {
            "ok": True,
            "output_shape": list(out.shape),
            "l2_norm": round(l2, 6),
            "is_finite": True,
            "model_id": manifest.get("model_id"),
            "version": manifest.get("version"),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

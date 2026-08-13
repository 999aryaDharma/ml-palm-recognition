"""
TorchScript Exporter for PalmNet-Lite Artifact.

Exports backbone to TorchScript model.pt, creates manifest.json, threshold.json, metrics.json.
Enforces fail-closed rules when exporting for backend deployment.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from palm_recognition.paths import resolve_ml_path
from palm_recognition.models.inference import PalmNetLiteInferenceWrapper


def export_palmnet_lite(
    backbone_class,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    version: str = "1.0.0",
    threshold_data: Optional[dict] = None,
    metrics_data: Optional[dict] = None,
    param_count: Optional[int] = None,
    deploy_backend: bool = False,
) -> Path:
    """Export PalmNet-Lite backbone to TorchScript artifact.

    Args:
        backbone_class: PalmNetLite factory function
        checkpoint_path: path to checkpoint_phase2_best.pth
        output_dir: target output directory
        version: artifact version string
        threshold_data: dictionary from validation threshold calibration
        metrics_data: dictionary from holdout evaluation
        param_count: trainable parameter count
        deploy_backend: if True, strict fail-closed checks apply (requires validation threshold)

    Returns:
        Path to generated model.pt
    """
    checkpoint_path = resolve_ml_path(checkpoint_path)
    output_dir = resolve_ml_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Export FAIL: Checkpoint not found at {checkpoint_path}")

    # Strict Fail-Closed Check for Backend Deployment
    if deploy_backend:
        if not threshold_data:
            raise ValueError("Export FAIL (deploy_backend): threshold_data is required for backend deployment. Run validation calibration first.")
        if threshold_data.get("calibration_split") != "validation":
            raise ValueError(f"Export FAIL (deploy_backend): threshold must be calibrated on 'validation' split, got '{threshold_data.get('calibration_split')}'")
        if not metrics_data:
            raise ValueError("Export FAIL (deploy_backend): metrics_data is required for backend deployment.")

    # 1. Load backbone state_dict
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_key = "backbone_state_dict" if "backbone_state_dict" in ckpt else "model_state_dict"
    state_dict = ckpt.get(state_key, ckpt)

    backbone = backbone_class()
    backbone.load_state_dict(state_dict, strict=True)
    backbone.eval()

    # 2. Canonical inference wrapper
    wrapper = PalmNetLiteInferenceWrapper(backbone)
    wrapper.eval()

    # 3. TorchScript export
    dummy = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        eager_out = wrapper(dummy)

    try:
        scripted = torch.jit.script(wrapper)
    except Exception as e:
        print(f"[Export] torch.jit.script failed: {e}, falling back to torch.jit.trace")
        scripted = torch.jit.trace(wrapper, dummy)

    # 4. Verify output
    scripted.eval()
    with torch.no_grad():
        scripted_out = scripted(dummy)

    assert scripted_out.shape == (1, 128), f"Output shape mismatch: {scripted_out.shape}"
    assert torch.isfinite(scripted_out).all(), "Output contains NaN or Inf"
    l2_norm = scripted_out.norm(p=2, dim=1).item()
    assert abs(l2_norm - 1.0) < 1e-3, f"Output L2 norm is not ~1: {l2_norm:.6f}"

    max_diff = (eager_out - scripted_out).abs().max().item()

    # 5. Write model.pt
    model_pt_path = output_dir / "model.pt"
    scripted.save(str(model_pt_path))
    model_size_mb = model_pt_path.stat().st_size / (1024 * 1024)

    # 6. Write manifest.json
    manifest = {
        "model_id": "palmnet-lite-scratch",
        "name": "PalmNet-Lite (Scratch)",
        "version": version,
        "architecture": "PalmNetLite",
        "architecture_version": "v1",
        "training_mode": "scratch",
        "deployable": deploy_backend,
        "input_shape": [3, 112, 112],
        "output_dim": 128,
        "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "output_normalized": True,
        "parameter_count": param_count,
        "model_size_mb": round(model_size_mb, 3),
        "exported_at": datetime.now().isoformat(),
        "export_verification": {
            "output_shape": list(scripted_out.shape),
            "is_finite": True,
            "l2_norm": round(l2_norm, 6),
            "eager_vs_scripted_max_diff": round(max_diff, 8),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)

    # 7. Write threshold.json
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
            "note": "Dev artifact — uncalibrated threshold.",
        }
        _write_json(output_dir / "threshold.json", default_threshold)

    # 8. Write metrics.json
    if metrics_data:
        clean_metrics = {k: v for k, v in metrics_data.items() if not k.startswith("_")}
        _write_json(output_dir / "metrics.json", clean_metrics)

    print(f"[Export] Artifact successfully exported to: {output_dir}")
    return model_pt_path


def verify_artifact(artifact_dir: str | Path) -> dict:
    """Verify an exported artifact directory."""
    artifact_dir = resolve_ml_path(artifact_dir)
    model_pt = artifact_dir / "model.pt"
    manifest_path = artifact_dir / "manifest.json"

    if not model_pt.exists():
        return {"ok": False, "error": f"model.pt not found in {artifact_dir}"}

    try:
        model = torch.jit.load(str(model_pt), map_location="cpu")
        model.eval()

        dummy = torch.randn(1, 3, 112, 112)
        with torch.no_grad():
            out = model(dummy)

        assert out.shape == (1, 128)
        assert torch.isfinite(out).all()
        l2 = out.norm(p=2, dim=1).item()
        assert abs(l2 - 1.0) < 1e-2

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

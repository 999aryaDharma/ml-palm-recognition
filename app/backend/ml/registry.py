"""
Model Registry — Architecture-agnostic runtime model discovery and execution.

Features:
1. Discovers local TorchScript artifacts under backend/ml/models/.
2. Validates manifest schema and runs forward smoke test before loading.
3. Implements PIL Image ROI preprocessing according to manifest metadata.
4. Exposes extract_embedding(roi: PIL.Image) -> np.ndarray [128] L2-normalized float32.
5. Semantic version sorting for model directory discovery.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def parse_semver(v_str: str) -> tuple[int, ...]:
    """Parse version string into integer tuple for semantic version sorting.

    e.g. '1.10.0' -> (1, 10, 0)
    """
    try:
        parts = [int(p) for p in v_str.strip().split(".") if p.isdigit()]
        return tuple(parts) if parts else (0, 0, 0)
    except Exception:
        return (0, 0, 0)


def preprocess_roi(roi: Image.Image, manifest: dict | None = None) -> torch.Tensor:
    """Preprocess PIL Image ROI into normalized torch.Tensor [1, 3, H, W].

    Uses manifest input_shape and normalization parameters if provided.
    Defaults: 112x112 RGB, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5].
    """
    if not isinstance(roi, Image.Image):
        raise TypeError(f"Expected PIL Image ROI, got {type(roi)}")

    roi_rgb = roi.convert("RGB")

    target_size = 112
    if manifest and "input_shape" in manifest and isinstance(manifest["input_shape"], list):
        shape = manifest["input_shape"]
        if len(shape) == 3:
            target_size = shape[1]

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    if manifest and "normalization" in manifest and isinstance(manifest["normalization"], dict):
        norm = manifest["normalization"]
        mean = norm.get("mean", mean)
        std = norm.get("std", std)

    if roi_rgb.size != (target_size, target_size):
        roi_rgb = roi_rgb.resize((target_size, target_size), Image.Resampling.BILINEAR)

    img_np = np.array(roi_rgb, dtype=np.float32) / 255.0  # HWC, [0, 1]
    mean_np = np.array(mean, dtype=np.float32)
    std_np = np.array(std, dtype=np.float32)
    img_np = (img_np - mean_np) / std_np

    tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


@dataclass
class ModelRuntime:
    """Runtime instance for a loaded model version."""
    model_id: str
    version: str
    name: str
    training_mode: str
    manifest: dict
    model: object              # TorchScript model
    threshold: float
    threshold_data: dict
    device: torch.device
    metrics: dict = field(default_factory=dict)

    @torch.no_grad()
    def extract_embedding(self, roi: Image.Image) -> Optional[np.ndarray]:
        """Extract L2-normalized 128-D embedding from PIL Image ROI.

        Args:
            roi: PIL Image of palm region

        Returns:
            1D float32 numpy array of shape (128,) or None if extraction fails.
        """
        if roi is None:
            return None
        try:
            tensor = preprocess_roi(roi, self.manifest).to(self.device)
            emb = self.model(tensor)
            emb = F.normalize(emb, p=2, dim=1)
            arr = emb.cpu().numpy()[0].astype(np.float32)
            if arr.shape != (128,) or not np.isfinite(arr).all():
                return None
            return arr
        except Exception as e:
            print(f"[ModelRuntime:{self.model_id}] Error extracting embedding: {e}")
            return None


class ModelRegistry:
    """Discovers, validates, and manages local TorchScript model artifacts."""

    def __init__(self, models_dir: str | Path):
        self.models_dir = Path(models_dir)
        self._registry: dict[str, ModelRuntime] = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def discover(self) -> int:
        """Scan models_dir for valid artifacts.

        Returns:
            Number of valid models successfully loaded.
        """
        self._registry.clear()

        if not self.models_dir.exists():
            print(f"[ModelRegistry] models_dir does not exist: {self.models_dir}")
            return 0

        for model_dir in sorted(self.models_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
            model_id = model_dir.name

            # Sort version directories by semver descending
            version_dirs = [d for d in model_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
            version_dirs.sort(key=lambda d: parse_semver(d.name), reverse=True)

            for version_dir in version_dirs:
                version = version_dir.name
                runtime = self._try_load(model_id, version, version_dir)
                if runtime:
                    self._registry[model_id] = runtime
                    print(f"[ModelRegistry] [OK] Loaded model: '{model_id}' v{version} "
                          f"(threshold={runtime.threshold:.4f})")

                    break  # Use highest valid version

        return len(self._registry)

    def _try_load(
        self,
        model_id: str,
        version: str,
        artifact_dir: Path,
    ) -> Optional[ModelRuntime]:
        """Try loading and validating artifact from directory."""
        model_pt = artifact_dir / "model.pt"
        manifest_path = artifact_dir / "manifest.json"
        threshold_path = artifact_dir / "threshold.json"

        if not model_pt.exists():
            return None

        try:
            # 1. Load manifest if present
            manifest = {}
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)

            # Validate manifest keys consistency
            if manifest.get("model_id") and manifest["model_id"] != model_id:
                print(f"[ModelRegistry] Manifest model_id mismatch: {manifest['model_id']} vs {model_id}")
                return None
            if manifest.get("version") and manifest["version"] != version:
                print(f"[ModelRegistry] Manifest version mismatch: {manifest['version']} vs {version}")
                return None

            # 2. Load TorchScript model
            model = torch.jit.load(str(model_pt), map_location=self._device)
            model.eval()

            # 3. Discovery smoke test: run dummy tensor
            dummy = torch.randn(1, 3, 112, 112).to(self._device)
            with torch.no_grad():
                out = model(dummy)
                out = F.normalize(out, p=2, dim=1)

            if out.shape != (1, 128) or not torch.isfinite(out).all():
                print(f"[ModelRegistry] Smoke test failed for {model_id} v{version}: shape={out.shape}")
                return None

            # 4. Load threshold
            threshold_data = {}
            threshold = 0.50
            if threshold_path.exists():
                with open(threshold_path) as f:
                    threshold_data = json.load(f)
                    threshold = float(threshold_data.get("threshold", 0.50))

            # 5. Load metrics if present
            metrics = {}
            metrics_path = artifact_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)

            name = manifest.get("name", model_id)
            training_mode = manifest.get("training_mode", _infer_training_mode(model_id))

            return ModelRuntime(
                model_id=model_id,
                version=version,
                name=name,
                training_mode=training_mode,
                manifest=manifest,
                model=model,
                threshold=threshold,
                threshold_data=threshold_data,
                device=self._device,
                metrics=metrics,
            )

        except Exception as e:
            print(f"[ModelRegistry] Failed to load {model_id} v{version}: {e}")
            return None

    def get(self, model_id: str) -> ModelRuntime:
        if model_id not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"Model '{model_id}' not found in registry. Available models: {available}"
            )
        return self._registry[model_id]

    def list_available(self) -> list[dict]:
        return [
            {
                "id": runtime.model_id,
                "name": runtime.name,
                "version": runtime.version,
                "training_mode": runtime.training_mode,
                "threshold": runtime.threshold,
            }
            for runtime in self._registry.values()
        ]

    def is_available(self, model_id: str) -> bool:
        return model_id in self._registry

    @property
    def count(self) -> int:
        return len(self._registry)


def _infer_training_mode(model_id: str) -> str:
    if "pretrained" in model_id:
        return "pretrained"
    if "scratch" in model_id:
        return "scratch"
    return "unknown"

"""
Model Registry — Architecture-agnostic runtime model discovery.

Menemukan dan me-load dua artifact:
    mobilefacenet-pretrained  (existing frozen artifact)
    palmnet-lite-scratch      (new trained artifact)

Backend tidak perlu import PalmNetLite atau MobileFaceNet class untuk inference.
Cukup torch.jit.load() TorchScript artifact.

Directory structure:
    backend/ml/models/
    ├── registry.json             (optional override)
    ├── mobilefacenet-pretrained/
    │   └── 1.0.0/
    │       ├── model.pt
    │       ├── manifest.json
    │       └── threshold.json
    └── palmnet-lite-scratch/
        └── 1.0.0/
            ├── model.pt
            ├── manifest.json
            ├── threshold.json
            └── metrics.json
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


@dataclass
class ModelRuntime:
    """Runtime artifact untuk satu model version."""
    model_id: str
    version: str
    name: str
    training_mode: str
    manifest: dict
    model: object              # TorchScript model
    threshold: float
    threshold_data: dict
    metrics: dict = field(default_factory=dict)

    def extract_embedding(self, tensor: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalized embedding dari image tensor.

        Args:
            tensor: [B, 3, 112, 112] float32 normalized

        Returns:
            [B, 128] L2-normalized embedding
        """
        with torch.no_grad():
            emb = self.model(tensor)
        return emb


class ModelRegistry:
    """Discovers dan menyimpan local TorchScript artifacts.

    Usage:
        registry = ModelRegistry("ml/models")
        registry.discover()
        runtime = registry.get("palmnet-lite-scratch")
        emb = runtime.extract_embedding(tensor)
    """

    def __init__(self, models_dir: str | Path):
        self.models_dir = Path(models_dir)
        self._registry: dict[str, ModelRuntime] = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def discover(self) -> int:
        """Scan models_dir untuk valid artifacts.

        Returns:
            jumlah artifact yang berhasil di-load
        """
        self._registry.clear()

        if not self.models_dir.exists():
            print(f"[ModelRegistry] models_dir tidak ada: {self.models_dir}")
            return 0

        for model_dir in sorted(self.models_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
            model_id = model_dir.name

            # Cari version directory (misal 1.0.0/)
            for version_dir in sorted(model_dir.iterdir(), reverse=True):
                if not version_dir.is_dir():
                    continue
                version = version_dir.name
                runtime = self._try_load(model_id, version, version_dir)
                if runtime:
                    self._registry[model_id] = runtime
                    print(f"[ModelRegistry] ✓ Loaded: {model_id} v{version} "
                          f"(threshold={runtime.threshold:.4f})")
                    break  # gunakan version terbaru yang berhasil di-load

        return len(self._registry)

    def _try_load(
        self,
        model_id: str,
        version: str,
        artifact_dir: Path,
    ) -> Optional[ModelRuntime]:
        """Coba load artifact dari directory."""
        model_pt = artifact_dir / "model.pt"
        manifest_path = artifact_dir / "manifest.json"
        threshold_path = artifact_dir / "threshold.json"

        if not model_pt.exists():
            return None

        try:
            # Load TorchScript — backend tidak perlu tahu arsitekturnya
            model = torch.jit.load(str(model_pt), map_location=self._device)
            model.eval()

            # Load manifest
            manifest = {}
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)

            # Load threshold
            threshold_data = {}
            threshold = 0.50  # default
            if threshold_path.exists():
                with open(threshold_path) as f:
                    threshold_data = json.load(f)
                    threshold = float(threshold_data.get("threshold", 0.50))

            # Load metrics (optional)
            metrics = {}
            metrics_path = artifact_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)

            # Name dan training mode dari manifest
            name = manifest.get("name", model_id)
            training_mode = manifest.get("training_mode", "unknown")
            if not name or name == model_id:
                # Fallback naming
                name = _default_name(model_id)
            if not training_mode or training_mode == "unknown":
                training_mode = _infer_training_mode(model_id)

            return ModelRuntime(
                model_id=model_id,
                version=version,
                name=name,
                training_mode=training_mode,
                manifest=manifest,
                model=model,
                threshold=threshold,
                threshold_data=threshold_data,
                metrics=metrics,
            )

        except Exception as e:
            print(f"[ModelRegistry] Gagal load {model_id}: {e}")
            return None

    def get(self, model_id: str) -> ModelRuntime:
        """Get ModelRuntime berdasarkan model_id.

        Raises:
            KeyError jika model_id tidak tersedia
        """
        if model_id not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"Model '{model_id}' tidak tersedia. Available: {available}"
            )
        return self._registry[model_id]

    def list_available(self) -> list[dict]:
        """Return list of available model metadata untuk API response."""
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


def _default_name(model_id: str) -> str:
    """Generate display name dari model_id."""
    name_map = {
        "mobilefacenet-pretrained": "MobileFaceNet (Pretrained)",
        "palmnet-lite-scratch": "PalmNet-Lite (Scratch)",
    }
    return name_map.get(model_id, model_id.replace("-", " ").title())


def _infer_training_mode(model_id: str) -> str:
    """Infer training mode dari model_id."""
    if "pretrained" in model_id:
        return "pretrained"
    if "scratch" in model_id:
        return "scratch"
    return "unknown"

"""
Training Checkpoint utilities untuk PalmNet-Lite.

Menyimpan dan memuat checkpoint dengan schema yang konsisten.
Checkpoint hanya untuk resume/training/debug — bukan artifact runtime.

Schema checkpoint Phase 1:
    run_id, phase, architecture_version, epoch,
    backbone_state_dict, classifier_state_dict,
    optimizer_state_dict, scheduler_state_dict,
    best_metric, config

Schema checkpoint Phase 2:
    run_id, phase, epoch,
    backbone_state_dict, arcface_state_dict,
    optimizer_state_dict, scheduler_state_dict,
    current_margin, validation_metrics, config
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    payload: dict[str, Any],
) -> None:
    """Simpan checkpoint ke disk.

    Args:
        path:    target file path (akan di-mkdir parents)
        payload: dict yang akan disimpan
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load checkpoint dari disk.

    Args:
        path:         file path
        map_location: device target untuk tensor

    Returns:
        dict checkpoint
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {path}")
    return torch.load(path, map_location=map_location)


def save_phase1_checkpoint(
    checkpoint_dir: str | Path,
    run_id: str,
    epoch: int,
    backbone: torch.nn.Module,
    classifier: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_metric: float,
    config: dict,
    is_best: bool = False,
) -> None:
    """Simpan checkpoint Phase 1.

    Menyimpan last.pth selalu.
    Menyimpan best.pth jika is_best=True.
    """
    payload = {
        "run_id": run_id,
        "phase": "phase1",
        "architecture_version": "v1",
        "epoch": epoch,
        "backbone_state_dict": backbone.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_metric": best_metric,
        "config": config,
    }

    ckpt_dir = Path(checkpoint_dir)
    save_checkpoint(ckpt_dir / "checkpoint_phase1_last.pth", payload)
    if is_best:
        save_checkpoint(ckpt_dir / "checkpoint_phase1_best.pth", payload)


def save_phase2_checkpoint(
    checkpoint_dir: str | Path,
    run_id: str,
    epoch: int,
    backbone: torch.nn.Module,
    arcface_head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    current_margin: float,
    validation_metrics: dict,
    config: dict,
    is_best: bool = False,
) -> None:
    """Simpan checkpoint Phase 2."""
    payload = {
        "run_id": run_id,
        "phase": "phase2",
        "architecture_version": "v1",
        "epoch": epoch,
        "backbone_state_dict": backbone.state_dict(),
        "arcface_state_dict": arcface_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "current_margin": current_margin,
        "validation_metrics": validation_metrics,
        "config": config,
    }

    ckpt_dir = Path(checkpoint_dir)
    save_checkpoint(ckpt_dir / "checkpoint_phase2_last.pth", payload)
    if is_best:
        save_checkpoint(ckpt_dir / "checkpoint_phase2_best.pth", payload)

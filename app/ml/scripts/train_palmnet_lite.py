"""
Main training script untuk PalmNet-Lite Scratch.

Usage:
    cd app/ml
    python -m scripts.train_palmnet_lite \\
        --config configs/palmnet_lite_scratch.yaml \\
        [--phase 1|2|all] \\
        [--phase1-checkpoint path/to/checkpoint_phase1_best.pth]

Pipeline:
    Phase 0: Dataset sanity check
    Phase 1: Softmax representation learning (scratch)
    Phase 2: ArcFace metric learning (dari Phase 1 checkpoint)

Semua hasil disimpan ke:
    app/ml/artifacts/trained_logs/palmnet-lite-scratch/<run_id>/
    app/ml/checkpoints/palmnet-lite-scratch/
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Path setup — bisa dijalankan dari app/ml/ atau root
_SCRIPT_DIR = Path(__file__).resolve().parent
_ML_DIR = _SCRIPT_DIR.parent
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))


def set_seed(seed: int) -> None:
    """Set seed untuk reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_phase0_sanity(config: dict, device: torch.device) -> dict:
    """Phase 0: Validasi dataset dan pipeline sanity check."""
    from palm_recognition.data.dataset import PalmDataset, get_train_transform, get_val_transform, build_dataloaders
    from palm_recognition.models.palmnet_lite import PalmNetLite
    from palm_recognition.models.initialization import initialize_scratch_weights

    ds_cfg = config.get("dataset", {})
    train_csv = ds_cfg.get("train_csv", "app/ml/data/splits/train.csv")
    val_csv   = ds_cfg.get("val_csv",   "app/ml/data/splits/val.csv")
    test_csv  = ds_cfg.get("test_csv",  "app/ml/data/splits/test.csv")

    print("\n=== PHASE 0: Dataset Sanity Check ===")

    # Check file existence
    for csv_path, name in [(train_csv, "train"), (val_csv, "val"), (test_csv, "test")]:
        if not Path(csv_path).exists():
            print(f"  ERROR: {name}.csv tidak ditemukan: {csv_path}")
            sys.exit(1)

    train_ds = PalmDataset(train_csv, transform=get_train_transform())
    val_ds   = PalmDataset(val_csv,   transform=get_val_transform())

    import pandas as pd
    test_df = pd.read_csv(test_csv)

    # Identity isolation check
    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)

    train_palms = set(train_df["palm_id"].astype(str).unique()) if "palm_id" in train_df.columns else set()
    val_palms   = set(val_df["palm_id"].astype(str).unique()) if "palm_id" in val_df.columns else set()
    test_palms  = set(test_df["palm_id"].astype(str).unique()) if "palm_id" in test_df.columns else set()

    overlap = train_palms & test_palms
    if overlap:
        print(f"  WARNING: {len(overlap)} palm ID overlap antara train dan test!")

    summary = {
        "train_images": len(train_ds),
        "val_images": len(val_ds),
        "test_images": len(test_df),
        "train_classes": train_ds.num_classes,
        "val_classes": val_ds.num_classes,
        "train_identities": len(train_palms),
        "val_identities": len(val_palms),
        "test_identities": len(test_palms),
        "train_test_identity_overlap": len(overlap),
    }
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Smoke batch
    batch_size = config.get("phase1", {}).get("batch_size", 64)
    loader_tr, loader_val, num_classes = build_dataloaders(
        train_csv, val_csv, batch_size=min(batch_size, 8), num_workers=0
    )
    images, labels = next(iter(loader_tr))
    assert images.shape[1:] == (3, 112, 112), f"Shape salah: {images.shape}"
    assert images.dtype == torch.float32
    assert labels.max() < num_classes

    # Forward smoke test
    backbone = PalmNetLite()
    initialize_scratch_weights(backbone)
    backbone.eval()
    with torch.no_grad():
        out = backbone(images.to(device))
    assert out.shape == (images.shape[0], 128)
    assert torch.isfinite(out).all(), "Forward output mengandung NaN/Inf"

    # Backward smoke test
    backbone.train()
    backbone.to(device)
    classifier = torch.nn.Linear(128, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(classifier.parameters()), lr=1e-3
    )
    optimizer.zero_grad()
    emb = backbone(images.to(device))
    logits = classifier(emb)
    loss = torch.nn.functional.cross_entropy(logits, labels.to(device))
    loss.backward()
    grads_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in list(backbone.parameters()) + list(classifier.parameters())
        if p.requires_grad
    )

    summary["smoke_forward_ok"] = True
    summary["smoke_backward_ok"] = grads_ok
    summary["num_train_classes"] = num_classes
    print(f"  Smoke forward: OK")
    print(f"  Smoke backward: {'OK' if grads_ok else 'FAIL'}")
    print(f"  Num classes (train): {num_classes}")
    print("Phase 0 PASSED\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train PalmNet-Lite from scratch")
    parser.add_argument("--config", type=str,
                        default="configs/palmnet_lite_scratch.yaml")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["0", "1", "2", "all"],
                        help="Phase yang dijalankan")
    parser.add_argument("--phase1-checkpoint", type=str, default=None,
                        help="Path ke checkpoint Phase 1 untuk skip Phase 1 dan langsung ke Phase 2")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("run", {}).get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    from palm_recognition.models.palmnet_lite import PalmNetLite, build_palmnet_lite
    from palm_recognition.models.initialization import initialize_scratch_weights
    from palm_recognition.training.logging import RunLogger
    from palm_recognition.data.dataset import build_dataloaders

    ds_cfg = config.get("dataset", {})
    train_csv = ds_cfg.get("train_csv", "app/ml/data/splits/train.csv")
    val_csv   = ds_cfg.get("val_csv",   "app/ml/data/splits/val.csv")

    paths_cfg = config.get("paths", {})
    checkpoint_dir  = paths_cfg.get("checkpoint_dir",  "app/ml/checkpoints/palmnet-lite-scratch")
    trained_logs    = paths_cfg.get("trained_logs_dir", "app/ml/artifacts/trained_logs/palmnet-lite-scratch")

    # ── Phase 0 ───────────────────────────────────────────────────────────────
    if args.phase in ("0", "all"):
        phase0_summary = run_phase0_sanity(config, device)

    # ── Build DataLoaders ─────────────────────────────────────────────────────
    p1_cfg = config.get("phase1", {})
    batch_size = p1_cfg.get("batch_size", 64)
    train_loader, val_loader, num_classes = build_dataloaders(
        train_csv, val_csv, batch_size=batch_size, num_workers=0
    )
    print(f"DataLoaders: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, classes={num_classes}")

    # ── Initialize RunLogger ──────────────────────────────────────────────────
    run_logger = RunLogger(base_dir=trained_logs, seed=seed)

    # Build run.json awal
    import torch
    run_metadata = {
        "seed": seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "num_train_classes": num_classes,
        "train_images": len(train_loader.dataset),
        "val_images": len(val_loader.dataset),
    }
    if device.type == "cuda":
        run_metadata["gpu"] = torch.cuda.get_device_name(0)

    # Build model dan hitung param count
    model_cfg = config.get("model", {})
    backbone = build_palmnet_lite(cfg=model_cfg)
    param_info = backbone.count_parameters()
    trainable_params = param_info["trainable"]
    print(f"\nPalmNet-Lite v1 parameter count: {trainable_params:,}")

    # Sanity range check
    assert 350_000 <= trainable_params <= 450_000, (
        f"Parameter count {trainable_params:,} di luar sanity range [350k, 450k]. "
        "Cek architecture implementation."
    )
    print(f"  [OK] Parameter count dalam sanity range [350k, 450k]")

    run_metadata["parameter_count"] = trainable_params
    run_logger.init_run(config=config, extra=run_metadata)
    run_logger.add_note(f"Training dimulai. Device: {device}, seed: {seed}")

    print(f"\nRun ID: {run_logger.run_id}")
    print(f"Run dir: {run_logger.run_dir}")

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    phase1_summary = None
    if args.phase in ("1", "all") and args.phase1_checkpoint is None:
        initialize_scratch_weights(backbone, prelu_init=model_cfg.get("prelu_init", 0.25))
        from palm_recognition.training.softmax_stage import run_phase1
        try:
            phase1_summary = run_phase1(
                backbone=backbone,
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=num_classes,
                config=p1_cfg,
                checkpoint_dir=checkpoint_dir,
                run_logger=run_logger,
                device=device,
                run_id=run_logger.run_id,
            )
            run_logger.update_run_json({"phase1_summary": phase1_summary})
        except Exception as e:
            run_logger.add_note(f"Phase 1 FAILED: {e}")
            run_logger.close(status="failed", failure_reason=str(e))
            raise

        # Phase 1 gate check
        best_val_acc = phase1_summary.get("best_val_accuracy", 0)
        if best_val_acc < 0.5:
            run_logger.add_note(
                f"WARN: Phase 1 val_accuracy={best_val_acc:.4f} < 0.5 — pipeline mungkin bermasalah"
            )
            print(f"\n⚠️  Phase 1 val_accuracy={best_val_acc:.4f} < 0.5 — cek pipeline sebelum Phase 2")
        elif best_val_acc < 0.7:
            run_logger.add_note(
                f"INFO: Phase 1 val_accuracy={best_val_acc:.4f} antara 0.5-0.7 — cukup untuk lanjut"
            )

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    phase2_summary = None
    if args.phase in ("2", "all"):
        # Load backbone dari Phase 1 checkpoint (own, bukan external)
        phase1_ckpt_path = args.phase1_checkpoint or str(
            Path(checkpoint_dir) / "checkpoint_phase1_best.pth"
        )
        if not Path(phase1_ckpt_path).exists():
            print(f"ERROR: Phase 1 checkpoint tidak ditemukan: {phase1_ckpt_path}")
            run_logger.close(status="failed", failure_reason="Phase 1 checkpoint tidak ditemukan")
            sys.exit(1)

        print(f"\nLoading Phase 1 backbone dari: {phase1_ckpt_path}")
        ckpt = torch.load(phase1_ckpt_path, map_location=device)
        backbone2 = build_palmnet_lite(cfg=model_cfg)
        backbone2.load_state_dict(ckpt["backbone_state_dict"])
        run_logger.add_note(f"Phase 2 dimulai dari Phase 1 checkpoint: {phase1_ckpt_path}")

        p2_cfg = config.get("phase2", {})
        from palm_recognition.training.arcface_stage import run_phase2
        try:
            phase2_summary = run_phase2(
                backbone=backbone2,
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=num_classes,
                config=p2_cfg,
                checkpoint_dir=checkpoint_dir,
                run_logger=run_logger,
                device=device,
                run_id=run_logger.run_id,
            )
            run_logger.update_run_json({"phase2_summary": phase2_summary})
        except Exception as e:
            run_logger.add_note(f"Phase 2 FAILED: {e}")
            run_logger.close(status="failed", failure_reason=str(e))
            raise

    run_logger.close(status="completed")
    print(f"\n✓ Training selesai. Run ID: {run_logger.run_id}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Logs: {run_logger.run_dir}")

    if phase1_summary:
        print(f"\n  Phase 1 best val_accuracy: {phase1_summary.get('best_val_accuracy', 'N/A'):.4f} "
              f"@ epoch {phase1_summary.get('best_epoch', 'N/A')}")
    if phase2_summary:
        print(f"  Phase 2 best cosine_gap:  {phase2_summary.get('best_cosine_gap', 'N/A'):.4f} "
              f"@ epoch {phase2_summary.get('best_epoch', 'N/A')}")

    print(f"\nNext steps:")
    print(f"  Evaluate:  python -m scripts.evaluate_palmnet_lite --run-id {run_logger.run_id}")
    print(f"  Export:    python -m scripts.export_palmnet_lite --run-id {run_logger.run_id}")


if __name__ == "__main__":
    main()

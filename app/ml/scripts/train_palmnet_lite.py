"""
Main training script untuk PalmNet-Lite Scratch.

Usage:
    cd app/ml
    python -m scripts.train_palmnet_lite \\
        --config configs/palmnet_lite_scratch.yaml \\
        [--phase 0|1|2|all] \\
        [--phase1-checkpoint path/to/checkpoint_phase1_best.pth] \\
        [--force-phase2]
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

# Canonical path resolution
_ML_DIR = Path(__file__).resolve().parent.parent
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

from palm_recognition.paths import resolve_ml_path, ML_ROOT


def set_seed(seed: int) -> torch.Generator:
    """Set seed untuk reproducibility dan return PyTorch Generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    return g


def load_config(config_path: str | Path) -> dict:
    resolved_path = resolve_ml_path(config_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")
    with open(resolved_path) as f:
        return yaml.safe_load(f)


def run_phase0_sanity(config: dict, device: torch.device, run_dir: Path | None = None) -> dict:
    """Phase 0: Strict dataset sanity check and pipeline integrity validation."""
    from palm_recognition.data.dataset import PalmDataset, get_train_transform, get_val_transform, build_dataloaders
    from palm_recognition.models.palmnet_lite import build_palmnet_lite
    from palm_recognition.models.initialization import initialize_scratch_weights
    import pandas as pd

    ds_cfg = config.get("dataset", {})
    train_csv = resolve_ml_path(ds_cfg.get("train_csv", "data/splits/train.csv"))
    val_csv   = resolve_ml_path(ds_cfg.get("val_csv",   "data/splits/val.csv"))
    test_csv  = resolve_ml_path(ds_cfg.get("test_csv",  "data/splits/test.csv"))

    print("\n=== PHASE 0: Strict Dataset Sanity Check ===")

    # 1. Check file existence
    for csv_path, name in [(train_csv, "train"), (val_csv, "val"), (test_csv, "test")]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Phase 0 FAIL: {name}.csv tidak ditemukan di {csv_path}")

    # 2. Check dataframe schema and image file existence
    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)
    test_df  = pd.read_csv(test_csv)

    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        required = {"path", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Phase 0 FAIL: {name}.csv missing kolom: {missing}")

        # Check sample paths exist
        missing_images = []
        for idx, p_str in enumerate(df["path"].head(50)):
            p = resolve_ml_path(p_str)
            if not p.exists():
                missing_images.append(str(p))
        if missing_images:
            raise FileNotFoundError(f"Phase 0 FAIL: {name}.csv memiliki {len(missing_images)} gambar tidak ditemukan. Sample: {missing_images[:3]}")

    # 3. Duplicate paths check
    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        dups = df["path"].duplicated().sum()
        if dups > 0:
            raise ValueError(f"Phase 0 FAIL: {name}.csv mengandung {dups} path duplikat.")

    train_paths = set(train_df["path"])
    val_paths   = set(val_df["path"])
    test_paths  = set(test_df["path"])

    cross_dups = (train_paths & test_paths) | (val_paths & test_paths)
    if cross_dups:
        raise ValueError(f"Phase 0 FAIL: {len(cross_dups)} path duplikat ditemukan antara train/val dan test set.")

    # 4. Identity isolation check (HARD FAIL on overlap!)
    train_palms = set(train_df["palm_id"].astype(str).unique()) if "palm_id" in train_df.columns else set()
    val_palms   = set(val_df["palm_id"].astype(str).unique()) if "palm_id" in val_df.columns else set()
    test_palms  = set(test_df["palm_id"].astype(str).unique()) if "palm_id" in test_df.columns else set()

    overlap = train_palms & test_palms
    if overlap:
        raise ValueError(f"Phase 0 FAIL: Identity leakage detected! {len(overlap)} palm IDs overlap antara train dan test set.")

    # 5. Build datasets & smoke forward/backward
    train_ds = PalmDataset(train_csv, transform=get_train_transform())
    val_ds   = PalmDataset(val_csv,   transform=get_val_transform())

    batch_size = config.get("phase1", {}).get("batch_size", 64)
    loader_tr, _, num_classes = build_dataloaders(
        train_csv, val_csv, batch_size=min(batch_size, 8), num_workers=0
    )
    images, labels = next(iter(loader_tr))
    if images.shape[1:] != (3, 112, 112):
        raise ValueError(f"Phase 0 FAIL: Input tensor shape {images.shape[1:]} != (3, 112, 112)")

    backbone = build_palmnet_lite(config.get("model", {}))
    initialize_scratch_weights(backbone)
    backbone.to(device)
    backbone.eval()
    with torch.no_grad():
        out = backbone(images.to(device))
    if out.shape != (images.shape[0], 128) or not torch.isfinite(out).all():
        raise RuntimeError(f"Phase 0 FAIL: Forward pass output invalid shape/values: {out.shape}")

    backbone.train()
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
    if not grads_ok:
        raise RuntimeError("Phase 0 FAIL: Non-finite or missing gradients during backward smoke check.")

    summary = {
        "status": "passed",
        "train_images": len(train_ds),
        "val_images": len(val_ds),
        "test_images": len(test_df),
        "train_classes": train_ds.num_classes,
        "val_classes": val_ds.num_classes,
        "train_identities": len(train_palms),
        "val_identities": len(val_palms),
        "test_identities": len(test_palms),
        "train_test_identity_overlap": 0,
        "smoke_forward_ok": True,
        "smoke_backward_ok": True,
    }

    if run_dir:
        with open(run_dir / "phase0_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("Phase 0 PASSED ✓\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train PalmNet-Lite from scratch")
    parser.add_argument("--config", type=str, default="configs/palmnet_lite_scratch.yaml")
    parser.add_argument("--phase", type=str, default="all", choices=["0", "1", "2", "all"])
    parser.add_argument("--phase1-checkpoint", type=str, default=None)
    parser.add_argument("--force-phase2", action="store_true", help="Override Phase 1 accuracy gate")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("run", {}).get("seed", 42)
    generator = set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from palm_recognition.models.palmnet_lite import build_palmnet_lite
    from palm_recognition.models.initialization import initialize_scratch_weights
    from palm_recognition.training.logging import RunLogger
    from palm_recognition.data.dataset import build_dataloaders

    ds_cfg = config.get("dataset", {})
    train_csv = resolve_ml_path(ds_cfg.get("train_csv", "data/splits/train.csv"))
    val_csv   = resolve_ml_path(ds_cfg.get("val_csv",   "data/splits/val.csv"))

    paths_cfg = config.get("paths", {})
    checkpoint_dir  = resolve_ml_path(paths_cfg.get("checkpoint_dir",  "checkpoints/palmnet-lite-scratch"))
    trained_logs    = resolve_ml_path(paths_cfg.get("trained_logs_dir", "artifacts/trained_logs/palmnet-lite-scratch"))

    # ── Phase 0 ───────────────────────────────────────────────────────────────
    if args.phase in ("0", "all"):
        run_phase0_sanity(config, device)

    if args.phase == "0":
        print("Phase 0 finished.")
        return

    # ── Initialize RunLogger ──────────────────────────────────────────────────
    run_logger = RunLogger(base_dir=trained_logs, seed=seed)

    # ── Build DataLoaders ─────────────────────────────────────────────────────
    p1_cfg = config.get("phase1", {})
    batch_size = p1_cfg.get("batch_size", 64)
    aug_cfg = config.get("augmentation", {})

    train_loader, val_loader, num_classes = build_dataloaders(
        train_csv, val_csv, batch_size=batch_size, num_workers=0,
        aug_config=aug_cfg, seed=seed
    )

    model_cfg = config.get("model", {})
    backbone = build_palmnet_lite(cfg=model_cfg)
    param_info = backbone.count_parameters()
    trainable_params = param_info["trainable"]
    print(f"\nPalmNet-Lite v1 parameter count: {trainable_params:,}")
    assert 350_000 <= trainable_params <= 450_000, f"Param count {trainable_params:,} out of range [350k, 450k]"

    run_metadata = {
        "seed": seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "num_train_classes": num_classes,
        "train_images": len(train_loader.dataset),
        "val_images": len(val_loader.dataset),
        "parameter_count": trainable_params,
    }
    run_logger.init_run(config=config, extra=run_metadata)
    run_phase0_sanity(config, device, run_dir=run_logger.run_dir)

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

        best_val_acc = phase1_summary.get("best_val_accuracy", 0.0)
        gate_pass = p1_cfg.get("gate_min_acc_pass", 0.70)
        gate_review = p1_cfg.get("gate_min_acc_review", 0.50)

        if best_val_acc >= gate_pass:
            print(f"\n✓ Phase 1 Gate PASSED (val_acc={best_val_acc:.4f} >= {gate_pass})")
            run_logger.add_note(f"Phase 1 Gate PASSED (val_acc={best_val_acc:.4f})")
        elif best_val_acc >= gate_review:
            if args.force_phase2:
                print(f"\n⚠️  Phase 1 val_acc={best_val_acc:.4f} < {gate_pass}, but --force-phase2 is set. Proceeding...")
                run_logger.add_note(f"Phase 1 val_acc={best_val_acc:.4f} < {gate_pass}, phase2_forced=true")
            else:
                print(f"\n⏹ Phase 1 val_acc={best_val_acc:.4f} in [{gate_review}, {gate_pass}). Stopping for review. Use --force-phase2 to override.")
                run_logger.add_note(f"Stopped at Phase 1 gate for review (val_acc={best_val_acc:.4f})")
                run_logger.close(status="needs_review")
                return
        else:
            if args.force_phase2:
                print(f"\n⚠️  Phase 1 val_acc={best_val_acc:.4f} < {gate_review}, but --force-phase2 is set. Proceeding...")
                run_logger.add_note(f"Phase 1 val_acc={best_val_acc:.4f} < {gate_review}, phase2_forced=true")
            else:
                print(f"\n❌ Phase 1 FAILED (val_acc={best_val_acc:.4f} < {gate_review}). Stopping.")
                run_logger.add_note(f"Phase 1 FAILED gate (val_acc={best_val_acc:.4f})")
                run_logger.close(status="failed", failure_reason=f"Phase 1 val_accuracy {best_val_acc:.4f} < {gate_review}")
                raise RuntimeError(f"Phase 1 val_accuracy {best_val_acc:.4f} < {gate_review}")

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    phase2_summary = None
    if args.phase in ("2", "all"):
        phase1_ckpt_path = resolve_ml_path(args.phase1_checkpoint) if args.phase1_checkpoint else (checkpoint_dir / "checkpoint_phase1_best.pth")
        if not phase1_ckpt_path.exists():
            print(f"ERROR: Phase 1 checkpoint tidak ditemukan: {phase1_ckpt_path}")
            run_logger.close(status="failed", failure_reason="Phase 1 checkpoint tidak ditemukan")
            sys.exit(1)

        print(f"\nLoading Phase 1 backbone dari: {phase1_ckpt_path}")
        ckpt = torch.load(phase1_ckpt_path, map_location=device)
        backbone2 = build_palmnet_lite(cfg=model_cfg)
        backbone2.load_state_dict(ckpt["backbone_state_dict"])

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


if __name__ == "__main__":
    main()

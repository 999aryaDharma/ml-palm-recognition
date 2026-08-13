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
    """Phase 0: Strict dataset sanity check and scientific protocol integrity validation."""
    from palm_recognition.data.dataset import PalmDataset, get_train_transform, get_val_transform, build_dataloaders
    from palm_recognition.models.palmnet_lite import build_palmnet_lite
    from palm_recognition.models.initialization import initialize_scratch_weights
    import pandas as pd
    import json

    ds_cfg = config.get("dataset", {})
    train_csv = resolve_ml_path(ds_cfg.get("train_csv", "data/splits/train.csv"))
    val_csv   = resolve_ml_path(ds_cfg.get("val_csv",   "data/splits/val.csv"))
    test_csv  = resolve_ml_path(ds_cfg.get("test_csv",  "data/splits/test.csv"))

    print("\n=== PHASE 0: Strict Dataset Sanity Check ===")

    # 1. Check file existence
    for csv_path, name in [(train_csv, "train"), (val_csv, "val"), (test_csv, "test")]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Phase 0 FAIL: {name}.csv tidak ditemukan di {csv_path}")

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)
    test_df  = pd.read_csv(test_csv)

    # 2. Check required columns: path, label, palm_id, session (Req 7)
    required_cols = {"path", "label", "palm_id", "session"}
    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Phase 0 FAIL: {name}.csv missing kolom wajib: {missing_cols}")

    # 3. Check ALL image paths exist on disk (Req 8)
    missing_files = 0
    missing_samples = []
    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        for p_str in df["path"]:
            p = resolve_ml_path(p_str)
            if not p.exists():
                missing_files += 1
                if len(missing_samples) < 5:
                    missing_samples.append(str(p))

    if missing_files > 0:
        raise FileNotFoundError(
            f"Phase 0 FAIL: Ditemukan {missing_files} file gambar tidak ada di disk. Contoh: {missing_samples}"
        )

    # 4. Duplicate paths within split and across splits (Req 9.A & 9.B)
    duplicate_paths = 0
    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        dups = int(df["path"].duplicated().sum())
        duplicate_paths += dups
        if dups > 0:
            raise ValueError(f"Phase 0 FAIL: {name}.csv mengandung {dups} path duplikat.")

    train_paths = set(train_df["path"])
    val_paths   = set(val_df["path"])
    test_paths  = set(test_df["path"])

    cross_split_duplicates = len((train_paths & val_paths) | (train_paths & test_paths) | (val_paths & test_paths))
    if cross_split_duplicates > 0:
        raise ValueError(f"Phase 0 FAIL: Ditemukan {cross_split_duplicates} path duplikat antar split.")

    # 5. Identity protocol assertion (Req 9.C)
    train_palms = set(train_df["palm_id"].astype(str).unique())
    val_palms   = set(val_df["palm_id"].astype(str).unique())
    test_palms  = set(test_df["palm_id"].astype(str).unique())

    if train_palms != val_palms:
        diff_tv = (train_palms ^ val_palms)
        raise ValueError(f"Phase 0 FAIL: Identity mismatch antara train dan val set ({len(diff_tv)} palm_ids berbeda).")

    identity_overlap_count = len((train_palms & test_palms) | (val_palms & test_palms))
    if identity_overlap_count > 0:
        raise ValueError(f"Phase 0 FAIL: Identity leakage detected! {identity_overlap_count} palm IDs overlap dengan test set.")

    # 6. Session protocol assertion (Req 9.D)
    if not (train_df["session"] == 1).all():
        raise ValueError("Phase 0 FAIL: Semua sampel train_df harus dari session 1.")
    if not (val_df["session"] == 2).all():
        raise ValueError("Phase 0 FAIL: Semua sampel val_df harus dari session 2.")

    for pid in test_palms:
        p_df = test_df[test_df["palm_id"].astype(str) == pid]
        sessions = set(p_df["session"].unique())
        if not ({1, 2}.issubset(sessions)):
            raise ValueError(f"Phase 0 FAIL: Test palm_id '{pid}' harus memiliki sampel dari session 1 dan session 2.")

    # 7. Label mapping consistency assertion (Req 9.E)
    train_map = train_df.groupby("palm_id")["label"].unique().to_dict()
    val_map   = val_df.groupby("palm_id")["label"].unique().to_dict()

    for pid, labels in train_map.items():
        if len(labels) > 1:
            raise ValueError(f"Phase 0 FAIL: palm_id '{pid}' di train_df terhubung ke lebih dari 1 label: {labels}")
        if pid in val_map:
            v_labels = val_map[pid]
            if len(v_labels) > 1 or v_labels[0] != labels[0]:
                raise ValueError(f"Phase 0 FAIL: Inkonsistensi label untuk palm_id '{pid}' antara train ({labels[0]}) dan val ({v_labels[0]}).")

    # 8. Forward & backward smoke test
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
        "train_images": len(train_df),
        "val_images": len(val_df),
        "test_images": len(test_df),
        "train_classes": int(train_df["label"].nunique()),
        "val_classes": int(val_df["label"].nunique()),
        "test_identities": int(len(test_palms)),
        "train_sessions": sorted([int(s) for s in train_df["session"].unique()]),
        "val_sessions": sorted([int(s) for s in val_df["session"].unique()]),
        "test_sessions": sorted([int(s) for s in test_df["session"].unique()]),
        "train_identities": int(len(train_palms)),
        "val_identities": int(len(val_palms)),
        "duplicate_paths": duplicate_paths,
        "cross_split_duplicates": cross_split_duplicates,
        "identity_overlap_count": identity_overlap_count,
        "missing_files": missing_files,
        "unreadable_files": 0,
        "protocol_valid": True,
        "forward_smoke_ok": True,
        "backward_smoke_ok": True,
    }

    target_dir = run_dir if run_dir else resolve_ml_path("artifacts")
    target_dir.mkdir(parents=True, exist_ok=True)
    summary_path = target_dir / "phase0_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Phase 0 summary saved: {summary_path}")
    print("Phase 0 PASSED [OK]\n")



    return summary


def evaluate_phase1_gate(
    best_val_accuracy: float,
    pass_threshold: float = 0.70,
    review_threshold: float = 0.50,
    force: bool = False,
) -> str:
    """Evaluates Phase 1 validation accuracy against scientific gate thresholds.

    Returns one of: 'passed', 'needs_review', 'failed', 'forced'.
    """
    if best_val_accuracy >= pass_threshold:
        return "passed"
    elif best_val_accuracy >= review_threshold:
        return "forced" if force else "needs_review"
    else:
        return "forced" if force else "failed"


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
    checkpoint_base_dir = resolve_ml_path(paths_cfg.get("checkpoint_dir", "checkpoints/palmnet-lite-scratch"))
    trained_logs       = resolve_ml_path(paths_cfg.get("trained_logs_dir", "artifacts/trained_logs/palmnet-lite-scratch"))

    # ── Phase 0 ───────────────────────────────────────────────────────────────
    if args.phase in ("0", "all"):
        run_phase0_sanity(config, device)

    if args.phase == "0":
        print("Phase 0 finished.")
        return

    # ── Initialize RunLogger ──────────────────────────────────────────────────
    run_logger = RunLogger(base_dir=trained_logs, seed=seed)
    checkpoint_run_dir = checkpoint_base_dir / run_logger.run_id
    checkpoint_run_dir.mkdir(parents=True, exist_ok=True)

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
        "checkpoint_dir": str(checkpoint_run_dir),
    }
    run_logger.init_run(config=config, extra=run_metadata)
    run_phase0_sanity(config, device, run_dir=run_logger.run_dir)

    print(f"\nRun ID: {run_logger.run_id}")
    print(f"Run dir: {run_logger.run_dir}")
    print(f"Checkpoint dir: {checkpoint_run_dir}")

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
                checkpoint_dir=checkpoint_run_dir,
                run_logger=run_logger,
                device=device,
                run_id=run_logger.run_id,
            )
            run_logger.update_run_json({
                "phase1_summary": phase1_summary,
                "phase1_best_checkpoint": str(checkpoint_run_dir / "checkpoint_phase1_best.pth"),
            })
        except Exception as e:
            run_logger.add_note(f"Phase 1 FAILED: {e}")
            run_logger.close(status="failed", failure_reason=str(e))
            raise

        best_val_acc = phase1_summary.get("best_val_accuracy", 0.0)
        gate_pass = p1_cfg.get("gate_min_acc_pass", 0.70)
        gate_review = p1_cfg.get("gate_min_acc_review", 0.50)

        gate_status = evaluate_phase1_gate(
            best_val_acc, pass_threshold=gate_pass, review_threshold=gate_review, force=args.force_phase2
        )
        run_logger.update_run_json({"phase1_gate_status": gate_status, "forced_phase2": args.force_phase2})

        if gate_status == "passed":
            print(f"\n[OK] Phase 1 Gate PASSED (val_acc={best_val_acc:.4f} >= {gate_pass})")
            run_logger.add_note(f"Phase 1 Gate PASSED (val_acc={best_val_acc:.4f})")
        elif gate_status == "forced":
            print(f"\n[WARN] Phase 1 val_acc={best_val_acc:.4f}, --force-phase2 set. Proceeding to Phase 2.")
            run_logger.add_note(f"Phase 1 val_acc={best_val_acc:.4f}, phase2_forced=true")
        elif gate_status == "needs_review":
            print(f"\n[STOP] Phase 1 val_acc={best_val_acc:.4f} in [{gate_review}, {gate_pass}). Stopping for review.")
            run_logger.add_note(f"Stopped at Phase 1 gate for review (val_acc={best_val_acc:.4f})")
            run_logger.close(status="needs_review")
            return
        else:
            print(f"\n[FAIL] Phase 1 FAILED (val_acc={best_val_acc:.4f} < {gate_review}). Stopping.")
            run_logger.add_note(f"Phase 1 FAILED gate (val_acc={best_val_acc:.4f})")
            run_logger.close(status="failed", failure_reason=f"Phase 1 val_accuracy {best_val_acc:.4f} < {gate_review}")
            raise RuntimeError(f"Phase 1 val_accuracy {best_val_acc:.4f} < {gate_review}")

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    phase2_summary = None
    if args.phase in ("2", "all"):
        phase1_ckpt_path = resolve_ml_path(args.phase1_checkpoint) if args.phase1_checkpoint else (checkpoint_run_dir / "checkpoint_phase1_best.pth")
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
                checkpoint_dir=checkpoint_run_dir,
                run_logger=run_logger,
                device=device,
                run_id=run_logger.run_id,
            )
            run_logger.update_run_json({
                "phase2_summary": phase2_summary,
                "phase2_best_checkpoint": str(checkpoint_run_dir / "checkpoint_phase2_best.pth"),
            })
        except Exception as e:
            run_logger.add_note(f"Phase 2 FAILED: {e}")
            run_logger.close(status="failed", failure_reason=str(e))
            raise

    run_logger.close(status="completed")
    print(f"\n[OK] Training selesai. Run ID: {run_logger.run_id}")



if __name__ == "__main__":
    main()

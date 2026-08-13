"""
Phase 2 — ArcFace Metric Learning untuk PalmNet-Lite.

Input: backbone dari checkpoint_phase1_best.pth (own training hasil)
Goal: membentuk embedding geometry untuk biometric matching.

Best checkpoint selection:
Primary: highest cosine_gap
Tie-break: lower train_loss if cosine_gap is equal
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from palm_recognition.paths import resolve_ml_path


def compute_val_embedding_metrics(
    backbone: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_pairs: int = 50000,
) -> dict:
    backbone.eval()
    all_embs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            embs = backbone(images)
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu())
            all_labels.extend(labels.tolist())

    embs_np = torch.cat(all_embs, dim=0).numpy()
    labels_np = np.array(all_labels)

    label_to_idx: dict = defaultdict(list)
    for i, lbl in enumerate(labels_np):
        label_to_idx[int(lbl)].append(i)

    genuine_scores = []
    impostor_scores = []
    rng = np.random.default_rng(42)

    for lbl, idxs in label_to_idx.items():
        idxs = np.array(idxs)
        if len(idxs) >= 2:
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    s = float(np.dot(embs_np[idxs[i]], embs_np[idxs[j]]))
                    genuine_scores.append(s)

    num_classes = len(label_to_idx)
    class_list = list(label_to_idx.keys())
    n_imp = min(max_pairs, len(genuine_scores) * 3)
    imp_count = 0
    for _ in range(n_imp * 2):
        if imp_count >= n_imp:
            break
        c1, c2 = rng.choice(num_classes, 2, replace=False)
        idx1 = rng.choice(label_to_idx[class_list[c1]])
        idx2 = rng.choice(label_to_idx[class_list[c2]])
        s = float(np.dot(embs_np[idx1], embs_np[idx2]))
        impostor_scores.append(s)
        imp_count += 1

    if not genuine_scores:
        return {"mean_positive_cosine": 0.0, "mean_negative_cosine": 0.0, "cosine_gap": 0.0}

    mean_pos = float(np.mean(genuine_scores))
    mean_neg = float(np.mean(impostor_scores)) if impostor_scores else 0.0
    gap = mean_pos - mean_neg

    return {
        "mean_positive_cosine": round(mean_pos, 6),
        "mean_negative_cosine": round(mean_neg, 6),
        "cosine_gap": round(gap, 6),
        "n_genuine_pairs": len(genuine_scores),
        "n_impostor_pairs": len(impostor_scores),
    }


def run_phase2(
    backbone: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    config: dict,
    checkpoint_dir: str | Path,
    run_logger,
    device: torch.device,
    run_id: str,
) -> dict:
    from palm_recognition.losses.arcface import ArcFaceLoss, LinearMarginWarmup
    from palm_recognition.training.checkpointing import save_phase2_checkpoint

    checkpoint_dir = resolve_ml_path(checkpoint_dir)

    epochs = config.get("epochs", 50)
    lr = config.get("lr", 1e-4)
    weight_decay = config.get("weight_decay", 1e-4)
    arcface_margin = config.get("arcface_margin", 0.30)
    arcface_scale = config.get("arcface_scale", 32.0)
    margin_warmup_epochs = config.get("margin_warmup_epochs", 8)
    min_lr = config.get("min_lr", 1e-6)
    grad_clip = config.get("grad_clip_norm", 5.0)

    backbone = backbone.to(device)
    arcface = ArcFaceLoss(
        in_features=128,
        num_classes=num_classes,
        margin=0.0,
        scale=arcface_scale,
    ).to(device)
    margin_scheduler = LinearMarginWarmup(arcface, target_margin=arcface_margin,
                                          warmup_epochs=margin_warmup_epochs)

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(arcface.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=min_lr
    )

    best_cosine_gap = -float("inf")
    best_train_loss = float("inf")
    best_epoch = -1
    best_metrics: dict = {}
    history = []

    print(f"\n{'='*60}")
    print(f"Phase 2 — ArcFace Metric Learning")
    print(f"  Epochs: {epochs}, LR: {lr}, Margin: {arcface_margin}, Scale: {arcface_scale}")
    print(f"  Margin warmup: {margin_warmup_epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(epochs):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        current_margin = margin_scheduler.step(epoch)

        backbone.train()
        arcface.train()
        train_loss_acc = 0.0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            emb = backbone(images)
            loss = arcface(emb, labels)

            if not torch.isfinite(loss):
                run_logger.add_note(f"ABORT: non-finite loss at epoch {epoch+1}")
                raise RuntimeError(f"Non-finite loss (ArcFace) at epoch {epoch+1}: {loss.item()}")

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(arcface.parameters()),
                    grad_clip,
                )
            optimizer.step()

            train_loss_acc += loss.item() * images.size(0)
            train_total += images.size(0)

        train_loss = train_loss_acc / max(train_total, 1)
        scheduler.step()

        emb_metrics = compute_val_embedding_metrics(backbone, val_loader, device)
        cosine_gap = emb_metrics.get("cosine_gap", 0.0)

        duration = time.time() - t0
        metrics = {
            "train_loss": round(train_loss, 6),
            **emb_metrics,
            "learning_rate": round(current_lr, 8),
            "arcface_margin": round(current_margin, 4),
            "epoch_duration_sec": round(duration, 2),
        }
        history.append({"epoch": epoch + 1, **metrics})
        run_logger.log_epoch(phase=2, epoch=epoch + 1, metrics=metrics)

        # Primary selection: cosine_gap. Tie-break: lower train_loss
        is_best = False
        if cosine_gap > best_cosine_gap + 1e-6:
            is_best = True
        elif abs(cosine_gap - best_cosine_gap) <= 1e-6 and train_loss < best_train_loss:
            is_best = True

        if is_best:
            best_cosine_gap = cosine_gap
            best_train_loss = train_loss
            best_epoch = epoch + 1
            best_metrics = emb_metrics.copy()

        save_phase2_checkpoint(
            checkpoint_dir=checkpoint_dir,
            run_id=run_id,
            epoch=epoch + 1,
            backbone=backbone,
            arcface_head=arcface,
            optimizer=optimizer,
            scheduler=scheduler,
            current_margin=current_margin,
            validation_metrics=emb_metrics,
            config=config,
            is_best=is_best,
        )

        star = " ★" if is_best else ""
        print(
            f"[Phase2] Epoch {epoch+1:03d}/{epochs} | "
            f"loss {train_loss:.4f} | "
            f"gap {cosine_gap:.4f} "
            f"(pos={emb_metrics.get('mean_positive_cosine', 0):.4f} "
            f"neg={emb_metrics.get('mean_negative_cosine', 0):.4f}) | "
            f"m={current_margin:.3f} lr={current_lr:.2e} | {duration:.1f}s{star}"
        )

    summary = {
        "best_epoch": best_epoch,
        "best_cosine_gap": best_cosine_gap,
        "best_metrics": best_metrics,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "epochs_run": epochs,
        "arcface_margin": arcface_margin,
        "arcface_scale": arcface_scale,
        "num_classes": num_classes,
    }
    run_logger.save_phase_summary(2, summary)
    print(f"\nPhase 2 selesai. Best cosine_gap={best_cosine_gap:.4f} at epoch {best_epoch}")
    return summary

"""
Phase 1 — Softmax Representation Learning untuk PalmNet-Lite.

Goal: mengajarkan backbone scratch membangun feature dasar yang diskriminatif.

Pipeline:
    PalmNetLite (random init)
    -> raw 128-D embedding
    -> Linear classifier (num_train_classes)
    -> CrossEntropyLoss (label_smoothing=0.1)
    -> backprop seluruh backbone

Tidak ada layer yang difreeze.
Backbone + classifier semua trainable.

Setelah Phase 1 selesai:
    - simpan best_phase1.pth (by val_accuracy)
    - buang classifier
    - lanjut Phase 2 dengan ArcFace
"""
from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def run_phase1(
    backbone: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    config: dict,
    checkpoint_dir: str,
    run_logger,
    device: torch.device,
    run_id: str,
) -> dict:
    """Jalankan Phase 1 Softmax Representation Learning.

    Args:
        backbone:        PalmNetLite (random-initialized)
        train_loader:    training DataLoader
        val_loader:      validation DataLoader
        num_classes:     jumlah kelas training (palm identity count)
        config:          phase1 config dict (lr, epochs, dll)
        checkpoint_dir:  path untuk menyimpan checkpoint
        run_logger:      RunLogger instance
        device:          torch device
        run_id:          string identifier run ini

    Returns:
        dict summary Phase 1
    """
    from palm_recognition.training.checkpointing import save_phase1_checkpoint

    epochs = config.get("epochs", 40)
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 1e-4)
    label_smoothing = config.get("label_smoothing", 0.1)
    warmup_epochs = config.get("warmup_epochs", 3)
    min_lr = config.get("min_lr", 1e-6)
    grad_clip = config.get("grad_clip_norm", 5.0)

    # ── Model ──────────────────────────────────────────────────────────────────
    backbone = backbone.to(device)
    classifier = nn.Linear(128, num_classes).to(device)
    nn.init.xavier_normal_(classifier.weight)
    nn.init.zeros_(classifier.bias)

    # ── Optimizer — seluruh backbone + classifier ──────────────────────────────
    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(classifier.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # ── LR Schedule: linear warmup + cosine ──────────────────────────────────
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return max(min_lr / lr, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    history = []

    print(f"\n{'='*60}")
    print(f"Phase 1 — Softmax Representation Learning")
    print(f"  Epochs: {epochs}, LR: {lr}, Classes: {num_classes}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Train ─────────────────────────────────────────────────────────────
        backbone.train()
        classifier.train()
        train_loss_acc = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            emb = backbone(images)
            logits = classifier(emb)
            loss = loss_fn(logits, labels)

            if not torch.isfinite(loss):
                run_logger.add_note(f"WARN: non-finite loss at epoch {epoch+1}, aborting")
                raise RuntimeError(f"Non-finite loss at epoch {epoch+1}: {loss.item()}")

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(classifier.parameters()),
                    grad_clip,
                )
            optimizer.step()

            train_loss_acc += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        train_loss = train_loss_acc / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # ── Validation ────────────────────────────────────────────────────────
        backbone.eval()
        classifier.eval()
        val_loss_acc = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                emb = backbone(images)
                logits = classifier(emb)
                loss = loss_fn(logits, labels)
                val_loss_acc += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        val_loss = val_loss_acc / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        scheduler.step()

        duration = time.time() - t0
        metrics = {
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 6),
            "learning_rate": round(current_lr, 8),
            "epoch_duration_sec": round(duration, 2),
        }
        history.append({"epoch": epoch + 1, **metrics})
        run_logger.log_epoch(phase=1, epoch=epoch + 1, metrics=metrics)

        # ── Best checkpoint selection ──────────────────────────────────────────
        is_best = False
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1
            is_best = True

        save_phase1_checkpoint(
            checkpoint_dir=checkpoint_dir,
            run_id=run_id,
            epoch=epoch + 1,
            backbone=backbone,
            classifier=classifier,
            optimizer=optimizer,
            scheduler=scheduler,
            best_metric=best_val_acc,
            config=config,
            is_best=is_best,
        )

        star = " ★" if is_best else ""
        print(
            f"[Phase1] Epoch {epoch+1:03d}/{epochs} | "
            f"loss {train_loss:.4f}/{val_loss:.4f} | "
            f"acc {train_acc:.4f}/{val_acc:.4f} | "
            f"lr {current_lr:.2e} | {duration:.1f}s{star}"
        )

    summary = {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_val_loss": history[-1]["val_loss"] if history else None,
        "epochs_run": epochs,
        "num_classes": num_classes,
    }
    run_logger.save_phase_summary(1, summary)
    print(f"\nPhase 1 selesai. Best val_acc={best_val_acc:.4f} at epoch {best_epoch}")
    return summary

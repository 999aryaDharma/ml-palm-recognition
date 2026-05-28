"""
Phase 1 Training: Softmax Warm-up

Tujuan:
- Domain adaptation dari MS-Celeb-1M (wajah) ke palm
- Stabilkan backbone sebelum ArcFace di Phase 2
- Pakai cross-entropy biasa (lebih stabil dari ArcFace untuk awal training)

Strategi 2-step:
  Step A (frozen backbone, 5 epoch):
    - Freeze semua layer backbone, hanya train classifier head baru
    - LR tinggi (1e-3) — fokus belajar mapping feature → kelas palm
  Step B (unfreeze, 10 epoch):
    - Unfreeze SEMUA layer
    - LR rendah (1e-4) — fine-tune feature representation halus
    - Save best checkpoint berdasarkan val accuracy

Gate: kalau best val acc < 75% setelah Phase 1, JANGAN lanjut ke Phase 2.
Kemungkinan masalah: ROI extraction error, label mismatch, dataset issue.

Output:
- ml/models/checkpoint_phase1_best.pth
- TensorBoard logs di ml/artifacts/training_logs/phase1_<timestamp>/
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    SPLITS_DIR,
    MODELS_DIR,
    LOGS_DIR,
    PRETRAINED_PATH,
    EMBEDDING_DIM,
    PHASE1_EPOCHS_FROZEN,
    PHASE1_EPOCHS_UNFROZEN,
    PHASE1_LR_FROZEN,
    PHASE1_LR_UNFROZEN,
    PHASE1_BATCH_SIZE,
    PHASE1_TARGET_VAL_ACC,
    RANDOM_SEED,
)
from mobilefacenet import MobileFaceNet, load_mobilefacenet
from data_loader import build_dataloaders


# ============================================================================
# Phase 1 Model: backbone + softmax classifier head
# ============================================================================
class Phase1Model(nn.Module):
    """
    Wrapper: backbone MobileFaceNet + linear softmax classifier.

    Saat training Phase 1, output adalah logit kelas (num_classes).
    Setelah Phase 1 selesai, classifier dibuang — yang dipakai untuk Phase 2
    hanyalah backbone-nya saja (embedding).
    """

    def __init__(self, backbone: MobileFaceNet, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(EMBEDDING_DIM, num_classes)

    def forward(self, x):
        emb = self.backbone(x)  # (B, 128)
        logits = self.classifier(emb)
        return logits

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


# ============================================================================
# Train / Eval Loops
# ============================================================================
def run_epoch(model, loader, optimizer, criterion, device, desc=""):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.3f}"})

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_phase1(
    train_csv: Path,
    val_csv: Path,
    pretrained_path: Path | None,
    output_dir: Path,
    logs_dir: Path,
    epochs_frozen: int = PHASE1_EPOCHS_FROZEN,
    epochs_unfrozen: int = PHASE1_EPOCHS_UNFROZEN,
    batch_size: int = PHASE1_BATCH_SIZE,
    lr_frozen: float = PHASE1_LR_FROZEN,
    lr_unfrozen: float = PHASE1_LR_UNFROZEN,
    num_workers: int = 4,
    seed: int = RANDOM_SEED,
):
    # Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Data
    train_loader, val_loader, num_classes = build_dataloaders(
        train_csv, val_csv, batch_size=batch_size, num_workers=num_workers
    )
    print(f"Num classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    if pretrained_path and Path(pretrained_path).exists():
        print(f"Loading pretrained weights: {pretrained_path}")
        backbone = load_mobilefacenet(str(pretrained_path), strict=False)
    else:
        print("⚠️  Tidak ada pretrained — training from random init (akurasi akan rendah)")
        backbone = MobileFaceNet()

    model = Phase1Model(backbone, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    # TensorBoard
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=str(logs_dir / run_name))

    best_val_acc = 0.0
    best_epoch = -1
    global_epoch = 0

    # ====== STEP A: Frozen backbone ======
    print(f"\n=== Phase 1A: Frozen backbone ({epochs_frozen} epoch, lr={lr_frozen}) ===")
    model.freeze_backbone()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr_frozen
    )

    for epoch in range(epochs_frozen):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, criterion, device,
            desc=f"P1A Epoch {epoch+1}/{epochs_frozen}"
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0

        print(
            f"  Epoch {epoch+1}/{epochs_frozen} ({dt:.1f}s) | "
            f"Train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
            f"Val loss: {val_loss:.4f} acc: {val_acc:.3f}"
        )

        writer.add_scalar("phase1a/train_loss", train_loss, global_epoch)
        writer.add_scalar("phase1a/train_acc", train_acc, global_epoch)
        writer.add_scalar("phase1a/val_loss", val_loss, global_epoch)
        writer.add_scalar("phase1a/val_acc", val_acc, global_epoch)
        global_epoch += 1

    # ====== STEP B: Unfrozen ======
    print(f"\n=== Phase 1B: Unfreeze ({epochs_unfrozen} epoch, lr={lr_unfrozen}) ===")
    model.unfreeze_backbone()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_unfrozen)

    for epoch in range(epochs_unfrozen):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, criterion, device,
            desc=f"P1B Epoch {epoch+1}/{epochs_unfrozen}"
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0

        print(
            f"  Epoch {epoch+1}/{epochs_unfrozen} ({dt:.1f}s) | "
            f"Train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
            f"Val loss: {val_loss:.4f} acc: {val_acc:.3f}"
        )

        writer.add_scalar("phase1b/train_loss", train_loss, global_epoch)
        writer.add_scalar("phase1b/train_acc", train_acc, global_epoch)
        writer.add_scalar("phase1b/val_loss", val_loss, global_epoch)
        writer.add_scalar("phase1b/val_acc", val_acc, global_epoch)
        global_epoch += 1

        # Save best checkpoint (hanya backbone — classifier dibuang nanti)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            ckpt_path = output_dir / "checkpoint_phase1_best.pth"
            torch.save({
                "epoch": global_epoch,
                "backbone_state_dict": model.backbone.state_dict(),
                "classifier_state_dict": model.classifier.state_dict(),
                "val_acc": val_acc,
                "num_classes": num_classes,
            }, ckpt_path)
            print(f"  ✅ Best checkpoint saved → {ckpt_path}")

    writer.close()

    # ====== Gate Check ======
    print("\n" + "=" * 60)
    print("=== PHASE 1 COMPLETE ===")
    print(f"Best val accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    if best_val_acc < PHASE1_TARGET_VAL_ACC:
        print(f"⚠️  Val acc {best_val_acc:.3f} < target {PHASE1_TARGET_VAL_ACC}")
        print(f"   Investigasi dulu sebelum lanjut ke Phase 2:")
        print(f"   - Cek ROI extraction quality (sample visual inspection)")
        print(f"   - Cek label mapping di train.csv/val.csv konsisten")
        print(f"   - Cek pretrained loading (apakah keys cocok?)")
    else:
        print(f"✅ Val acc {best_val_acc:.3f} >= target {PHASE1_TARGET_VAL_ACC}")
        print(f"   Lanjut ke Phase 2 (ArcFace fine-tune).")
    print("=" * 60)

    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Training (Softmax Warm-up)")
    parser.add_argument("--train-csv", type=str, default=str(SPLITS_DIR / "train.csv"))
    parser.add_argument("--val-csv", type=str, default=str(SPLITS_DIR / "val.csv"))
    parser.add_argument("--pretrained", type=str, default=str(PRETRAINED_PATH))
    parser.add_argument("--output-dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--logs-dir", type=str, default=str(LOGS_DIR))
    parser.add_argument("--epochs-frozen", type=int, default=PHASE1_EPOCHS_FROZEN)
    parser.add_argument("--epochs-unfrozen", type=int, default=PHASE1_EPOCHS_UNFROZEN)
    parser.add_argument("--batch-size", type=int, default=PHASE1_BATCH_SIZE)
    parser.add_argument("--lr-frozen", type=float, default=PHASE1_LR_FROZEN)
    parser.add_argument("--lr-unfrozen", type=float, default=PHASE1_LR_UNFROZEN)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    train_phase1(
        train_csv=Path(args.train_csv),
        val_csv=Path(args.val_csv),
        pretrained_path=Path(args.pretrained),
        output_dir=Path(args.output_dir),
        logs_dir=Path(args.logs_dir),
        epochs_frozen=args.epochs_frozen,
        epochs_unfrozen=args.epochs_unfrozen,
        batch_size=args.batch_size,
        lr_frozen=args.lr_frozen,
        lr_unfrozen=args.lr_unfrozen,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

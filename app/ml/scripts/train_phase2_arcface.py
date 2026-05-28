"""
Phase 2 Training: ArcFace Fine-tuning

Tujuan:
- Replace softmax classifier dengan ArcFace loss
- Optimasi metric: tingkatkan cosine gap (mean_pos - mean_neg) di validation
- Bukan akurasi klasifikasi (closed-set), tapi separability embedding

Recipe:
- Load checkpoint Phase 1 (backbone-nya saja)
- ArcFace head dengan margin=0.5, scale=64
- Linear margin warm-up: 0 → 0.5 selama 5 epoch (hindari training collapse)
- AdamW optimizer, lr=5e-5, weight_decay=1e-4
- Cosine annealing LR schedule
- Gradient clipping (max_norm=5.0)
- 30 epoch total, save best berdasarkan cosine_gap

Gate: best cosine_gap > 0.4 di val set → siap untuk evaluation final.
Kalau < 0.4: tuning ulang hyperparameter.

Monitoring metric (cosine gap):
  gap = mean(positive_similarities) - mean(negative_similarities)
  - positive: pair yang sama kelas
  - negative: pair yang beda kelas
  Lebih besar = lebih separated = lebih bagus.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    SPLITS_DIR,
    MODELS_DIR,
    LOGS_DIR,
    EMBEDDING_DIM,
    PHASE2_EPOCHS,
    PHASE2_BATCH_SIZE,
    PHASE2_LR,
    PHASE2_WEIGHT_DECAY,
    PHASE2_GRAD_CLIP_NORM,
    PHASE2_TARGET_COSINE_GAP,
    ARCFACE_MARGIN,
    ARCFACE_SCALE,
    ARCFACE_MARGIN_WARMUP_EPOCHS,
    RANDOM_SEED,
)
from mobilefacenet import MobileFaceNet
from arcface_loss import ArcFaceLoss, LinearMarginScheduler
from data_loader import build_dataloaders


@torch.no_grad()
def compute_cosine_gap(model, loader, device, max_pairs_per_sample: int = 50):
    """
    Hitung cosine_gap = mean(positive_sim) - mean(negative_sim) di val set.

    Implementasi efisien: hitung embedding sekali, lalu sample pairs di matriks
    similarity untuk hindari O(N²) brute force.

    Args:
        max_pairs_per_sample: untuk tiap sample, bandingkan dengan max N tetangga
                              di matriks (default 50). Trade-off akurasi vs speed.
    """
    model.eval()
    all_embs = []
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        embs = model(imgs)
        embs = F.normalize(embs, dim=1)
        all_embs.append(embs.cpu())
        all_labels.append(labels.clone())

    all_embs = torch.cat(all_embs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    n = all_embs.size(0)

    if n < 2:
        return 0.0, 0.0, 0.0

    # Hitung similarity matrix (efficient: matrix mult)
    sim_matrix = all_embs @ all_embs.T  # (N, N)

    pos_sims = []
    neg_sims = []

    for i in range(n):
        # Bandingkan dengan rentang [i+1, i+1+max_pairs_per_sample]
        end = min(i + 1 + max_pairs_per_sample, n)
        for j in range(i + 1, end):
            sim = sim_matrix[i, j].item()
            if all_labels[i].item() == all_labels[j].item():
                pos_sims.append(sim)
            else:
                neg_sims.append(sim)

    if not pos_sims or not neg_sims:
        return 0.0, 0.0, 0.0

    mean_pos = sum(pos_sims) / len(pos_sims)
    mean_neg = sum(neg_sims) / len(neg_sims)
    gap = mean_pos - mean_neg
    return gap, mean_pos, mean_neg


def train_phase2(
    train_csv: Path,
    val_csv: Path,
    phase1_checkpoint: Path,
    output_dir: Path,
    logs_dir: Path,
    num_epochs: int = PHASE2_EPOCHS,
    batch_size: int = PHASE2_BATCH_SIZE,
    lr: float = PHASE2_LR,
    weight_decay: float = PHASE2_WEIGHT_DECAY,
    margin: float = ARCFACE_MARGIN,
    scale: float = ARCFACE_SCALE,
    margin_warmup_epochs: int = ARCFACE_MARGIN_WARMUP_EPOCHS,
    grad_clip_norm: float = PHASE2_GRAD_CLIP_NORM,
    num_workers: int = 4,
    seed: int = RANDOM_SEED,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, num_classes = build_dataloaders(
        train_csv, val_csv, batch_size=batch_size, num_workers=num_workers
    )
    print(f"Num classes: {num_classes}")

    # Model: pakai backbone dari Phase 1
    backbone = MobileFaceNet()
    if not Path(phase1_checkpoint).exists():
        raise FileNotFoundError(
            f"Phase 1 checkpoint tidak ditemukan: {phase1_checkpoint}\n"
            f"Jalankan train_phase1_softmax.py dulu."
        )
    ckpt = torch.load(phase1_checkpoint, map_location=device)
    backbone.load_state_dict(ckpt["backbone_state_dict"])
    backbone = backbone.to(device)
    print(f"Loaded Phase 1 checkpoint: val_acc={ckpt.get('val_acc', 'N/A')}")

    # ArcFace loss
    arcface = ArcFaceLoss(
        in_features=EMBEDDING_DIM,
        num_classes=num_classes,
        margin=margin,
        scale=scale,
    ).to(device)
    margin_scheduler = LinearMarginScheduler(arcface, margin, margin_warmup_epochs)

    # Optimizer: gabungkan param backbone + arcface weight matrix
    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(arcface.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # TensorBoard
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=str(logs_dir / run_name))

    best_cosine_gap = -1.0
    best_epoch = -1

    print(f"\n=== Phase 2: ArcFace Fine-tune ({num_epochs} epoch, lr={lr}) ===")
    print(f"Margin warm-up: 0 → {margin} over {margin_warmup_epochs} epoch")

    for epoch in range(num_epochs):
        current_margin = margin_scheduler.step(epoch)
        backbone.train()
        arcface.train()

        total_loss = 0.0
        num_samples = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (m={current_margin:.3f})")
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            embeddings = backbone(imgs)  # raw embeddings
            loss = arcface(embeddings, labels)

            loss.backward()

            # Gradient clipping (penting untuk ArcFace stability!)
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(arcface.parameters()),
                max_norm=grad_clip_norm,
            )

            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            num_samples += imgs.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        lr_scheduler.step()
        avg_loss = total_loss / max(1, num_samples)
        dt = time.time() - t0

        # Eval: cosine gap
        gap, mean_pos, mean_neg = compute_cosine_gap(backbone, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch+1:2d}/{num_epochs} ({dt:.1f}s) | "
            f"Loss: {avg_loss:.4f} | "
            f"Cosine gap: {gap:.4f} (pos={mean_pos:.3f}, neg={mean_neg:.3f}) | "
            f"LR: {current_lr:.2e}"
        )

        writer.add_scalar("phase2/train_loss", avg_loss, epoch)
        writer.add_scalar("phase2/cosine_gap", gap, epoch)
        writer.add_scalar("phase2/mean_pos_sim", mean_pos, epoch)
        writer.add_scalar("phase2/mean_neg_sim", mean_neg, epoch)
        writer.add_scalar("phase2/learning_rate", current_lr, epoch)
        writer.add_scalar("phase2/margin", current_margin, epoch)

        # Save best
        if gap > best_cosine_gap:
            best_cosine_gap = gap
            best_epoch = epoch + 1
            ckpt_path = output_dir / "checkpoint_phase2_best.pth"
            torch.save({
                "epoch": epoch + 1,
                "backbone_state_dict": backbone.state_dict(),
                "arcface_state_dict": arcface.state_dict(),
                "cosine_gap": gap,
                "mean_pos_sim": mean_pos,
                "mean_neg_sim": mean_neg,
                "num_classes": num_classes,
                "margin": margin,
                "scale": scale,
            }, ckpt_path)
            print(f"    ✅ Best checkpoint saved (gap: {gap:.4f}) → {ckpt_path}")

    writer.close()

    # ====== Gate ======
    print("\n" + "=" * 60)
    print("=== PHASE 2 COMPLETE ===")
    print(f"Best cosine gap: {best_cosine_gap:.4f} (epoch {best_epoch})")
    if best_cosine_gap < PHASE2_TARGET_COSINE_GAP:
        print(f"⚠️  Cosine gap {best_cosine_gap:.3f} < target {PHASE2_TARGET_COSINE_GAP}")
        print(f"   Pertimbangkan tuning:")
        print(f"   - Coba margin = 0.3 atau 0.4 (lebih lembut)")
        print(f"   - Tambah epoch (mis. 50)")
        print(f"   - Cek learning rate (terlalu tinggi → divergent)")
    else:
        print(f"✅ Cosine gap {best_cosine_gap:.3f} >= target {PHASE2_TARGET_COSINE_GAP}")
        print(f"   Siap untuk evaluation final + export.")
    print("=" * 60)

    return best_cosine_gap


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Training (ArcFace Fine-tune)")
    parser.add_argument("--train-csv", type=str, default=str(SPLITS_DIR / "train.csv"))
    parser.add_argument("--val-csv", type=str, default=str(SPLITS_DIR / "val.csv"))
    parser.add_argument("--phase1-checkpoint", type=str,
                        default=str(MODELS_DIR / "checkpoint_phase1_best.pth"))
    parser.add_argument("--output-dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--logs-dir", type=str, default=str(LOGS_DIR))
    parser.add_argument("--epochs", type=int, default=PHASE2_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=PHASE2_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=PHASE2_LR)
    parser.add_argument("--margin", type=float, default=ARCFACE_MARGIN)
    parser.add_argument("--scale", type=float, default=ARCFACE_SCALE)
    parser.add_argument("--margin-warmup-epochs", type=int, default=ARCFACE_MARGIN_WARMUP_EPOCHS)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    train_phase2(
        train_csv=Path(args.train_csv),
        val_csv=Path(args.val_csv),
        phase1_checkpoint=Path(args.phase1_checkpoint),
        output_dir=Path(args.output_dir),
        logs_dir=Path(args.logs_dir),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        margin=args.margin,
        scale=args.scale,
        margin_warmup_epochs=args.margin_warmup_epochs,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

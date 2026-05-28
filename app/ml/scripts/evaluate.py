"""
Evaluation komprehensif untuk model palm biometric.

Protocol: cross-session, open-set
- Test set: 100 individu × 2 tangan = 200 kelas yang BELUM PERNAH dilihat
  saat training (tidak ada di train.csv / val.csv)
- Untuk setiap kelas (individu+hand):
  - Sesi 1: enroll (rata-rata embedding sebagai template)
  - Sesi 2: query (tiap gambar di-test melawan SEMUA template)
- Hitung genuine score (query vs template kelas sama) dan
  impostor score (query vs template kelas berbeda).

Metrics yang dihitung:
- EER (Equal Error Rate) — titik dimana FAR = FRR
- TAR@FAR=0.1% — True Accept Rate saat False Accept Rate 0.1%
- TAR@FAR=0.01% — versi lebih ketat
- ROC AUC
- Rank-1 accuracy (identifikasi)

Output:
- ml/artifacts/eval_metrics.json
- ml/artifacts/figures/roc_curve.png
- ml/artifacts/figures/far_frr_curve.png
- ml/artifacts/figures/score_distributions.png
- ml/artifacts/threshold.json (untuk backend)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_curve, auc

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    SPLITS_DIR,
    MODELS_DIR,
    ARTIFACTS_DIR,
    FIGURES_DIR,
    THRESHOLD_OUTPUT,
    EVAL_FAR_TARGETS,
)
from mobilefacenet import MobileFaceNet
from data_loader import get_val_transform


@torch.no_grad()
def extract_embeddings(model, image_paths, transform, device, batch_size=64):
    """
    Extract embedding untuk list of image path. Batched untuk efisiensi.
    """
    model.eval()
    embeddings = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_tensors = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            img_np = np.array(img)
            try:
                # albumentations style
                t = transform(image=img_np)["image"]
            except (TypeError, KeyError):
                # torchvision style
                t = transform(img)
            batch_tensors.append(t)
        batch = torch.stack(batch_tensors).to(device)
        embs = model(batch)
        embs = F.normalize(embs, dim=1)
        embeddings.append(embs.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def evaluate_biometric(model, test_csv, device, session_split=0.5):
    """
    Cross-session evaluation.

    Args:
        model: trained backbone (output 128-dim embedding)
        test_csv: path ke test.csv dengan kolom 'path', 'label' (kelas string)
        device: torch device
        session_split: fraksi gambar pertama (sorted) yang dianggap sesi 1
                       (enrollment), sisanya sesi 2 (query). Default 0.5.

    Returns:
        dict dengan semua metrics + arrays untuk plotting.
    """
    import pandas as pd

    df = pd.read_csv(test_csv)
    if "label" not in df.columns or "path" not in df.columns:
        raise ValueError("test.csv harus punya kolom 'path' dan 'label'")

    transform = get_val_transform()

    # Group by class label
    class_to_paths = defaultdict(list)
    for _, row in df.iterrows():
        class_to_paths[row["label"]].append(row["path"])

    enroll_paths_all = []
    enroll_class_idx = []  # paralel dengan enroll_paths_all
    query_paths_all = []
    query_class_idx = []

    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(class_to_paths.keys()))}

    for cls, paths in class_to_paths.items():
        paths_sorted = sorted(paths)
        if len(paths_sorted) < 2:
            continue  # butuh minimal 1 enroll + 1 query

        split_idx = max(1, int(len(paths_sorted) * session_split))
        enroll = paths_sorted[:split_idx]
        query = paths_sorted[split_idx:]
        if len(query) == 0:
            query = [paths_sorted[-1]]
            enroll = paths_sorted[:-1]

        cls_idx = class_to_idx[cls]
        enroll_paths_all.extend(enroll)
        enroll_class_idx.extend([cls_idx] * len(enroll))
        query_paths_all.extend(query)
        query_class_idx.extend([cls_idx] * len(query))

    num_classes = len(class_to_idx)
    print(f"Test set: {num_classes} kelas, "
          f"{len(enroll_paths_all)} enroll images, {len(query_paths_all)} query images")

    # Extract embeddings
    print("Extracting enroll embeddings...")
    enroll_embs = extract_embeddings(model, enroll_paths_all, transform, device)
    print("Extracting query embeddings...")
    query_embs = extract_embeddings(model, query_paths_all, transform, device)

    enroll_class_idx = np.array(enroll_class_idx)
    query_class_idx = np.array(query_class_idx)

    # Build enrollment templates: rata-rata embedding per kelas
    templates = np.zeros((num_classes, enroll_embs.shape[1]), dtype=np.float32)
    counts = np.zeros(num_classes, dtype=np.int32)
    for emb, cls in zip(enroll_embs, enroll_class_idx):
        templates[cls] += emb
        counts[cls] += 1
    templates /= counts[:, None].clip(min=1)
    # L2-normalize template setelah averaging
    templates = templates / np.linalg.norm(templates, axis=1, keepdims=True).clip(min=1e-8)

    # Hitung score: query vs ALL templates (matrix mult)
    # Shape: (num_query, num_classes)
    score_matrix = query_embs @ templates.T

    # Genuine vs impostor
    genuine_scores = []
    impostor_scores = []
    for q_idx, true_cls in enumerate(query_class_idx):
        for c in range(num_classes):
            s = float(score_matrix[q_idx, c])
            if c == true_cls:
                genuine_scores.append(s)
            else:
                impostor_scores.append(s)

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    # Rank-1 identification accuracy
    pred_class = score_matrix.argmax(axis=1)
    rank1_correct = (pred_class == query_class_idx).sum()
    rank1_acc = rank1_correct / len(query_class_idx)

    # ROC + EER + TAR@FAR
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    all_labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores)),
    ])

    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)

    fnr = 1 - tpr
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    eer_threshold = float(thresholds[eer_idx])

    tar_at_far = {}
    for far_target in EVAL_FAR_TARGETS:
        idx = int(np.argmin(np.abs(fpr - far_target)))
        tar_at_far[f"tar_at_far_{far_target}"] = float(tpr[idx])

    results = {
        "num_classes": num_classes,
        "num_genuine_pairs": int(len(genuine_scores)),
        "num_impostor_pairs": int(len(impostor_scores)),
        "rank1_accuracy": float(rank1_acc),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "roc_auc": float(roc_auc),
        "mean_genuine_score": float(genuine_scores.mean()),
        "mean_impostor_score": float(impostor_scores.mean()),
        **tar_at_far,
        # untuk plotting
        "_fpr": fpr,
        "_tpr": tpr,
        "_thresholds": thresholds,
        "_genuine_scores": genuine_scores,
        "_impostor_scores": impostor_scores,
    }

    return results


def print_metrics(results):
    print("\n" + "=" * 60)
    print("=== EVALUATION RESULTS ===")
    print(f"Kelas:                 {results['num_classes']}")
    print(f"Genuine pairs:         {results['num_genuine_pairs']:,}")
    print(f"Impostor pairs:        {results['num_impostor_pairs']:,}")
    print()
    print(f"Rank-1 accuracy:       {results['rank1_accuracy']*100:.2f}%")
    print(f"EER:                   {results['eer']*100:.2f}%   (target < 1%)")
    print(f"EER threshold:         {results['eer_threshold']:.4f}")
    print(f"ROC AUC:               {results['roc_auc']:.4f}")
    print()
    for far in EVAL_FAR_TARGETS:
        key = f"tar_at_far_{far}"
        if key in results:
            print(f"TAR @ FAR={far*100:.3f}%:     {results[key]*100:.2f}%")
    print()
    print(f"Mean genuine score:    {results['mean_genuine_score']:.4f}")
    print(f"Mean impostor score:   {results['mean_impostor_score']:.4f}")
    print(f"Gap:                   {results['mean_genuine_score']-results['mean_impostor_score']:.4f}")
    print("=" * 60)


def save_plots(results, output_dir: Path):
    """Save ROC, FAR/FRR, dan score distribution plots."""
    import matplotlib
    matplotlib.use("Agg")  # headless backend
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fpr = results["_fpr"]
    tpr = results["_tpr"]
    thresholds = results["_thresholds"]
    fnr = 1 - tpr

    # 1. ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="navy", lw=2, label=f"ROC (AUC = {results['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    plt.xscale("log")
    plt.xlim([1e-5, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Accept Rate (log scale)")
    plt.ylabel("True Accept Rate")
    plt.title("ROC Curve - Palm Biometric (Cross-Session)")
    plt.legend(loc="lower right")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close()

    # 2. FAR vs FRR
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, fpr, color="red", lw=2, label="FAR (False Accept)")
    plt.plot(thresholds, fnr, color="blue", lw=2, label="FRR (False Reject)")
    plt.axvline(x=results["eer_threshold"], color="green", linestyle="--",
                label=f"EER threshold = {results['eer_threshold']:.3f}")
    plt.axhline(y=results["eer"], color="green", linestyle=":", alpha=0.5,
                label=f"EER = {results['eer']*100:.2f}%")
    plt.xlabel("Threshold (cosine similarity)")
    plt.ylabel("Error rate")
    plt.title("FAR vs FRR")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "far_frr_curve.png", dpi=150)
    plt.close()

    # 3. Score distributions
    plt.figure(figsize=(10, 6))
    plt.hist(results["_genuine_scores"], bins=50, alpha=0.6, color="green",
             label=f"Genuine (n={len(results['_genuine_scores'])})", density=True)
    plt.hist(results["_impostor_scores"], bins=50, alpha=0.6, color="red",
             label=f"Impostor (n={len(results['_impostor_scores'])})", density=True)
    plt.axvline(x=results["eer_threshold"], color="black", linestyle="--",
                label=f"EER threshold = {results['eer_threshold']:.3f}")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.title("Score Distribution: Genuine vs Impostor")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "score_distributions.png", dpi=150)
    plt.close()

    print(f"Plots saved to: {output_dir}")


def save_threshold(results, output_path: Path):
    """Save threshold + summary metrics ke JSON untuk backend consumption."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    threshold_data = {
        "threshold": results["eer_threshold"],
        "eer": results["eer"],
        "rank1_accuracy": results["rank1_accuracy"],
        "roc_auc": results["roc_auc"],
        "model": "mobilefacenet_arcface_tongji",
        "calibrated_on": "tongji_test_holdout",
        "note": (
            "EER point threshold. Lower threshold = more strict (higher security, "
            "lower convenience). Higher threshold = more permissive."
        ),
    }
    for far in EVAL_FAR_TARGETS:
        key = f"tar_at_far_{far}"
        if key in results:
            threshold_data[key] = results[key]

    with open(output_path, "w") as f:
        json.dump(threshold_data, f, indent=2)
    print(f"Threshold saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate palm biometric model")
    parser.add_argument("--checkpoint", type=str,
                        default=str(MODELS_DIR / "checkpoint_phase2_best.pth"))
    parser.add_argument("--test-csv", type=str, default=str(SPLITS_DIR / "test.csv"))
    parser.add_argument("--output-dir", type=str, default=str(ARTIFACTS_DIR))
    parser.add_argument("--figures-dir", type=str, default=str(FIGURES_DIR))
    parser.add_argument("--threshold-out", type=str, default=str(THRESHOLD_OUTPUT))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MobileFaceNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["backbone_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Cosine gap saat training: {ckpt.get('cosine_gap', 'N/A')}")

    results = evaluate_biometric(model, args.test_csv, device)
    print_metrics(results)

    save_plots(results, args.figures_dir)
    save_threshold(results, args.threshold_out)

    # Save full metrics (tanpa array besar)
    metrics_clean = {k: v for k, v in results.items() if not k.startswith("_")}
    output_path = Path(args.output_dir) / "eval_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics_clean, f, indent=2)
    print(f"Full metrics saved: {output_path}")


if __name__ == "__main__":
    main()

"""
Metrics utilities untuk biometric evaluation.

Plot generation:
  - phase1_loss.png, phase1_accuracy.png
  - phase2_loss.png, phase2_cosine_gap.png
  - genuine_impostor_distribution.png
  - roc_curve.png

Semua plot berasal dari actual training logs / evaluation results.
TIDAK ada placeholder graph.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def print_biometric_metrics(results: dict) -> None:
    """Print ringkasan metrics ke stdout."""
    print("\n" + "=" * 60)
    print("=== BIOMETRIC EVALUATION RESULTS ===")
    print(f"Identities:           {results.get('num_classes', 'N/A')}")
    print(f"Genuine pairs:        {results.get('num_genuine_pairs', 0):,}")
    print(f"Impostor pairs:       {results.get('num_impostor_pairs', 0):,}")
    print()
    print(f"Rank-1 accuracy:      {results.get('rank1_accuracy', 0)*100:.2f}%")
    print(f"EER:                  {results.get('eer', 0)*100:.3f}%")
    print(f"EER threshold:        {results.get('eer_threshold', 0):.4f}")
    print(f"ROC AUC:              {results.get('roc_auc', 0):.4f}")
    print()
    for key in ["tar_at_far_0.001", "tar_at_far_0.0001"]:
        if key in results:
            far_pct = float(key.split("_")[-1]) * 100
            print(f"TAR @ FAR={far_pct:.3f}%:   {results[key]*100:.2f}%")
    print()
    print(f"Mean genuine sim:     {results.get('mean_genuine_score', 0):.4f} ± {results.get('std_genuine_score', 0):.4f}")
    print(f"Mean impostor sim:    {results.get('mean_impostor_score', 0):.4f} ± {results.get('std_impostor_score', 0):.4f}")
    print(f"Cosine gap:           {results.get('cosine_gap', 0):.4f}")
    print("=" * 60)


def save_training_plots(
    history_csv_path: str | Path,
    figures_dir: str | Path,
) -> None:
    """Generate training curve plots dari history.csv."""
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib/pandas tidak tersedia, skip plot generation")
        return

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(history_csv_path)
    df1 = df[df["phase"] == 1]
    df2 = df[df["phase"] == 2]

    # Phase 1 Loss
    if len(df1) > 0 and "train_loss" in df1.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df1["epoch"], df1["train_loss"], label="Train Loss", color="steelblue")
        if "val_loss" in df1.columns:
            ax.plot(df1["epoch"], df1["val_loss"], label="Val Loss", color="orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Phase 1 — Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "phase1_loss.png", dpi=150)
        plt.close()

    # Phase 1 Accuracy
    if len(df1) > 0 and "train_accuracy" in df1.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df1["epoch"], df1["train_accuracy"], label="Train Acc", color="steelblue")
        if "val_accuracy" in df1.columns:
            ax.plot(df1["epoch"], df1["val_accuracy"], label="Val Acc", color="orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Phase 1 — Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(figures_dir / "phase1_accuracy.png", dpi=150)
        plt.close()

    # Phase 2 Loss
    if len(df2) > 0 and "train_loss" in df2.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df2["epoch"], df2["train_loss"], label="Train Loss", color="steelblue")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Phase 2 — ArcFace Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "phase2_loss.png", dpi=150)
        plt.close()

    # Phase 2 Cosine Gap
    if len(df2) > 0 and "cosine_gap" in df2.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df2["epoch"], df2["cosine_gap"], label="Cosine Gap", color="green")
        if "mean_positive_cosine" in df2.columns:
            ax.plot(df2["epoch"], df2["mean_positive_cosine"], label="Mean Genuine", color="blue", linestyle="--")
        if "mean_negative_cosine" in df2.columns:
            ax.plot(df2["epoch"], df2["mean_negative_cosine"], label="Mean Impostor", color="red", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Phase 2 — Cosine Gap (Genuine vs Impostor)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "phase2_cosine_gap.png", dpi=150)
        plt.close()

    print(f"Training plots saved to: {figures_dir}")


def save_evaluation_plots(results: dict, figures_dir: str | Path) -> None:
    """Generate evaluation plots dari evaluation results dict."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib tidak tersedia, skip evaluation plot generation")
        return

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fpr = results.get("_fpr")
    tpr = results.get("_tpr")
    thresholds = results.get("_thresholds")
    genuine_scores = results.get("_genuine_scores")
    impostor_scores = results.get("_impostor_scores")

    if fpr is not None and tpr is not None:
        # ROC Curve
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="navy", lw=2,
                label=f"ROC (AUC={results.get('roc_auc', 0):.4f})")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random")
        ax.set_xscale("log")
        ax.set_xlim([1e-5, 1.0])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel("False Accept Rate (log scale)")
        ax.set_ylabel("True Accept Rate")
        ax.set_title("PalmNet-Lite — ROC Curve (Cross-Session Evaluation)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "roc_curve.png", dpi=150)
        plt.close()

        # FAR/FRR
        fnr = 1.0 - tpr
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(thresholds, fpr, color="red", lw=2, label="FAR")
        ax.plot(thresholds, fnr, color="blue", lw=2, label="FRR")
        ax.axvline(x=results.get("eer_threshold", 0), color="green", linestyle="--",
                   label=f"EER threshold={results.get('eer_threshold', 0):.3f}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Error Rate")
        ax.set_title("FAR vs FRR — PalmNet-Lite")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "far_frr_curve.png", dpi=150)
        plt.close()

    if genuine_scores is not None and impostor_scores is not None:
        # Score distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(genuine_scores, bins=50, alpha=0.6, color="green",
                label=f"Genuine (n={len(genuine_scores):,})", density=True)
        ax.hist(impostor_scores, bins=50, alpha=0.6, color="red",
                label=f"Impostor (n={len(impostor_scores):,})", density=True)
        ax.axvline(x=results.get("eer_threshold", 0), color="black", linestyle="--",
                   label=f"EER threshold={results.get('eer_threshold', 0):.3f}")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Density")
        ax.set_title("PalmNet-Lite — Score Distribution: Genuine vs Impostor")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "genuine_impostor_distribution.png", dpi=150)
        plt.close()

    print(f"Evaluation plots saved to: {figures_dir}")

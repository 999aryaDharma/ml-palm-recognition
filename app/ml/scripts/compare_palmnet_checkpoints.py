"""
Compare PalmNet-Lite checkpoints using strictly validation data (val.csv).

Usage:
    python -m scripts.compare_palmnet_checkpoints --run-id 20260813-132816-seed42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import auc, roc_curve

_ML_DIR = Path(__file__).resolve().parent.parent
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

from palm_recognition.data.dataset import PalmDataset, get_val_transform
from palm_recognition.models.palmnet_lite import build_palmnet_lite
from palm_recognition.paths import resolve_ml_path


@torch.no_grad()
def extract_val_embeddings(
    model: torch.nn.Module,
    val_csv: str | Path,
    transform,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract L2-normalized embeddings for validation images."""
    dataset = PalmDataset(val_csv, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    model.eval()
    all_embs = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        embs = model(images)
        embs = F.normalize(embs, p=2, dim=1)
        all_embs.append(embs.cpu().numpy())
        all_labels.extend(labels.tolist())

    return np.concatenate(all_embs, axis=0), np.array(all_labels)


def compute_val_metrics(embs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute validation biometric metrics from embeddings."""
    num_samples = len(embs)
    sim_matrix = embs @ embs.T  # [N, N] cosine similarity matrix

    triu_i, triu_j = np.triu_indices(num_samples, k=1)
    same_label = labels[triu_i] == labels[triu_j]

    genuine_scores = sim_matrix[triu_i[same_label], triu_j[same_label]]
    impostor_scores = sim_matrix[triu_i[~same_label], triu_j[~same_label]]

    mean_genuine = float(np.mean(genuine_scores))
    mean_impostor = float(np.mean(impostor_scores))
    cosine_gap = float(mean_genuine - mean_impostor)

    all_scores = np.concatenate([genuine_scores, impostor_scores])
    all_labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores)),
    ])

    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = float(auc(fpr, tpr))

    fnr = 1.0 - tpr
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    val_eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])

    return {
        "val_eer": val_eer,
        "eer_threshold": eer_threshold,
        "roc_auc": roc_auc,
        "mean_genuine": mean_genuine,
        "mean_impostor": mean_impostor,
        "cosine_gap": cosine_gap,
        "num_genuine_pairs": int(len(genuine_scores)),
        "num_impostor_pairs": int(len(impostor_scores)),
    }


def compare_checkpoints(
    run_id: str,
    config_path: str | Path = "configs/palmnet_lite_scratch.yaml",
) -> dict:
    """Compare Phase 1 Best, Phase 2 Best, Phase 2 Last checkpoints on val.csv."""
    config_file = resolve_ml_path(config_path)
    with open(config_file) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds_cfg = config.get("dataset", {})
    val_csv = resolve_ml_path(ds_cfg.get("val_csv", "data/splits/val.csv"))

    paths_cfg = config.get("paths", {})
    checkpoint_base = resolve_ml_path(
        paths_cfg.get("checkpoint_dir", "checkpoints/palmnet-lite-scratch")
    )
    trained_logs = resolve_ml_path(
        paths_cfg.get("trained_logs_dir", "artifacts/trained_logs/palmnet-lite-scratch")
    )

    ckpt_dir = checkpoint_base / run_id
    run_dir = trained_logs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if not val_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found: {val_csv}")

    checkpoints_to_compare = [
        ("phase1_best", ckpt_dir / "checkpoint_phase1_best.pth"),
        ("phase2_best", ckpt_dir / "checkpoint_phase2_best.pth"),
        ("phase2_last", ckpt_dir / "checkpoint_phase2_last.pth"),
    ]

    transform = get_val_transform()
    results = {}

    for name, path in checkpoints_to_compare:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=device)
        backbone = build_palmnet_lite(config.get("model", {}))
        state_dict = ckpt.get("backbone_state_dict", ckpt)
        backbone.load_state_dict(state_dict)
        backbone.to(device)

        embs, labels = extract_val_embeddings(
            backbone, val_csv, transform, device, batch_size=64
        )
        metrics = compute_val_metrics(embs, labels)
        metrics["checkpoint_path"] = str(path)
        results[name] = metrics

    # Select recommendation: lowest val_eer -> highest roc_auc -> highest cosine_gap
    sorted_candidates = sorted(
        results.keys(),
        key=lambda k: (
            results[k]["val_eer"],
            -results[k]["roc_auc"],
            -results[k]["cosine_gap"],
        ),
    )
    recommended = sorted_candidates[0]

    comparison_summary = {
        "run_id": run_id,
        "evaluation_split": "validation",
        "val_csv": str(val_csv),
        "checkpoints": results,
        "recommended_checkpoint": recommended,
        "recommendation_reason": (
            f"Candidate '{recommended}' achieved the lowest validation EER ({results[recommended]['val_eer']:.6f}) "
            f"with ROC-AUC ({results[recommended]['roc_auc']:.6f}) and cosine gap ({results[recommended]['cosine_gap']:.6f})."
        ),
    }

    # Save JSON summary
    json_path = run_dir / "checkpoint_comparison.json"
    with open(json_path, "w") as f:
        json.dump(comparison_summary, f, indent=2)

    # Save CSV summary
    csv_rows = []
    for name, m in results.items():
        csv_rows.append({
            "checkpoint": name,
            "val_eer": m["val_eer"],
            "eer_threshold": m["eer_threshold"],
            "roc_auc": m["roc_auc"],
            "mean_genuine": m["mean_genuine"],
            "mean_impostor": m["mean_impostor"],
            "cosine_gap": m["cosine_gap"],
            "recommended": name == recommended,
        })
    csv_df = pd.DataFrame(csv_rows)
    csv_path = run_dir / "checkpoint_comparison.csv"
    csv_df.to_csv(csv_path, index=False)

    print(f"\n=== Validation-Only Checkpoint Comparison (Run ID: {run_id}) ===")
    print(f"{'checkpoint':<22} {'val_eer':<9} {'roc_auc':<9} {'genuine':<9} {'impostor':<10} {'gap':<8}")
    print("-" * 72)
    for row in csv_rows:
        print(
            f"{row['checkpoint']:<22} "
            f"{row['val_eer']:<9.6f} "
            f"{row['roc_auc']:<9.6f} "
            f"{row['mean_genuine']:<9.6f} "
            f"{row['mean_impostor']:<10.6f} "
            f"{row['cosine_gap']:<8.6f}"
        )
    print("-" * 72)
    print(f"\nRecommendation: '{recommended}'")
    print(f"Reason: {comparison_summary['recommendation_reason']}\n")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}\n")

    return comparison_summary


def main():
    parser = argparse.ArgumentParser(description="Compare PalmNet-Lite Checkpoints on Validation Set")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID of the training run")
    parser.add_argument(
        "--config", type=str, default="configs/palmnet_lite_scratch.yaml", help="Path to config YAML"
    )
    args = parser.parse_args()

    compare_checkpoints(run_id=args.run_id, config_path=args.config)


if __name__ == "__main__":
    main()

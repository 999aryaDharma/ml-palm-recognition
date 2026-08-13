"""
Evaluate PalmNet-Lite: calibration + final evaluation.

Usage:
    python -m scripts.evaluate_palmnet_lite \\
        --checkpoint checkpoints/palmnet-lite-scratch/checkpoint_phase2_best.pth \\
        --run-id <run_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

_ML_DIR = Path(__file__).resolve().parent.parent
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

from palm_recognition.paths import resolve_ml_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate PalmNet-Lite")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/palmnet-lite-scratch/checkpoint_phase2_best.pth")
    parser.add_argument("--config", type=str, default="configs/palmnet_lite_scratch.yaml")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    config_path = resolve_ml_path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from palm_recognition.models.palmnet_lite import build_palmnet_lite
    from palm_recognition.data.dataset import get_val_transform
    from palm_recognition.evaluation.embeddings import (
        evaluate_biometric_session_aware,
        calibrate_threshold_from_val,
    )
    from palm_recognition.evaluation.metrics import (
        print_biometric_metrics,
        save_evaluation_plots,
    )

    ds_cfg = config.get("dataset", {})
    val_csv  = resolve_ml_path(ds_cfg.get("val_csv",  "data/splits/val.csv"))
    test_csv = resolve_ml_path(ds_cfg.get("test_csv", "data/splits/test.csv"))

    paths_cfg = config.get("paths", {})
    trained_logs = resolve_ml_path(paths_cfg.get("trained_logs_dir", "artifacts/trained_logs/palmnet-lite-scratch"))

    if args.run_id:
        run_dir = trained_logs / args.run_id
        figures_dir = run_dir / "figures"
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = resolve_ml_path("artifacts")
        figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = resolve_ml_path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone = build_palmnet_lite(config.get("model", {}))
    state_dict = ckpt.get("backbone_state_dict", ckpt)
    backbone.load_state_dict(state_dict)
    backbone.to(device)
    backbone.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    transform = get_val_transform()

    # Phase 3: Validation calibration (STRICTLY on val_csv)
    print("\n=== Phase 3: Validation Threshold Calibration ===")
    eval_cfg = config.get("evaluation", {})
    threshold_data = calibrate_threshold_from_val(
        model=backbone,
        val_csv=val_csv,
        transform=transform,
        device=device,
        model_id="palmnet-lite-scratch",
        version=paths_cfg.get("artifact_version", "1.0.0"),
    )
    print(f"Calibrated Threshold (Val EER Point): {threshold_data['threshold']:.4f}")
    print(f"Validation EER:                      {threshold_data['eer']:.4f}")

    thresh_path = run_dir / "threshold.json"
    with open(thresh_path, "w") as f:
        json.dump(threshold_data, f, indent=2)
    print(f"Calibrated threshold saved: {thresh_path}")

    # Phase 4: Final hold-out evaluation on test_csv using FROZEN validation threshold
    print("\n=== Phase 4: Final Hold-out Evaluation ===")
    far_targets = eval_cfg.get("far_targets", [0.001, 0.0001])
    results = evaluate_biometric_session_aware(
        model=backbone,
        test_csv=test_csv,
        transform=transform,
        device=device,
        far_targets=far_targets,
    )

    # Label test EER clearly as diagnostic_test_eer
    results["diagnostic_test_eer"] = results.pop("eer", 0.0)
    results["diagnostic_test_eer_threshold"] = results.pop("eer_threshold", 0.0)
    results["calibrated_validation_threshold"] = threshold_data["threshold"]

    # Compute performance metrics on test set at the FROZEN validation threshold
    gen = results["_genuine_scores"]
    imp = results["_impostor_scores"]
    val_th = threshold_data["threshold"]
    far_val_th = float((imp >= val_th).mean())
    frr_val_th = float((gen < val_th).mean())
    tar_val_th = 1.0 - frr_val_th

    results["far_at_calibrated_threshold"] = far_val_th
    results["frr_at_calibrated_threshold"] = frr_val_th
    results["tar_at_calibrated_threshold"] = tar_val_th

    print_biometric_metrics(results)
    save_evaluation_plots(results, figures_dir)

    clean_metrics = {k: v for k, v in results.items() if not k.startswith("_")}
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(clean_metrics, f, indent=2)
    print(f"\nFinal metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()

"""
Evaluate PalmNet-Lite: calibration + final evaluation.

Usage:
    cd app/ml
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

_ML_DIR = Path(__file__).resolve().parent.parent
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))


def main():
    parser = argparse.ArgumentParser(description="Evaluate PalmNet-Lite")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/palmnet-lite-scratch/checkpoint_phase2_best.pth")
    parser.add_argument("--config", type=str,
                        default="configs/palmnet_lite_scratch.yaml")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
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
    val_csv  = ds_cfg.get("val_csv",  "app/ml/data/splits/val.csv")
    test_csv = ds_cfg.get("test_csv", "app/ml/data/splits/test.csv")

    paths_cfg = config.get("paths", {})
    trained_logs = paths_cfg.get("trained_logs_dir",
                                  "app/ml/artifacts/trained_logs/palmnet-lite-scratch")

    # Tentukan figures dir
    if args.run_id:
        run_dir = Path(trained_logs) / args.run_id
        figures_dir = run_dir / "figures"
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        figures_dir = Path("app/ml/artifacts/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load backbone
    ckpt = torch.load(args.checkpoint, map_location=device)
    backbone = build_palmnet_lite()
    state_dict = ckpt.get("backbone_state_dict", ckpt)
    backbone.load_state_dict(state_dict)
    backbone.to(device)
    backbone.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    transform = get_val_transform()

    # Phase 3: Validation calibration (JANGAN gunakan test set)
    print("\n=== Phase 3: Validation Calibration ===")
    eval_cfg = config.get("evaluation", {})
    threshold_data = calibrate_threshold_from_val(
        model=backbone,
        val_csv=val_csv,
        transform=transform,
        device=device,
        model_id="palmnet-lite-scratch",
        version=paths_cfg.get("artifact_version", "1.0.0"),
    )
    print(f"EER threshold: {threshold_data['threshold']:.4f}")
    print(f"EER:           {threshold_data['eer']:.4f}")

    # Simpan threshold ke run dir jika ada
    if args.run_id:
        thresh_path = run_dir / "threshold.json"
        with open(thresh_path, "w") as f:
            json.dump(threshold_data, f, indent=2)
        print(f"Threshold saved: {thresh_path}")

    # Phase 4: Final hold-out evaluation (test set, session-aware)
    print("\n=== Phase 4: Final Hold-out Evaluation ===")
    far_targets = eval_cfg.get("far_targets", [0.001, 0.0001])
    results = evaluate_biometric_session_aware(
        model=backbone,
        test_csv=test_csv,
        transform=transform,
        device=device,
        far_targets=far_targets,
    )

    print_biometric_metrics(results)
    save_evaluation_plots(results, figures_dir)

    # Simpan metrics
    clean = {k: v for k, v in results.items() if not k.startswith("_")}
    if args.run_id:
        metrics_path = run_dir / "metrics.json"
    else:
        metrics_path = Path("app/ml/artifacts/eval_metrics_palmnet.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    print("\nNext step:")
    version = paths_cfg.get("artifact_version", "1.0.0")
    print(f"  python -m scripts.export_palmnet_lite --checkpoint {args.checkpoint} --version {version}")


if __name__ == "__main__":
    main()

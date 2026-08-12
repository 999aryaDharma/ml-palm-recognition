"""
Export PalmNet-Lite ke TorchScript artifact.

Usage:
    cd app/ml
    python -m scripts.export_palmnet_lite \\
        --checkpoint checkpoints/palmnet-lite-scratch/checkpoint_phase2_best.pth \\
        --threshold path/to/threshold.json \\
        --metrics path/to/metrics.json \\
        --version 1.0.0 \\
        [--deploy-backend]

Output:
    app/backend/ml/models/palmnet-lite-scratch/1.0.0/
        model.pt
        manifest.json
        threshold.json
        metrics.json
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
    parser = argparse.ArgumentParser(description="Export PalmNet-Lite artifact")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/palmnet-lite-scratch/checkpoint_phase2_best.pth")
    parser.add_argument("--version", type=str, default="1.0.0")
    parser.add_argument("--threshold", type=str, default=None,
                        help="Path ke threshold.json (dari evaluation)")
    parser.add_argument("--metrics", type=str, default=None,
                        help="Path ke metrics.json (dari evaluation)")
    parser.add_argument("--deploy-backend", action="store_true",
                        help="Simpan ke backend/ml/models/ directory")
    args = parser.parse_args()

    from palm_recognition.models.palmnet_lite import PalmNetLite, build_palmnet_lite
    from palm_recognition.artifacts.exporter import export_palmnet_lite, verify_artifact

    # Tentukan output directory
    if args.deploy_backend:
        output_dir = Path(f"../backend/ml/models/palmnet-lite-scratch/{args.version}")
    else:
        output_dir = Path(f"artifacts/models/palmnet-lite-scratch/{args.version}")

    # Load threshold jika ada
    threshold_data = None
    if args.threshold and Path(args.threshold).exists():
        with open(args.threshold) as f:
            threshold_data = json.load(f)

    # Load metrics jika ada
    metrics_data = None
    if args.metrics and Path(args.metrics).exists():
        with open(args.metrics) as f:
            metrics_data = json.load(f)

    # Count params
    backbone_temp = build_palmnet_lite()
    param_count = sum(p.numel() for p in backbone_temp.parameters() if p.requires_grad)

    print(f"\n=== Export PalmNet-Lite v{args.version} ===")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output:     {output_dir}")
    print(f"  Params:     {param_count:,}")

    model_pt = export_palmnet_lite(
        backbone_class=lambda: build_palmnet_lite(),
        checkpoint_path=args.checkpoint,
        output_dir=output_dir,
        version=args.version,
        threshold_data=threshold_data,
        metrics_data=metrics_data,
        param_count=param_count,
    )

    # Verify artifact
    print("\n=== Verifying Artifact ===")
    result = verify_artifact(output_dir)
    if result["ok"]:
        print(f"  ✓ model.pt loaded successfully")
        print(f"  ✓ Output shape: {result['output_shape']}")
        print(f"  ✓ L2 norm: {result['l2_norm']}")
    else:
        print(f"  ✗ Verification FAILED: {result['error']}")
        sys.exit(1)

    print(f"\n✓ Artifact siap di: {output_dir}")
    print(f"  Files: model.pt, manifest.json, threshold.json, metrics.json")


if __name__ == "__main__":
    main()

"""
Export PalmNet-Lite ke TorchScript artifact.

Usage:
    python -m scripts.export_palmnet_lite \\
        --checkpoint checkpoints/palmnet-lite-scratch/checkpoint_phase2_best.pth \\
        --threshold artifacts/trained_logs/palmnet-lite-scratch/<run_id>/threshold.json \\
        --metrics artifacts/trained_logs/palmnet-lite-scratch/<run_id>/metrics.json \\
        --version 1.0.0 \\
        [--deploy-backend]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ML_DIR = Path(__file__).resolve().parent.parent
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

from palm_recognition.paths import resolve_ml_path, REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description="Export PalmNet-Lite artifact")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/palmnet-lite-scratch/checkpoint_phase2_best.pth")
    parser.add_argument("--version", type=str, default="1.0.0")
    parser.add_argument("--threshold", type=str, default=None, help="Path ke threshold.json (dari evaluation)")
    parser.add_argument("--metrics", type=str, default=None, help="Path ke metrics.json (dari evaluation)")
    parser.add_argument("--deploy-backend", action="store_true", help="Export ke backend/ml/models/ directory")
    args = parser.parse_args()

    from palm_recognition.models.palmnet_lite import build_palmnet_lite
    from palm_recognition.artifacts.exporter import export_palmnet_lite, verify_artifact

    checkpoint_path = resolve_ml_path(args.checkpoint)

    if args.deploy_backend:
        output_dir = (REPO_ROOT / f"app/backend/ml/models/palmnet-lite-scratch/{args.version}").resolve()
    else:
        output_dir = resolve_ml_path(f"artifacts/models/palmnet-lite-scratch/{args.version}")

    threshold_data = None
    if args.threshold:
        t_path = resolve_ml_path(args.threshold)
        if t_path.exists():
            with open(t_path) as f:
                threshold_data = json.load(f)

    metrics_data = None
    if args.metrics:
        m_path = resolve_ml_path(args.metrics)
        if m_path.exists():
            with open(m_path) as f:
                metrics_data = json.load(f)

    backbone_temp = build_palmnet_lite()
    param_count = sum(p.numel() for p in backbone_temp.parameters() if p.requires_grad)

    print(f"\n=== Export PalmNet-Lite v{args.version} ===")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output:     {output_dir}")
    print(f"  Params:     {param_count:,}")

    model_pt = export_palmnet_lite(
        backbone_class=lambda: build_palmnet_lite(),
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        version=args.version,
        threshold_data=threshold_data,
        metrics_data=metrics_data,
        param_count=param_count,
        deploy_backend=args.deploy_backend,
    )

    print("\n=== Verifying Artifact ===")
    result = verify_artifact(output_dir)
    if result["ok"]:
        print(f"  ✓ model.pt loaded successfully")
        print(f"  ✓ Output shape: {result['output_shape']}")
        print(f"  ✓ L2 norm: {result['l2_norm']}")
    else:
        print(f"  ✗ Verification FAILED: {result['error']}")
        sys.exit(1)

    print(f"\n✓ Artifact ready at: {output_dir}")


if __name__ == "__main__":
    main()

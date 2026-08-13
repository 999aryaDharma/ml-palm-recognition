# PalmNet-Lite Scratch — Training Quickstart

## Prerequisites

- Python environment: `conda activate ML`
- Dataset: `app/ml/data/splits/train.csv`, `val.csv`, `test.csv` sudah ada
- Working directory: `app/ml/` (semua perintah dari sini)

---

## Run Training

```bash
cd app/ml

# Phase 0 + 1 + 2 (full pipeline)
python -m scripts.train_palmnet_lite --config configs/palmnet_lite_scratch.yaml

# Only Phase 1 (Softmax warmup)
python -m scripts.train_palmnet_lite --config configs/palmnet_lite_scratch.yaml --phase 1

# Only Phase 2 (ArcFace, dari checkpoint Phase 1 run-specific)
python -m scripts.train_palmnet_lite --config configs/palmnet_lite_scratch.yaml --phase 2 \
  --phase1-checkpoint checkpoints/palmnet-lite-scratch/<run_id>/checkpoint_phase1_best.pth
```

## Checkpoint Directory Structure

Training output checkpoints are isolated per run directory:
```
checkpoints/palmnet-lite-scratch/<run_id>/
    checkpoint_phase1_best.pth
    checkpoint_phase1_last.pth
    checkpoint_phase2_best.pth
    checkpoint_phase2_last.pth
```

## Evaluate

```bash
python -m scripts.evaluate_palmnet_lite \
  --checkpoint checkpoints/palmnet-lite-scratch/<run_id>/checkpoint_phase2_best.pth \
  --run-id <run_id>
```

## Export ke Backend

```bash
python -m scripts.export_palmnet_lite \
  --checkpoint checkpoints/palmnet-lite-scratch/<run_id>/checkpoint_phase2_best.pth \
  --threshold artifacts/trained_logs/palmnet-lite-scratch/<run_id>/threshold.json \
  --metrics artifacts/trained_logs/palmnet-lite-scratch/<run_id>/metrics.json \
  --version 1.0.0 \
  --deploy-backend
```


---

## Architecture

| Layer            | Output Shape     | Notes                          |
|------------------|-----------------|--------------------------------|
| Input            | [B, 3, 112, 112]| RGB normalized (mean=std=0.5)  |
| Stem Conv3x3 s2  | [B, 32, 56, 56] | ConvBnPrelu                    |
| DW Stem 3x3 s1   | [B, 32, 56, 56] | Depthwise ConvBnPrelu          |
| Stage 1 (2×)     | [B, 32, 56, 56] | InvRes, s=1                    |
| Stage 2 (3×)     | [B, 64, 28, 28] | InvRes, s=2 first              |
| Stage 3 (4×)     | [B, 96, 14, 14] | InvRes, s=2 first              |
| Stage 4 (2×)     | [B, 128, 7, 7]  | InvRes, s=2 first              |
| Final Proj 1x1   | [B, 256, 7, 7]  | ConvBnPrelu                    |
| GDConv 7x7       | [B, 256, 1, 1]  | Global Depthwise Conv          |
| Flatten + FC     | [B, 128]        | Linear(256, 128) + BN1d        |
| (Inference)      | [B, 128]        | + L2 normalize                 |

**Parameter count:** ~393,472 (sanity range 350k-450k)

---

## Training Protocol

**Phase 1 (Softmax Representation Learning):**
- All layers trainable from random init
- CrossEntropyLoss + label smoothing 0.1
- AdamW, LR: 1e-3 → cosine decay
- 40 epochs, batch 64

**Phase 2 (ArcFace Metric Learning):**
- Backbone dari Phase 1 best checkpoint
- ArcFace margin warmup: 0 → 0.30 in 8 epochs
- Scale: 32.0
- AdamW, LR: 1e-4 → cosine decay
- 50 epochs, batch 64

**Best model selection:**
- Phase 1: highest val_accuracy
- Phase 2: highest cosine_gap (mean genuine - mean impostor)

---

## Logs

Setiap run menghasilkan:
```
artifacts/trained_logs/palmnet-lite-scratch/<YYYYMMDD-HHMMSS-seed42>/
  run.json           # run metadata & config
  history.csv        # per-epoch metrics
  notes.md           # debug notes
  phase1_summary.json
  phase2_summary.json
  validation_metrics.json
  metrics.json       # final test set metrics
  figures/           # training curves, ROC, distribution plots
```

---

## Test

```bash
cd app/ml
python tests/test_palmnet.py
# 14 tests: model construction, shape, param count, gradients,
# L2 norm, TorchScript, checkpoint, template avg, artifact verify
```

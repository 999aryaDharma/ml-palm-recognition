# Training, Evaluation, Trained Logs, and Model Artifacts

## 1. Tujuan

Dokumen ini menetapkan pipeline training dan output artifact untuk dua model:

- `mobilefacenet-pretrained`
- `palmnet-lite-scratch`

Keduanya menggunakan pola logging dan artifact yang sama agar evaluasi mudah dibandingkan dan backend dapat memuat model melalui kontrak yang konsisten.

## 2. Prinsip Reproducibility

Setiap training run harus dapat ditelusuri. Minimal informasi yang dicatat:

- `model_id`;
- `model_version` kandidat;
- timestamp;
- random seed;
- dataset split;
- architecture config;
- optimizer;
- learning rate;
- scheduler;
- batch size;
- epoch;
- loss;
- validation metric;
- best epoch;
- checkpoint path;
- final evaluation metrics.

Tidak perlu menggunakan MLflow/W&B untuk scope tugas. File JSON/CSV dan TensorBoard lokal sudah cukup.

## 3. Folder Trained Logs

Target struktur:

```text
app/ml/artifacts/
└── trained_logs/
    ├── mobilefacenet-pretrained/
    │   └── <run_id>/
    │       ├── run.json
    │       ├── history.csv
    │       ├── metrics.json
    │       ├── tensorboard/
    │       └── notes.md
    │
    └── palmnet-lite-scratch/
        └── <run_id>/
            ├── run.json
            ├── history.csv
            ├── metrics.json
            ├── tensorboard/
            └── notes.md
```

Contoh `run_id`:

```text
20260812-221500-seed42
```

## 4. `run.json`

Contoh isi:

```json
{
  "model_id": "palmnet-lite-scratch",
  "run_id": "20260812-221500-seed42",
  "seed": 42,
  "dataset": "Tongji Palmprint",
  "input_size": [3, 112, 112],
  "embedding_dim": 128,
  "training_mode": "scratch",
  "stages": ["softmax", "arcface"],
  "optimizer": "AdamW",
  "initial_lr": 0.001,
  "batch_size": 64
}
```

## 5. `history.csv`

Minimal kolom:

```text
stage,epoch,train_loss,val_loss,val_accuracy,cosine_gap,learning_rate
```

Kolom boleh bertambah selama konsisten.

## 6. PalmNet-Lite Training Pipeline

### Stage A — Softmax Representation Learning

PalmNet-Lite dimulai dari random initialization.

```text
Random Initialization
       -> PalmNet-Lite backbone
       -> temporary linear classifier
       -> Cross Entropy
       -> best scratch checkpoint
```

Semua backbone layer trainable sejak epoch pertama.

Tidak ada freeze-pretrained phase.

Baseline hyperparameter awal:

```text
optimizer     : AdamW
learning rate : 1e-3
weight decay  : 1e-4
batch size    : 32 atau 64
epoch         : 30-50
scheduler     : cosine annealing
```

Nilai aktual harus berasal dari config dan dicatat di trained log.

### Stage B — ArcFace Metric Learning

Stage B memuat checkpoint terbaik Stage A yang dibuat sendiri.

```text
Own Stage-A Checkpoint
       -> PalmNet-Lite backbone
       -> ArcFace head
       -> metric learning
       -> best embedding checkpoint
```

Checkpoint Stage A adalah internal project checkpoint sehingga tidak melanggar ketentuan scratch training.

## 7. MobileFaceNet Training Pipeline

Pipeline existing pretrained tetap dipertahankan dan boleh berbeda dari scratch pipeline.

```text
External pretrained MobileFaceNet
       -> adaptation/fine-tuning
       -> ArcFace
       -> best embedding checkpoint
```

Training logs tetap ditulis ke struktur yang sama, tetapi `training_mode` harus menunjukkan `pretrained` atau `fine_tune` agar provenance jelas.

## 8. Evaluation

Model final harus dievaluasi menggunakan protocol yang sama sebanyak mungkin.

Minimal metric:

- Rank-1 identification accuracy;
- Equal Error Rate (EER);
- ROC-AUC;
- TAR @ FAR 0.1%;
- TAR @ FAR 0.01% jika dataset/protocol memungkinkan;
- mean genuine cosine similarity;
- mean impostor cosine similarity;
- parameter count;
- artifact size;
- optional local inference latency.

Hasil ditulis ke `metrics.json` milik run terkait.

## 9. Threshold

Setiap model mempunyai threshold sendiri.

Threshold tidak boleh menjadi global file yang dipakai semua model.

Contoh:

```json
{
  "metric": "cosine_similarity",
  "threshold": 0.48,
  "calibration": "validation",
  "eer": 0.021
}
```

Nilai threshold harus berasal dari evaluation/calibration model tersebut, bukan disamakan secara manual dengan model lain.

## 10. Artifact Bundle

Artifact yang siap dipakai aplikasi mempunyai struktur:

```text
app/ml/artifacts/models/
├── mobilefacenet-pretrained/
│   └── 1.0.0/
│       ├── model.pt
│       ├── manifest.json
│       ├── threshold.json
│       └── metrics.json
│
└── palmnet-lite-scratch/
    └── 1.0.0/
        ├── model.pt
        ├── manifest.json
        ├── threshold.json
        └── metrics.json
```

`model.pt` adalah inference-only TorchScript.

Classifier head dan ArcFace training head tidak perlu dibawa ke runtime artifact.

## 11. Manifest Contract

Minimal `manifest.json`:

```json
{
  "model_id": "palmnet-lite-scratch",
  "version": "1.0.0",
  "display_name": "PalmNet-Lite (Scratch)",
  "architecture": "PalmNetLite",
  "training_mode": "scratch",
  "format": "torchscript",
  "input": {
    "shape": [3, 112, 112],
    "color_space": "RGB",
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5]
  },
  "output": {
    "type": "embedding",
    "dimension": 128,
    "l2_normalized": true
  },
  "files": {
    "model": "model.pt",
    "threshold": "threshold.json",
    "metrics": "metrics.json"
  }
}
```

## 12. Generic Exporter

Exporter tidak boleh hardcode `MobileFaceNet()`.

Target design:

```text
model factory
    -> load checkpoint berdasarkan model_id
    -> inference wrapper
    -> L2 normalize
    -> torch.jit.script/trace
    -> verification
    -> artifact bundle
```

CLI target:

```bash
python scripts/export_artifact.py \
  --model palmnet-lite-scratch \
  --checkpoint checkpoints/palmnet-lite-scratch/best.pth \
  --version 1.0.0 \
  --deploy-backend
```

## 13. Artifact Verification

Export dianggap valid jika:

- TorchScript dapat di-load ulang;
- dummy input `(1,3,112,112)` diterima;
- output `(1,128)`;
- output tidak mengandung NaN/Inf;
- L2 norm mendekati 1;
- output TorchScript mendekati eager model dalam toleransi numerik;
- manifest cocok dengan runtime output;
- threshold dan metrics tersedia.

## 14. Backend Deployment untuk Local App

Untuk scope lokal, `--deploy-backend` cukup melakukan copy artifact bundle ke:

```text
app/backend/ml/models/
```

Tidak diperlukan artifact server, registry remote, release service, atau object storage.

## 15. Trained Logs vs Runtime Artifact

Keduanya harus dipisahkan secara konseptual:

```text
trained_logs/
    = bukti proses eksperimen dan training

models/<model>/<version>/
    = artifact final untuk aplikasi
```

Jangan menaruh seluruh checkpoint optimizer/training state ke backend. Backend hanya membutuhkan inference artifact, manifest, threshold, dan optional metrics.

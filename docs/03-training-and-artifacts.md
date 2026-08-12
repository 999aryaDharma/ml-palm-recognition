# PalmNet-Lite Training, Evaluation, Trained Logs, and Model Artifacts

## 1. Tujuan

Dokumen ini menetapkan pipeline aktif untuk **PalmNet-Lite Scratch**.

MobileFaceNet tidak termasuk pipeline training baru. Artifact MobileFaceNet yang sudah tersedia diperlakukan sebagai frozen runtime artifact dan hanya dipertahankan agar aplikasi dapat memilih dua model.

Dengan demikian ada dua konsep yang harus dipisahkan:

```text
PalmNet-Lite
= research + training + evaluation + export

MobileFaceNet
= existing inference artifact only
```

## 2. Reproducibility PalmNet-Lite

Setiap training run PalmNet-Lite harus dapat ditelusuri. Minimal informasi yang dicatat:

- `model_id`;
- `run_id`;
- timestamp;
- random seed;
- dataset dan split;
- architecture config;
- weight initialization;
- augmentation config;
- optimizer;
- learning rate;
- scheduler;
- batch size;
- epoch;
- loss;
- validation metric;
- best epoch;
- checkpoint path;
- final evaluation metrics;
- catatan eksperimen jika ada perubahan penting.

Tidak perlu MLflow/W&B. File JSON/CSV dan TensorBoard lokal sudah cukup.

## 3. Trained Logs

Active trained logs hanya dibuat oleh eksperimen PalmNet-Lite.

Target struktur:

```text
app/ml/artifacts/
└── trained_logs/
    └── palmnet-lite-scratch/
        ├── <run_id-1>/
        │   ├── run.json
        │   ├── history.csv
        │   ├── metrics.json
        │   ├── tensorboard/
        │   └── notes.md
        │
        └── <run_id-2>/
            └── ...
```

Contoh `run_id`:

```text
20260812-221500-seed42
```

Jika repository memiliki historical log MobileFaceNet, file tersebut boleh dipertahankan sebagai arsip. Namun pipeline baru **tidak boleh membuat retraining run MobileFaceNet hanya demi menyamakan struktur log**.

Provenance MobileFaceNet cukup disimpan pada manifest/metadata artifact existing.

## 4. `run.json`

Contoh:

```json
{
  "model_id": "palmnet-lite-scratch",
  "run_id": "20260812-221500-seed42",
  "seed": 42,
  "dataset": "Tongji Palmprint",
  "input_size": [3, 112, 112],
  "embedding_dim": 128,
  "initialization": "kaiming_random",
  "training_mode": "scratch",
  "stages": ["softmax", "arcface"],
  "optimizer": "AdamW",
  "initial_lr": 0.001,
  "batch_size": 64
}
```

`run.json` harus menggambarkan config aktual. Nilai di atas hanya contoh.

## 5. `history.csv`

Minimal kolom:

```text
stage,epoch,train_loss,val_loss,val_accuracy,cosine_gap,learning_rate
```

Kolom boleh ditambah sesuai kebutuhan, misalnya margin ArcFace atau training duration.

## 6. PalmNet-Lite Training Pipeline

### Stage A — Softmax Representation Learning

PalmNet-Lite selalu dimulai dari random initialization.

```text
Random Initialization
       -> PalmNet-Lite backbone
       -> temporary linear classifier
       -> Cross Entropy Loss
       -> validation
       -> best Stage-A checkpoint
```

Semua backbone layer trainable sejak epoch pertama.

Tidak ada:

- external checkpoint load;
- freeze pretrained backbone;
- pretrained warm-up;
- transfer learning.

Baseline hyperparameter awal:

```text
optimizer     : AdamW
learning rate : 1e-3
weight decay  : 1e-4
batch size    : 32 atau 64
epoch         : 30-50
scheduler     : cosine annealing
```

Nilai ini harus diperlakukan sebagai starting configuration, bukan hasil final sebelum eksperimen dijalankan.

### Stage B — ArcFace Metric Learning

Jika Stage A telah menghasilkan representation yang layak, checkpoint terbaik Stage A menjadi initialization internal Stage B.

```text
Own PalmNet-Lite Stage-A Checkpoint
       -> PalmNet-Lite backbone
       -> ArcFace head
       -> metric learning
       -> validation embedding metric
       -> best Stage-B checkpoint
```

Stage-A checkpoint tetap memenuhi prinsip scratch karena weight tersebut dihasilkan sendiri dari random initialization pada proyek ini.

Jika hasil eksperimen menunjukkan pipeline lain lebih stabil, perubahan harus dicatat pada trained logs dan `06-report-writing-source.md`.

## 7. MobileFaceNet Existing Artifact

MobileFaceNet tidak dilatih dalam scope tugas.

Yang dilakukan hanya:

- mempertahankan `model.pt`/artifact existing;
- mempertahankan threshold existing jika digunakan runtime;
- menambahkan/merapikan manifest bila diperlukan untuk ModelRegistry;
- melakukan smoke test load/inference saat integrasi;
- tidak mengubah bobot model.

Tidak perlu:

- training script MobileFaceNet baru;
- optimizer config MobileFaceNet;
- new trained log;
- checkpoint selection MobileFaceNet;
- evaluation ulang kecuali suatu saat dibutuhkan untuk eksperimen pembanding.

## 8. Evaluation PalmNet-Lite

PalmNet-Lite final harus dievaluasi menggunakan protocol biometric yang sesuai.

Metric minimum:

- Rank-1 identification accuracy;
- Equal Error Rate (EER);
- ROC-AUC;
- TAR @ FAR 0.1%;
- TAR @ FAR 0.01% jika protocol memungkinkan;
- mean genuine cosine similarity;
- mean impostor cosine similarity;
- genuine-impostor gap;
- parameter count;
- artifact size;
- local inference latency bila berguna untuk laporan.

Hasil ditulis ke `metrics.json` run terkait dan kemudian diringkas ke living report source.

Tidak perlu membuat tabel perbandingan MobileFaceNet kecuali perbandingan benar-benar diputuskan kemudian.

## 9. Threshold PalmNet-Lite

PalmNet-Lite mempunyai threshold hasil calibration sendiri.

Contoh:

```json
{
  "metric": "cosine_similarity",
  "threshold": 0.48,
  "calibration": "validation",
  "eer": 0.021
}
```

Nilai ini hanya contoh.

Aturan metodologis:

- calibration dilakukan pada validation/calibration data;
- final test tidak dipakai untuk tuning threshold;
- threshold PalmNet-Lite tidak disalin dari MobileFaceNet.

## 10. Runtime Artifact Bundle

Target local artifact directory:

```text
app/backend/ml/models/
├── mobilefacenet-pretrained/
│   └── 1.0.0/
│       ├── model.pt             # existing, frozen
│       ├── manifest.json
│       └── threshold.json
│
└── palmnet-lite-scratch/
    └── 1.0.0/
        ├── model.pt             # hasil export tugas
        ├── manifest.json
        ├── threshold.json
        └── metrics.json
```

MobileFaceNet tidak harus memiliki trained logs baru agar bisa berada di registry runtime.

## 11. PalmNet-Lite Manifest Contract

Minimal:

```json
{
  "model_id": "palmnet-lite-scratch",
  "version": "1.0.0",
  "display_name": "PalmNet-Lite (Scratch)",
  "architecture": "PalmNetLite",
  "training_mode": "scratch",
  "initialization": "random",
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

MobileFaceNet existing dapat memiliki manifest lebih sederhana selama memenuhi runtime contract yang dibutuhkan registry.

## 12. Exporter PalmNet-Lite

Exporter bertanggung jawab mengubah best checkpoint menjadi inference-only TorchScript.

Target flow:

```text
PalmNet-Lite best checkpoint
    -> construct PalmNetLite architecture
    -> load own trained weights
    -> inference wrapper
    -> L2 normalize
    -> torch.jit.script/trace
    -> artifact verification
    -> copy artifact bundle to backend
```

Target CLI:

```bash
python scripts/export_artifact.py \
  --model palmnet-lite-scratch \
  --checkpoint checkpoints/palmnet-lite-scratch/best.pth \
  --version 1.0.0 \
  --deploy-backend
```

Exporter tidak perlu mendukung retraining/export-from-checkpoint MobileFaceNet kecuali benar-benar diperlukan. MobileFaceNet existing cukup diregistrasikan dari artifact yang sudah ada.

## 13. Artifact Verification

PalmNet-Lite export dianggap valid jika:

- TorchScript dapat di-load ulang;
- dummy input `(1,3,112,112)` diterima;
- output shape `(1,128)`;
- output tidak mengandung NaN/Inf;
- L2 norm mendekati 1;
- output TorchScript mendekati eager model dalam toleransi numerik;
- manifest sesuai dengan runtime output;
- threshold dan metrics tersedia.

MobileFaceNet existing cukup melalui smoke test runtime:

- artifact dapat di-load;
- menerima shared input contract;
- menghasilkan 128-D embedding;
- tetap berfungsi pada app.

## 14. Trained Logs vs Runtime Artifact

Pemisahan konsep:

```text
trained_logs/palmnet-lite-scratch/
    = bukti proses penelitian dan training PalmNet-Lite

backend/ml/models/mobilefacenet-pretrained/
    = frozen existing runtime artifact

backend/ml/models/palmnet-lite-scratch/
    = final runtime artifact hasil tugas
```

Jangan membuat training data palsu/placeholder untuk MobileFaceNet hanya agar terlihat simetris. Dokumentasi harus mencerminkan provenance yang sebenarnya.

## 15. Living Report Update Rule

Setelah eksperimen penting, informasi berikut harus dipindahkan/diringkas ke `06-report-writing-source.md`:

- konfigurasi run;
- alasan perubahan hyperparameter;
- best epoch;
- training curve summary;
- final metrics;
- model parameter count;
- artifact size;
- inference test;
- hasil integrasi aplikasi;
- keterbatasan yang ditemukan.

Dengan demikian trained logs menjadi sumber bukti teknis, sedangkan `06-report-writing-source.md` menjadi sumber narasi ilmiah yang terus diperbarui.

# Codebase Modularity Plan

## 1. Tujuan

Codebase harus mendukung dua runtime model tanpa menduplikasi pipeline aplikasi, tetapi hanya **PalmNet-Lite** yang menjadi model aktif untuk research/training pada tugas ini.

Prinsip utama:

- PalmNet-Lite source architecture terpisah dari trainer;
- trainer terpisah dari evaluation;
- evaluation terpisah dari runtime backend;
- exporter fokus pada PalmNet-Lite hasil training;
- MobileFaceNet existing diperlakukan sebagai frozen TorchScript artifact;
- backend bersifat architecture-agnostic;
- ModelRegistry mengelola dua artifact runtime;
- trained logs hanya wajib untuk PalmNet-Lite;
- shared utilities digunakan jika behavior memang sama.

## 2. Target Struktur ML

```text
app/ml/
├── configs/
│   └── palmnet_lite_scratch.yaml
│
├── palm_recognition/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── blocks.py
│   │   └── palmnet_lite.py
│   │
│   ├── data/
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   └── splits.py
│   │
│   ├── losses/
│   │   └── arcface.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── softmax_stage.py
│   │   ├── arcface_stage.py
│   │   ├── checkpointing.py
│   │   └── logging.py
│   │
│   ├── evaluation/
│   │   ├── embeddings.py
│   │   ├── identification.py
│   │   ├── verification.py
│   │   └── metrics.py
│   │
│   └── artifacts/
│       ├── exporter.py
│       ├── manifest.py
│       └── verifier.py
│
├── scripts/
│   ├── train_palmnet_lite.py
│   ├── evaluate_palmnet_lite.py
│   └── export_palmnet_lite.py
│
├── checkpoints/
│   └── palmnet-lite-scratch/
│
└── artifacts/
    └── trained_logs/
        └── palmnet-lite-scratch/
```

Tidak perlu membuat config/trainer/checkpoint baru untuk MobileFaceNet karena model tersebut tidak dilatih ulang.

## 3. PalmNet-Lite Architecture Responsibility

`palmnet_lite.py` hanya mendefinisikan network graph.

Contoh:

```python
class PalmNetLite(nn.Module):
    def __init__(...): ...
    def forward(...): ...
```

Architecture class tidak boleh:

- membaca dataset;
- mengetahui filesystem checkpoint default;
- menjalankan optimizer;
- menyimpan trained logs;
- melakukan evaluation;
- meng-copy artifact ke backend.

## 4. Reusable Building Blocks

`blocks.py` berisi primitive reusable untuk PalmNet-Lite, misalnya:

- Conv + BatchNorm + PReLU;
- depthwise convolution;
- pointwise projection;
- inverted residual bottleneck;
- Global Depthwise Convolution block.

Tujuannya agar arsitektur mudah dibaca dan tidak berisi duplikasi kode layer.

## 5. Trainer Responsibility

Trainer generic bertanggung jawab atas mechanics:

- forward;
- loss;
- backward;
- optimizer step;
- scheduler step;
- validation loop;
- checkpoint callback;
- trained-log callback.

Perbedaan Stage A Softmax dan Stage B ArcFace berada pada stage module, bukan banyak `if` di satu loop besar.

## 6. Configuration

Semua hyperparameter PalmNet-Lite berada pada config terpusat.

Contoh:

```yaml
model:
  id: palmnet-lite-scratch
  embedding_dim: 128
  input_channels: 3
  input_size: 112
  initialization: kaiming

training:
  mode: scratch
  batch_size: 64
  seed: 42

softmax:
  epochs: 40
  optimizer: adamw
  lr: 0.001

arcface:
  epochs: 40
  margin: 0.3
  scale: 32
```

Nilai aktual boleh berubah berdasarkan eksperimen. Setiap perubahan penting harus tercermin pada trained log dan living report source.

## 7. Evaluation Module

Evaluation menerima embedding model secara abstrak:

```python
results = evaluate_biometric(
    model=model,
    test_dataset=test_dataset,
    ...
)
```

Module evaluation tidak perlu mengetahui apakah pada runtime app ada MobileFaceNet.

Fokusnya adalah mengevaluasi PalmNet-Lite secara objektif.

## 8. Exporter

Exporter PalmNet-Lite menerima:

```text
checkpoint
model config
version
metrics
threshold
```

Kemudian menghasilkan inference-only TorchScript artifact.

Tidak perlu membuat generic training exporter untuk MobileFaceNet hanya karena aplikasi punya dua model. MobileFaceNet existing cukup ditempatkan pada runtime artifact folder dengan manifest yang sesuai.

## 9. Backend Target Structure

```text
app/backend/ml/
├── detection.py
├── roi.py
├── quality.py
├── matcher.py
│
├── runtime/
│   ├── recognizer.py
│   ├── registry.py
│   └── manifest.py
│
├── cache.py
│
└── models/
    ├── registry.json
    ├── mobilefacenet-pretrained/
    │   └── 1.0.0/
    └── palmnet-lite-scratch/
        └── 1.0.0/
```

Backend tidak mengimpor training package dan tidak perlu mengimpor `PalmNetLite` untuk inference jika `model.pt` sudah TorchScript.

## 10. Frozen MobileFaceNet Boundary

Source/training logic MobileFaceNet existing tidak perlu menjadi bagian dari refactor research baru.

Yang penting untuk runtime:

```text
existing model.pt
+ manifest
+ threshold
```

Jika code lama MobileFaceNet masih ada di repository karena historical reason, jangan biarkan code tersebut menjadi dependency wajib bagi training PalmNet-Lite atau backend runtime.

## 11. Service Responsibilities

### EnrollmentService

- detection;
- ROI;
- memilih model runtime sesuai request/flow;
- menghasilkan embedding;
- quality information.

### IdentificationService

- memilih runtime berdasarkan `model_id`;
- menghasilkan query embedding;
- mengambil template yang kompatibel;
- cosine matching;
- threshold decision;
- response metadata.

### Repository Layer

Persistence/filtering berdasarkan:

```text
user_id
model_id
model_version
```

## 12. Model Registry Responsibility

`ModelRegistry` hanya bagian runtime.

Tanggung jawab:

- membaca local artifact directory;
- membaca manifest;
- load TorchScript;
- expose available model list;
- lookup model berdasarkan ID/version;
- expose threshold dan metadata.

Registry tidak bertanggung jawab atas:

- training;
- checkpoint selection;
- optimizer;
- evaluation eksperimen.

## 13. No DRY Overcorrection

Modular bukan berarti semua hal dibuat generic tanpa kebutuhan.

Contoh keputusan:

- PalmNet-Lite training: modular dan reusable;
- MobileFaceNet training: tidak dibangun ulang karena tidak dibutuhkan;
- TorchScript runtime loading: shared;
- cosine matcher: shared;
- architecture source: hanya diperlukan untuk PalmNet-Lite research/export;
- trained logs: hanya untuk model yang benar-benar dilatih pada tugas.

Ini menghindari overengineering.

## 14. Dependency Direction

```text
PalmNet-Lite scripts
  -> training/evaluation/artifacts
      -> model/data/loss

Backend
  -> runtime registry
      -> exported TorchScript artifacts
```

Backend tidak bergantung pada training code.

MobileFaceNet runtime tidak bergantung pada source training code.

## 15. Testing Strategy

### PalmNet-Lite Model Tests

- forward shape;
- parameter sanity;
- random initialization path;
- no external checkpoint load;
- backward pass smoke test.

### Training Tests

- one mini epoch;
- checkpoint save/load;
- trained log creation;
- Stage A -> Stage B handoff.

### Artifact Tests

- PalmNet-Lite export;
- reload TorchScript;
- output shape 128;
- L2 norm;
- manifest validation.

### Registry Tests

- existing MobileFaceNet discovered;
- PalmNet-Lite discovered setelah artifact tersedia;
- invalid artifact ditolak;
- requested model selected correctly.

### API Tests

- `GET /models`;
- identification menggunakan MobileFaceNet artifact;
- identification menggunakan PalmNet-Lite artifact;
- invalid `model_id`;
- response memuat model metadata.

## 16. Development Order

Urutan implementasi yang disarankan:

1. rapikan shared data/evaluation utilities seperlunya;
2. implement PalmNet-Lite architecture;
3. implement scratch initialization;
4. implement PalmNet-Lite training + trained logs;
5. evaluate dan tune PalmNet-Lite;
6. export PalmNet-Lite TorchScript artifact;
7. buat runtime ModelRegistry;
8. registrasikan MobileFaceNet existing;
9. registrasikan PalmNet-Lite artifact;
10. buat frontend selector dan model-aware API;
11. lakukan local end-to-end test;
12. update `06-report-writing-source.md` dengan hasil aktual.

## 17. Refactor Rule

Refactor hanya dilakukan jika memberi nilai pada tujuan tugas.

Jangan:

- merancang ulang training MobileFaceNet;
- membuat migration architecture production;
- membuat remote registry;
- membuat abstraction untuk skenario hipotetis yang tidak digunakan.

Prioritas utama adalah codebase PalmNet-Lite yang bersih, eksperimen yang dapat direproduksi, artifact yang dapat dipakai aplikasi, dan integrasi dua runtime model yang sederhana.

# Codebase Modularity Plan

## 1. Tujuan

Codebase harus mendukung dua model tanpa menduplikasi seluruh pipeline. Pemisahan dilakukan berdasarkan responsibility, bukan berdasarkan membuat dua aplikasi ML yang terpisah.

Target utama:

- model architecture terpisah dari trainer;
- trainer terpisah dari evaluation;
- evaluation terpisah dari runtime backend;
- artifact exporter generic;
- backend tidak mengimpor source architecture untuk inference TorchScript;
- pilihan model ditentukan melalui registry/factory;
- configuration model-specific;
- trained logs model-specific;
- shared utilities digunakan bersama bila kontraknya sama.

## 2. Target Struktur ML

```text
app/ml/
├── configs/
│   ├── mobilefacenet_pretrained.yaml
│   └── palmnet_lite_scratch.yaml
│
├── palm_recognition/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── blocks.py
│   │   ├── mobilefacenet.py
│   │   ├── palmnet_lite.py
│   │   └── factory.py
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
│   ├── train.py
│   ├── evaluate.py
│   └── export_artifact.py
│
├── checkpoints/
│   ├── mobilefacenet-pretrained/
│   └── palmnet-lite-scratch/
│
└── artifacts/
    ├── trained_logs/
    └── models/
```

Struktur aktual tidak harus dipindahkan sekaligus. Refactor boleh dilakukan bertahap, tetapi dependency direction harus menuju target ini.

## 3. Model Factory

`factory.py` menjadi satu tempat untuk membuat architecture berdasarkan model identifier.

Contoh interface:

```python
def build_model(model_id: str, config) -> nn.Module:
    ...
```

Mapping:

```text
mobilefacenet-pretrained -> MobileFaceNet
palmnet-lite-scratch     -> PalmNetLite
```

Factory membuat architecture. Loading pretrained weight atau scratch initialization dilakukan oleh training orchestration berdasarkan `training_mode`, bukan disembunyikan di dalam architecture class.

Hal ini mencegah PalmNet-Lite secara tidak sengaja memuat external weights.

## 4. Architecture Class Responsibility

Architecture class hanya bertanggung jawab atas network graph.

Contoh:

```python
class PalmNetLite(nn.Module):
    def __init__(...): ...
    def forward(...): ...
```

Class tidak boleh:

- membaca dataset;
- mengetahui filesystem checkpoint default;
- menjalankan optimizer;
- menyimpan TensorBoard log;
- melakukan evaluation;
- copy artifact ke backend.

## 5. Trainer Responsibility

Trainer bertanggung jawab atas generic training mechanics:

- forward;
- loss;
- backward;
- optimizer step;
- scheduler step;
- validation loop;
- checkpoint callback;
- trained-log callback.

Logic spesifik stage ditempatkan pada `softmax_stage.py` dan `arcface_stage.py`.

## 6. Configuration

Hyperparameter tidak boleh tersebar sebagai magic number.

Setiap model mempunyai config terpisah, sementara common settings dapat berada pada config shared/default.

Contoh PalmNet-Lite config:

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

Actual values dapat berubah setelah eksperimen.

## 7. Evaluation Reuse

Evaluation harus menerima abstract embedding model, bukan hardcode MobileFaceNet.

Target:

```python
results = evaluate_biometric(
    model=model,
    test_dataset=test_dataset,
    ...
)
```

Dengan demikian metric code digunakan identik untuk kedua model.

## 8. Exporter Reuse

Exporter menerima:

```text
model_id
checkpoint
model config
version
metrics
threshold
```

Kemudian menghasilkan artifact bundle yang konsisten.

Tidak boleh ada exporter terpisah yang menyalin 90% logic hanya karena architecture berbeda.

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
    └── palmnet-lite-scratch/
```

Backend `recognizer.py` bersifat architecture-agnostic karena hanya memuat TorchScript artifact.

## 10. Service Responsibilities

### EnrollmentService

Bertanggung jawab atas:

- image -> detection;
- ROI extraction;
- model inference;
- quality information;
- menghasilkan embedding untuk persistence.

Tidak bertanggung jawab atas SQL detail.

### IdentificationService

Bertanggung jawab atas:

- memilih runtime berdasarkan request model;
- image -> ROI -> query embedding;
- mengambil model-compatible templates;
- matcher;
- threshold decision;
- response model metadata.

Tidak boleh hardcode `MobileFaceNet` atau `PalmNetLite`.

### Repository Layer

Bertanggung jawab atas persistence dan query filtering berdasarkan:

```text
user_id
model_id
model_version
```

## 11. No DRY Overcorrection

Modular bukan berarti semua perbedaan dipaksa masuk satu function besar dengan banyak `if`.

Shared logic dipakai bersama jika behavior benar-benar sama. Logic yang berbeda secara konseptual boleh terpisah.

Contoh:

- MobileFaceNet pretrained initialization berbeda dari PalmNet-Lite scratch initialization -> pisahkan strategy;
- training epoch loop sama -> reuse Trainer;
- model architecture berbeda -> class terpisah;
- evaluation metric sama -> reuse evaluation module.

## 12. Dependency Direction

Target arah dependency:

```text
scripts
  -> training/evaluation/artifacts
      -> models/data/losses

backend
  -> exported TorchScript artifact
```

Backend tidak seharusnya bergantung pada training package.

## 13. Testing Strategy

Minimal test groups:

### Model Tests

- forward shape PalmNet-Lite;
- random initialization path;
- MobileFaceNet factory;
- parameter sanity.

### Artifact Tests

- export;
- reload TorchScript;
- output shape;
- L2 norm;
- manifest validation.

### Registry Tests

- discover two models;
- reject invalid manifest;
- select requested model;
- unavailable model behavior.

### Matching Tests

- reject/filter mismatched `model_id`;
- reject/filter mismatched `model_version`;
- cosine matcher correctness.

### API Tests

- `GET /models`;
- identify using MobileFaceNet;
- identify using PalmNet-Lite;
- invalid model id;
- response contains model metadata.

## 14. Refactor Rule

Refactor harus dilakukan dengan prinsip:

1. pertahankan behavior existing MobileFaceNet;
2. extract shared abstractions;
3. tambahkan PalmNet-Lite;
4. baru ubah app menjadi model-aware;
5. hapus compatibility code yang tidak lagi dibutuhkan untuk local assignment.

Karena tidak ada kebutuhan migrasi production, jangan mempertahankan struktur lama hanya demi backward compatibility yang tidak memberi nilai pada tugas.

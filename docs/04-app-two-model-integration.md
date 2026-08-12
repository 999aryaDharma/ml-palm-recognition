# Local Application Integration — Two Model Runtime Selection

## 1. Tujuan

Aplikasi lokal harus dapat menggunakan dua model palm recognition melalui pipeline runtime yang sama:

- `mobilefacenet-pretrained` — existing frozen artifact;
- `palmnet-lite-scratch` — model baru hasil tugas.

Fokus pengembangan bukan melatih dua model. Fokusnya adalah membuat satu runtime aplikasi yang dapat memilih artifact mana yang digunakan untuk menghasilkan embedding.

## 2. Scope

Desain hanya untuk penggunaan lokal pada tugas.

Tidak diperlukan:

- distributed serving;
- remote model registry;
- model download service;
- production migration;
- user migration;
- backward compatibility data lama.

SQLite dapat di-reset saat schema berubah.

## 3. Model Registry

Backend menggunakan `ModelRegistry` untuk mendaftarkan artifact lokal.

Target:

```text
backend/ml/models/
├── registry.json
├── mobilefacenet-pretrained/
│   └── 1.0.0/
│       ├── model.pt
│       ├── manifest.json
│       └── threshold.json
└── palmnet-lite-scratch/
    └── 1.0.0/
        ├── model.pt
        ├── manifest.json
        ├── threshold.json
        └── metrics.json
```

MobileFaceNet adalah artifact existing. Registry hanya perlu memuat dan memvalidasi artifact tersebut, bukan mengetahui cara model itu dahulu dilatih.

PalmNet-Lite masuk registry setelah training, evaluation, dan export selesai.

## 4. Shared Runtime Contract

Kedua artifact harus dapat digunakan melalui interface yang sama:

```text
Input  : [B, 3, 112, 112] float32 RGB
Output : [B, 128] float32 L2-normalized embedding
```

Backend inference tidak perlu mengetahui source architecture jika artifact telah berbentuk TorchScript.

## 5. Runtime Abstraction

Target abstraction:

```text
ModelRuntime
├── model_id
├── version
├── manifest
├── recognizer
└── threshold
```

Contoh penggunaan:

```python
runtime = registry.get(model_id)
embedding = runtime.recognizer.extract_embedding(roi)
threshold = runtime.threshold
```

Tidak diperlukan conditional branch berdasarkan class `MobileFaceNet` atau `PalmNetLite` di service layer.

## 6. Model Discovery API

Backend menyediakan:

```text
GET /models
```

Contoh:

```json
{
  "models": [
    {
      "id": "mobilefacenet-pretrained",
      "name": "MobileFaceNet (Existing Pretrained)",
      "version": "1.0.0",
      "source": "existing_artifact"
    },
    {
      "id": "palmnet-lite-scratch",
      "name": "PalmNet-Lite (Scratch)",
      "version": "1.0.0",
      "source": "project_training"
    }
  ]
}
```

Frontend membangun selector dari response ini.

## 7. Selection per Request

Model dipilih per request.

```text
POST /identify
multipart/form-data:
  image=<frame>
  model_id=palmnet-lite-scratch
```

atau:

```text
model_id=mobilefacenet-pretrained
```

Jangan menggunakan mutable global `active_model` karena pilihan model seharusnya menjadi bagian dari request.

## 8. Frontend

Frontend hanya membutuhkan selector sederhana:

```text
Recognition Model
[ MobileFaceNet (Existing)  v ]
[ PalmNet-Lite (Scratch)      ]
```

Frontend tidak memuat TorchScript. Inference tetap dilakukan di FastAPI backend.

Pilihan model dapat disimpan di state/local storage untuk kenyamanan demo.

## 9. Enrollment untuk Scope Tugas

Tidak perlu memikirkan migrasi user lama.

Setelah schema model-aware diterapkan, database demo boleh di-reset lalu enrollment dilakukan kembali.

Implementasi boleh memilih salah satu dari dua pola:

### Opsi sederhana

Enrollment menerima `model_id`, sehingga template dibuat hanya untuk model yang dipilih.

### Opsi demo yang lebih nyaman

Satu capture ROI dijalankan ke kedua model dan menghasilkan dua set embedding.

```text
Palm ROI
  ├── MobileFaceNet -> embedding existing-model
  └── PalmNet-Lite  -> embedding scratch-model
```

Kedua opsi valid untuk scope lokal. Pilih yang paling sederhana saat implementasi tanpa menambah kompleksitas yang tidak memberi nilai pada tugas.

## 10. Template Schema

Minimal:

```text
Template
├── id
├── user_id
├── model_id
├── model_version
├── embedding
├── quality_score
└── captured_at
```

## 11. Embedding Compatibility

Aturan keras:

```text
query.model_id == template.model_id
AND
query.model_version == template.model_version
```

baru similarity boleh dihitung.

Dua embedding 128-D dari model berbeda tidak otomatis berada pada ruang representasi yang sama.

## 12. Model-Aware Cache

Target API:

```python
cache.get_all(model_id, model_version)
cache.get_user(user_id, model_id, model_version)
```

Untuk aplikasi lokal dengan data kecil, cache dapat dibuat sederhana.

## 13. Identification Pipeline

```text
Frontend chooses model
       -> POST /identify + model_id
       -> ModelRegistry.get(model_id)
       -> hand detection
       -> palm ROI
       -> selected TorchScript artifact
       -> 128-D embedding
       -> templates filtered by model_id + version
       -> cosine similarity
       -> selected-model threshold
       -> identified / unknown
```

Response sebaiknya menyertakan model metadata:

```json
{
  "status": "identified",
  "model_id": "palmnet-lite-scratch",
  "model_version": "1.0.0",
  "user_id": 1,
  "user_name": "Example",
  "raw_score": 0.72,
  "threshold": 0.48,
  "latency_ms": 34
}
```

## 14. Threshold

Threshold bersifat model-specific.

```text
MobileFaceNet -> existing/calibrated MobileFaceNet threshold
PalmNet-Lite  -> PalmNet-Lite validation calibration threshold
```

Threshold satu model tidak boleh digunakan untuk model lain.

## 15. Startup

```text
load settings
-> initialize detector/ROI
-> ModelRegistry.discover()
-> load existing MobileFaceNet artifact
-> load PalmNet-Lite artifact jika tersedia
-> initialize model-aware cache
-> start API
```

Selama PalmNet-Lite belum selesai ditraining/export, registry boleh hanya menampilkan MobileFaceNet sebagai available model.

Setelah artifact PalmNet-Lite tersedia, model kedua langsung muncul melalui registry tanpa membuat pipeline service baru.

## 16. Integrasi sebagai Bagian Hasil Tugas

Yang perlu dicatat untuk laporan PalmNet-Lite:

- artifact berhasil di-load backend;
- input/output contract valid;
- model muncul pada `GET /models`;
- frontend dapat memilih PalmNet-Lite;
- enrollment/identification menggunakan model yang dipilih;
- response menunjukkan `model_id`;
- local inference berjalan tanpa error;
- latency dapat dicatat bila relevan.

MobileFaceNet hanya digunakan untuk membuktikan bahwa runtime app memang dapat menampung lebih dari satu artifact, bukan sebagai fokus analisis ilmiah utama.

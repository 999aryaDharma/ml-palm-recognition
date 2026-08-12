# Local Application Integration — Two Model Selection

## 1. Tujuan

Aplikasi lokal harus dapat menggunakan dua model palm recognition melalui pipeline aplikasi yang sama:

- `mobilefacenet-pretrained`
- `palmnet-lite-scratch`

Frontend memilih model. Backend menjalankan model yang dipilih, mengambil threshold model tersebut, dan membandingkan query embedding hanya dengan template yang berasal dari model serta versi yang sama.

## 2. Scope

Desain ini hanya untuk penggunaan lokal pada tugas. Tidak diperlukan distributed serving, dynamic model download, remote registry, atau user migration.

Database dapat di-reset setelah perubahan schema.

## 3. Model Registry

Backend mengganti konsep satu recognizer global menjadi `ModelRegistry`.

Target struktur:

```text
backend/ml/models/
├── registry.json
├── mobilefacenet-pretrained/
│   └── 1.0.0/
│       ├── model.pt
│       ├── manifest.json
│       ├── threshold.json
│       └── metrics.json
└── palmnet-lite-scratch/
    └── 1.0.0/
        ├── model.pt
        ├── manifest.json
        ├── threshold.json
        └── metrics.json
```

ModelRegistry bertanggung jawab atas:

- discovery model artifact lokal;
- validasi manifest;
- load TorchScript;
- penyimpanan runtime metadata;
- lookup berdasarkan `model_id` dan version;
- penyediaan threshold model.

ModelRegistry tidak bertanggung jawab atas training.

## 4. Runtime Abstraction

Target abstraction:

```text
ModelRuntime
├── model_id
├── version
├── manifest
├── recognizer
└── threshold
```

Penggunaan:

```python
runtime = registry.get(model_id)
embedding = runtime.recognizer.extract_embedding(roi)
threshold = runtime.threshold
```

Dengan demikian service tidak perlu mempunyai conditional logic berbasis nama arsitektur.

## 5. API Model Discovery

Backend menyediakan endpoint lokal:

```text
GET /models
```

Contoh response:

```json
{
  "models": [
    {
      "id": "mobilefacenet-pretrained",
      "name": "MobileFaceNet (Pretrained)",
      "version": "1.0.0",
      "training_mode": "pretrained"
    },
    {
      "id": "palmnet-lite-scratch",
      "name": "PalmNet-Lite (Scratch)",
      "version": "1.0.0",
      "training_mode": "scratch"
    }
  ]
}
```

Frontend membangun selector dari response ini agar daftar model tidak di-hardcode di beberapa file.

## 6. Selection per Request

Pemilihan model harus terjadi per request, bukan melalui global mutable active-model state.

Contoh identification request:

```text
POST /identify
multipart/form-data:
  image=<frame>
  model_id=palmnet-lite-scratch
```

Alasan:

- implementasi sederhana;
- service stateless terhadap pilihan model;
- mudah diuji;
- tidak ada risiko global model switch memengaruhi request lain;
- frontend bebas mengganti model kapan saja.

## 7. Frontend Selector

UI cukup menampilkan pilihan sederhana:

```text
Recognition Model
[ MobileFaceNet (Pretrained) v ]

atau

[ PalmNet-Lite (Scratch)      ]
```

Pilihan disimpan pada state halaman/local storage jika dibutuhkan untuk kenyamanan demo.

Frontend tidak memuat `.pt` model. Inference tetap dilakukan FastAPI backend.

## 8. Enrollment Strategy

Untuk scope tugas, desain yang direkomendasikan adalah satu sesi capture dapat menghasilkan template untuk kedua model.

```text
Captured Palm Image
       -> Hand Detection / ROI
       -> same normalized ROI
          |              |
          v              v
 MobileFaceNet       PalmNet-Lite
          |              |
          v              v
 embedding A        embedding B
          |              |
          +------DB------+ 
```

Jika implementasi awal ingin lebih sederhana, endpoint enrollment juga boleh menerima `model_id` dan user melakukan enrollment per-model. Namun target akhir yang lebih nyaman untuk demo adalah menyimpan output kedua model dari capture yang sama.

Tidak perlu menangani migration user lama. Database boleh di-reset sebelum demo dua-model.

## 9. Template Schema

Schema target minimum:

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

`model_id` dan `model_version` adalah bagian penting karena embedding space tidak kompatibel antar model atau antar retraining version.

## 10. Embedding Compatibility Rule

Aturan keras:

```text
query.model_id == template.model_id
AND
query.model_version == template.model_version
```

baru cosine similarity boleh dihitung.

Meskipun kedua model menghasilkan 128 dimensi, nilai embedding tidak berada di ruang representasi yang sama.

## 11. Model-Aware Cache

Embedding cache harus mendukung lookup berdasarkan model identity.

Target API:

```python
cache.get_all(model_id, model_version)
cache.get_user(user_id, model_id, model_version)
```

Boleh juga menggunakan nested structure:

```text
cache[model_id][version][user_id] -> embeddings[]
```

Untuk dataset demo lokal yang kecil, optimisasi memory kompleks tidak diperlukan.

## 12. Identification Pipeline

```text
Frontend model selection
       -> POST /identify + model_id
       -> ModelRegistry.get(model_id)
       -> hand detection
       -> palm ROI
       -> selected model inference
       -> 128-D normalized query embedding
       -> cache filtered by model_id + version
       -> cosine matching
       -> selected model threshold
       -> identified / unknown
       -> response includes model metadata
```

Response sebaiknya memuat:

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

## 13. Threshold Rule

Threshold diambil dari artifact model yang dipilih.

Jangan memakai satu `threshold.json` global.

```text
MobileFaceNet -> threshold MobileFaceNet
PalmNet-Lite -> threshold PalmNet-Lite
```

Confidence display di UI boleh ditambahkan, tetapi scientific metrics dan logs harus tetap menyimpan raw cosine score.

## 14. Startup

Saat FastAPI startup:

```text
load settings
-> initialize detector/ROI dependencies
-> ModelRegistry.discover()
-> load both TorchScript artifacts
-> initialize model-aware embedding cache
-> start API
```

Jika satu artifact belum tersedia saat development, registry boleh menandainya unavailable dan `GET /models` hanya mengembalikan model valid yang berhasil di-load.

## 15. Local-Only Simplification

Karena scope tugas lokal:

- semua model boleh preload ke memory;
- SQLite tetap digunakan;
- model artifacts disimpan di repository/local filesystem;
- tidak perlu model download endpoint;
- tidak perlu artifact signing;
- tidak perlu database migration strategy production;
- tidak perlu permission untuk memilih model;
- database dapat dibuat ulang ketika schema model-aware diperkenalkan.

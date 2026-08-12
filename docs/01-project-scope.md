# Project Scope — Dual-Model Local Palm Recognition

## 1. Latar Belakang

Repository saat ini sudah memiliki ekosistem palm recognition berbasis MobileFaceNet, termasuk preprocessing, training, evaluation, TorchScript export, backend FastAPI, database SQLite, dan frontend lokal. Untuk kebutuhan tugas mata kuliah AI, proyek akan dikembangkan dengan menambahkan satu model baru yang dibangun dari awal tanpa pretrained weights.

Model baru tersebut diberi nama **PalmNet-Lite**. Desainnya terinspirasi prinsip efisiensi MobileFaceNet, tetapi konfigurasi arsitektur, initialization, dan proses training dibuat sendiri untuk domain palm recognition.

## 2. Tujuan Utama

Tujuan proyek adalah membangun aplikasi palm recognition lokal yang dapat menjalankan dua model berbeda melalui pipeline aplikasi yang sama:

- `mobilefacenet-pretrained`
- `palmnet-lite-scratch`

PalmNet-Lite Scratch menjadi model utama untuk memenuhi ketentuan tugas bahwa jaringan saraf tiruan harus dibangun dan dilatih dari awal.

## 3. Definisi Model

### 3.1 MobileFaceNet Pretrained

MobileFaceNet existing tetap dipertahankan sebagai model pembanding.

Karakteristik:

- arsitektur MobileFaceNet;
- menggunakan pretrained weights eksternal sebagai initialization;
- dapat melalui adaptation/fine-tuning pada dataset palm;
- menghasilkan embedding 128 dimensi;
- diekspor menjadi artifact inference TorchScript.

Model ini bukan model yang digunakan untuk memenuhi klaim “trained from scratch”.

### 3.2 PalmNet-Lite Scratch

PalmNet-Lite adalah model baru untuk tugas.

Karakteristik wajib:

- source architecture ditulis sendiri menggunakan primitive PyTorch;
- tidak menggunakan `torchvision.models` atau model pretrained lain sebagai backbone;
- tidak memuat external checkpoint saat initialization;
- semua learnable weights dimulai dari random initialization;
- seluruh backbone dilatih sejak awal;
- menggunakan dataset palm yang sama dengan pipeline eksperimen;
- menghasilkan embedding 128 dimensi;
- diekspor menjadi artifact inference TorchScript.

Checkpoint hasil tahap training sebelumnya dalam proyek sendiri boleh digunakan untuk tahap training berikutnya. Contoh: checkpoint Softmax PalmNet-Lite boleh dipakai sebagai initialization untuk tahap ArcFace karena checkpoint tersebut merupakan hasil training sendiri, bukan pretrained eksternal.

## 4. Scope Implementasi

### In Scope

- PalmNet-Lite architecture;
- scratch weight initialization;
- training pipeline PalmNet-Lite;
- penggunaan kembali MobileFaceNet pretrained existing;
- shared dataset pipeline;
- shared ROI preprocessing;
- Softmax/Cross Entropy representation learning;
- ArcFace metric-learning stage;
- per-model training logs;
- per-model checkpoints;
- per-model metrics;
- per-model threshold;
- generic TorchScript artifact export;
- backend model registry;
- frontend model selector;
- model-aware enrollment;
- model-aware identification;
- local SQLite database;
- model-specific biometric templates;
- local experiment documentation;
- scientific-report reference document.

### Out of Scope

- cloud deployment;
- production migration;
- existing-user migration;
- migration of old embeddings;
- backward-compatible database migrations;
- authentication/authorization redesign;
- production observability;
- load balancing;
- remote model repository;
- artifact CDN;
- model download service;
- browser inference;
- ONNX Runtime Web;
- TensorFlow Lite;
- mobile native deployment;
- production data retention policy.

Jika schema SQLite berubah selama pengerjaan, database dapat dihapus dan dibuat ulang. Ini adalah keputusan scope agar tugas fokus pada AI, model training, evaluasi, dan modularitas codebase.

## 5. Shared Inference Contract

Walaupun arsitektur dan weight berbeda, dua model harus mempunyai interface inference yang konsisten.

### Input

```text
shape       : [B, 3, 112, 112]
color       : RGB
dtype       : float32
normalisasi : mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
```

### Output

```text
shape       : [B, 128]
dtype       : float32
representation: biometric embedding
normalization : L2-normalized
```

Dengan kontrak ini backend tidak perlu mengetahui implementasi internal model ketika melakukan inference.

## 6. Model Identity

Model harus menggunakan identifier eksplisit:

```text
mobilefacenet-pretrained
palmnet-lite-scratch
```

Setiap artifact juga memiliki `model_version`, misalnya `1.0.0`.

Kombinasi berikut menjadi identitas embedding space:

```text
(model_id, model_version)
```

Embedding dari model atau versi yang berbeda tidak boleh dibandingkan satu sama lain.

## 7. Acceptance Criteria

Implementasi dianggap selesai apabila:

- PalmNet-Lite dapat dibuat tanpa memuat pretrained weights;
- training PalmNet-Lite dapat dijalankan end-to-end;
- setiap training run menghasilkan structured trained logs;
- PalmNet-Lite menghasilkan 128-D normalized embedding;
- MobileFaceNet existing tetap dapat digunakan;
- kedua model dapat diekspor ke TorchScript;
- backend dapat menemukan kedua artifact melalui model registry;
- frontend dapat memilih model;
- enrollment dapat menghasilkan template untuk model yang dipilih atau kedua model;
- identification hanya membandingkan embedding dengan template dari model dan versi yang sama;
- threshold dibaca dari artifact model masing-masing;
- evaluasi menghasilkan minimal Rank-1, EER, ROC-AUC, dan TAR@FAR;
- codebase dipisahkan antara model, data, training, evaluation, artifact export, dan runtime integration;
- tersedia satu dokumen khusus sebagai dasar penulisan laporan ilmiah.

## 8. Non-Goal

Tujuan proyek bukan membuktikan bahwa PalmNet-Lite harus selalu mengalahkan MobileFaceNet pretrained. Hasil yang valid dapat menunjukkan PalmNet-Lite lebih baik, setara, atau lebih buruk.

Nilai akademiknya terletak pada:

- perancangan arsitektur sendiri;
- training dari random initialization;
- eksperimen yang dapat direproduksi;
- evaluasi objektif;
- analisis trade-off antara scratch learning dan pretrained transfer learning;
- integrasi dua model pada aplikasi melalui interface yang modular.

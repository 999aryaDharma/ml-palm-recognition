# Project Scope — PalmNet-Lite Scratch with Dual-Model Local Runtime

## 1. Latar Belakang

Repository saat ini sudah memiliki aplikasi palm recognition lokal dengan artifact MobileFaceNet yang telah selesai dilatih dan dapat digunakan untuk inference. Untuk kebutuhan tugas mata kuliah AI, proyek tidak akan melatih ulang MobileFaceNet. Pengembangan difokuskan pada perancangan dan pelatihan satu model baru dari awal tanpa pretrained weights.

Model baru tersebut diberi nama **PalmNet-Lite**. Desainnya mengambil inspirasi dari prinsip lightweight biometric CNN seperti depthwise convolution, pointwise convolution, inverted residual bottleneck, PReLU, dan Global Depthwise Convolution, tetapi konfigurasi network dan seluruh proses pembelajarannya dibuat untuk proyek ini.

Setelah PalmNet-Lite selesai, aplikasi lokal akan dapat menggunakan dua artifact model:

- `mobilefacenet-pretrained` — artifact existing, frozen;
- `palmnet-lite-scratch` — artifact baru hasil tugas.

## 2. Fokus Utama Tugas

Fokus utama tugas adalah:

1. merancang arsitektur PalmNet-Lite;
2. menginisialisasi seluruh parameter PalmNet-Lite secara acak;
3. melatih PalmNet-Lite dari awal menggunakan dataset palm;
4. mengevaluasi kemampuan model menghasilkan biometric embedding;
5. mengekspor model menjadi artifact inference;
6. mengintegrasikan artifact PalmNet-Lite ke aplikasi yang sudah dapat menjalankan MobileFaceNet;
7. menjaga codebase tetap modular sehingga dua model dapat dipilih melalui pipeline runtime yang sama.

MobileFaceNet bukan objek training baru dan untuk sementara bukan objek pembandingan wajib dalam laporan ilmiah.

## 3. Definisi Model

### 3.1 MobileFaceNet Pretrained — Frozen Existing Artifact

Posisi MobileFaceNet dalam tugas ini adalah **runtime model existing**.

Karakteristik:

- artifact sudah tersedia;
- performa existing dianggap cukup untuk kebutuhan demo aplikasi;
- tidak dilakukan retraining;
- tidak dilakukan fine-tuning baru;
- tidak dilakukan hyperparameter tuning baru;
- tidak perlu dibuat ulang training pipeline-nya;
- tetap tersedia pada backend agar dapat dipilih dari frontend;
- metadata artifact, threshold, dan runtime contract tetap dipertahankan.

Jika di masa depan dibutuhkan perbandingan, artifact ini dapat dipakai sebagai pembanding tanpa mengubah scope utama PalmNet-Lite.

### 3.2 PalmNet-Lite Scratch — Model Utama Tugas

PalmNet-Lite adalah model baru yang dirancang dan dilatih untuk tugas.

Karakteristik wajib:

- source architecture ditulis sendiri menggunakan primitive PyTorch;
- tidak menggunakan `torchvision.models` atau pretrained backbone;
- tidak memuat external checkpoint saat initialization;
- semua learnable weights dimulai dari random initialization;
- seluruh backbone dilatih dari awal;
- input inference tetap `3 x 112 x 112` agar kompatibel dengan pipeline aplikasi;
- output berupa embedding 128 dimensi;
- artifact final diekspor ke TorchScript;
- memiliki threshold dan metrics sendiri;
- memiliki trained logs yang dapat ditelusuri.

Checkpoint hasil training PalmNet-Lite sendiri boleh digunakan sebagai initialization untuk tahap berikutnya. Contoh: checkpoint terbaik tahap Softmax boleh menjadi initialization internal untuk tahap ArcFace karena checkpoint tersebut adalah hasil training proyek sendiri.

## 4. Scope Implementasi

### In Scope

- desain PalmNet-Lite;
- scratch weight initialization;
- dataset dan preprocessing pipeline untuk PalmNet-Lite;
- training PalmNet-Lite;
- Softmax/Cross Entropy representation-learning stage;
- ArcFace metric-learning stage jika digunakan sesuai hasil eksperimen;
- validation dan final evaluation PalmNet-Lite;
- trained logs per run PalmNet-Lite;
- checkpoint dan best-model selection;
- model-specific metrics dan threshold;
- TorchScript artifact export PalmNet-Lite;
- mempertahankan MobileFaceNet artifact existing;
- backend model registry untuk dua runtime model;
- frontend model selector;
- model-aware enrollment dan identification;
- local SQLite database;
- codebase modular;
- satu living scientific-report source document.

### Out of Scope

- retraining MobileFaceNet;
- fine-tuning MobileFaceNet;
- hyperparameter optimization MobileFaceNet;
- kewajiban membuat benchmark PalmNet-Lite vs MobileFaceNet;
- cloud deployment;
- production migration;
- existing-user migration;
- migration old embeddings;
- backward-compatible database migration strategy;
- authentication redesign;
- distributed model serving;
- remote artifact registry;
- browser inference;
- mobile native deployment;
- production observability.

Karena scope adalah tugas lokal, database SQLite dapat dihapus dan dibuat ulang jika schema berubah.

## 5. Shared Inference Contract

Dua runtime artifact harus memiliki interface inference yang konsisten agar backend tidak perlu mengetahui detail architecture.

### Input

```text
shape       : [B, 3, 112, 112]
color       : RGB
dtype       : float32
normalisasi : mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
```

### Output

```text
shape          : [B, 128]
dtype          : float32
representation : biometric embedding
normalization  : L2-normalized
```

Kontrak yang sama tidak berarti embedding space kedua model sama.

## 6. Model Identity dan Embedding Space

Identifier runtime:

```text
mobilefacenet-pretrained
palmnet-lite-scratch
```

Artifact juga memiliki `model_version`.

Identitas embedding space ditentukan oleh:

```text
(model_id, model_version)
```

Query embedding hanya boleh dibandingkan dengan template yang berasal dari model dan versi yang sama.

## 7. Scope Laporan Ilmiah

Laporan ilmiah utama mengusulkan **PalmNet-Lite** sebagai model scratch untuk palm recognition.

Isi utama laporan:

- masalah dan tujuan;
- desain arsitektur;
- alasan penggunaan building block;
- dataset dan preprocessing;
- random initialization;
- training pipeline;
- hyperparameter;
- training curves;
- validation dan evaluation metrics;
- keunggulan model;
- keterbatasan model;
- export artifact;
- integrasi ke aplikasi lokal;
- modularitas implementasi.

Untuk sementara laporan **tidak wajib melakukan perbandingan dengan MobileFaceNet**.

Jika kemudian pembandingan dianggap memberi nilai ilmiah, bagian tersebut dapat ditambahkan sebagai eksperimen tambahan, bukan sebagai syarat keberhasilan PalmNet-Lite.

## 8. Acceptance Criteria

Implementasi dianggap selesai apabila:

- PalmNet-Lite dapat diinstansiasi tanpa pretrained weights;
- random initialization dapat diverifikasi;
- training dapat dijalankan end-to-end;
- trained logs mencatat konfigurasi dan perkembangan training setiap run;
- best checkpoint dapat dipilih secara deterministik dari validation metric;
- evaluasi final menghasilkan metric biometric yang relevan;
- PalmNet-Lite menghasilkan embedding 128-D yang L2-normalized saat inference;
- PalmNet-Lite dapat diekspor menjadi TorchScript;
- artifact hasil export dapat di-load backend;
- MobileFaceNet artifact existing tetap berfungsi tanpa retraining;
- backend dapat menyediakan dua model melalui registry/runtime abstraction;
- frontend dapat memilih model;
- matching hanya menggunakan template dari model/version yang kompatibel;
- threshold diambil dari artifact model masing-masing;
- codebase memisahkan architecture, data, training, evaluation, artifact export, dan runtime integration;
- tersedia `06-report-writing-source.md` sebagai living source untuk penulisan laporan.

## 9. Non-Goal

Tujuan tugas **bukan** membuktikan PalmNet-Lite lebih baik dari MobileFaceNet.

Tujuan akademik utamanya adalah menunjukkan bahwa sebuah lightweight biometric network dapat:

- dirancang secara eksplisit;
- dilatih tanpa pretrained weights;
- menghasilkan embedding palm;
- dievaluasi secara objektif;
- diekspor menjadi artifact inference;
- diintegrasikan secara modular ke aplikasi nyata skala lokal.

Keunggulan dan keterbatasan PalmNet-Lite harus disimpulkan berdasarkan hasil eksperimen model itu sendiri. Klaim komparatif terhadap MobileFaceNet hanya dibuat jika eksperimen pembanding benar-benar dilakukan.

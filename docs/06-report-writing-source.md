# Living Report Source — PalmNet-Lite Scratch Palm Recognition

> Dokumen ini adalah **sumber utama penulisan laporan ilmiah** untuk tugas mata kuliah AI. Dokumen ini bukan laporan final dan harus diperbarui secara berkala mengikuti implementasi, training, evaluasi, export artifact, dan integrasi aplikasi.
>
> Angka yang belum berasal dari eksperimen aktual harus ditandai sebagai `TBD` dan tidak boleh ditulis sebagai hasil.

---

## 0. Status Dokumen

**Project:** `ml-palm-recognition`  
**Model utama penelitian:** `PalmNet-Lite Scratch`  
**Scope:** local-only academic assignment  
**Status:** desain / implementasi / training / evaluasi / integrasi — diperbarui sesuai progres  

### Update Checklist

Gunakan checklist ini setiap kali progres berubah:

- [x] arsitektur final PalmNet-Lite sudah sesuai implementasi;
- [x] parameter count aktual sudah dicatat (393,472 params);
- [x] scratch initialization implementation selesai;
- [x] local inference & preprocessing ROI berhasil;
- [x] ModelRegistry berhasil memuat PalmNet-Lite dan MobileFaceNet;
- [x] frontend dapat memilih model per request pada scanner dan enrollment;
- [x] dataset dan split aktual (Phase 0 PASSED: 4000 train, 4000 val, 4000 test images; 400 train identities, 400 val identities, 200 test identities);
- [ ] hyperparameter run terbaik;
- [ ] training curve;
- [ ] best epoch;
- [ ] evaluation metrics;
- [ ] threshold calibration;
- [ ] artifact size final.

---

# 1. Judul Sementara

Judul utama yang disarankan:

**Perancangan PalmNet-Lite sebagai Lightweight Convolutional Neural Network From Scratch untuk Pengenalan Telapak Tangan**

Alternatif:

**Implementasi Jaringan Saraf Konvolusional Ringan dari Awal untuk Palm Recognition Menggunakan PalmNet-Lite**

Judul tidak perlu menyebut MobileFaceNet selama penelitian utama belum melakukan perbandingan formal.

---

# 2. Latar Belakang

Biometrik merupakan pendekatan identifikasi atau verifikasi individu berdasarkan karakteristik biologis. Salah satu karakteristik yang dapat digunakan adalah telapak tangan karena memiliki pola garis utama, lipatan, tekstur, dan struktur lokal yang berbeda antar individu. Sistem palm recognition berbasis deep learning dapat mempelajari representasi fitur secara otomatis dari citra telapak tangan dan mengubahnya menjadi embedding numerik yang dapat digunakan untuk proses pencocokan.

Pada tugas mata kuliah AI ini terdapat ketentuan bahwa model jaringan saraf harus dibangun dari awal dan tidak menggunakan model pretrained sebagai dasar pembelajaran. Oleh karena itu dikembangkan **PalmNet-Lite**, sebuah lightweight convolutional neural network yang seluruh bobotnya diinisialisasi secara acak dan dilatih menggunakan dataset palm. Model dirancang dengan mengadaptasi prinsip arsitektur jaringan ringan seperti depthwise convolution, pointwise convolution, inverted residual bottleneck, dan Global Depthwise Convolution, namun konfigurasi jaringan ditentukan khusus untuk kebutuhan proyek.

Aplikasi existing tetap memiliki artifact MobileFaceNet pretrained sebagai model runtime lain. Keberadaan model tersebut tidak menjadikannya bagian dari proses training PalmNet-Lite. Fokus penelitian sementara adalah merancang, melatih, mengevaluasi, dan mengintegrasikan PalmNet-Lite. Perbandingan formal dengan MobileFaceNet dapat ditambahkan kemudian apabila dibutuhkan.

---

# 3. Rumusan Masalah

Rumusan masalah utama:

1. Bagaimana merancang jaringan CNN ringan untuk menghasilkan representasi embedding telapak tangan tanpa menggunakan pretrained weights?
2. Bagaimana proses training PalmNet-Lite dari random initialization hingga mampu menghasilkan biometric embedding yang diskriminatif?
3. Bagaimana performa PalmNet-Lite pada tugas palm identification dan verification berdasarkan metrik biometric?
4. Bagaimana model hasil training dapat diekspor menjadi artifact dan diintegrasikan secara modular ke aplikasi lokal?

### Rumusan tambahan opsional jika nanti dilakukan perbandingan

Tambahkan hanya jika benar-benar dilakukan:

> Bagaimana performa PalmNet-Lite dibandingkan dengan artifact MobileFaceNet pretrained pada protocol evaluasi yang sama?

Jangan masukkan pertanyaan ini ke laporan final jika eksperimennya tidak dilakukan.

---

# 4. Tujuan

Tujuan proyek:

1. merancang arsitektur PalmNet-Lite untuk palm recognition;
2. melatih seluruh parameter PalmNet-Lite dari random initialization;
3. menghasilkan embedding biometric 128 dimensi;
4. mengevaluasi kemampuan model pada identification dan verification;
5. mengekspor model menjadi inference artifact yang dapat digunakan backend lokal;
6. mengintegrasikan PalmNet-Lite ke aplikasi yang mampu memilih antara PalmNet-Lite dan MobileFaceNet existing;
7. membangun codebase yang modular antara model, data, training, evaluation, artifact, dan runtime aplikasi.

---

# 5. Batasan

Batasan proyek:

- aplikasi dijalankan secara lokal;
- model utama penelitian adalah PalmNet-Lite;
- PalmNet-Lite tidak memuat external pretrained weights;
- MobileFaceNet existing tidak ditraining ulang;
- MobileFaceNet tidak wajib menjadi pembanding penelitian pada tahap awal;
- input PalmNet-Lite berupa ROI telapak berukuran `112 x 112` RGB;
- output berupa embedding `128-D`;
- similarity menggunakan cosine similarity;
- dataset, split, preprocessing, dan protocol evaluasi mengikuti implementasi aktual yang dicatat pada dokumen ini;
- tidak membahas production deployment;
- tidak membahas migrasi user existing;
- SQLite lokal dapat di-reset;
- generalisasi hasil dibatasi pada dataset dan protocol eksperimen yang digunakan.

---

# 6. Konsep Model yang Diusulkan

## 6.1 PalmNet-Lite

PalmNet-Lite adalah lightweight CNN yang dirancang sebagai feature extractor untuk menghasilkan biometric embedding telapak tangan.

Karakteristik utama:

- dibuat menggunakan primitive PyTorch;
- random initialization;
- tidak menggunakan pretrained backbone;
- depthwise convolution;
- pointwise `1x1` convolution;
- inverted residual bottleneck;
- PReLU activation;
- Global Depthwise Convolution;
- embedding 128 dimensi;
- L2 normalization saat inference.

## 6.2 Posisi MobileFaceNet Existing

Aplikasi memiliki artifact MobileFaceNet pretrained yang sudah selesai dan tidak termasuk pipeline training baru.

Posisinya dalam proyek:

```text
MobileFaceNet existing
= runtime model option

PalmNet-Lite
= model penelitian + training + evaluation + integration
```

Untuk sementara laporan tidak perlu membandingkan hasil keduanya.

---

# 7. Arsitektur PalmNet-Lite

## 7.1 Desain Konseptual

Desain awal:

```text
Input 3x112x112
    -> Conv Stem 3x3, C=32, stride=2
    -> Depthwise Stem, C=32
    -> Inverted Residual Stage, C=32, repeat=2
    -> Inverted Residual Stage, C=64, repeat=3, downsample
    -> Inverted Residual Stage, C=96, repeat=4, downsample
    -> Inverted Residual Stage, C=128, repeat=2, downsample
    -> Conv 1x1 Projection, C=256
    -> Global Depthwise Conv 7x7
    -> Linear Projection
    -> 128-D Embedding
    -> L2 Normalization
```

**Penting:** tabel final laporan harus mengikuti source code final. Jika konfigurasi berubah saat tuning, bagian ini wajib diperbarui.

## 7.2 Tabel Arsitektur

| Stage | Operator | Output Shape | Repeat | Stride |
|---|---|---|---:|---:|
| Input | RGB image | 3x112x112 | - | - |
| Stem | Conv 3x3 + BN + PReLU | 32x56x56 | 1 | 2 |
| DW Stem | Depthwise Conv | 32x56x56 | 1 | 1 |
| Stage 1 | Inverted Residual | 32x56x56 | 2 | 1 |
| Stage 2 | Inverted Residual | 64x28x28 | 3 | 2 |
| Stage 3 | Inverted Residual | 96x14x14 | 4 | 2 |
| Stage 4 | Inverted Residual | 128x7x7 | 2 | 2 |
| Projection | Conv 1x1 | 256x7x7 | 1 | 1 |
| Spatial | GDConv 7x7 | 256x1x1 | 1 | 1 |
| Embedding | Linear/Projection | 128 | 1 | - |

### Nilai Aktual Setelah Implementasi

```text
Total parameters : 393,472
Trainable params : 393,472
FP32 model size  : ~1.5 MB
TorchScript size : TBD (setelah full training & export)
```


---

# 8. Alasan Pemilihan Komponen Arsitektur

## 8.1 Depthwise Convolution

Depthwise convolution menjalankan operasi spatial convolution secara terpisah pada setiap channel. Dibanding regular convolution, pendekatan ini dapat mengurangi jumlah parameter dan komputasi.

Alasan digunakan:

- menjaga model tetap ringan;
- cocok untuk local inference;
- tetap mempelajari pola spatial lokal pada telapak;
- memungkinkan jaringan memiliki beberapa stage tanpa parameter terlalu besar.

Tidak boleh menyatakan depthwise convolution otomatis memberikan akurasi lebih tinggi. Keunggulan utamanya dalam desain ini adalah efisiensi parameter.

## 8.2 Pointwise Convolution

Convolution `1x1` digunakan untuk menggabungkan informasi antar-channel setelah depthwise operation dan mengatur dimensionality feature.

## 8.3 Inverted Residual Bottleneck

Block terdiri dari:

```text
1x1 expansion
 -> 3x3 depthwise convolution
 -> 1x1 linear projection
 -> optional residual connection
```

Alasan:

- memberikan intermediate feature space yang lebih besar;
- menjaga input/output channel relatif kecil;
- residual connection membantu aliran gradient;
- efisien untuk lightweight network.

## 8.4 PReLU

PReLU mempunyai parameter slope yang dapat dipelajari untuk activation negatif. Digunakan agar jaringan tidak selalu membuang seluruh respon negatif seperti pada ReLU standar.

## 8.5 Global Depthwise Convolution

GDConv mempelajari spatial weighting pada feature map terakhir sebelum embedding dibentuk.

Alasan desain:

- feature pada lokasi spatial berbeda tidak harus dianggap memiliki kontribusi identik;
- bobot spatial dipelajari;
- operasi tetap efisien karena dilakukan per-channel.

Klaim bahwa GDConv lebih baik dari Global Average Pooling hanya boleh dibuat jika nantinya dilakukan ablation experiment.

## 8.6 Embedding 128 Dimensi

Embedding 128-D dipilih sebagai representasi yang cukup kompak untuk pencocokan biometric sekaligus konsisten dengan kontrak runtime aplikasi existing.

Pemilihan 128 dimensi tidak berarti embedding PalmNet-Lite kompatibel dengan embedding MobileFaceNet.

---

# 9. Random Initialization

PalmNet-Lite dimulai dari bobot acak, misalnya menggunakan Kaiming initialization untuk convolution layer.

```text
random initialization
 -> forward pass
 -> calculate loss
 -> backpropagation
 -> optimizer update
 -> repeat for each epoch
```

Tidak ada knowledge parameter yang diambil dari model eksternal.

### Implementasi Aktual

```text
Initialization method : TBD setelah source final
Random seed           : TBD
```

---

# 10. Dataset dan Data Protocol

## 10.1 Dataset

```text
Dataset name        : Tongji Palmprint / UPDATE JIKA BERUBAH
Total subjects      : TBD berdasarkan dataset final
Total palm classes  : TBD
Total images        : TBD
Session structure   : TBD
```

## 10.2 Split

Catat split aktual setelah implementation audit:

```text
Training   : TBD
Validation : TBD
Testing    : TBD
```

Identity leakage antara train dan open-set test harus dihindari sesuai protocol yang dipilih.

## 10.3 Preprocessing

Pipeline aktual:

```text
Raw image
 -> hand detection / landmark
 -> palm ROI extraction
 -> orientation normalization jika digunakan
 -> resize 112x112
 -> RGB conversion
 -> normalization
 -> tensor
```

Catatan metodologis: jika MediaPipe Hand Landmarker tetap digunakan untuk ROI, laporan harus membedakan dengan jelas bahwa pretrained component tersebut digunakan pada **preprocessing/detection**, sedangkan **PalmNet-Lite recognition network** dilatih from scratch. Jangan membuat klaim bahwa seluruh sistem bebas pretrained apabila itu tidak benar.

## 10.4 Augmentation

Isi setelah config final:

```text
Rotation      : TBD
Crop/Scale    : TBD
Brightness    : TBD
Contrast      : TBD
Blur          : TBD
Noise         : TBD
Horizontal flip: tidak digunakan jika kiri/kanan berbeda identity
```

---

# 11. Training Methodology

## 11.1 Stage A — Softmax Representation Learning

```text
PalmNet-Lite Random Init
 -> backbone
 -> 128-D feature
 -> temporary classifier
 -> Cross Entropy Loss
```

Tujuan Stage A adalah membentuk representation awal yang mampu membedakan kelas training.

Seluruh backbone trainable sejak awal.

### Hyperparameter Run Terbaik

| Parameter | Value |
|---|---|
| Seed | TBD |
| Epoch | TBD |
| Batch size | TBD |
| Optimizer | TBD |
| Initial learning rate | TBD |
| Weight decay | TBD |
| Scheduler | TBD |
| Label smoothing | TBD |
| Best epoch | TBD |
| Best validation accuracy | TBD |

## 11.2 Stage B — ArcFace Metric Learning

Checkpoint terbaik Stage A dapat digunakan sebagai own-project initialization untuk Stage B.

```text
Own PalmNet-Lite checkpoint
 -> backbone
 -> 128-D embedding
 -> ArcFace head
 -> angular metric learning
```

ArcFace head hanya digunakan saat training dan tidak dibawa ke inference artifact.

### Hyperparameter Run Terbaik

| Parameter | Value |
|---|---|
| Epoch | TBD |
| Batch size | TBD |
| Optimizer | TBD |
| Learning rate | TBD |
| Margin | TBD |
| Scale | TBD |
| Margin warm-up | TBD |
| Best epoch | TBD |
| Best validation embedding metric | TBD |

---

# 12. Trained Logs dan Reproducibility

Semua run PalmNet-Lite disimpan pada:

```text
app/ml/artifacts/trained_logs/
└── palmnet-lite-scratch/
    └── <run_id>/
        ├── run.json
        ├── history.csv
        ├── metrics.json
        ├── tensorboard/
        └── notes.md
```

Trained logs berfungsi sebagai evidence untuk:

- convergence;
- pemilihan best checkpoint;
- perubahan hyperparameter;
- reproducibility;
- pembuatan grafik laporan.

### Run yang Dipilih untuk Laporan

```text
Selected run_id : TBD
Reason          : TBD
Checkpoint      : TBD
```

---

# 13. Evaluation Protocol

## 13.1 Identification

Query embedding dibandingkan dengan enrollment templates dan identity dengan similarity tertinggi menjadi kandidat hasil identifikasi.

Metric:

**Rank-1 Accuracy**.

## 13.2 Verification

Cosine similarity:

```text
similarity = cosine(query_embedding, reference_embedding)
```

Metric:

- FAR;
- FRR;
- EER;
- ROC-AUC;
- TAR @ FAR target.

## 13.3 Genuine dan Impostor Distribution

Genuine score berasal dari pasangan identity yang sama. Impostor score berasal dari identity berbeda.

Separation kedua distribusi membantu melihat discriminative quality embedding.

---

# 14. Hasil Training dan Evaluasi

> Bagian ini hanya boleh diisi dengan hasil yang benar-benar dihasilkan program.

## 14.1 Training Result

```text
Stage A best epoch        : TBD
Stage A best val accuracy : TBD
Stage B best epoch        : TBD
Stage B best cosine gap   : TBD
Training duration         : TBD
```

## 14.2 Final Metrics

| Metric | PalmNet-Lite |
|---|---:|
| Parameter Count | TBD |
| TorchScript Size | TBD |
| Rank-1 Accuracy | TBD |
| EER | TBD |
| ROC-AUC | TBD |
| TAR @ FAR 0.1% | TBD |
| TAR @ FAR 0.01% | TBD |
| Mean Genuine Similarity | TBD |
| Mean Impostor Similarity | TBD |
| Genuine-Impostor Gap | TBD |
| Local Inference Latency | TBD / optional |

## 14.3 Grafik untuk Laporan

Minimal yang sebaiknya dihasilkan:

1. training loss per epoch;
2. validation loss/accuracy Stage A;
3. ArcFace/embedding metric progress Stage B;
4. ROC curve;
5. genuine vs impostor distribution;
6. FAR vs FRR curve.

Masukkan path artifact aktual setelah tersedia:

```text
training curve          : TBD
roc curve               : TBD
score distribution      : TBD
far/frr curve           : TBD
```

---

# 15. Threshold Calibration

Threshold keputusan tidak boleh dipilih hanya karena terlihat cocok pada demo.

Calibration dilakukan menggunakan validation/calibration data.

```text
Calibration source : TBD
Threshold          : TBD
Criterion          : TBD (EER / target FAR / lainnya)
```

Final test digunakan untuk mengukur performa, bukan tuning threshold.

---

# 16. Export Artifact

Best PalmNet-Lite checkpoint diekspor menjadi TorchScript inference artifact.

```text
checkpoint
 -> PalmNetLite architecture
 -> load own weights
 -> inference wrapper
 -> L2 normalization
 -> TorchScript
 -> verification
 -> backend artifact directory
```

Artifact bundle:

```text
palmnet-lite-scratch/
└── 1.0.0/
    ├── model.pt
    ├── manifest.json
    ├── threshold.json
    └── metrics.json
```

### Artifact Aktual

```text
Version          : TBD
Path             : TBD
TorchScript size : TBD
Verification     : TBD
```

---

# 17. Pipeline Aplikasi

```text
Camera / image
 -> hand detection
 -> ROI extraction
 -> preprocessing
 -> selected runtime model
 -> 128-D embedding
 -> model-compatible template lookup
 -> cosine similarity
 -> model-specific threshold
 -> identified / unknown
```

Aplikasi memiliki dua runtime option:

```text
MobileFaceNet Existing Artifact
PalmNet-Lite Scratch Artifact
```

Frontend mengirim `model_id`. Backend memilih artifact melalui ModelRegistry.

---

# 18. Mengapa Model Registry Digunakan

Sebelumnya backend hanya berorientasi pada satu `palm_recognizer.pt`. Dengan dua model, hardcoded single-model path akan meningkatkan coupling.

ModelRegistry menyediakan abstraction:

```text
model_id
 -> manifest
 -> TorchScript model
 -> threshold
 -> runtime recognizer
```

Keuntungan:

- backend service tidak perlu mengetahui class arsitektur;
- MobileFaceNet existing tidak perlu source training untuk inference;
- PalmNet-Lite dapat diintegrasikan setelah export;
- frontend dapat menampilkan daftar model;
- threshold terikat pada model;
- runtime code tidak diduplikasi.

---

# 19. Model-Aware Embedding

Walaupun MobileFaceNet dan PalmNet-Lite sama-sama menghasilkan 128-D vector, embedding space keduanya berbeda.

Similarity hanya boleh dihitung jika:

```text
query.model_id == template.model_id
AND
query.model_version == template.model_version
```

Ini merupakan requirement integrasi aplikasi, bukan topik utama perbandingan model.

---

# 20. Codebase Modularity

Target pemisahan responsibility:

```text
models/
    architecture

data/
    dataset + transforms + split

training/
    optimization + checkpoint + logs

evaluation/
    biometric metrics

artifacts/
    export + manifest + verification

backend/runtime/
    model registry + TorchScript loading

frontend/
    model selection + interaction
```

Tujuan modularitas:

- menghindari coupling training dan runtime;
- memudahkan pengujian;
- menghindari duplikasi;
- mempermudah audit bahwa PalmNet-Lite benar-benar scratch;
- membuat artifact hasil penelitian langsung dapat digunakan aplikasi.

---

# 21. Hasil Integrasi Aplikasi

Isi setelah selesai:

| Integration Check | Result |
|---|---|
| PalmNet-Lite TorchScript load | TBD |
| `GET /models` menampilkan PalmNet-Lite | TBD |
| Frontend selector tersedia | TBD |
| Enrollment dengan PalmNet-Lite | TBD |
| Identification dengan PalmNet-Lite | TBD |
| Raw cosine score tersedia | TBD |
| Model-specific threshold digunakan | TBD |
| End-to-end local demo | TBD |

Catat juga kendala yang ditemukan:

```text
TBD
```

---

# 22. Keunggulan Model yang Diusulkan

Keunggulan yang dapat dikemukakan sebagai **desain**, sebelum hasil tersedia:

1. seluruh learnable weight recognition model dilatih dari awal;
2. arsitektur lightweight menggunakan depthwise operation;
3. embedding relatif kompak (128-D);
4. inference artifact dapat digunakan secara lokal;
5. arsitektur disusun modular;
6. training provenance dapat ditelusuri melalui trained logs;
7. model tidak bergantung pada pretrained recognition backbone.

Keunggulan performa seperti "lebih akurat", "lebih cepat", atau "lebih baik dari MobileFaceNet" tidak boleh diklaim sebelum ada data.

---

# 23. Keterbatasan

Keterbatasan desain/eksperimen yang kemungkinan relevan:

- scratch model sangat bergantung pada jumlah dan variasi training data;
- model kecil dapat memiliki capacity lebih rendah;
- generalisasi terhadap kamera, background, lighting, dan pose dunia nyata belum tentu sama dengan dataset;
- hyperparameter search terbatas oleh resource/waktu tugas;
- preprocessing ROI dapat menjadi sumber error terpisah dari recognition model;
- jika MediaPipe digunakan, keseluruhan sistem bukan sepenuhnya bebas pretrained component walaupun PalmNet-Lite sendiri scratch;
- evaluation hanya merepresentasikan dataset/protocol yang digunakan;
- local demo bukan validasi production biometric system.

Tambahkan keterbatasan aktual setelah eksperimen.

---

# 24. Pembahasan Hasil

Gunakan struktur berikut setelah metrics tersedia:

### 24.1 Convergence

Bahas:

- apakah loss turun stabil;
- apakah terdapat overfitting;
- epoch terbaik;
- dampak Stage A terhadap Stage B;
- perubahan yang dilakukan selama eksperimen.

### 24.2 Embedding Quality

Bahas:

- genuine mean;
- impostor mean;
- separation/gap;
- ROC;
- EER;
- TAR pada FAR tertentu.

### 24.3 Model Complexity

Bahas:

- parameter count;
- model size;
- inference latency jika diukur;
- trade-off complexity dan performance.

### 24.4 Integrasi Aplikasi

Bahas bahwa model tidak hanya berhenti sebagai checkpoint training, tetapi diekspor dan digunakan pada full local recognition pipeline.

---

# 25. Opsional — Perbandingan dengan MobileFaceNet

**Bagian ini tidak digunakan untuk sementara.**

Jika nantinya dosen meminta atau penelitian ingin diperluas, tambahkan eksperimen perbandingan yang fair dengan artifact MobileFaceNet existing menggunakan dataset/protocol test yang sama.

Tabel opsional:

| Metric | PalmNet-Lite Scratch | MobileFaceNet Existing |
|---|---:|---:|
| Parameters | TBD | TBD |
| Artifact Size | TBD | TBD |
| Rank-1 | TBD | TBD |
| EER | TBD | TBD |
| ROC-AUC | TBD | TBD |
| TAR@FAR | TBD | TBD |
| Latency | TBD | TBD |

Jika tidak dilakukan, hapus bagian perbandingan dari laporan final.

---

# 26. Kesimpulan — Template Sementara

Jangan gunakan sebagai kesimpulan final sebelum metrics tersedia.

> Proyek merancang PalmNet-Lite sebagai lightweight convolutional neural network untuk palm recognition yang dilatih dari random initialization tanpa menggunakan pretrained recognition weights. Arsitektur menggunakan depthwise convolution, inverted residual bottleneck, Global Depthwise Convolution, dan menghasilkan embedding 128 dimensi. Model dilatih melalui pipeline yang terdokumentasi menggunakan trained logs, dievaluasi menggunakan metrik biometric, kemudian diekspor menjadi TorchScript artifact dan diintegrasikan ke aplikasi lokal melalui ModelRegistry. Hasil kuantitatif dan keterbatasan model disimpulkan berdasarkan eksperimen yang dilakukan.

Setelah eksperimen selesai, template ini harus diubah berdasarkan hasil aktual dan tidak boleh menyatakan keberhasilan performa yang tidak didukung data.

---

# 27. Catatan Perubahan Penelitian

Gunakan bagian ini untuk mencatat keputusan penting agar alasan perubahan tidak hilang.

## Entry Template

```text
Date:
Run/Commit:
Change:
Reason:
Observed result:
Decision:
Report sections affected:
```

### Entries

Belum ada hasil eksperimen baru setelah dokumen ini dibuat.

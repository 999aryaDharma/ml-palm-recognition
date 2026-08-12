# Scientific Report Reference — PalmNet-Lite Scratch vs MobileFaceNet Pretrained

Dokumen ini menjadi acuan utama untuk penulisan laporan ilmiah tugas mata kuliah AI. Isinya bukan laporan final, tetapi kerangka argumentasi, metodologi, arsitektur, pipeline, alasan desain, metrik, dan batas klaim yang harus konsisten dengan implementasi.

## 1. Judul Sementara

Contoh judul:

**Perancangan Lightweight Convolutional Neural Network PalmNet-Lite dari Awal untuk Pengenalan Telapak Tangan dan Perbandingannya dengan MobileFaceNet Pretrained**

Alternatif lebih ringkas:

**PalmNet-Lite: Perancangan CNN Ringan From Scratch untuk Palm Recognition**

## 2. Latar Belakang

Sistem biometric recognition membutuhkan representasi feature yang mampu membedakan identitas berdasarkan karakteristik biologis. Pada palm recognition, informasi diskriminatif dapat berasal dari pola garis utama telapak, tekstur, lipatan, dan struktur lokal lainnya.

Model pretrained dapat mempercepat transfer pengetahuan, tetapi penggunaan model pretrained tidak menunjukkan secara langsung kemampuan sebuah jaringan mempelajari representasi palm dari awal. Karena tugas mata kuliah mensyaratkan pembangunan jaringan saraf tiruan tanpa model pretrained, proyek ini merancang PalmNet-Lite sebagai lightweight CNN yang seluruh parameternya diinisialisasi secara acak dan dilatih menggunakan dataset palm.

MobileFaceNet pretrained tetap dipertahankan sebagai pembanding agar eksperimen dapat menunjukkan trade-off antara pendekatan transfer learning dan model scratch yang lebih sederhana serta domain-specific.

## 3. Rumusan Masalah

Rumusan masalah yang dapat digunakan:

1. Bagaimana merancang jaringan CNN ringan untuk menghasilkan embedding palmprint tanpa menggunakan pretrained weights?
2. Seberapa baik PalmNet-Lite yang dilatih dari awal dalam melakukan palm identification dan verification?
3. Bagaimana performa PalmNet-Lite scratch dibandingkan dengan MobileFaceNet pretrained pada pipeline aplikasi dan protocol evaluasi yang sama?
4. Bagaimana dua model dengan embedding space berbeda dapat diintegrasikan secara modular pada satu aplikasi palm recognition lokal?

## 4. Tujuan Penelitian

Tujuan:

1. merancang PalmNet-Lite sebagai CNN ringan untuk palm recognition;
2. melatih PalmNet-Lite sepenuhnya dari random initialization;
3. menghasilkan biometric embedding 128 dimensi;
4. mengevaluasi model menggunakan metric identification dan verification;
5. membandingkan hasil model scratch dengan MobileFaceNet pretrained;
6. mengintegrasikan dua model dalam satu aplikasi lokal dengan model-selection mechanism yang modular.

## 5. Batasan Penelitian

Batasan:

- aplikasi dijalankan secara lokal;
- penelitian fokus pada pengembangan model dan integrasi demo;
- tidak membahas production deployment;
- tidak membahas migrasi data user lama;
- input model berupa ROI telapak berukuran 112x112 RGB;
- output kedua model berupa embedding 128 dimensi;
- similarity dihitung menggunakan cosine similarity;
- PalmNet-Lite tidak menggunakan pretrained backbone;
- MobileFaceNet digunakan sebagai model pretrained pembanding;
- hasil evaluasi dibatasi pada dataset dan protocol eksperimen yang digunakan.

## 6. Perbedaan Dua Model

### MobileFaceNet Pretrained

MobileFaceNet adalah lightweight biometric CNN yang menggunakan operasi efisien seperti depthwise convolution dan bottleneck. Pada project ini MobileFaceNet existing menggunakan pretrained weights kemudian diadaptasi pada data palm.

Posisinya dalam penelitian adalah sebagai model pembanding berbasis pretrained/transfer learning.

### PalmNet-Lite Scratch

PalmNet-Lite adalah model yang dirancang dalam project dengan prinsip lightweight CNN. Model tidak memuat pretrained weights dan semua learnable parameters dimulai dari random initialization.

PalmNet-Lite mengambil inspirasi struktural dari prinsip MobileFaceNet, namun tidak menyalin konfigurasi arsitektur secara identik.

Konfigurasi utama:

```text
Input 3x112x112
 -> Conv Stem 32
 -> Depthwise Stem 32
 -> Inverted Residual x2, C=32
 -> Inverted Residual x3, C=64
 -> Inverted Residual x4, C=96
 -> Inverted Residual x2, C=128
 -> 1x1 Projection, C=256
 -> Global Depthwise Conv 7x7
 -> 128-D Embedding
 -> L2 Normalization
```

## 7. Alasan Pemilihan Arsitektur

### 7.1 Depthwise Convolution

Regular convolution memproses channel dan spatial dimension secara bersamaan. Depthwise convolution memproses spatial feature secara terpisah pada setiap channel sehingga jumlah parameter dan computation dapat dikurangi.

Alasan digunakan:

- sesuai dengan tujuan model ringan;
- memperkecil jumlah parameter;
- tetap mampu menangkap local spatial patterns;
- cocok sebagai building block untuk aplikasi lokal dengan resource terbatas.

### 7.2 Pointwise Convolution

Conv 1x1 digunakan setelah depthwise convolution untuk menggabungkan informasi antar channel.

Kombinasi depthwise + pointwise menghasilkan alternatif convolution yang lebih efisien dibanding regular convolution besar.

### 7.3 Inverted Residual Bottleneck

Bottleneck menggunakan expansion, depthwise convolution, dan linear projection.

Tujuannya:

- mengekstrak feature pada intermediate dimensionality yang lebih besar;
- menjaga efisiensi pada input/output channel;
- memungkinkan residual connection jika shape sama;
- membantu gradient flow pada jaringan yang lebih dalam.

### 7.4 PReLU

PReLU memungkinkan slope pada bagian negatif activation dipelajari selama training.

PReLU digunakan karena biometric representation membutuhkan feature yang diskriminatif dan activation adaptif dapat mempertahankan informasi dibanding memotong semua nilai negatif secara langsung.

### 7.5 Global Depthwise Convolution

Feature map terakhir berukuran 7x7. GDConv menggunakan learned spatial kernel per channel untuk mengubah feature map menjadi representasi 1x1.

Alasan dipilih:

- area spatial tidak diperlakukan identik seperti pada simple average pooling;
- model dapat belajar spatial weighting untuk feature telapak;
- tetap mempertahankan sifat lightweight karena convolution dilakukan depthwise.

Pernyataan bahwa GDConv pasti lebih baik daripada GAP tidak boleh dibuat tanpa eksperimen. Pada laporan, GDConv harus dijelaskan sebagai pilihan desain berdasarkan karakteristik biometric spatial representation.

### 7.6 Embedding 128 Dimensi

Kedua model menggunakan 128-D output agar interface inference konsisten.

Embedding tidak berarti kedua model memiliki ruang feature yang sama. Embedding MobileFaceNet dan PalmNet-Lite tetap tidak kompatibel karena dihasilkan oleh parameter jaringan yang berbeda.

## 8. Random Initialization

PalmNet-Lite menggunakan random initialization, misalnya Kaiming initialization pada convolution layer.

Secara konsep:

```text
random weights
 -> forward prediction
 -> calculate loss
 -> backpropagation
 -> update weights
 -> repeated across epochs
```

Tidak ada knowledge dari external trained model pada initialization PalmNet-Lite.

Hal ini menjadi pembeda utama dari MobileFaceNet pretrained.

## 9. Dataset Pipeline

Pipeline konseptual:

```text
Raw Palm Dataset
 -> preprocessing / ROI extraction
 -> resize 112x112
 -> normalization
 -> data augmentation (training only)
 -> train / validation / test protocol
 -> DataLoader
 -> model training
```

Augmentation dapat meliputi rotasi ringan, crop/scale ringan, perubahan brightness/contrast, blur ringan, dan noise sesuai kebutuhan.

Horizontal flip tidak digunakan jika telapak kiri dan kanan diperlakukan sebagai kelas/identity yang berbeda karena flip dapat mengubah makna biometric sample.

## 10. Training Pipeline PalmNet-Lite

### Stage A — Softmax Representation Learning

```text
PalmNet-Lite Random Init
 -> 128-D feature
 -> temporary linear classifier
 -> Cross Entropy Loss
```

Tujuan tahap ini adalah membangun representasi awal yang mampu membedakan kelas palm pada training set.

Berbeda dari pretrained fine-tuning, seluruh backbone PalmNet-Lite trainable sejak awal.

### Stage B — ArcFace Metric Learning

Checkpoint terbaik Stage A digunakan sebagai initialization internal untuk tahap ArcFace.

```text
Own PalmNet-Lite checkpoint
 -> embedding
 -> ArcFace head
 -> angular metric learning
```

ArcFace mendorong embedding kelas yang sama menjadi lebih kompak dan kelas berbeda menjadi lebih terpisah secara angular.

ArcFace head digunakan saat training saja dan dibuang saat inference.

## 11. MobileFaceNet Training Pipeline

Pipeline MobileFaceNet dapat menggunakan pretrained initialization dan fine-tuning.

Secara konseptual:

```text
External pretrained weights
 -> domain adaptation on palm
 -> metric-learning refinement
 -> inference embedding model
```

Perbedaan initialization harus dicatat jelas agar perbandingan tidak disalahartikan sebagai perbandingan dua model scratch.

## 12. Evaluation Protocol

Evaluation harus sama untuk kedua model sejauh memungkinkan.

### Identification

Model menghasilkan query embedding dan membandingkannya terhadap enrollment template.

Metric utama:

**Rank-1 Accuracy**

Mengukur proporsi query yang identity dengan similarity tertinggi sama dengan identity sebenarnya.

### Verification

Verification mengukur apakah dua sample berasal dari identity yang sama.

Score:

```text
cosine_similarity(query_embedding, reference_embedding)
```

Metric utama:

- False Accept Rate (FAR);
- False Reject Rate (FRR);
- Equal Error Rate (EER);
- True Accept Rate pada target FAR;
- ROC-AUC.

### Genuine dan Impostor Score

Genuine score berasal dari pasangan identity yang sama.

Impostor score berasal dari pasangan identity berbeda.

Model yang baik diharapkan menghasilkan distribusi genuine dan impostor yang semakin terpisah.

## 13. Threshold

Setiap model memiliki threshold hasil calibration sendiri.

Tidak benar menggunakan threshold MobileFaceNet untuk PalmNet-Lite hanya karena output keduanya 128 dimensi.

Secara konseptual:

```text
if cosine_score >= threshold_model:
    accept
else:
    reject
```

Threshold merupakan karakteristik model dan protocol data, bukan konstanta universal.

## 14. App Pipeline

Pipeline aplikasi:

```text
Camera/Image
 -> palm detection
 -> ROI extraction
 -> preprocessing
 -> selected model
 -> 128-D normalized embedding
 -> retrieve compatible enrollment templates
 -> cosine similarity
 -> selected model threshold
 -> identified / unknown
```

Frontend memilih model melalui `model_id`.

Backend memiliki model registry yang memuat artifact lokal kedua model.

## 15. Mengapa Model Registry Digunakan

Tanpa registry, backend cenderung memiliki hardcoded path seperti satu `palm_recognizer.pt` dan banyak conditional logic.

ModelRegistry membuat runtime modular:

```text
model_id
 -> manifest
 -> model.pt
 -> threshold
 -> recognizer runtime
```

Keuntungannya:

- service identification tidak mengetahui detail architecture;
- penambahan artifact baru tidak membutuhkan duplikasi pipeline;
- model metadata terpusat;
- threshold dan version terikat pada model;
- frontend dapat mendapatkan model list melalui API.

## 16. Mengapa Embedding Dipisahkan per Model

Embedding adalah coordinate representation yang dipelajari oleh network tertentu.

Walaupun dua network menghasilkan vector 128-D, coordinate axes tidak mempunyai semantic alignment otomatis.

Karena itu:

```text
MobileFaceNet embedding
!=
PalmNet-Lite embedding space
```

Similarity hanya valid antara query dan template yang dihasilkan model serta model version yang sama.

## 17. Codebase Modularity sebagai Bagian Metodologi Implementasi

Codebase dipisahkan menjadi:

```text
models
 -> definisi architecture

data
 -> dataset dan transform

training
 -> optimization process

evaluation
 -> biometric metrics

artifacts
 -> inference export

backend runtime
 -> model loading dan inference

frontend
 -> model selection dan interaction
```

Alasan modularitas:

- mengurangi coupling;
- memudahkan eksperimen model;
- mencegah duplication training/evaluation code;
- memisahkan research code dan inference runtime;
- memudahkan validasi bahwa PalmNet-Lite benar-benar scratch.

## 18. Trained Logs

Setiap model memiliki trained logs sendiri.

Logs berguna sebagai evidence bahwa model benar-benar melalui proses training dan untuk menganalisis convergence.

Data yang layak divisualisasikan dalam laporan:

- training loss per epoch;
- validation loss;
- classification accuracy pada Stage A;
- cosine gap atau metric-learning validation statistic pada Stage B;
- learning rate;
- best epoch.

## 19. Hasil yang Harus Ditampilkan

Tabel hasil minimum:

| Metric | MobileFaceNet Pretrained | PalmNet-Lite Scratch |
|---|---:|---:|
| Parameter Count | ... | ... |
| Model Artifact Size | ... | ... |
| Rank-1 Accuracy | ... | ... |
| EER | ... | ... |
| ROC-AUC | ... | ... |
| TAR @ FAR 0.1% | ... | ... |
| Mean Genuine Similarity | ... | ... |
| Mean Impostor Similarity | ... | ... |
| Local Inference Latency (optional) | ... | ... |

Grafik yang direkomendasikan:

1. training/validation loss PalmNet-Lite;
2. metric-learning progress;
3. ROC curve kedua model;
4. genuine vs impostor score distribution;
5. FAR/FRR curve;
6. optional parameter/model-size comparison.

## 20. Cara Menginterpretasi Hasil

Jangan berasumsi model scratch harus mengalahkan pretrained.

Kemungkinan hasil:

### PalmNet-Lite lebih rendah performanya

Interpretasi valid:

- pretrained model memiliki prior representation dari dataset besar;
- dataset palm training terbatas;
- PalmNet-Lite mempunyai capacity lebih kecil;
- scratch learning membutuhkan data/hyperparameter lebih besar.

### PalmNet-Lite mendekati pretrained

Interpretasi:

- architecture lightweight domain-specific mampu belajar feature efektif;
- pretrained knowledge memberikan keuntungan terbatas untuk domain ini;
- model scratch mungkin menawarkan trade-off complexity-performance yang baik.

### PalmNet-Lite lebih baik

Jangan langsung menyimpulkan scratch selalu lebih baik.

Kemungkinan:

- architecture lebih cocok terhadap dataset;
- transfer dari face domain kurang optimal;
- tuning PalmNet-Lite lebih cocok;
- evaluation protocol menguntungkan model tertentu.

Kesimpulan harus dibatasi pada dataset dan eksperimen project.

## 21. Threats to Validity / Keterbatasan

Keterbatasan yang sebaiknya disebutkan:

- dataset mungkin tidak mewakili variasi penggunaan dunia nyata;
- lighting, camera, pose, dan background demo lokal dapat berbeda dari dataset training;
- dataset size membatasi kemampuan generalisasi scratch model;
- hyperparameter search terbatas oleh waktu/resource tugas;
- MobileFaceNet dan PalmNet-Lite berbeda architecture sekaligus initialization sehingga perbandingan bukan controlled isolation penuh terhadap efek pretraining;
- threshold bergantung pada calibration protocol;
- hasil tidak dapat digeneralisasikan ke semua dataset palm recognition.

## 22. Kesimpulan yang Aman

Bentuk kesimpulan yang aman:

> Penelitian berhasil merancang PalmNet-Lite sebagai lightweight convolutional neural network yang dilatih sepenuhnya dari random initialization untuk menghasilkan biometric embedding palm 128 dimensi. Model diintegrasikan bersama MobileFaceNet pretrained pada aplikasi lokal melalui model registry dan model-aware biometric templates. Evaluasi dilakukan menggunakan metric identification dan verification yang sama sehingga performa kedua pendekatan dapat dibandingkan secara objektif dalam batas dataset dan protocol eksperimen yang digunakan.

Isi angka performa hanya setelah final experiment selesai.

## 23. Hal yang Tidak Boleh Diklaim Tanpa Bukti

Hindari klaim:

- PalmNet-Lite lebih baik hanya karena lebih baru;
- scratch selalu lebih baik daripada pretrained;
- GDConv selalu lebih baik daripada GAP tanpa ablation;
- PReLU pasti lebih baik daripada ReLU tanpa eksperimen;
- model aman untuk security production;
- model general untuk semua populasi dan kondisi;
- threshold hasil dataset bersifat universal;
- 128-D embedding dua model kompatibel;
- MobileFaceNet merupakan model yang dibuat from scratch dalam tugas.

## 24. Evidence dari Codebase untuk Laporan

Saat implementasi selesai, laporan sebaiknya dapat ditelusuri ke evidence berikut:

```text
PalmNet-Lite source architecture
configuration file
random initialization implementation
trained_logs PalmNet-Lite
trained_logs MobileFaceNet
checkpoint metadata
metrics.json
threshold.json
TorchScript manifest
model registry
API model selector
screenshots demo aplikasi
```

Dengan demikian laporan tidak hanya berisi deskripsi teoritis, tetapi mempunyai artefak implementasi dan eksperimen yang dapat diverifikasi.

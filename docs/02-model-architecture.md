# PalmNet-Lite Scratch — Model Architecture Specification

## 1. Tujuan Desain

PalmNet-Lite adalah lightweight convolutional neural network untuk menghasilkan biometric embedding dari ROI telapak tangan. Model ini dibuat khusus untuk tugas mata kuliah AI dan dilatih sepenuhnya dari random initialization.

Desainnya mengambil inspirasi dari MobileFaceNet pada prinsip efisiensi komputasi, khususnya:

- depthwise convolution;
- pointwise convolution;
- inverted residual bottleneck;
- linear projection;
- Global Depthwise Convolution (GDConv);
- compact biometric embedding.

PalmNet-Lite bukan pretrained MobileFaceNet yang diganti namanya. Model memiliki konfigurasi channel dan jumlah block sendiri dan seluruh learnable parameters dibuat serta dilatih dalam project ini.

## 2. Design Goals

Model dirancang dengan tujuan:

1. cukup ringan untuk inference lokal;
2. mampu belajar feature palmprint dari dataset terbatas;
3. menghasilkan fixed-size embedding untuk cosine matching;
4. arsitektur cukup jelas untuk dijelaskan dalam laporan ilmiah;
5. dapat diekspor ke TorchScript;
6. menggunakan operasi standar PyTorch agar implementasi mudah diaudit;
7. mempunyai output contract yang sama dengan MobileFaceNet existing.

## 3. Input dan Output

Input model:

```text
[B, 3, 112, 112]
```

Output backbone:

```text
[B, 128]
```

Pada inference wrapper, output wajib melalui L2 normalization:

```text
z_normalized = z / ||z||2
```

Embedding normalized digunakan untuk cosine similarity.

## 4. Proposed Architecture

Konfigurasi awal PalmNet-Lite:

| Stage | Operator | Expansion | Output Channel | Repeat | Stride | Spatial Output |
|---|---|---:|---:|---:|---:|---|
| Input | RGB image | - | 3 | - | - | 112x112 |
| Stem | Conv 3x3 + BN + PReLU | - | 32 | 1 | 2 | 56x56 |
| DW Stem | DWConv 3x3 + BN + PReLU | - | 32 | 1 | 1 | 56x56 |
| Stage 1 | Inverted Residual | 2 | 32 | 2 | 1 | 56x56 |
| Stage 2 | Inverted Residual | 2 | 64 | 3 | 2 pada block pertama | 28x28 |
| Stage 3 | Inverted Residual | 2 | 96 | 4 | 2 pada block pertama | 14x14 |
| Stage 4 | Inverted Residual | 2 | 128 | 2 | 2 pada block pertama | 7x7 |
| Projection | Conv 1x1 + BN + PReLU | - | 256 | 1 | 1 | 7x7 |
| Spatial Aggregation | GDConv 7x7 linear | - | 256 | 1 | 1 | 1x1 |
| Embedding | Linear/1x1 projection + BN | - | 128 | 1 | - | 128-D |

Konfigurasi ini adalah baseline awal. Perubahan hyperparameter boleh dilakukan berdasarkan validation experiment, tetapi perubahan harus dicatat di trained logs dan report.

## 5. Building Blocks

### 5.1 ConvBlock

Komponen standar:

```text
Conv2D
  -> BatchNorm2D
  -> PReLU
```

Untuk linear projection, activation dapat dihilangkan.

### 5.2 Depthwise Convolution

Depthwise convolution menggunakan:

```text
groups = input_channels
```

Tujuan utamanya mengurangi parameter dan operasi dibanding regular convolution sambil tetap mempelajari spatial feature per-channel.

### 5.3 Pointwise Convolution

Conv `1x1` digunakan untuk:

- channel expansion;
- menggabungkan informasi antar channel setelah depthwise convolution;
- channel projection.

### 5.4 Inverted Residual Block

Struktur block:

```text
Input
  -> 1x1 Expansion Conv
  -> 3x3 Depthwise Conv
  -> 1x1 Linear Projection
  -> Residual Add jika shape kompatibel
```

Residual connection digunakan hanya jika:

```text
stride == 1
and
input_channels == output_channels
```

### 5.5 Global Depthwise Convolution

Feature map terakhir berukuran `7x7`. GDConv menggunakan kernel spatial `7x7` per channel sehingga setiap channel mempelajari bobot agregasi spatial sendiri.

GDConv dipilih sebagai alternatif Global Average Pooling karena biometric recognition membutuhkan representasi yang mempertahankan struktur spatial diskriminatif pada area telapak.

Keputusan ini adalah hipotesis desain yang dapat dibahas dalam laporan dan, jika waktu memungkinkan, diuji terhadap GAP pada eksperimen tambahan. Eksperimen GAP bukan acceptance criterion utama.

## 6. Weight Initialization

PalmNet-Lite tidak boleh melakukan load external checkpoint ketika model pertama kali dibuat.

Initialization yang direkomendasikan:

- Conv2D: Kaiming normal;
- Linear: Kaiming/Xavier sesuai activation;
- BatchNorm weight: 1;
- BatchNorm bias: 0;
- PReLU menggunakan initialization default atau value eksplisit yang terdokumentasi.

Contoh prinsip:

```python
model = PalmNetLite()
initialize_weights(model)
```

Bukan:

```python
model.load_state_dict(external_pretrained_checkpoint)
```

## 7. Training Head vs Inference Backbone

PalmNet-Lite dipisah menjadi backbone dan temporary training head.

### Softmax Stage

```text
PalmNetLite Backbone
       -> 128-D embedding
       -> Linear Classifier
       -> class logits
```

Classifier hanya digunakan saat representation warm-up dan tidak masuk artifact final.

### ArcFace Stage

```text
PalmNetLite Backbone
       -> 128-D embedding
       -> ArcFace Head
       -> metric-learning loss
```

ArcFace head juga tidak masuk artifact final.

### Inference

```text
PalmNetLite Backbone
       -> 128-D embedding
       -> L2 Normalize
```

Dengan pemisahan ini, model artifact hanya berisi komponen yang diperlukan untuk embedding extraction.

## 8. Alasan Tidak Menyalin MobileFaceNet 1:1

Tujuan tugas adalah menunjukkan kemampuan merancang dan melatih jaringan sendiri, bukan hanya menjalankan kembali arsitektur existing.

Karena itu PalmNet-Lite mempertahankan ide lightweight biometric CNN tetapi menggunakan konfigurasi sendiri:

- stem lebih kecil;
- channel progression sendiri: `32 -> 64 -> 96 -> 128`;
- jumlah bottleneck sendiri;
- projection 256 channel;
- embedding tetap 128-D untuk kompatibilitas interface aplikasi.

Hal ini membuat hubungan dengan MobileFaceNet dapat dijelaskan sebagai inspirasi desain, bukan penggunaan model pretrained atau duplikasi identik.

## 9. Reuse yang Diizinkan

PalmNet-Lite boleh menggunakan komponen project yang bukan pretrained backbone, seperti:

- dataset loader;
- augmentation utilities;
- ROI extraction pipeline;
- ArcFace loss implementation;
- metric evaluation;
- optimizer/scheduler utilities;
- TorchScript exporter;
- backend inference abstraction.

Larangan utama hanya pada initialization backbone PalmNet-Lite dari external pretrained weights.

## 10. Model Contract Tests

Minimal unit/smoke tests:

```text
input shape            = (2, 3, 112, 112)
raw embedding shape    = (2, 128)
TorchScript shape      = (2, 128)
normalized norm        ~= 1.0
NaN/Inf output         = false
parameter count        tercatat
```

Test tambahan memastikan factory `palmnet-lite-scratch` tidak menerima atau memuat pretrained checkpoint secara implisit.

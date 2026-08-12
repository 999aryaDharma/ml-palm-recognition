# PalmNet-Lite Scratch — Detailed Model Architecture Specification

## 1. Status Dokumen

Dokumen ini adalah **canonical architecture specification** untuk model baru tugas mata kuliah AI.

Model identifier:

```text
model_id             : palmnet-lite-scratch
architecture_name    : PalmNetLite
architecture_version : v1
training_mode        : scratch
```

Jika implementasi berbeda dari specification ini pada channel, jumlah block, expansion ratio, aggregation layer, embedding dimension, atau activation policy, perubahan tersebut harus dianggap sebagai **arsitektur/experiment revision**, dicatat pada trained log, dan direfleksikan ke `06-report-writing-source.md`.

MobileFaceNet existing bukan bagian dari architecture specification ini. MobileFaceNet hanya menjadi frozen runtime artifact pada aplikasi.

---

## 2. Tujuan Arsitektur

PalmNet-Lite adalah lightweight convolutional neural network untuk menghasilkan biometric embedding dari Region of Interest (ROI) telapak tangan.

Model dirancang dengan lima tujuan utama:

1. **From scratch** — seluruh learnable weight dibuat dari random initialization tanpa external pretrained checkpoint.
2. **Lightweight** — parameter dan operasi dijaga kecil dengan depthwise convolution dan bottleneck.
3. **Spatially aware** — feature map akhir tidak langsung dirata-ratakan; digunakan Global Depthwise Convolution agar agregasi spatial dapat dipelajari.
4. **Embedding oriented** — output utama bukan probabilitas kelas, tetapi vector representasi 128 dimensi yang dapat digunakan untuk cosine matching.
5. **Runtime compatible** — inference contract sama dengan pipeline aplikasi existing: input RGB `112x112`, output `128-D` embedding.

Arsitektur mengambil inspirasi konsep dari MobileFaceNet/MobileNet-style efficient convolution, tetapi konfigurasi PalmNet-Lite v1 ditentukan khusus dalam proyek ini dan tidak memuat bobot dari model tersebut.

---

## 3. Batas Klaim

Klaim yang boleh dibuat:

> PalmNet-Lite adalah lightweight CNN yang dirancang dalam proyek ini menggunakan depthwise convolution, inverted residual bottleneck, learned spatial aggregation, dan compact embedding, kemudian dilatih sepenuhnya dari random initialization pada dataset palm.

Klaim yang tidak boleh dibuat tanpa bukti eksperimen:

- PalmNet-Lite lebih baik dari MobileFaceNet;
- GDConv pasti lebih baik dari Global Average Pooling;
- 128 dimensi adalah ukuran embedding optimal;
- expansion ratio 2 adalah konfigurasi terbaik;
- arsitektur ini state-of-the-art.

Semua pilihan tersebut adalah **design choices** yang memiliki alasan teknis, bukan bukti superioritas otomatis.

---

## 4. Shared Inference Contract

### 4.1 Input

```text
Tensor shape : [B, 3, 112, 112]
dtype        : float32
color space  : RGB
range awal   : [0, 255] pada image
```

Normalisasi sebelum backbone:

```text
x = x / 255.0
x = (x - 0.5) / 0.5
```

Sehingga nilai input network berada kira-kira pada:

```text
[-1, 1]
```

Equivalent config:

```text
mean = [0.5, 0.5, 0.5]
std  = [0.5, 0.5, 0.5]
```

### 4.2 Raw Backbone Output

```text
shape : [B, 128]
type  : raw embedding
```

Backbone **tidak wajib melakukan L2 normalization di dalam `forward()`**. Ini disengaja agar raw embedding dapat digunakan oleh training head seperti Softmax classifier dan ArcFace.

### 4.3 Inference Output

Inference wrapper melakukan:

```text
z = backbone(x)
z = z / ||z||2
```

Output final:

```text
shape   : [B, 128]
norm L2 : ~1.0
```

Cosine similarity kemudian dapat dihitung dengan dot product terhadap embedding lain yang juga L2-normalized.

---

## 5. Architecture Summary

Baseline PalmNet-Lite v1:

| Stage | Operator | Expansion | Output C | Repeat | First Stride | Output HxW |
|---|---|---:|---:|---:|---:|---:|
| Input | RGB | - | 3 | - | - | 112x112 |
| Stem | Conv 3x3 + BN + PReLU | - | 32 | 1 | 2 | 56x56 |
| DW Stem | DWConv 3x3 + BN + PReLU | - | 32 | 1 | 1 | 56x56 |
| Stage 1 | Inverted Residual | 2 | 32 | 2 | 1 | 56x56 |
| Stage 2 | Inverted Residual | 2 | 64 | 3 | 2 | 28x28 |
| Stage 3 | Inverted Residual | 2 | 96 | 4 | 2 | 14x14 |
| Stage 4 | Inverted Residual | 2 | 128 | 2 | 2 | 7x7 |
| Projection | Conv 1x1 + BN + PReLU | - | 256 | 1 | 1 | 7x7 |
| Spatial Aggregation | GDConv 7x7 + BN, linear | - | 256 | 1 | 1 | 1x1 |
| Embedding | Flatten + Linear + BN1d | - | 128 | 1 | - | 128-D |

Expected trainable parameter count untuk konfigurasi ini sekitar:

```text
~393,000 parameters
```

Angka final pada laporan harus diambil langsung dari implementasi (`sum(p.numel() ...)`) setelah model selesai dibuat. Range sanity test yang disarankan:

```text
350,000 <= total_params <= 450,000
```

Range tersebut adalah guard untuk mendeteksi kesalahan konfigurasi, bukan target ilmiah.

---

## 6. Layer-by-Layer Specification

Notasi:

```text
C  = channel
H  = height
W  = width
t  = expansion ratio
DW = depthwise convolution
PW = pointwise 1x1 convolution
s  = stride
```

### 6.1 Stem

Input:

```text
[B, 3, 112, 112]
```

Operation:

```text
Conv2d(
    in_channels=3,
    out_channels=32,
    kernel_size=3,
    stride=2,
    padding=1,
    bias=False
)
-> BatchNorm2d(32)
-> PReLU(32)
```

Output:

```text
[B, 32, 56, 56]
```

Tujuan:

- melakukan downsampling awal;
- membangun feature dasar edge/texture;
- menjaga channel awal kecil agar biaya komputasi tidak membesar terlalu cepat.

### 6.2 Depthwise Stem

Operation:

```text
Conv2d(
    in_channels=32,
    out_channels=32,
    kernel_size=3,
    stride=1,
    padding=1,
    groups=32,
    bias=False
)
-> BatchNorm2d(32)
-> PReLU(32)
```

Output:

```text
[B, 32, 56, 56]
```

Layer ini menambahkan spatial filtering dengan parameter kecil sebelum masuk ke bottleneck stack.

---

## 7. Inverted Residual Block Specification

Setiap block mempunyai tiga operasi utama:

```text
Input
  -> PW Expansion 1x1
  -> BN
  -> PReLU
  -> DWConv 3x3
  -> BN
  -> PReLU
  -> PW Linear Projection 1x1
  -> BN
  -> optional residual add
Output
```

Tidak ada activation setelah linear projection.

### 7.1 Hidden Channel

Untuk expansion ratio `t`:

```text
hidden_channels = input_channels * t
```

PalmNet-Lite v1 menggunakan:

```text
t = 2
```

pada seluruh inverted residual block.

### 7.2 Expansion Operation

```text
Conv2d(C_in, C_hidden, kernel=1, stride=1, padding=0, bias=False)
BatchNorm2d(C_hidden)
PReLU(C_hidden)
```

### 7.3 Depthwise Operation

```text
Conv2d(
    C_hidden,
    C_hidden,
    kernel=3,
    stride=block_stride,
    padding=1,
    groups=C_hidden,
    bias=False
)
BatchNorm2d(C_hidden)
PReLU(C_hidden)
```

### 7.4 Linear Projection

```text
Conv2d(C_hidden, C_out, kernel=1, stride=1, padding=0, bias=False)
BatchNorm2d(C_out)
```

Tidak menggunakan PReLU setelah projection karena projection diperlakukan sebagai linear bottleneck.

### 7.5 Residual Connection

Residual hanya aktif jika:

```text
block_stride == 1
AND
C_in == C_out
```

Jika kondisi benar:

```text
output = input + block(input)
```

Jika salah:

```text
output = block(input)
```

Tidak digunakan projection shortcut tambahan. Hal ini menjaga block sederhana dan ringan.

---

## 8. Exact Bottleneck Layout

### 8.1 Stage 1 — Preserve Resolution

Input:

```text
[B, 32, 56, 56]
```

#### Block 1.1

```text
C_in   = 32
C_hide = 64
C_out  = 32
stride = 1
residual = yes
```

Output:

```text
[B, 32, 56, 56]
```

#### Block 1.2

```text
C_in   = 32
C_hide = 64
C_out  = 32
stride = 1
residual = yes
```

Output Stage 1:

```text
[B, 32, 56, 56]
```

Tujuan Stage 1 adalah memperdalam local texture extraction tanpa mengurangi resolusi spatial.

### 8.2 Stage 2 — 56x56 to 28x28

#### Block 2.1

```text
C_in   = 32
C_hide = 64
C_out  = 64
stride = 2
residual = no
```

Output:

```text
[B, 64, 28, 28]
```

#### Block 2.2

```text
C_in   = 64
C_hide = 128
C_out  = 64
stride = 1
residual = yes
```

#### Block 2.3

Sama dengan Block 2.2.

Output Stage 2:

```text
[B, 64, 28, 28]
```

### 8.3 Stage 3 — 28x28 to 14x14

#### Block 3.1

```text
C_in   = 64
C_hide = 128
C_out  = 96
stride = 2
residual = no
```

Output:

```text
[B, 96, 14, 14]
```

#### Block 3.2

```text
C_in   = 96
C_hide = 192
C_out  = 96
stride = 1
residual = yes
```

#### Block 3.3

Sama dengan Block 3.2.

#### Block 3.4

Sama dengan Block 3.2.

Output Stage 3:

```text
[B, 96, 14, 14]
```

Stage ini menjadi bagian utama feature extraction karena mempunyai jumlah block terbanyak.

### 8.4 Stage 4 — 14x14 to 7x7

#### Block 4.1

```text
C_in   = 96
C_hide = 192
C_out  = 128
stride = 2
residual = no
```

Output:

```text
[B, 128, 7, 7]
```

#### Block 4.2

```text
C_in   = 128
C_hide = 256
C_out  = 128
stride = 1
residual = yes
```

Output Stage 4:

```text
[B, 128, 7, 7]
```

---

## 9. Final Projection

Input:

```text
[B, 128, 7, 7]
```

Operation:

```text
Conv2d(128, 256, kernel=1, stride=1, padding=0, bias=False)
-> BatchNorm2d(256)
-> PReLU(256)
```

Output:

```text
[B, 256, 7, 7]
```

Tujuan projection adalah membentuk feature space yang lebih kaya sebelum spatial aggregation tanpa menambah spatial resolution.

---

## 10. Global Depthwise Convolution (GDConv)

Input:

```text
[B, 256, 7, 7]
```

Operation:

```text
Conv2d(
    in_channels=256,
    out_channels=256,
    kernel_size=7,
    stride=1,
    padding=0,
    groups=256,
    bias=False
)
-> BatchNorm2d(256)
```

Tidak menggunakan activation setelah GDConv.

Output:

```text
[B, 256, 1, 1]
```

### Alasan Desain

Global Average Pooling menggunakan bobot spatial yang tetap dan sama besar untuk seluruh lokasi. GDConv membuat setiap channel memiliki kernel spatial yang dipelajari.

Hipotesis desainnya adalah pola palmprint memiliki lokasi/struktur spatial yang dapat memberikan kontribusi tidak seragam terhadap biometric representation.

Namun laporan harus menuliskan ini sebagai **alasan pemilihan**, bukan klaim bahwa GDConv terbukti selalu lebih baik.

---

## 11. Embedding Layer

Input:

```text
[B, 256, 1, 1]
```

Operation:

```text
Flatten()
-> Linear(256, 128, bias=False)
-> BatchNorm1d(128)
```

Tidak ada activation setelah embedding BatchNorm.

Raw output:

```text
[B, 128]
```

### Mengapa 128 Dimensi

128 dimensi dipilih karena:

- compact untuk penyimpanan template;
- cukup kecil untuk cosine matching lokal;
- konsisten dengan interface aplikasi existing;
- memberi ruang feature yang lebih besar daripada representation yang sangat kecil tanpa membuat artifact terlalu besar.

Pemilihan 128-D bukan klaim optimum dan dapat dinyatakan sebagai architectural design constraint.

---

## 12. Activation Policy

PalmNet-Lite v1 menggunakan channel-wise PReLU pada nonlinear layer.

PReLU digunakan pada:

- stem;
- depthwise stem;
- inverted residual expansion;
- inverted residual depthwise;
- final projection.

PReLU **tidak digunakan** pada:

- inverted residual linear projection;
- GDConv output;
- final embedding layer.

Initial negative slope baseline:

```text
0.25
```

Parameter slope kemudian dipelajari selama training.

---

## 13. Batch Normalization Policy

BatchNorm digunakan setelah setiap convolution dan pada final embedding.

Baseline:

```text
BatchNorm2d eps      = 1e-5
BatchNorm2d momentum = 0.1
BatchNorm1d eps      = 1e-5
BatchNorm1d momentum = 0.1
```

Convolution/Linear yang langsung diikuti BatchNorm menggunakan:

```text
bias=False
```

Pada inference, model harus berada pada:

```python
model.eval()
```

agar running statistics BatchNorm tidak berubah.

---

## 14. Weight Initialization Specification

Model **harus** diinisialisasi tanpa membaca external checkpoint.

Initialization baseline:

### Conv2d

```text
Kaiming Normal
mode = fan_out
nonlinearity = leaky_relu
```

PReLU menggunakan initial slope `0.25`.

### Linear Embedding / Temporary Classifier

```text
Xavier Normal
bias = 0 jika bias ada
```

### BatchNorm

```text
weight/gamma = 1.0
bias/beta    = 0.0
```

### ArcFace Weight

Diinisialisasi terpisah oleh training head menggunakan Xavier/normal initialization yang eksplisit di implementation loss/head.

### Verification Rule

Construction path harus bersifat:

```python
model = PalmNetLite(config)
initialize_scratch_weights(model)
```

Tidak boleh terdapat hidden behavior seperti:

```python
PalmNetLite(pretrained=True)
load_default_checkpoint()
download_weights()
```

Unit test harus memastikan path scratch tidak membutuhkan file `.pth`, `.pt`, URL, atau pretrained model registry.

---

## 15. Training Heads

Backbone dan training head harus terpisah.

### 15.1 Stage A — Classification Head

```text
PalmNetLite backbone
-> raw 128-D embedding
-> Linear(128, num_classes)
-> logits
```

Classifier hanya digunakan untuk Cross Entropy training dan dibuang setelah Stage A.

### 15.2 Stage B — ArcFace Head

```text
PalmNetLite backbone
-> raw 128-D embedding
-> ArcFace angular classifier
-> loss
```

ArcFace melakukan normalization yang dibutuhkan dalam computation loss.

ArcFace head tidak masuk artifact inference final.

### 15.3 Final Inference Wrapper

```text
PalmNetLite backbone
-> raw 128-D embedding
-> L2 normalize(dim=1)
-> final 128-D unit embedding
```

---

## 16. No Dropout Baseline

PalmNet-Lite v1 tidak menggunakan Dropout pada backbone baseline.

Alasan:

- network sudah relatif kecil;
- BatchNorm dan augmentation sudah memberi regularization;
- mengurangi variabel eksperimen awal;
- biometric embedding membutuhkan inference path deterministik.

Jika overfitting signifikan ditemukan, dropout boleh menjadi eksperimen revisi, tetapi harus dicatat sebagai perubahan configuration/architecture.

---

## 17. Architecture Configuration Contract

Target config minimal:

```yaml
model:
  id: palmnet-lite-scratch
  architecture: PalmNetLite
  architecture_version: v1
  input_channels: 3
  input_size: 112
  embedding_dim: 128
  activation: prelu
  prelu_init: 0.25
  expansion_ratio: 2
  stem_channels: 32
  stage_channels: [32, 64, 96, 128]
  stage_repeats: [2, 3, 4, 2]
  projection_channels: 256
  spatial_aggregation: gdconv
  initialization: kaiming_random
```

Implementation tidak boleh menyebar nilai tersebut sebagai magic number di banyak file jika config sudah menjadi source of truth.

---

## 18. Forward Shape Trace

Shape trace yang harus lulus smoke test:

```text
Input              [B,   3, 112, 112]
Stem               [B,  32,  56,  56]
DW Stem            [B,  32,  56,  56]
Stage 1            [B,  32,  56,  56]
Stage 2            [B,  64,  28,  28]
Stage 3            [B,  96,  14,  14]
Stage 4            [B, 128,   7,   7]
Projection         [B, 256,   7,   7]
GDConv             [B, 256,   1,   1]
Flatten            [B, 256]
Embedding          [B, 128]
Inference L2 Norm  [B, 128]
```

---

## 19. Architecture Tests

Minimal automated tests:

### 19.1 Construction Test

- model dapat dibuat tanpa checkpoint;
- tidak ada file/network access;
- seluruh parameter `requires_grad=True` pada backbone saat awal training.

### 19.2 Shape Test

Input:

```text
(2, 3, 112, 112)
```

Expected raw output:

```text
(2, 128)
```

### 19.3 Parameter Budget Test

Expected:

```text
350k - 450k trainable parameters
```

Exact count dicatat setelah implementation final.

### 19.4 Gradient Test

Satu dummy backward pass harus menghasilkan gradient finite pada parameter utama.

### 19.5 Numerical Test

Output tidak boleh memiliki:

```text
NaN
Inf
```

### 19.6 Inference Normalization Test

Setelah inference wrapper:

```text
||embedding_i||2 ~= 1.0
```

dengan toleransi numerik, misalnya `1e-5` sampai `1e-4`.

### 19.7 TorchScript Test

- eager model dapat dibungkus untuk inference;
- TorchScript export berhasil;
- TorchScript output shape sama;
- max absolute difference terhadap eager output berada dalam toleransi yang ditetapkan.

---

## 20. Architecture Freeze Rule untuk Eksperimen

Sebelum full training pertama, lakukan smoke test arsitektur dan parameter count. Setelah full training dimulai, configuration PalmNet-Lite v1 dianggap **frozen untuk run tersebut**.

Jika kemudian dilakukan perubahan seperti:

- `32 -> 48` stem channel;
- expansion ratio `2 -> 4`;
- jumlah Stage 3 block `4 -> 6`;
- GDConv diganti GAP;
- embedding `128 -> 256`;

maka perubahan harus:

1. menghasilkan run baru;
2. tersimpan di `run.json`;
3. tidak menimpa checkpoint run lama;
4. dicatat alasan perubahannya;
5. diperbarui pada living report jika konfigurasi baru dipilih sebagai final.

---

## 21. Keunggulan Desain yang Diharapkan

Keunggulan yang **secara arsitektural diharapkan**, sebelum hasil training diketahui:

- jumlah parameter relatif kecil;
- computational path sederhana;
- dapat dilatih dari awal tanpa ketergantungan pretrained weight;
- embedding compact;
- TorchScript-friendly;
- modular terhadap training head;
- spatial aggregation dapat dipelajari melalui GDConv.

Keunggulan performa recognition hanya boleh ditulis setelah evaluation selesai.

---

## 22. Keterbatasan Desain yang Sudah Diketahui

Keterbatasan yang dapat dijelaskan bahkan sebelum hasil training:

- scratch model lebih bergantung pada jumlah dan keragaman dataset;
- lightweight capacity dapat membatasi representational power;
- BatchNorm dapat sensitif terhadap batch size yang terlalu kecil;
- fixed `112x112` ROI dapat membuang detail jika preprocessing kurang baik;
- GDConv `7x7` mengasumsikan final feature map memiliki ukuran tepat `7x7`;
- architecture hyperparameter belum dibuktikan optimal;
- model recognition scratch tidak berarti seluruh preprocessing system bebas pretrained dependency jika detector/landmarker existing masih digunakan.

---

## 23. Canonical Implementation Mapping

Target modular mapping:

```text
palm_recognition/models/blocks.py
    ConvBNPReLU
    LinearConvBN
    InvertedResidual
    GlobalDepthwiseConv

palm_recognition/models/palmnet_lite.py
    PalmNetLite
    architecture assembly only

palm_recognition/models/initialization.py
    initialize_scratch_weights

palm_recognition/models/inference.py
    EmbeddingInferenceWrapper
```

Architecture module tidak boleh memiliki tanggung jawab training loop, dataset loading, checkpoint saving, evaluation, atau backend deployment.

Dokumen phase training yang menggunakan architecture ini berada di `03-training-and-artifacts.md`.

# PalmNet-Lite — Detailed Training, Evaluation, Trained Logs, and Artifact Specification

## 1. Status Dokumen

Dokumen ini adalah **canonical training specification** untuk PalmNet-Lite Scratch.

Scope aktif:

```text
PalmNet-Lite
= design + scratch initialization + training + evaluation + export

MobileFaceNet
= frozen existing runtime artifact only
```

MobileFaceNet tidak dilatih ulang, tidak di-fine-tune, dan tidak menjadi dependency dari pipeline training PalmNet-Lite.

---

## 2. Tujuan Training

Training PalmNet-Lite bertujuan menghasilkan backbone yang memetakan ROI palm ke embedding 128 dimensi dengan sifat:

- sample dari palm identity yang sama memiliki cosine similarity tinggi;
- sample dari palm identity berbeda memiliki cosine similarity lebih rendah;
- representation dapat digeneralisasikan ke identity yang tidak digunakan sebagai kelas training;
- artifact final dapat digunakan untuk enrollment dan identification pada aplikasi lokal.

Training tidak diarahkan hanya untuk memaksimalkan closed-set classification accuracy. Classification digunakan sebagai **representation warm-up**, sedangkan tujuan akhir adalah biometric embedding.

---

## 3. Global Training Constraints

Aturan yang berlaku untuk seluruh run:

1. PalmNet-Lite selalu dimulai dari random initialization pada awal Phase 1.
2. Tidak ada external model checkpoint yang boleh dimuat ke backbone.
3. Dataset split tidak boleh berubah di tengah run.
4. Test set tidak boleh digunakan untuk hyperparameter tuning.
5. Semua hyperparameter aktual harus tersimpan di `run.json`.
6. Setiap full run memiliki folder sendiri dan tidak boleh menimpa run lama.
7. Best checkpoint dipilih berdasarkan validation metric yang didefinisikan sebelum training.
8. Semua metric final yang masuk laporan harus dapat ditelusuri ke `run_id` dan checkpoint.
9. Random seed harus dicatat.
10. Training code tidak boleh diam-diam fallback ke pretrained weights jika checkpoint tidak ditemukan.

---

## 4. Canonical Training Flow

```text
Phase 0 — Dataset & Pipeline Sanity
        |
        v
Phase 1 — Scratch Softmax Representation Learning
        |
        | best Stage-1 checkpoint
        v
Phase 2 — ArcFace Metric Learning
        |
        | best Stage-2 checkpoint
        v
Phase 3 — Validation Calibration & Model Selection
        |
        v
Phase 4 — Final Hold-out Evaluation
        |
        v
Phase 5 — TorchScript Export & App Integration
```

Setiap phase memiliki input, output, gate, dan artifact yang jelas.

---

# PHASE 0 — DATASET & PIPELINE SANITY

## 5. Tujuan Phase 0

Phase 0 memastikan model tidak membuang waktu GPU pada pipeline data yang salah.

Yang divalidasi:

- path dataset valid;
- label mapping konsisten;
- jumlah class sesuai ekspektasi;
- train/validation/test split tidak overlap;
- image dapat dibaca;
- transform menghasilkan tensor shape yang benar;
- ROI tidak kosong/rusak;
- augmentation tidak merusak identitas palm;
- class distribution diketahui;
- dataloader dapat menghasilkan batch stabil.

---

## 6. Dataset Protocol Baseline

Protocol repository saat ini dapat dipertahankan sebagai baseline:

```text
Total palm identities : 600 palm classes

Palm 1-400
    Session 1 -> Train
    Session 2 -> Validation

Palm 401-600
    Session 1 -> Enrollment side of Test
    Session 2 -> Query side of Test
```

Interpretasi:

- `400` palm identities digunakan untuk belajar representation;
- `200` palm identities tidak pernah digunakan sebagai class saat training;
- final evaluation menguji generalisasi embedding ke unseen identities.

Jika preprocessing menghasilkan jumlah sample aktual berbeda karena rejection/quality filter, jumlah final harus dicatat di run metadata.

---

## 7. Split Integrity Rules

Harus diverifikasi sebelum training:

```text
train_identity ∩ test_identity = empty
validation_identity = training identities, cross-session
```

Tidak boleh:

- image duplikat muncul di train dan test;
- test identity masuk classifier training;
- hyperparameter dipilih berdasarkan final test result.

Script sanity sebaiknya menghasilkan summary:

```text
train images
validation images
test images
train identities
validation identities
test identities
missing files
corrupt files
rejected ROI
```

---

## 8. Input Transform

### Train Transform Baseline

Urutan konseptual:

```text
Load RGB ROI
-> mild rotation
-> mild resized crop / scale jitter
-> optional brightness/contrast/saturation jitter
-> optional mild Gaussian blur
-> optional mild Gaussian noise
-> normalize mean/std
-> tensor [3,112,112]
```

Baseline limits:

```text
rotation       : ±10 to ±15 degree
crop scale     : 0.90 - 1.00
crop ratio     : ~0.85 - 1.15
brightness     : up to ±0.25
contrast       : up to ±0.25
saturation     : up to ±0.15
hue            : small only
Gaussian blur  : low probability
noise          : low probability
```

Tidak digunakan:

- horizontal flip jika left/right palm adalah identity berbeda;
- vertical flip;
- extreme rotation;
- aggressive perspective distortion.

### Validation/Test Transform

Deterministic:

```text
Resize/ensure 112x112
-> normalize
-> tensor
```

Tidak ada augmentation acak.

---

## 9. Phase 0 Smoke Batch

Sebelum full training:

- ambil minimal satu batch;
- verifikasi shape `(B,3,112,112)`;
- verifikasi labels integer valid;
- jalankan forward PalmNet-Lite;
- hitung temporary classification loss;
- lakukan satu backward;
- pastikan gradient finite.

Gate Phase 0:

```text
PASS jika:
- dataset integrity valid
- batch valid
- forward valid
- loss finite
- backward finite

FAIL -> training tidak boleh dilanjutkan
```

---

# PHASE 1 — SOFTMAX REPRESENTATION LEARNING

## 10. Tujuan Phase 1

Phase 1 mengajarkan backbone scratch membangun feature dasar yang diskriminatif sebelum diberikan angular metric objective.

Flow:

```text
PalmNetLite random initialization
        -> raw 128-D embedding
        -> temporary Linear classifier
        -> CrossEntropyLoss
        -> backpropagation through ALL backbone layers
```

Tidak ada frozen stage.

---

## 11. Model Phase 1

```text
backbone = PalmNetLite(embedding_dim=128)
classifier = Linear(128, num_train_classes)
```

Forward:

```text
embedding = backbone(images)
logits = classifier(embedding)
```

Classifier hanya training head dan tidak ikut export final.

---

## 12. Initialization Phase 1

Backbone:

- Conv2d: Kaiming Normal;
- Linear embedding: Xavier Normal;
- BatchNorm gamma=1, beta=0;
- PReLU initial slope=0.25.

Classifier:

```text
Xavier Normal
bias = 0
```

Initialization seed harus mengikuti run seed.

---

## 13. Loss Phase 1

Baseline:

```text
CrossEntropyLoss
```

Optional label smoothing:

```text
0.0 - 0.1
```

Default baseline yang disarankan:

```text
label_smoothing = 0.1
```

Jika implementasi awal ingin lebih sederhana, `0.0` boleh digunakan, tetapi config aktual wajib dicatat.

---

## 14. Optimizer Phase 1

Baseline:

```text
optimizer    : AdamW
initial_lr   : 1e-3
weight_decay : 1e-4
```

Semua parameter backbone + classifier masuk optimizer.

Tidak boleh ada parameter backbone yang frozen.

---

## 15. LR Schedule Phase 1

Recommended baseline:

```text
Linear warmup 3 epoch
        -> CosineAnnealingLR sampai akhir phase
```

Jika implementasi warmup memperumit code tanpa manfaat pada smoke run, baseline minimal boleh langsung CosineAnnealing.

Source of truth tetap config run.

Suggested:

```text
max_epochs   = 40
warmup_epoch = 3
min_lr       = 1e-6
```

Range eksplorasi jika diperlukan:

```text
30 - 60 epochs
```

Jangan otomatis menambah epoch tanpa melihat validation curve.

---

## 16. Batch Size Phase 1

Baseline:

```text
batch_size = 64
```

Fallback jika memory terbatas:

```text
32
```

Karena backbone menggunakan BatchNorm, batch terlalu kecil sebaiknya dihindari.

Jika batch `<16`, training harus diberi catatan khusus karena BatchNorm stability dapat terpengaruh.

---

## 17. Gradient Handling Phase 1

Baseline:

```text
optimizer.zero_grad(set_to_none=True)
forward
loss.backward()
gradient clipping optional
optimizer.step()
```

Recommended safety clip:

```text
max_grad_norm = 5.0
```

Log gradient norm jika debugging instability diperlukan.

---

## 18. Phase 1 Epoch Metrics

Setiap epoch minimal mencatat:

```text
train_loss
train_accuracy
val_loss
val_accuracy
learning_rate
epoch_duration_sec
```

Optional:

```text
grad_norm
GPU memory
```

---

## 19. Phase 1 Checkpoint Strategy

Save:

```text
checkpoint_phase1_last.pth
checkpoint_phase1_best.pth
```

`best` dipilih berdasarkan:

```text
highest validation accuracy
```

Jika val accuracy sama, tie-breaker:

```text
lower validation loss
```

Checkpoint content minimal:

```text
run_id
phase
architecture_version
epoch
backbone_state_dict
classifier_state_dict
optimizer_state_dict
scheduler_state_dict
best_metric
config snapshot
```

---

## 20. Phase 1 Early Stopping

Tidak wajib pada run pertama.

Jika digunakan:

```text
monitor  : val_accuracy
patience : 8-10 epochs
min_delta: kecil, mis. 0.001
```

Early stopping tidak boleh terlalu agresif karena scratch model membutuhkan waktu membentuk feature awal.

---

## 21. Phase 1 Gate

Tujuan gate bukan menetapkan angka “sakral”, tetapi mencegah Phase 2 dijalankan pada backbone yang jelas gagal belajar.

Gate minimum:

```text
1. train loss menurun secara meaningful
2. validation accuracy jauh di atas random chance
3. output embedding finite
4. tidak terjadi training collapse
5. best checkpoint tersedia
```

Operational starting gate:

```text
val_accuracy >= 0.70
```

Target preferensi:

```text
val_accuracy >= 0.75
```

Angka ini boleh direvisi setelah melihat karakteristik dataset, tetapi perubahan harus dicatat.

Jika gagal:

Investigasi urutan:

1. label mapping;
2. split/session;
3. ROI quality;
4. augmentation terlalu agresif;
5. normalization;
6. learning rate;
7. batch size;
8. architecture bug.

Jangan langsung menambah epoch sebagai respons pertama terhadap pipeline yang salah.

---

# PHASE 2 — ARCFACE METRIC LEARNING

## 22. Tujuan Phase 2

Phase 2 mengubah objective dari sekadar class separation menuju **embedding geometry** yang cocok untuk biometric matching.

Input backbone Phase 2 berasal dari:

```text
checkpoint_phase1_best.pth
```

Checkpoint ini adalah hasil training internal proyek, sehingga tetap konsisten dengan klaim model from scratch.

---

## 23. Model Phase 2

```text
PalmNetLite backbone
+ ArcFace head
```

Phase 1 linear classifier dibuang.

Backbone state:

```text
load own Phase-1 backbone weights
ALL backbone layers trainable
```

Tidak ada freeze stage baseline.

---

## 24. ArcFace Objective

Konsep:

```text
normalize embedding
normalize class weights
compute cosine theta
apply additive angular margin to target class
scale logits
Cross Entropy
```

Baseline hyperparameter awal:

```text
margin m = 0.30
scale  s = 32
```

Alasan mulai lebih konservatif daripada margin tinggi:

- scratch dataset relatif terbatas;
- margin terlalu agresif dapat membuat optimization sulit;
- tujuan run pertama adalah stable metric learning.

Candidate exploration jika diperlukan:

```text
margin: 0.20, 0.30, 0.40
scale : 32, optional 64
```

Jangan melakukan grid search besar untuk scope tugas.

---

## 25. Margin Warmup

Recommended:

```text
margin epoch 1 -> 0.0
linear increase
hingga target m pada epoch 5-10
```

Baseline:

```text
margin_warmup_epochs = 8
```

Tujuan:

- menghindari angular constraint terlalu keras pada awal phase;
- memberi backbone waktu beradaptasi dari classifier representation ke metric representation.

---

## 26. Optimizer Phase 2

Baseline:

```text
optimizer    : AdamW
lr           : 1e-4
weight_decay : 1e-4
```

Optimizer mencakup:

```text
backbone parameters
+
ArcFace head parameters
```

Candidate lower LR jika representation rusak terlalu cepat:

```text
5e-5
```

---

## 27. LR Scheduler Phase 2

Baseline:

```text
CosineAnnealingLR
```

Suggested:

```text
max_epochs = 40-60
min_lr     = 1e-6
```

Starting config:

```text
50 epochs
```

---

## 28. Gradient Safety Phase 2

ArcFace dapat lebih sensitif terhadap gradient instability.

Wajib baseline:

```text
grad_clip_norm = 5.0
```

Jika loss menjadi NaN:

- hentikan run;
- jangan save checkpoint NaN sebagai best;
- catat failure pada `notes.md`;
- periksa margin, scale, LR, normalization, dan implementation ArcFace.

---

## 29. Validation Embedding Metrics Phase 2

Phase 2 tidak dipilih berdasarkan classification accuracy saja.

Setiap validation cycle hitung minimal:

```text
mean_positive_cosine
mean_negative_cosine
cosine_gap = mean_positive - mean_negative
```

Recommended tambahan:

```text
validation EER
ROC-AUC
```

Jika pair calculation mahal, gunakan deterministic pair sampling dengan seed tetap dan jumlah pair yang dicatat.

---

## 30. Pair Construction Validation

Positive pair:

```text
same palm identity
cross image/session jika memungkinkan
```

Negative pair:

```text
different palm identity
```

Jumlah pair dan sampling strategy harus dicatat agar metric antar run dapat dibandingkan secara fair.

Hindari hanya mengambil negative pair yang terlalu mudah jika sampling dapat dikontrol.

---

## 31. Phase 2 Best Checkpoint

Save:

```text
checkpoint_phase2_last.pth
checkpoint_phase2_best.pth
```

Primary selection metric baseline:

```text
highest validation cosine_gap
```

Tie-breaker:

```text
lower validation EER
```

Jika EER belum dihitung per epoch, tie-breaker dapat menggunakan lower training loss atau higher mean-positive, tetapi decision harus konsisten dalam run.

Checkpoint content:

```text
run_id
phase
epoch
backbone_state_dict
arcface_state_dict
optimizer_state_dict
scheduler_state_dict
margin state
validation metrics
config snapshot
```

---

## 32. Phase 2 Gate

Operational target awal:

```text
cosine_gap >= 0.30 : acceptable candidate
cosine_gap >= 0.40 : strong preferred gate
```

Lebih penting daripada angka absolut:

```text
mean_positive > mean_negative
clear separation exists
validation EER finite and reasonable
```

Jika gap rendah:

Investigasi:

1. Phase 1 checkpoint quality;
2. ArcFace implementation;
3. margin terlalu besar;
4. scale terlalu besar;
5. LR terlalu tinggi;
6. pair metric bug;
7. class imbalance/sample scarcity;
8. ROI variation antar session.

---

# PHASE 3 — VALIDATION CALIBRATION

## 33. Tujuan Phase 3

Setelah best Phase 2 checkpoint dipilih, validation data digunakan untuk:

- menentukan operating threshold;
- menghasilkan diagnostic metric;
- memastikan model siap untuk final test.

Final test belum disentuh pada tahap tuning ini.

---

## 34. Embedding Extraction Validation

Untuk seluruh validation samples:

```text
model.eval()
no_grad()
extract raw embedding
L2 normalize
cache embedding
```

Jangan melakukan augmentation acak pada validation extraction.

---

## 35. Threshold Calibration

Hitung genuine dan impostor scores.

Baseline threshold options:

### EER Threshold

Pilih threshold pada titik:

```text
FAR ~= FRR
```

Simpan sebagai diagnostic/reference threshold.

### FAR-Constrained Threshold

Jika aplikasi ingin mode lebih ketat, threshold dapat dipilih pada target FAR tertentu.

Untuk scope tugas, threshold utama dapat menggunakan EER threshold terlebih dahulu selama dijelaskan dengan benar.

Output:

```text
threshold.json
```

Contoh schema:

```json
{
  "model_id": "palmnet-lite-scratch",
  "version": "1.0.0",
  "metric": "cosine_similarity",
  "threshold": 0.48,
  "calibration_split": "validation",
  "selection": "eer",
  "eer": 0.021
}
```

Nilai hanya contoh.

---

## 36. Phase 3 Diagnostic Outputs

Simpan:

```text
validation_metrics.json
validation_score_distribution.png
validation_roc_curve.png
validation_far_frr_curve.png
threshold.json
```

Plot adalah evidence untuk laporan, bukan input ke runtime.

---

# PHASE 4 — FINAL HOLD-OUT EVALUATION

## 37. Tujuan Phase 4

Final evaluation mengukur kemampuan embedding pada identities yang tidak menjadi training classes.

Protocol baseline:

```text
Test palms 401-600
Session 1 -> enrollment template
Session 2 -> query
```

Threshold yang digunakan harus sudah ditentukan pada validation calibration.

Final test tidak boleh digunakan untuk memilih ulang hyperparameter.

---

## 38. Enrollment Template Construction

Untuk setiap test palm identity:

1. extract embedding seluruh Session-1 sample;
2. L2 normalize setiap embedding;
3. average embeddings;
4. L2 normalize kembali average vector.

Template:

```text
1 x 128 unit vector per palm identity
```

Jika ingin mengikuti multi-template app behavior, evaluation boleh menyediakan additional protocol, tetapi protocol utama harus konsisten dan terdokumentasi.

---

## 39. Query Matching

Untuk setiap Session-2 query:

```text
query embedding
-> cosine similarity terhadap semua enrollment templates
-> rank similarity descending
```

Identification prediction:

```text
argmax cosine similarity
```

---

## 40. Final Metrics

Minimal:

### Identification

```text
Rank-1 Accuracy
```

### Verification

```text
EER
ROC-AUC
TAR @ FAR 0.1%
TAR @ FAR 0.01% jika estimasi cukup stabil
```

### Distribution

```text
mean genuine similarity
std genuine similarity
mean impostor similarity
std impostor similarity
genuine-impostor gap
```

### Efficiency

```text
parameter count
checkpoint/artifact size
optional CPU inference latency
optional GPU inference latency
```

Final metrics masuk:

```text
trained_logs/<run_id>/metrics.json
```

kemudian diringkas ke `06-report-writing-source.md`.

---

## 41. Interpretation Rule

Jangan hanya menulis “accuracy tinggi”.

Interpretasi harus melihat:

- apakah genuine dan impostor distribution terpisah;
- EER;
- behavior pada low FAR;
- generalisasi cross-session;
- kemungkinan mismatch dataset vs webcam/local app.

Jika Rank-1 tinggi tetapi EER buruk, model belum tentu bagus untuk verification thresholding.

---

# PHASE 5 — FINAL ARTIFACT EXPORT

## 42. Checkpoint Source

Artifact final hanya boleh berasal dari:

```text
checkpoint_phase2_best.pth
```

atau checkpoint final lain yang secara eksplisit dipilih dari validation process.

Jangan export `last` checkpoint jika bukan best tanpa alasan tercatat.

---

## 43. Inference Wrapper

Artifact runtime hanya berisi:

```text
PalmNetLite backbone
-> L2 normalization
```

Tidak berisi:

- Phase 1 classifier;
- ArcFace head;
- optimizer;
- scheduler;
- training-only logging code.

Contract:

```text
input  : [B,3,112,112]
output : [B,128], L2 normalized
```

---

## 44. TorchScript Export Validation

Setelah export:

1. load eager backbone;
2. load best checkpoint;
3. wrap inference normalization;
4. run deterministic dummy/reference batch;
5. export `torch.jit.script` jika compatible;
6. fallback trace hanya jika justified;
7. reload artifact;
8. compare eager vs artifact output.

Acceptance:

```text
shape correct
finite output
L2 norm ~= 1
max_abs_diff <= 1e-4 (target)
```

---

## 45. Runtime Artifact Bundle

```text
backend/ml/models/
└── palmnet-lite-scratch/
    └── 1.0.0/
        ├── model.pt
        ├── manifest.json
        ├── threshold.json
        └── metrics.json
```

MobileFaceNet existing berada pada folder sibling-nya tetapi tidak berasal dari pipeline ini.

---

# TRAINED LOG SPECIFICATION

## 46. Directory Layout

```text
app/ml/artifacts/trained_logs/
└── palmnet-lite-scratch/
    └── <run_id>/
        ├── run.json
        ├── history.csv
        ├── metrics.json
        ├── phase1_summary.json
        ├── phase2_summary.json
        ├── validation_metrics.json
        ├── notes.md
        ├── tensorboard/
        └── figures/
```

Suggested `run_id`:

```text
YYYYMMDD-HHMMSS-seed<seed>
```

---

## 47. `run.json` Required Schema

Minimal fields:

```json
{
  "run_id": "20260812-221500-seed42",
  "model_id": "palmnet-lite-scratch",
  "architecture": "PalmNetLite",
  "architecture_version": "v1",
  "training_mode": "scratch",
  "seed": 42,
  "dataset": "Tongji Palmprint",
  "split_protocol": "400-train-cross-session-val-200-unseen-test",
  "input_shape": [3, 112, 112],
  "embedding_dim": 128,
  "initialization": "kaiming_random",
  "phase1": {},
  "phase2": {},
  "status": "running"
}
```

Saat selesai:

```text
status = completed
```

Jika gagal:

```text
status = failed
failure_reason = ...
```

Jangan menghapus failed run; failed run berguna sebagai evidence eksperimen.

---

## 48. `history.csv` Schema

Recommended columns:

```text
phase,
epoch,
train_loss,
train_accuracy,
val_loss,
val_accuracy,
mean_positive_cosine,
mean_negative_cosine,
cosine_gap,
val_eer,
learning_rate,
arcface_margin,
epoch_duration_sec
```

Field yang tidak relevan pada phase tertentu boleh kosong.

---

## 49. `notes.md`

Isi hanya catatan penting yang tidak cocok dalam numeric log, misalnya:

- augmentation dikurangi karena validation drop;
- run dihentikan karena NaN;
- batch size diubah karena VRAM;
- checkpoint dipilih manual dengan alasan khusus;
- ROI anomaly ditemukan.

Bukan diary panjang; catat keputusan yang memengaruhi interpretasi hasil.

---

# REPRODUCIBILITY

## 50. Seed Handling

Set minimal:

```text
python random
numpy
torch CPU
torch CUDA
DataLoader generator
```

Jika deterministic mode penuh menurunkan performa secara signifikan, cukup catat limitation; tidak harus memaksa semua CUDA op deterministic untuk tugas lokal.

Baseline first run:

```text
seed = 42
```

Jika waktu memungkinkan, final configuration dapat dijalankan minimal 2-3 seed untuk melihat stability, tetapi ini bukan acceptance criterion wajib.

---

## 51. Device Handling

Priority:

```text
CUDA jika tersedia
CPU fallback
```

Run metadata harus menyimpan minimal:

```text
device type
GPU name jika ada
PyTorch version
```

Optional:

```text
CUDA version
```

---

# FAILURE HANDLING

## 52. Failure Categories

### Data Failure

Contoh:

- file missing;
- label corrupt;
- empty dataset;
- split leakage.

Action:

```text
abort sebelum training
```

### Numerical Failure

Contoh:

- NaN loss;
- Inf embedding;
- exploding gradient.

Action:

```text
abort current run
save failure metadata
investigate LR/margin/implementation
```

### Learning Failure

Contoh:

- train accuracy random;
- validation tidak meningkat;
- cosine gap ~0.

Action:

```text
investigate pipeline dulu
baru tuning
```

### Overfitting

Pattern:

```text
train accuracy naik tinggi
validation stagnan/turun
```

Candidate response:

- stronger but realistic augmentation;
- weight decay adjustment;
- lower model capacity only if clearly needed;
- earlier checkpoint selection;
- evaluate dataset/session imbalance.

---

# CONFIG SPECIFICATION

## 53. Recommended Initial Config

```yaml
run:
  seed: 42

model:
  id: palmnet-lite-scratch
  architecture: PalmNetLite
  architecture_version: v1
  input_size: 112
  input_channels: 3
  embedding_dim: 128
  initialization: kaiming_random

phase1:
  epochs: 40
  batch_size: 64
  optimizer: adamw
  lr: 0.001
  weight_decay: 0.0001
  label_smoothing: 0.1
  warmup_epochs: 3
  scheduler: cosine
  min_lr: 0.000001
  grad_clip_norm: 5.0

phase2:
  epochs: 50
  batch_size: 64
  optimizer: adamw
  lr: 0.0001
  weight_decay: 0.0001
  arcface_margin: 0.30
  arcface_scale: 32.0
  margin_warmup_epochs: 8
  scheduler: cosine
  min_lr: 0.000001
  grad_clip_norm: 5.0

evaluation:
  metric: cosine_similarity
  threshold_calibration: eer
  far_targets: [0.001, 0.0001]
```

Config ini adalah baseline implementasi pertama, bukan klaim optimum.

---

# IMPLEMENTATION BOUNDARIES

## 54. Module Responsibilities

Target:

```text
models/
    architecture only

initialization/
    random initialization policy

data/
    dataset, transform, split

training/
    generic loop + phase orchestration

losses/
    ArcFace

evaluation/
    embedding extraction + metrics

logging/
    run/history/summary persistence

artifacts/
    TorchScript export + manifest
```

Training script tidak boleh memiliki seluruh implementation dalam satu file besar.

---

## 55. Recommended CLI Separation

```bash
python scripts/validate_data.py --config configs/palmnet_lite_scratch.yaml

python scripts/train_palmnet.py --config configs/palmnet_lite_scratch.yaml

python scripts/evaluate_palmnet.py \
  --run-id <run_id> \
  --checkpoint best

python scripts/export_artifact.py \
  --run-id <run_id> \
  --version 1.0.0 \
  --deploy-backend
```

Training orchestration boleh menjalankan Phase 1 dan Phase 2 dalam satu command, tetapi internal state/log tetap dibedakan per phase.

---

# REPORT INTEGRATION

## 56. Data yang Harus Masuk Living Report

Setelah final run dipilih, update `06-report-writing-source.md` dengan:

### Architecture

```text
exact parameter count
architecture version
input/output
```

### Phase 1

```text
hyperparameter
best epoch
best val accuracy
train/val curve interpretation
```

### Phase 2

```text
margin
scale
best epoch
positive cosine
negative cosine
cosine gap
validation EER
```

### Final Evaluation

```text
Rank-1
EER
ROC-AUC
TAR@FAR
score distribution
```

### Artifact

```text
TorchScript size
input/output verification
local inference latency jika diukur
```

### Integration

```text
backend registry loaded
frontend model selector works
enrollment using PalmNet works
identification using PalmNet works
```

Jangan copy raw log seluruhnya ke laporan. Living report berisi **ringkasan dan interpretasi evidence**.

---

## 57. Definition of Done

Training subsystem dianggap selesai jika:

- Phase 0 sanity pass;
- Phase 1 dapat training scratch end-to-end;
- Phase 1 best checkpoint tercatat;
- Phase 2 dapat melanjutkan dari own Phase-1 checkpoint;
- ArcFace training stabil;
- best Phase-2 checkpoint tercatat;
- validation threshold berhasil dikalibrasi;
- final hold-out evaluation selesai;
- metrics dan figures tersimpan;
- trained log lengkap;
- TorchScript artifact berhasil diekspor;
- artifact dapat dimuat backend;
- hasil final diperbarui ke living report.

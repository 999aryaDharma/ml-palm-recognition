# Palm Recognition Assignment Documentation

Dokumentasi ini menjadi acuan pengembangan proyek `ml-palm-recognition` untuk kebutuhan tugas mata kuliah AI.

## Tujuan

Proyek akan memiliki dua model palm recognition yang dapat digunakan secara bergantian pada aplikasi lokal:

1. **MobileFaceNet Pretrained** — model existing berbasis MobileFaceNet dengan pretrained weights dan fine-tuning pada data palm.
2. **PalmNet-Lite Scratch** — model baru yang dirancang sendiri dengan prinsip arsitektur lightweight yang terinspirasi MobileFaceNet, tetapi seluruh bobot diinisialisasi secara acak dan dilatih dari awal menggunakan dataset palm.

PalmNet-Lite Scratch adalah model utama yang memenuhi kebutuhan tugas untuk membangun jaringan saraf tiruan dari awal tanpa pretrained weights.

## Scope

Scope proyek dibatasi untuk kebutuhan tugas dan penggunaan lokal.

Termasuk dalam scope:

- training dan evaluasi dua model;
- model baru PalmNet-Lite dari scratch;
- penggunaan MobileFaceNet existing sebagai model pretrained pembanding;
- model selector pada aplikasi;
- model registry di backend;
- model-specific threshold dan biometric templates;
- trained logs terpisah untuk setiap model;
- export artifact TorchScript;
- evaluasi biometric;
- dokumentasi ilmiah arsitektur dan metodologi;
- codebase ML dan backend yang modular.

Tidak termasuk dalam scope:

- deployment cloud;
- multi-tenant atau production infrastructure;
- user authentication kompleks;
- CI/CD production;
- backward compatibility data lama;
- migrasi user existing;
- distributed model serving;
- browser-side inference;
- mobile deployment;
- model quantization untuk production.

Karena aplikasi hanya digunakan secara lokal untuk tugas, database SQLite dapat di-reset ketika terjadi perubahan schema yang signifikan.

## Dokumen

- `01-project-scope.md` — tujuan, batasan, definisi kedua model, dan acceptance criteria.
- `02-model-architecture.md` — desain PalmNet-Lite dan hubungan arsitekturalnya dengan MobileFaceNet.
- `03-training-and-artifacts.md` — pipeline training, trained logs, checkpoint, evaluation, dan export artifact.
- `04-app-two-model-integration.md` — bagaimana backend dan frontend memilih serta menjalankan dua model.
- `05-codebase-modularity.md` — target struktur codebase modular dan tanggung jawab setiap modul.
- `06-scientific-report-reference.md` — bahan utama untuk penulisan laporan ilmiah, termasuk alasan desain, arsitektur, pipeline, metodologi, dan evaluasi.

## Prinsip Utama

1. PalmNet-Lite tidak boleh memuat pretrained weights apa pun.
2. MobileFaceNet pretrained yang sudah ada tetap dipertahankan.
3. Kedua model harus mempunyai kontrak inference yang sama: input `3 x 112 x 112`, output embedding `128-D` yang L2-normalized.
4. Embedding dari dua model tidak boleh dibandingkan atau dicampur.
5. Setiap training run harus memiliki log dan metadata yang dapat ditelusuri.
6. Export artifact harus dapat langsung dimuat backend tanpa bergantung pada source class model.
7. Frontend hanya memilih `model_id`; inference tetap dilakukan backend.
8. Struktur source code harus modular agar training, evaluation, export, dan runtime tidak saling bergantung secara berlebihan.

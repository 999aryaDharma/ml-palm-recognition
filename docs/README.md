# Palm Recognition Assignment Documentation

Dokumentasi ini menjadi acuan pengembangan proyek `ml-palm-recognition` untuk kebutuhan tugas mata kuliah AI dan penggunaan lokal.

## Tujuan

Aplikasi lokal akan memiliki dua model palm recognition yang dapat dipilih saat runtime:

1. **MobileFaceNet Pretrained** — artifact existing yang sudah selesai dilatih, sudah memiliki performa yang baik, dan **tidak akan dilatih ulang dalam scope tugas ini**.
2. **PalmNet-Lite Scratch** — model baru yang dirancang sendiri dengan prinsip lightweight CNN yang terinspirasi MobileFaceNet, tetapi seluruh bobot diinisialisasi secara acak dan dilatih dari awal menggunakan dataset palm.

Fokus akademik, eksperimen, training, evaluasi, dan penulisan laporan adalah **PalmNet-Lite Scratch**.

MobileFaceNet dipertahankan untuk kebutuhan aplikasi agar user dapat memilih model inference. Untuk sementara MobileFaceNet **bukan bagian wajib dari eksperimen pembandingan pada laporan**. Perbandingan dapat ditambahkan kemudian jika memang dibutuhkan.

## Scope

Scope proyek dibatasi untuk kebutuhan tugas dan penggunaan lokal.

Termasuk dalam scope:

- merancang PalmNet-Lite dari scratch;
- training dan evaluasi PalmNet-Lite;
- mempertahankan artifact MobileFaceNet existing tanpa retraining;
- model selector pada aplikasi;
- model registry di backend;
- model-specific threshold dan biometric templates;
- structured trained logs untuk setiap training run PalmNet-Lite;
- mempertahankan metadata artifact MobileFaceNet existing;
- export artifact TorchScript PalmNet-Lite;
- evaluasi biometric PalmNet-Lite;
- dokumentasi arsitektur dan metodologi;
- codebase ML dan backend yang modular;
- satu living document sebagai sumber penulisan laporan ilmiah.

Tidak termasuk dalam scope:

- retraining MobileFaceNet;
- tuning MobileFaceNet;
- kewajiban membandingkan PalmNet-Lite dengan MobileFaceNet;
- deployment cloud;
- production infrastructure;
- existing-user migration;
- backward compatibility data lama;
- browser-side inference;
- mobile deployment;
- production-grade model serving.

Karena aplikasi hanya digunakan secara lokal untuk tugas, database SQLite dapat di-reset jika schema berubah. Tidak perlu menghabiskan waktu untuk migration strategy user lama.

## Dokumen

- `01-project-scope.md` — tujuan, batasan, posisi MobileFaceNet existing, PalmNet-Lite scratch, dan acceptance criteria.
- `02-model-architecture.md` — desain PalmNet-Lite dan alasan arsitekturnya.
- `03-training-and-artifacts.md` — pipeline training PalmNet-Lite, trained logs, checkpoint, evaluation, dan export artifact; MobileFaceNet hanya dibahas sebagai frozen artifact.
- `04-app-two-model-integration.md` — bagaimana backend dan frontend dapat memilih MobileFaceNet atau PalmNet-Lite.
- `05-codebase-modularity.md` — target struktur modular dengan pemisahan research/training code dan runtime model loading.
- `06-report-writing-source.md` — **single living document** untuk bahan penulisan laporan ilmiah. Dokumen ini diperbarui secara berkala setelah training, evaluasi, dan integrasi aplikasi.

## Prinsip Utama

1. PalmNet-Lite tidak boleh memuat pretrained weights apa pun.
2. MobileFaceNet existing diperlakukan sebagai frozen runtime artifact dan tidak masuk pipeline training baru.
3. Fokus eksperimen dan laporan utama adalah PalmNet-Lite.
4. Perbandingan dengan MobileFaceNet bersifat opsional dan dapat ditambahkan kemudian tanpa mengubah tujuan utama penelitian.
5. Kedua model harus mempunyai kontrak inference yang konsisten: input `3 x 112 x 112`, output embedding `128-D` yang L2-normalized.
6. Embedding dari dua model tidak boleh dibandingkan atau dicampur untuk proses matching.
7. Setiap training run PalmNet-Lite harus memiliki log dan metadata yang dapat ditelusuri.
8. Export artifact PalmNet-Lite harus dapat langsung dimuat backend melalui runtime abstraction yang sama dengan MobileFaceNet.
9. Frontend hanya memilih `model_id`; inference tetap dilakukan backend.
10. Struktur source code harus modular agar architecture, data, training, evaluation, artifact export, dan runtime integration memiliki tanggung jawab yang jelas.

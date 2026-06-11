# Frontend UX Refresh

Patch untuk membuat UX lebih smooth, hilangkan rasa "sat-set / auto-reload",
dan kasih feedback yang jelas di setiap aksi.

## Cara pasang (drop-in)

1. **Tambah file baru:**
   - `js/ux.js` → letakkan di folder `js/` (sejajar dengan `utils.js`)
   - `css/ux.css` → letakkan di folder `css/`

2. **Tambahkan `ux.css` di semua HTML yang menggunakan halaman ini**
   (taruh setelah `components.css`):
   ```html
   <link rel="stylesheet" href="css/ux.css">
   ```
   Untuk halaman di folder `demo/`, pakai `../css/ux.css`.

3. **Replace 7 file di `js/pages/`** dengan versi dari folder ini:
   - `users.js`
   - `user-detail.js`
   - `enroll.js`
   - `payment.js`
   - `attendance.js`
   - `access.js`
   - `patient.js`

   API, HTML, dan komponen lain (modal, toast, scanner) **tidak perlu** diubah.

## Apa yang berubah (UX-wise)

### 1. Hilangkan auto-reload / redirect mendadak
| File | Sebelum | Sesudah |
|------|---------|---------|
| `payment.js` | `window.location.reload()` setelah "Selesai" | Reset state in-place, scanner siap pakai lagi, log terminal tidak hilang |
| `enroll.js` | `window.location.href = "index.html"` setelah sukses & cancel | `history.back()` via helper `smoothBack()` |
| `user-detail.js` | `setTimeout(() => location.href = "users.html", 2000)` saat ID invalid / delete sukses | Inline empty state dengan tombol "Kembali"; setelah delete: fade-out + history.back |
| `enroll.js` | Auto-restart 5 detik saat error ("Mengulang dalam 5 detik...") | Tombol "Coba dari Awal" / "Kembali" — **user yang putuskan** |

### 2. Feedback loading konsisten
- `js/ux.js` punya `withLoading(btn, label, asyncFn)` → spinner + disable + auto-restore. Mencegah double-click & kasih sinyal "lagi diproses".
- Daftar pengguna & daftar otorisasi: skeleton shimmer (bukan layar kosong / teks "Memuat...").
- Patient card: skeleton card khusus saat fetch rekam medis.
- Counter absensi: animasi "bump" saat naik.

### 3. Transisi step yang smooth
- Step enroll (name → capture → verifying → success) pakai fade-in (`ux-step`).
- Patient card masuk dengan fade animation.
- Receipt payment muncul dengan fade.
- Entry baru di access log & attendance list: slide-in dengan highlight sebentar (mata user otomatis tertarik ke baris baru).
- Hapus kartu user: fade-out + collapse (optimistic, rollback kalau API gagal).

### 4. Error handling yang manusiawi
- Payment gagal → modal "Coba Lagi" / "Batal" (bukan toast lalu reset diam-diam).
- Attendance gagal submit → idem.
- Patient fetch gagal → tombol "Coba Lagi" inline di panel.
- Authorized list gagal load → tombol "Coba lagi" inline.
- Camera permission denied → toast jelas + label tombol berubah jadi "Coba Aktifkan Scanner Lagi".

### 5. Bonus
- Tombol opsional `#btn-success-enroll-another` di step success enroll: kalau kamu tambahkan tombol dengan ID itu di HTML, user bisa langsung daftarkan pengguna lain tanpa reload halaman.
- `prefers-reduced-motion` dihormati: animasi mati otomatis bagi user yang setting aksesibilitas tinggi.

## Catatan teknis

- `enroll.js`: dead-code retry loop di `addTemplate` (yang unreachable karena `throw` mendahuluinya) sudah dirapikan jadi retry yang benar-benar jalan (max 3x dengan jeda 1.2s).
- Semua fungsi di `ux.js` no-op-safe: kalau elemen `null`, tidak akan throw.
- File rewrites tetap kompatibel dengan API existing (`apiFetch`, `PalmScanner`, `showModal`, `toast`, dll.) — tidak ada breaking change ke kontrak komponen.

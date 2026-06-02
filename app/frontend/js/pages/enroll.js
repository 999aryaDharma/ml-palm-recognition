// ============================================================
// js/pages/enroll.js — Enrollment page logic (FIXED v2)
//
// FLOW YANG BENAR (setelah perbaikan):
//
//   [Step 1] User isi nama
//            → Validasi: nama tidak boleh kosong
//            → BELUM ke backend sama sekali
//
//   [Step 2] Kamera aktif, auto-capture mulai
//            → Setiap frame dikirim ke POST /validate-frame
//            → Jika quality gate GAGAL: tampilkan hint, ulangi capture
//            → Jika quality gate LOLOS untuk pertama kali:
//                 1. POST /users  ← createUser() dipanggil SEKARANG
//                 2. POST /users/{id}/templates dengan blob yang sama
//                 3. sampleCount = 1, isUserCreated = true
//
//   [Step 3] Capture sampel 2-5
//            → POST /users/{id}/templates langsung (user sudah ada)
//            → Progress bertambah hanya jika berhasil
//
//   [Step 4] Verifikasi: POST /identify
//            → Jika cocok dengan user yang baru dibuat → Sukses
//            → Jika gagal → retry capture (user & template tetap ada)
//
// MENGAPA INI BENAR:
//   - User di backend HANYA dibuat setelah terbukti telapak bisa di-detect
//   - Tidak ada lagi user dengan 0 template "hantu"
//   - Kamera harus menyala dan telapak terdeteksi sebelum apapun disimpan
//   - Cancel setelah user dibuat = hapus user dari backend (cleanup)
// ============================================================

import { mountNavbar } from "../components/navbar.js";
import { WebcamCapture } from "../components/webcam.js";
import { createUser, addTemplate, deleteUser } from "../api/users.js";
import { identify } from "../api/identify.js";
import { apiFetch, BASE_URL } from "../api/client.js";
import { toast } from "../components/toast.js";
import { QUALITY_HINTS, sleep } from "../utils.js";

// ── State ──────────────────────────────────────────────────
let webcam = null;
let currentUserId = null; // null SAMPAI scan pertama berhasil + createUser() dipanggil
let currentUserName = "";
let pendingName = ""; // nama dari step 1, belum dikirim ke backend
let sampleCount = 0;
const MAX_SAMPLES = 5;
let isCapturing = false;
let isUserCreated = false; // guard: true setelah createUser() berhasil
let steps = {}; // mapping untuk DOM section langkah-langkah

const ENROLL_HINTS = [
  "🔍 Tahan posisi telapak lurus dan tegak di depan kamera",
  "🔄 Sampel 2/5: Miringkan telapak sedikit ke arah kiri",
  "🔄 Sampel 3/5: Miringkan telapak sedikit ke arah kanan",
  "⬆️ Sampel 4/5: Majukan telapak sedikit mendekati kamera",
  "⬇️ Sampel 5/5: Mundurkan telapak sedikit menjauhi kamera",
];

// ── DOM Elements ───────────────────────────────────────────
let inputName, btnToCapture, btnCancelCapture;
let videoEl, scannerHint, scannerLoading, scanline;
let sampleBadge, sampleDots, successName;

// ── Init ───────────────────────────────────────────────────
async function init() {
  mountNavbar();

  // Retrieve elements inside init
  inputName = document.getElementById("input-name");
  btnToCapture = document.getElementById("btn-to-capture");
  btnCancelCapture = document.getElementById("btn-cancel-capture");
  videoEl = document.getElementById("enroll-video");
  scannerHint = document.getElementById("scanner-hint");
  scannerLoading = document.getElementById("scanner-loading");
  scanline = document.getElementById("scanline");
  sampleBadge = document.getElementById("sample-count-badge");
  sampleDots = document.querySelectorAll(".step-dot");
  successName = document.getElementById("success-name");

  steps = {
    name: document.getElementById("step-name"),
    capture: document.getElementById("step-capture"),
    verifying: document.getElementById("step-verifying"),
    success: document.getElementById("step-success"),
  };

  if (!inputName || !btnToCapture) {
    console.error("[Enroll] Essential DOM elements not found!");
    return;
  }

  // Fix: Handle form submission to prevent page reload
  const formName = document.getElementById("form-name");
  if (formName) {
    formName.addEventListener("submit", (e) => {
      e.preventDefault();
      goToCaptureStep();
    });
  }

  btnCancelCapture?.addEventListener("click", cancelEnrollment);

  inputName.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      goToCaptureStep();
    }
  });

  inputName.focus();

  window.addEventListener("beforeunload", cleanupOnExit);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && webcam) webcam.stop();
  });
}

function showStep(stepName) {
  Object.keys(steps).forEach((key) => {
    steps[key].classList.toggle("hidden", key !== stepName);
  });
}

// ── New: Handle cancel dengan confirmation ketat ──
async function handleCancelCapture() {
  // GUARD: Jika sudah ada template, warning LEBIH ketat
  if (sampleCount > 0) {
    let message = "";

    if (sampleCount < MAX_SAMPLES) {
      message = `⚠️  PERINGATAN: Anda sudah mengumpulkan ${sampleCount}/5 sampel.\n\n`;
      message += `Jika Anda membatalkan SEKARANG:\n`;
      message += `  • Semua ${sampleCount} template akan DIHAPUS\n`;
      message += `  • Anda harus mulai dari awal\n`;
      message += `  • Tinggal ${MAX_SAMPLES - sampleCount} sampel lagi!\n\n`;
      message += `Yakin ingin membatalkan?`;
    } else {
      message = `⚠️  CRITICAL: Anda sudah 5/5 sampel!\n\n`;
      message += `Jika Anda membatalkan:\n`;
      message += `  • Semua 5 template akan DIHAPUS\n`;
      message += `  • Enrollment DIBATALKAN\n\n`;
      message += `LANJUTKAN KE VERIFIKASI! Tinggal 1 langkah!\n\n`;
      message += `Yakin ingin membatalkan?`;
    }

    const confirmed = confirm(message);
    if (!confirmed) return;
  }

  await cancelEnrollment();
}

// ── STEP 1 → STEP 2 ───────────────────────────────────────
// Hanya validasi nama secara lokal, langsung aktifkan kamera.
// createUser() BELUM dipanggil di sini.
async function goToCaptureStep() {
  const name = inputName.value.trim();

  if (!name || name.length < 2) {
    toast.warning("Nama minimal 2 karakter.");
    inputName.focus();
    return;
  }

  // GUARD: Ensure fresh state
  if (isUserCreated && currentUserId) {
    console.warn("[Enroll] goToCaptureStep called but user already created");
    return;
  }

  pendingName = name;
  currentUserName = name;

  btnToCapture.disabled = true;
  btnToCapture.textContent = "Menyiapkan kamera...";

  showStep("capture");
  updateProgress();
  updateSampleStatus();
  await initWebcam();
}

// ── Kamera ─────────────────────────────────────────────────
async function initWebcam() {
  setHint("🎥 Menyalakan kamera...");

  // Bersihkan retry button lama jika ada
  document.getElementById("btn-retry-camera")?.remove();

  try {
    webcam = new WebcamCapture(videoEl, {
      onCapture: handleCapture,
      captureInterval: 1500, // Faster polling, but we gate it with isCapturing
    });

    await webcam.start();
    setHint(ENROLL_HINTS[0] || "🖐 Tunjukkan telapak tangan ke kamera");
    webcam.startAutoCapture();
  } catch (err) {
    setHint("❌ Tidak dapat mengakses kamera", true);
    toast.error(
      "Gagal mengakses kamera. Pastikan izin sudah diberikan di browser.",
    );
    showRetryButton();
  }
}

// ── Handle Capture ─────────────────────────────────────────
async function handleCapture(blob) {
  if (isCapturing || sampleCount >= MAX_SAMPLES) return;
  isCapturing = true;

  scannerLoading.classList.remove("hidden");
  scanline.classList.remove("hidden");

  try {
    if (!isUserCreated) {
      // ============================================================
      // FASE 1: User belum ada di backend.
      // Langkah: validate-frame dulu → jika lolos → createUser → addTemplate
      // Ini memastikan user HANYA dibuat jika telapak memang bisa dideteksi.
      // ============================================================
      setHint("🔍 Mendeteksi telapak...");

      // Cek quality gate tanpa menyimpan apapun
      const validateForm = new FormData();
      validateForm.append("image", blob, "frame.jpg");

      let validateOk = false;
      try {
        await apiFetch("/validate-frame", {
          method: "POST",
          body: validateForm,
        });
        validateOk = true;
      } catch (valErr) {
        // Quality gate gagal — tampilkan hint, jangan buat user
        const hint =
          QUALITY_HINTS[valErr.error] || "Arahkan telapak tangan ke kamera";
        setHint(hint);
        isCapturing = false;
        scannerLoading.classList.add("hidden");
        return;
      }

      if (!validateOk) {
        isCapturing = false;
        scannerLoading.classList.add("hidden");
        return;
      }

      // Frame valid → sekarang baru buat user di backend
      setHint("⏳ Mendaftarkan pengguna...");
      let newUser;
      try {
        newUser = await createUser(pendingName);
      } catch (createErr) {
        setHint("❌ Gagal terhubung ke server. Cek backend.", true);
        toast.error("Backend tidak dapat dihubungi.");
        isCapturing = false;
        scannerLoading.classList.add("hidden");
        return;
      }

      // Upload template dengan blob yang sama (sudah terbukti valid)
      try {
        await addTemplate(newUser.id, blob);
        currentUserId = newUser.id;
        isUserCreated = true;
        sampleCount = 1;
        updateProgress();

        // UX: Freeze kamera sesaat untuk feedback sukses
        videoEl.pause();
        webcam.stopAutoCapture();
        scanline.classList.add("hidden");

        toast.success(`Template 1/5 berhasil disimpan.`, "Berhasil");
        setHint(`✅ Sampel 1/5 OK!`);
        await sleep(1500); // Tahan freeze

        // Resume
        const nextHint = ENROLL_HINTS[sampleCount] || "Tahan posisi...";
        setHint(nextHint);
        videoEl.play();
        webcam.startAutoCapture();

        // Jeda tambahan agar user sempat mengubah pose tangan sebelum kamera mengambil sampel lagi
        await sleep(2000);
      } catch (templateErr) {
        // Aneh tapi bisa terjadi (race condition): validasi lolos tapi addTemplate gagal
        // Hapus user yang baru dibuat agar tidak jadi "hantu"
        await deleteUser(newUser.id).catch(() => {});
        const hint =
          QUALITY_HINTS[templateErr.error] || "Posisikan ulang telapak";
        setHint(hint);
        isCapturing = false;
        scannerLoading.classList.add("hidden");
        return;
      }
    } else {
      // ============================================================
      // FASE 2: User sudah ada. Langsung upload template berikutnya.
      // ============================================================
      setHint(
        `📷 Mengekstrak sampel ${sampleCount + 1}/5... (Usaha ${sampleAttempts}/${MAX_ATTEMPTS_PER_SAMPLE})`,
      );

      try {
        const addTemplateResp = await addTemplate(currentUserId, blob);
        const qualityScore = addTemplateResp.quality_score || lastQualityScore;
        updateQualityDisplay(qualityScore);

        sampleCount++;
        sampleAttempts = 0;
        updateProgress();
        updateSampleStatus();

        // PENTING: Stop webcam SEBELUM freeze untuk mencegah race condition
        webcam.stopAutoCapture();

        // UX: Freeze kamera sesaat
        videoEl.pause();
        scanline.classList.add("hidden");

        // GUARD KETAT: Check sampel == 5
        if (sampleCount >= MAX_SAMPLES) {
          if (sampleCount !== MAX_SAMPLES) {
            console.warn(
              `[Enroll] Anomali: sampleCount=${sampleCount} > MAX_SAMPLES=${MAX_SAMPLES}`,
            );
          }

          setHint(
            "✅ 5/5 Sampel terkumpul! Mempersiapkan verifikasi...",
            "success",
          );
          toast.success("5 template berhasil dikumpulkan.", "Sukses");

          await sleep(2000);
          videoEl.play();

          // PENTING: Verification hanya dari sini
          await runVerification();
        } else {
          setHint(
            `✅ Sampel ${sampleCount}/5 OK! Quality: ${Math.round(qualityScore * 100)}%`,
          );
          toast.success(`✓ Sampel ${sampleCount}/5 berhasil!`, "Berhasil");

          await sleep(1500);

          const nextHint = ENROLL_HINTS[sampleCount] || "Tahan posisi...";
          setHint(nextHint);
          updateSampleStatus();
          videoEl.play();

          // PENTING: Restart SETELAH freeze
          webcam.startAutoCapture();

          await sleep(2000);
        }
      } catch (err) {
        const qualityScore = err.quality_score || 0;
        updateQualityDisplay(qualityScore);

        const hint =
          QUALITY_HINTS[err.error] || "Geser tangan sedikit dan tahan";
        setHint(
          `❌ ${hint} (Usaha ${sampleAttempts}/${MAX_ATTEMPTS_PER_SAMPLE})`,
        );
      }
    }
  } finally {
    // Only un-flag capturing if we are still collecting samples
    if (sampleCount < MAX_SAMPLES) {
      isCapturing = false;
      scannerLoading.classList.add("hidden");
      if (webcam?.isRunning && videoEl.paused === false) {
        scanline.classList.remove("hidden");
      }
    }
  }
}

// ── Verifikasi Akhir ───────────────────────────────────────
async function runVerification() {
  // GUARD 1: Pastikan benar-benar sudah 5 sampel
  if (sampleCount !== MAX_SAMPLES) {
    console.error(
      `[Enroll] CRITICAL: runVerification called but sampleCount=${sampleCount} !== MAX_SAMPLES=${MAX_SAMPLES}`,
    );
    toast.error(
      `ERROR: Sampel belum lengkap (${sampleCount}/${MAX_SAMPLES}). Lanjutkan capture.`,
    );
    showStep("capture");
    isCapturing = false;
    webcam.startAutoCapture();
    return;
  }

  // GUARD 2: Verifikasi ke backend bahwa benar-benar tersimpan 5 template
  try {
    const verifyReadyEndpoint = `/users/${currentUserId}/verify-ready`;
    try {
      const readyCheck = await apiFetch(verifyReadyEndpoint);

      if (!readyCheck.ready) {
        const backendCount = readyCheck.template_count || 0;
        console.warn(
          `[Enroll] Backend verify-ready returned false: only ${backendCount}/${readyCheck.required} templates`,
        );
        sampleCount = backendCount;
        updateProgress();
        toast.warning(
          `Template belum cukup di sistem (${backendCount}/${MAX_SAMPLES}). Lanjutkan capture.`,
        );
        showStep("capture");
        isCapturing = false;
        updateSampleStatus();
        webcam.startAutoCapture();
        return;
      }
    } catch (checkErr) {
      console.warn(
        "[Enroll] verify-ready endpoint not available, proceeding without backend check",
      );
    }
  } catch (err) {
    console.error("[Enroll] Error during ready check:", err);
  }

  showStep("verifying");
  await sleep(1200);

  try {
    let verifyBlob = null;
    if (webcam?.isRunning) {
      verifyBlob = await webcam.captureFrame();
    }

    // GUARD 3: Frame HARUS berhasil diambil
    if (!verifyBlob) {
      throw new Error(
        "Kamera tidak responsif. Silakan periksa koneksi dan izin kamera, lalu coba lagi.",
      );
    }

    const result = await identify(verifyBlob);

    if (result.status === "identified" && result.user?.id === currentUserId) {
      webcam.stop();
      successName.textContent = currentUserName;
      showStep("success");
      toast.success(
        `Enrollment berhasil! 5 template terverifikasi untuk ${currentUserName}.`,
        "Berhasil",
      );
    } else if (result.status === "identified") {
      throw new Error(
        `Verifikasi gagal: terdeteksi sebagai ${result.user?.name || "orang lain"}. Template tidak konsisten.`,
      );
    } else {
      throw new Error(
        "Template belum terverifikasi dengan baik. Kualitas tidak konsisten antar variasi posisi.",
      );
    }
  } catch (err) {
    console.error("[Enroll] Verification error:", err);
    toast.error(err.message || "Verifikasi gagal");

    showStep("verificationFailed");
  }
}

// ── Progress UI ────────────────────────────────────────────
function updateProgress() {
  if (sampleBadge) {
    sampleBadge.textContent = `${sampleCount} / ${MAX_SAMPLES}`;
    sampleBadge.className = "badge";
    if (sampleCount === 0) {
      sampleBadge.classList.add("badge--processing");
    } else if (sampleCount >= MAX_SAMPLES) {
      sampleBadge.classList.add("badge--identified");
    } else {
      sampleBadge.classList.add("badge--scanning");
    }
  }

  sampleDots.forEach((dot, idx) => {
    dot.classList.remove("active", "completed");
    if (idx < sampleCount) {
      dot.classList.add("completed");
    } else if (idx === sampleCount && sampleCount < MAX_SAMPLES) {
      dot.classList.add("active");
    }
  });
}

function setHint(text, isError = false) {
  if (!scannerHint) return;
  scannerHint.textContent = text;
  scannerHint.style.color = isError ? "var(--color-coral)" : "";
}

function showRetryButton() {
  document.getElementById("btn-retry-camera")?.remove();
  const btn = document.createElement("button");
  btn.id = "btn-retry-camera";
  btn.className = "btn btn--secondary";
  btn.textContent = "Coba Lagi";
  btn.style.marginTop = "var(--space-4)";
  btn.onclick = async () => {
    btn.remove();
    await initWebcam();
  };
  const scannerEl = document.getElementById("scanner-container");
  scannerEl?.insertAdjacentElement("afterend", btn);
}

// ── Cancel ─────────────────────────────────────────────────
async function cancelEnrollment() {
  webcam?.stop();

  if (isUserCreated && currentUserId) {
    // Hapus user dari backend — enrollment dibatalkan
    await deleteUser(currentUserId).catch(() => {});
  }

  resetLocalState();
  window.history.length > 1
    ? window.history.back()
    : (window.location.href = "index.html");
}

// ── Cleanup saat tab ditutup / navigasi pergi ──────────────
function cleanupOnExit() {
  webcam?.stop();

  // Hanya hapus jika user terlanjur dibuat tapi BELUM ada template sama sekali
  // Jika sudah ada > 0 template, biarkan saja (user bisa hapus manual nanti)
  // Ini menghindari user terhapus saat berpindah ke halaman detail/success.
  if (isUserCreated && currentUserId && sampleCount === 0) {
    const url = `${BASE_URL}/users/${currentUserId}`;
    if (navigator.sendBeacon) {
      // sendBeacon tidak support DELETE secara standar,
      // tapi kita coba fetch dengan keepalive sebagai alternatif modern
      fetch(url, { method: "DELETE", keepalive: true }).catch(() => {});
    }
  }
}

function resetLocalState() {
  currentUserId = null;
  currentUserName = "";
  pendingName = "";
  sampleCount = 0;
  isUserCreated = false;
  isCapturing = false;
  updateProgress();
  btnToCapture.disabled = false;
  btnToCapture.textContent = "Mulai Enrollment";
  inputName.value = "";
}

// ── Run ────────────────────────────────────────────────────
init();

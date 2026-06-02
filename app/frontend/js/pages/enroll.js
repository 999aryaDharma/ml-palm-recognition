// ============================================================
// js/pages/enroll.js — Enrollment page logic (LOCAL BUFFER)
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
let currentUserId = null;
let currentUserName = "";
let pendingName = "";
let sampleCount = 0;
const MAX_SAMPLES = 5;
let isCapturing = false;
let isUserCreated = false;
let capturedBlobs = []; // ARRAY LOKAL: Menyimpan 5 foto di RAM terlebih dahulu

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
let sampleBadge, sampleDots, successName, steps;

async function init() {
  mountNavbar();

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

  const formName = document.getElementById("form-name");
  if (formName) {
    formName.addEventListener("submit", (e) => {
      e.preventDefault();
      goToCaptureStep();
    });
  }

  btnCancelCapture?.addEventListener("click", cancelEnrollment);
  window.addEventListener("beforeunload", cleanupOnExit);
}

function showStep(stepName) {
  Object.keys(steps).forEach((key) => {
    if (steps[key]) steps[key].classList.toggle("hidden", key !== stepName);
  });
}

async function goToCaptureStep() {
  const name = inputName.value.trim();
  if (!name || name.length < 2) {
    toast.warning("Nama minimal 2 karakter.");
    return;
  }

  pendingName = name;
  currentUserName = name;
  btnToCapture.disabled = true;

  showStep("capture");
  updateProgress();
  await initWebcam();
}

async function initWebcam() {
  setHint("🎥 Menyalakan kamera...");
  try {
    webcam = new WebcamCapture(videoEl, {
      onCapture: handleCapture,
      captureInterval: 1200,
    });
    await webcam.start();
    setHint(ENROLL_HINTS[0]);
    webcam.startAutoCapture();
  } catch (err) {
    setHint("❌ Tidak dapat mengakses kamera", true);
    toast.error("Gagal mengakses kamera.");
  }
}

async function handleCapture(blob) {
  if (isCapturing || capturedBlobs.length >= MAX_SAMPLES) return;
  isCapturing = true;

  scannerLoading.classList.remove("hidden");
  scanline.classList.remove("hidden");

  try {
    setHint(`🔍 Menganalisis kualitas sampel ${capturedBlobs.length + 1}/5...`);

    // UJI KUALITAS SAJA, Jangan simpan ke database
    const validateForm = new FormData();
    validateForm.append("image", blob, "frame.jpg");

    try {
      await apiFetch("/validate-frame", { method: "POST", body: validateForm });
    } catch (valErr) {
      setHint(
        QUALITY_HINTS[valErr.error] || "Arahkan telapak tangan ke kamera",
      );
      return;
    }

    // JIKA LOLOS -> Simpan frame di RAM (Local Buffer)
    capturedBlobs.push(blob);
    sampleCount = capturedBlobs.length;
    updateProgress();

    if (sampleCount >= MAX_SAMPLES) {
      setHint("✅ 5/5 Sampel terkumpul! Menyimpan ke server...", "success");
      scanline.classList.add("hidden");

      webcam.stopAutoCapture(); // Hentikan kamera
      await sleep(1000);

      await finalizeEnrollment(); // PROSES UPLOAD DIMULAI
    } else {
      setHint(`✅ Sampel ${sampleCount}/5 OK!`);
      await sleep(1000);
      setHint(ENROLL_HINTS[sampleCount] || "Tahan posisi...");
      await sleep(1500); // Waktu agar user mengubah pose
    }
  } finally {
    if (sampleCount < MAX_SAMPLES) {
      isCapturing = false;
      scannerLoading.classList.add("hidden");
      if (webcam?.isRunning) scanline.classList.remove("hidden");
    }
  }
}

// ── PROSES UPLOAD (Hanya berjalan jika sudah pasti ada 5 gambar) ──
async function finalizeEnrollment() {
  showStep("verifying");
  setHint("Mendaftarkan pengguna ke server...", "info");

  try {
    // 1. BUAT USER DI DATABASE (Dijamin aman dari duplikat)
    let newUser;
    try {
      newUser = await createUser(pendingName);
    } catch (createErr) {
      if (createErr.status === 409 || createErr.error === "user_exists") {
        toast.warning(`Nama "${pendingName}" sudah terdaftar. Silakan ganti.`);
        cancelEnrollment();
        return;
      }
      throw new Error("Gagal membuat profil di server.");
    }

    currentUserId = newUser.id;
    isUserCreated = true;

    // 2. UNGGAH 5 TEMPLATE SECARA BERURUTAN
    for (let i = 0; i < capturedBlobs.length; i++) {
      setHint(`Mengunggah template biometrik ${i + 1} dari 5...`, "info");
      await addTemplate(currentUserId, capturedBlobs[i]);

      // JEDA SANGAT PENTING: Mencegah SQLite Error / Lock
      await sleep(600);
    }

    // 3. VERIFIKASI AKHIR
    setHint("Memverifikasi kecocokan akhir...", "info");
    await sleep(1000);

    let verifyBlob = await webcam.captureFrame();
    if (!verifyBlob) throw new Error("Kamera terputus saat verifikasi.");

    const result = await identify(verifyBlob);

    if (result.status === "identified" && result.user?.id === currentUserId) {
      webcam.stop();
      if (successName) successName.textContent = currentUserName;
      showStep("success");
      toast.success(
        `Enrollment berhasil untuk ${currentUserName}.`,
        "Berhasil",
      );
    } else {
      throw new Error("Biometrik tidak konsisten. Silakan ulangi.");
    }
  } catch (error) {
    toast.error(error.message);

    // Auto Rollback: Jika terjadi error di tengah unggahan, hapus user
    if (currentUserId) {
      await deleteUser(currentUserId).catch(() => {});
    }

    resetLocalState();
    showStep("capture");
    await initWebcam();
  }
}

// ── Utilities ──────────────────────────────────────────────
function updateProgress() {
  if (sampleBadge) {
    sampleBadge.textContent = `${sampleCount} / ${MAX_SAMPLES}`;
    sampleBadge.className =
      "badge " +
      (sampleCount >= MAX_SAMPLES ? "badge--identified" : "badge--scanning");
  }
  sampleDots.forEach((dot, idx) => {
    dot.classList.remove("active", "completed");
    if (idx < sampleCount) dot.classList.add("completed");
    else if (idx === sampleCount) dot.classList.add("active");
  });
}

function setHint(text, isError = false) {
  if (!scannerHint) return;
  scannerHint.textContent = text;
  scannerHint.style.color = isError ? "var(--color-coral)" : "";
}

function resetLocalState() {
  capturedBlobs = [];
  sampleCount = 0;
  currentUserId = null;
  isUserCreated = false;
  isCapturing = false;
  updateProgress();
}

async function cancelEnrollment() {
  webcam?.stop();
  if (isUserCreated && currentUserId) {
    await deleteUser(currentUserId).catch(() => {});
  }
  resetLocalState();
  window.history.length > 1
    ? window.history.back()
    : (window.location.href = "index.html");
}

function cleanupOnExit() {
  webcam?.stop();
  if (isUserCreated && currentUserId && sampleCount < MAX_SAMPLES) {
    const url = `${BASE_URL}/users/${currentUserId}`;
    if (navigator.sendBeacon)
      fetch(url, { method: "DELETE", keepalive: true }).catch(() => {});
  }
}

init();

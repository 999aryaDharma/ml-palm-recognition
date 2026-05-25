// ============================================================
// js/pages/enroll.js — Enrollment page logic
// ============================================================

import { mountNavbar } from "../components/navbar.js";
import { WebcamCapture } from "../components/webcam.js";
import { createUser, addTemplate, deleteUser } from "../api/users.js";
import { identify } from "../api/identify.js";
import { toast } from "../components/toast.js";
import { QUALITY_HINTS, sleep } from "../utils.js";

// ── State ──────────────────────────────────────────────────
let webcam = null;
let currentUserId = null;
let currentUserName = "";
let sampleCount = 0;
const MAX_SAMPLES = 5;
let isCapturing = false;

// ── DOM Elements ───────────────────────────────────────────
const steps = {
  name: document.getElementById("step-name"),
  capture: document.getElementById("step-capture"),
  verifying: document.getElementById("step-verifying"),
  success: document.getElementById("step-success"),
};

const inputName = document.getElementById("input-name");
const btnToCapture = document.getElementById("btn-to-capture");
const btnCancelCapture = document.getElementById("btn-cancel-capture");

const videoEl = document.getElementById("enroll-video");
const scannerHint = document.getElementById("scanner-hint");
const scannerLoading = document.getElementById("scanner-loading");
const scanline = document.getElementById("scanline");
const sampleBadge = document.getElementById("sample-count-badge");
const sampleDots = document.querySelectorAll(".step-dot");
const successName = document.getElementById("success-name");

// ── Initialization ─────────────────────────────────────────
function init() {
  mountNavbar();

  btnToCapture.addEventListener("click", startEnrollment);
  btnCancelCapture.addEventListener("click", cancelEnrollment);

  // Auto-focus name input
  inputName.focus();
}

/** Show a specific step and hide others */
function showStep(stepName) {
  Object.keys(steps).forEach((key) => {
    steps[key].classList.toggle("hidden", key !== stepName);
  });
}

// ── Flow ───────────────────────────────────────────────────

/** Step 1 → 2: Create user and start camera */
async function startEnrollment() {
  const name = inputName.value.trim();
  if (!name) {
    toast.warning("Silakan masukkan nama lengkap.");
    return;
  }

  try {
    btnToCapture.disabled = true;
    btnToCapture.textContent = "Mendaftarkan...";

    const user = await createUser(name);
    currentUserId = user.id;
    currentUserName = user.name;

    showStep("capture");
    await initWebcam();
  } catch (err) {
    toast.error(err.message || "Gagal mendaftarkan user.");
    btnToCapture.disabled = false;
    btnToCapture.textContent = "Mulai Enrollment";
  }
}

/** Initialize webcam and start auto-capture */
async function initWebcam() {
  try {
    webcam = new WebcamCapture(videoEl, {
      onCapture: handleCapture,
      captureInterval: 1800, // Slightly slower for feedback clarity
    });

    await webcam.start();
    scannerHint.textContent = "Tunjukkan telapak tangan";
    webcam.startAutoCapture();
  } catch (err) {
    toast.error("Gagal mengakses kamera: " + err.message);
    cancelEnrollment();
  }
}

/** Handle auto-capture results */
async function handleCapture(blob) {
  if (isCapturing || sampleCount >= MAX_SAMPLES) return;
  isCapturing = true;

  scannerLoading.classList.remove("hidden");
  scanline.classList.remove("hidden");
  scannerHint.textContent = "Memproses...";

  try {
    const result = await addTemplate(currentUserId, blob);

    // Success capture
    sampleCount++;
    updateProgress();
    toast.success(`Sampel ${sampleCount} berhasil diambil.`, "Sampel Terambil");

    if (sampleCount >= MAX_SAMPLES) {
      webcam.stopAutoCapture();
      scanline.classList.add("hidden");
      await runVerification();
    } else {
      scannerHint.textContent = "Tahan stabil...";
    }
  } catch (err) {
    // Quality gate error
    const hint = QUALITY_HINTS[err.error] || "Coba posisikan kembali telapak";
    scannerHint.textContent = hint;
    // Don't show toast for quality errors to avoid spam, just update hint
  } finally {
    isCapturing = false;
    scannerLoading.classList.add("hidden");
    if (sampleCount < MAX_SAMPLES) {
      setTimeout(() => scanline.classList.add("hidden"), 500);
    }
  }
}

/** Step 2 → 3: Final verification gate */
async function runVerification() {
  showStep("verifying");

  // Wait a bit to show the "Verifying" state for UX
  await sleep(1500);

  try {
    const blob = await webcam.captureFrame();
    const result = await identify(blob);

    if (result.status === "identified" && result.user.id === currentUserId) {
      // SUCCESS!
      webcam.stop();
      successName.textContent = currentUserName;
      showStep("success");
      toast.success("Enrollment selesai sepenuhnya.", "Berhasil");
    } else {
      throw new Error("Verifikasi identitas gagal setelah pendaftaran.");
    }
  } catch (err) {
    toast.error(err.message || "Gagal verifikasi pendaftaran.");
    // If verification fails, we might want to let them retry capture or delete the partial user
    // For now, let's allow them to go back to capture
    sampleCount = 0;
    updateProgress();
    showStep("capture");
    webcam.startAutoCapture();
  }
}

/** Update UI progress */
function updateProgress() {
  sampleBadge.textContent = `${sampleCount} / ${MAX_SAMPLES}`;
  sampleDots.forEach((dot, idx) => {
    dot.classList.toggle("active", idx < sampleCount);
    dot.classList.toggle("completed", idx < sampleCount);
  });
}

/** Cancel and cleanup */
async function cancelEnrollment() {
  if (webcam) webcam.stop();

  // If we already created a user, delete it from backend
  if (currentUserId) {
    try {
      await deleteUser(currentUserId);
    } catch (_) {}
  }

  window.location.href = "index.html";
}

// ── Run ────────────────────────────────────────────────────
init();

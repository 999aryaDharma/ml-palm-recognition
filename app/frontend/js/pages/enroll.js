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

// ── Helper untuk debug ────────────────────────────────────
function logDebug(msg) {
  console.log(msg);
  // Also write to DOM for visibility
  const debugEl = document.getElementById("debug-log") || createDebugLog();
  const logEntry = document.createElement("div");
  logEntry.textContent = msg;
  logEntry.style.fontSize = "11px";
  logEntry.style.padding = "2px";
  logEntry.style.borderBottom = "1px solid #ccc";
  debugEl.appendChild(logEntry);
  debugEl.scrollTop = debugEl.scrollHeight;
}

function createDebugLog() {
  const el = document.createElement("div");
  el.id = "debug-log";
  el.style.position = "fixed";
  el.style.bottom = "0";
  el.style.left = "0";
  el.style.right = "0";
  el.style.height = "150px";
  el.style.overflow = "auto";
  el.style.background = "#f0f0f0";
  el.style.border = "1px solid #999";
  el.style.zIndex = "9999";
  el.style.fontFamily = "monospace";
  el.style.display = "none"; // hidden by default
  document.body.appendChild(el);
  return el;
}

// Show debug log when needed
if (window.location.search.includes("debug=1")) {
  document.addEventListener("DOMContentLoaded", () => {
    const debugEl = document.getElementById("debug-log");
    if (debugEl) debugEl.style.display = "block";
  });
}

// ── Initialization ─────────────────────────────────────────
function init() {
  try {
    logDebug("[Init] Starting...");
    mountNavbar();
    logDebug("[Init] Navbar mounted");

    btnToCapture.addEventListener("click", () => {
      logDebug("[Init] btn-to-capture clicked");
      startEnrollment();
    });
    btnCancelCapture.addEventListener("click", () => {
      logDebug("[Init] btn-cancel-capture clicked");
      cancelEnrollment();
    });

    // Auto-focus name input
    inputName.focus();
    logDebug("[Init] Page ready");

    // Add cleanup on page leave
    window.addEventListener("beforeunload", () => {
      if (webcam) webcam.stop();
    });
    document.addEventListener("visibilitychange", () => {
      if (document.hidden && webcam) webcam.stop();
    });
  } catch (err) {
    logDebug("[Init] ERROR: " + err.message);
    console.error("[Enrollment] Init error:", err);
  }
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

    logDebug("[startEnrollment] Creating user: " + name);
    const user = await createUser(name);
    logDebug("[startEnrollment] User created, id=" + user.id);
    currentUserId = user.id;
    currentUserName = user.name;

    logDebug("[startEnrollment] Showing capture step...");
    showStep("capture");
    logDebug("[startEnrollment] Step changed, initializing webcam...");
    await initWebcam();
  } catch (err) {
    logDebug("[startEnrollment] CAUGHT ERROR: " + err.message);
    console.error("[Enrollment] Error in startEnrollment:", err);
    toast.error(err.message || "Gagal mendaftarkan user.");
    btnToCapture.disabled = false;
    btnToCapture.textContent = "Mulai Enrollment";
  }
}

/** Initialize webcam and start auto-capture */
async function initWebcam() {
  try {
    console.log("[Enrollment] Initializing webcam...");
    webcam = new WebcamCapture(videoEl, {
      onCapture: handleCapture,
      captureInterval: 1800, // Slightly slower for feedback clarity
    });

    console.log("[Enrollment] Starting webcam stream...");
    await webcam.start();
    console.log("[Enrollment] Webcam started successfully");

    scannerHint.textContent = "Tunjukkan telapak tangan";
    webcam.startAutoCapture();
    console.log("[Enrollment] Auto-capture started");
  } catch (err) {
    console.error("[Enrollment] Webcam error:", err);
    const errorMsg = err.message || "Gagal mengakses kamera";
    toast.error("Gagal mengakses kamera: " + errorMsg);

    // Show a retry button instead of auto-redirecting
    scannerHint.textContent =
      "❌ " + errorMsg + " — Klik 'Coba Lagi' atau 'Batal'";
    scannerHint.style.background = "rgba(239, 68, 68, 0.2)";

    // Optionally add a retry button
    const btnRetry = document.createElement("button");
    btnRetry.className = "btn btn--primary";
    btnRetry.textContent = "Coba Lagi";
    btnRetry.onclick = () => {
      scannerHint.textContent = "Menghidupkan kamera...";
      scannerHint.style.background = "";
      btnRetry.remove();
      initWebcam();
    };
    document.querySelector(".surface-card").appendChild(btnRetry);

    // Don't call cancelEnrollment() - let user try again
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
  console.log(
    "[Enrollment] Canceling enrollment, currentUserId:",
    currentUserId,
  );
  if (webcam) webcam.stop();

  // If we already created a user, delete it from backend
  if (currentUserId) {
    try {
      console.log("[Enrollment] Deleting user", currentUserId);
      await deleteUser(currentUserId);
    } catch (err) {
      console.error("[Enrollment] Failed to delete user:", err);
    }
  }

  console.log("[Enrollment] Redirecting to index.html");
  window.history.back();
}

// ── Run ────────────────────────────────────────────────────
init();

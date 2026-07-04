// ============================================================
// js/pages/enroll.js — Enrollment (UX-improved)
//
// Perubahan UX:
// • Hilangkan window.location.href = "index.html" → pakai
//   history.back() / tombol "Mulai lagi" yang user-driven.
// • Step transitions pakai fadeSwap (bukan toggle hidden mendadak).
// • Error state: BUKAN auto-restart 5 detik, melainkan tampilkan
//   tombol "Coba Lagi" / "Mulai dari Awal". User yang putuskan.
// • Quality reject: hint jelas + tidak menutupi scanner.
// • Hapus dead code retry yang unreachable.
// ============================================================

import { mountNavbar } from "../components/navbar.js";
import { WebcamCapture } from "../components/webcam.js";
import {
  createUser,
  addTemplate,
  deleteUser,
  verifyUserReady,
} from "../api/users.js";
import { identify } from "../api/identify.js";
import { BASE_URL } from "../api/client.js";
import { toast } from "../components/toast.js";
import { QUALITY_HINTS, sleep } from "../utils.js";
import { fadeSwap, smoothBack, withLoading } from "../ux.js";

const VERIFY_STEPS_CONFIG = [
  { id: "create-user", label: "Membuat profil pengguna" },
  { id: "upload-1", label: "Template biometrik 1/5" },
  { id: "upload-2", label: "Template biometrik 2/5" },
  { id: "upload-3", label: "Template biometrik 3/5" },
  { id: "upload-4", label: "Template biometrik 4/5" },
  { id: "upload-5", label: "Template biometrik 5/5" },
  { id: "verify", label: "Verifikasi kelengkapan" },
];

const ENROLL_HINTS = [
  "🔍 Tahan telapak lurus dan tegak di depan kamera",
  "↙ Sampel 2: Miringkan telapak sedikit ke kiri",
  "↗ Sampel 3: Miringkan telapak sedikit ke kanan",
  "⬆ Sampel 4: Majukan tangan sedikit ke kamera",
  "⬇ Sampel 5: Mundurkan tangan sedikit dari kamera",
];

let webcam = null;
let currentUserId = null;
let currentUserName = "";
let pendingName = "";
let sampleCount = 0;
const MAX_SAMPLES = 5;
let isCapturing = false;
let isUserCreated = false;
let capturedBlobs = [];

let inputName, btnToCapture, btnCancelCapture;
let videoEl, scannerHint, scannerLoading, scannerLoadingLabel, scanline;
let sampleBadge, sampleDots, captureQualityHint;
let successName, successTemplateCount, steps;

async function init() {
  mountNavbar();

  inputName = document.getElementById("input-name");
  btnToCapture = document.getElementById("btn-to-capture");
  btnCancelCapture = document.getElementById("btn-cancel-capture");
  videoEl = document.getElementById("enroll-video");
  scannerHint = document.getElementById("scanner-hint");
  scannerLoading = document.getElementById("scanner-loading");
  scannerLoadingLabel = document.getElementById("scanner-loading-label");
  scanline = document.getElementById("scanline");
  sampleBadge = document.getElementById("sample-count-badge");
  sampleDots = document.querySelectorAll(".step-dot");
  captureQualityHint = document.getElementById("capture-quality-hint");
  successName = document.getElementById("success-name");
  successTemplateCount = document.getElementById("success-template-count");

  steps = {
    name: document.getElementById("step-name"),
    capture: document.getElementById("step-capture"),
    verifying: document.getElementById("step-verifying"),
    success: document.getElementById("step-success"),
  };

  document.getElementById("form-name")?.addEventListener("submit", (e) => {
    e.preventDefault();
    goToCaptureStep();
  });

  btnCancelCapture?.addEventListener("click", (e) => {
    e.preventDefault();
    cancelEnrollment();
  });

  // Tombol "Selesai" di step success → kembali, bukan reload
  document.getElementById("btn-success-done")?.addEventListener("click", (e) => {
    e.preventDefault();
    smoothBack("index.html");
  });

  // Tombol opsional "Daftarkan pengguna lain" — reset state in-place
  document
    .getElementById("btn-success-enroll-another")
    ?.addEventListener("click", (e) => {
      e.preventDefault();
      restartFromName();
    });

  window.addEventListener("beforeunload", cleanupOnExit);
}

// ── Step navigation (smooth fade) ──────────────────────────
async function showStep(stepName) {
  // Cari container parent untuk fade. Kalau tidak ada, fallback ke toggle biasa.
  const targets = Object.entries(steps);
  for (const [key, el] of targets) {
    if (!el) continue;
    if (key === stepName) {
      el.classList.remove("hidden");
      el.classList.add("ux-step");
      // Re-trigger animasi
      void el.offsetWidth;
    } else {
      el.classList.add("hidden");
      el.classList.remove("ux-step");
    }
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ── Step 1 → 2 ────────────────────────────────────────────
async function goToCaptureStep() {
  const name = inputName.value.trim();
  if (!name || name.length < 2) {
    toast.warning("Nama minimal 2 karakter.");
    return;
  }

  pendingName = name;
  currentUserName = name;

  await withLoading(btnToCapture, "Memproses…", async () => {
    await sleep(280);
    await showStep("capture");
    updateCaptureProgress();
    await initWebcam();
  });
}

async function initWebcam() {
  setHint("🎥 Menyalakan kamera...");
  clearQualityHint();

  try {
    webcam = new WebcamCapture(videoEl, {
      onCapture: handleCapture,
      captureInterval: 1200,
    });
    await webcam.start();
    setHint(ENROLL_HINTS[0]);
    scanline?.classList.remove("hidden");
    webcam.startAutoCapture();
  } catch (err) {
    setHint("❌ Tidak dapat mengakses kamera", true);
    toast.error("Izin kamera ditolak. Aktifkan kamera di pengaturan browser.");
  }
}

// ── Capture handler ────────────────────────────────────────
async function handleCapture(blob) {
  if (isCapturing || capturedBlobs.length >= MAX_SAMPLES) return;
  isCapturing = true;

  if (scannerLoadingLabel)
    scannerLoadingLabel.textContent = `Memeriksa kualitas sampel ${capturedBlobs.length + 1}/5...`;
  scannerLoading?.classList.remove("hidden");

  try {
    const validateForm = new FormData();
    validateForm.append("image", blob, "frame.jpg");

    try {
      const { apiFetch } = await import("../api/client.js");
      await apiFetch("/validate-frame", { method: "POST", body: validateForm });
    } catch (valErr) {
      scannerLoading?.classList.add("hidden");
      const hint =
        QUALITY_HINTS[valErr.error] || "🖐 Arahkan telapak tangan ke kamera";
      setHint(hint);
      showQualityHint(hint);
      return;
    }

    scannerLoading?.classList.add("hidden");
    clearQualityHint();

    capturedBlobs.push(blob);
    sampleCount = capturedBlobs.length;
    updateCaptureProgress();

    if (sampleCount >= MAX_SAMPLES) {
      setHint("✅ 5/5 sampel berhasil! Memulai proses penyimpanan...");
      scanline?.classList.add("hidden");
      webcam.stopAutoCapture();
      await sleep(600);
      await finalizeEnrollment();
    } else {
      setHint(`✅ Sampel ${sampleCount}/5 berhasil diambil!`);
      await sleep(800);
      setHint(ENROLL_HINTS[sampleCount] || "🖐 Siap untuk sampel berikutnya...");
      await sleep(1400);
    }
  } finally {
    if (sampleCount < MAX_SAMPLES) {
      isCapturing = false;
      scannerLoading?.classList.add("hidden");
    }
  }
}

// ── Finalize ───────────────────────────────────────────────
async function finalizeEnrollment() {
  await showStep("verifying");
  initVerifyUI();

  let stepsDone = 0;
  const totalSteps = VERIFY_STEPS_CONFIG.length;
  const advance = () => setVerifyProgress(++stepsDone, totalSteps);

  try {
    // 1. Create user
    setVerifyStep("create-user", "active");
    setVerifyMsg("Membuat profil pengguna di server...");

    let newUser;
    try {
      newUser = await createUser(pendingName);
    } catch (createErr) {
      setVerifyStep("create-user", "error", "Gagal membuat profil");
      if (createErr.status === 409 || createErr.error === "user_exists") {
        // Bukan auto-reset 5 detik — kasih user opsi
        return showRecoveryPrompt({
          title: "Nama sudah terdaftar",
          message: `Pengguna dengan nama "${pendingName}" sudah ada. Ganti nama lalu coba lagi.`,
          actionLabel: "Ganti Nama",
          onAction: async () => {
            resetLocalState();
            webcam?.stop();
            await showStep("name");
            inputName?.focus();
          },
        });
      }
      throw new Error("Gagal membuat profil pengguna di server.");
    }

    setVerifyStep("create-user", "done");
    advance();
    currentUserId = newUser.id;
    isUserCreated = true;

    // 2-6. Upload templates
    for (let i = 0; i < capturedBlobs.length; i++) {
      const stepId = `upload-${i + 1}`;
      setVerifyStep(stepId, "active");
      setVerifyMsg(`Mengunggah template biometrik ${i + 1}/5...`);

      const uniqueFile = new File(
        [capturedBlobs[i]],
        `template_${i + 1}.jpg`,
        { type: "image/jpeg" },
      );

      // Retry tipis untuk lawan backend hot-reload
      let lastErr = null;
      let ok = false;
      for (let attempt = 1; attempt <= 3 && !ok; attempt++) {
        try {
          await addTemplate(currentUserId, uniqueFile);
          ok = true;
        } catch (uploadErr) {
          lastErr = uploadErr;
          if (attempt < 3) await sleep(1200);
        }
      }
      if (!ok) {
        setVerifyStep(stepId, "error", `Template ${i + 1}/5 gagal diunggah`);
        throw new Error(
          `Template ${i + 1} gagal: ${lastErr?.message || "Koneksi terputus"}.`,
        );
      }

      setVerifyStep(stepId, "done", `Template ${i + 1}/5 tersimpan ✓`);
      advance();
      await sleep(220);
    }

    // 7. Verify
    setVerifyStep("verify", "active");
    setVerifyMsg("Memverifikasi kelengkapan template di database...");
    await sleep(300);

    const readyCheck = await verifyUserReady(currentUserId);
    if (!readyCheck.ready || readyCheck.template_count < 5) {
      setVerifyStep("verify", "error");
      throw new Error(
        `Hanya ${readyCheck.template_count}/5 template yang tersimpan.`,
      );
    }

    setVerifyStep(
      "verify",
      "done",
      `${readyCheck.template_count}/5 template terverifikasi ✓`,
    );
    advance();
    setVerifyProgress(totalSteps, totalSteps);
    setVerifyMsg("Enrollment selesai! Semua template tersimpan dengan aman.");

    // Optional identify check (silent)
    try {
      const verifyBlob = await webcam.captureFrame();
      if (verifyBlob) await identify(verifyBlob);
    } catch (_) {}

    webcam.stop();
    if (successName) successName.textContent = currentUserName;
    if (successTemplateCount)
      successTemplateCount.textContent = `${readyCheck.template_count} template tersimpan`;

    await sleep(500);
    await showStep("success");
    toast.success(`Enrollment berhasil untuk ${currentUserName}.`, "Selamat!");
  } catch (error) {
    setVerifyMsg(`❌ ${error.message}`);
    toast.error(error.message, "Enrollment Gagal");

    // Tidak auto-restart. User pilih sendiri.
    showRecoveryPrompt({
      title: "Enrollment gagal",
      message: error.message,
      actionLabel: "Coba dari Awal",
      onAction: async () => {
        if (isUserCreated && currentUserId) {
          await deleteUser(currentUserId).catch(() => {});
        }
        resetLocalState();
        await showStep("capture");
        await initWebcam();
      },
      secondaryLabel: "Kembali",
      onSecondary: () => smoothBack("index.html"),
    });
  }
}

// ── Recovery prompt (inline, tidak modal) ─────────────────
function showRecoveryPrompt({
  title,
  message,
  actionLabel,
  onAction,
  secondaryLabel,
  onSecondary,
}) {
  const list = document.getElementById("verify-steps-list");
  const container = list?.parentElement || steps.verifying;
  if (!container) return;

  const promptId = "enroll-recovery-prompt";
  document.getElementById(promptId)?.remove();

  const node = document.createElement("div");
  node.id = promptId;
  node.className = "ux-anim-fade-in";
  node.style.cssText =
    "margin-top:var(--space-4);padding:var(--space-4);border:1px solid var(--color-border);border-radius:12px;background:var(--color-surface-warm);text-align:center";
  node.innerHTML = `
    <div style="font-size:1.6rem;margin-bottom:8px">⚠️</div>
    <h3 style="margin:0 0 4px 0">${escapeHtml(title)}</h3>
    <p class="text-sm" style="color:var(--color-coffee-light);margin:0 0 var(--space-3) 0">${escapeHtml(message)}</p>
    <div style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap">
      <button type="button" class="btn btn--primary btn--sm" data-act="primary">${escapeHtml(actionLabel)}</button>
      ${secondaryLabel ? `<button type="button" class="btn btn--secondary btn--sm" data-act="secondary">${escapeHtml(secondaryLabel)}</button>` : ""}
    </div>
  `;
  container.appendChild(node);

  node.querySelector('[data-act="primary"]')?.addEventListener("click", () => {
    node.remove();
    onAction?.();
  });
  node.querySelector('[data-act="secondary"]')?.addEventListener("click", () => {
    node.remove();
    onSecondary?.();
  });
}

// ── Verify UI helpers ──────────────────────────────────────
function initVerifyUI() {
  const list = document.getElementById("verify-steps-list");
  if (!list) return;
  list.innerHTML = VERIFY_STEPS_CONFIG.map(
    (s) => `
    <div class="vstep vstep--pending" id="vstep-${s.id}" role="listitem">
      <div class="vstep__indicator" aria-hidden="true">–</div>
      <span class="vstep__label">${s.label}</span>
    </div>
  `,
  ).join("");
  setVerifyProgress(0, VERIFY_STEPS_CONFIG.length);
  setVerifyMsg("Mempersiapkan proses enrollment...");
}

function setVerifyStep(stepId, state, overrideLabel) {
  const el = document.getElementById(`vstep-${stepId}`);
  if (!el) return;
  el.className = `vstep vstep--${state}`;
  const indicator = el.querySelector(".vstep__indicator");
  if (indicator) {
    if (state === "active")
      indicator.innerHTML = '<div class="spinner spinner--sm"></div>';
    else if (state === "done") indicator.textContent = "✓";
    else if (state === "error") indicator.textContent = "✕";
    else indicator.textContent = "–";
  }
  if (overrideLabel) {
    const label = el.querySelector(".vstep__label");
    if (label) label.textContent = overrideLabel;
  }
  if (state === "active") {
    el.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
  }
}

function setVerifyProgress(done, total) {
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  const fill = document.getElementById("verify-progress-fill");
  const label = document.getElementById("verify-pct");
  if (fill) fill.style.width = `${pct}%`;
  if (label) label.textContent = `${pct}%`;
}

function setVerifyMsg(msg) {
  const el = document.getElementById("verify-status-msg");
  if (el) el.textContent = msg;
}

// ── Capture UI helpers ─────────────────────────────────────
function updateCaptureProgress() {
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

function showQualityHint(text) {
  if (!captureQualityHint) return;
  captureQualityHint.textContent = text;
  captureQualityHint.style.display = "flex";
}

function clearQualityHint() {
  if (!captureQualityHint) return;
  captureQualityHint.textContent = "";
  captureQualityHint.style.display = "none";
}

// ── State reset & cleanup ──────────────────────────────────
function resetLocalState() {
  capturedBlobs = [];
  sampleCount = 0;
  currentUserId = null;
  isUserCreated = false;
  isCapturing = false;
  updateCaptureProgress();
  clearQualityHint();
}

async function restartFromName() {
  webcam?.stop();
  resetLocalState();
  pendingName = "";
  currentUserName = "";
  if (inputName) inputName.value = "";
  await showStep("name");
  inputName?.focus();
}

async function cancelEnrollment() {
  webcam?.stop();
  if (isUserCreated && currentUserId) {
    await deleteUser(currentUserId).catch(() => {});
  }
  resetLocalState();
  smoothBack("index.html");
}

function escapeHtml(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[c],
  );
}

function cleanupOnExit() {
  webcam?.stop();
  if (isUserCreated && currentUserId && sampleCount < MAX_SAMPLES) {
    const url = `${BASE_URL}/users/${currentUserId}`;
    fetch(url, { method: "DELETE", keepalive: true }).catch(() => {});
  }
}

init();

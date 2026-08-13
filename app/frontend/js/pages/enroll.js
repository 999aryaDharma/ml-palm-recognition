// ============================================================
// js/pages/enroll.js — 5-sample multi-model enrollment
//
// One capture set is shared by all active biometric models. The backend performs
// detection/ROI once per frame and fans the ROI out to each registered model.
// ============================================================

import { mountNavbar } from "../components/navbar.js";
import { WebcamCapture } from "../components/webcam.js";
import {
  createUser,
  addTemplateMulti,
  deleteUser,
  verifyUserReadyAll,
} from "../api/users.js";
import { BASE_URL, apiFetch } from "../api/client.js";
import { toast } from "../components/toast.js";
import { QUALITY_HINTS, sleep } from "../utils.js";
import { smoothBack, withLoading } from "../ux.js";

const MAX_SAMPLES = 5;
const VERIFY_STEPS_CONFIG = [
  { id: "create-user", label: "Membuat profil pengguna" },
  { id: "upload-1", label: "Memproses sampel 1/5 ke semua model" },
  { id: "upload-2", label: "Memproses sampel 2/5 ke semua model" },
  { id: "upload-3", label: "Memproses sampel 3/5 ke semua model" },
  { id: "upload-4", label: "Memproses sampel 4/5 ke semua model" },
  { id: "upload-5", label: "Memproses sampel 5/5 ke semua model" },
  { id: "verify", label: "Verifikasi template semua model" },
];

const POSE_GUIDE = [
  { icon: "🖐", short: "Lurus", hint: "🔍 Tahan telapak lurus dan tegak di depan kamera" },
  { icon: "↙", short: "Kiri", hint: "↙ Sampel 2: Miringkan telapak sedikit ke kiri" },
  { icon: "↗", short: "Kanan", hint: "↗ Sampel 3: Miringkan telapak sedikit ke kanan" },
  { icon: "⬆", short: "Dekat", hint: "⬆ Sampel 4: Majukan tangan sedikit mendekati kamera" },
  { icon: "⬇", short: "Jauh", hint: "⬇ Sampel 5: Mundurkan tangan sedikit menjauhi kamera" },
];

const _thumbUrls = [];

let webcam = null;
let currentUserId = null;
let currentUserName = "";
let pendingName = "";
let sampleCount = 0;
let isCapturing = false;
let isUserCreated = false;
let capturedBlobs = [];

let inputName, btnToCapture, btnCancelCapture;
let videoEl, scannerHint, scannerLoading, scannerLoadingLabel, scanline;
let sampleBadge, sampleDots, captureQualityHint;
let captureEyebrow, scannerSampleNum;
let poseInstructionIcon, poseInstructionText;
let successName, successTemplateCount, successThumbs;
let steps;

async function init() {
  mountNavbar();

  // Enrollment always targets every active model, so a per-model selector would
  // be misleading on this page. Identification pages keep their selector.
  const selectorContainer = document.getElementById("navbar-model-selector");
  if (selectorContainer) selectorContainer.innerHTML = "";

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
  captureEyebrow = document.getElementById("capture-eyebrow");
  scannerSampleNum = document.getElementById("scanner-sample-num");
  poseInstructionIcon = document.getElementById("pose-instruction-icon");
  poseInstructionText = document.getElementById("pose-instruction-text");
  successName = document.getElementById("success-name");
  successTemplateCount = document.getElementById("success-template-count");
  successThumbs = document.getElementById("success-thumbs");

  steps = {
    name: document.getElementById("step-name"),
    capture: document.getElementById("step-capture"),
    verifying: document.getElementById("step-verifying"),
    profile: document.getElementById("step-profile"),
    success: document.getElementById("step-success"),
  };

  document.getElementById("form-name")?.addEventListener("submit", (event) => {
    event.preventDefault();
    goToCaptureStep();
  });
  document.getElementById("form-profile")?.addEventListener("submit", (event) => {
    event.preventDefault();
    submitProfile();
  });
  btnCancelCapture?.addEventListener("click", (event) => {
    event.preventDefault();
    cancelEnrollment();
  });
  document.getElementById("btn-success-done")?.addEventListener("click", (event) => {
    event.preventDefault();
    smoothBack("index.html");
  });
  document.getElementById("btn-success-enroll-another")?.addEventListener("click", (event) => {
    event.preventDefault();
    restartFromName();
  });

  window.addEventListener("beforeunload", cleanupOnExit);
}

async function showStep(stepName) {
  for (const [key, element] of Object.entries(steps)) {
    if (!element) continue;
    if (key === stepName) {
      element.classList.remove("hidden");
      element.classList.add("ux-step");
      void element.offsetWidth;
    } else {
      element.classList.add("hidden");
      element.classList.remove("ux-step");
    }
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function goToCaptureStep() {
  const name = inputName?.value.trim() || "";
  if (name.length < 2) {
    toast.warning("Nama minimal 2 karakter.");
    return;
  }

  const { checkHealth } = await import("../api/client.js");
  const { online } = await checkHealth();
  if (!online) {
    toast.error("Server backend offline. Jalankan FastAPI terlebih dahulu.");
    return;
  }

  pendingName = name;
  currentUserName = name;

  await withLoading(btnToCapture, "Memproses…", async () => {
    await showStep("capture");
    resetCaptureUI();
    await initWebcam();
  });
}

async function submitProfile() {
  const nik = document.getElementById("input-nik")?.value.trim();
  const kelas = document.getElementById("input-kelas")?.value.trim();
  const balance = parseFloat(document.getElementById("input-balance")?.value || "0");
  const button = document.getElementById("btn-submit-profile");

  if (!nik || !kelas) {
    toast.warning("Lengkapi NIK dan Kelas terlebih dahulu.");
    return;
  }

  await withLoading(button, "Menyimpan…", async () => {
    try {
      const { addProfile } = await import("../api/users.js");
      await addProfile(currentUserId, {
        nik,
        kelas_jabatan: kelas,
        initial_balance: balance,
      });
      await showStep("success");
      toast.success(`Enrollment berhasil untuk ${currentUserName}!`, "Selamat!");
    } catch (error) {
      toast.error(error.message || "Gagal menyimpan data diri.");
    }
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
    setHint(POSE_GUIDE[findFirstEmptySlot()]?.hint || POSE_GUIDE[0].hint);
    scanline?.classList.remove("hidden");
    webcam.startAutoCapture();
  } catch {
    setHint("❌ Tidak dapat mengakses kamera", true);
    toast.error("Izin kamera ditolak. Aktifkan kamera di pengaturan browser.");
  }
}

async function validateFrameForAllModels(blob) {
  const form = new FormData();
  form.append("image", blob, "frame.jpg");
  const result = await apiFetch("/validate-frame", { method: "POST", body: form });
  if (result.status === "error") {
    const error = new Error(result.message || "Frame tidak valid.");
    error.error = result.error;
    throw error;
  }
  return result;
}

async function handleCapture(blob) {
  if (isCapturing || sampleCount >= MAX_SAMPLES) return;
  isCapturing = true;
  const index = findFirstEmptySlot();

  if (scannerLoadingLabel) {
    scannerLoadingLabel.textContent = `Memeriksa sampel ${index + 1}/5 untuk semua model...`;
  }
  scannerLoading?.classList.remove("hidden");
  setSlotActive(index);

  try {
    await validateFrameForAllModels(blob);

    capturedBlobs[index] = blob;
    sampleCount = capturedBlobs.filter(Boolean).length;
    renderSampleThumbnail(blob, index);
    clearQualityHint();
    clearSlotActive(index);
    updateCaptureProgress();

    if (sampleCount >= MAX_SAMPLES) {
      setHint("✅ 5/5 sampel valid. Membuat template untuk semua model...");
      scanline?.classList.add("hidden");
      webcam.stopAutoCapture();
      await sleep(500);
      await finalizeEnrollment();
      return;
    }

    setHint(`✅ Sampel ${sampleCount}/5 valid untuk semua model.`);
    await sleep(500);
    const next = findFirstEmptySlot();
    updatePoseGuide(next);
    setHint(POSE_GUIDE[next]?.hint || "🖐 Siap untuk sampel berikutnya...");
  } catch (error) {
    clearSlotActive(index);
    const hint = QUALITY_HINTS[error.error] || error.message || "🖐 Arahkan telapak tangan ke kamera";
    setHint(hint);
    showQualityHint(hint);
  } finally {
    if (sampleCount < MAX_SAMPLES) isCapturing = false;
    scannerLoading?.classList.add("hidden");
  }
}

async function uploadCaptureWithRetry(blob, index) {
  let lastError = null;
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      return await addTemplateMulti(currentUserId, blob);
    } catch (error) {
      lastError = error;
      if (attempt < 3) await sleep(1200);
    }
  }
  throw new Error(
    `Sampel ${index + 1} gagal disimpan setelah 3 percobaan: ${lastError?.message || "koneksi bermasalah"}.`,
  );
}

async function finalizeEnrollment() {
  await showStep("verifying");
  initVerifyUI();

  let stepsDone = 0;
  const totalSteps = VERIFY_STEPS_CONFIG.length;
  const advance = () => setVerifyProgress(++stepsDone, totalSteps);

  try {
    setVerifyStep("create-user", "active");
    setVerifyMsg("Membuat profil pengguna di server...");

    const newUser = await createUser(pendingName);
    currentUserId = newUser.id;
    isUserCreated = true;
    setVerifyStep("create-user", "done", "Profil pengguna siap ✓");
    advance();

    let detectedModelCount = 0;
    for (let index = 0; index < capturedBlobs.length; index++) {
      const stepId = `upload-${index + 1}`;
      setVerifyStep(stepId, "active");
      setVerifyMsg(`Membuat template sampel ${index + 1}/5 untuk semua model...`);

      const result = await uploadCaptureWithRetry(capturedBlobs[index], index);
      detectedModelCount = result.model_count;
      setVerifyStep(
        stepId,
        "done",
        `Sampel ${index + 1}/5 → ${result.model_count} model tersimpan ✓`,
      );
      advance();
      await sleep(250);
    }

    setVerifyStep("verify", "active");
    setVerifyMsg("Memverifikasi template per model di database...");
    const readyCheck = await verifyUserReadyAll(currentUserId);

    if (!readyCheck.ready) {
      const incomplete = Object.entries(readyCheck.models)
        .filter(([, info]) => !info.ready)
        .map(([modelId, info]) => `${modelId}: ${info.template_count}/${readyCheck.required}`)
        .join(", ");
      throw new Error(`Enrollment belum lengkap: ${incomplete}`);
    }

    const modelCount = Object.keys(readyCheck.models).length || detectedModelCount;
    const totalTemplates = Object.values(readyCheck.models).reduce(
      (sum, info) => sum + info.template_count,
      0,
    );

    setVerifyStep(
      "verify",
      "done",
      `${MAX_SAMPLES} sampel × ${modelCount} model terverifikasi ✓`,
    );
    advance();
    setVerifyProgress(totalSteps, totalSteps);
    setVerifyMsg("✅ Semua model siap. Lanjut isi data diri.");

    webcam?.stop();
    buildSuccessScreen({ modelCount, totalTemplates });
    await sleep(350);
    await showStep("profile");
  } catch (error) {
    setVerifyMsg(`❌ ${error.message}`);
    toast.error(error.message, "Enrollment Gagal");
    console.error("Enrollment error:", error);

    // This flow only creates new users, so rollback the whole user to avoid
    // partially-enrolled model namespaces or duplicate templates on retry.
    if (isUserCreated && currentUserId) {
      await deleteUser(currentUserId).catch((rollbackError) => {
        console.error("Rollback failed:", rollbackError);
      });
    }
    isUserCreated = false;
    currentUserId = null;

    showRecoveryPrompt({
      title: "Enrollment gagal",
      message: error.message,
      actionLabel: "Coba dari Awal",
      onAction: async () => {
        resetLocalState();
        await showStep("capture");
        await initWebcam();
      },
      secondaryLabel: "Kembali ke Dashboard",
      onSecondary: () => smoothBack("index.html"),
    });
  }
}

function renderSampleThumbnail(blob, index) {
  const slot = document.getElementById(`sample-slot-${index}`);
  if (!slot) return;
  const url = URL.createObjectURL(blob);
  _thumbUrls.push(url);
  slot.innerHTML = "";

  const image = document.createElement("img");
  image.src = url;
  image.alt = `Sampel ${index + 1}`;
  image.className = "sample-slot__img";

  const check = document.createElement("div");
  check.className = "sample-slot__check";
  check.textContent = "✓";
  check.setAttribute("aria-label", `Sampel ${index + 1} berhasil`);

  slot.appendChild(image);
  slot.appendChild(check);
  slot.classList.add("sample-slot--captured");
}

function buildSuccessScreen({ modelCount, totalTemplates }) {
  if (successName) successName.textContent = currentUserName;
  if (successTemplateCount) {
    successTemplateCount.textContent = `${MAX_SAMPLES} sampel × ${modelCount} model = ${totalTemplates} template tersimpan`;
  }
  if (!successThumbs) return;

  successThumbs.innerHTML = "";
  capturedBlobs.forEach((blob, index) => {
    const url = URL.createObjectURL(blob);
    _thumbUrls.push(url);
    const node = document.createElement("div");
    node.className = "success-thumb";
    node.innerHTML = `
      <img src="${url}" alt="Sampel ${index + 1}">
      <div class="success-thumb__label">S${index + 1}</div>
    `;
    successThumbs.appendChild(node);
  });
}

function updatePoseGuide(nextIndex) {
  const poseItems = document.querySelectorAll(".pose-item");
  poseItems.forEach((item, index) => {
    item.classList.remove("pose-item--active", "pose-item--done");
    if (index < nextIndex) item.classList.add("pose-item--done");
    else if (index === nextIndex) item.classList.add("pose-item--active");
  });

  const pose = POSE_GUIDE[nextIndex] || POSE_GUIDE[MAX_SAMPLES - 1];
  if (poseInstructionIcon) poseInstructionIcon.textContent = pose.icon;
  if (poseInstructionText) poseInstructionText.textContent = pose.hint.replace(/^.{1,3}\s/, "");
  if (captureEyebrow) captureEyebrow.textContent = `STAGE 2 — SAMPLE ${nextIndex + 1} / 5`;
  if (scannerSampleNum) scannerSampleNum.textContent = String(nextIndex + 1);
}

function findFirstEmptySlot() {
  for (let index = 0; index < MAX_SAMPLES; index++) {
    if (!capturedBlobs[index]) return index;
  }
  return MAX_SAMPLES - 1;
}

function setSlotActive(index) {
  document.getElementById(`sample-slot-${index}`)?.classList.add("sample-slot--active-slot");
}

function clearSlotActive(index) {
  document.getElementById(`sample-slot-${index}`)?.classList.remove("sample-slot--active-slot");
}

function showRecoveryPrompt({ title, message, actionLabel, onAction, secondaryLabel, onSecondary }) {
  const container = document.getElementById("verify-steps-list")?.parentElement || steps.verifying;
  if (!container) return;

  const promptId = "enroll-recovery-prompt";
  document.getElementById(promptId)?.remove();
  const node = document.createElement("div");
  node.id = promptId;
  node.className = "ux-anim-fade-in";
  node.style.cssText =
    "margin-top:var(--space-5);padding:var(--space-5);border:1px solid var(--color-border);" +
    "border-radius:12px;background:var(--color-surface-warm);text-align:center;";
  node.innerHTML = `
    <div style="font-size:1.8rem;margin-bottom:var(--space-3)">⚠️</div>
    <h3 style="margin:0 0 var(--space-2)">${escapeHtml(title)}</h3>
    <p class="text-sm" style="color:var(--color-coffee-light);margin:0 0 var(--space-5)">${escapeHtml(message)}</p>
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

function initVerifyUI() {
  const list = document.getElementById("verify-steps-list");
  if (!list) return;
  list.innerHTML = VERIFY_STEPS_CONFIG.map(
    (step) => `
      <div class="vstep vstep--pending" id="vstep-${step.id}" role="listitem">
        <div class="vstep__indicator" aria-hidden="true">–</div>
        <span class="vstep__label">${step.label}</span>
      </div>
    `,
  ).join("");
  setVerifyProgress(0, VERIFY_STEPS_CONFIG.length);
  setVerifyMsg("Mempersiapkan multi-model enrollment...");
}

function setVerifyStep(stepId, state, overrideLabel) {
  const element = document.getElementById(`vstep-${stepId}`);
  if (!element) return;
  element.className = `vstep vstep--${state}`;
  const indicator = element.querySelector(".vstep__indicator");
  if (indicator) {
    if (state === "active") indicator.innerHTML = '<div class="spinner spinner--sm"></div>';
    else if (state === "done") indicator.textContent = "✓";
    else if (state === "error") indicator.textContent = "✕";
    else indicator.textContent = "–";
  }
  if (overrideLabel) {
    const label = element.querySelector(".vstep__label");
    if (label) label.textContent = overrideLabel;
  }
}

function setVerifyProgress(done, total) {
  const percentage = total > 0 ? Math.round((done / total) * 100) : 0;
  const fill = document.getElementById("verify-progress-fill");
  const label = document.getElementById("verify-pct");
  if (fill) fill.style.width = `${percentage}%`;
  if (label) label.textContent = `${percentage}%`;
}

function setVerifyMsg(message) {
  const element = document.getElementById("verify-status-msg");
  if (element) element.textContent = message;
}

function resetCaptureUI() {
  sampleCount = 0;
  capturedBlobs = new Array(MAX_SAMPLES).fill(null);
  for (let index = 0; index < MAX_SAMPLES; index++) {
    const slot = document.getElementById(`sample-slot-${index}`);
    if (!slot) continue;
    slot.classList.remove("sample-slot--captured", "sample-slot--active-slot");
    slot.innerHTML = `
      <div class="sample-slot__number">${index + 1}</div>
      <div class="sample-slot__icon">${POSE_GUIDE[index].icon}</div>
    `;
  }
  updatePoseGuide(0);
  updateCaptureProgressUI();
  clearQualityHint();
  revokeThumbs();
}

function updateCaptureProgress() {
  updateCaptureProgressUI();
  if (sampleCount < MAX_SAMPLES) updatePoseGuide(findFirstEmptySlot());
}

function updateCaptureProgressUI() {
  if (sampleBadge) {
    sampleBadge.textContent = `${sampleCount} / ${MAX_SAMPLES}`;
    sampleBadge.className =
      "badge " + (sampleCount >= MAX_SAMPLES ? "badge--identified" : "badge--scanning");
  }
  document.getElementById("sample-stepper-wrapper")?.setAttribute("aria-valuenow", String(sampleCount));
  sampleDots.forEach((dot, index) => {
    dot.classList.remove("active", "completed");
    if (index < sampleCount) dot.classList.add("completed");
    else if (index === sampleCount) dot.classList.add("active");
  });
}

function setHint(text, isError = false) {
  if (!scannerHint) return;
  scannerHint.textContent = text;
  scannerHint.style.color = isError ? "var(--color-coral)" : "";
}

function showQualityHint(text) {
  if (!captureQualityHint) return;
  captureQualityHint.textContent = `⚠️ ${text}`;
  captureQualityHint.classList.remove("hidden");
}

function clearQualityHint() {
  if (!captureQualityHint) return;
  captureQualityHint.classList.add("hidden");
  captureQualityHint.textContent = "";
}

function resetLocalState() {
  capturedBlobs = new Array(MAX_SAMPLES).fill(null);
  sampleCount = 0;
  currentUserId = null;
  isUserCreated = false;
  isCapturing = false;
  revokeThumbs();
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

function revokeThumbs() {
  _thumbUrls.forEach((url) => URL.revokeObjectURL(url));
  _thumbUrls.length = 0;
}

function cleanupOnExit() {
  webcam?.stop();
  revokeThumbs();
  if (isUserCreated && currentUserId) {
    fetch(`${BASE_URL}/users/${currentUserId}`, {
      method: "DELETE",
      keepalive: true,
    }).catch(() => {});
  }
}

function escapeHtml(value) {
  return String(value).replace(
    /[&<>"']/g,
    (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;",
    })[char],
  );
}

init();

// ============================================================
// js/pages/enroll.js — Interactive 5-Sample Enrollment
//
// Alur:
//   Step 1: Input nama → cek backend online
//   Step 2: Auto-capture 5 sampel dari pose berbeda
//           - Tiap sampel: validate-frame → simpan blob di RAM
//           - Feedback per-sampel: thumbnail, pose guide, dot
//   Step 3: Finalize → createUser → upload 5 template ke DB
//           - Per-template: retry 3x, delay 800ms antar upload
//           - Verifikasi via /users/:id/verify-ready
//   Step 4: Success screen dengan pratinjau 5 thumbnail
//
// Jaminan:
//   - TIDAK ada user dibuat sebelum 5 blob berhasil dikumpulkan
//   - TIDAK ada template yang diupload sebagian → rollback kalau gagal
//   - SQLite lock dicegah dengan delay 800ms + busy_timeout 30s
// ============================================================

import { mountNavbar }               from "../components/navbar.js";
import { WebcamCapture }             from "../components/webcam.js";
import { ModelSelector }             from "../components/model-selector.js";
import { createUser, addTemplate, deleteUser, verifyUserReady } from "../api/users.js";
import { BASE_URL }                  from "../api/client.js";
import { toast }                     from "../components/toast.js";
import { QUALITY_HINTS, sleep }      from "../utils.js";
import { smoothBack, withLoading }   from "../ux.js";


// ── Konfigurasi ───────────────────────────────────────────
const MAX_SAMPLES = 5;

/** Label verifikasi untuk step-verifying */
const VERIFY_STEPS_CONFIG = [
  { id: "create-user", label: "Membuat profil pengguna" },
  { id: "upload-1",    label: "Template biometrik 1/5" },
  { id: "upload-2",    label: "Template biometrik 2/5" },
  { id: "upload-3",    label: "Template biometrik 3/5" },
  { id: "upload-4",    label: "Template biometrik 4/5" },
  { id: "upload-5",    label: "Template biometrik 5/5" },
  { id: "verify",      label: "Verifikasi kelengkapan" },
];

/** Panduan pose per sampel */
const POSE_GUIDE = [
  { icon: "🖐",  short: "Lurus",  hint: "🔍 Tahan telapak lurus dan tegak di depan kamera" },
  { icon: "↙",  short: "Kiri",   hint: "↙ Sampel 2: Miringkan telapak sedikit ke kiri" },
  { icon: "↗",  short: "Kanan",  hint: "↗ Sampel 3: Miringkan telapak sedikit ke kanan" },
  { icon: "⬆",  short: "Dekat",  hint: "⬆ Sampel 4: Majukan tangan sedikit mendekati kamera" },
  { icon: "⬇",  short: "Jauh",   hint: "⬇ Sampel 5: Mundurkan tangan sedikit menjauhi kamera" },
];

/** Revoke URL cache untuk thumbnail */
const _thumbUrls = [];

// ── State ─────────────────────────────────────────────────
let webcam         = null;
let currentUserId  = null;
let currentUserName = "";
let pendingName    = "";
let sampleCount    = 0;
let isCapturing    = false;
let isUserCreated  = false;
let capturedBlobs  = [];   // Array lokal: 5 blob di RAM, upload saat finalizeEnrollment()
let modelSelector = null;
let selectedEnrollmentModelId = null;
let selectedEnrollmentVersion = null;


// ── DOM refs ──────────────────────────────────────────────
let inputName, btnToCapture, btnCancelCapture;
let videoEl, scannerHint, scannerLoading, scannerLoadingLabel, scanline;
let sampleBadge, sampleDots, captureQualityHint;
let captureEyebrow, scannerSampleNum;
let poseInstructionIcon, poseInstructionText;
let successName, successTemplateCount, successThumbs;
let steps;

// ─────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────
async function init() {
  mountNavbar();

  // DOM cache
  inputName            = document.getElementById("input-name");
  btnToCapture         = document.getElementById("btn-to-capture");
  btnCancelCapture     = document.getElementById("btn-cancel-capture");
  videoEl              = document.getElementById("enroll-video");
  scannerHint          = document.getElementById("scanner-hint");
  scannerLoading       = document.getElementById("scanner-loading");
  scannerLoadingLabel  = document.getElementById("scanner-loading-label");
  scanline             = document.getElementById("scanline");
  sampleBadge          = document.getElementById("sample-count-badge");
  sampleDots           = document.querySelectorAll(".step-dot");
  captureQualityHint   = document.getElementById("capture-quality-hint");
  captureEyebrow       = document.getElementById("capture-eyebrow");
  scannerSampleNum     = document.getElementById("scanner-sample-num");
  poseInstructionIcon  = document.getElementById("pose-instruction-icon");
  poseInstructionText  = document.getElementById("pose-instruction-text");
  successName          = document.getElementById("success-name");
  successTemplateCount = document.getElementById("success-template-count");
  successThumbs        = document.getElementById("success-thumbs");

  steps = {
    name:      document.getElementById("step-name"),
    capture:   document.getElementById("step-capture"),
    verifying: document.getElementById("step-verifying"),
    profile:   document.getElementById("step-profile"),
    success:   document.getElementById("step-success"),
  };

  const selectorContainer = document.getElementById("navbar-model-selector");
  if (selectorContainer) {
    modelSelector = new ModelSelector(selectorContainer);
    await modelSelector.init();
    selectedEnrollmentModelId = modelSelector.selectedModelId;
    selectedEnrollmentVersion = modelSelector.selectedVersion;
    modelSelector.onChange((id) => {
      selectedEnrollmentModelId = id;
      selectedEnrollmentVersion = modelSelector.selectedVersion;
    });
  }

  // Event listeners
  document.getElementById("form-name")?.addEventListener("submit", (e) => {
    e.preventDefault();
    goToCaptureStep();
  });


  document.getElementById("form-profile")?.addEventListener("submit", (e) => {
    e.preventDefault();
    submitProfile();
  });

  btnCancelCapture?.addEventListener("click", (e) => {
    e.preventDefault();
    cancelEnrollment();
  });

  document.getElementById("btn-success-done")?.addEventListener("click", (e) => {
    e.preventDefault();
    smoothBack("index.html");
  });

  document.getElementById("btn-success-enroll-another")?.addEventListener("click", (e) => {
    e.preventDefault();
    restartFromName();
  });

  window.addEventListener("beforeunload", cleanupOnExit);
}

// ─────────────────────────────────────────────────────────
// STEP NAVIGATION
// ─────────────────────────────────────────────────────────
async function showStep(stepName) {
  for (const [key, el] of Object.entries(steps)) {
    if (!el) continue;
    if (key === stepName) {
      el.classList.remove("hidden");
      el.classList.add("ux-step");
      void el.offsetWidth; // re-trigger animation
    } else {
      el.classList.add("hidden");
      el.classList.remove("ux-step");
    }
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ─────────────────────────────────────────────────────────
// STEP 1 → 2: Go to Capture
// ─────────────────────────────────────────────────────────
async function goToCaptureStep() {
  const name = inputName.value.trim();
  if (!name || name.length < 2) {
    toast.warning("Nama minimal 2 karakter.");
    return;
  }

  // Cek backend online sebelum memulai
  const { checkHealth } = await import("../api/client.js");
  try {
    const { online } = await checkHealth();
    if (!online) {
      toast.error("Server backend offline. Jalankan FastAPI terlebih dahulu.");
      return;
    }
  } catch {
    toast.error("Gagal memeriksa koneksi ke server backend.");
    return;
  }

  pendingName     = name;
  currentUserName = name;

  await withLoading(btnToCapture, "Memproses…", async () => {
    await sleep(280);
    await showStep("capture");
    resetCaptureUI();
    await initWebcam();
  });
}

// ─────────────────────────────────────────────────────────
// STEP 4 → 5: Submit Profile & Go to Success
// ─────────────────────────────────────────────────────────
async function submitProfile() {
  const nik = document.getElementById("input-nik")?.value.trim();
  const kelas = document.getElementById("input-kelas")?.value.trim();
  const balance = parseFloat(document.getElementById("input-balance")?.value || "0");
  const btn = document.getElementById("btn-submit-profile");

  if (!nik || !kelas) {
    toast.warning("Lengkapi NIK dan Kelas terlebih dahulu.");
    return;
  }

  await withLoading(btn, "Menyimpan…", async () => {
    try {
      const { addProfile } = await import("../api/users.js");
      await addProfile(currentUserId, { nik: nik, kelas_jabatan: kelas, initial_balance: balance });
      await sleep(500);
      await showStep("success");
      toast.success(`Enrollment berhasil untuk ${currentUserName}!`, "Selamat!");
    } catch (err) {
      toast.error(err.message || "Gagal menyimpan data diri.");
    }
  });
}

// ─────────────────────────────────────────────────────────
// WEBCAM
// ─────────────────────────────────────────────────────────
async function initWebcam() {
  modelSelector?.setDisabled(true);
  setHint("🎥 Menyalakan kamera...");
  clearQualityHint();


  try {
    webcam = new WebcamCapture(videoEl, {
      onCapture: handleCapture,
      captureInterval: 1200,
    });
    await webcam.start();
    setHint(POSE_GUIDE[0].hint);
    scanline?.classList.remove("hidden");
    webcam.startAutoCapture();
  } catch {
    setHint("❌ Tidak dapat mengakses kamera", true);
    toast.error("Izin kamera ditolak. Aktifkan kamera di pengaturan browser.");
  }
}

// ─────────────────────────────────────────────────────────
// CAPTURE HANDLER  (dipanggil otomatis tiap captureInterval)
// ─────────────────────────────────────────────────────────
async function handleCapture(blob) {
  // Guard: jangan capture jika sedang memproses atau sudah semua slot terisi
  if (isCapturing || sampleCount >= MAX_SAMPLES) return;
  isCapturing = true;

  // Index slot yang akan diisi = slot thumbnail pertama yang masih kosong.
  // Ini PENTING saat recapture: capturedBlobs mungkin sudah ada 3 blob di index
  // 0,1,3 dan slot 2 kosong — kita harus isi slot 2, bukan index 3.
  const idx = findFirstEmptySlot();

  // Tampilkan spinner loading di scanner
  if (scannerLoadingLabel) {
    scannerLoadingLabel.textContent = `Memeriksa kualitas sampel ${idx + 1}/5...`;
  }
  scannerLoading?.classList.remove("hidden");

  // Tandai slot yang sedang diproses dengan animasi pulse
  setSlotActive(idx);

  try {
    // ── 1. Validasi kualitas frame ke backend (full pipeline, TANPA simpan ke DB) ──
    const form = new FormData();
    form.append("image", blob, "frame.jpg");

    try {
      const { apiFetch } = await import("../api/client.js");
      const result = await apiFetch("/validate-frame", { method: "POST", body: form });
      if (result.status === "error") {
        throw { detail: result };
      }
    } catch (valErr) {
      // Frame gagal quality gate → tunjukkan hint, JANGAN tambah ke array
      scannerLoading?.classList.add("hidden");
      clearSlotActive(idx);

      const hint = QUALITY_HINTS[valErr.error] || "🖐 Arahkan telapak tangan ke kamera";
      setHint(hint);
      showQualityHint(hint);
      return; // isCapturing akan direset di finally
    }

    // ── 2. Frame lolos full pipeline → simpan di RAM di slot yang benar ──
    scannerLoading?.classList.add("hidden");
    clearQualityHint();
    clearSlotActive(idx);

    // Simpan ke slot yg tepat (bisa jadi bukan akhir array kalau ada recapture)
    capturedBlobs[idx] = blob;
    sampleCount = capturedBlobs.filter(Boolean).length;

    // Render thumbnail di slot panel kanan
    renderSampleThumbnail(blob, idx);

    // Update semua progress UI
    updateCaptureProgress();

    if (sampleCount >= MAX_SAMPLES) {
      // ── Semua 5 sampel terkumpul → lanjut ke finalize ──
      setHint("✅ 5/5 sampel berhasil dikumpulkan! Menyimpan ke server...");
      scanline?.classList.add("hidden");
      webcam.stopAutoCapture();

      // Animasi sukses singkat sebelum pindah step
      await sleep(700);
      await finalizeEnrollment();
    } else {
      // ── Masih ada sampel berikutnya → kasih feedback ──
      setHint(`✅ Sampel ${sampleCount}/5 berhasil diambil!`);
      await sleep(700);

      // Arahkan ke slot kosong berikutnya
      const nextEmpty = findFirstEmptySlot();
      updatePoseGuide(nextEmpty < MAX_SAMPLES ? nextEmpty : sampleCount);
      setHint(POSE_GUIDE[nextEmpty < MAX_SAMPLES ? nextEmpty : sampleCount]?.hint
        || "🖐 Siap untuk sampel berikutnya...");
      await sleep(1300);
    }
  } finally {
    // Reset lock hanya kalau belum selesai semua (kalau 5 → finalizeEnrollment yg handle)
    if (sampleCount < MAX_SAMPLES) {
      isCapturing = false;
      scannerLoading?.classList.add("hidden");
    }
  }
}

// ─────────────────────────────────────────────────────────
// Error codes yang PASTI gagal kalau di-retry dengan blob sama
// (masalah ada di konten gambar, bukan jaringan/database)
// ─────────────────────────────────────────────────────────
const ML_UNRECOVERABLE_ERRORS = new Set([
  "detection_failed",
  "roi_extraction_failed",
  "image_too_blurry",
]);

// ─────────────────────────────────────────────────────────
// FINALIZE: Create user + upload 5 template → verify
//
// Strategi upload:
//  - ML error (blur/detection) → skip blob, lanjut berikutnya
//    → kalau kurang dari 5 sukses, kembali ke capture step
//  - Network/DB error → retry hingga 3×
//  - User tidak dibuat sebelum 5 blob berhasil dikumpulkan
// ─────────────────────────────────────────────────────────
async function finalizeEnrollment() {
  await showStep("verifying");
  initVerifyUI();

  let stepsDone = 0;
  const totalSteps = VERIFY_STEPS_CONFIG.length;
  const advance = () => setVerifyProgress(++stepsDone, totalSteps);

  try {
    // ── 1. Buat user di database ──────────────────────────
    setVerifyStep("create-user", "active");
    
    if (!isUserCreated) {
      setVerifyMsg("Membuat profil pengguna di server...");
      let newUser;
      try {
        newUser = await createUser(pendingName);
      } catch (createErr) {
        setVerifyStep("create-user", "error", "Gagal membuat profil");
  
        if (createErr.status === 409 || createErr.error === "user_exists") {
          return showRecoveryPrompt({
            title: "Nama sudah terdaftar",
            message: `Pengguna "${pendingName}" sudah ada. Ganti nama lalu coba lagi.`,
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
      currentUserId  = newUser.id;
      isUserCreated  = true;
    }
    
    setVerifyStep("create-user", "done", "Profil pengguna siap ✓");
    advance();

    // ── 2–6. Upload template satu per satu ───────────────
    // Blob yang gagal karena ML (blur/detection) di-skip → recapture nanti.
    // Blob yang gagal karena network/DB → retry max 3x.
    let uploadedCount = 0;
    const failedSlots = []; // index blob yang tidak bisa di-upload karena ML

    for (let i = 0; i < capturedBlobs.length; i++) {
      const stepId = `upload-${i + 1}`;
      setVerifyStep(stepId, "active");
      setVerifyMsg(`Mengunggah template biometrik ${i + 1}/5...`);

      // Beri waktu SQLite commit transaction sebelumnya
      await sleep(800);

      let lastErr  = null;
      let ok       = false;
      let isMLFail = false;

      for (let attempt = 1; attempt <= 3 && !ok && !isMLFail; attempt++) {
        try {
          await addTemplate(currentUserId, capturedBlobs[i], selectedEnrollmentModelId);
          ok = true;
        } catch (uploadErr) {
          lastErr = uploadErr;
          // ML error: blob ini tidak bisa di-embed, retry sama saja gagal
          if (ML_UNRECOVERABLE_ERRORS.has(uploadErr.error)) {
            isMLFail = true;
          } else if (attempt < 3) {
            // Network/DB error: tunggu lalu coba lagi
            await sleep(1500);
          }
        }
      }

      if (ok) {
        uploadedCount++;
        setVerifyStep(stepId, "done", `Template ${i + 1}/5 tersimpan ✓`);
        advance();
      } else if (isMLFail) {
        // Blob ini tidak bisa di-embed karena kualitas gambar
        failedSlots.push(i);
        setVerifyStep(
          stepId,
          "error",
          `Template ${i + 1}/5 — kualitas gambar kurang (akan di-scan ulang)`
        );
        // Jangan throw — lanjut ke blob berikutnya
      } else {
        // Network/DB error fatal setelah 3× retry
        setVerifyStep(stepId, "error", `Template ${i + 1}/5 — koneksi bermasalah`);
        throw new Error(
          `Upload template ${i + 1} gagal setelah 3 percobaan: ${
            lastErr?.message || "Koneksi terputus"
          }.`
        );
      }
    }

    // ── Cek apakah ada slot yang perlu di-recapture ──────
    if (failedSlots.length > 0) {
      // Kita sudah punya sebagian template di DB.
      // Minta user scan ulang hanya slot yang gagal.
      setVerifyMsg(
        `⚠️ ${uploadedCount}/5 template tersimpan. Perlu scan ulang ${failedSlots.length} sampel.`
      );
      toast.warning(
        `${failedSlots.length} sampel kurang jelas. Sistem akan membuka kamera untuk scan ulang.`,
        "Kualitas Gambar"
      );

      // Set blob yang gagal ke null (jangan di-splice agar index tidak bergeser)
      for (let j = failedSlots.length - 1; j >= 0; j--) {
        const idx = failedSlots[j];
        capturedBlobs[idx] = null;
        // Reset thumbnail slot ke state kosong
        const slot = document.getElementById(`sample-slot-${idx}`);
        if (slot) {
          slot.classList.remove("sample-slot--captured");
          slot.innerHTML = `
            <div class="sample-slot__number">${idx + 1}</div>
            <div class="sample-slot__icon">${POSE_GUIDE[idx]?.icon || "🖐"}</div>
          `;
        }
      }
      sampleCount = capturedBlobs.filter(Boolean).length;

      await sleep(1000);

      // Kembali ke capture step — kamera buka lagi
      await showStep("capture");
      updateCaptureProgressUI();
      isCapturing = false;
      await initWebcam();

      // Set hint sesuai dengan sampel berikutnya yang perlu di-capture
      const nextSlot = failedSlots[0];
      updatePoseGuide(nextSlot);
      setHint(`🔄 Scan ulang sampel ${nextSlot + 1}: ${POSE_GUIDE[nextSlot]?.hint || ""}`);
      return; // Tunggu user scan lagi; finalizeEnrollment akan dipanggil ulang saat 5 blob terkumpul
    }

    // ── 7. Verifikasi via /users/:id/verify-ready ────────
    setVerifyStep("verify", "active");
    setVerifyMsg("Memverifikasi kelengkapan template di database...");
    await sleep(800);

    const readyCheck = await verifyUserReady(currentUserId, selectedEnrollmentModelId, selectedEnrollmentVersion);

    if (!readyCheck.ready || readyCheck.template_count < 5) {
      setVerifyStep("verify", "error");
      throw new Error(
        `Hanya ${readyCheck.template_count}/5 template yang tersimpan. Coba ulangi enrollment.`
      );
    }

    setVerifyStep("verify", "done", `${readyCheck.template_count}/5 template terverifikasi ✓`);
    advance();
    setVerifyProgress(totalSteps, totalSteps);
    setVerifyMsg("✅ Template tersimpan. Lanjut isi data diri.");

    // ── 8. Tampilkan Form Profil ─────────────────────────────
    webcam.stop();
    buildSuccessScreen(readyCheck.template_count); // Build the thumbnails for later

    await sleep(500);
    await showStep("profile");

  } catch (error) {
    setVerifyMsg(`❌ ${error.message}`);
    toast.error(error.message, "Enrollment Gagal");
    console.error("Enrollment error:", error);

    // Rollback: Hapus user IMMEDIATELY kalau upload gagal agar tidak ada sisa user cacat di DB
    if (isUserCreated && currentUserId) {
      console.log(`Rolling back user ${currentUserId}...`);
      await deleteUser(currentUserId).catch((err) => {
        console.error("Rollback failed:", err);
      });
      isUserCreated = false;
      currentUserId = null;
    }

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

// ─────────────────────────────────────────────────────────
// THUMBNAIL RENDERING
// ─────────────────────────────────────────────────────────

/** Render thumbnail gambar ke slot pada panel progress */
function renderSampleThumbnail(blob, index) {
  const slot = document.getElementById(`sample-slot-${index}`);
  if (!slot) return;

  const url = URL.createObjectURL(blob);
  _thumbUrls.push(url);

  // Bersihkan isi slot (icon + nomor)
  slot.innerHTML = "";

  const img = document.createElement("img");
  img.src       = url;
  img.alt       = `Sampel ${index + 1}`;
  img.className = "sample-slot__img";

  const check = document.createElement("div");
  check.className   = "sample-slot__check";
  check.textContent = "✓";
  check.setAttribute("aria-label", `Sampel ${index + 1} berhasil`);

  slot.appendChild(img);
  slot.appendChild(check);
  slot.classList.add("sample-slot--captured");
}

/** Render 5 thumbnail di success screen */
function buildSuccessScreen(templateCount) {
  if (successName)          successName.textContent = currentUserName;
  if (successTemplateCount) successTemplateCount.textContent = `${templateCount} template tersimpan`;

  if (successThumbs) {
    successThumbs.innerHTML = "";
    capturedBlobs.forEach((blob, i) => {
      const url = URL.createObjectURL(blob);
      _thumbUrls.push(url);

      const div = document.createElement("div");
      div.className = "success-thumb";
      div.innerHTML = `
        <img src="${url}" alt="Sampel ${i + 1}">
        <div class="success-thumb__label">S${i + 1}</div>
      `;
      successThumbs.appendChild(div);
    });
  }
}

// ─────────────────────────────────────────────────────────
// POSE GUIDE UPDATES
// ─────────────────────────────────────────────────────────

/** Tandai pose item `index` sebagai active, sebelumnya jadi done */
function updatePoseGuide(nextIndex) {
  // Update bar poses
  const poseItems = document.querySelectorAll(".pose-item");
  poseItems.forEach((item, i) => {
    item.classList.remove("pose-item--active", "pose-item--done");
    if (i < nextIndex)       item.classList.add("pose-item--done");
    else if (i === nextIndex) item.classList.add("pose-item--active");
  });

  // Update instruction card bawah panel
  const pose = POSE_GUIDE[nextIndex] || POSE_GUIDE[MAX_SAMPLES - 1];
  if (poseInstructionIcon) poseInstructionIcon.textContent = pose.icon;
  if (poseInstructionText) poseInstructionText.textContent = pose.hint.replace(/^.{1,3}\s/, "");

  // Update eyebrow heading
  if (captureEyebrow) captureEyebrow.textContent = `STAGE 2 — SAMPLE ${nextIndex + 1} / 5`;
  if (scannerSampleNum) scannerSampleNum.textContent = String(nextIndex + 1);
}

/** Temukan index slot pertama yang masih kosong di capturedBlobs.
 * Dipakai saat recapture agar blob ditempatkan di slot yang benar.
 */
function findFirstEmptySlot() {
  for (let i = 0; i < MAX_SAMPLES; i++) {
    if (!capturedBlobs[i]) return i;
  }
  return MAX_SAMPLES - 1; // fallback (tidak seharusnya terjadi)
}

/** Mark slot sebagai sedang diproses (pulse animation) */
function setSlotActive(index) {
  document.getElementById(`sample-slot-${index}`)?.classList.add("sample-slot--active-slot");
}

/** Hapus state active dari slot */
function clearSlotActive(index) {
  document.getElementById(`sample-slot-${index}`)?.classList.remove("sample-slot--active-slot");
}

// ─────────────────────────────────────────────────────────
// RECOVERY PROMPT (inline di step-verifying)
// ─────────────────────────────────────────────────────────
function showRecoveryPrompt({ title, message, actionLabel, onAction, secondaryLabel, onSecondary }) {
  const container = document.getElementById("verify-steps-list")?.parentElement
    || steps.verifying;
  if (!container) return;

  const promptId = "enroll-recovery-prompt";
  document.getElementById(promptId)?.remove();

  const node = document.createElement("div");
  node.id        = promptId;
  node.className = "ux-anim-fade-in";
  node.style.cssText =
    "margin-top:var(--space-5);padding:var(--space-5);border:1px solid var(--color-border);" +
    "border-radius:12px;background:var(--color-surface-warm);text-align:center;";
  node.innerHTML = `
    <div style="font-size:1.8rem;margin-bottom:var(--space-3)">⚠️</div>
    <h3 style="margin:0 0 var(--space-2)">${escapeHtml(title)}</h3>
    <p class="text-sm" style="color:var(--color-coffee-light);margin:0 0 var(--space-5)">
      ${escapeHtml(message)}
    </p>
    <div style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap">
      <button type="button" class="btn btn--primary btn--sm" data-act="primary">${escapeHtml(actionLabel)}</button>
      ${secondaryLabel
        ? `<button type="button" class="btn btn--secondary btn--sm" data-act="secondary">${escapeHtml(secondaryLabel)}</button>`
        : ""}
    </div>
  `;
  container.appendChild(node);

  node.querySelector('[data-act="primary"]')?.addEventListener("click", () => {
    node.remove(); onAction?.();
  });
  node.querySelector('[data-act="secondary"]')?.addEventListener("click", () => {
    node.remove(); onSecondary?.();
  });
}

// ─────────────────────────────────────────────────────────
// VERIFY UI HELPERS
// ─────────────────────────────────────────────────────────
function initVerifyUI() {
  const list = document.getElementById("verify-steps-list");
  if (!list) return;
  list.innerHTML = VERIFY_STEPS_CONFIG.map((s) => `
    <div class="vstep vstep--pending" id="vstep-${s.id}" role="listitem">
      <div class="vstep__indicator" aria-hidden="true">–</div>
      <span class="vstep__label">${s.label}</span>
    </div>
  `).join("");
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
    else if (state === "done")  indicator.textContent = "✓";
    else if (state === "error") indicator.textContent = "✕";
    else                        indicator.textContent = "–";
  }
  if (overrideLabel) {
    const label = el.querySelector(".vstep__label");
    if (label) label.textContent = overrideLabel;
  }
  if (state === "active") el.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
}

function setVerifyProgress(done, total) {
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  const fill  = document.getElementById("verify-progress-fill");
  const label = document.getElementById("verify-pct");
  if (fill)  fill.style.width    = `${pct}%`;
  if (label) label.textContent   = `${pct}%`;
}

function setVerifyMsg(msg) {
  const el = document.getElementById("verify-status-msg");
  if (el) el.textContent = msg;
}

// ─────────────────────────────────────────────────────────
// CAPTURE UI HELPERS
// ─────────────────────────────────────────────────────────

/** Reset semua UI capture ke state awal */
function resetCaptureUI() {
  sampleCount  = 0;
  capturedBlobs = new Array(MAX_SAMPLES).fill(null);

  // Reset thumbnail slots
  for (let i = 0; i < MAX_SAMPLES; i++) {
    const slot = document.getElementById(`sample-slot-${i}`);
    if (slot) {
      slot.classList.remove("sample-slot--captured", "sample-slot--active-slot");
      slot.innerHTML = `
        <div class="sample-slot__number">${i + 1}</div>
        <div class="sample-slot__icon">${POSE_GUIDE[i].icon}</div>
      `;
    }
  }

  // Reset pose guide
  const poseItems = document.querySelectorAll(".pose-item");
  poseItems.forEach((item, i) => {
    item.classList.remove("pose-item--active", "pose-item--done");
    if (i === 0) item.classList.add("pose-item--active");
  });

  // Reset instruction card
  if (poseInstructionIcon) poseInstructionIcon.textContent = POSE_GUIDE[0].icon;
  if (poseInstructionText) poseInstructionText.textContent = POSE_GUIDE[0].hint.replace(/^.{1,3}\s/, "");

  // Reset dots & badge
  updateCaptureProgressUI();

  // Reset header
  if (captureEyebrow)   captureEyebrow.textContent  = "STAGE 2 — SAMPLE 1 / 5";
  if (scannerSampleNum) scannerSampleNum.textContent = "1";

  clearQualityHint();
  revokeThumbs();
}

function updateCaptureProgress() {
  updateCaptureProgressUI();

  // Update pose guide setelah sampel berhasil (kecuali yang terakhir)
  if (sampleCount < MAX_SAMPLES) {
    updatePoseGuide(sampleCount);
  }
}

function updateCaptureProgressUI() {
  if (sampleBadge) {
    sampleBadge.textContent = `${sampleCount} / ${MAX_SAMPLES}`;
    sampleBadge.className   = "badge " +
      (sampleCount >= MAX_SAMPLES ? "badge--identified" : "badge--scanning");
  }

  // ARIA progress bar
  const stepper = document.getElementById("sample-stepper-wrapper");
  if (stepper) stepper.setAttribute("aria-valuenow", String(sampleCount));

  sampleDots.forEach((dot, idx) => {
    dot.classList.remove("active", "completed");
    if (idx < sampleCount)    dot.classList.add("completed");
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
  captureQualityHint.textContent = `⚠️ ${text}`;
  captureQualityHint.classList.remove("hidden");
}

function clearQualityHint() {
  if (!captureQualityHint) return;
  captureQualityHint.classList.add("hidden");
  captureQualityHint.textContent = "";
}

// ─────────────────────────────────────────────────────────
// STATE RESET & CLEANUP
// ─────────────────────────────────────────────────────────
function resetLocalState() {
  modelSelector?.setDisabled(false);
  capturedBlobs  = new Array(MAX_SAMPLES).fill(null);

  sampleCount    = 0;
  currentUserId  = null;
  isUserCreated  = false;
  isCapturing    = false;
  revokeThumbs();
}

async function restartFromName() {
  webcam?.stop();
  resetLocalState();
  pendingName     = "";
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

/** Revoke semua object URLs agar tidak leak memory */
function revokeThumbs() {
  _thumbUrls.forEach((url) => URL.revokeObjectURL(url));
  _thumbUrls.length = 0;
}

function cleanupOnExit() {
  webcam?.stop();
  revokeThumbs();
  // Rollback user yang dibuat tapi tidak selesai
  if (isUserCreated && currentUserId && sampleCount < MAX_SAMPLES) {
    const url = `${BASE_URL}/users/${currentUserId}`;
    fetch(url, { method: "DELETE", keepalive: true }).catch(() => {});
  }
}

// ─────────────────────────────────────────────────────────
// UTILITIES
// ─────────────────────────────────────────────────────────
function escapeHtml(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[c]
  );
}

// ── Bootstrap ─────────────────────────────────────────────
init();

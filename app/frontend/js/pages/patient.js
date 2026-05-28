// ============================================================
// js/pages/patient.js — Patient Check-in demo
// Full implementation: scanner → patient card → confirm check-in
// ============================================================

import { mountNavbar } from "../components/navbar.js";
import { PalmScanner } from "../components/palm-scanner.js";
import { DiagnosticTerminal } from "../components/terminal.js";
import { showModal } from "../components/modal.js";
import { toast } from "../components/toast.js";
import { apiFetch } from "../api/client.js";
import { maskString } from "../utils.js";

// ── State ──────────────────────────────────────────────────
let scanner = null;
let terminal = null;
let isProcessing = false;
let pendingUser = null;
let pendingPatient = null;

// ── DOM ────────────────────────────────────────────────────
const scannerContainer = document.getElementById("scanner-container");
const scannerVideo = document.getElementById("scanner-video");
const scannerHint = document.getElementById("scanner-hint");
const scannerResult = document.getElementById("scanner-result");
const scannerPlaceholder = document.getElementById("scanner-placeholder");
const terminalBody = document.getElementById("terminal-body");
const patientPanel = document.getElementById("patient-panel");
const btnStartScan = document.getElementById("btn-start-scan");
const btnConfirmCheckin = document.getElementById("btn-confirm-checkin");

// ── Init ───────────────────────────────────────────────────
function init() {
  mountNavbar();

  terminal = new DiagnosticTerminal(terminalBody);
  terminal.addLog("CAMERA_READY", "Patient check-in system initialized.");

  scanner = new PalmScanner({
    containerEl: scannerContainer,
    videoEl: scannerVideo,
    hintEl: scannerHint,
    resultEl: scannerResult,
    placeholderEl: scannerPlaceholder,
    logFn: (tag, msg) => terminal.addLog(tag, msg),
    onIdentified: handleIdentified,
    onUnknown: (score) => {
      terminal.addLog("RESULT", `UNKNOWN — score ${score.toFixed(4)}`);
      toast.warning("Pasien tidak dikenali. Silakan hubungi resepsionis.");
      showPatientNotFound();
    },
    autoResetMs: 5000,
  });

  btnStartScan?.addEventListener("click", startScanner);
  btnConfirmCheckin?.addEventListener("click", confirmCheckin);

  window.addEventListener("beforeunload", () => scanner?.stop());
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) scanner?.stop();
  });
}

async function startScanner() {
  if (btnStartScan) {
    btnStartScan.disabled = true;
    btnStartScan.innerHTML = `<span class="spinner spinner--sm"></span> Scanning…`;
  }
  clearPatientCard();
  terminal.addLog("SYSTEM", "Patient scanner started.");
  await scanner.start();
}

// ── Identification callback ────────────────────────────────
async function handleIdentified(user, score, latency) {
  if (isProcessing) return;
  isProcessing = true;

  terminal.addLog("MATCHING", `${user.name} — score ${score.toFixed(4)}`);
  scanner.stop();

  try {
    const result = await apiFetch("/demos/patient/checkin", {
      method: "POST",
      body: JSON.stringify({ user_id: user.id, match_score: score }),
    });

    terminal.addLog("RESULT", `PATIENT FOUND — ${user.name}`);
    pendingUser = result.user;
    pendingPatient = result.patient;
    showPatientCard(result.user, result.patient, score);
    if (btnConfirmCheckin) btnConfirmCheckin.classList.remove("hidden");
  } catch (err) {
    terminal.addLog("ERROR", err.message);
    toast.error("Gagal mengambil data pasien.");
  } finally {
    isProcessing = false;
    resetStartButton();
  }
}

// ── Patient card ───────────────────────────────────────────
function showPatientCard(user, patient, score) {
  if (!patientPanel) return;

  patientPanel.innerHTML = `
    <div class="patient-card">
      <div class="patient-header">
        <div style="width:56px;height:56px;border-radius:12px;background:var(--color-coffee);display:flex;align-items:center;justify-content:center;font-family:var(--font-heading);font-size:1.5rem;color:white;flex-shrink:0">
          ${escHtml(user.name[0])}
        </div>
        <div>
          <div style="font-family:var(--font-heading);font-size:1.2rem;font-weight:700">${escHtml(user.name)}</div>
          <div style="display:flex;gap:8px;margin-top:4px">
            <span class="badge badge--identified"><span class="status-dot"></span>Teridentifikasi</span>
            <span style="font-family:monospace;font-size:11px;color:var(--color-coffee-light)">Score: ${score.toFixed(4)}</span>
          </div>
        </div>
      </div>

      <div class="patient-meta">
        <div>
          <div class="patient-field-label">NIK</div>
          <div class="patient-field-value" style="font-family:monospace">${escHtml(patient.nik)}</div>
        </div>
        <div>
          <div class="patient-field-label">No. Rekam Medik</div>
          <div class="patient-field-value" style="font-family:monospace">${escHtml(patient.rekam_medik)}</div>
        </div>
        <div>
          <div class="patient-field-label">Dokter PJ</div>
          <div class="patient-field-value">${escHtml(patient.dokter)}</div>
        </div>
        <div>
          <div class="patient-field-label">Jadwal</div>
          <div class="patient-field-value">${escHtml(patient.jadwal)}</div>
        </div>
        <div>
          <div class="patient-field-label">Kunjungan Terakhir</div>
          <div class="patient-field-value">${escHtml(patient.last_visit)}</div>
        </div>
        <div>
          <div class="patient-field-label">Waktu Check-in</div>
          <div class="patient-field-value">${new Date().toLocaleTimeString("id-ID")}</div>
        </div>
      </div>

      <div class="hint-box hint-box--info" style="margin-top:16px">
        <p class="text-xs">Periksa data pasien lalu klik <strong>Konfirmasi Check-in</strong> untuk menyelesaikan proses.</p>
      </div>
    </div>
  `;
}

function showPatientNotFound() {
  if (!patientPanel) return;
  patientPanel.innerHTML = `
    <div style="text-align:center;padding:40px 20px">
      <div style="font-size:3rem;margin-bottom:16px">❌</div>
      <h3 style="color:var(--color-coral)">Pasien Tidak Dikenali</h3>
      <p class="text-sm" style="margin-top:8px">Silakan hubungi petugas resepsionis untuk proses manual.</p>
    </div>
  `;
}

function clearPatientCard() {
  if (!patientPanel) return;
  patientPanel.innerHTML = `
    <div style="text-align:center;padding:40px 20px;opacity:0.4">
      <div style="font-size:3rem;margin-bottom:12px">🏥</div>
      <p>Pindai telapak tangan pasien untuk melihat data rekam medis.</p>
    </div>
  `;
  if (btnConfirmCheckin) btnConfirmCheckin.classList.add("hidden");
  pendingUser = null;
  pendingPatient = null;
}

async function confirmCheckin() {
  if (!pendingUser) return;

  showModal({
    title: "Konfirmasi Check-in Pasien",
    icon: "✅",
    message: `Check-in untuk <strong>${escHtml(pendingUser.name)}</strong> pada ${new Date().toLocaleString("id-ID")} akan dicatat ke sistem.`,
    confirmLabel: "Konfirmasi Check-in",
    confirmVariant: "success",
    onConfirm: async () => {
      toast.success(
        `Check-in pasien ${pendingUser.name} berhasil.`,
        "Berhasil",
      );
      terminal.addLog("RESULT", `CHECK-IN CONFIRMED — ${pendingUser.name}`);
      if (btnConfirmCheckin) btnConfirmCheckin.classList.add("hidden");

      // Reset after 2s
      setTimeout(() => {
        clearPatientCard();
        resetStartButton();
      }, 2000);
    },
  });
}

function resetStartButton() {
  if (btnStartScan) {
    btnStartScan.disabled = false;
    btnStartScan.textContent = "Pindai Pasien";
  }
}

function escHtml(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;",
      })[c],
  );
}

init();

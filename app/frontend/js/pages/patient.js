// ============================================================
// js/pages/patient.js — Patient Check-in demo
// Fixed: API call, patient card render, confirm flow, terminal
// ============================================================

import { mountNavbar } from "../components/navbar.js";
import { PalmScanner } from "../components/palm-scanner.js";
import { DiagnosticTerminal } from "../components/terminal.js";
import { showModal } from "../components/modal.js";
import { toast } from "../components/toast.js";
import { apiFetch } from "../api/client.js";

// ── State ──────────────────────────────────────────────────
let scanner = null;
let terminal = null;
let isProcessing = false;
let pendingUser = null;
let pendingPatient = null;

// ── DOM ────────────────────────────────────────────────────
const scannerContainer = document.getElementById("scanner-container");
const terminalBody = document.getElementById("terminal-body");
const patientPanel = document.getElementById("patient-panel");
const btnStartScan = document.getElementById("btn-start-scan");
const btnConfirmCheckin = document.getElementById("btn-confirm-checkin");

// ── Init ───────────────────────────────────────────────────
function init() {
  mountNavbar();

  terminal = new DiagnosticTerminal(terminalBody);
  terminal.addLog("SYSTEM", "Patient check-in system initialized.");
  terminal.addLog("INFO", "Klik 'Pindai Pasien' untuk memulai identifikasi.");

  scanner = new PalmScanner({
    containerEl: scannerContainer,
    videoEl: document.getElementById("scanner-video"),
    hintEl: document.getElementById("scanner-hint"),
    resultEl: document.getElementById("scanner-result"),
    placeholderEl: document.getElementById("scanner-placeholder"),
    logFn: (tag, msg) => terminal.addLog(tag, msg),
    onIdentified: handleIdentified,
    onUnknown: (score) => {
      terminal.addLog("RESULT", `UNKNOWN — score ${score.toFixed(4)}`);
      toast.warning("Pasien tidak dikenali. Silakan hubungi resepsionis.");
      showPatientNotFound();
      enableStartButton();
    },
    autoResetMs: 6000,
    captureIntervalMs: 1500,
  });

  btnStartScan?.addEventListener("click", startScanner);
  btnConfirmCheckin?.addEventListener("click", confirmCheckin);

  window.addEventListener("beforeunload", () => scanner?.stop());
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) scanner?.stop();
  });
}

// ── Scanner ────────────────────────────────────────────────
async function startScanner() {
  if (btnStartScan) {
    btnStartScan.disabled = true;
    btnStartScan.innerHTML = `<span class="spinner spinner--sm"></span> Scanning…`;
  }
  clearPatientCard();
  terminal.addLog("SYSTEM", "Patient scanner started.");
  await scanner.start();
}

function enableStartButton() {
  if (btnStartScan) {
    btnStartScan.disabled = false;
    btnStartScan.textContent = "🖐 Pindai Pasien";
  }
}

// ── Identification callback ────────────────────────────────
async function handleIdentified(user, score, latency) {
  if (isProcessing) return;
  isProcessing = true;

  terminal.addLog(
    "MATCHING",
    `${user.name} — score ${score.toFixed(4)} — ${latency}ms`,
  );

  // Stop auto-scan while showing patient card
  scanner.stop();

  try {
    const result = await apiFetch("/demos/patient/checkin", {
      method: "POST",
      body: JSON.stringify({ user_id: user.id, match_score: score }),
    });

    terminal.addLog("RESULT", `✅ PATIENT FOUND — ${user.name}`);
    terminal.addLog(
      "PATIENT",
      `Rekam medik: ${result.patient?.rekam_medik || "—"}`,
    );
    terminal.addLog("PATIENT", `Dokter PJ: ${result.patient?.dokter || "—"}`);

    pendingUser = result.user || user;
    pendingPatient = result.patient || {};

    showPatientCard(pendingUser, pendingPatient, score);

    if (btnConfirmCheckin) btnConfirmCheckin.classList.remove("hidden");
  } catch (err) {
    terminal.addLog("ERROR", err.message || "Gagal mengambil data pasien");
    toast.error("Gagal mengambil data pasien. Cek koneksi backend.");
    showPatientError(err.message);
    enableStartButton();
  } finally {
    isProcessing = false;
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
          <div style="display:flex;gap:8px;margin-top:4px;flex-wrap:wrap">
            <span class="badge badge--identified">
              <span class="status-dot status-dot--pulse"></span>Teridentifikasi
            </span>
            <span style="font-family:monospace;font-size:11px;color:var(--color-coffee-light)">
              Score: ${score.toFixed(4)}
            </span>
          </div>
        </div>
      </div>

      <div class="patient-meta">
        <div>
          <div class="patient-field-label">NIK</div>
          <div class="patient-field-value" style="font-family:monospace">${escHtml(patient.nik || "—")}</div>
        </div>
        <div>
          <div class="patient-field-label">No. Rekam Medik</div>
          <div class="patient-field-value" style="font-family:monospace">${escHtml(patient.rekam_medik || "—")}</div>
        </div>
        <div>
          <div class="patient-field-label">Dokter PJ</div>
          <div class="patient-field-value">${escHtml(patient.dokter || "—")}</div>
        </div>
        <div>
          <div class="patient-field-label">Jadwal</div>
          <div class="patient-field-value">${escHtml(patient.jadwal || "—")}</div>
        </div>
        <div>
          <div class="patient-field-label">Kunjungan Terakhir</div>
          <div class="patient-field-value">${escHtml(patient.last_visit || "—")}</div>
        </div>
        <div>
          <div class="patient-field-label">Waktu Check-in</div>
          <div class="patient-field-value">${new Date().toLocaleString("id-ID")}</div>
        </div>
      </div>

      <div class="hint-box hint-box--info" style="margin-top:16px">
        <p class="text-xs">Periksa data pasien lalu klik <strong>Konfirmasi Check-in</strong> di bawah untuk menyelesaikan proses.</p>
      </div>
    </div>
  `;
}

function showPatientNotFound() {
  if (!patientPanel) return;
  patientPanel.innerHTML = `
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:var(--space-10) var(--space-6);text-align:center">
      <div style="font-size:3rem;margin-bottom:var(--space-4)">❌</div>
      <h3 style="color:var(--color-coral)">Pasien Tidak Dikenali</h3>
      <p class="text-sm" style="margin-top:8px;color:var(--color-coffee-light)">
        Silakan hubungi petugas resepsionis untuk proses manual.
      </p>
    </div>
  `;
  if (btnConfirmCheckin) btnConfirmCheckin.classList.add("hidden");
}

function showPatientError(message) {
  if (!patientPanel) return;
  patientPanel.innerHTML = `
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:var(--space-10) var(--space-6);text-align:center">
      <div style="font-size:3rem;margin-bottom:var(--space-4)">⚠️</div>
      <h3 style="color:var(--color-honey)">Gagal Mengambil Data</h3>
      <p class="text-sm" style="margin-top:8px;color:var(--color-coffee-light)">${escHtml(message || "Cek koneksi backend.")}</p>
    </div>
  `;
  if (btnConfirmCheckin) btnConfirmCheckin.classList.add("hidden");
}

function clearPatientCard() {
  if (!patientPanel) return;
  patientPanel.innerHTML = `
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:var(--space-10) var(--space-4);opacity:0.35;text-align:center">
      <div style="font-size:3rem;margin-bottom:var(--space-4)">🏥</div>
      <p class="text-sm">Pindai telapak tangan pasien untuk melihat data rekam medis.</p>
    </div>
  `;
  if (btnConfirmCheckin) btnConfirmCheckin.classList.add("hidden");
  pendingUser = null;
  pendingPatient = null;
}

// ── Confirm check-in ───────────────────────────────────────
async function confirmCheckin() {
  if (!pendingUser) return;

  showModal({
    title: "Konfirmasi Check-in Pasien",
    icon: "✅",
    message: `
      Check-in untuk <strong>${escHtml(pendingUser.name)}</strong>
      pada <strong>${new Date().toLocaleString("id-ID")}</strong>
      akan dicatat ke sistem.
    `,
    confirmLabel: "Konfirmasi Check-in",
    confirmVariant: "success",
    onConfirm: async () => {
      toast.success(
        `Check-in pasien ${pendingUser.name} berhasil.`,
        "Berhasil",
      );
      terminal.addLog("RESULT", `CHECK-IN CONFIRMED — ${pendingUser.name}`);

      if (btnConfirmCheckin) btnConfirmCheckin.classList.add("hidden");

      // Reset after short delay so user can see success state
      setTimeout(() => {
        clearPatientCard();
        enableStartButton();
      }, 2000);
    },
    onCancel: () => {
      // Allow re-confirm
    },
  });
}

// ── Escape helper ──────────────────────────────────────────
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

// ============================================================
// js/pages/patient.js — Patient Check-in (UX-improved)
//
// Perubahan UX:
// • Patient card masuk dengan fade animation (bukan tiba-tiba).
// • Loading state khusus saat fetch data pasien (skeleton card).
// • Error fetch: tombol "Coba lagi" inline.
// • Setelah confirm success → fade-out + reset, bukan setTimeout
//   tanpa indikator.
// ============================================================

import { mountNavbar } from "../components/navbar.js";
import { PalmScanner } from "../components/palm-scanner.js";
import { DiagnosticTerminal } from "../components/terminal.js";
import { showModal } from "../components/modal.js";
import { toast } from "../components/toast.js";
import { apiFetch } from "../api/client.js";
import { animateIn, fadeSwap } from "../ux.js";

let scanner = null;
let terminal = null;
let isProcessing = false;
let pendingUser = null;
let pendingPatient = null;

let scannerContainer, terminalCard, patientPanel, btnStartScan, btnConfirmCheckin;

async function init() {
  mountNavbar();

  scannerContainer = document.getElementById("scanner-container");
  terminalCard = document.querySelector(".terminal-card");
  patientPanel = document.getElementById("patient-panel");
  btnStartScan = document.getElementById("btn-start-scan");
  btnConfirmCheckin = document.getElementById("btn-confirm-checkin");

  if (!btnStartScan || !terminalCard || !scannerContainer) {
    console.error("[Patient] Required DOM elements not found!");
    return;
  }

  terminal = new DiagnosticTerminal(terminalCard);
  document.addEventListener("toast", (e) => {
    if (e.detail && toast[e.detail.type]) toast[e.detail.type](e.detail.message);
  });

  terminal.addLog("SYSTEM", "Patient check-in system initialized.");
  terminal.addLog("INSTRUCTION", "Langkah 1: Klik 'Aktifkan Scanner Pasien'.");
  terminal.addLog("INSTRUCTION", "Langkah 2: Arahkan telapak tangan pasien ke kamera.");

  scanner = new PalmScanner({
    containerEl: scannerContainer,
    videoEl: document.getElementById("scanner-video"),
    hintEl: document.getElementById("scanner-hint"),
    resultEl: document.getElementById("scanner-result"),
    placeholderEl: document.getElementById("scanner-placeholder"),
    boundingBoxLayerEl: document.getElementById("palm-box-layer"),
    logFn: (tag, msg) => terminal?.addLog(tag, msg),
    onIdentified: handleIdentified,
    onUnknown: (score) => {
      terminal?.addLog("RESULT", `UNKNOWN — score ${score.toFixed(4)}`);
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

async function startScanner() {
  if (btnStartScan) {
    btnStartScan.disabled = true;
    btnStartScan.innerHTML = `<span class="spinner spinner--sm"></span> Scanning…`;
  }
  clearPatientCard();
  terminal.addLog("SYSTEM", "Patient scanner started.");
  const ok = await scanner.start();
  if (!ok) {
    enableStartButton();
    toast.error("Gagal menyalakan kamera. Periksa izin browser.");
  }
}

function enableStartButton() {
  if (btnStartScan) {
    btnStartScan.disabled = false;
    btnStartScan.textContent = "🖐 Pindai Pasien";
  }
}

async function handleIdentified(user, score, latency) {
  if (isProcessing) return;
  isProcessing = true;

  terminal.addLog(
    "MATCHING",
    `${user.name} — score ${score.toFixed(4)} — ${latency}ms`,
  );
  scanner.pause();
  showPatientLoading();

  try {
    const result = await apiFetch("/demos/patient/checkin", {
      method: "POST",
      body: JSON.stringify({ user_id: user.id, match_score: score }),
    });

    terminal.addLog("RESULT", `✅ PATIENT FOUND — ${user.name}`);
    pendingUser = result.user || user;
    pendingPatient = result.patient || {};

    showPatientCard(pendingUser, pendingPatient, score);
    if (btnConfirmCheckin) btnConfirmCheckin.classList.remove("hidden");
  } catch (err) {
    terminal.addLog("ERROR", err.message || "Gagal mengambil data pasien");
    showPatientError(err.message, () => handleIdentified(user, score, latency));
    enableStartButton();
    scanner.resume();
  } finally {
    isProcessing = false;
  }
}

function showPatientLoading() {
  if (!patientPanel) return;
  patientPanel.innerHTML = `
    <div class="patient-card ux-anim-fade-in" style="padding:var(--space-5)">
      <div style="display:flex;gap:12px;align-items:center;margin-bottom:16px">
        <span class="ux-skeleton ux-skeleton--avatar" style="width:56px;height:56px"></span>
        <div style="flex:1">
          <span class="ux-skeleton ux-skeleton--line" style="width:50%;height:18px"></span>
          <div style="height:8px"></div>
          <span class="ux-skeleton ux-skeleton--line" style="width:30%"></span>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
        ${Array.from({ length: 6 })
          .map(
            () => `
          <div>
            <span class="ux-skeleton ux-skeleton--line" style="width:60%;height:9px"></span>
            <div style="height:6px"></div>
            <span class="ux-skeleton ux-skeleton--line" style="width:80%"></span>
          </div>
        `,
          )
          .join("")}
      </div>
    </div>
  `;
}

function showPatientCard(user, patient, score) {
  if (!patientPanel) return;
  patientPanel.innerHTML = `
    <div class="patient-card ux-anim-fade-in">
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
            <span style="font-family:monospace;font-size:11px;color:var(--color-coffee-light)">Score: ${score.toFixed(4)}</span>
          </div>
        </div>
      </div>
      <div class="patient-meta">
        <div><div class="patient-field-label">NIK</div><div class="patient-field-value" style="font-family:monospace">${escHtml(patient.nik || "—")}</div></div>
        <div><div class="patient-field-label">No. Rekam Medik</div><div class="patient-field-value" style="font-family:monospace">${escHtml(patient.rekam_medik || "—")}</div></div>
        <div><div class="patient-field-label">Dokter PJ</div><div class="patient-field-value">${escHtml(patient.dokter || "—")}</div></div>
        <div><div class="patient-field-label">Jadwal</div><div class="patient-field-value">${escHtml(patient.jadwal || "—")}</div></div>
        <div><div class="patient-field-label">Kunjungan Terakhir</div><div class="patient-field-value">${escHtml(patient.last_visit || "—")}</div></div>
        <div><div class="patient-field-label">Waktu Check-in</div><div class="patient-field-value">${new Date().toLocaleString("id-ID")}</div></div>
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
    <div class="ux-anim-fade-in" style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:var(--space-10) var(--space-6);text-align:center">
      <div class="status-icon status-icon--error" style="margin-bottom:var(--space-4)">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
      </div>
      <h3 style="color:var(--color-coral)">Pasien Tidak Dikenali</h3>
      <p class="text-sm" style="margin-top:8px;color:var(--color-coffee-light)">Silakan hubungi petugas resepsionis untuk proses manual.</p>
    </div>
  `;
  if (btnConfirmCheckin) btnConfirmCheckin.classList.add("hidden");
}

function showPatientError(message, onRetry) {
  if (!patientPanel) return;
  patientPanel.innerHTML = `
    <div class="ux-anim-fade-in" style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:var(--space-10) var(--space-6);text-align:center;gap:var(--space-3)">
      <div class="status-icon status-icon--warning">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
      </div>
      <h3 style="color:var(--color-honey);margin:0">Gagal Mengambil Data</h3>
      <p class="text-sm" style="color:var(--color-coffee-light);margin:0">${escHtml(message || "Cek koneksi backend.")}</p>
      <button type="button" class="btn btn--secondary btn--sm" id="patient-retry">Coba Lagi</button>
    </div>
  `;
  patientPanel.querySelector("#patient-retry")?.addEventListener("click", () => onRetry?.());
  if (btnConfirmCheckin) btnConfirmCheckin.classList.add("hidden");
}

function clearPatientCard() {
  if (!patientPanel) return;
  patientPanel.innerHTML = `
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:var(--space-10) var(--space-4);opacity:0.35;text-align:center">
      <div style="margin-bottom:var(--space-4)">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 6v4"/><path d="M14 14h-4"/><path d="M14 18h-4"/><path d="M14 8h-4"/><path d="M18 12h2a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-6a2 2 0 0 1 2-2h2"/><path d="M6 22V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v18"/></svg>
      </div>
      <p class="text-sm">Pindai telapak tangan pasien untuk melihat data rekam medis.</p>
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
    icon: "success",
    message: `Check-in untuk <strong>${escHtml(pendingUser.name)}</strong> pada <strong>${new Date().toLocaleString("id-ID")}</strong> akan dicatat ke sistem.`,
    confirmLabel: "Konfirmasi Check-in",
    confirmVariant: "success",
    onConfirm: async () => {
      toast.success(`Check-in pasien ${pendingUser.name} berhasil.`, "Berhasil");
      terminal.addLog("RESULT", `CHECK-IN CONFIRMED — ${pendingUser.name}`);
      if (btnConfirmCheckin) btnConfirmCheckin.classList.add("hidden");

      // Fade-out lalu reset → user lihat transisi yang halus
      await fadeSwap(patientPanel, () => clearPatientCard(), 260);
      enableStartButton();
      scanner.resume();
    },
    onCancel: () => scanner.resume(),
  });
}

function escHtml(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[c],
  );
}

init();

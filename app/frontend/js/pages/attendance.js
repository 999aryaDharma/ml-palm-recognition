// ============================================================
// js/pages/attendance.js — Palm Attendance demo logic
// Fixed: result display in list, count tracking, proper scanner elements
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
let currentMode = "checkin"; // "checkin" | "checkout"
let isProcessing = false;
let countCheckin = 0;
let countCheckout = 0;

// ── DOM ────────────────────────────────────────────────────
const scannerContainer = document.getElementById("scanner-container");
const terminalBody = document.getElementById("terminal-body");
const attendanceList = document.getElementById("attendance-list");
const btnCheckin = document.getElementById("btn-mode-checkin");
const btnCheckout = document.getElementById("btn-mode-checkout");
const btnStartScan = document.getElementById("btn-start-scan");
const modeLabelEl = document.getElementById("current-mode-label");
const countCheckinEl = document.getElementById("count-checkin");
const countCheckoutEl = document.getElementById("count-checkout");

// ── Init ───────────────────────────────────────────────────
function init() {
  mountNavbar();

  terminal = new DiagnosticTerminal(terminalBody);
  terminal.addLog("SYSTEM", "Attendance system initialized.");
  terminal.addLog(
    "INFO",
    "Pilih mode Check-in atau Check-out lalu klik Mulai Scan.",
  );

  // Mode buttons
  btnCheckin?.addEventListener("click", () => setMode("checkin"));
  btnCheckout?.addEventListener("click", () => setMode("checkout"));

  // Scanner
  scanner = new PalmScanner({
    containerEl: scannerContainer,
    videoEl: document.getElementById("scanner-video"),
    hintEl: document.getElementById("scanner-hint"),
    resultEl: document.getElementById("scanner-result"),
    placeholderEl: document.getElementById("scanner-placeholder"),
    logFn: (tag, msg) => terminal.addLog(tag, msg),
    onIdentified: handleIdentified,
    onUnknown: (score) => {
      toast.warning("Pengguna tidak dikenali. Coba scan ulang.");
      terminal.addLog("RESULT", `UNKNOWN — score ${score.toFixed(4)}`);
      enableStartButton();
    },
    autoResetMs: 4000,
    captureIntervalMs: 1500,
  });

  btnStartScan?.addEventListener("click", startScanner);

  window.addEventListener("beforeunload", () => scanner?.stop());
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) scanner?.stop();
  });
}

// ── Mode toggle ────────────────────────────────────────────
function setMode(mode) {
  currentMode = mode;
  if (modeLabelEl)
    modeLabelEl.textContent = mode === "checkin" ? "Check-in" : "Check-out";

  btnCheckin?.classList.toggle("btn--primary", mode === "checkin");
  btnCheckin?.classList.toggle("btn--secondary", mode !== "checkin");
  btnCheckout?.classList.toggle("btn--primary", mode === "checkout");
  btnCheckout?.classList.toggle("btn--secondary", mode !== "checkout");

  terminal.addLog("SYSTEM", `Mode: ${mode.toUpperCase()} aktif.`);
}

// ── Scanner ────────────────────────────────────────────────
async function startScanner() {
  if (btnStartScan) {
    btnStartScan.disabled = true;
    btnStartScan.innerHTML = `<span class="spinner spinner--sm"></span> Scanning…`;
  }
  terminal.addLog("SYSTEM", `Attendance scanner (${currentMode}) started.`);
  await scanner.start();
}

function enableStartButton() {
  if (btnStartScan) {
    btnStartScan.disabled = false;
    btnStartScan.textContent = "🖐 Mulai Scan Absensi";
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

  const modeText = currentMode === "checkin" ? "Check-in" : "Check-out";
  const modeIcon = currentMode === "checkin" ? "📋" : "🚪";

  showModal({
    title: `Konfirmasi ${modeText}`,
    icon: modeIcon,
    message: `
      <div class="flex flex-col gap-4">
        <p>Catat <strong>${modeText}</strong> untuk:</p>
        <div style="display:flex;align-items:center;gap:12px;padding:12px;background:var(--color-surface-warm);border-radius:12px;border:1px solid var(--color-border)">
          <div style="width:40px;height:40px;border-radius:8px;background:var(--color-coffee);display:flex;align-items:center;justify-content:center;color:white;font-weight:700">
            ${escHtml(user.name[0])}
          </div>
          <div>
            <div style="font-weight:600">${escHtml(user.name)}</div>
            <div class="text-xs" style="color:var(--color-coffee-light)">Score: ${(score * 100).toFixed(1)}% · ${latency}ms</div>
          </div>
        </div>
        <p class="text-sm">Waktu: <strong>${new Date().toLocaleTimeString("id-ID")}</strong></p>
      </div>
    `,
    confirmLabel: `Konfirmasi ${modeText}`,
    confirmVariant: currentMode === "checkin" ? "success" : "primary",
    onConfirm: () => submitAttendance(user, score),
    onCancel: () => {
      isProcessing = false;
      scanner.resume();
      enableStartButton();
    },
  });
}

// ── Submit to backend ──────────────────────────────────────
async function submitAttendance(user, score) {
  try {
    const result = await apiFetch("/demos/attendance/checkin", {
      method: "POST",
      body: JSON.stringify({
        user_id: user.id,
        mode: currentMode,
        match_score: score,
      }),
    });

    terminal.addLog(
      "RESULT",
      `${currentMode.toUpperCase()} recorded — ${user.name}`,
    );
    toast.success(
      `${currentMode === "checkin" ? "Check-in" : "Check-out"} berhasil: ${user.name}`,
      "Absensi Tercatat",
    );

    // Update counters
    if (currentMode === "checkin") {
      countCheckin++;
      if (countCheckinEl) countCheckinEl.textContent = String(countCheckin);
    } else {
      countCheckout++;
      if (countCheckoutEl) countCheckoutEl.textContent = String(countCheckout);
    }

    addAttendanceEntry(user, result);
  } catch (err) {
    terminal.addLog("ERROR", err.message || "Gagal mencatat absensi");
    toast.error("Gagal mencatat kehadiran. Coba lagi.");
  } finally {
    isProcessing = false;
    scanner.resume();
    enableStartButton();
  }
}

// ── Render attendance entry in list ───────────────────────
function addAttendanceEntry(user, result) {
  if (!attendanceList) return;

  // Remove empty-state placeholder on first real entry
  const placeholder = attendanceList.querySelector("p");
  if (placeholder) placeholder.remove();

  const time = result?.timestamp
    ? new Date(result.timestamp).toLocaleTimeString("id-ID")
    : new Date().toLocaleTimeString("id-ID");

  const entry = document.createElement("div");
  entry.className = "attendance-entry";
  entry.innerHTML = `
    <span class="attendance-mode attendance-mode--${currentMode}">
      ${currentMode === "checkin" ? "IN" : "OUT"}
    </span>
    <span class="attendance-name">${escHtml(user.name)}</span>
    <span class="attendance-time">${time}</span>
  `;
  // Newest entry at top
  attendanceList.insertBefore(entry, attendanceList.firstChild);
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

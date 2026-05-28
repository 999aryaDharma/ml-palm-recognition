// ============================================================
// js/pages/attendance.js — Palm Attendance demo logic
// Full implementation: check-in / check-out mode toggle
// ============================================================

import { mountNavbar } from "../components/navbar.js";
import { PalmScanner } from "../components/palm-scanner.js";
import { DiagnosticTerminal } from "../components/terminal.js";
import { showModal } from "../components/modal.js";
import { toast } from "../components/toast.js";
import { formatTime, formatDate, logTimestamp } from "../utils.js";

// ── State ──────────────────────────────────────────────────
let scanner = null;
let terminal = null;
let currentMode = "checkin"; // "checkin" | "checkout"
let isProcessing = false;

// ── DOM ────────────────────────────────────────────────────
const scannerContainer = document.getElementById("scanner-container");
const scannerVideo = document.getElementById("scanner-video");
const scannerHint = document.getElementById("scanner-hint");
const scannerResult = document.getElementById("scanner-result");
const scannerPlaceholder = document.getElementById("scanner-placeholder");
const terminalBody = document.getElementById("terminal-body");
const attendanceList = document.getElementById("attendance-list");
const btnCheckin = document.getElementById("btn-mode-checkin");
const btnCheckout = document.getElementById("btn-mode-checkout");
const btnStartScan = document.getElementById("btn-start-scan");
const modeLabel = document.getElementById("current-mode-label");

// ── Init ───────────────────────────────────────────────────
function init() {
  mountNavbar();

  terminal = new DiagnosticTerminal(terminalBody);
  terminal.addLog("CAMERA_READY", "Attendance system initialized.");

  // Mode toggle
  btnCheckin?.addEventListener("click", () => setMode("checkin"));
  btnCheckout?.addEventListener("click", () => setMode("checkout"));

  // Scanner init
  scanner = new PalmScanner({
    containerEl: scannerContainer,
    videoEl: scannerVideo,
    hintEl: scannerHint,
    resultEl: scannerResult,
    placeholderEl: scannerPlaceholder,
    logFn: (tag, msg) => terminal.addLog(tag, msg),
    onIdentified: handleIdentified,
    onUnknown: (score) => {
      toast.warning("Pengguna tidak dikenali. Coba scan ulang.");
      terminal.addLog("RESULT", `UNKNOWN — score ${score.toFixed(4)}`);
    },
    autoResetMs: 4000,
  });

  btnStartScan?.addEventListener("click", startScanner);

  // Cleanup
  window.addEventListener("beforeunload", () => scanner?.stop());
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) scanner?.stop();
  });
}

function setMode(mode) {
  currentMode = mode;
  if (modeLabel)
    modeLabel.textContent = mode === "checkin" ? "Check-in" : "Check-out";

  btnCheckin?.classList.toggle("btn--primary", mode === "checkin");
  btnCheckin?.classList.toggle("btn--secondary", mode !== "checkin");
  btnCheckout?.classList.toggle("btn--primary", mode === "checkout");
  btnCheckout?.classList.toggle("btn--secondary", mode !== "checkout");

  terminal.addLog("SYSTEM", `Mode switched to ${mode.toUpperCase()}`);
}

async function startScanner() {
  if (btnStartScan) {
    btnStartScan.disabled = true;
    btnStartScan.innerHTML = `<span class="spinner spinner--sm"></span> Scanning…`;
  }
  terminal.addLog("SYSTEM", "Attendance scanner started.");
  await scanner.start();
}

// ── Handle identification ──────────────────────────────────
async function handleIdentified(user, score, latency) {
  if (isProcessing) return;
  isProcessing = true;

  terminal.addLog(
    "MATCHING",
    `${user.name} score ${score.toFixed(4)} latency ${latency}ms`,
  );

  const modeLabel = currentMode === "checkin" ? "Check-in" : "Check-out";
  const modeIcon = currentMode === "checkin" ? "📋" : "🚪";

  showModal({
    title: `Konfirmasi ${modeLabel}`,
    icon: modeIcon,
    message: `
      <div class="flex flex-col gap-4">
        <p>Catat <strong>${modeLabel}</strong> untuk:</p>
        <div style="display:flex;align-items:center;gap:12px;padding:12px;background:var(--color-surface-warm);border-radius:12px;border:1px solid var(--color-border)">
          <div style="width:40px;height:40px;border-radius:8px;background:var(--color-coffee);display:flex;align-items:center;justify-content:center;color:white;font-weight:700">${user.name[0]}</div>
          <div>
            <div style="font-weight:600">${escHtml(user.name)}</div>
            <div style="font-size:12px;color:var(--color-coffee-light)">Score: ${(score * 100).toFixed(1)}% · ${latency}ms</div>
          </div>
        </div>
        <p class="text-sm">Waktu: <strong>${new Date().toLocaleTimeString("id-ID")}</strong></p>
      </div>
    `,
    confirmLabel: `Konfirmasi ${modeLabel}`,
    confirmVariant: currentMode === "checkin" ? "success" : "primary",
    onConfirm: () => submitAttendance(user, score),
    onCancel: () => {
      isProcessing = false;
      scanner.resume();
    },
  });
}

async function submitAttendance(user, score) {
  try {
    const { apiFetch } = await import("../api/client.js");
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
    );
    addAttendanceEntry(user, result);
  } catch (err) {
    terminal.addLog("ERROR", err.message);
    toast.error("Gagal mencatat kehadiran. Coba lagi.");
  } finally {
    isProcessing = false;
    scanner.resume();
    resetStartButton();
  }
}

function addAttendanceEntry(user, result) {
  // Remove empty state if present
  const empty = attendanceList?.querySelector(".empty-state");
  if (empty) empty.remove();

  const entry = document.createElement("div");
  entry.className = "attendance-entry";
  entry.innerHTML = `
    <span class="attendance-mode attendance-mode--${currentMode}">${currentMode === "checkin" ? "IN" : "OUT"}</span>
    <span class="attendance-name">${escHtml(user.name)}</span>
    <span class="attendance-time">${new Date().toLocaleTimeString("id-ID")}</span>
  `;
  attendanceList?.insertBefore(entry, attendanceList.firstChild);
}

function resetStartButton() {
  if (btnStartScan) {
    btnStartScan.disabled = false;
    btnStartScan.textContent = "Mulai Scan";
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

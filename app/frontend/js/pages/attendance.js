// ============================================================
// js/pages/attendance.js — Palm Attendance demo logic
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
let scannerStarted = false;
let countCheckin = 0;
let countCheckout = 0;

// ── DOM ────────────────────────────────────────────────────
let btnCheckin, btnCheckout, btnStartScan, btnStopScan;
let terminalCard, scannerContainer, attendanceList;
let modeLabelEl, countCheckinEl, countCheckoutEl;

// ── Init ───────────────────────────────────────────────────
async function init() {
  mountNavbar();

  // Retrieve elements inside init
  btnCheckin = document.getElementById("btn-mode-checkin");
  btnCheckout = document.getElementById("btn-mode-checkout");
  btnStartScan = document.getElementById("btn-start-scan");
  btnStopScan = document.getElementById("btn-stop-scan");
  terminalCard = document.querySelector(".terminal-card");
  scannerContainer = document.getElementById("scanner-container");
  attendanceList = document.getElementById("attendance-list");
  modeLabelEl = document.getElementById("current-mode-label");
  countCheckinEl = document.getElementById("count-checkin");
  countCheckoutEl = document.getElementById("count-checkout");

  if (!btnStartScan || !terminalCard || !scannerContainer) {
    console.error("[Attendance] Required DOM elements not found!");
    return;
  }

  terminal = new DiagnosticTerminal(terminalCard);

  // Listen for terminal events (like copy log)
  document.addEventListener("toast", (e) => {
    if (e.detail && toast[e.detail.type]) {
      toast[e.detail.type](e.detail.message);
    }
  });

  terminal.addLog("SYSTEM", "Attendance system initialized.");
  terminal.addLog(
    "INSTRUCTION",
    "Langkah 1: Pilih mode Check-in atau Check-out.",
  );
  terminal.addLog("INSTRUCTION", "Langkah 2: Klik 'Mulai Scan Absensi'.");

  // Load from localStorage
  loadAttendanceFromStorage();

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
    boundingBoxLayerEl: document.getElementById("palm-box-layer"),
    logFn: (tag, msg) => terminal?.addLog(tag, msg),
    onIdentified: handleIdentified,
    onUnknown: (score) => {
      terminal?.addLog(
        "RESULT",
        `ATTENDANCE UNKNOWN — score ${score.toFixed(4)}`,
      );
    },
    autoResetMs: 4000,
    captureIntervalMs: 1500,
    autoResumeOnIdentified: false,
    pauseOnUnknown: true,
  });

  btnStartScan?.addEventListener("click", startScanner);

  btnStopScan?.addEventListener("click", () => {
    scanner?.stop();
    scannerStarted = false;
    if (btnStopScan) btnStopScan.hidden = true;
    enableStartButton();
    terminal.addLog("SYSTEM", "Attendance scanner stopped by user.");
  });

  window.addEventListener("beforeunload", () => scanner?.stop());
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) scanner?.stop();
  });
}

function loadAttendanceFromStorage() {
  const data = localStorage.getItem("palmid_attendance_logs");
  if (!data) return;

  try {
    const logs = JSON.parse(data);
    logs.forEach((entry) => {
      addAttendanceEntryToDOM(entry.user, entry.mode, entry.time);
      if (entry.mode === "checkin") countCheckin++;
      else countCheckout++;
    });
    updateCounters();
  } catch (err) {
    console.error("Failed to load attendance logs:", err);
  }
}

function saveAttendanceToStorage(user, mode, time) {
  const data = localStorage.getItem("palmid_attendance_logs");
  let logs = [];
  if (data) {
    try {
      logs = JSON.parse(data);
    } catch (e) {}
  }
  logs.unshift({ user, mode, time });
  // Keep last 20
  if (logs.length > 20) logs = logs.slice(0, 20);
  localStorage.setItem("palmid_attendance_logs", JSON.stringify(logs));
}

function updateCounters() {
  if (countCheckinEl) countCheckinEl.textContent = String(countCheckin);
  if (countCheckoutEl) countCheckoutEl.textContent = String(countCheckout);
}

// ── Mode toggle ────────────────────────────────────────────
function setMode(mode) {
  currentMode = mode;

  // Accessibility: aria-pressed for semantic state
  btnCheckin?.setAttribute("aria-pressed", String(mode === "checkin"));
  btnCheckout?.setAttribute("aria-pressed", String(mode === "checkout"));

  if (modeLabelEl)
    modeLabelEl.textContent =
      mode === "checkin" ? "Check-in aktif" : "Check-out aktif";

  btnCheckin?.classList.toggle("btn--primary", mode === "checkin");
  btnCheckin?.classList.toggle("btn--secondary", mode !== "checkin");
  btnCheckout?.classList.toggle("btn--primary", mode === "checkout");
  btnCheckout?.classList.toggle("btn--secondary", mode !== "checkout");

  terminal.addLog("SYSTEM", `Mode: ${mode.toUpperCase()} aktif.`);
}

// ── Scanner ────────────────────────────────────────────────
async function startScanner() {
  if (scannerStarted) return;
  scannerStarted = true;

  if (btnStartScan) {
    btnStartScan.disabled = true;
    btnStartScan.innerHTML = `<span class="spinner spinner--sm"></span> Scanning…`;
  }
  terminal.addLog("SYSTEM", `Attendance scanner (${currentMode}) started.`);

  const ok = await scanner.start();
  if (!ok) {
    scannerStarted = false;
    enableStartButton("Coba Aktifkan Scanner Lagi");
    return;
  }

  if (btnStopScan) btnStopScan.hidden = false;
}

function enableStartButton(label = "🖐 Mulai Scan Absensi") {
  if (btnStartScan) {
    btnStartScan.disabled = false;
    btnStartScan.textContent = label;
  }
}

// ── Identification callback ────────────────────────────────
async function handleIdentified(user, score, latency) {
  if (isProcessing) return;

  terminal.addLog(
    "MATCHING",
    `${user.name} — score ${score.toFixed(4)} — ${latency}ms`,
  );

  // P0 UX FIX: Explicitly pause scanner during modal
  scanner.pause();

  const modeText = currentMode === "checkin" ? "Check-in" : "Check-out";
  const modeIcon = currentMode === "checkin" ? "clipboard" : "doorOpen";

  showModal({
    title: `Konfirmasi ${modeText}`,
    icon: modeIcon,
    message: `
      <div class="flex flex-col gap-4">
        <p>Catat <strong>${modeText}</strong> untuk:</p>
        <div class="attendance-user-summary" style="display:flex;align-items:center;gap:12px;padding:12px;background:var(--color-surface-warm);border-radius:12px;border:1px solid var(--color-border)">
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
  isProcessing = true;
  try {
    const result = await apiFetch("/demos/attendance/checkin", {
      method: "POST",
      body: JSON.stringify({
        user_id: user.id,
        mode: currentMode,
        match_score: score,
      }),
    });

    const timeStr = result?.timestamp
      ? new Date(result.timestamp).toLocaleTimeString("id-ID")
      : new Date().toLocaleTimeString("id-ID");

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
    } else {
      countCheckout++;
    }
    updateCounters();

    addAttendanceEntryToDOM(user, currentMode, timeStr);
    saveAttendanceToStorage(user, currentMode, timeStr);
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
function addAttendanceEntryToDOM(user, mode, time) {
  if (!attendanceList) return;

  // Remove empty-state placeholder on first real entry
  const placeholder = document.getElementById("attendance-empty");
  if (placeholder) placeholder.remove();

  const entry = document.createElement("div");
  entry.className = "attendance-entry";
  entry.innerHTML = `
    <span class="attendance-mode attendance-mode--${mode}">
      ${mode === "checkin" ? "IN" : "OUT"}
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

// ============================================================
// js/pages/patient.js — Palm Patient Check-in demo logic
// ============================================================

import { mountNavbar } from "../components/navbar.js";
import { PalmScanner } from "../components/palm-scanner.js";
import { patientCheckin } from "../api/demos.js";
import { showModal } from "../components/modal.js";
import { toast } from "../components/toast.js";
import { logTimestamp } from "../utils.js";

// ── State ──────────────────────────────────────────────────
let scanner = null;
let isCheckingIn = false;
const HOSPITAL_ID = "rumahsakit-01";

// ── DOM Elements ───────────────────────────────────────────
const scannerContainer = document.getElementById("scanner-container");
const btnStartPatient = document.getElementById("btn-start-patient");
const terminalBody = document.getElementById("terminal-body");
const checkoutPanel = document.getElementById("checkout-panel");
const receiptPanel = document.getElementById("receipt-panel");

// ── Initialization ─────────────────────────────────────────
function init() {
  mountNavbar();

  // Setup PalmScanner
  scanner = new PalmScanner({
    containerEl: scannerContainer,
    videoEl: document.getElementById("scanner-video"),
    hintEl: document.getElementById("scanner-hint"),
    resultEl: document.getElementById("scanner-result"),
    placeholderEl: document.getElementById("scanner-placeholder"),
    logFn: addLog,
    onIdentified: handleIdentified,
    onUnknown: (score) => {
      toast.warning(
        "Pasien tidak dikenali. Silakan hubungi resepsionis.",
        "Gagal",
      );
    },
  });

  btnStartPatient.addEventListener("click", () => {
    scanner.start();
    btnStartPatient.disabled = true;
    btnStartPatient.innerHTML = `<span class="spinner spinner--sm"></span> Menunggu Scan...`;
    addLog(
      "SYSTEM",
      "Patient check-in mode activated. Waiting for palm scan...",
    );
  });

  // Cleanup on page leave
  window.addEventListener("beforeunload", () => {
    if (scanner) scanner.stop();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && scanner) scanner.stop();
  });

  addLog("SYSTEM", "Ready for patient check-in.");
}

/**
 * Handle successful biometric identification
 */
async function handleIdentified(user, score, latency) {
  if (isCheckingIn) return;

  addLog(
    "PATIENT",
    `Patient ${user.name} identified with score ${score.toFixed(4)}`,
  );

  showModal({
    title: "Konfirmasi Check-in Pasien",
    message: `
      <div class="flex flex-col gap-4">
        <p>Daftar check-in untuk pasien:</p>
        <div class="surface-card-warm flex items-center gap-4" style="padding: var(--space-4)">
          <div class="user-avatar" style="width: 40px; height: 40px; font-size: 1rem">${user.name[0]}</div>
          <div>
            <div style="font-weight: 600">${user.name}</div>
            <div class="text-xs text-muted">ID: #${user.id} · Match: ${(score * 100).toFixed(1)}%</div>
          </div>
        </div>
        <p class="text-sm">Waktu check-in: <strong>${new Date().toLocaleTimeString("id-ID")}</strong></p>
      </div>
    `,
    icon: "🏥",
    confirmLabel: "Konfirmasi Check-in",
    confirmVariant: "success",
    onConfirm: () => processCheckin(user),
    onCancel: () => {
      addLog("SYSTEM", "Patient check-in cancelled.");
      scanner.resume();
    },
  });
}

/**
 * Finalize patient check-in with API call
 */
async function processCheckin(user) {
  isCheckingIn = true;
  addLog("API_POST", `/demos/patient/checkin (user_id: ${user.id})`);

  try {
    const result = await patientCheckin(user.id);

    addLog("RESULT", "Patient check-in recorded successfully.");

    // UI Transition
    checkoutPanel.classList.add("hidden");
    showReceipt(user, result);
    toast.success("Check-in pasien berhasil dicatat.", "Sukses");

    // Stop camera
    scanner.stop();
  } catch (err) {
    addLog("ERROR", err.message || "Failed to record check-in");
    toast.error("Gagal mencatat check-in. Silakan coba lagi.");
    scanner.resume();
  } finally {
    isCheckingIn = false;
  }
}

/**
 * Display patient check-in receipt
 */
function showReceipt(user, result) {
  receiptPanel.classList.remove("hidden");

  const now = new Date();
  document.getElementById("receipt-txn-id").textContent =
    "PAT-" + Date.now().toString().slice(-6);
  document.getElementById("receipt-date").textContent =
    now.toLocaleString("id-ID");
  document.getElementById("receipt-user").textContent = user.name;
  document.getElementById("receipt-amount").textContent =
    "Check-in " + now.toLocaleTimeString("id-ID");

  document.getElementById("btn-finish").onclick = () => {
    window.location.reload();
  };
}

/**
 * Utility to add a log line to the terminal
 */
function addLog(tag, msg) {
  const line = document.createElement("div");
  line.className = "log-line";
  line.innerHTML = `
    <span class="log-time">[${logTimestamp()}]</span>
    <span class="log-tag log-tag--${mapTagToClass(tag)}">${tag}</span>
    <span class="log-msg">${msg}</span>
  `;
  terminalBody.appendChild(line);
  terminalBody.scrollTop = terminalBody.scrollHeight;
}

/**
 * Map tag to CSS class name
 */
function mapTagToClass(tag) {
  const mapping = {
    SYSTEM: "camera",
    PATIENT: "detect",
    API_POST: "match",
    RESULT: "result",
    ERROR: "error",
  };
  return mapping[tag] || "camera";
}

// ── Run ────────────────────────────────────────────────────
init();

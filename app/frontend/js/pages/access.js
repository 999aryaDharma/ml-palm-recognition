// ============================================================
// js/pages/access.js — Palm Access Control demo logic
// ============================================================

import { mountNavbar } from "../components/navbar.js";
import { PalmScanner } from "../components/palm-scanner.js";
import { accessCheck } from "../api/demos.js";
import { showModal } from "../components/modal.js";
import { toast } from "../components/toast.js";
import { logTimestamp } from "../utils.js";

// ── State ──────────────────────────────────────────────────
let scanner = null;
let isProcessingAccess = false;
const DOOR_ID = "door-01-secured";

// ── DOM Elements ───────────────────────────────────────────
const scannerContainer = document.getElementById("scanner-container");
const btnStartAccess = document.getElementById("btn-start-access");
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
      toast.warning("Akses ditolak. Pengguna tidak dikenali.", "Ditolak");
    },
  });

  btnStartAccess.addEventListener("click", () => {
    scanner.start();
    btnStartAccess.disabled = true;
    btnStartAccess.innerHTML = `<span class="spinner spinner--sm"></span> Menunggu Scan...`;
    addLog("SYSTEM", "Access control mode activated. Waiting for palm scan...");
  });

  // Cleanup on page leave
  window.addEventListener("beforeunload", () => {
    if (scanner) scanner.stop();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && scanner) scanner.stop();
  });

  addLog("SYSTEM", "Ready for access control.");
}

/**
 * Handle successful biometric identification
 */
async function handleIdentified(user, score, latency) {
  if (isProcessingAccess) return;

  addLog(
    "ACCESS",
    `User ${user.name} identified with score ${score.toFixed(4)}`,
  );

  showModal({
    title: "Konfirmasi Akses",
    message: `
      <div class="flex flex-col gap-4">
        <p>Memberikan akses ke pintu: <strong>${DOOR_ID.toUpperCase()}</strong></p>
        <div class="surface-card-warm flex items-center gap-4" style="padding: var(--space-4)">
          <div class="user-avatar" style="width: 40px; height: 40px; font-size: 1rem">${user.name[0]}</div>
          <div>
            <div style="font-weight: 600">${user.name}</div>
            <div class="text-xs text-muted">ID: #${user.id} · Match: ${(score * 100).toFixed(1)}%</div>
          </div>
        </div>
      </div>
    `,
    icon: "🔓",
    confirmLabel: "Buka Akses",
    confirmVariant: "success",
    onConfirm: () => processAccess(user),
    onCancel: () => {
      addLog("SYSTEM", "Access request cancelled.");
      scanner.resume();
    },
  });
}

/**
 * Finalize access check with API call
 */
async function processAccess(user) {
  isProcessingAccess = true;
  addLog("API_POST", `/demos/access/check (user_id: ${user.id})`);

  try {
    const result = await accessCheck(user.id, DOOR_ID);

    addLog("RESULT", "Access GRANTED to " + DOOR_ID);

    // UI Transition
    checkoutPanel.classList.add("hidden");
    showReceipt(user, result);
    toast.success("Akses diberikan.", "Sukses");

    // Stop camera
    scanner.stop();
  } catch (err) {
    addLog("ERROR", err.message || "Failed to grant access");
    toast.error("Gagal memberikan akses. Silakan coba lagi.");
    scanner.resume();
  } finally {
    isProcessingAccess = false;
  }
}

/**
 * Display access receipt
 */
function showReceipt(user, result) {
  receiptPanel.classList.remove("hidden");

  const now = new Date();
  document.getElementById("receipt-txn-id").textContent =
    "ACC-" + Date.now().toString().slice(-6);
  document.getElementById("receipt-date").textContent =
    now.toLocaleString("id-ID");
  document.getElementById("receipt-user").textContent = user.name;
  document.getElementById("receipt-amount").textContent = DOOR_ID.toUpperCase();

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
    ACCESS: "detect",
    API_POST: "match",
    RESULT: "result",
    ERROR: "error",
  };
  return mapping[tag] || "camera";
}

// ── Run ────────────────────────────────────────────────────
init();

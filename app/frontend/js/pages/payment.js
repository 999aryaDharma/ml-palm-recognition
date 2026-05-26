// ============================================================
// js/pages/payment.js — Palm Payment demo logic
// ============================================================

import { mountNavbar } from "../components/navbar.js";
import { PalmScanner } from "../components/palm-scanner.js";
import { paymentPay } from "../api/demos.js";
import { showModal } from "../components/modal.js";
import { toast } from "../components/toast.js";
import { formatRupiah, logTimestamp, generateTxnId, sleep } from "../utils.js";

// ── Utilities ──────────────────────────────────────────────
function escapeHtml(text) {
  const map = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  };
  return String(text).replace(/[&<>"']/g, (char) => map[char]);
}

// ── State ──────────────────────────────────────────────────
let scanner = null;
const MERCHANT_NAME = "Toko Maju Jaya";
const ORDER_TOTAL = 127500;
let isPaymentProcessing = false;

// ── DOM Elements ───────────────────────────────────────────
const scannerContainer = document.getElementById("scanner-container");
const btnStartPayment = document.getElementById("btn-start-payment");
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
      toast.warning("Pengguna tidak dikenali. Silakan hubungi kasir.", "Gagal");
    },
  });

  btnStartPayment.addEventListener("click", () => {
    scanner.start();
    btnStartPayment.disabled = true;
    btnStartPayment.innerHTML = `<span class="spinner spinner--sm"></span> Menunggu Scan...`;
    addLog("SYSTEM", "Payment mode activated. Waiting for palm scan...");
  });

  // Cleanup on page leave
  window.addEventListener("beforeunload", () => {
    if (scanner) scanner.stop();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && scanner) scanner.stop();
  });

  addLog("SYSTEM", "Ready to process payment.");
}

/**
 * Handle successful biometric identification
 */
async function handleIdentified(user, score, latency) {
  if (isPaymentProcessing) return;

  addLog("PAYMENT", `User ${user.name} matched with score ${score.toFixed(4)}`);

  showModal({
    title: "Konfirmasi Pembayaran",
    message: `
      <div class="flex flex-col gap-4">
        <p>Bayar pesanan sebesar <strong>${formatRupiah(ORDER_TOTAL)}</strong> atas nama:</p>
        <div class="surface-card-warm flex items-center gap-4" style="padding: var(--space-4)">
          <div class="user-avatar" style="width: 40px; height: 40px; font-size: 1rem">${escapeHtml(user.name[0])}</div>
          <div>
            <div style="font-weight: 600">${escapeHtml(user.name)}</div>
            <div class="text-xs text-muted">ID: #${user.id} · Match: ${(score * 100).toFixed(1)}%</div>
          </div>
        </div>
      </div>
    `,
    icon: "💳",
    confirmLabel: "Konfirmasi & Bayar",
    confirmVariant: "success",
    onConfirm: () => processPayment(user),
    onCancel: () => {
      addLog("SYSTEM", "Payment cancelled by user.");
      scanner.resume();
    },
  });
}

/**
 * Finalize payment with API call
 */
async function processPayment(user) {
  isPaymentProcessing = true;
  addLog("API_POST", `/demos/payment/pay (user_id: ${user.id})`);

  try {
    const result = await paymentPay(user.id, ORDER_TOTAL, MERCHANT_NAME);

    addLog(
      "RESULT",
      "Payment SUCCESS. Transaction ID: " +
        (result.transaction_id || "TXN-" + Date.now()),
    );

    // UI Transition
    checkoutPanel.classList.add("hidden");
    showReceipt(user, result);
    toast.success("Pembayaran berhasil diproses.", "Sukses");

    // Stop camera as we are in receipt view
    scanner.stop();
  } catch (err) {
    addLog("ERROR", err.message || "Failed to process payment");
    toast.error("Gagal memproses pembayaran. Silakan coba lagi.");
    scanner.resume();
  } finally {
    isPaymentProcessing = false;
  }
}

/**
 * Display digital receipt
 */
function showReceipt(user, result) {
  receiptPanel.classList.remove("hidden");

  document.getElementById("receipt-txn-id").textContent =
    result.transaction_id || generateTxnId();
  document.getElementById("receipt-date").textContent =
    new Date().toLocaleString("id-ID");
  document.getElementById("receipt-user").textContent = user.name;
  document.getElementById("receipt-amount").textContent =
    formatRupiah(ORDER_TOTAL);

  document.getElementById("btn-finish").onclick = () => {
    window.location.reload(); // Simplest reset for demo
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
    PAYMENT: "detect",
    API_POST: "match",
    RESULT: "result",
    ERROR: "error",
  };
  return mapping[tag] || "camera";
}

// ── Run ────────────────────────────────────────────────────
init();

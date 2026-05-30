// ============================================================
// js/pages/payment.js — Palm Payment demo logic
// Fixed: scanner start on button click, btn state, terminal clear
// ============================================================

import { mountNavbar } from "../components/navbar.js";
import { PalmScanner } from "../components/palm-scanner.js";
import { DiagnosticTerminal } from "../components/terminal.js";
import { paymentPay } from "../api/demos.js";
import { showModal } from "../components/modal.js";
import { toast } from "../components/toast.js";
import { formatRupiah, logTimestamp, generateTxnId } from "../utils.js";

// ── Utilities ──────────────────────────────────────────────
function escapeHtml(text) {
  const map = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  };
  return String(text).replace(/[&<>"']/g, (c) => map[c]);
}

// ── Constants ──────────────────────────────────────────────
const MERCHANT_NAME = "Toko Maju Jaya";
const ORDER_TOTAL = 127500;

// ── State ──────────────────────────────────────────────────
let scanner = null;
let terminal = null;
let isPaymentProcessing = false;
let scannerStarted = false;

// ── DOM ────────────────────────────────────────────────────
const scannerContainer = document.getElementById("scanner-container");
const btnStartPayment = document.getElementById("btn-start-payment");
const terminalBody = document.getElementById("terminal-body");
const checkoutPanel = document.getElementById("checkout-panel");
const receiptPanel = document.getElementById("receipt-panel");
const btnClearLog = document.getElementById("btn-clear-log");

// ── Initialization ─────────────────────────────────────────
function init() {
  mountNavbar();

  terminal = new DiagnosticTerminal(terminalBody);
  terminal.addLog("SYSTEM", "Palm Payment module loaded.");
  terminal.addLog("SYSTEM", "Klik 'Bayar dengan Telapak' untuk mulai.");

  // Build PalmScanner — note: scanner is created once, start() called on button click
  scanner = new PalmScanner({
    containerEl: scannerContainer,
    videoEl: document.getElementById("scanner-video"),
    hintEl: document.getElementById("scanner-hint"),
    resultEl: document.getElementById("scanner-result"),
    placeholderEl: document.getElementById("scanner-placeholder"),
    logFn: (tag, msg) => terminal.addLog(tag, msg),
    onIdentified: handleIdentified,
    onUnknown: (score) => {
      toast.warning("Pengguna tidak dikenali. Silakan hubungi kasir.", "Gagal");
    },
    captureIntervalMs: 1500,
    autoResetMs: 4000,
  });

  btnStartPayment.addEventListener("click", async () => {
    if (scannerStarted) return; // prevent double-click
    scannerStarted = true;
    btnStartPayment.disabled = true;
    btnStartPayment.innerHTML = `<span class="spinner spinner--sm"></span> Menunggu Scan...`;
    terminal.addLog(
      "SYSTEM",
      "Payment mode activated. Waiting for palm scan...",
    );
    await scanner.start();
  });

  btnClearLog?.addEventListener("click", () => terminal.clear());

  // Cleanup on page leave
  window.addEventListener("beforeunload", () => scanner?.stop());
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) scanner?.stop();
  });

  terminal.addLog("SYSTEM", "Ready to process payment.");
}

// ── Handle identification ──────────────────────────────────
async function handleIdentified(user, score, latency) {
  if (isPaymentProcessing) return;

  terminal.addLog(
    "PAYMENT",
    `User ${escapeHtml(user.name)} matched — score ${score.toFixed(4)}`,
  );

  showModal({
    title: "Konfirmasi Pembayaran",
    message: `
      <div class="flex flex-col gap-4">
        <p>Bayar pesanan sebesar <strong>${formatRupiah(ORDER_TOTAL)}</strong> atas nama:</p>
        <div style="display:flex;align-items:center;gap:12px;padding:12px;background:var(--color-surface-warm);border-radius:12px;border:1px solid var(--color-border)">
          <div style="width:40px;height:40px;border-radius:8px;background:var(--color-coffee);display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:1.1rem">
            ${escapeHtml(user.name[0])}
          </div>
          <div>
            <div style="font-weight:600">${escapeHtml(user.name)}</div>
            <div class="text-xs" style="color:var(--color-coffee-light)">ID: #${user.id} · Match: ${(score * 100).toFixed(1)}% · ${latency}ms</div>
          </div>
        </div>
      </div>
    `,
    icon: "💳",
    confirmLabel: "Konfirmasi & Bayar",
    confirmVariant: "success",
    onConfirm: () => processPayment(user, score),
    onCancel: () => {
      terminal.addLog("SYSTEM", "Payment cancelled by user.");
      scanner.resume();
    },
  });
}

// ── Process payment ────────────────────────────────────────
async function processPayment(user, score) {
  isPaymentProcessing = true;
  terminal.addLog(
    "API_POST",
    `/demos/payment/pay (user_id: ${user.id}, amount: ${ORDER_TOTAL})`,
  );

  try {
    const result = await paymentPay(user.id, ORDER_TOTAL, MERCHANT_NAME);

    terminal.addLog(
      "RESULT",
      "Payment SUCCESS — Transaction ID: " +
        (result.transaction_id || "TXN-" + Date.now()),
    );

    checkoutPanel.classList.add("hidden");
    showReceipt(user, result);
    toast.success("Pembayaran berhasil diproses.", "Sukses");

    scanner.stop(); // release camera — we're on receipt screen now
  } catch (err) {
    terminal.addLog("ERROR", err.message || "Failed to process payment");
    toast.error("Gagal memproses pembayaran. Silakan coba lagi.");
    scanner.resume();
    isPaymentProcessing = false;
  }
}

// ── Receipt ────────────────────────────────────────────────
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
    window.location.reload();
  };
}

// ── Run ────────────────────────────────────────────────────
init();

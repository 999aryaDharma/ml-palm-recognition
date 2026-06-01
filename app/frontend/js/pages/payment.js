// ============================================================
// js/pages/payment.js — Palm Payment demo logic
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
let btnStartPayment, btnStopPayment, terminalCard, scannerContainer;
let checkoutPanel, receiptPanel;

// ── Initialization ─────────────────────────────────────────
async function init() {
  mountNavbar();

  // Retrieve elements inside init
  btnStartPayment = document.getElementById("btn-start-payment");
  btnStopPayment = document.getElementById("btn-stop-payment");
  terminalCard = document.querySelector(".terminal-card");
  scannerContainer = document.getElementById("scanner-container");
  checkoutPanel = document.getElementById("checkout-panel");
  receiptPanel = document.getElementById("receipt-panel");

  if (!btnStartPayment || !terminalCard || !scannerContainer) {
    console.error("[Payment] Required DOM elements not found!");
    return;
  }

  terminal = new DiagnosticTerminal(terminalCard);

  // Listen for terminal events (like copy log)
  document.addEventListener("toast", (e) => {
    if (e.detail && toast[e.detail.type]) {
      toast[e.detail.type](e.detail.message);
    }
  });

  terminal.addLog("SYSTEM", "Palm Payment module loaded.");
  terminal.addLog(
    "INSTRUCTION",
    "Langkah 1: Klik tombol 'Bayar dengan Telapak'.",
  );

  // Build PalmScanner
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
      terminal?.addLog("RESULT", `PAYMENT UNKNOWN — score ${score.toFixed(4)}`);
    },
    captureIntervalMs: 1500,
    autoResetMs: 4000,
    autoResumeOnIdentified: false,
    pauseOnUnknown: true,
  });

  btnStartPayment.addEventListener("click", async () => {
    if (scannerStarted) return;

    try {
      scannerStarted = true;
      setPaymentButtonLoading();

      terminal.addLog(
        "SYSTEM",
        "Payment mode activated. Waiting for palm scan...",
      );
      terminal.addLog(
        "INSTRUCTION",
        "Langkah 2: Hadapkan telapak tangan ke kamera.",
      );

      const ok = await scanner.start();
      if (!ok) {
        throw new Error("Failed to start scanner");
      }

      if (btnStopPayment) btnStopPayment.hidden = false;
    } catch (err) {
      console.error("[Payment] Error starting scanner:", err);
      scannerStarted = false;
      resetPaymentButton("Coba Aktifkan Scanner Lagi");
      terminal.addLog("ERROR", err.message || "Failed to start camera");
    }
  });

  btnStopPayment?.addEventListener("click", () => {
    scanner?.stop();
    scannerStarted = false;
    if (btnStopPayment) btnStopPayment.hidden = true;
    resetPaymentButton();
    terminal.addLog("SYSTEM", "Payment scanner stopped by user.");
  });

  // Cleanup on page leave
  window.addEventListener("beforeunload", () => scanner?.stop());
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) scanner?.stop();
  });

  terminal.addLog("SYSTEM", "Ready to process payment.");
}

function setPaymentButtonLoading() {
  btnStartPayment.disabled = true;
  btnStartPayment.innerHTML = `<span class="spinner spinner--sm"></span> Menunggu Scan...`;
}

function resetPaymentButton(label = "Bayar dengan Telapak") {
  btnStartPayment.disabled = false;
  btnStartPayment.innerHTML = `💳 ${label}`;
}

// ── Handle identification ──────────────────────────────────
async function handleIdentified(user, score, latency) {
  if (isPaymentProcessing) return;

  terminal.addLog(
    "PAYMENT",
    `User ${escapeHtml(user.name)} matched — score ${score.toFixed(4)}`,
  );

  // P0 UX FIX: Explicitly pause scanner before modal
  scanner.pause();

  showModal({
    title: "Konfirmasi Pembayaran",
    message: `
      <div class="flex flex-col gap-4">
        <p>Bayar pesanan sebesar <strong>${formatRupiah(ORDER_TOTAL)}</strong> atas nama:</p>
        <div class="payment-user-summary" style="display:flex;align-items:center;gap:12px;padding:12px;background:var(--color-surface-warm);border-radius:12px;border:1px solid var(--color-border)">
          <div class="payment-user-avatar" style="width:40px;height:40px;border-radius:8px;background:var(--color-coffee);display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:1.1rem">
            ${escapeHtml(user.name[0])}
          </div>
          <div>
            <div style="font-weight:600">${escapeHtml(user.name)}</div>
            <div class="text-xs" style="color:var(--color-coffee-light)">ID: #${user.id} · Match: ${(score * 100).toFixed(1)}% · ${latency}ms</div>
          </div>
        </div>
      </div>
    `,
    icon: "creditCard", // Use SVG icon key
    initialFocus: "cancel", // P1 UX FIX: Safe focus for financial action
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

    // Stop and hide scanner
    scanner?.stop();
    if (scannerContainer) scannerContainer.style.display = "none";
    if (btnStartPayment) btnStartPayment.style.display = "none";
    if (btnStopPayment) btnStopPayment.style.display = "none";

    checkoutPanel.classList.add("hidden");
    showReceipt(user, result, score);
    toast.success("Pembayaran berhasil diproses.", "Sukses");
  } catch (err) {
    terminal.addLog("ERROR", err.message || "Failed to process payment");
    toast.error("Gagal memproses pembayaran. Silakan coba lagi.");
    scanner.resume();
    isPaymentProcessing = false;
  }
}

// ── Receipt ────────────────────────────────────────────────
function showReceipt(user, result, score) {
  receiptPanel.classList.remove("hidden");

  const txnId = result.transaction_id || generateTxnId();
  const dateStr = new Date().toLocaleString("id-ID");

  document.getElementById("receipt-txn-id").textContent = txnId;
  document.getElementById("receipt-date").textContent = dateStr;
  document.getElementById("receipt-user").textContent = user.name;
  document.getElementById("receipt-amount").textContent =
    formatRupiah(ORDER_TOTAL);
  document.getElementById("receipt-score").textContent = score.toFixed(4);

  // Persistence: Store last transaction
  localStorage.setItem(
    "palmid_last_payment",
    JSON.stringify({
      txnId,
      user: user.name,
      amount: ORDER_TOTAL,
      date: dateStr,
      score,
    }),
  );

  document.getElementById("btn-finish").onclick = () => {
    localStorage.removeItem("palmid_last_payment");
    window.location.reload();
  };
}

// ── Run ────────────────────────────────────────────────────
init();

// ============================================================
// js/pages/payment.js — Palm Payment (UX-improved)
//
// Perubahan UX:
// • Tidak ada window.location.reload() di tombol "Selesai".
//   Selesai → reset state in-place, scanner siap pakai lagi
//   tanpa kehilangan terminal log.
// • Error pembayaran kasih opsi "Coba lagi" jelas, bukan
//   diam-diam resume scanner.
// • Receipt muncul dengan animasi smooth.
// ============================================================

import { mountNavbar } from "../components/navbar.js";
import { PalmScanner } from "../components/palm-scanner.js";
import { DiagnosticTerminal } from "../components/terminal.js";
import { paymentPay } from "../api/demos.js";
import { showModal } from "../components/modal.js";
import { toast } from "../components/toast.js";
import { formatRupiah, generateTxnId } from "../utils.js";
import { animateIn, fadeSwap } from "../ux.js";

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

const MERCHANT_NAME = "Toko Maju Jaya";
const ORDER_TOTAL = 127500;

let scanner = null;
let terminal = null;
let isPaymentProcessing = false;
let scannerStarted = false;

let btnStartPayment, btnStopPayment, terminalCard, scannerContainer;
let checkoutPanel, receiptPanel;

async function init() {
  mountNavbar();

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

  btnStartPayment.addEventListener("click", startPayment);
  btnStopPayment?.addEventListener("click", stopPayment);

  window.addEventListener("beforeunload", () => scanner?.stop());
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) scanner?.stop();
  });

  terminal.addLog("SYSTEM", "Ready to process payment.");
}

async function startPayment() {
  if (scannerStarted) return;

  try {
    scannerStarted = true;
    setPaymentButtonLoading();

    terminal.addLog("SYSTEM", "Payment mode activated. Waiting for palm scan...");
    terminal.addLog("INSTRUCTION", "Langkah 2: Hadapkan telapak tangan ke kamera.");

    const ok = await scanner.start();
    if (!ok) throw new Error("Gagal mengaktifkan kamera");

    if (btnStopPayment) btnStopPayment.hidden = false;
  } catch (err) {
    console.error("[Payment] Error starting scanner:", err);
    scannerStarted = false;
    resetPaymentButton("Coba Aktifkan Scanner Lagi");
    terminal.addLog("ERROR", err.message || "Failed to start camera");
    toast.error("Tidak bisa menyalakan kamera. Periksa izin browser.");
  }
}

function stopPayment() {
  scanner?.stop();
  scannerStarted = false;
  if (btnStopPayment) btnStopPayment.hidden = true;
  resetPaymentButton();
  terminal.addLog("SYSTEM", "Payment scanner stopped by user.");
}

function setPaymentButtonLoading() {
  btnStartPayment.disabled = true;
  btnStartPayment.innerHTML = `<span class="spinner spinner--sm"></span> Menunggu Scan...`;
}

function resetPaymentButton(label = "Bayar dengan Telapak") {
  btnStartPayment.disabled = false;
  btnStartPayment.innerHTML = `💳 ${label}`;
}

async function handleIdentified(user, score, latency) {
  if (isPaymentProcessing) return;

  terminal.addLog(
    "PAYMENT",
    `User ${escapeHtml(user.name)} matched — score ${score.toFixed(4)}`,
  );

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
    icon: "creditCard",
    initialFocus: "cancel",
    confirmLabel: "Konfirmasi & Bayar",
    confirmVariant: "success",
    onConfirm: () => processPayment(user, score),
    onCancel: () => {
      terminal.addLog("SYSTEM", "Payment cancelled by user.");
      scanner.resume();
    },
  });
}

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

    scanner?.stop();
    scannerStarted = false;
    if (btnStopPayment) btnStopPayment.hidden = true;

    // Fade swap: checkout panel keluar, receipt masuk
    if (checkoutPanel) {
      checkoutPanel.classList.add("hidden");
    }
    showReceipt(user, result, score);
    toast.success("Pembayaran berhasil diproses.", "Sukses");
  } catch (err) {
    terminal.addLog("ERROR", err.message || "Failed to process payment");
    // Recovery jelas: kasih opsi retry, bukan diam-diam resume
    showModal({
      title: "Pembayaran Gagal",
      message: `Maaf, transaksi tidak dapat diproses.<br><br><span class="text-xs" style="color:var(--color-coffee-light)">${escapeHtml(err.message || "Kesalahan jaringan")}</span>`,
      icon: "error",
      confirmLabel: "Coba Lagi",
      confirmVariant: "primary",
      onConfirm: () => processPayment(user, score),
      onCancel: () => {
        isPaymentProcessing = false;
        scanner?.resume();
      },
    });
    return;
  } finally {
    isPaymentProcessing = false;
  }
}

function showReceipt(user, result, score) {
  if (!receiptPanel) return;
  receiptPanel.classList.remove("hidden");
  animateIn(receiptPanel, "fade");

  const txnId = result.transaction_id || generateTxnId();
  const dateStr = new Date().toLocaleString("id-ID");

  setText("receipt-txn-id", txnId);
  setText("receipt-date", dateStr);
  setText("receipt-user", user.name);
  setText("receipt-amount", formatRupiah(ORDER_TOTAL));
  setText("receipt-score", score.toFixed(4));

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

  const btnFinish = document.getElementById("btn-finish");
  if (btnFinish) {
    btnFinish.onclick = async () => {
      localStorage.removeItem("palmid_last_payment");
      await resetToFreshCheckout();
    };
  }
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

/**
 * Reset UI ke kondisi awal TANPA reload halaman.
 * Receipt fade out → checkout panel fade in → scanner siap pakai.
 */
async function resetToFreshCheckout() {
  // Receipt out
  if (receiptPanel) {
    await fadeSwap(receiptPanel, () => receiptPanel.classList.add("hidden"));
  }
  // Checkout panel back
  if (checkoutPanel) {
    checkoutPanel.classList.remove("hidden");
    animateIn(checkoutPanel, "fade");
  }
  // Scanner UI reset
  if (scannerContainer) scannerContainer.style.display = "";
  resetPaymentButton();
  if (btnStartPayment) btnStartPayment.style.display = "";
  if (btnStopPayment) {
    btnStopPayment.style.display = "";
    btnStopPayment.hidden = true;
  }
  terminal.addLog("SYSTEM", "Siap untuk transaksi berikutnya.");
}

init();

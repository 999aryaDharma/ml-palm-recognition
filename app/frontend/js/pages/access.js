// ============================================================
// js/pages/access.js — Access Control (UX-improved)
//
// Perubahan UX:
// • Skeleton row saat memuat daftar otorisasi (bukan teks "Memuat...")
// • Toggle otorisasi pakai feedback halus (label fade, opsional revert)
// • Access log entry baru → animasi slide-in + highlight sebentar
// • Door panel transition lebih jelas
// • Error fetch authorized → tombol "Coba lagi" inline
// ============================================================

import { mountNavbar } from "../components/navbar.js";
import { PalmScanner } from "../components/palm-scanner.js";
import { DiagnosticTerminal } from "../components/terminal.js";
import { toast } from "../components/toast.js";
import { apiFetch } from "../api/client.js";
import { getUsers } from "../api/users.js";
import { renderSkeleton, animateIn, renderEmptyState } from "../ux.js";

let scanner = null;
let terminal = null;
let isProcessing = false;
let scannerStarted = false;

let scannerContainer, terminalCard, doorPanelInner, doorStatus, doorIcon;
let authorizedList, btnStartScan, accessLog, btnRefreshAuth;

async function init() {
  mountNavbar();

  scannerContainer = document.getElementById("scanner-container");
  terminalCard = document.querySelector(".terminal-card");
  doorPanelInner = document.getElementById("door-panel-inner");
  doorStatus = document.getElementById("door-status");
  doorIcon = document.getElementById("door-icon");
  authorizedList = document.getElementById("authorized-list");
  btnStartScan = document.getElementById("btn-start-scan");
  accessLog = document.getElementById("access-log");
  btnRefreshAuth = document.getElementById("btn-refresh-auth");

  if (!btnStartScan || !terminalCard || !scannerContainer) {
    console.error("[Access] Required DOM elements not found!");
    return;
  }

  terminal = new DiagnosticTerminal(terminalCard);
  document.addEventListener("toast", (e) => {
    if (e.detail && toast[e.detail.type]) toast[e.detail.type](e.detail.message);
  });

  terminal.addLog("SYSTEM", "Access control system initialized.");
  terminal.addLog("INSTRUCTION", "Langkah 1: Pilih pengguna yang diotorisasi.");
  terminal.addLog("INSTRUCTION", "Langkah 2: Klik 'Aktifkan Scanner Area'.");

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
      triggerAccessDenied("Unknown", score, "unknown");
      enableStartButton();
    },
    autoResetMs: 4000,
    captureIntervalMs: 1500,
  });

  btnStartScan?.addEventListener("click", startScanner);
  btnRefreshAuth?.addEventListener("click", () => loadAuthorizedPanel(true));

  loadAuthorizedPanel();

  window.addEventListener("beforeunload", () => scanner?.stop());
  
  let wasScanning = false;
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      wasScanning = scannerStarted;
      if (scannerStarted) {
        scanner?.stop();
        scannerStarted = false;
      }
    } else {
      if (wasScanning) {
        startScanner();
      }
    }
  });

  // Reactive backend offline protection
  document.addEventListener("backend-status-change", (e) => {
    const online = e.detail.online;
    if (online) {
      if (btnStartScan && btnStartScan.disabled && btnStartScan.innerHTML.includes("Offline")) {
        enableStartButton();
      }
    } else {
      if (btnStartScan) {
        btnStartScan.disabled = true;
        btnStartScan.innerHTML = "⚠️ Server Offline";
      }
    }
  });
}

async function startScanner() {
  if (scannerStarted) return;
  scannerStarted = true;
  if (btnStartScan) {
    btnStartScan.disabled = true;
    btnStartScan.innerHTML = `<span class="spinner spinner--sm"></span> Scanning…`;
  }
  terminal.addLog("SYSTEM", "Access scanner activated.");
  const ok = await scanner.start();
  if (!ok) {
    scannerStarted = false;
    enableStartButton("Coba Aktifkan Scanner Lagi");
    toast.error("Gagal menyalakan kamera. Periksa izin browser.");
  }
}

function enableStartButton(label = "🖐 Mulai Scan Akses") {
  scannerStarted = false;
  if (btnStartScan) {
    btnStartScan.disabled = false;
    btnStartScan.textContent = label;
  }
}

async function handleIdentified(user, score, latency) {
  if (isProcessing) return;
  isProcessing = true;

  terminal.addLog(
    "MATCHING",
    `${user.name} — score ${score.toFixed(4)} — ${latency}ms`,
  );

  try {
    const authList = await apiFetch("/demos/access/authorized");
    const record = authList.find((r) => r.user_id === user.id);
    const authorized = record?.authorized ?? false;

    if (authorized) {
      terminal.addLog("RESULT", `✅ ACCESS GRANTED — ${user.name}`);
      triggerAccessGranted(user, score);
    } else {
      terminal.addLog(
        "RESULT",
        `🔒 ACCESS DENIED — ${user.name} (tidak diotorisasi)`,
      );
      triggerAccessDenied(user.name, score, "not_authorized");
    }

    apiFetch("/demo-logs", {
      method: "POST",
      body: JSON.stringify({
        user_id: user.id,
        demo_type: "access",
        match_score: score,
        payload: {
          granted: authorized,
          reason: authorized ? "authorized" : "not_authorized",
        },
      }),
    }).catch(() => {});

    addAccessLogEntry(user.name, authorized, score);
    scanner.pause();
  } catch (err) {
    terminal.addLog("ERROR", err.message || "Gagal memeriksa otorisasi");
    toast.error("Gagal memeriksa otorisasi.");
  } finally {
    isProcessing = false;
    setTimeout(() => {
      resetDoor();
      scanner.resume();
      enableStartButton();
    }, 3500);
  }
}

// ── Door ───────────────────────────────────────────────────
function triggerAccessGranted(user, score) {
  if (doorIcon) doorIcon.textContent = "🔓";
  if (doorStatus) {
    doorStatus.textContent = `✅ Access Granted — ${escHtml(user.name)}`;
    doorStatus.className = "door-status door-status--unlocked";
  }
  doorPanelInner?.classList.add("door-panel--granted");
  doorPanelInner?.classList.remove("door-panel--denied");
  animateIn(doorPanelInner, "fade");
  toast.success(`Akses diberikan: ${user.name}`, "Access Granted");
}

function triggerAccessDenied(name, score, reason) {
  if (doorIcon) doorIcon.textContent = "🔒";
  if (doorStatus) {
    doorStatus.textContent =
      reason === "not_authorized"
        ? `🔒 Access Denied — ${escHtml(name)} (tidak diotorisasi)`
        : `🔒 Access Denied — Pengguna tidak dikenali`;
    doorStatus.className = "door-status door-status--denied";
  }
  doorPanelInner?.classList.add("door-panel--denied");
  doorPanelInner?.classList.remove("door-panel--granted");
  animateIn(doorPanelInner, "fade");
  toast.error(
    reason === "not_authorized"
      ? `Akses ditolak: ${name} tidak diotorisasi.`
      : "Pengguna tidak dikenali.",
    "Access Denied",
  );
}

function resetDoor() {
  if (doorIcon) doorIcon.textContent = "🔒";
  if (doorStatus) {
    doorStatus.textContent = "Terdaftar & Terkunci";
    doorStatus.className = "door-status door-status--locked";
  }
  doorPanelInner?.classList.remove("door-panel--granted", "door-panel--denied");
}

// ── Authorized panel ───────────────────────────────────────
async function loadAuthorizedPanel(isRefresh = false) {
  if (!authorizedList) return;

  // Skeleton state
  renderSkeleton(authorizedList, 4, "row");

  try {
    const [users, authData] = await Promise.all([
      getUsers(),
      apiFetch("/demos/access/authorized"),
    ]);

    const authMap = {};
    for (const a of authData) authMap[a.user_id] = a.authorized;

    if (users.length === 0) {
      renderEmptyState(authorizedList, {
        variant: "empty",
        title: "Belum ada pengguna",
        message: "Daftarkan pengguna baru terlebih dahulu.",
        actionLabel: "Buka halaman Enroll",
        onAction: () => window.location.assign("../enroll.html"),
      });
      return;
    }

    authorizedList.innerHTML = users
      .map(
        (u) => `
        <div class="authorized-row ux-anim-fade-in" data-uid="${u.id}">
          <div style="display:flex;align-items:center;gap:8px">
            <div style="width:28px;height:28px;border-radius:6px;background:var(--color-coffee);display:flex;align-items:center;justify-content:center;color:white;font-size:11px;font-weight:700;flex-shrink:0">
              ${escHtml(u.name[0])}
            </div>
            <span style="font-size:13px;font-weight:500">${escHtml(u.name)}</span>
          </div>
          <label style="display:flex;align-items:center;gap:6px;cursor:pointer;user-select:none">
            <input type="checkbox" data-uid="${u.id}"
              ${authMap[u.id] ? "checked" : ""}
              style="width:16px;height:16px;accent-color:var(--color-matcha);cursor:pointer"/>
            <span class="auth-label" style="font-size:11px;color:var(--color-coffee-light);transition:color .2s ease">
              ${authMap[u.id] ? "Diotorisasi" : "Ditolak"}
            </span>
          </label>
        </div>
      `,
      )
      .join("");

    authorizedList.querySelectorAll("input[type=checkbox]").forEach((cb) => {
      cb.addEventListener("change", async () => {
        const uid = Number(cb.dataset.uid);
        const auth = cb.checked;
        const label = cb.nextElementSibling;
        cb.disabled = true;
        // Optimistic label update
        if (label) {
          label.textContent = auth ? "Diotorisasi" : "Ditolak";
          label.style.color = "var(--color-coffee)";
        }
        try {
          await apiFetch(`/demos/access/authorized/${uid}?authorized=${auth}`, {
            method: "PUT",
          });
          terminal.addLog(
            "ADMIN",
            `User #${uid} otorisasi → ${auth ? "GRANTED" : "DENIED"}`,
          );
          toast.info(`Otorisasi user #${uid} diperbarui.`);
        } catch {
          cb.checked = !auth;
          if (label) label.textContent = !auth ? "Diotorisasi" : "Ditolak";
          toast.error("Gagal memperbarui otorisasi.");
        } finally {
          cb.disabled = false;
          if (label) setTimeout(() => (label.style.color = ""), 400);
        }
      });
    });

    if (isRefresh) toast.info("Daftar otorisasi diperbarui.");
  } catch (err) {
    renderEmptyState(authorizedList, {
      variant: "error",
      title: "Gagal memuat daftar",
      message: err.message || "Periksa koneksi backend.",
      actionLabel: "Coba lagi",
      onAction: () => loadAuthorizedPanel(true),
    });
  }
}

// ── Access log ─────────────────────────────────────────────
function addAccessLogEntry(name, granted, score) {
  if (!accessLog) return;
  const placeholder = accessLog.querySelector("p");
  if (placeholder) placeholder.remove();

  const entry = document.createElement("div");
  entry.className = "access-log-entry";
  entry.innerHTML = `
    <span style="
      padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700;flex-shrink:0;
      background:${granted ? "var(--color-matcha-soft)" : "var(--color-coral-soft)"};
      color:${granted ? "var(--color-matcha)" : "var(--color-coral)"}">
      ${granted ? "GRANTED" : "DENIED"}
    </span>
    <span style="flex:1;font-weight:500">${escHtml(name)}</span>
    <span style="font-family:monospace;color:var(--color-coffee-light);font-size:10px">${score.toFixed(3)}</span>
    <span style="color:var(--color-coffee-light);font-size:10px">${new Date().toLocaleTimeString("id-ID")}</span>
  `;
  accessLog.insertBefore(entry, accessLog.firstChild);
  animateIn(entry, "slide");
}

function escHtml(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[c],
  );
}

init();

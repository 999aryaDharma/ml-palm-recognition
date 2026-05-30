// ============================================================
// js/pages/access.js — Access Control demo
// Fixed: door animation classes, access log render, auth panel, terminal
// ============================================================

import { mountNavbar } from "../components/navbar.js";
import { PalmScanner } from "../components/palm-scanner.js";
import { DiagnosticTerminal } from "../components/terminal.js";
import { toast } from "../components/toast.js";
import { apiFetch } from "../api/client.js";
import { getUsers } from "../api/users.js";

// ── State ──────────────────────────────────────────────────
let scanner = null;
let terminal = null;
let isProcessing = false;

// ── DOM ────────────────────────────────────────────────────
const scannerContainer = document.getElementById("scanner-container");
const terminalBody = document.getElementById("terminal-body");
const doorPanelInner = document.getElementById("door-panel-inner"); // inner div for animation class
const doorStatus = document.getElementById("door-status");
const doorIcon = document.getElementById("door-icon");
const authorizedList = document.getElementById("authorized-list");
const btnStartScan = document.getElementById("btn-start-scan");
const accessLog = document.getElementById("access-log");
const btnRefreshAuth = document.getElementById("btn-refresh-auth");

// ── Init ───────────────────────────────────────────────────
function init() {
  mountNavbar();

  terminal = new DiagnosticTerminal(terminalBody);
  terminal.addLog("SYSTEM", "Access control system initialized.");
  terminal.addLog("INFO", "Centang pengguna yang diotorisasi di panel kiri.");
  terminal.addLog("INFO", "Klik Mulai Scan untuk mengaktifkan pemindai.");

  scanner = new PalmScanner({
    containerEl: scannerContainer,
    videoEl: document.getElementById("scanner-video"),
    hintEl: document.getElementById("scanner-hint"),
    resultEl: document.getElementById("scanner-result"),
    placeholderEl: document.getElementById("scanner-placeholder"),
    logFn: (tag, msg) => terminal.addLog(tag, msg),
    onIdentified: handleIdentified,
    onUnknown: (score) => {
      terminal.addLog("RESULT", `UNKNOWN — score ${score.toFixed(4)}`);
      triggerAccessDenied("Unknown", score, "unknown");
      enableStartButton();
    },
    autoResetMs: 4000,
    captureIntervalMs: 1500,
  });

  btnStartScan?.addEventListener("click", startScanner);
  btnRefreshAuth?.addEventListener("click", () => loadAuthorizedPanel());

  loadAuthorizedPanel();

  window.addEventListener("beforeunload", () => scanner?.stop());
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) scanner?.stop();
  });
}

// ── Scanner ────────────────────────────────────────────────
async function startScanner() {
  if (btnStartScan) {
    btnStartScan.disabled = true;
    btnStartScan.innerHTML = `<span class="spinner spinner--sm"></span> Scanning…`;
  }
  terminal.addLog("SYSTEM", "Access scanner activated.");
  await scanner.start();
}

function enableStartButton() {
  if (btnStartScan) {
    btnStartScan.disabled = false;
    btnStartScan.textContent = "🖐 Mulai Scan Akses";
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

  try {
    // Fetch current authorization list
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

    // Log to backend (fire and forget)
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
  } catch (err) {
    terminal.addLog("ERROR", err.message || "Gagal memeriksa otorisasi");
    toast.error("Gagal memeriksa otorisasi.");
  } finally {
    isProcessing = false;
    // Reset door after delay, then re-enable scan
    setTimeout(() => {
      resetDoor();
      scanner.resume();
      enableStartButton();
    }, 3500);
  }
}

// ── Door animation ─────────────────────────────────────────
function triggerAccessGranted(user, score) {
  if (doorIcon) doorIcon.textContent = "🔓";
  if (doorStatus) {
    doorStatus.textContent = `✅ Access Granted — ${escHtml(user.name)}`;
    doorStatus.className = "door-status door-status--unlocked";
  }
  doorPanelInner?.classList.add("door-panel--granted");
  doorPanelInner?.classList.remove("door-panel--denied");
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

// ── Authorized users panel ─────────────────────────────────
async function loadAuthorizedPanel() {
  if (!authorizedList) return;
  authorizedList.innerHTML = `<p class="text-xs" style="color:var(--color-coffee-light)">Memuat daftar...</p>`;

  try {
    const [users, authData] = await Promise.all([
      getUsers(),
      apiFetch("/demos/access/authorized"),
    ]);

    // Build map: user_id → authorized bool
    const authMap = {};
    for (const a of authData) authMap[a.user_id] = a.authorized;

    if (users.length === 0) {
      authorizedList.innerHTML = `
        <p class="text-xs" style="color:var(--color-coffee-light)">
          Belum ada pengguna. <a href="../enroll.html" style="color:var(--color-matcha)">Daftarkan pengguna</a> terlebih dahulu.
        </p>`;
      return;
    }

    authorizedList.innerHTML = users
      .map(
        (u) => `
        <div class="authorized-row">
          <div style="display:flex;align-items:center;gap:8px">
            <div style="width:28px;height:28px;border-radius:6px;background:var(--color-coffee);display:flex;align-items:center;justify-content:center;color:white;font-size:11px;font-weight:700;flex-shrink:0">
              ${escHtml(u.name[0])}
            </div>
            <span style="font-size:13px;font-weight:500">${escHtml(u.name)}</span>
          </div>
          <label style="display:flex;align-items:center;gap:6px;cursor:pointer;user-select:none">
            <input
              type="checkbox"
              data-uid="${u.id}"
              ${authMap[u.id] ? "checked" : ""}
              style="width:16px;height:16px;accent-color:var(--color-matcha);cursor:pointer"
            />
            <span class="auth-label" style="font-size:11px;color:var(--color-coffee-light)">
              ${authMap[u.id] ? "Diotorisasi" : "Ditolak"}
            </span>
          </label>
        </div>
      `,
      )
      .join("");

    // Attach toggle handlers
    authorizedList.querySelectorAll("input[type=checkbox]").forEach((cb) => {
      cb.addEventListener("change", async () => {
        const uid = Number(cb.dataset.uid);
        const auth = cb.checked;
        const label = cb.nextElementSibling;
        cb.disabled = true;
        try {
          await apiFetch(`/demos/access/authorized/${uid}?authorized=${auth}`, {
            method: "PUT",
          });
          if (label) label.textContent = auth ? "Diotorisasi" : "Ditolak";
          terminal.addLog(
            "ADMIN",
            `User #${uid} otorisasi → ${auth ? "GRANTED" : "DENIED"}`,
          );
          toast.info(`Otorisasi user #${uid} diperbarui.`);
        } catch {
          cb.checked = !auth; // revert on failure
          toast.error("Gagal memperbarui otorisasi.");
        } finally {
          cb.disabled = false;
        }
      });
    });
  } catch (err) {
    authorizedList.innerHTML = `<p class="text-xs" style="color:var(--color-coral)">Gagal memuat daftar: ${escHtml(err.message)}</p>`;
  }
}

// ── Access log ─────────────────────────────────────────────
function addAccessLogEntry(name, granted, score) {
  if (!accessLog) return;

  // Remove placeholder text on first entry
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

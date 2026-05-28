// ============================================================
// js/pages/access.js — Access Control demo
// Full implementation: door animation, authorized panel, scanner
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
const scannerVideo = document.getElementById("scanner-video");
const scannerHint = document.getElementById("scanner-hint");
const scannerResult = document.getElementById("scanner-result");
const scannerPlaceholder = document.getElementById("scanner-placeholder");
const terminalBody = document.getElementById("terminal-body");
const doorPanel = document.getElementById("door-panel");
const doorStatus = document.getElementById("door-status");
const doorIcon = document.getElementById("door-icon");
const authorizedList = document.getElementById("authorized-list");
const btnStartScan = document.getElementById("btn-start-scan");
const accessLog = document.getElementById("access-log");

// ── Init ───────────────────────────────────────────────────
function init() {
  mountNavbar();

  terminal = new DiagnosticTerminal(terminalBody);
  terminal.addLog("CAMERA_READY", "Access control system initialized.");

  scanner = new PalmScanner({
    containerEl: scannerContainer,
    videoEl: scannerVideo,
    hintEl: scannerHint,
    resultEl: scannerResult,
    placeholderEl: scannerPlaceholder,
    logFn: (tag, msg) => terminal.addLog(tag, msg),
    onIdentified: handleIdentified,
    onUnknown: (score) => {
      terminal.addLog("RESULT", `UNKNOWN — score ${score.toFixed(4)}`);
      triggerAccessDenied("Unknown user", score);
    },
    autoResetMs: 4000,
  });

  btnStartScan?.addEventListener("click", startScanner);
  loadAuthorizedPanel();

  window.addEventListener("beforeunload", () => scanner?.stop());
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) scanner?.stop();
  });
}

async function startScanner() {
  if (btnStartScan) {
    btnStartScan.disabled = true;
    btnStartScan.innerHTML = `<span class="spinner spinner--sm"></span> Scanning…`;
  }
  terminal.addLog("SYSTEM", "Access scanner activated.");
  await scanner.start();
}

// ── Identification callback ────────────────────────────────
async function handleIdentified(user, score, latency) {
  if (isProcessing) return;
  isProcessing = true;

  terminal.addLog("MATCHING", `${user.name} — score ${score.toFixed(4)}`);

  try {
    // Check authorization via backend (or check local list for speed)
    const result = await apiFetch("/demos/access/authorized");
    const record = result.find((r) => r.user_id === user.id);
    const authorized = record?.authorized ?? false;

    terminal.addLog(
      "RESULT",
      authorized
        ? `ACCESS GRANTED — ${user.name}`
        : `ACCESS DENIED — ${user.name} (not authorized)`,
    );

    if (authorized) {
      triggerAccessGranted(user, score);
    } else {
      triggerAccessDenied(user.name, score, "not_authorized");
    }

    // Log to backend
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
    terminal.addLog("ERROR", err.message);
    toast.error("Gagal memeriksa otorisasi.");
  } finally {
    isProcessing = false;
    setTimeout(() => resetDoor(), 3000);
    resetStartButton();
  }
}

// ── Door animation ─────────────────────────────────────────
function triggerAccessGranted(user, score) {
  setDoorState(
    "unlocked",
    `🔓 Access Granted — ${escHtml(user.name)}`,
    "matcha",
  );
  doorPanel?.classList.add("door--open");
  toast.success(`Akses diberikan: ${user.name}`, "Access Granted");
}

function triggerAccessDenied(name, score, reason = "unknown") {
  const msg =
    reason === "not_authorized"
      ? `🔒 Access Denied — ${escHtml(name)} (tidak diotorisasi)`
      : `🔒 Access Denied — Pengguna tidak dikenali`;
  setDoorState("locked", msg, "coral");
  doorPanel?.classList.remove("door--open");
  toast.error(
    reason === "not_authorized"
      ? `Akses ditolak: ${name} tidak diotorisasi.`
      : "Pengguna tidak dikenali.",
  );
}

function setDoorState(state, labelText, color) {
  if (doorStatus) {
    doorStatus.textContent = labelText;
    doorStatus.className = `door-status door-status--${state}`;
  }
  if (doorIcon) {
    doorIcon.textContent = state === "unlocked" ? "🔓" : "🔒";
  }
}

function resetDoor() {
  if (doorStatus) {
    doorStatus.textContent = "🔒 Terdaftar & Terkunci";
    doorStatus.className = "door-status door-status--locked";
  }
  if (doorIcon) doorIcon.textContent = "🔒";
  doorPanel?.classList.remove("door--open");
}

// ── Authorized panel ───────────────────────────────────────
async function loadAuthorizedPanel() {
  if (!authorizedList) return;
  try {
    const [users, authorized] = await Promise.all([
      getUsers(),
      apiFetch("/demos/access/authorized"),
    ]);
    const authMap = Object.fromEntries(
      authorized.map((a) => [a.user_id, a.authorized]),
    );

    if (users.length === 0) {
      authorizedList.innerHTML = `<p class="text-xs text-muted">Belum ada pengguna terdaftar.</p>`;
      return;
    }

    authorizedList.innerHTML = users
      .map(
        (u) => `
      <div class="authorized-row" style="display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--color-border)">
        <span style="font-size:13px;font-weight:500">${escHtml(u.name)}</span>
        <label style="display:flex;align-items:center;gap:8px;cursor:pointer">
          <input type="checkbox" data-uid="${u.id}" ${authMap[u.id] ? "checked" : ""}
            style="width:16px;height:16px;accent-color:var(--color-matcha)" />
          <span style="font-size:11px;color:var(--color-coffee-light)">${authMap[u.id] ? "Diotorisasi" : "Ditolak"}</span>
        </label>
      </div>
    `,
      )
      .join("");

    // Toggle handler
    authorizedList.querySelectorAll("input[type=checkbox]").forEach((cb) => {
      cb.addEventListener("change", async () => {
        const uid = Number(cb.dataset.uid);
        const auth = cb.checked;
        const span = cb.nextElementSibling;
        try {
          await apiFetch(`/demos/access/authorized/${uid}?authorized=${auth}`, {
            method: "PUT",
          });
          if (span) span.textContent = auth ? "Diotorisasi" : "Ditolak";
          terminal.addLog("SYSTEM", `User #${uid} authorization → ${auth}`);
        } catch {
          cb.checked = !auth;
          toast.error("Gagal memperbarui otorisasi.");
        }
      });
    });
  } catch {
    authorizedList.innerHTML = `<p class="text-xs" style="color:var(--color-coral)">Gagal memuat daftar.</p>`;
  }
}

// ── Access log ─────────────────────────────────────────────
function addAccessLogEntry(name, granted, score) {
  if (!accessLog) return;
  const entry = document.createElement("div");
  entry.style.cssText =
    "display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid var(--color-border);font-size:12px";
  entry.innerHTML = `
    <span style="padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700;background:${granted ? "var(--color-matcha-soft)" : "var(--color-coral-soft)"};color:${granted ? "var(--color-matcha)" : "var(--color-coral)"}">${granted ? "GRANTED" : "DENIED"}</span>
    <span style="flex:1">${escHtml(name)}</span>
    <span style="font-family:monospace;color:var(--color-coffee-light)">${score.toFixed(3)}</span>
    <span style="color:var(--color-coffee-light)">${new Date().toLocaleTimeString("id-ID")}</span>
  `;
  accessLog.insertBefore(entry, accessLog.firstChild);
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

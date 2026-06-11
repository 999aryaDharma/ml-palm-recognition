// ============================================================
// js/pages/user-detail.js — Detail page (UX-improved)
//
// Perubahan UX:
// • Tidak ada auto-redirect lagi. ID invalid / not-found →
//   inline empty state + tombol "Kembali", user yang putuskan.
// • Delete sukses: kartu detail fade out, lalu history.back().
// • Skeleton placeholder saat fetch detail.
// • Tombol "Test Scan" jadi toggle (Stop tersedia).
// ============================================================

import { mountNavbar } from "../components/navbar.js";
import { getUser, deleteUser } from "../api/users.js";
import { PalmScanner } from "../components/palm-scanner.js";
import { toast } from "../components/toast.js";
import { showModal } from "../components/modal.js";
import { formatDate } from "../utils.js";
import { smoothBack, renderEmptyState, fadeSwap } from "../ux.js";

// ── State ──────────────────────────────────────────────────
let scanner = null;
let currentUserId = null;
let userData = null;
let scanActive = false;

// ── DOM ────────────────────────────────────────────────────
const nameTitle = document.getElementById("user-name-title");
const idSubtitle = document.getElementById("user-id-subtitle");
const templateCount = document.getElementById("template-count");
const enrolledDate = document.getElementById("enrolled-date");
const templateStatus = document.getElementById("template-status");
const btnDeleteUser = document.getElementById("btn-delete-user");
const btnTestScan = document.getElementById("btn-test-scan");
const detailRoot =
  document.getElementById("user-detail-root") || document.querySelector("main");

async function init() {
  mountNavbar();

  const params = new URLSearchParams(window.location.search);
  currentUserId = params.get("id");

  if (!currentUserId) {
    // Jangan redirect otomatis — kasih opsi ke user
    showInvalidState("ID Pengguna tidak ditemukan di URL.");
    return;
  }

  showLoadingSkeleton();
  await loadUserDetail();

  if (!userData) return; // Error sudah ditangani di loadUserDetail

  // Setup Scanner
  scanner = new PalmScanner({
    containerEl: document.getElementById("scanner-container"),
    videoEl: document.getElementById("scanner-video"),
    hintEl: document.getElementById("scanner-hint"),
    resultEl: document.getElementById("scanner-result"),
    placeholderEl: document.getElementById("scanner-placeholder"),
    boundingBoxLayerEl: document.getElementById("palm-box-layer"),
    onIdentified: (user, score) => {
      if (String(user.id) === String(currentUserId)) {
        toast.success(`Identitas terkonfirmasi sebagai ${user.name}`);
      } else {
        toast.warning(
          `Terdeteksi sebagai ${user.name} (ID: ${user.id}) — bukan pengguna ini.`,
        );
      }
    },
    onUnknown: (score) => {
      toast.error(
        `Pengguna tidak dikenali (skor: ${score.toFixed(4)})`,
      );
    },
  });

  btnTestScan?.addEventListener("click", toggleScan);
  btnDeleteUser?.addEventListener("click", confirmDelete);

  window.addEventListener("beforeunload", () => scanner?.stop());

  let wasScanning = false;
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      wasScanning = scanActive;
      if (scanActive) {
        scanner?.stop();
        scanActive = false;
        if (btnTestScan) {
          btnTestScan.disabled = false;
          btnTestScan.textContent = "Mulai Test Scan";
        }
      }
    } else {
      if (wasScanning) {
        toggleScan();
      }
    }
  });

  // Reactive backend offline protection
  document.addEventListener("backend-status-change", (e) => {
    const online = e.detail.online;
    if (online) {
      if (btnTestScan && btnTestScan.disabled && btnTestScan.innerHTML.includes("Offline")) {
        btnTestScan.disabled = false;
        btnTestScan.textContent = "Mulai Test Scan";
      }
    } else {
      if (btnTestScan) {
        if (scanActive) {
          scanner?.stop();
          scanActive = false;
        }
        btnTestScan.disabled = true;
        btnTestScan.innerHTML = "⚠️ Server Offline";
      }
    }
  });
}

function showLoadingSkeleton() {
  if (nameTitle) nameTitle.innerHTML = `<span class="ux-skeleton ux-skeleton--line" style="width:180px;height:24px;display:inline-block"></span>`;
  if (idSubtitle) idSubtitle.innerHTML = `<span class="ux-skeleton ux-skeleton--line" style="width:100px;display:inline-block"></span>`;
  if (templateCount) templateCount.textContent = "—";
  if (enrolledDate) enrolledDate.textContent = "—";
  if (templateStatus) templateStatus.innerHTML = `<span class="ux-skeleton ux-skeleton--line" style="width:120px;display:inline-block"></span>`;
}

function showInvalidState(message) {
  if (!detailRoot) {
    toast.error(message);
    return;
  }
  renderEmptyState(detailRoot, {
    variant: "error",
    title: "Pengguna tidak ditemukan",
    message,
    actionLabel: "Kembali ke daftar pengguna",
    onAction: () => smoothBack("users.html"),
  });
}

async function loadUserDetail() {
  try {
    userData = await getUser(currentUserId);

    if (nameTitle) nameTitle.textContent = userData.name;
    if (idSubtitle) idSubtitle.textContent = `ID: #${userData.id}`;
    if (templateCount)
      templateCount.textContent = `${userData.template_count || 0} template`;
    if (enrolledDate) enrolledDate.textContent = formatDate(userData.enrolled_at);

    if (templateStatus) {
      if ((userData.template_count || 0) >= 5) {
        templateStatus.innerHTML = `<span class="badge badge--identified">Sangat Stabil</span>`;
      } else {
        templateStatus.innerHTML = `<span class="badge badge--unknown">Kurang Stabil (${userData.template_count}/5)</span>`;
      }
    }
  } catch (err) {
    userData = null;
    showInvalidState(err.message || "Tidak bisa mengambil data dari server.");
  }
}

function toggleScan() {
  if (!scanner || !btnTestScan) return;
  if (scanActive) {
    scanner.stop();
    scanActive = false;
    btnTestScan.disabled = false;
    btnTestScan.textContent = "Mulai Test Scan";
  } else {
    scanner.start();
    scanActive = true;
    btnTestScan.innerHTML = `<span class="spinner spinner--sm" style="margin-right:6px"></span> Berhenti Scan`;
  }
}

function confirmDelete() {
  if (!userData) return;
  showModal({
    title: "Hapus Pengguna",
    message: `Apakah Anda yakin ingin menghapus <strong>${escapeHtml(userData.name)}</strong>? Seluruh data biometrik yang terkait akan dihapus secara permanen.`,
    icon: "🗑️",
    confirmLabel: "Hapus Pengguna",
    confirmVariant: "danger",
    onConfirm: async () => {
      try {
        await deleteUser(currentUserId);
        toast.success(`Pengguna ${userData.name} telah dihapus.`);
        scanner?.stop();
        // Fade keluar halaman lalu kembali — bukan timeout redirect mendadak
        if (detailRoot) {
          await fadeSwap(
            detailRoot,
            () => {
              detailRoot.innerHTML = `
                <div class="ux-empty-state" style="padding:var(--space-10);text-align:center">
                  <div style="font-size:2rem">✅</div>
                  <h2 style="margin:.5rem 0">Pengguna telah dihapus</h2>
                  <p class="text-sm" style="color:var(--color-coffee-light)">Mengembalikan Anda ke daftar pengguna…</p>
                </div>
              `;
            },
            220,
          );
        }
        setTimeout(() => smoothBack("users.html"), 700);
      } catch (err) {
        toast.error("Gagal menghapus pengguna: " + err.message);
      }
    },
  });
}

function escapeHtml(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[c],
  );
}

init();

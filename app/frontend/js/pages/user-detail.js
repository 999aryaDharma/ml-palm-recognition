// ============================================================
// js/pages/user-detail.js — User detail page logic
// ============================================================

import { mountNavbar } from "../components/navbar.js";
import { getUser, deleteUser } from "../api/users.js";
import { PalmScanner } from "../components/palm-scanner.js";
import { toast } from "../components/toast.js";
import { showModal } from "../components/modal.js";
import { formatDate } from "../utils.js";

// ── State ──────────────────────────────────────────────────
let scanner = null;
let currentUserId = null;
let userData = null;

// ── DOM Elements ───────────────────────────────────────────
const nameTitle = document.getElementById("user-name-title");
const idSubtitle = document.getElementById("user-id-subtitle");
const templateCount = document.getElementById("template-count");
const enrolledDate = document.getElementById("enrolled-date");
const templateStatus = document.getElementById("template-status");
const btnDeleteUser = document.getElementById("btn-delete-user");
const btnTestScan = document.getElementById("btn-test-scan");

// ── Initialization ─────────────────────────────────────────
async function init() {
  mountNavbar();

  // Get ID from URL
  const params = new URLSearchParams(window.location.search);
  currentUserId = params.get("id");

  if (!currentUserId) {
    toast.error("ID Pengguna tidak valid.");
    setTimeout(() => (window.location.href = "users.html"), 2000);
    return;
  }

  await loadUserDetail();

  // Setup Scanner
  scanner = new PalmScanner({
    containerEl: document.getElementById("scanner-container"),
    videoEl: document.getElementById("scanner-video"),
    hintEl: document.getElementById("scanner-hint"),
    resultEl: document.getElementById("scanner-result"),
    placeholderEl: document.getElementById("scanner-placeholder"),
    onIdentified: (user, score) => {
      if (user.id == currentUserId) {
        toast.success(`Berhasil! Identitas terkonfirmasi sebagai ${user.name}`);
      } else {
        toast.warning(`Peringatan: Terdeteksi sebagai ${user.name} (ID: ${user.id})`);
      }
    },
    onUnknown: (score) => {
      toast.error(`Gagal: Pengguna tidak dikenali (Score: ${score.toFixed(4)})`);
    },
  });

  btnTestScan.addEventListener("click", () => {
    scanner.start();
    btnTestScan.disabled = true;
    btnTestScan.textContent = "Scanning...";
  });

  btnDeleteUser.addEventListener("click", confirmDelete);
}

/** Fetch user data from backend */
async function loadUserDetail() {
  try {
    userData = await getUser(currentUserId);

    nameTitle.textContent = userData.name;
    idSubtitle.textContent = `ID: #${userData.id}`;
    templateCount.textContent = `${userData.template_count || 0} template`;
    enrolledDate.textContent = formatDate(userData.enrolled_at);

    if ((userData.template_count || 0) >= 5) {
      templateStatus.innerHTML = `<span class="badge badge--identified">Sangat Stabil</span>`;
    } else {
      templateStatus.innerHTML = `<span class="badge badge--unknown">Kurang Stabil (${userData.template_count}/5)</span>`;
    }
  } catch (err) {
    toast.error("Gagal memuat detail pengguna: " + err.message);
    nameTitle.textContent = "Error";
  }
}

/** Show confirmation modal before deleting */
function confirmDelete() {
  showModal({
    title: "Hapus Pengguna",
    message: `Apakah Anda yakin ingin menghapus <strong>${userData.name}</strong>? Seluruh data biometrik yang terkait akan dihapus secara permanen.`,
    icon: "🗑️",
    confirmLabel: "Hapus Pengguna",
    confirmVariant: "danger",
    onConfirm: async () => {
      try {
        await deleteUser(currentUserId);
        toast.success(`Pengguna ${userData.name} telah dihapus.`);
        setTimeout(() => (window.location.href = "users.html"), 1500);
      } catch (err) {
        toast.error("Gagal menghapus pengguna: " + err.message);
      }
    },
  });
}

// ── Run ────────────────────────────────────────────────────
init();

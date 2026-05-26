// ============================================================
// js/pages/users.js — Users list page logic
// ============================================================

import { mountNavbar } from "../components/navbar.js";
import { getUsers, deleteUser } from "../api/users.js";
import { toast } from "../components/toast.js";
import { showModal } from "../components/modal.js";
import { formatDate, getInitial, debounce } from "../utils.js";

// ── State ──────────────────────────────────────────────────
let allUsers = [];

// ── DOM Elements ───────────────────────────────────────────
const grid = document.getElementById("users-grid");
const searchInput = document.getElementById("user-search");
const emptyState = document.getElementById("empty-state");

// ── Initialization ─────────────────────────────────────────
function init() {
  mountNavbar();
  loadUsers();

  searchInput.addEventListener("input", debounce(handleSearch, 300));
}

/** Fetch users from backend and render */
async function loadUsers() {
  try {
    // Keep skeletons for a moment for smooth UX
    allUsers = await getUsers();
    renderUsers(allUsers);
  } catch (err) {
    toast.error("Gagal memuat daftar pengguna: " + err.message);
    grid.innerHTML = "";
  }
}

/** Render user cards to the grid */
function renderUsers(users) {
  if (users.length === 0) {
    grid.innerHTML = "";
    grid.classList.add("hidden");
    emptyState.classList.remove("hidden");
    return;
  }

  emptyState.classList.add("hidden");
  grid.classList.remove("hidden");

  grid.innerHTML = users
    .map(
      (user) => `
    <div class="user-card" data-id="${user.id}">
      <div class="user-avatar">${getInitial(user.name)}</div>
      <div class="user-info">
        <div class="user-name">${user.name}</div>
        <div class="user-meta">
          <span>${user.template_count || 0} template</span> • 
          <span>Terdaftar: ${formatDate(user.enrolled_at)}</span>
        </div>
      </div>
      <div class="user-actions">
        <a href="user-detail.html?id=${user.id}" class="btn btn--sm btn--secondary" title="Lihat Detail">
           Detail
        </a>
        <button class="btn btn--sm btn--ghost btn-delete" data-id="${user.id}" data-name="${user.name}" title="Hapus Pengguna">
           <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M3 6h18m-2 0v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6m3 0V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>
           </svg>
        </button>
      </div>
    </div>
  `,
    )
    .join("");

  // Attach delete events
  grid.querySelectorAll(".btn-delete").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const { id, name } = btn.dataset;
      confirmDelete(id, name);
    });
  });
}

/** Filter users based on search input */
function handleSearch() {
  const query = searchInput.value.toLowerCase().trim();
  const filtered = allUsers.filter((u) => u.name.toLowerCase().includes(query));
  renderUsers(filtered);
}

/** Show confirmation modal before deleting */
function confirmDelete(id, name) {
  // Escape user name to prevent XSS
  const escapedName = String(name).replace(/[<>&]/g, (c) => {
    const map = { "<": "&lt;", ">": "&gt;", "&": "&amp;" };
    return map[c];
  });

  showModal({
    title: "Hapus Pengguna",
    message: `Apakah Anda yakin ingin menghapus <strong>${escapedName}</strong>? Seluruh data biometrik yang terkait akan dihapus secara permanen.`,
    icon: "🗑️",
    confirmLabel: "Hapus Pengguna",
    confirmVariant: "danger",
    onConfirm: async () => {
      try {
        await deleteUser(id);
        toast.success(`Pengguna ${escapedName} telah dihapus.`);
        loadUsers(); // reload list
      } catch (err) {
        toast.error("Gagal menghapus pengguna: " + err.message);
      }
    },
  });
}

// ── Run ────────────────────────────────────────────────────
init();

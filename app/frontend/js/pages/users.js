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
function escapeHtml(value) {
  return String(value ?? "").replace(
    /[&<>"']/g,
    (char) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;",
      })[char],
  );
}

function splitFormattedDate(value) {
  const formatted = formatDate(value);

  if (!formatted || formatted === "—") {
    return {
      date: "—",
      time: "—",
    };
  }

  const parts = formatted.split("pukul");

  return {
    date: parts[0]?.trim() || formatted,
    time: parts[1]?.trim() || "—",
  };
}

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
    .map((user) => {
      const safeName = escapeHtml(user.name);
      const templateCount = user.template_count || 0;
      const dateInfo = splitFormattedDate(user.enrolled_at);
      const compactDate = `${dateInfo.date} · ${dateInfo.time}`;

      return `
        <article class="user-card" data-id="${user.id}">
          <div class="user-card__top">
            <div class="user-card__avatar" aria-hidden="true">
              ${getInitial(user.name)}
            </div>

            <div class="user-card__main">
              <h3 class="user-card__name" title="${safeName}">
                ${safeName}
              </h3>
              <span class="user-card__template">
                ${templateCount} template
              </span>
            </div>
          </div>

          <div class="user-card__meta">
            <span class="user-card__meta-label">Terdaftar</span>
            <span class="user-card__meta-value" title="${escapeHtml(compactDate)}">
              ${escapeHtml(compactDate)}
            </span>
          </div>

          <div class="user-card__actions">
            <a
              href="user-detail.html?id=${encodeURIComponent(user.id)}"
              class="btn btn--secondary btn--sm user-card__detail"
              aria-label="Lihat detail pengguna ${safeName}"
            >
              Detail
            </a>

            <button
              type="button"
              class="btn-icon-danger btn-delete"
              data-id="${user.id}"
              aria-label="Hapus pengguna ${safeName}"
              title="Hapus Pengguna"
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
                stroke="currentColor" stroke-width="2" stroke-linecap="round"
                stroke-linejoin="round" aria-hidden="true">
                <path d="M3 6h18" />
                <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
                <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
                <path d="M10 11v6" />
                <path d="M14 11v6" />
              </svg>
            </button>
          </div>
        </article>
      `;
    })
    .join("");

  grid.querySelectorAll(".btn-delete").forEach((btn) => {
    btn.addEventListener("click", () => {
      const selectedUser = users.find(
        (user) => String(user.id) === String(btn.dataset.id),
      );

      if (!selectedUser) return;
      confirmDelete(selectedUser.id, selectedUser.name);
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
    icon: "error", // Use SVG icon key (red alert circle)
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

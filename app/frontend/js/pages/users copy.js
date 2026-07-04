// ============================================================
// js/pages/users.js — Users list (UX-improved)
//
// Perubahan UX:
// • Skeleton card saat initial load (bukan layar kosong)
// • Optimistic delete: kartu fade-out instan, rollback on error
// • Empty state pencarian dengan tombol "Reset" (bukan layar kosong)
// • Tombol delete jadi loading state saat request berjalan
// ============================================================

import { mountNavbar } from "../components/navbar.js";
import { getUsers, deleteUser } from "../api/users.js";
import { toast } from "../components/toast.js";
import { showModal } from "../components/modal.js";
import { formatDate, getInitial, debounce } from "../utils.js";
import {
  renderSkeleton,
  animateOutAndRemove,
  withLoading,
  renderEmptyState,
} from "../ux.js";

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
  searchInput?.addEventListener("input", debounce(handleSearch, 250));
}

async function loadUsers() {
  // Skeleton dulu — user langsung tahu "lagi loading", bukan layar kosong
  if (grid) {
    grid.classList.remove("hidden");
    renderSkeleton(grid, 6, "card");
  }
  emptyState?.classList.add("hidden");

  try {
    allUsers = await getUsers();
    renderUsers(allUsers);
  } catch (err) {
    renderEmptyState(grid, {
      variant: "error",
      title: "Gagal memuat pengguna",
      message: err.message || "Periksa koneksi backend lalu coba lagi.",
      actionLabel: "Coba lagi",
      onAction: loadUsers,
    });
  }
}

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
  if (!formatted || formatted === "—") return { date: "—", time: "—" };
  const parts = formatted.split("pukul");
  return {
    date: parts[0]?.trim() || formatted,
    time: parts[1]?.trim() || "—",
  };
}

function renderUsers(users) {
  if (!grid) return;

  if (users.length === 0) {
    grid.innerHTML = "";
    grid.classList.add("hidden");
    if (emptyState) {
      emptyState.classList.remove("hidden");
      // Kalau pencarian aktif, tawarkan reset — bukan diam saja
      const query = searchInput?.value?.trim();
      if (query) {
        renderEmptyState(emptyState, {
          variant: "empty",
          title: `Tidak ada hasil untuk "${escapeHtml(query)}"`,
          message: "Coba kata kunci lain atau reset pencarian.",
          actionLabel: "Reset pencarian",
          onAction: () => {
            if (searchInput) searchInput.value = "";
            renderUsers(allUsers);
          },
        });
      }
    }
    return;
  }

  emptyState?.classList.add("hidden");
  grid.classList.remove("hidden");

  grid.innerHTML = users
    .map((user) => {
      const safeName = escapeHtml(user.name);
      const templateCount = user.template_count || 0;
      const dateInfo = splitFormattedDate(user.enrolled_at);
      const compactDate = `${dateInfo.date} · ${dateInfo.time}`;

      return `
        <article class="user-card ux-anim-fade-in" data-id="${user.id}">
          <div class="user-card__top">
            <div class="user-card__avatar" aria-hidden="true">${getInitial(user.name)}</div>
            <div class="user-card__main">
              <h3 class="user-card__name" title="${safeName}">${safeName}</h3>
              <span class="user-card__template">${templateCount} template</span>
            </div>
          </div>
          <div class="user-card__meta">
            <span class="user-card__meta-label">Terdaftar</span>
            <span class="user-card__meta-value" title="${escapeHtml(compactDate)}">${escapeHtml(compactDate)}</span>
          </div>
          <div class="user-card__actions">
            <a href="user-detail.html?id=${encodeURIComponent(user.id)}"
               class="btn btn--secondary btn--sm user-card__detail"
               aria-label="Lihat detail pengguna ${safeName}">Detail</a>
            <button type="button" class="btn-icon-danger btn-delete"
              data-id="${user.id}" aria-label="Hapus pengguna ${safeName}" title="Hapus Pengguna">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
                stroke="currentColor" stroke-width="2" stroke-linecap="round"
                stroke-linejoin="round" aria-hidden="true">
                <path d="M3 6h18" /><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
                <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
                <path d="M10 11v6" /><path d="M14 11v6" />
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
        (u) => String(u.id) === String(btn.dataset.id),
      );
      if (!selectedUser) return;
      confirmDelete(selectedUser.id, selectedUser.name, btn);
    });
  });
}

function handleSearch() {
  const query = searchInput.value.toLowerCase().trim();
  const filtered = allUsers.filter((u) => u.name.toLowerCase().includes(query));
  renderUsers(filtered);
}

function confirmDelete(id, name, triggerBtn) {
  const safeName = escapeHtml(name);
  showModal({
    title: "Hapus Pengguna",
    message: `Apakah Anda yakin ingin menghapus <strong>${safeName}</strong>? Seluruh data biometrik yang terkait akan dihapus secara permanen.`,
    icon: "error",
    confirmLabel: "Hapus Pengguna",
    confirmVariant: "danger",
    onConfirm: async () => {
      // Optimistic: kartu fade-out langsung, hapus dari state lokal.
      // Kalau gagal di server → rollback (re-render).
      const card = grid?.querySelector(`.user-card[data-id="${id}"]`);
      const previousUsers = [...allUsers];
      allUsers = allUsers.filter((u) => String(u.id) !== String(id));
      if (card) await animateOutAndRemove(card);

      try {
        await deleteUser(id);
        toast.success(`Pengguna ${name} telah dihapus.`);
        // Re-render kalau jadi kosong (untuk munculin empty state)
        if (allUsers.length === 0) renderUsers(allUsers);
      } catch (err) {
        // Rollback
        allUsers = previousUsers;
        renderUsers(allUsers);
        toast.error("Gagal menghapus pengguna: " + err.message);
      }
    },
  });
}

init();

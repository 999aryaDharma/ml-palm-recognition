// ============================================================
// js/components/toast.js — Toast notification system
// ============================================================

import { ICONS as SVG_ICONS } from "../icons.js";

const MAX_TOASTS = 3;
let container = null;

function getContainer() {
  if (!container) {
    container = document.getElementById("toast-container");
    if (!container) {
      container = document.createElement("div");
      container.id = "toast-container";
      document.body.appendChild(container);
    }
  }
  return container;
}

const ICON_MAP = {
  success: SVG_ICONS.checkCircle(20),
  error: SVG_ICONS.alertCircle(20),
  warning: SVG_ICONS.alertTriangle(20),
  info: SVG_ICONS.info(20),
};

/**
 * Show a toast notification.
 * @param {string} message
 * @param {'info'|'success'|'warning'|'error'} type
 * @param {number} duration ms
 * @param {string} [title] optional title
 */
export function showToast(message, type = "info", duration = 3500, title = "") {
  const c = getContainer();

  // Limit max visible toasts
  const existing = c.querySelectorAll(".toast");
  if (existing.length >= MAX_TOASTS) {
    existing[0].remove();
  }

  const toast = document.createElement("div");
  toast.className = `toast toast--${type}`;
  toast.setAttribute("role", type === "error" ? "alert" : "status");
  toast.setAttribute("aria-live", type === "error" ? "assertive" : "polite");

  const displayTitle =
    title ||
    {
      success: "Berhasil",
      error: "Terjadi Kesalahan",
      warning: "Perhatian",
      info: "Informasi",
    }[type];

  toast.innerHTML = `
    <span class="toast-icon" aria-hidden="true">${ICON_MAP[type]}</span>
    <div class="toast-body">
      <div class="toast-title">${displayTitle}</div>
      <div class="toast-msg">${message}</div>
    </div>
    <button class="toast-close" aria-label="Tutup notifikasi">&times;</button>
  `;

  c.appendChild(toast);

  // Trigger enter animation
  requestAnimationFrame(() => {
    requestAnimationFrame(() => toast.classList.add("toast--visible"));
  });

  // Auto remove
  const timer = setTimeout(() => removeToast(toast), duration);

  // Close button action
  toast.querySelector(".toast-close")?.addEventListener("click", (e) => {
    e.stopPropagation();
    clearTimeout(timer);
    removeToast(toast);
  });

  // Click anywhere to dismiss (optional, kept for convenience)
  toast.addEventListener("click", () => {
    clearTimeout(timer);
    removeToast(toast);
  });
}

function removeToast(toast) {
  toast.classList.remove("toast--visible");
  toast.addEventListener("transitionend", () => toast.remove(), { once: true });
}

// Convenience helpers
export const toast = {
  success: (msg, title) => showToast(msg, "success", 3500, title),
  error: (msg, title) => showToast(msg, "error", 5000, title),
  warning: (msg, title) => showToast(msg, "warning", 4000, title),
  info: (msg, title) => showToast(msg, "info", 3500, title),
};

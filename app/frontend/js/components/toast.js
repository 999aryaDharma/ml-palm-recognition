// ============================================================
// js/components/toast.js — Toast notification system
// ============================================================

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

const ICONS = {
  success: "✅",
  error: "❌",
  warning: "⚠️",
  info: "ℹ️",
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
    <span class="toast-icon">${ICONS[type]}</span>
    <div class="toast-body">
      <div class="toast-title">${displayTitle}</div>
      <div class="toast-msg">${message}</div>
    </div>
  `;

  c.appendChild(toast);

  // Trigger enter animation
  requestAnimationFrame(() => {
    requestAnimationFrame(() => toast.classList.add("toast--visible"));
  });

  // Auto remove
  const timer = setTimeout(() => removeToast(toast), duration);

  // Click to dismiss
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

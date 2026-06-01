// ============================================================
// js/components/modal.js — Reusable confirm modal
// ============================================================

import { ICONS } from "../icons.js";

let activeModal = null;
let lastFocusedElement = null;

const ICON_MAP = {
  success: ICONS.checkCircle(40),
  error: ICONS.alertCircle(40),
  warning: ICONS.alertTriangle(40),
  info: ICONS.info(40),
  creditCard: ICONS.creditCard(),
  clipboard: ICONS.clipboard(),
  doorOpen: ICONS.doorOpen(),
  hospital: ICONS.hospital(),
};

/**
 * Show a confirmation modal.
 * @param {object} opts
 * @param {string} opts.title
 * @param {string} opts.message   — can include HTML
 * @param {string} [opts.icon]    — key from ICON_MAP
 * @param {string} [opts.confirmLabel]  default "Konfirmasi"
 * @param {string} [opts.cancelLabel]   default "Batal"
 * @param {string} [opts.confirmVariant] btn variant: 'primary'|'danger'|'success'
 * @param {'confirm'|'cancel'} [opts.initialFocus] which button to focus first
 * @param {Function} opts.onConfirm
 * @param {Function} [opts.onCancel]
 */
export function showModal({
  title,
  message,
  icon = "",
  confirmLabel = "Konfirmasi",
  cancelLabel = "Batal",
  confirmVariant = "primary",
  initialFocus,
  onConfirm,
  onCancel,
}) {
  // Save last focused element for restoration
  lastFocusedElement = document.activeElement;

  // Remove previous modal
  closeModal();

  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop";
  backdrop.setAttribute("role", "dialog");
  backdrop.setAttribute("aria-modal", "true");
  backdrop.setAttribute("aria-labelledby", "modal-title");
  backdrop.setAttribute("aria-describedby", "modal-desc");

  const iconHtml = ICON_MAP[icon] 
    ? `<div class="modal-icon" aria-hidden="true">${ICON_MAP[icon]}</div>` 
    : "";

  backdrop.innerHTML = `
    <div class="modal">
      ${iconHtml}
      <h3 class="modal-title" id="modal-title">${title}</h3>
      <div class="modal-body" id="modal-desc">${message}</div>
      <div class="modal-actions">
        <button class="btn btn--ghost" id="modal-cancel">${cancelLabel}</button>
        <button class="btn btn--${confirmVariant}" id="modal-confirm">${confirmLabel}</button>
      </div>
    </div>
  `;

  document.body.appendChild(backdrop);
  activeModal = backdrop;

  // Open animation
  requestAnimationFrame(() => {
    requestAnimationFrame(() => backdrop.classList.add("is-open"));
  });

  const cancelBtn = backdrop.querySelector("#modal-cancel");
  const confirmBtn = backdrop.querySelector("#modal-confirm");

  // Initial focus logic
  if (initialFocus === "cancel") {
    cancelBtn.focus();
  } else if (initialFocus === "confirm") {
    confirmBtn.focus();
  } else {
    // Default legacy logic
    if (confirmVariant === "danger") {
      cancelBtn.focus();
    } else {
      confirmBtn.focus();
    }
  }

  // Trap focus inside modal
  backdrop.addEventListener("keydown", handleTrapFocus);

  // Events
  cancelBtn.addEventListener("click", () => {
    closeModal();
    onCancel?.();
  });

  confirmBtn.addEventListener("click", () => {
    closeModal();
    onConfirm?.();
  });

  // Close on backdrop click
  backdrop.addEventListener("click", (e) => {
    if (e.target === backdrop) {
      closeModal();
      onCancel?.();
    }
  });

  // ESC to close
  const escHandler = (e) => {
    if (e.key === "Escape") {
      closeModal();
      onCancel?.();
      document.removeEventListener("keydown", escHandler);
    }
  };
  document.addEventListener("keydown", escHandler);
}

export function closeModal() {
  if (!activeModal) return;
  activeModal.classList.remove("is-open");
  const m = activeModal;
  m.addEventListener(
    "transitionend",
    () => {
      m.remove();
      // Restore focus
      if (lastFocusedElement && typeof lastFocusedElement.focus === "function") {
        lastFocusedElement.focus();
      }
    },
    { once: true },
  );
  activeModal = null;
}

function handleTrapFocus(e) {
  if (e.key !== "Tab") return;
  const focusable = activeModal?.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
  );
  if (!focusable || focusable.length === 0) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (e.shiftKey) {
    if (document.activeElement === first) {
      e.preventDefault();
      last.focus();
    }
  } else {
    if (document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }
}

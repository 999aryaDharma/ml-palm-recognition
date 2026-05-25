// ============================================================
// js/components/modal.js — Reusable confirm modal
// ============================================================

let activeModal = null;

/**
 * Show a confirmation modal.
 * @param {object} opts
 * @param {string} opts.title
 * @param {string} opts.message   — can include HTML
 * @param {string} [opts.icon]    — emoji or text
 * @param {string} [opts.confirmLabel]  default "Konfirmasi"
 * @param {string} [opts.cancelLabel]   default "Batal"
 * @param {string} [opts.confirmVariant] btn variant: 'primary'|'danger'|'success'
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
  onConfirm,
  onCancel,
}) {
  // Remove previous modal
  closeModal();

  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop";
  backdrop.setAttribute("role", "dialog");
  backdrop.setAttribute("aria-modal", "true");
  backdrop.setAttribute("aria-labelledby", "modal-title");

  backdrop.innerHTML = `
    <div class="modal">
      ${icon ? `<div class="modal-icon">${icon}</div>` : ""}
      <h3 class="modal-title" id="modal-title">${title}</h3>
      <div class="modal-body">${message}</div>
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

  // Focus cancel by default for destructive actions
  const cancelBtn = backdrop.querySelector("#modal-cancel");
  const confirmBtn = backdrop.querySelector("#modal-confirm");

  if (confirmVariant === "danger") {
    cancelBtn.focus();
  } else {
    confirmBtn.focus();
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
  m.addEventListener("transitionend", () => m.remove(), { once: true });
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

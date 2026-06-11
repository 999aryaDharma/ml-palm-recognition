// ============================================================
// js/ux.js — Shared UX helpers
//
// Tujuan: hilangkan auto-reload, kasih feedback konsisten,
// transisi halus, dan recovery yang user-driven.
//
// Cara pakai: import { withLoading, fadeSwap, smoothBack,
//   renderSkeleton, animateIn } from "./ux.js";
// ============================================================

/**
 * Bungkus tombol jadi loading state, auto-restore saat selesai.
 * Mencegah double-click + kasih spinner inline yang konsisten.
 *
 * @param {HTMLButtonElement|HTMLElement} btn
 * @param {string} loadingLabel  Label saat loading (HTML allowed)
 * @param {() => Promise<any>} task  Async task yang dijalankan
 * @returns {Promise<any>}
 */
export async function withLoading(btn, loadingLabel, task) {
  if (!btn) return task();
  const original = btn.innerHTML;
  const wasDisabled = btn.disabled;
  btn.disabled = true;
  btn.setAttribute("aria-busy", "true");
  btn.innerHTML = `<span class="spinner spinner--sm" style="display:inline-block;vertical-align:middle;margin-right:8px"></span>${loadingLabel}`;
  try {
    return await task();
  } finally {
    btn.innerHTML = original;
    btn.disabled = wasDisabled;
    btn.removeAttribute("aria-busy");
  }
}

/**
 * Fade-out elemen lama, jalankan callback (biasanya ganti konten),
 * lalu fade-in. Bikin transisi step jauh lebih smooth.
 *
 * @param {HTMLElement} el
 * @param {() => void | Promise<void>} mutate
 * @param {number} duration ms
 */
export async function fadeSwap(el, mutate, duration = 220) {
  if (!el) return mutate?.();
  el.style.transition = `opacity ${duration}ms ease, transform ${duration}ms ease`;
  el.style.opacity = "0";
  el.style.transform = "translateY(4px)";
  await new Promise((r) => setTimeout(r, duration));
  await mutate?.();
  // Force reflow so transition restarts
  void el.offsetHeight;
  el.style.opacity = "1";
  el.style.transform = "translateY(0)";
  await new Promise((r) => setTimeout(r, duration));
  el.style.transition = "";
  el.style.transform = "";
}

/**
 * Pengganti `window.location.href = "..."`.
 * Kembali ke halaman sebelumnya kalau ada history, kalau tidak
 * fallback ke URL yang diberikan. Tetap pakai navigasi (multi-page
 * app) tapi tidak terasa seperti reload random.
 *
 * @param {string} fallback URL bila history kosong
 */
export function smoothBack(fallback = "index.html") {
  if (window.history.length > 1) {
    window.history.back();
  } else {
    window.location.assign(fallback);
  }
}

/**
 * Render skeleton placeholder ke container.
 * @param {HTMLElement} el
 * @param {number} count Jumlah skeleton card
 * @param {string} variant "card" | "row"
 */
export function renderSkeleton(el, count = 4, variant = "card") {
  if (!el) return;
  const tpl =
    variant === "row"
      ? `<div class="ux-skeleton-row"><div class="ux-skeleton ux-skeleton--avatar"></div><div style="flex:1"><div class="ux-skeleton ux-skeleton--line" style="width:60%"></div><div class="ux-skeleton ux-skeleton--line" style="width:40%;margin-top:6px"></div></div></div>`
      : `<div class="ux-skeleton-card">
           <div class="ux-skeleton ux-skeleton--avatar"></div>
           <div class="ux-skeleton ux-skeleton--line" style="width:70%"></div>
           <div class="ux-skeleton ux-skeleton--line" style="width:50%"></div>
           <div class="ux-skeleton ux-skeleton--line" style="width:90%;margin-top:auto"></div>
         </div>`;
  el.innerHTML = Array.from({ length: count }, () => tpl).join("");
}

/**
 * Tambahkan class animasi masuk ke elemen baru, lalu auto-cleanup.
 * @param {HTMLElement} el
 * @param {"fade"|"slide"} kind
 */
export function animateIn(el, kind = "fade") {
  if (!el) return;
  const cls = kind === "slide" ? "ux-anim-slide-in" : "ux-anim-fade-in";
  el.classList.add(cls);
  setTimeout(() => el.classList.remove(cls), 600);
}

/**
 * Animasi keluar (fade + collapse) lalu hapus elemen.
 * @param {HTMLElement} el
 * @param {number} duration ms
 */
export async function animateOutAndRemove(el, duration = 240) {
  if (!el) return;
  el.style.transition = `opacity ${duration}ms ease, transform ${duration}ms ease, max-height ${duration}ms ease, margin ${duration}ms ease, padding ${duration}ms ease`;
  el.style.maxHeight = el.offsetHeight + "px";
  void el.offsetHeight;
  el.style.opacity = "0";
  el.style.transform = "scale(0.96)";
  el.style.maxHeight = "0px";
  el.style.marginTop = "0";
  el.style.marginBottom = "0";
  el.style.paddingTop = "0";
  el.style.paddingBottom = "0";
  el.style.overflow = "hidden";
  await new Promise((r) => setTimeout(r, duration));
  el.remove();
}

/**
 * Tampilkan inline empty/error state dengan tombol aksi
 * (pengganti auto-redirect setelah toast).
 *
 * @param {HTMLElement} container
 * @param {{title:string, message:string, actionLabel?:string, onAction?:()=>void, variant?:"error"|"empty"}} opts
 */
export function renderEmptyState(container, opts) {
  if (!container) return;
  const color =
    opts.variant === "error" ? "var(--color-coral)" : "var(--color-coffee)";
  container.innerHTML = `
    <div class="ux-empty-state" style="
      display:flex;flex-direction:column;align-items:center;justify-content:center;
      padding:var(--space-10) var(--space-6);text-align:center;gap:var(--space-3);
      animation:ux-fade-in .35s ease both;">
      <div style="font-size:2rem">${opts.variant === "error" ? "⚠️" : "📭"}</div>
      <h3 style="color:${color};margin:0">${opts.title}</h3>
      <p class="text-sm" style="color:var(--color-coffee-light);max-width:42ch;margin:0">${opts.message}</p>
      ${
        opts.actionLabel
          ? `<button type="button" class="btn btn--secondary btn--sm" id="ux-empty-action" style="margin-top:var(--space-2)">${opts.actionLabel}</button>`
          : ""
      }
    </div>
  `;
  if (opts.actionLabel && opts.onAction) {
    container
      .querySelector("#ux-empty-action")
      ?.addEventListener("click", opts.onAction);
  }
}

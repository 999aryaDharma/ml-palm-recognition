// ============================================================
// js/utils.js — Helper utilities
// ============================================================

/** Map backend error codes → user-friendly hint copy */
export const QUALITY_HINTS = {
  no_hand_detected: "🖐 Tunjukkan telapak tangan ke kamera",
  detection_failed: "🖐 Telapak belum terbaca. Pastikan tangan terlihat penuh",
  landmarks_occluded: "👁 Pastikan seluruh jari dan telapak terlihat",
  palm_facing_wrong: "🔄 Hadapkan telapak tangan ke kamera",
  hand_too_small: "↔ Dekatkan tangan ke kamera",
  fingers_not_open: "✋ Buka jari sedikit lebih lebar",
  image_too_blurry: "📷 Tahan tangan diam sebentar",
  roi_extraction_failed: "🧭 Posisikan telapak di tengah frame",
  network_error: "🌐 Backend tidak dapat dihubungi. Periksa server",
  camera_permission_denied: "🎥 Izin kamera diperlukan untuk memindai telapak",
};

/**
 * Format number as Indonesian Rupiah.
 * @param {number} amount
 * @returns {string} e.g. "Rp 127.500"
 */
export function formatRupiah(amount) {
  return new Intl.NumberFormat("id-ID", {
    style: "currency",
    currency: "IDR",
    minimumFractionDigits: 0,
  }).format(amount);
}

/**
 * Format ISO date string to Indonesian locale.
 * @param {string} isoString
 * @returns {string} e.g. "24 Mei 2026, 09.41"
 */
export function formatDate(isoString) {
  if (!isoString) return "—";
  return new Date(isoString).toLocaleString("id-ID", {
    day: "2-digit",
    month: "long",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/**
 * Format date only (no time).
 * @param {string} isoString
 */
export function formatDateOnly(isoString) {
  if (!isoString) return "—";
  return new Date(isoString).toLocaleDateString("id-ID", {
    day: "2-digit",
    month: "long",
    year: "numeric",
  });
}

/**
 * Format time only.
 * @param {string|Date} date
 */
export function formatTime(date) {
  return new Date(date).toLocaleTimeString("id-ID", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/**
 * Get current timestamp in log format: [HH:MM:SS.mmm]
 */
export function logTimestamp() {
  const now = new Date();
  const hh = String(now.getHours()).padStart(2, "0");
  const mm = String(now.getMinutes()).padStart(2, "0");
  const ss = String(now.getSeconds()).padStart(2, "0");
  const ms = String(now.getMilliseconds()).padStart(3, "0");
  return `${hh}:${mm}:${ss}.${ms}`;
}

/**
 * Get initial letters for avatar (first character of name).
 * @param {string} name
 */
export function getInitial(name) {
  return (name || "?").trim()[0].toUpperCase();
}

/**
 * Debounce a function.
 * @param {Function} fn
 * @param {number} delay ms
 */
export function debounce(fn, delay = 300) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

/**
 * Generate a random transaction ID.
 */
export function generateTxnId() {
  const now = new Date();
  const yy = now.getFullYear();
  const mm = String(now.getMonth() + 1).padStart(2, "0");
  const dd = String(now.getDate()).padStart(2, "0");
  const rand = String(Math.floor(Math.random() * 10000)).padStart(4, "0");
  return `TXN-${yy}${mm}${dd}-${rand}`;
}

/**
 * Mask a string — show first 4 and last 4 chars, rest as X.
 * Used for NIK masking.
 * @param {string} str
 */
export function maskString(str) {
  if (!str || str.length <= 8) return "****";
  return str.slice(0, 4) + "x".repeat(str.length - 8) + str.slice(-4);
}

/**
 * Clamp number between min and max.
 */
export function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

/**
 * Sleep/delay helper.
 * @param {number} ms
 */
export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ============================================================
// js/api/client.js — Base API client with error handling
// ============================================================

const BASE_URL = "http://localhost:8000";

/**
 * Core fetch wrapper. Throws a structured error object on non-2xx.
 * @param {string} path
 * @param {RequestInit} options
 */
async function apiFetch(path, options = {}) {
  const response = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: {
      // Don't set Content-Type for FormData (browser sets boundary automatically)
      ...(options.body instanceof FormData
        ? {}
        : { "Content-Type": "application/json" }),
      ...options.headers,
    },
  });

  if (!response.ok) {
    let errData = {};
    try {
      errData = await response.json();
    } catch (_) {}
    const err = new Error(errData.message || `HTTP ${response.status}`);
    err.status = response.status;
    err.error = errData.error || "unknown_error";
    err.detail = errData.detail || errData.message || "";
    throw err;
  }

  // Handle 204 No Content
  if (response.status === 204) return null;
  return response.json();
}

/**
 * GET /health — check if backend is online
 */
async function checkHealth() {
  try {
    const data = await apiFetch("/health");
    return { online: true, data };
  } catch (_) {
    return { online: false, data: null };
  }
}

export { apiFetch, checkHealth, BASE_URL };

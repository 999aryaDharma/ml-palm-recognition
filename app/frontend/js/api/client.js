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
    
    // FastAPI returns errors under "detail". Sometimes "detail" is an object, sometimes a string.
    let errorDetail = errData.detail;
    let errCode = "unknown_error";
    let errMsg = errData.message || `HTTP ${response.status}`;

    if (errorDetail && typeof errorDetail === "object") {
      errCode = errorDetail.error || errCode;
      errMsg = errorDetail.message || errMsg;
    } else if (typeof errorDetail === "string") {
      errMsg = errorDetail;
    } else {
      errCode = errData.error || errCode;
    }

    const err = new Error(errMsg);
    err.status = response.status;
    err.error = errCode;
    err.detail = errorDetail || errMsg;
    // Note: Browser native fetch will still log a 400 to console, 
    // this is normal and expected for quality gates. We just throw it 
    // so the UI can handle the state.
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

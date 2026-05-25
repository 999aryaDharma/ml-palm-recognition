// ============================================================
// js/api/identify.js — Identification API call
// ============================================================

import { apiFetch } from "./client.js";

/**
 * POST /identify
 * Send a palm image blob for identification.
 * @param {Blob} imageBlob
 * @returns {{ status: 'identified'|'unknown', user: {id,name}|null, score: number, latency_ms: number }}
 */
export const identify = (imageBlob) => {
  const form = new FormData();
  form.append("image", imageBlob, "frame.jpg");
  return apiFetch("/identify", { method: "POST", body: form });
};

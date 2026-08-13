// ============================================================
// js/api/identify.js — Identification API call
// ============================================================

import { apiFetch } from "./client.js";

/**
 * POST /identify
 * Send a palm image blob for identification with optional model_id.
 * @param {Blob} imageBlob
 * @param {string} [modelId]
 * @returns {Promise<object>}
 */
export const identify = (imageBlob, modelId = null) => {
  const form = new FormData();
  form.append("image", imageBlob, "frame.jpg");
  if (modelId) {
    form.append("model_id", modelId);
  }
  return apiFetch("/identify", { method: "POST", body: form });
};

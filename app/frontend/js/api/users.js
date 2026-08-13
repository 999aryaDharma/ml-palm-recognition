// ============================================================
// js/api/users.js — User management API calls
// ============================================================

import { apiFetch } from "./client.js";

/** GET /users */
export const getUsers = () => apiFetch("/users");

/** GET /users/:id */
export const getUser = (id) => apiFetch(`/users/${id}`);

/** POST /users { name } */
export const createUser = (name) =>
  apiFetch("/users", {
    method: "POST",
    body: JSON.stringify({ name }),
  });

/** DELETE /users/:id */
export const deleteUser = (id) =>
  apiFetch(`/users/${id}`, { method: "DELETE" });

/**
 * POST /users/:id/templates
 * Upload a single palm image as a template for one selected model.
 */
export const addTemplate = (userId, imageBlob, modelId = null) => {
  const form = new FormData();
  form.append("image", imageBlob, "frame.jpg");
  if (modelId) {
    form.append("model_id", modelId);
  }
  return apiFetch(`/users/${userId}/templates`, { method: "POST", body: form });
};

/**
 * POST /users/:id/templates/multi
 * One captured frame fans out to every active registry model.
 */
export const addTemplateMulti = (userId, imageBlob) => {
  const form = new FormData();
  form.append("image", imageBlob, "frame.jpg");
  return apiFetch(`/users/${userId}/templates/multi`, {
    method: "POST",
    body: form,
  });
};

/** GET /users/:id/templates */
export const getTemplates = (userId) => apiFetch(`/users/${userId}/templates`);

/** GET /users/:id/verify-ready */
export const verifyUserReady = (userId, modelId = null, modelVersion = null) => {
  const params = new URLSearchParams();
  if (modelId) params.append("model_id", modelId);
  if (modelVersion) params.append("model_version", modelVersion);
  const query = params.toString() ? `?${params.toString()}` : "";
  return apiFetch(`/users/${userId}/verify-ready${query}`);
};

/** GET /users/:id/verify-ready-all */
export const verifyUserReadyAll = (userId) =>
  apiFetch(`/users/${userId}/verify-ready-all`);

/** POST /users/:id/profile */
export const addProfile = (userId, payload) =>
  apiFetch(`/users/${userId}/profile`, {
    method: "POST",
    body: JSON.stringify(payload),
  });

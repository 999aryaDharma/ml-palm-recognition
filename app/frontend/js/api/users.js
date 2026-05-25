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
 * Upload a single palm image as a template for enrollment.
 * @param {number} userId
 * @param {Blob} imageBlob
 */
export const addTemplate = (userId, imageBlob) => {
  const form = new FormData();
  form.append("image", imageBlob, "frame.jpg");
  return apiFetch(`/users/${userId}/templates`, { method: "POST", body: form });
};

/** GET /users/:id/templates */
export const getTemplates = (userId) => apiFetch(`/users/${userId}/templates`);

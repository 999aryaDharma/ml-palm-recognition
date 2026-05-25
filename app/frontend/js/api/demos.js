// ============================================================
// js/api/demos.js — Demo module API calls
// ============================================================

import { apiFetch } from "./client.js";

// ── Payment ───────────────────────────────────────────────
export const paymentPay = (userId, amount, merchant) =>
  apiFetch("/demos/payment/pay", {
    method: "POST",
    body: JSON.stringify({ user_id: userId, amount, merchant }),
  });

// ── Attendance ────────────────────────────────────────────
export const attendanceCheckin = (userId, mode = "checkin") =>
  apiFetch("/demos/attendance/checkin", {
    method: "POST",
    body: JSON.stringify({ user_id: userId, mode }),
  });

// ── Access ────────────────────────────────────────────────
export const getAuthorizedUsers = () => apiFetch("/demos/access/authorized");

export const setAuthorized = (userId, authorized) =>
  apiFetch(`/demos/access/authorized/${userId}`, {
    method: "PUT",
    body: JSON.stringify({ authorized }),
  });

// ── Patient ───────────────────────────────────────────────
export const patientCheckin = (userId) =>
  apiFetch("/demos/patient/checkin", {
    method: "POST",
    body: JSON.stringify({ user_id: userId }),
  });

// ── Logs ─────────────────────────────────────────────────
export const getDemoLogs = (demoType, limit = 20) => {
  const qs = new URLSearchParams({ limit });
  if (demoType) qs.set("demo_type", demoType);
  return apiFetch(`/demo-logs?${qs}`);
};

// ── Seed / Reset ──────────────────────────────────────────
export const seedDemoData = () =>
  apiFetch("/seed-demo-data", { method: "POST" });
export const resetDemoData = () =>
  apiFetch("/seed-demo-data", { method: "DELETE" });

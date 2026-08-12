// js/api/models.js — Model selection API client

import { apiFetch } from './client.js';

/**
 * GET /models — list semua model yang tersedia di registry
 * @returns {{ models: Array, active_model_id: string }}
 */
async function listModels() {
  return apiFetch('/models');
}

/**
 * GET /models/active — get active model info
 * @returns {{ model_id: string, name: string, version: string, training_mode: string, threshold: number, metrics: object }}
 */
async function getActiveModel() {
  return apiFetch('/models/active');
}

/**
 * POST /models/active — set active model
 * @param {string} modelId
 */
async function setActiveModel(modelId) {
  return apiFetch('/models/active', {
    method: 'POST',
    body: JSON.stringify({ model_id: modelId }),
  });
}

export { listModels, getActiveModel, setActiveModel };

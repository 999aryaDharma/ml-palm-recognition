// js/components/model-selector.js
// Reusable model selector component untuk halaman yang butuh model switching.
//
// Usage:
//   import { ModelSelector } from './components/model-selector.js';
//   const selector = new ModelSelector('#model-selector-container');
//   await selector.init();
//   selector.onChange(modelId => { /* re-identify atau re-enroll */ });

import { listModels, setActiveModel } from '../api/models.js';

export class ModelSelector {
  /**
   * @param {string|HTMLElement} container - CSS selector atau element
   */
  constructor(container) {
    this._el = typeof container === 'string'
      ? document.querySelector(container)
      : container;
    this._onChangeCb = null;
    this._models = [];
    this._activeId = null;
  }

  /** Fetch model list dan render selector. */
  async init() {
    if (!this._el) return;
    try {
      const data = await listModels();
      this._models = data.models || [];
      this._activeId = data.active_model_id || null;
      this._render();
    } catch (e) {
      this._renderError();
    }
  }

  /** Register callback yang dipanggil saat model berubah. */
  onChange(cb) {
    this._onChangeCb = cb;
  }

  /** Get current active model_id. */
  get activeModelId() {
    return this._activeId;
  }

  _render() {
    if (this._models.length === 0) {
      this._el.innerHTML = '';
      return;
    }

    const html = `
      <div class="model-selector-wrapper">
        <label class="model-selector-label" for="model-selector-select">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="2" y="3" width="20" height="14" rx="2"/><polyline points="8 21 12 17 16 21"/>
          </svg>
          Model
        </label>
        <div class="model-selector-select-wrap">
          <select id="model-selector-select" class="model-selector-select">
            ${this._models.map(m => `
              <option value="${m.id}" ${m.id === this._activeId ? 'selected' : ''}>
                ${m.name}
              </option>
            `).join('')}
          </select>
          <span class="model-selector-badge" id="model-selector-badge">
            ${this._getBadgeHtml(this._activeId)}
          </span>
        </div>
      </div>
    `;
    this._el.innerHTML = html;
    this._el.querySelector('#model-selector-select')
      .addEventListener('change', (e) => this._handleChange(e.target.value));
  }

  _renderError() {
    this._el.innerHTML = `<span class="model-selector-error">Model registry tidak tersedia</span>`;
  }

  async _handleChange(modelId) {
    if (modelId === this._activeId) return;

    const select = this._el.querySelector('#model-selector-select');
    const badge = this._el.querySelector('#model-selector-badge');
    if (select) select.disabled = true;
    if (badge) badge.innerHTML = `<span class="model-badge-loading">Switching…</span>`;

    try {
      await setActiveModel(modelId);
      this._activeId = modelId;
      if (badge) badge.innerHTML = this._getBadgeHtml(modelId);
      if (this._onChangeCb) this._onChangeCb(modelId);
    } catch (e) {
      // Revert selection
      if (select) select.value = this._activeId;
      if (badge) badge.innerHTML = this._getBadgeHtml(this._activeId);
      console.error('[ModelSelector] Failed to switch model:', e);
    } finally {
      if (select) select.disabled = false;
    }
  }

  _getBadgeHtml(modelId) {
    const model = this._models.find(m => m.id === modelId);
    if (!model) return '';
    const mode = model.training_mode || 'unknown';
    const cls = mode === 'scratch' ? 'badge-scratch' : 'badge-pretrained';
    const label = mode === 'scratch' ? 'Scratch' : 'Pretrained';
    return `<span class="model-badge ${cls}">${label}</span>`;
  }
}

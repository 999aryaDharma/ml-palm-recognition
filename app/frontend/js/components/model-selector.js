// js/components/model-selector.js
// Reusable client-side model selector component.
// Discovers models via GET /models and stores selected modelId locally.
// Does NOT mutate global server active model.

import { listModels } from '../api/models.js';

export class ModelSelector {
  /**
   * @param {string|HTMLElement} container
   */
  constructor(container) {
    this._el = typeof container === 'string'
      ? document.querySelector(container)
      : container;
    this._onChangeCb = null;
    this._models = [];
    this._selectedId = null;
  }

  async init() {
    if (!this._el) return;
    try {
      const data = await listModels();
      this._models = data.models || [];
      // Default to first available or default_model_id
      this._selectedId = data.default_model_id || (this._models[0] ? this._models[0].id : null);
      this._render();
    } catch (e) {
      this._renderError();
    }
  }

  onChange(cb) {
    this._onChangeCb = cb;
  }

  get selectedModelId() {
    return this._selectedId;
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
              <option value="${m.id}" ${m.id === this._selectedId ? 'selected' : ''}>
                ${m.name}
              </option>
            `).join('')}
          </select>
          <span class="model-selector-badge" id="model-selector-badge">
            ${this._getBadgeHtml(this._selectedId)}
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

  _handleChange(modelId) {
    if (modelId === this._selectedId) return;

    this._selectedId = modelId;
    const badge = this._el.querySelector('#model-selector-badge');
    if (badge) badge.innerHTML = this._getBadgeHtml(modelId);

    if (this._onChangeCb) {
      this._onChangeCb(modelId);
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

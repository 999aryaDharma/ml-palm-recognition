// ============================================================
// js/components/terminal.js — Diagnostic terminal
// ============================================================

import { logTimestamp } from "../utils.js";

const TAG_CLASSES = {
  CAMERA_READY: "log-tag--camera",
  CAPTURE: "log-tag--camera",
  DETECTION: "log-tag--detect",
  ROI_EXTRACTION: "log-tag--roi",
  EMBEDDING: "log-tag--embed",
  MATCHING: "log-tag--match",
  RESULT: "log-tag--result",
  ERROR: "log-tag--error",
};

/**
 * DiagnosticTerminal — renders an ML log terminal.
 *
 * @param {HTMLElement} bodyEl — the scrollable log body element
 * @param {number} [maxLines] — max log lines to keep (default 80)
 */
export class DiagnosticTerminal {
  constructor(bodyEl, maxLines = 80) {
    this.bodyEl = bodyEl;
    this.maxLines = maxLines;
    this._lines = 0;
  }

  /**
   * Add a log line.
   * @param {string} tag  — e.g. 'DETECTION', 'RESULT', 'ERROR'
   * @param {string} msg
   */
  addLog(tag, msg) {
    if (!this.bodyEl) return;

    // Trim old lines
    while (this._lines >= this.maxLines) {
      this.bodyEl.firstElementChild?.remove();
      this._lines--;
    }

    const tagClass = TAG_CLASSES[tag] || "log-tag--embed";
    const line = document.createElement("div");
    line.className = "log-line";
    line.innerHTML = `
      <span class="log-time">${logTimestamp()}</span>
      <span class="log-tag ${tagClass}">${tag.padEnd(14)}</span>
      <span class="log-msg">${escapeHtml(msg)}</span>
    `;

    this.bodyEl.appendChild(line);
    this._lines++;

    // Auto-scroll to bottom
    this.bodyEl.scrollTop = this.bodyEl.scrollHeight;
  }

  /** Clear all log lines. */
  clear() {
    if (!this.bodyEl) return;
    this.bodyEl.innerHTML = "";
    this._lines = 0;
  }

  /** Add a separator line. */
  addSeparator() {
    if (!this.bodyEl) return;
    const sep = document.createElement("div");
    sep.style.cssText = `
      border-top: 1px dashed var(--color-border);
      margin: 4px 0;
      opacity: 0.5;
    `;
    this.bodyEl.appendChild(sep);
    this._lines++;
  }
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

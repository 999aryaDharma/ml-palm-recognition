// ============================================================
// js/components/terminal.js — Diagnostic terminal
// ============================================================

import { logTimestamp } from "../utils.js";
import { ICONS } from "../icons.js";

const TAG_CLASSES = {
  SYSTEM: "log-tag--camera",
  INFO: "log-tag--embed",
  INSTRUCTION: "log-tag--match",
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
 * DiagnosticTerminal — renders an ML log terminal with controls.
 * Supports auto-scrolling, tag coloring, and copying logs to clipboard.
 */
export class DiagnosticTerminal {
  /**
   * @param {HTMLElement} containerEl — the .terminal-card element
   * @param {object} [opts]
   * @param {number} [opts.maxLines] — max log lines to keep (default 100)
   * @param {boolean} [opts.compact] — start in compact mode
   */
  constructor(containerEl, opts = {}) {
    this.containerEl = containerEl;
    this.bodyEl = containerEl.querySelector(".terminal__body");
    this.maxLines = opts.maxLines || 100;
    this.isCompact = opts.compact || false;
    this._lines = 0;

    this._initControls();
  }

  _initControls() {
    const header = this.containerEl.querySelector(".terminal__header");
    if (!header) return;

    // Add controls to header if they don't exist
    if (!header.querySelector(".terminal__actions")) {
      const actions = document.createElement("div");
      actions.className = "terminal__actions";
      actions.style.cssText = "margin-left: auto; display: flex; gap: 8px;";
      actions.innerHTML = `
        <button class="btn btn--ghost btn--sm" id="term-copy-btn" title="Salin Log">${ICONS.copy()}</button>
        <button class="btn btn--ghost btn--sm" id="term-clear-btn" title="Bersihkan Log">${ICONS.trash()}</button>
      `;
      header.appendChild(actions);

      actions.querySelector("#term-copy-btn").addEventListener("click", () => this.copyToClipboard());
      actions.querySelector("#term-clear-btn").addEventListener("click", () => this.clear());
    }
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
    
    // Accessibility: new logs are additions
    line.setAttribute("role", "listitem");

    const cleanTag = tag.padEnd(14).replace(/ /g, "&nbsp;");

    line.innerHTML = `
      <span class="log-time">${logTimestamp()}</span>
      <span class="log-tag ${tagClass}">${cleanTag}</span>
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

  /** Copy all logs to clipboard as plain text. */
  async copyToClipboard() {
    const text = Array.from(this.bodyEl.querySelectorAll(".log-line"))
      .map((line) => {
        const time = line.querySelector(".log-time").textContent;
        const tag = line.querySelector(".log-tag").textContent.trim();
        const msg = line.querySelector(".log-msg").textContent;
        return `[${time}] ${tag}: ${msg}`;
      })
      .join("\n");

    try {
      await navigator.clipboard.writeText(text);
      // We don't have direct access to toast here, but we can use a native alert or custom event
      const event = new CustomEvent("toast", { 
        detail: { message: "Log berhasil disalin ke clipboard", type: "success" } 
      });
      document.dispatchEvent(event);
    } catch (err) {
      console.error("Failed to copy logs:", err);
    }
  }

  /** Add a separator line. */
  addSeparator() {
    if (!this.bodyEl) return;
    const sep = document.createElement("div");
    sep.className = "terminal__separator";
    sep.style.cssText = `
      border-top: 1px dashed var(--color-border);
      margin: 8px 0;
      opacity: 0.3;
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

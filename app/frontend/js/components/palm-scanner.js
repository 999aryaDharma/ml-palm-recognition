// ============================================================
// js/components/palm-scanner.js — PalmScanner component
// Reusable for all 4 demo modules + user detail test
// ============================================================

import { WebcamCapture } from "./webcam.js";
import { identify } from "../api/identify.js";
import { QUALITY_HINTS, logTimestamp } from "../utils.js";

/**
 * PalmScanner wraps webcam + polling identification loop.
 *
 * HTML structure expected:
 *   <div class="scanner" id="scanner">
 *     <video id="scanner-video" autoplay muted playsinline></video>
 *     <div class="scanner__target">...</div>
 *     <div class="scanner__scanline"></div>
 *     <div class="scanner__hint" id="scanner-hint">Tunjukkan telapak tangan</div>
 *     <div class="scanner__processing"><div class="spinner"></div></div>
 *     <div class="scanner__result" id="scanner-result"></div>
 *     <div class="scanner__placeholder" id="scanner-placeholder">...</div>
 *   </div>
 *
 * Usage:
 *   const scanner = new PalmScanner({
 *     containerEl: document.getElementById('scanner'),
 *     videoEl:     document.getElementById('scanner-video'),
 *     hintEl:      document.getElementById('scanner-hint'),
 *     resultEl:    document.getElementById('scanner-result'),
 *     logFn:       (tag, msg) => terminal.addLog(tag, msg),
 *     onIdentified: (user, score, latency) => { ... },
 *     onUnknown:    (score) => { ... },
 *   });
 *   await scanner.start();
 */
export class PalmScanner {
  constructor(opts = {}) {
    this.containerEl = opts.containerEl;
    this.videoEl = opts.videoEl;
    this.hintEl = opts.hintEl;
    this.resultEl = opts.resultEl;
    this.placeholderEl = opts.placeholderEl;

    this.onIdentified = opts.onIdentified || (() => {});
    this.onUnknown = opts.onUnknown || (() => {});
    this.onError = opts.onError || (() => {});
    this.logFn = opts.logFn || (() => {});

    this.autoResetMs = opts.autoResetMs ?? 4000;
    this.captureIntervalMs = opts.captureIntervalMs ?? 1500;

    this.isProcessing = false;
    this._resetTimer = null;

    this._webcam = new WebcamCapture(this.videoEl, {
      captureInterval: this.captureIntervalMs,
      onCapture: (blob) => this._handleCapture(blob),
    });
  }

  /** Start camera and identification loop. */
  async start() {
    this._showPlaceholder(false);
    this._setHint("🎥 Menyalakan kamera...");

    try {
      await this._webcam.start();
    } catch (err) {
      this._setHint(QUALITY_HINTS.camera_permission_denied, "error");
      this._setState("error");
      this.logFn("ERROR", "Camera access denied: " + err.message);
      return;
    }

    this._setState("camera-ready");
    this._setHint("🖐 Tunjukkan telapak tangan ke kamera");
    this.logFn(
      "CAMERA_READY",
      `Stream ${this.videoEl.videoWidth}x${this.videoEl.videoHeight} initialized`,
    );
    this._webcam.startAutoCapture();
    this._setState("scanning");
  }

  /** Stop scanner. */
  stop() {
    clearTimeout(this._resetTimer);
    this._webcam.stop();
    this._setState("idle");
  }

  /** Resume scanning after a result. */
  resume() {
    clearTimeout(this._resetTimer);
    this._hideResult();
    if (!this._webcam.isRunning) return;
    this._webcam.startAutoCapture();
    this._setState("scanning");
    this._setHint("🖐 Tunjukkan telapak tangan ke kamera");
  }

  // ── Internal ─────────────────────────────────────────────

  async _handleCapture(blob) {
    if (this.isProcessing) return;
    this.isProcessing = true;

    this._setState("processing");
    this.logFn("CAPTURE", "Frame captured, sending to backend...");

    try {
      const result = await identify(blob);

      if (result.status === "identified") {
        this.logFn("DETECTION", "21 landmarks detected");
        this.logFn("ROI_EXTRACTION", "Palm ROI normalized to 112×112");
        this.logFn("EMBEDDING", "128-d vector generated");
        this.logFn(
          "MATCHING",
          `Top score ${result.score.toFixed(4)} / threshold match`,
        );
        this.logFn("RESULT", `IDENTIFIED: ${result.user.name}`);

        this._webcam.stopAutoCapture();
        this._setState("identified");
        this._showResult(
          "success",
          result.user,
          result.score,
          result.latency_ms,
        );
        this.onIdentified(result.user, result.score, result.latency_ms);

        this._resetTimer = setTimeout(() => this.resume(), this.autoResetMs);
      } else {
        // Unknown
        this.logFn("RESULT", `UNKNOWN — score ${result.score.toFixed(4)}`);
        this._setHint("⚠️ Pengguna tidak dikenali. Coba scan ulang", "warning");
        this.onUnknown(result.score);
        // Don't stop loop; continue scanning
        this._setState("scanning");
      }
    } catch (err) {
      const hint = QUALITY_HINTS[err.error] || "⚠️ Coba scan ulang";
      this._setHint(hint, err.error ? "" : "error");
      this.logFn("ERROR", err.error || err.message || "Unknown error");
      this._setState("scanning"); // Continue loop; don't block on quality errors
      this.onError(err);
    } finally {
      this.isProcessing = false;
    }
  }

  _setState(state) {
    if (!this.containerEl) return;
    // Remove all state classes
    this.containerEl.classList.remove(
      "scanner--scanning",
      "scanner--processing",
      "scanner--identified",
      "scanner--error",
      "scanner--idle",
    );
    if (state !== "idle") this.containerEl.classList.add(`scanner--${state}`);
  }

  _setHint(text, type = "") {
    if (!this.hintEl) return;
    this.hintEl.textContent = text;
    this.hintEl.className = "scanner__hint";
    if (type) this.hintEl.classList.add(`hint--${type}`);
  }

  _showResult(type, user, score, latency) {
    if (!this.resultEl) return;
    const isSuccess = type === "success";
    this.resultEl.className = `scanner__result scanner__result--${isSuccess ? "success" : "unknown"} is-visible`;
    this.resultEl.innerHTML = `
      <div class="result-icon">${isSuccess ? "✅" : "⚠️"}</div>
      <div class="result-name">${isSuccess ? user.name : "Tidak Dikenali"}</div>
      <div class="result-score">${
        isSuccess
          ? `Score: ${score.toFixed(4)} · ${latency}ms`
          : `Score: ${score.toFixed(4)}`
      }</div>
      <div class="result-action-bar">
        <button class="btn btn--secondary btn--sm" id="result-retry-btn">
          Scan Ulang
        </button>
      </div>
    `;
    this.resultEl
      .querySelector("#result-retry-btn")
      ?.addEventListener("click", () => this.resume(), { once: true });
  }

  _hideResult() {
    if (!this.resultEl) return;
    this.resultEl.classList.remove("is-visible");
    this.resultEl.innerHTML = "";
  }

  _showPlaceholder(show) {
    if (!this.placeholderEl) return;
    this.placeholderEl.style.display = show ? "flex" : "none";
  }
}

// ============================================================
// js/components/palm-scanner.js — PalmScanner component
// Reusable for all 4 demo modules + user detail test
// ============================================================

import { WebcamCapture } from "./webcam.js";
import { identify } from "../api/identify.js";
import { QUALITY_HINTS, logTimestamp } from "../utils.js";
import { ICONS } from "../icons.js";
import { PalmBoundingBox } from "./palm-bounding-box.js";

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
    this.processingLabelEl = opts.processingLabelEl;

    this.onIdentified = opts.onIdentified || (() => {});
    this.onUnknown = opts.onUnknown || (() => {});
    this.onError = opts.onError || (() => {});
    this.logFn = opts.logFn || (() => {});

    this.autoResetMs = opts.autoResetMs ?? 4000;
    this.captureIntervalMs = opts.captureIntervalMs ?? 1500;

    // New options from UI/UX report
    this.autoResumeOnIdentified = opts.autoResumeOnIdentified ?? true;
    this.pauseOnUnknown = opts.pauseOnUnknown ?? true;

    this.isProcessing = false;
    this._resetTimer = null;

    // Initialize PalmBoundingBox
    this.boundingBox =
      opts.boundingBox ||
      new PalmBoundingBox({
        containerEl: this.containerEl,
        layerEl: opts.boundingBoxLayerEl,
        logFn: this.logFn,
      });

    this._webcam = new WebcamCapture(this.videoEl, {
      captureInterval: this.captureIntervalMs,
      onCapture: (blob) => this._handleCapture(blob),
    });
  }

  /**
   * Start camera and identification loop.
   * @returns {Promise<boolean>} Success status
   */
  async start() {
    this._showPlaceholder(false);
    this._setState("requesting-camera");
    this._setHint("🎥 Menyalakan kamera...");

    try {
      await this._webcam.start();
    } catch (err) {
      this._setHint(QUALITY_HINTS.camera_permission_denied, "error");
      this._setState("error");
      this._showResult({
        type: "error",
        title: "Kamera Tidak Dapat Diakses",
        detail: "Berikan izin kamera di browser, lalu coba lagi.",
        primaryActionLabel: "Coba Lagi",
      });
      this.logFn("ERROR", "Camera access denied: " + err.message);
      this.onError(err);
      return false;
    }

    this._setState("camera-ready");
    this._setHint("🖐 Tunjukkan telapak tangan ke kamera");
    this.logFn(
      "CAMERA_READY",
      `Stream ${this.videoEl.videoWidth}x${this.videoEl.videoHeight} initialized`,
    );
    this._webcam.startAutoCapture();
    this._setState("scanning");
    this.boundingBox.searching();
    return true;
  }

  /** Stop scanner completely. */
  stop() {
    clearTimeout(this._resetTimer);
    this._webcam.stop();
    this._setState("idle");
    this._hideResult();
    this.boundingBox.clear();
  }

  /** Pause scanning loop without stopping webcam. */
  pause() {
    clearTimeout(this._resetTimer);
    this._webcam.stopAutoCapture();
    this._setState("paused");
  }

  /** Resume scanning after a result or pause. */
  resume() {
    clearTimeout(this._resetTimer);
    this._hideResult();
    if (!this._webcam || !this._webcam.isRunning) return;
    this._webcam.startAutoCapture();
    this._setState("scanning");
    this._setHint("🖐 Tunjukkan telapak tangan ke kamera");
    this.boundingBox.searching();
  }

  // ── Internal ─────────────────────────────────────────────

  async _handleCapture(blob) {
    if (this.isProcessing) return;
    this.isProcessing = true;

    this._setState("processing");
    this._setHint("🔍 Mendeteksi telapak tangan...", "info");
    if (this.processingLabelEl)
      this.processingLabelEl.textContent = "Mendeteksi...";
    this.logFn("CAPTURE", "Frame captured, sending to backend...");

    try {
      const result = await identify(blob);

      // Immediate feedback: Update bounding box as soon as we have detection data
      if (result.bbox) {
        this.boundingBox.update({
          bbox: result.bbox,
          landmarks: result.landmarks || [],
          quality: result.quality_score,
          score: result.score,
          label: result.status === "identified" ? "PALM MATCHED" : "PALM DETECTED",
          state: result.status === "identified" ? "matched" : "detected",
        });
        this.containerEl?.classList.add("scanner--detected");
        this.containerEl?.classList.remove("scanner--no-palm");
      }

      // Granular feedback loop for UX transparency
      this._setHint("📐 Mengekstrak ROI...", "info");
      if (this.processingLabelEl) this.processingLabelEl.textContent = "Mengekstrak...";
      await new Promise((r) => setTimeout(r, 200));

      this._setHint("🔢 Mencocokkan biometrik...", "info");
      if (this.processingLabelEl)
        this.processingLabelEl.textContent = "Mencocokkan...";
      await new Promise((r) => setTimeout(r, 200));

      if (result.status === "identified") {
        this.logFn("DETECTION", "21 landmarks detected");
        this.logFn("ROI_EXTRACTION", "Palm ROI normalized to 112×112");
        this.logFn("EMBEDDING", "128-d vector generated");
        this.logFn(
          "MATCHING",
          `Top score ${result.score.toFixed(4)} / threshold match`,
        );
        this.logFn("RESULT", `IDENTIFIED: ${result.user.name}`);

        if (!this.autoResumeOnIdentified) {
          this._webcam.stopAutoCapture();
        }
        
        this._setState("identified");
        this._setHint("✅ Identitas Terdeteksi", "success");

        this._showResult({
          type: "success",
          title: "Identitas Terdeteksi",
          name: result.user.name,
          score: result.score,
          latency: result.latency_ms,
          primaryActionLabel: "Lanjutkan",
        });

        this.onIdentified(result.user, result.score, result.latency_ms);

        if (this.autoResumeOnIdentified) {
          // Hanya sembunyikan result overlay setelah beberapa saat,
          // Kamera TETAP lanjut mendeteksi di background tanpa jeda.
          this._resetTimer = setTimeout(() => {
              this._hideResult();
              this._setState("scanning");
              this._setHint("🖐 Tunjukkan telapak tangan ke kamera");
              this.boundingBox.searching();
          }, this.autoResetMs);
        }
      } else {
        // Unknown
        this.logFn("RESULT", `UNKNOWN — score ${result.score.toFixed(4)}`);

        if (this.pauseOnUnknown) {
          this._webcam.stopAutoCapture();
          this._setState("unknown");
          this._setHint("⚠️ Pengguna tidak dikenali", "warning");
          this._showResult({
            type: "unknown",
            title: "Pengguna Tidak Dikenali",
            detail: "Template biometric tidak cocok dengan data terdaftar.",
            score: result.score,
            latency: result.latency_ms,
            primaryActionLabel: "Scan Ulang",
          });
        } else {
          this._setHint(
            "⚠️ Pengguna tidak dikenali. Coba scan ulang",
            "warning",
          );
          this._setState("scanning");
        }

        this.onUnknown(result.score);
      }
    } catch (err) {
      // Hand not detected or other quality error
      this.containerEl?.classList.remove("scanner--detected");
      
      // Update bounding box if data is available in error detail
      const errorDetail = err.detail || err;
      if (errorDetail.bbox) {
         this.boundingBox.update({
            bbox: errorDetail.bbox,
            landmarks: errorDetail.landmarks || [],
            quality: errorDetail.quality_score,
            label: "QUALITY LOW",
            state: "quality-low",
         });
         this.containerEl?.classList.add("scanner--detected");
      } else if (
        errorDetail.error === "no_hand_detected" ||
        errorDetail.error === "detection_failed"
      ) {
        this.containerEl?.classList.add("scanner--no-palm");
        this.boundingBox.lost("NO PALM");
      } else {
        this.boundingBox.searching();
      }

      const hint = QUALITY_HINTS[errorDetail.error] || "⚠️ Coba scan ulang";
      this._setHint(hint, errorDetail.error ? "" : "error");

      // ... rest same
      if (err.status !== 400 || !errorDetail.error) {
        this.logFn("ERROR", errorDetail.error || err.message || "Unknown error");
        console.error("[PalmScanner] Unhandled Error:", err);
      } else {
        this.logFn("QUALITY_GATE", errorDetail.message || hint);
      }

      this._setState("scanning");
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
      "scanner--unknown",
      "scanner--error",
      "scanner--idle",
      "scanner--paused",
      "scanner--requesting-camera",
      "scanner--camera-ready",
      "scanner--no-palm",
    );

    // Only remove "scanner--detected" if we are going back to idle, error, or specifically told to
    if (state === "idle" || state === "error" || state === "camera-ready") {
      this.containerEl.classList.remove("scanner--detected");
    }

    if (state !== "idle") this.containerEl.classList.add(`scanner--${state}`);
  }

  _setHint(text, type = "") {
    if (!this.hintEl) return;
    this.hintEl.textContent = text;
    this.hintEl.className = "scanner__hint";
    if (type) this.hintEl.classList.add(`hint--${type}`);

    // Accessibility: hints are status updates
    this.hintEl.setAttribute("role", "status");
    this.hintEl.setAttribute("aria-live", "polite");
  }

  /**
   * Show result overlay.
   * @param {Object} opts result options
   * @param {string} opts.type - 'success', 'unknown', or 'error'
   * @param {string} opts.title - result title
   * @param {string} [opts.name] - user name (for success)
   * @param {string} [opts.detail] - result detail text
   * @param {number} [opts.score] - match score
   * @param {number} [opts.latency] - match latency
   * @param {string} [opts.primaryActionLabel] - button text
   */
  _showResult(opts) {
    if (!this.resultEl) return;

    const {
      type,
      title,
      name,
      detail,
      score,
      latency,
      primaryActionLabel = "Scan Ulang",
    } = opts;
    const isSuccess = type === "success";

    this.resultEl.className = `scanner__result scanner__result--${type} is-visible`;

    // Accessibility
    this.resultEl.setAttribute(
      "role",
      type === "error" || type === "unknown" ? "alert" : "status",
    );
    this.resultEl.setAttribute(
      "aria-live",
      type === "error" || type === "unknown" ? "assertive" : "polite",
    );

    let icon = ICONS.alertTriangle(48);
    if (type === "success") icon = ICONS.checkCircle(48);
    if (type === "error") icon = ICONS.alertCircle(48);

    const nameHtml = name ? `<div class="result-name">${name}</div>` : "";
    const titleHtml = title ? `<div class="result-title">${title}</div>` : "";
    const detailHtml = detail
      ? `<div class="result-detail">${detail}</div>`
      : "";
    const scoreText = score ? `Score: ${score.toFixed(4)}` : "";
    const latencyText = latency ? ` · ${latency}ms` : "";
    const statsHtml =
      score || latency
        ? `<div class="result-score">${scoreText}${latencyText}</div>`
        : "";

    this.resultEl.innerHTML = `
      <div class="result-icon">${icon}</div>
      ${titleHtml}
      ${nameHtml}
      ${detailHtml}
      ${statsHtml}
      <div class="result-action-bar">
        <button class="btn btn--secondary btn--sm" id="result-retry-btn">
          ${primaryActionLabel}
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
    this.resultEl.removeAttribute("role");
    this.resultEl.removeAttribute("aria-live");
  }

  _showPlaceholder(show) {
    if (!this.placeholderEl) return;
    this.placeholderEl.style.display = show ? "flex" : "none";
  }
}

// ============================================================
// js/components/webcam.js — WebcamCapture class
// Phase F2 deliverable
// ============================================================

/**
 * WebcamCapture — manages webcam stream and frame capture.
 *
 * Usage:
 *   const cam = new WebcamCapture(videoEl, { onCapture: async (blob) => { ... } });
 *   await cam.start();
 *   cam.startAutoCapture();
 *   // later:
 *   cam.stop();
 */
export class WebcamCapture {
  /**
   * @param {HTMLVideoElement} videoEl
   * @param {object} opts
   * @param {(blob: Blob) => void|Promise<void>} [opts.onCapture]
   * @param {number} [opts.captureInterval] ms between auto-captures (default 1500)
   * @param {number} [opts.width]  preferred webcam width
   * @param {number} [opts.height] preferred webcam height
   */
  constructor(videoEl, opts = {}) {
    this.videoEl = videoEl;
    this.onCapture = opts.onCapture || (() => {});
    this.captureInterval = opts.captureInterval ?? 1500;
    this.width = opts.width ?? 640;
    this.height = opts.height ?? 480;

    this.stream = null;
    this._timer = null;
    this._canvas = document.createElement("canvas");
  }

  /** Start webcam stream. Throws if permission denied or no camera. */
  async start() {
    if (this.stream) return; // already running

    const constraints = {
      video: {
        width: { ideal: this.width },
        height: { ideal: this.height },
        facingMode: "user",
        frameRate: { ideal: 30 },
      },
      audio: false,
    };

    try {
      console.log("[WebcamCapture] Requesting camera with constraints:", constraints);
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (err) {
      console.warn("[WebcamCapture] Preferred constraints failed, trying fallback...", err);
      // Fallback: minimal constraints
      this.stream = await navigator.mediaDevices.getUserMedia({ 
        video: true, 
        audio: false 
      });
    }

    this.videoEl.srcObject = this.stream;
    this.videoEl.setAttribute("playsinline", ""); // iOS fix
    this.videoEl.muted = true; // Extra safety for autoplay

    await new Promise((resolve, reject) => {
      const onLoaded = () => {
        console.log(`[WebcamCapture] Video ready: ${this.videoEl.videoWidth}x${this.videoEl.videoHeight}`);
        resolve();
      };
      
      if (this.videoEl.readyState >= 2) {
        return onLoaded();
      }
      this.videoEl.addEventListener("loadeddata", onLoaded, { once: true });
      this.videoEl.addEventListener("error", (e) => reject(new Error("Video element error")), { once: true });
      
      // Fallback timeout in case event doesn't fire
      setTimeout(onLoaded, 3000);
    });

    try {
      await this.videoEl.play();
      console.log("[WebcamCapture] Video playing");
    } catch (err) {
      console.warn("[WebcamCapture] Video play failed, might need user interaction:", err);
    }
  }

  /** Capture one frame as a JPEG Blob. Returns null if video not ready. */
  captureFrame(quality = 0.85) {
    const video = this.videoEl;
    if (!video || video.readyState < 2) return Promise.resolve(null);

    const w = video.videoWidth || this.width;
    const h = video.videoHeight || this.height;

    this._canvas.width = w;
    this._canvas.height = h;

    const ctx = this._canvas.getContext("2d");
    // Mirror to match video display (canvas captures pre-mirror)
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, w, h);
    ctx.restore();

    return new Promise((resolve) =>
      this._canvas.toBlob(resolve, "image/jpeg", quality),
    );
  }

  /** Start automatic capture loop. Calls onCapture(blob) each interval. */
  startAutoCapture() {
    if (this._timer) return;
    this._timer = setInterval(async () => {
      const blob = await this.captureFrame();
      if (blob) this.onCapture(blob);
    }, this.captureInterval);
  }

  /** Stop automatic capture loop without stopping the stream. */
  stopAutoCapture() {
    clearInterval(this._timer);
    this._timer = null;
  }

  /** Stop everything and release camera. */
  stop() {
    this.stopAutoCapture();
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
    this.videoEl.srcObject = null;
  }

  /** True if camera stream is active. */
  get isRunning() {
    return !!(this.stream && this.stream.active);
  }
}

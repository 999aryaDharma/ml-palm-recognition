// ============================================================
// js/components/palm-bounding-box.js
// Shared visual overlay for palm detection bounding box
//
// BUG FIXES v2:
//   1. ReferenceError: "const cornerMarkup = cornerMarkup(box)" — nama
//      variabel lokal bentrok dengan nama fungsi module. Diganti ke
//      _buildCorners(box).
//   2. CSS opacity: layer awal opacity:0, hanya muncul jika container
//      memiliki class scanner--palm-*. State "searching" sekarang juga
//      di-apply saat scanner pertama aktif (sudah ada), dan state class
//      di containerEl dipertahankan saat _setState() dipanggil.
//   3. Landmark coordinates: landmarks dari backend sudah normalized
//      (0-1), tinggal kali 100 untuk viewBox 0-100. Pastikan tidak
//      double-normalize.
// ============================================================

export class PalmBoundingBox {
  constructor({ containerEl, layerEl, logFn = () => {} } = {}) {
    this.containerEl = containerEl;
    this.layerEl = layerEl;
    this.logFn = logFn;
    this.lastBox = null;
    this.state = "idle";

    this._ensureLayer();
    // BUG FIX: Jangan panggil clear() di constructor karena akan
    // menghapus class dari containerEl sebelum scanner start.
    // Layer SVG sudah kosong saat baru dibuat.
    if (this.layerEl) {
      this.layerEl.innerHTML = "";
    }
  }

  _ensureLayer() {
    if (this.layerEl) return;
    if (!this.containerEl) return;

    this.layerEl = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "svg",
    );
    this.layerEl.classList.add("palm-box-layer");
    this.layerEl.setAttribute("viewBox", "0 0 100 100");
    this.layerEl.setAttribute("preserveAspectRatio", "none");
    this.layerEl.setAttribute("aria-hidden", "true");

    const video = this.containerEl.querySelector("video");
    if (video?.nextSibling) {
      this.containerEl.insertBefore(this.layerEl, video.nextSibling);
    } else {
      this.containerEl.prepend(this.layerEl);
    }
  }

  setState(state) {
    this.state = state;
    if (!this.containerEl) return;

    this.containerEl.classList.remove(
      "scanner--palm-searching",
      "scanner--palm-detected",
      "scanner--palm-quality-low",
      "scanner--palm-matched",
      "scanner--palm-lost",
    );

    if (state && state !== "idle") {
      this.containerEl.classList.add(`scanner--palm-${state}`);
    }
  }

  clear() {
    if (!this.layerEl) return;
    this.lastBox = null;
    this.setState("idle");
    this.layerEl.innerHTML = "";
  }

  searching() {
    this.setState("searching");
    if (!this.layerEl) return;

    this.layerEl.innerHTML = `
      <g class="palm-box__searching">
        <rect class="palm-box__scanner-area" x="25" y="20" width="50" height="60" rx="2" />
        <text class="palm-box__label" x="50" y="95" text-anchor="middle">SEARCHING PALM</text>
      </g>
    `;
  }

  lost(reason = "NO PALM") {
    this.setState("lost");
    if (!this.layerEl) return;

    this.layerEl.innerHTML = `
      <g class="palm-box__lost">
        <rect class="palm-box__scanner-area" x="25" y="20" width="50" height="60" rx="2" />
        <text class="palm-box__label" x="50" y="95" text-anchor="middle">${_escapeSvgText(reason)}</text>
      </g>
    `;
  }

  update({
    bbox,
    landmarks = [],
    quality = null,
    score = null,
    label = "PALM",
    state = "detected",
  } = {}) {
    if (!this.layerEl || !bbox) {
      this.lost("NO PALM");
      return;
    }

    const box = _normalizeBox(bbox);
    
    // FLIP THE X COORDINATES VISUALLY
    // Because the video element is mirrored via CSS (`transform: scaleX(-1)`), 
    // but the backend calculates coordinates on the raw, unmirrored image,
    // we must horizontally flip the bounding box coordinates so they visually
    // align with the mirrored video on screen.
    box.x = 100 - box.x - box.width;

    this.lastBox = box;

    const qualityNumber = Number(quality);
    const hasQuality = Number.isFinite(qualityNumber);
    const qualityLabel = hasQuality
      ? `Q:${Math.round(qualityNumber * 100)}%`
      : "";

    const scoreNumber = Number(score);
    const hasScore = Number.isFinite(scoreNumber);
    const scoreLabel = hasScore ? `S:${scoreNumber.toFixed(3)}` : "";

    const finalState =
      state === "matched"
        ? "matched"
        : hasQuality && qualityNumber < 0.75
          ? "quality-low"
          : "detected";

    const cornersHtml = _buildCorners(box);

    const landmarkDots = landmarks
      .map((point) => {
        // Flip landmark X visually as well
        const x = 100 - (_clamp01(point.x) * 100);
        const y = _clamp01(point.y) * 100;
        return `<circle class="palm-box__dot" cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="0.6" />`;
      })
      .join("");

    const displayLabel = `${_escapeSvgText(label)} ${qualityLabel} ${scoreLabel}`.trim();

    this.layerEl.innerHTML = `
      <g class="palm-box__group">
        <rect class="palm-box__rect"
          x="${box.x.toFixed(2)}" y="${box.y.toFixed(2)}"
          width="${box.width.toFixed(2)}" height="${box.height.toFixed(2)}"
          rx="1" />

        <rect class="palm-box__scanline"
          x="${box.x.toFixed(2)}" y="${box.y.toFixed(2)}"
          width="${box.width.toFixed(2)}" height="0.5"
          rx="0.25" />

        ${cornersHtml}

        <g class="palm-box__label-group" transform="translate(${(box.x + box.width / 2).toFixed(2)}, ${Math.max(4, box.y - 3).toFixed(2)})">
           <rect class="palm-box__label-bg" x="-15" y="-3" width="30" height="4.5" rx="0.5" />
           <text class="palm-box__label" text-anchor="middle" y="0.5">${displayLabel}</text>
        </g>

        <g class="palm-box__landmarks">${landmarkDots}</g>
      </g>
    `;

    this.setState(finalState);
  }
}

// ── Private module-level helpers (prefix _ agar tidak bentrok) ────────────

function _normalizeBox(bbox) {
  // Terima bbox normalized (0-1) atau persen (0-100)
  // Deteksi: jika semua nilai <= 1.0, anggap normalized
  const isNormalized =
    bbox.x <= 1 && bbox.y <= 1 && bbox.width <= 1 && bbox.height <= 1;

  return {
    x: isNormalized ? bbox.x * 100 : bbox.x,
    y: isNormalized ? bbox.y * 100 : bbox.y,
    width: isNormalized ? bbox.width * 100 : bbox.width,
    height: isNormalized ? bbox.height * 100 : bbox.height,
  };
}

// BUG FIX 1: Fungsi ini sebelumnya bernama "cornerMarkup" sama dengan
// variabel yang menampung hasilnya di update(), menyebabkan ReferenceError.
function _buildCorners(box) {
  const { x, y, width, height } = box;
  const s = 4; // corner size

  return `
    <g class="palm-box__corners">
      <line class="palm-box__corner" x1="${x}"       y1="${y + s}"      x2="${x}"       y2="${y}" />
      <line class="palm-box__corner" x1="${x}"       y1="${y}"          x2="${x + s}"   y2="${y}" />

      <line class="palm-box__corner" x1="${x + width - s}" y1="${y}"    x2="${x + width}" y2="${y}" />
      <line class="palm-box__corner" x1="${x + width}"     y1="${y}"    x2="${x + width}" y2="${y + s}" />

      <line class="palm-box__corner" x1="${x}"       y1="${y + height - s}" x2="${x}"       y2="${y + height}" />
      <line class="palm-box__corner" x1="${x}"       y1="${y + height}"     x2="${x + s}"   y2="${y + height}" />

      <line class="palm-box__corner" x1="${x + width - s}" y1="${y + height}" x2="${x + width}" y2="${y + height}" />
      <line class="palm-box__corner" x1="${x + width}"     y1="${y + height - s}" x2="${x + width}" y2="${y + height}" />
    </g>
  `;
}

function _clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function _escapeSvgText(value) {
  return String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;",
      })[c],
  );
}

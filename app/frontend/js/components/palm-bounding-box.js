// ============================================================
// js/components/palm-bounding-box.js
// Shared visual overlay for palm detection bounding box
// ============================================================

export class PalmBoundingBox {
  constructor({ containerEl, layerEl, logFn = () => {} } = {}) {
    this.containerEl = containerEl;
    this.layerEl = layerEl;
    this.logFn = logFn;
    this.lastBox = null;
    this.state = "idle";

    this._ensureLayer();
    this.clear();
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
        <text class="palm-box__label" x="50" y="95" text-anchor="middle">
          SEARCHING PALM
        </text>
      </g>
    `;
  }

  lost(reason = "NO PALM") {
    this.setState("lost");

    if (!this.layerEl) return;

    this.layerEl.innerHTML = `
      <g class="palm-box__lost">
        <rect class="palm-box__scanner-area" x="25" y="20" width="50" height="60" rx="2" />
        <text class="palm-box__label" x="50" y="95" text-anchor="middle">
          ${escapeSvgText(reason)}
        </text>
      </g>
    `;
  }

  update({
    bbox,
    landmarks = [],
    quality = null,
    label = "PALM",
    state = "detected",
  } = {}) {
    if (!this.layerEl || !bbox) {
      this.lost("NO PALM");
      return;
    }

    const box = normalizeBox(bbox);
    this.lastBox = box;

    const qualityNumber = Number(quality);
    const hasQuality = Number.isFinite(qualityNumber);
    const qualityLabel = hasQuality
      ? `${Math.round(qualityNumber * 100)}%`
      : "";

    const finalState =
      state === "matched"
        ? "matched"
        : hasQuality && qualityNumber < 0.75
          ? "quality-low"
          : "detected";

    const cornerMarkup = cornerMarkup(box);

    const landmarkMarkup = landmarks
      .map((point) => {
        const x = clamp01(point.x) * 100;
        const y = clamp01(point.y) * 100;
        return `<circle class="palm-box__dot" cx="${x}" cy="${y}" r="0.8" />`;
      })
      .join("");

    this.layerEl.innerHTML = `
      <g class="palm-box__group">
        <rect class="palm-box__rect" x="${box.x}" y="${box.y}" width="${box.width}" height="${box.height}" rx="1" />

        ${cornerMarkup}

        <text class="palm-box__label" x="${box.x + box.width / 2}" y="${box.y - 2}" text-anchor="middle">
          ${escapeSvgText(label)} ${qualityLabel}
        </text>

        <g class="palm-box__landmarks">
          ${landmarkMarkup}
        </g>
      </g>
    `;

    this.setState(finalState);
  }
}

function normalizeBox(bbox) {
  // Support normalized box: {x:0.2,y:0.1,width:0.5,height:0.6}
  // Support percent box:     {x:20,y:10,width:50,height:60}
  const isNormalized =
    bbox.x <= 1 && bbox.y <= 1 && bbox.width <= 1 && bbox.height <= 1;

  const x = isNormalized ? bbox.x * 100 : bbox.x;
  const y = isNormalized ? bbox.y * 100 : bbox.y;
  const width = isNormalized ? bbox.width * 100 : bbox.width;
  const height = isNormalized ? bbox.height * 100 : bbox.height;

  return { x, y, width, height };
}

function cornerMarkup(box) {
  const { x, y, width, height } = box;
  const cornerSize = 4;

  return `
    <g class="palm-box__corners">
      <!-- Top-left -->
      <line class="palm-box__corner" x1="${x}" y1="${y + cornerSize}" x2="${x}" y2="${y}" />
      <line class="palm-box__corner" x1="${x}" y1="${y}" x2="${x + cornerSize}" y2="${y}" />

      <!-- Top-right -->
      <line class="palm-box__corner" x1="${x + width - cornerSize}" y1="${y}" x2="${x + width}" y2="${y}" />
      <line class="palm-box__corner" x1="${x + width}" y1="${y}" x2="${x + width}" y2="${y + cornerSize}" />

      <!-- Bottom-left -->
      <line class="palm-box__corner" x1="${x}" y1="${y + height - cornerSize}" x2="${x}" y2="${y + height}" />
      <line class="palm-box__corner" x1="${x}" y1="${y + height}" x2="${x + cornerSize}" y2="${y + height}" />

      <!-- Bottom-right -->
      <line class="palm-box__corner" x1="${x + width - cornerSize}" y1="${y + height}" x2="${x + width}" y2="${y + height}" />
      <line class="palm-box__corner" x1="${x + width}" y1="${y + height - cornerSize}" x2="${x + width}" y2="${y + height}" />
    </g>
  `;
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function escapeSvgText(value) {
  return String(value ?? "").replace(
    /[&<>"']/g,
    (char) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;",
      })[char],
  );
}

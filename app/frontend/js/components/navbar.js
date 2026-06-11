// ============================================================
// js/components/navbar.js — Shared navbar renderer
// ============================================================

import { checkHealth } from "../api/client.js";

const NAV_ITEMS = [
  { href: "index.html", label: "Home", icon: svgHome() },
  { href: "enroll.html", label: "Enrollment", icon: svgEnroll() },
  { href: "users.html", label: "Users", icon: svgUsers() },
  {
    label: "Demo",
    icon: svgDemo(),
    children: [
      { href: "demo/payment.html", label: "💳 Payment" },
      { href: "demo/attendance.html", label: "📋 Absensi" },
      { href: "demo/patient.html", label: "🏥 Patient" },
    ],
  },
  { href: "about.html", label: "About", icon: svgAbout() },
];

/**
 * Mount navbar into .app-navbar element.
 */
export function mountNavbar() {
  try {
    const el = document.querySelector(".app-navbar");
    if (!el) return;

    // Detect project root relative to current path
    // If we are in 'demo/payment.html', path depth is 1.
    const currentPath = window.location.pathname;
    const isSubDir = currentPath.includes("/demo/");
    const rootPrefix = isSubDir ? "../" : "";

    el.innerHTML = `
      <a href="${rootPrefix}index.html" class="navbar-brand" aria-label="PalmID Home">
        <div class="navbar-logo-mark" aria-hidden="true">
          ${svgPalmIcon()}
        </div>
        <div>
          <div class="navbar-brand-name">PalmID</div>
        </div>
      </a>

      <nav class="navbar-nav" aria-label="Main navigation">
        ${NAV_ITEMS.map((item) => renderNavItem(item, currentPath, rootPrefix)).join("")}
      </nav>

      <div class="backend-status backend-status--checking" id="backend-status" aria-live="polite">
        <span class="status-dot status-dot--pulse"></span>
        <span id="backend-status-label">Memeriksa...</span>
      </div>
    `;

    pollBackendStatus().catch(() => {});
    initPageTransitions();
  } catch (err) {
    console.error("mountNavbar failed:", err);
  }
}

function renderNavItem(item, currentPath, rootPrefix) {
  const isMatch = (href) => {
    if (!href) return false;
    // Simple path matching
    return (
      currentPath.endsWith(href) ||
      (currentPath.endsWith("/") && href === "index.html")
    );
  };

  if (item.children) {
    const isActive = item.children.some((c) => isMatch(c.href));
    return `
      <div class="nav-dropdown">
        <button class="nav-link${isActive ? " active" : ""}" aria-haspopup="true">
          ${item.icon}
          <span>${item.label}</span>
          <svg class="dropdown-chevron" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>
        </button>
        <div class="nav-dropdown-menu" role="menu">
          ${item.children
            .map((c) => {
              // If we're already in demo/ and the target is demo/X,
              // we need to adjust the link to just X.html
              let finalHref = rootPrefix + c.href;
              if (
                currentPath.includes("/demo/") &&
                c.href.startsWith("demo/")
              ) {
                finalHref = c.href.replace("demo/", "");
              }

              return `
                <a href="${finalHref}" class="nav-dropdown-item${isMatch(c.href) ? " active" : ""}" role="menuitem">
                  <span>${c.label}</span>
                </a>
              `;
            })
            .join("")}
        </div>
      </div>
    `;
  }

  const isActive = isMatch(item.href);
  return `
    <a href="${rootPrefix}${item.href}" class="nav-link${isActive ? " active" : ""}" aria-current="${isActive ? "page" : "false"}">
      ${item.icon}
      <span>${item.label}</span>
    </a>
  `;
}

async function pollBackendStatus() {
  const statusEl = document.getElementById("backend-status");
  const labelEl = document.getElementById("backend-status-label");
  if (!statusEl || !labelEl) return;

  try {
    const { online } = await checkHealth();

    statusEl.className = `backend-status backend-status--${online ? "online" : "offline"}`;
    const dot = statusEl.querySelector(".status-dot");
    if (dot) {
      dot.className = `status-dot${online ? " status-dot--pulse" : ""}`;
    }
    labelEl.textContent = online ? "Backend Online" : "Backend Offline";

    // Dispatch status to other page components reactively
    document.dispatchEvent(
      new CustomEvent("backend-status-change", { detail: { online } }),
    );
  } catch (err) {
    statusEl.className = "backend-status backend-status--offline";
    labelEl.textContent = "Backend Offline";
    document.dispatchEvent(
      new CustomEvent("backend-status-change", { detail: { online: false } }),
    );
  }

  // Re-poll every 30s instead of 15s to be gentler
  setTimeout(() => pollBackendStatus().catch(() => {}), 30_000);
}

function initPageTransitions() {
  if (typeof window === "undefined") return;

  // Add the ready class on load (ensures smooth fade in)
  requestAnimationFrame(() => {
    document.body.classList.add("ux-page-ready");
  });

  if (window.uxTransitionInitialized) return;
  window.uxTransitionInitialized = true;

  // Create top progress bar if not exists
  let loader = document.getElementById("ux-top-loader");
  if (!loader) {
    loader = document.createElement("div");
    loader.id = "ux-top-loader";
    document.body.appendChild(loader);
  }

  // Intercept anchor clicks for smooth transition
  document.addEventListener("click", (e) => {
    const anchor = e.target.closest("a");
    if (anchor) {
      const href = anchor.getAttribute("href");

      // Skip if not a regular link or target blank
      if (
        !href ||
        href.startsWith("#") ||
        href.startsWith("javascript:") ||
        anchor.target === "_blank" ||
        anchor.hasAttribute("download") ||
        anchor.hasAttribute("data-no-transition")
      ) {
        return;
      }

      // Check origin
      try {
        const url = new URL(anchor.href, window.location.href);
        if (url.origin !== window.location.origin) return;

        e.preventDefault();

        // Start progress loader
        loader.style.width = "40%";
        setTimeout(() => {
          loader.style.width = "80%";
        }, 80);

        // Fade out page body
        document.body.classList.remove("ux-page-ready");

        setTimeout(() => {
          loader.style.width = "100%";
          window.location.href = anchor.href;
        }, 220); // must match opacity transition duration in ux.css
      } catch (err) {
        console.error("Link interception error:", err);
      }
    }
  });
}

// ── SVG Icons ─────────────────────────────────────────────

function svgPalmIcon() {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M18 11V6a2 2 0 0 0-4 0v5"/>
    <path d="M14 10V4a2 2 0 0 0-4 0v6"/>
    <path d="M10 10.5V6a2 2 0 0 0-4 0v8"/>
    <path d="M6 14a4 4 0 0 0 4 4h4a4 4 0 0 0 4-4v-2.5"/>
  </svg>`;
}

function svgHome() {
  return `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>
  </svg>`;
}

function svgEnroll() {
  return `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><line x1="19" y1="8" x2="19" y2="14"/><line x1="22" y1="11" x2="16" y2="11"/>
  </svg>`;
}

function svgUsers() {
  return `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>
  </svg>`;
}

function svgDemo() {
  return `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <polygon points="5 3 19 12 5 21 5 3"/>
  </svg>`;
}

function svgAbout() {
  return `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>
  </svg>`;
}

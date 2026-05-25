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
      { href: "demo/access.html", label: "🚪 Access" },
      { href: "demo/patient.html", label: "🏥 Patient" },
    ],
  },
  { href: "settings.html", label: "Settings", icon: svgSettings() },
];

/**
 * Mount navbar into .app-navbar element.
 * Highlights the active link based on current pathname.
 */
export function mountNavbar() {
  try {
    const el = document.querySelector(".app-navbar");
    if (!el) {
      console.warn("Navbar mount point (.app-navbar) not found.");
      return;
    }

    const currentPath = window.location.pathname;

    el.innerHTML = `
      <a href="index.html" class="navbar-brand" aria-label="Palm Biometric Home">
        <div class="navbar-logo-mark" aria-hidden="true">
          ${svgPalmIcon()}
        </div>
        <div>
          <div class="navbar-brand-name">PalmID</div>
          <div class="navbar-brand-sub">BIOMETRIC v0.5</div>
        </div>
      </a>

      <nav class="navbar-nav" aria-label="Main navigation">
        ${NAV_ITEMS.map((item) => renderNavItem(item, currentPath)).join("")}
      </nav>

      <div class="backend-status backend-status--checking" id="backend-status" aria-live="polite">
        <span class="status-dot status-dot--pulse"></span>
        <span id="backend-status-label">Memeriksa...</span>
      </div>
    `;

    // Initialize dropdown listeners if needed (optional since we use hover in CSS)
    
    // Start background polling
    pollBackendStatus().catch(err => console.error("Initial health check failed:", err));
  } catch (err) {
    console.error("mountNavbar failed:", err);
  }
}

function renderNavItem(item, currentPath) {
  // Normalize paths for matching
  const isMatch = (href) => {
    if (!href) return false;
    // Relative match: ends with the href or just filename
    return currentPath.endsWith(href) || 
           (currentPath === "/" && href === "index.html") ||
           (currentPath.endsWith("/") && href === "index.html");
  };

  if (item.children) {
    const isActive = item.children.some((c) => isMatch(c.href));
    return `
      <div class="nav-dropdown">
        <button class="nav-link${isActive ? " active" : ""}" aria-haspopup="true" aria-expanded="${isActive}">
          ${item.icon}
          <span>${item.label}</span>
        </button>
        <div class="nav-dropdown-menu" role="menu">
          ${item.children
            .map(
              (c) => `
            <a href="${c.href}" class="nav-dropdown-item${isMatch(c.href) ? " active" : ""}" role="menuitem">
              ${c.label}
            </a>
          `,
            )
            .join("")}
        </div>
      </div>
    `;
  }

  const isActive = isMatch(item.href);
  return `
    <a href="${item.href}" class="nav-link${isActive ? " active" : ""}" aria-current="${isActive ? "page" : "false"}">
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
  } catch (err) {
    statusEl.className = "backend-status backend-status--offline";
    labelEl.textContent = "Backend Offline";
  }

  // Re-poll every 30s instead of 15s to be gentler
  setTimeout(() => pollBackendStatus().catch(() => {}), 30_000);
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

function svgSettings() {
  return `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
  </svg>`;
}

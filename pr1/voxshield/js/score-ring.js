/**
 * score-ring.js — VOXSHIELD
 *
 * Manages the SVG circular authenticity score ring:
 *  - Smooth stroke-dashoffset CSS transition animation
 *  - Demo mode: shows illustrative score, clearly labelled
 *  - Live mode: shows WAITING FOR ANALYSIS until a model calls setScore()
 *  - setDemoScore(): show a specific score in demo mode (used by upload + call)
 *  - Public setScore() hook for future real model integration
 */

// ─── Constants ────────────────────────────────────────────────────────────────

/** Must match the SVG circle's r attribute */
const RING_RADIUS = 90;

/** Full circumference: 2π × r ≈ 565.49 */
const RING_CIRCUMFERENCE = 2 * Math.PI * RING_RADIUS;

// ─── State ────────────────────────────────────────────────────────────────────

/** Last score value rendered (0–100) */
let currentScore = 94;

/** Whether demo mode is enabled */
let isDemoMode = true;

/** Cached DOM element references (populated in initScoreRing) */
let elements = {
  /** @type {SVGCircleElement|null} */ progress:  null,
  /** @type {SVGTextElement|null}   */ scoreText: null,
  /** @type {SVGTextElement|null}   */ labelText: null,
  /** @type {HTMLElement|null}      */ demoBadge: null,
  /** @type {HTMLElement|null}      */ demoNote:  null,
  /** @type {HTMLElement|null}      */ ariaDesc:  null,
};

// ─── Initialization ───────────────────────────────────────────────────────────

/**
 * Bind DOM elements and trigger the initial ring animation.
 * Call once, after the DOM is ready.
 */
export function initScoreRing() {
  elements.progress  = document.getElementById('ring-progress');
  elements.scoreText = document.getElementById('ring-score-text');
  elements.labelText = document.getElementById('ring-label-text');
  elements.demoBadge = document.getElementById('demo-badge');
  elements.demoNote  = document.getElementById('demo-note');
  elements.ariaDesc  = document.getElementById('score-aria-desc');

  // Two rAF delay: browser must paint the initial dashoffset before the
  // CSS transition can animate from it.
  requestAnimationFrame(() => {
    requestAnimationFrame(() => setDemoMode(isDemoMode));
  });
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Enable or disable demo mode.
 *
 * ON  → ring shows `currentScore` with "DEMO SCORE" label.
 * OFF → ring empties and shows "WAITING FOR ANALYSIS" until setScore() is called.
 *
 * @param {boolean} enabled
 */
export function setDemoMode(enabled) {
  isDemoMode = enabled;

  if (enabled) {
    _animateRing(currentScore);
    _setLabel('DEMO SCORE');
    _setScoreText(`${Math.round(currentScore)}%`);
    _updateAriaDesc(currentScore, true);
    if (elements.demoBadge) elements.demoBadge.style.display = '';
    if (elements.demoNote)  elements.demoNote.textContent =
      'Showing illustrative scores. Not calculated from audio or an AI model.';
  } else {
    _animateRing(0);
    _setLabel('WAITING FOR ANALYSIS');
    _setScoreText('—');
    _updateAriaDesc(0, false);
    if (elements.demoBadge) elements.demoBadge.style.display = 'none';
    if (elements.demoNote)  elements.demoNote.textContent =
      'Demo mode disabled. Connect an anti-spoofing model to see real scores.';
  }
}

/**
 * Show a specific score value in DEMO mode with a custom label.
 * Used by the File Upload analysis simulation and the Live Call demo ticker.
 * Has no effect if demo mode is off (use setScore() instead).
 *
 * @param {number} score   0–100
 * @param {string} [label] Ring label text
 */
export function setDemoScore(score, label = 'DEMO SCORE') {
  if (!isDemoMode) return;

  currentScore = Math.max(0, Math.min(100, score));
  _animateRing(currentScore);
  _setLabel(label);
  _setScoreText(`${Math.round(currentScore)}%`);
  _updateAriaDesc(currentScore, true);
}

/**
 * Set a real authenticity score from a connected model (0–100).
 * Only has visual effect when demo mode is OFF.
 * ← Hook for future FastAPI / WebSocket integration.
 *
 * @param {number} score
 * @param {string} [label]
 */
export function setScore(score, label = 'AUTHENTICITY SCORE') {
  currentScore = Math.max(0, Math.min(100, score));

  if (!isDemoMode) {
    _animateRing(currentScore);
    _setLabel(label);
    _setScoreText(`${Math.round(currentScore)}%`);
    _updateAriaDesc(currentScore, false);
  }
}

/** @returns {boolean} */
export function getIsDemoMode() { return isDemoMode; }

/** @returns {number} Last score value (0–100) */
export function getCurrentScore() { return currentScore; }

// ─── Private Helpers ──────────────────────────────────────────────────────────

function _animateRing(score) {
  if (!elements.progress) return;
  const pct    = Math.max(0, Math.min(100, score)) / 100;
  const offset = RING_CIRCUMFERENCE * (1 - pct);
  elements.progress.style.strokeDashoffset = offset.toFixed(4);
}

function _setScoreText(text) {
  if (elements.scoreText) elements.scoreText.textContent = text;
}

function _setLabel(label) {
  if (elements.labelText) elements.labelText.textContent = label;
}

function _updateAriaDesc(score, isDemo) {
  if (!elements.ariaDesc) return;
  if (isDemo && score > 0) {
    elements.ariaDesc.textContent =
      `Authenticity score: ${Math.round(score)} percent. Demo score — not from live data.`;
  } else if (score === 0) {
    elements.ariaDesc.textContent = 'Waiting for analysis. No model is connected.';
  } else {
    elements.ariaDesc.textContent = `Authenticity score: ${Math.round(score)} percent.`;
  }
}

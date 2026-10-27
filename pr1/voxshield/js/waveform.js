/**
 * waveform.js — VOXSHIELD
 *
 * Handles:
 *  - DPR-aware Canvas sizing via ResizeObserver
 *  - Live waveform from real microphone AnalyserNode data (bar style, white glow)
 *  - Idle sine-wave animation (visually distinct from live data)
 *  - Switching the render target between canvases (monitor ↔ call tabs)
 *  - One-shot static waveform drawing for uploaded audio files
 *  - Clean resource teardown (cancelAnimationFrame, ResizeObserver.disconnect)
 */

import { getWaveformData, getIsMonitoring } from './audio.js';

// ─── Live Render State ────────────────────────────────────────────────────────

/** @type {HTMLCanvasElement|null} Currently active canvas for the render loop */
let canvas = null;

/** @type {CanvasRenderingContext2D|null} */
let ctx = null;

/** @type {number|null} requestAnimationFrame handle */
let rafId = null;

/** @type {ResizeObserver|null} Observes the canvas container for size changes */
let resizeObserver = null;

/** Phase counter for the idle sine wave animation */
let idlePhase = 0;

/** Whether the render loop is currently running */
let isRunning = false;

// ─── Initialization ───────────────────────────────────────────────────────────

/**
 * Initialize the waveform renderer targeting a specific canvas by ID.
 * Safe to call multiple times — cleanly tears down the previous instance first.
 *
 * @param {string} canvasId
 */
export function initWaveform(canvasId) {
  // Tear down any previous observer / canvas binding
  if (resizeObserver) {
    resizeObserver.disconnect();
    resizeObserver = null;
  }
  canvas = null;
  ctx    = null;

  const el = document.getElementById(canvasId);
  if (!el) {
    console.error(`[waveform] Canvas #${canvasId} not found.`);
    return;
  }

  canvas = el;
  ctx    = canvas.getContext('2d');

  // Watch the container (parent) for layout changes
  const target = canvas.parentElement || canvas;
  resizeObserver = new ResizeObserver(() => resizeCanvas());
  resizeObserver.observe(target);

  resizeCanvas();
}

/**
 * Switch the render loop target to a different canvas.
 * Stops the loop, reinitialises, then restarts.
 *
 * @param {string} canvasId
 */
export function switchCanvas(canvasId) {
  const wasRunning = isRunning;
  stopRenderLoop();
  initWaveform(canvasId);
  if (wasRunning) startRenderLoop();
}

/**
 * DPR-aware canvas resize. Called automatically on container resize.
 */
function resizeCanvas() {
  if (!canvas) return;

  const dpr  = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();

  if (rect.width === 0 || rect.height === 0) return;

  canvas.width  = Math.round(rect.width  * dpr);
  canvas.height = Math.round(rect.height * dpr);

  if (ctx) ctx.scale(dpr, dpr);
}

// ─── Render Loop ──────────────────────────────────────────────────────────────

/**
 * Start the render loop. Idempotent — safe to call when already running.
 */
export function startRenderLoop() {
  if (isRunning) return;
  isRunning = true;
  rafId = requestAnimationFrame(renderFrame);
}

/**
 * Stop the render loop and release the rAF handle.
 */
export function stopRenderLoop() {
  isRunning = false;
  if (rafId !== null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
}

/**
 * Fully destroy the renderer and clean up all resources.
 */
export function destroyWaveform() {
  stopRenderLoop();
  if (resizeObserver) {
    resizeObserver.disconnect();
    resizeObserver = null;
  }
  canvas = null;
  ctx    = null;
}

/** @private Main animation frame callback */
function renderFrame() {
  if (!isRunning) return;

  if (getIsMonitoring()) {
    drawLiveWaveform();
  } else {
    drawIdleWaveform();
  }

  rafId = requestAnimationFrame(renderFrame);
}

// ─── Live Waveform ────────────────────────────────────────────────────────────

/**
 * Draw a bar-style waveform from real AnalyserNode time-domain data.
 * White bars with a soft glow — reacts to actual microphone input.
 */
function drawLiveWaveform() {
  if (!ctx || !canvas) return;

  const data = getWaveformData();

  const dpr = window.devicePixelRatio || 1;
  const w   = canvas.width  / dpr;
  const h   = canvas.height / dpr;

  ctx.clearRect(0, 0, w, h);
  if (!data || data.length === 0) return;

  // Faint center reference line
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();

  const barCount   = Math.min(data.length, 200);
  const barWidth   = w / barCount;
  const barPadding = Math.max(barWidth * 0.15, 0.5);
  const centerY    = h / 2;

  ctx.shadowColor = 'rgba(255,255,255,0.55)';
  ctx.shadowBlur  = 8;
  ctx.fillStyle   = 'rgba(255,255,255,0.72)';

  for (let i = 0; i < barCount; i++) {
    const sample    = data[Math.floor((i / barCount) * data.length)];
    const amplitude = (sample / 128.0) - 1.0; // –1 … 1
    const barH      = Math.max(Math.abs(amplitude) * centerY * 1.6, 1);
    const x         = i * barWidth + barPadding / 2;
    const y         = centerY - barH / 2;
    ctx.fillRect(x, y, barWidth - barPadding, barH);
  }

  ctx.shadowBlur = 0;
}

// ─── Idle Waveform ────────────────────────────────────────────────────────────

/**
 * Draw a subtle, slowly-animating sine wave when the mic is inactive.
 * Intentionally dim and regular — clearly not real audio data.
 */
function drawIdleWaveform() {
  if (!ctx || !canvas) return;

  const dpr = window.devicePixelRatio || 1;
  const w   = canvas.width  / dpr;
  const h   = canvas.height / dpr;

  ctx.clearRect(0, 0, w, h);

  idlePhase += 0.012;

  const amp     = h * 0.06;
  const freq    = 2;
  const centerY = h / 2;

  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth   = 1.5;
  ctx.beginPath();

  for (let x = 0; x <= w; x += 2) {
    const t = (x / w) * Math.PI * 2 * freq + idlePhase;
    const y = centerY + Math.sin(t) * amp;
    x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Faint tick marks
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth   = 1;
  for (let i = 1; i < 10; i++) {
    const x = (i / 10) * w;
    ctx.beginPath();
    ctx.moveTo(x, centerY - 4);
    ctx.lineTo(x, centerY + 4);
    ctx.stroke();
  }
}

// ─── Static File Waveform ─────────────────────────────────────────────────────

/**
 * Draw a one-shot static overview waveform from a decoded AudioBuffer.
 * Used by the File Upload tab to visualise the uploaded audio.
 *
 * Renders the entire waveform (downsampled to canvas width) as a symmetric
 * bar chart — white bars, subtle glow, dark background.
 *
 * @param {AudioBuffer}       audioBuffer  - Decoded audio (from decodeAudioData)
 * @param {HTMLCanvasElement} targetCanvas - The canvas to draw into
 */
export function drawStaticWaveform(audioBuffer, targetCanvas) {
  if (!audioBuffer || !targetCanvas) return;

  const staticCtx = targetCanvas.getContext('2d');
  if (!staticCtx) return;

  // Size the canvas
  const dpr  = window.devicePixelRatio || 1;
  const rect  = targetCanvas.getBoundingClientRect();
  const cssW  = rect.width  || targetCanvas.clientWidth  || 600;
  const cssH  = rect.height || targetCanvas.clientHeight || 80;

  targetCanvas.width  = Math.round(cssW * dpr);
  targetCanvas.height = Math.round(cssH * dpr);
  staticCtx.scale(dpr, dpr);

  const w = cssW;
  const h = cssH;

  // Use channel 0 (left / mono)
  const rawData   = audioBuffer.getChannelData(0);
  const barCount  = Math.min(Math.floor(w / 3), 300);
  const blockSize = Math.floor(rawData.length / barCount);
  const centerY   = h / 2;
  const barW      = w / barCount;
  const gap       = Math.max(barW * 0.15, 0.5);

  staticCtx.clearRect(0, 0, w, h);

  // Center reference line
  staticCtx.strokeStyle = 'rgba(255,255,255,0.05)';
  staticCtx.lineWidth   = 1;
  staticCtx.beginPath();
  staticCtx.moveTo(0, centerY);
  staticCtx.lineTo(w, centerY);
  staticCtx.stroke();

  // Draw bars
  staticCtx.shadowColor = 'rgba(255,255,255,0.35)';
  staticCtx.shadowBlur  = 5;
  staticCtx.fillStyle   = 'rgba(255,255,255,0.55)';

  for (let i = 0; i < barCount; i++) {
    // RMS of the block for a cleaner envelope
    let sum = 0;
    for (let j = 0; j < blockSize; j++) {
      const s = rawData[i * blockSize + j] || 0;
      sum += s * s;
    }
    const rms   = Math.sqrt(sum / blockSize);
    const barH  = Math.max(rms * h * 2.2, 1);
    const x     = i * barW + gap / 2;
    const y     = centerY - barH / 2;
    staticCtx.fillRect(x, y, barW - gap, barH);
  }

  staticCtx.shadowBlur = 0;
}

/**
 * app.js — VOXSHIELD
 *
 * Root application controller.
 * Now connects to the FastAPI backend at BACKEND_URL for real AI analysis.
 *
 * Backend endpoints used:
 *   GET  /api/health          — startup readiness check
 *   GET  /api/model/status    — model load state
 *   POST /api/analyze         — file upload analysis
 *   WS   /ws/analyze          — live call streaming
 *
 * Demo mode remains available as a fallback when the backend is offline,
 * clearly labelled so users know results are not from a real model.
 */

import {
  initScoreRing,
  setDemoMode,
  setDemoScore,
  setScore,
  getIsDemoMode,
} from './score-ring.js';

import {
  initWaveform,
  startRenderLoop,
  switchCanvas,
  drawStaticWaveform,
} from './waveform.js';

import {
  startMonitoring,
  stopMonitoring,
  getIsMonitoring,
  getAudioLevel,
  getElapsedTime,
  applySettings,
  consumePcmSamples,
  enumerateAudioInputDevices,
} from './audio.js';

// ═══════════════════════════════════════════════════════════════
// BACKEND CONFIGURATION
// ═══════════════════════════════════════════════════════════════

/**
 * Base URL of the VOXSHIELD FastAPI backend.
 * Change this if your backend runs on a different host or port.
 */
const BACKEND_URL    = 'http://localhost:8001';
const WS_BACKEND_URL = 'ws://localhost:8001';

// ═══════════════════════════════════════════════════════════════
// GLOBAL STATE
// ═══════════════════════════════════════════════════════════════

/** @type {'monitor'|'upload'|'call'} */
let activeTab = 'monitor';

// In-memory session log (never stored/uploaded)
const sessions = [];
let sessionCounter = 0;

// Interval IDs for UI polling
let monitorPollId      = null;
let callPollId         = null;
let demoTickerInterval = null;
let demoTickerDown     = 5;
let demoTickerSource   = null;
let backendPollId      = null;

// Upload state
let uploadAudioBuffer = null;
let uploadObjectUrl   = null;
let isAnalysing       = false;

// Last real score received via WebSocket (monitor / call)
let lastStreamScore = null;

// Live Call WebSocket state
let callWebSocket   = null;
let monitorWebSocket = null;
let callSessionId   = null;
let backendOnline   = false;   // detected at init and refreshed periodically
let demoUserOverride = false;  // true once the user manually changes DEMO MODE

// ═══════════════════════════════════════════════════════════════
// DOM REFERENCES
// ═══════════════════════════════════════════════════════════════

// ─── Tabs ─────────────────────────────────────────────────────
const tabBtns = {
  monitor: document.getElementById('tab-btn-monitor'),
  upload:  document.getElementById('tab-btn-upload'),
  call:    document.getElementById('tab-btn-call'),
};
const tabPanels = {
  monitor: document.getElementById('tab-panel-monitor'),
  upload:  document.getElementById('tab-panel-upload'),
  call:    document.getElementById('tab-panel-call'),
};

// ─── Live Monitor ─────────────────────────────────────────────
const startBtn        = document.getElementById('start-btn');
const stopBtn         = document.getElementById('stop-btn');
const timerDisplay    = document.getElementById('timer-display');
const levelBarFill    = document.getElementById('level-bar-fill');
const levelValue      = document.getElementById('level-value');
const waveformOverlay = document.getElementById('waveform-overlay');
const micStatusBadge  = document.getElementById('mic-status-badge');
const micStatusText   = document.getElementById('mic-status-text');
const micError        = document.getElementById('mic-error');
const micErrorText    = document.getElementById('mic-error-text');

// ─── File Upload ──────────────────────────────────────────────
const uploadDropzone      = document.getElementById('upload-dropzone');
const fileInput           = document.getElementById('file-input');
const fileInfoPanel       = document.getElementById('file-info-panel');
const fileNameDisplay     = document.getElementById('file-name-display');
const fileSizeDisplay     = document.getElementById('file-size-display');
const fileTypeDisplay     = document.getElementById('file-type-display');
const fileDurationDisplay = document.getElementById('file-duration-display');
const removeFileBtn       = document.getElementById('remove-file-btn');
const fileWaveformCont    = document.getElementById('file-waveform-container');
const fileWaveformCanvas  = document.getElementById('file-waveform-canvas');
const audioElement        = document.getElementById('audio-element');
const audioPlayer         = document.getElementById('audio-player');
const playerPlayBtn       = document.getElementById('player-play-btn');
const playerPlayIcon      = document.getElementById('player-play-icon');
const playerSeek          = document.getElementById('player-seek');
const playerSeekFill      = document.getElementById('player-seek-fill');
const playerCurrentTime   = document.getElementById('player-current-time');
const playerTotalTime     = document.getElementById('player-total-time');
const analyzeBtn          = document.getElementById('analyze-btn');
const resetUploadBtn      = document.getElementById('reset-upload-btn');
const analysisSteps       = document.getElementById('analysis-steps');
const uploadResult        = document.getElementById('upload-result');
const resultClassification= document.getElementById('result-classification');
const uploadModeBadge     = document.getElementById('upload-mode-badge');
const step4Name           = document.getElementById('step-4-name');
const resultModeBadge     = document.getElementById('result-mode-badge');
const resultNote          = document.getElementById('result-note');
const fileError           = document.getElementById('file-error');
const fileErrorText       = document.getElementById('file-error-text');

// ─── Live Call ────────────────────────────────────────────────
const callDeviceSelect    = document.getElementById('call-device-select');
const refreshDevicesBtn   = document.getElementById('refresh-devices-btn');
const callWaveformOverlay = document.getElementById('call-waveform-overlay');
const callStatusBadge     = document.getElementById('call-status-badge');
const callStatusText      = document.getElementById('call-status-text');
const callLevelBarFill    = document.getElementById('call-level-bar-fill');
const callLevelValue      = document.getElementById('call-level-value');
const callLevelTrack      = document.getElementById('call-level-track');
const callTimerDisplay    = document.getElementById('call-timer-display');
const callStartBtn        = document.getElementById('call-start-btn');
const callStopBtn         = document.getElementById('call-stop-btn');
const callDemoTicker      = document.getElementById('call-demo-ticker');
const callTickerText      = document.getElementById('call-ticker-text');
const callError           = document.getElementById('call-error');
const callErrorText       = document.getElementById('call-error-text');

// ─── Shared right column ──────────────────────────────────────
const demoToggle       = document.getElementById('demo-toggle');
const scoreModeText    = document.getElementById('score-mode-text');
const historyList      = document.getElementById('history-list');
const historyEmpty     = document.getElementById('history-empty');
const clearHistoryBtn  = document.getElementById('clear-history-btn');
const settingsBtn      = document.getElementById('settings-btn');
const settingsModal    = document.getElementById('settings-modal');
const settingsCloseBtn = document.getElementById('settings-close-btn');
const fftSizeSelect    = document.getElementById('fft-size-select');
const smoothingSelect  = document.getElementById('smoothing-select');

// Analysis panel
const analysisMode           = document.getElementById('analysis-mode');
const analysisStatus         = document.getElementById('analysis-status');
const analysisClassification = document.getElementById('analysis-classification');
const analysisInterpretation = document.getElementById('analysis-interpretation');
const analysisMic            = document.getElementById('analysis-mic');
const analysisLevel          = document.getElementById('analysis-level');
const analysisProcessing     = document.getElementById('analysis-processing');
const analysisLastUpdate     = document.getElementById('analysis-last-update');

// ═══════════════════════════════════════════════════════════════
// INITIALISATION
// ═══════════════════════════════════════════════════════════════

async function init() {
  if (window.lucide) window.lucide.createIcons();

  initScoreRing();
  initWaveform('waveform-canvas');
  startRenderLoop();

  wireTabNav();
  wireMonitorTab();
  wireUploadTab();
  wireCallTab();
  wireShared();

  setAnalysisDefaults('LIVE MONITOR');
  updateScoreModePill('LIVE MONITOR');

  // Check backend health at startup and keep watching for it
  await refreshBackendStatus();
  startBackendPolling();
}

// ─── Backend health probe ─────────────────────────────────────

/**
 * Probe the backend and model, update connection state, and fall back to
 * demo mode when real analysis is unavailable. Returns true only when the
 * model is loaded and ready for inference.
 *
 * @param {{ silent?: boolean }} [options]
 * @returns {Promise<boolean>}
 */
async function refreshBackendStatus({ silent = false } = {}) {
  try {
    const res = await fetch(`${BACKEND_URL}/api/health`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) throw new Error(`Health check returned ${res.status}`);

    const data = await res.json();
    const modelRes = await fetch(`${BACKEND_URL}/api/model/status`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!modelRes.ok) throw new Error(`Model status returned ${modelRes.status}`);

    const modelData = await modelRes.json();
    const wasOnline = backendOnline;
    backendOnline = true;

    if (modelData.readiness === 'ready') {
      if (!demoUserOverride) {
        if (demoToggle) demoToggle.checked = false;
        setDemoMode(false);
      }
      updateDemoLabels(getIsDemoMode());
      updateStatusPill('BACKEND CONNECTED');
      if (!wasOnline) {
        console.info('[VOXSHIELD] Backend online — version', data.version);
        console.info('[VOXSHIELD] AASIST model ready on', modelData.device);
      }
      return true;
    }

    if (!demoUserOverride) {
      if (demoToggle) demoToggle.checked = true;
      setDemoMode(true);
    }
    updateDemoLabels(true);
    updateStatusPill('DEMO MODE — MODEL LOADING');
    if (!silent) {
      showBackendWarning(
        `Backend connected but model not loaded yet. ${modelData.error_detail || ''} ` +
        `Run: python models/AASIST/download_weights.py`
      );
    }
    return false;
  } catch (_) {
    backendOnline = false;
    if (!demoUserOverride) {
      if (demoToggle) demoToggle.checked = true;
      setDemoMode(true);
    }
    updateDemoLabels(true);
    updateStatusPill('DEMO MODE — BACKEND OFFLINE');
    if (!silent) {
      console.info('[VOXSHIELD] Backend offline — using demo mode.');
      showBackendWarning('Backend not detected at ' + BACKEND_URL + '. Demo mode active.');
    }
    return false;
  }
}

/** Retry the backend immediately; used before starting any analysis. */
async function ensureBackendReady() {
  if (backendOnline) return true;
  return refreshBackendStatus({ silent: true });
}

/** Poll while idle so starting the backend after page load is picked up. */
function startBackendPolling() {
  if (backendPollId !== null) return;
  backendPollId = setInterval(() => {
    if (!getIsMonitoring()) refreshBackendStatus({ silent: true });
  }, 15000);
}

/** Keep the upload UI honest about whether results come from AASIST or demo. */
function updateDemoLabels(isDemo) {
  if (uploadModeBadge) uploadModeBadge.textContent = isDemo ? 'DEMO MODE' : 'AASIST MODEL';
  if (step4Name)       step4Name.textContent       = isDemo ? 'MODEL INFERENCE [DEMO]' : 'AASIST INFERENCE';
  if (resultModeBadge) resultModeBadge.textContent = isDemo ? 'DEMO RESULT' : 'MODEL RESULT';
  if (resultNote) {
    resultNote.textContent = isDemo
      ? 'This is a simulated result for demonstration purposes. ' +
        'No anti-spoofing model is currently connected — scores are not calculated from actual audio content.'
      : 'Result produced by the AASIST anti-spoofing model. ' +
        'Detection is probabilistic and must not be treated as definitive proof of fraud.';
  }
}

function updateStatusPill(text) {
  const pill = document.querySelector('.status-pill');
  if (!pill) return;
  const textEl = pill.querySelector('.status-text');
  if (textEl) {
    textEl.textContent = text;
  }
  const dot = pill.querySelector('.status-dot');
  if (dot) {
    dot.classList.toggle('status-dot--ready', backendOnline);
    dot.classList.toggle('status-dot--demo', !backendOnline);
  }
}

function showBackendWarning(msg) {
  // Show a non-intrusive console warning; could be surfaced as a banner
  console.warn('[VOXSHIELD]', msg);
  // Update the status pill text in the header
  updateStatusPill(backendOnline ? 'BACKEND CONNECTED' : 'DEMO MODE');
}

// ═══════════════════════════════════════════════════════════════
// TAB NAVIGATION
// ═══════════════════════════════════════════════════════════════

function wireTabNav() {
  Object.values(tabBtns).forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });
}

function switchTab(tab) {
  if (tab === activeTab) return;

  if (getIsMonitoring()) {
    if (activeTab === 'monitor') handleStopMonitoring();
    if (activeTab === 'call')    handleCallStop();
  }

  Object.entries(tabBtns).forEach(([key, btn]) => {
    const isActive = key === tab;
    btn.setAttribute('aria-selected', String(isActive));
    btn.classList.toggle('tab-btn--active', isActive);
  });

  Object.entries(tabPanels).forEach(([key, panel]) => {
    panel.hidden = key !== tab;
  });

  activeTab = tab;

  if (tab === 'monitor') {
    switchCanvas('waveform-canvas');
    setAnalysisDefaults('LIVE MONITOR');
    updateScoreModePill('LIVE MONITOR');
  } else if (tab === 'upload') {
    startRenderLoop();
    setAnalysisDefaults('FILE UPLOAD');
    updateScoreModePill('FILE UPLOAD');
  } else if (tab === 'call') {
    switchCanvas('call-waveform-canvas');
    setAnalysisDefaults('LIVE CALL');
    updateScoreModePill('LIVE CALL');
    populateDevices();
  }
}

function updateScoreModePill(label) {
  if (scoreModeText) scoreModeText.textContent = label;
}

// ═══════════════════════════════════════════════════════════════
// LIVE MONITOR TAB — waveform plus real-time backend inference
// ═══════════════════════════════════════════════════════════════

function wireMonitorTab() {
  startBtn.addEventListener('click', handleStartMonitoring);
  stopBtn.addEventListener('click', handleStopMonitoring);
}

async function handleStartMonitoring() {
  if (getIsMonitoring()) return;

  hideMicError();
  startBtn.disabled = true;
  setMicStatus('REQUESTING…', false);

  const backendReady = await ensureBackendReady();

  try {
    await startMonitoring();
  } catch (err) {
    startBtn.disabled = false;
    setMicStatus('ERROR', false);
    showMicError(err.message);
    return;
  }

  stopBtn.disabled = false;
  setOverlayHidden(waveformOverlay, true);
  setMicStatus('ACTIVE', true);
  updateAnalysisOnMonitorStart();
  startMonitorPolling();
  lastStreamScore = null;

  if (backendReady && !getIsDemoMode()) {
    openMonitorWebSocket();
  } else {
    startDemoFallback('monitor');
  }
}

function handleStopMonitoring() {
  if (!getIsMonitoring()) return;

  const { durationMs } = stopMonitoring();
  if (monitorWebSocket) {
    monitorWebSocket.close(1000);
    monitorWebSocket = null;
  }
  stopMonitorPolling();
  stopDemoTicker();

  addSession({ type: 'monitor', durationMs, demoScore: lastStreamScore, source: 'Microphone' });
  lastStreamScore = null;

  startBtn.disabled = false;
  stopBtn.disabled  = true;
  setOverlayHidden(waveformOverlay, false);
  setMicStatus('IDLE', false);
  resetLevelMeter(levelBarFill, levelValue, null);
  if (timerDisplay) timerDisplay.textContent = '00:00';
  updateAnalysisOnStop('LIVE MONITOR');
}

function startMonitorPolling() {
  if (monitorPollId !== null) return;
  monitorPollId = setInterval(() => {
    if (!getIsMonitoring()) return;
    if (timerDisplay) timerDisplay.textContent = getElapsedTime();
    const level = getAudioLevel();
    const pct   = Math.round(level * 100);
    updateLevelMeter(levelBarFill, levelValue, null, pct, level);
    setAnalysisField(analysisLevel,      `${pct}% / ${levelToDb(level)}`, pct > 2);
    setAnalysisField(analysisLastUpdate, new Date().toLocaleTimeString(), false);
  }, 60);
}

function stopMonitorPolling() {
  if (monitorPollId !== null) { clearInterval(monitorPollId); monitorPollId = null; }
}

function setMicStatus(text, active) {
  if (micStatusText)  micStatusText.textContent = text;
  if (micStatusBadge) micStatusBadge.classList.toggle('active', active);
}

function openMonitorWebSocket() {
  monitorWebSocket = new WebSocket(`${WS_BACKEND_URL}/ws/analyze`);
  monitorWebSocket.binaryType = 'arraybuffer';
  monitorWebSocket.addEventListener('open', () => {
    monitorWebSocket.send(JSON.stringify({ type: 'session_init', sample_rate: 16000, channels: 1, encoding: 'pcm_float32' }));
    startAudioStreaming(monitorWebSocket);
  });
  monitorWebSocket.addEventListener('message', (event) => {
    if (typeof event.data !== 'string') return;
    try { handleWsMessage(JSON.parse(event.data), 'monitor'); } catch (_) {}
  });
  monitorWebSocket.addEventListener('error', () => {
    showMicError('Live analysis connection failed — switching to demo analysis.');
    if (getIsMonitoring()) startDemoFallback('monitor');
  });
  monitorWebSocket.addEventListener('close', (event) => {
    monitorWebSocket = null;
    if (event.code !== 1000 && getIsMonitoring()) {
      backendOnline = false;
      startDemoFallback('monitor');
    }
  });
}

// ═══════════════════════════════════════════════════════════════
// FILE UPLOAD TAB — real backend POST /api/analyze
// ═══════════════════════════════════════════════════════════════

function wireUploadTab() {
  uploadDropzone.addEventListener('click', (e) => {
    if (e.target !== fileInput) fileInput.click();
  });
  uploadDropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files?.[0]) handleFileSelected(fileInput.files[0]);
  });
  uploadDropzone.addEventListener('dragover', (e) => {
    e.preventDefault(); uploadDropzone.classList.add('drag-over');
  });
  uploadDropzone.addEventListener('dragleave', () => uploadDropzone.classList.remove('drag-over'));
  uploadDropzone.addEventListener('drop', (e) => {
    e.preventDefault(); uploadDropzone.classList.remove('drag-over');
    const file = e.dataTransfer?.files?.[0];
    if (file?.type?.startsWith('audio/')) handleFileSelected(file);
    else if (file) showFileError('Please drop an audio file (WAV, MP3, OGG, FLAC, M4A).');
  });

  removeFileBtn.addEventListener('click', resetUpload);
  analyzeBtn.addEventListener('click', runAnalysis);
  resetUploadBtn.addEventListener('click', resetUpload);

  playerPlayBtn.addEventListener('click', toggleAudioPlay);
  playerSeek.addEventListener('input', handleSeek);
  audioElement.addEventListener('timeupdate', updatePlayerProgress);
  audioElement.addEventListener('ended', onAudioEnded);
  audioElement.addEventListener('loadedmetadata', () => {
    if (playerTotalTime) playerTotalTime.textContent = formatTime(audioElement.duration);
  });
}

async function handleFileSelected(file) {
  resetUploadAnalysisState();
  hideFileError();

  if (!file.type.startsWith('audio/')) {
    showFileError('Unsupported file type. Please select a WAV, MP3, OGG, FLAC, or M4A file.');
    return;
  }

  fileNameDisplay.textContent  = truncateFilename(file.name, 40);
  fileSizeDisplay.textContent  = formatFileSize(file.size);
  fileTypeDisplay.textContent  = file.type.split('/')[1].toUpperCase() || 'AUDIO';
  fileDurationDisplay.textContent = '—';
  fileInfoPanel.hidden = false;
  analyzeBtn.disabled  = false;

  setAnalysisField(analysisMic,        file.name, false);
  setAnalysisField(analysisStatus,     'FILE LOADED', false);
  setAnalysisField(analysisProcessing, 'READY', false);
  setAnalysisField(analysisLastUpdate, new Date().toLocaleTimeString(), false);

  if (uploadObjectUrl) URL.revokeObjectURL(uploadObjectUrl);
  uploadObjectUrl  = URL.createObjectURL(file);
  audioElement.src = uploadObjectUrl;

  try {
    const arrayBuffer = await file.arrayBuffer();
    const offlineCtx  = new OfflineAudioContext(1, 1, 44100);
    uploadAudioBuffer = await offlineCtx.decodeAudioData(arrayBuffer);
    fileDurationDisplay.textContent = formatTime(uploadAudioBuffer.duration);
    fileWaveformCont.hidden = false;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      drawStaticWaveform(uploadAudioBuffer, fileWaveformCanvas);
    }));
    audioPlayer.hidden = false;
  } catch (err) {
    fileWaveformCont.hidden = true;
    audioPlayer.hidden      = false;
  }
}

/**
 * Send file to backend POST /api/analyze (if backend online),
 * or fall back to demo simulation (if offline or demo mode forced).
 */
async function runAnalysis() {
  if (isAnalysing) return;
  isAnalysing = true;

  analyzeBtn.disabled   = true;
  resetUploadBtn.hidden = true;
  uploadResult.hidden   = true;
  analysisSteps.hidden  = false;
  for (let i = 1; i <= 5; i++) setStepState(i, 'pending');

  setAnalysisField(analysisStatus,     'ANALYSING…', true);
  setAnalysisField(analysisProcessing, 'RUNNING', true);

  // ── Recheck the backend so it works even if it started after page load ────
  const backendReady = await ensureBackendReady();

  // ── Decide: real backend or demo ──────────────────────────────────────────
  if (backendReady && !getIsDemoMode()) {
    await runRealAnalysis();
  } else {
    await runDemoAnalysis();
  }

  isAnalysing = false;
}

/** Send audio to the real backend and parse the JSON response. */
async function runRealAnalysis() {
  const STEP_LABELS = [
    'UPLOADING TO BACKEND',
    'DECODING AUDIO',
    'EXTRACTING FEATURES',
    'AASIST INFERENCE',
    'COMPUTING SCORE',
  ];

  // Animate steps with estimated timings
  const TIMINGS = [400, 500, 600, 800, 300];

  for (let i = 0; i < STEP_LABELS.length; i++) {
    setStepState(i + 1, 'active');
    // Start next step partway through current one
    if (i < STEP_LABELS.length - 1) await delay(TIMINGS[i] / 2);
  }

  // Build FormData with the file
  const file    = fileInput.files[0];
  const formData = new FormData();
  formData.append('file', file);

  let data;
  try {
    const res = await fetch(`${BACKEND_URL}/api/analyze`, {
      method: 'POST',
      body:   formData,
      signal: AbortSignal.timeout(30000),
    });
    data = await res.json();

    if (!res.ok) {
      throw new Error(data?.detail?.message || data?.message || `Server error ${res.status}`);
    }
  } catch (err) {
    // Mark all steps failed and show error
    for (let i = 1; i <= 5; i++) setStepState(i, 'pending');
    showFileError(`Analysis failed: ${err.message}`);
    setAnalysisField(analysisStatus,     'ERROR', false);
    setAnalysisField(analysisProcessing, 'FAILED', false);
    resetUploadBtn.hidden = false;
    analyzeBtn.disabled   = false;
    return;
  }

  // Mark all steps done
  for (let i = 1; i <= 5; i++) setStepState(i, 'done');

  // Update score ring with real score
  const score = Math.round(data.raw_score * 100);
  setScore(score, 'FILE ANALYSIS');

  // Map classification
  const CLS_LABELS = {
    'likely_authentic':          'LIKELY AUTHENTIC',
    'potential_synthetic_speech':'POTENTIALLY SYNTHETIC',
    'inconclusive':              'INCONCLUSIVE',
  };
  const cls = CLS_LABELS[data.classification] || data.classification.toUpperCase();

  resultClassification.textContent = cls;
  uploadResult.hidden  = false;
  resetUploadBtn.hidden = false;

  setAnalysisField(analysisStatus,         'ANALYSIS COMPLETE', true);
  setAnalysisField(analysisClassification, cls, false);
  setAnalysisField(analysisInterpretation, data.score_interpretation, false);
  setAnalysisField(analysisProcessing,     `${Math.round(data.processing_latency_ms)} ms`, false);
  setAnalysisField(analysisLastUpdate,     new Date().toLocaleTimeString(), false);

  // Show any backend warnings
  if (data.warnings?.length) {
    const warnText = data.warnings.map(w => w.message).join(' | ');
    showFileError(`⚠ ${warnText}`);
  }

  addSession({
    type: 'upload', durationMs: 0, demoScore: score,
    source: fileNameDisplay?.textContent || 'Audio file',
  });
}

/** Demo simulation (5 fake steps, random score). Used when backend is offline or demo forced. */
async function runDemoAnalysis() {
  enableDemoFallback();

  const STEPS = [
    { id: 1, ms: 450  },
    { id: 2, ms: 900  },
    { id: 3, ms: 1300 },
    { id: 4, ms: 1600 },
    { id: 5, ms: 500  },
  ];
  for (const step of STEPS) {
    setStepState(step.id, 'active');
    await delay(step.ms);
    setStepState(step.id, 'done');
  }

  const score = Math.round(68 + Math.random() * 28);
  const cls   = score >= 80 ? 'LIKELY AUTHENTIC (DEMO)' : 'POTENTIALLY SYNTHETIC (DEMO)';

  setDemoScore(score, 'FILE ANALYSIS (DEMO)');
  resultClassification.textContent = cls;
  uploadResult.hidden   = false;
  resetUploadBtn.hidden = false;

  setAnalysisField(analysisStatus,         'ANALYSIS COMPLETE (DEMO)', true);
  setAnalysisField(analysisClassification, cls, false);
  setAnalysisField(analysisInterpretation, `Demo score ${score}% — not from a real model`, false);
  setAnalysisField(analysisProcessing,     'IDLE', false);
  setAnalysisField(analysisLastUpdate,     new Date().toLocaleTimeString(), false);

  addSession({
    type: 'upload', durationMs: 0, demoScore: score,
    source: fileNameDisplay?.textContent || 'Audio file',
  });
}

function setStepState(stepId, state) {
  const el       = document.getElementById(`step-${stepId}`);
  const statusEl = document.getElementById(`step-status-${stepId}`);
  if (!el) return;
  el.classList.remove('step--active', 'step--done');
  if (state === 'active') { el.classList.add('step--active'); if (statusEl) statusEl.textContent = 'RUNNING'; }
  else if (state === 'done') { el.classList.add('step--done'); if (statusEl) statusEl.textContent = 'DONE'; }
  else { if (statusEl) statusEl.textContent = 'PENDING'; }
}

function resetUpload() {
  if (uploadObjectUrl) { URL.revokeObjectURL(uploadObjectUrl); uploadObjectUrl = null; }
  if (audioElement)   { audioElement.pause(); audioElement.src = ''; }
  uploadAudioBuffer = null; isAnalysing = false;
  fileInput.value   = '';
  fileInfoPanel.hidden    = true; fileWaveformCont.hidden = true;
  audioPlayer.hidden      = true; analyzeBtn.disabled     = true;
  resetUploadBtn.hidden   = true; analysisSteps.hidden    = true;
  uploadResult.hidden     = true;
  hideFileError();
  for (let i = 1; i <= 5; i++) setStepState(i, 'pending');
  if (playerSeek)        playerSeek.value        = 0;
  if (playerSeekFill)    playerSeekFill.style.width = '0%';
  if (playerCurrentTime) playerCurrentTime.textContent = '0:00';
  if (playerTotalTime)   playerTotalTime.textContent   = '0:00';
  setPlayerIcon(false);
  setAnalysisDefaults('FILE UPLOAD');
  if (getIsDemoMode()) setDemoScore(94, 'DEMO SCORE');
}

function resetUploadAnalysisState() {
  isAnalysing = false; analysisSteps.hidden = true; uploadResult.hidden = true;
  resetUploadBtn.hidden = true; analyzeBtn.disabled = false;
  for (let i = 1; i <= 5; i++) setStepState(i, 'pending');
}

// ─ Audio player ───────────────────────────────────────────────

function toggleAudioPlay() {
  if (!audioElement) return;
  if (audioElement.paused) { audioElement.play().catch(() => {}); setPlayerIcon(true); }
  else { audioElement.pause(); setPlayerIcon(false); }
}
function handleSeek() {
  if (!audioElement?.duration) return;
  audioElement.currentTime = (parseFloat(playerSeek.value) / 100) * audioElement.duration;
}
function updatePlayerProgress() {
  if (!audioElement?.duration) return;
  const pct = (audioElement.currentTime / audioElement.duration) * 100;
  if (playerSeek)        playerSeek.value              = pct;
  if (playerSeekFill)    playerSeekFill.style.width    = `${pct}%`;
  if (playerCurrentTime) playerCurrentTime.textContent = formatTime(audioElement.currentTime);
}
function onAudioEnded() {
  setPlayerIcon(false);
  if (playerSeek)     playerSeek.value = 0;
  if (playerSeekFill) playerSeekFill.style.width = '0%';
}
function setPlayerIcon(playing) {
  if (!playerPlayIcon) return;
  playerPlayIcon.setAttribute('data-lucide', playing ? 'pause' : 'play');
  if (window.lucide) window.lucide.createIcons({ nodes: [playerPlayIcon] });
}

function showFileError(msg) { if (fileError) fileError.hidden = false; if (fileErrorText) fileErrorText.textContent = msg; }
function hideFileError()    { if (fileError) fileError.hidden = true; }
function formatFileSize(b)  { if (b < 1024) return `${b} B`; if (b < 1024**2) return `${(b/1024).toFixed(1)} KB`; return `${(b/1024**2).toFixed(2)} MB`; }
function truncateFilename(n, max) { if (n.length <= max) return n; const e = n.lastIndexOf('.') > 0 ? n.slice(n.lastIndexOf('.')) : ''; return n.slice(0, max - e.length - 1) + '…' + e; }

// ═══════════════════════════════════════════════════════════════
// LIVE CALL TAB — WebSocket WS /ws/analyze
// ═══════════════════════════════════════════════════════════════

function wireCallTab() {
  callStartBtn.addEventListener('click', handleCallStart);
  callStopBtn.addEventListener('click', handleCallStop);
  refreshDevicesBtn.addEventListener('click', populateDevices);
}

async function populateDevices() {
  const devices = await enumerateAudioInputDevices();
  while (callDeviceSelect.options.length > 0) callDeviceSelect.remove(0);
  callDeviceSelect.add(new Option('Default Microphone', ''));
  devices.forEach((d, i) => callDeviceSelect.add(new Option(d.label || `Audio Input ${i + 1}`, d.deviceId)));
}

async function handleCallStart() {
  if (getIsMonitoring()) return;

  hideCallError();
  callStartBtn.disabled = true;
  setCallStatus('REQUESTING…', false);

  const deviceId = callDeviceSelect.value || undefined;
  const backendReady = await ensureBackendReady();

  try {
    await startMonitoring(deviceId);
  } catch (err) {
    callStartBtn.disabled = false;
    setCallStatus('ERROR', false);
    showCallError(err.message);
    populateDevices();
    return;
  }

  populateDevices();
  callStopBtn.disabled = false;
  setOverlayHidden(callWaveformOverlay, true);
  setCallStatus('ACTIVE', true);
  updateAnalysisOnCallStart();
  startCallPolling();
  lastStreamScore = null;

  // Connect WebSocket for real backend inference if available
  if (backendReady && !getIsDemoMode()) {
    openCallWebSocket();
  } else {
    startDemoFallback('call');
  }
}

function handleCallStop() {
  if (!getIsMonitoring()) return;

  const { durationMs } = stopMonitoring();

  // Close WebSocket
  if (callWebSocket) {
    callWebSocket.close(1000);
    callWebSocket = null;
  }

  stopCallPolling();
  stopDemoTicker();

  addSession({
    type: 'call', durationMs,
    demoScore: lastStreamScore,
    source: callDeviceSelect.selectedOptions[0]?.text || 'Default Mic',
  });
  lastStreamScore = null;

  callStartBtn.disabled = false;
  callStopBtn.disabled  = true;
  setOverlayHidden(callWaveformOverlay, false);
  setCallStatus('IDLE', false);
  resetLevelMeter(callLevelBarFill, callLevelValue, callLevelTrack);
  if (callTimerDisplay) callTimerDisplay.textContent = '00:00';
  if (callDemoTicker)   callDemoTicker.hidden = true;
  updateAnalysisOnStop('LIVE CALL');
}

// ─ WebSocket streaming ────────────────────────────────────────

/** Open a WS connection to the backend and stream PCM audio from the mic. */
function openCallWebSocket() {
  callWebSocket = new WebSocket(`${WS_BACKEND_URL}/ws/analyze`);
  callWebSocket.binaryType = 'arraybuffer';

  callWebSocket.addEventListener('open', () => {
    // Send session_init
    callWebSocket.send(JSON.stringify({
      type:        'session_init',
      sample_rate: 16000,
      channels:    1,
      encoding:    'pcm_float32',
    }));
    // Start streaming audio in a loop
    startAudioStreaming(callWebSocket);
  });

  callWebSocket.addEventListener('message', (event) => {
    if (typeof event.data !== 'string') return;
    let msg;
    try { msg = JSON.parse(event.data); } catch { return; }
    handleWsMessage(msg);
  });

  callWebSocket.addEventListener('error', () => {
    console.warn('[VOXSHIELD] WS error — falling back to demo ticker.');
    if (getIsMonitoring()) startDemoFallback('call');
  });

  callWebSocket.addEventListener('close', (event) => {
    callWebSocket = null;
    if (event.code !== 1000 && getIsMonitoring()) {
      backendOnline = false;
      startDemoFallback('call');
    }
  });
}

/** Periodically read microphone samples and send as PCM float32 binary frames. */
let audioStreamInterval = null;

function startAudioStreaming(webSocket) {
  // Send genuine microphone PCM, resampled to the model's 16 kHz contract.
  audioStreamInterval = setInterval(() => {
    if (!getIsMonitoring() || !webSocket || webSocket.readyState !== WebSocket.OPEN) {
      clearInterval(audioStreamInterval); audioStreamInterval = null; return;
    }
    const samples = consumePcmSamples(16000);
    if (samples?.length) webSocket.send(samples.buffer);
  }, 250);
}

function handleWsMessage(msg, source = 'call') {
  switch (msg.type) {
    case 'session_started':
      callSessionId = msg.session_id;
      setAnalysisField(analysisProcessing, 'AASIST INFERENCE ACTIVE', true);
      if (source === 'monitor') setMicStatus('ANALYSING', true);
      break;

    case 'analysis_result': {
      stopDemoTicker();
      const score = Math.round(msg.raw_score * 100);
      setScore(score, source === 'monitor' ? 'LIVE MONITOR' : 'CALL ANALYSIS');
      lastStreamScore = score;
      const CLS = {
        'likely_authentic':          'LIKELY AUTHENTIC',
        'potential_synthetic_speech':'POSSIBLY SYNTHETIC',
        'inconclusive':              'INCONCLUSIVE',
      };
      const cls = CLS[msg.classification] || msg.classification.toUpperCase();
      setAnalysisField(analysisClassification, cls, false);
      setAnalysisField(analysisInterpretation, msg.score_interpretation, false);
      setAnalysisField(analysisLastUpdate, new Date().toLocaleTimeString(), false);
      setAnalysisField(analysisProcessing, `${Math.round(msg.processing_latency_ms)} ms latency`, false);
      if (source === 'call') {
        if (callTickerText) callTickerText.textContent = `LIVE ANALYSIS — score: ${score}%`;
        if (callDemoTicker) callDemoTicker.hidden = false;
      } else {
        setMicStatus('ANALYSED', true);
      }
      break;
    }

    case 'warning':
      console.warn('[WS warning]', msg.code, msg.message);
      break;

    case 'error':
      if (source === 'monitor') showMicError(`Backend: ${msg.message}`);
      else showCallError(`Backend: ${msg.message}`);
      break;

    case 'session_completed':
      if (callDemoTicker) callDemoTicker.hidden = true;
      break;
  }
}

// ─ Call polling ───────────────────────────────────────────────

function startCallPolling() {
  if (callPollId !== null) return;
  callPollId = setInterval(() => {
    if (!getIsMonitoring()) return;
    if (callTimerDisplay) callTimerDisplay.textContent = getElapsedTime();
    const level = getAudioLevel();
    const pct   = Math.round(level * 100);
    updateLevelMeter(callLevelBarFill, callLevelValue, callLevelTrack, pct, level);
    setAnalysisField(analysisLevel,      `${pct}% / ${levelToDb(level)}`, pct > 2);
    setAnalysisField(analysisLastUpdate, new Date().toLocaleTimeString(), false);
  }, 60);
}

function stopCallPolling() {
  if (callPollId !== null) { clearInterval(callPollId); callPollId = null; }
}

// ─ Demo analysis ticker ───────────────────────────────────────

/** Turn demo mode on when it is the only available analysis path. */
function enableDemoFallback() {
  if (getIsDemoMode()) return;
  if (demoToggle) demoToggle.checked = true;
  demoUserOverride = false;
  setDemoMode(true);
  updateDemoLabels(true);
}

function startDemoFallback(source) {
  enableDemoFallback();
  startDemoTicker(source);
}

function startDemoTicker(source) {
  stopDemoTicker();
  demoTickerSource = source;
  demoTickerDown = 5;

  if (source === 'call' && callDemoTicker) callDemoTicker.hidden = false;
  if (source === 'monitor') setMicStatus('DEMO ANALYSING', true);
  updateDemoTickerText();

  demoTickerInterval = setInterval(() => {
    if (!getIsMonitoring()) { stopDemoTicker(); return; }

    demoTickerDown--;
    if (demoTickerDown <= 0) {
      const score = Math.round(70 + Math.random() * 26);
      const label = source === 'call' ? 'CALL ANALYSIS (DEMO)' : 'LIVE MONITOR (DEMO)';
      setDemoScore(score, label);
      const cls = score >= 80 ? 'LIKELY AUTHENTIC (DEMO)' : 'POSSIBLY SYNTHETIC (DEMO)';
      setAnalysisField(analysisClassification, cls, false);
      setAnalysisField(analysisInterpretation, `Demo score ${score}% — illustrative only`, false);
      setAnalysisField(analysisProcessing, 'DEMO ANALYSIS', false);
      setAnalysisField(analysisLastUpdate, new Date().toLocaleTimeString(), false);
      demoTickerDown = 5;
    }
    updateDemoTickerText();
  }, 1000);
}

function stopDemoTicker() {
  if (demoTickerInterval !== null) { clearInterval(demoTickerInterval); demoTickerInterval = null; }
  demoTickerSource = null;
  if (callDemoTicker) callDemoTicker.hidden = true;
}

function updateDemoTickerText() {
  if (!callTickerText || demoTickerSource !== 'call') return;
  callTickerText.textContent = `DEMO ANALYSIS — next update in ${demoTickerDown}s`;
}

function setCallStatus(text, active) {
  if (callStatusText)  callStatusText.textContent = text;
  if (callStatusBadge) callStatusBadge.classList.toggle('active', active);
}

function showCallError(msg) { if (callError) callError.hidden = false; if (callErrorText) callErrorText.textContent = msg; }
function hideCallError()    { if (callError) callError.hidden = true; }

// ═══════════════════════════════════════════════════════════════
// ANALYSIS PANEL (shared right column)
// ═══════════════════════════════════════════════════════════════

function setAnalysisDefaults(modeName) {
  setAnalysisField(analysisMode,           modeName,               false);
  setAnalysisField(analysisStatus,         'WAITING FOR ANALYSIS', false);
  setAnalysisField(analysisClassification, '—',                    false);
  setAnalysisField(analysisInterpretation, '—',                    false);
  setAnalysisField(analysisMic,            '—',                    false);
  setAnalysisField(analysisLevel,          '—',                    false);
  setAnalysisField(analysisProcessing,     'IDLE',                 false);
  setAnalysisField(analysisLastUpdate,     '—',                    false);
}

function updateAnalysisOnMonitorStart() {
  setAnalysisField(analysisMode,       'LIVE MONITOR',      false);
  setAnalysisField(analysisStatus,     'MICROPHONE ACTIVE', true);
  setAnalysisField(analysisMic,        'MICROPHONE ACTIVE', true);
  setAnalysisField(analysisProcessing,
    backendOnline && !getIsDemoMode() ? 'CONNECTING TO AASIST…' : 'DEMO ANALYSIS', false);
  setAnalysisField(analysisLastUpdate, new Date().toLocaleTimeString(), false);
}

function updateAnalysisOnCallStart() {
  const device = callDeviceSelect.selectedOptions[0]?.text || 'Default Mic';
  setAnalysisField(analysisMode,       'LIVE CALL',                false);
  setAnalysisField(analysisStatus,     'CALL MONITORING ACTIVE',   true);
  setAnalysisField(analysisMic,        device,                     true);
  setAnalysisField(analysisProcessing,
    backendOnline && !getIsDemoMode() ? 'CONNECTING TO AASIST…' : 'DEMO ANALYSIS', false);
  setAnalysisField(analysisLastUpdate, new Date().toLocaleTimeString(), false);
}

function updateAnalysisOnStop(modeName) {
  setAnalysisField(analysisMode,       modeName,               false);
  setAnalysisField(analysisStatus,     'WAITING FOR ANALYSIS', false);
  setAnalysisField(analysisMic,        'STOPPED',              false);
  setAnalysisField(analysisProcessing, 'IDLE',                 false);
  setAnalysisField(analysisLevel,      '—',                    false);
  setAnalysisField(analysisLastUpdate, new Date().toLocaleTimeString(), false);
}

function setAnalysisField(el, text, active) {
  if (!el) return;
  el.textContent = text;
  el.classList.toggle('active', active);
}

// ═══════════════════════════════════════════════════════════════
// SESSION HISTORY
// ═══════════════════════════════════════════════════════════════

function addSession({ type, durationMs, demoScore, source }) {
  sessions.unshift({ id: ++sessionCounter, type, startTime: new Date(Date.now() - durationMs), durationMs, demoScore, source });
  renderHistoryList();
}

function renderHistoryList() {
  if (!historyList) return;
  historyList.querySelectorAll('.history-item').forEach(el => el.remove());
  if (sessions.length === 0) { if (historyEmpty) historyEmpty.hidden = false; return; }
  if (historyEmpty) historyEmpty.hidden = true;
  sessions.slice(0, 12).forEach(s => historyList.appendChild(buildHistoryItem(s)));
}

function buildHistoryItem(s) {
  const el = document.createElement('div');
  el.className = 'history-item'; el.setAttribute('role', 'listitem');
  const dotClass = s.demoScore !== null
    ? (s.type === 'upload' ? 'history-dot history-dot--file' : s.type === 'call' ? 'history-dot history-dot--call' : 'history-dot history-dot--demo')
    : 'history-dot';
  const typeLabel = { monitor: 'MONITOR', upload: 'FILE', call: 'CALL' }[s.type] || s.type.toUpperCase();
  const scoreStr  = s.demoScore !== null ? `${s.demoScore}%` : 'N/A';
  const durStr    = s.type === 'upload' ? s.source : formatDuration(s.durationMs);
  el.innerHTML = `<div class="${dotClass}" aria-hidden="true"></div><div class="history-meta"><span class="history-time">${s.startTime.toLocaleTimeString()} · ${typeLabel}</span><span class="history-duration">${durStr}</span></div><span class="history-score">${scoreStr}</span>`;
  return el;
}

function formatDuration(ms) { const s = Math.max(0, Math.floor(ms/1000)); const m = Math.floor(s/60); return m > 0 ? `${m}m ${s%60}s` : `${s}s`; }

// ═══════════════════════════════════════════════════════════════
// SHARED / SETTINGS
// ═══════════════════════════════════════════════════════════════

function wireShared() {
  clearHistoryBtn.addEventListener('click', () => { sessions.length = 0; sessionCounter = 0; renderHistoryList(); });
  demoToggle.addEventListener('change', () => {
    const on = demoToggle.checked;
    demoUserOverride = true;
    setDemoMode(on);
    updateDemoLabels(on);
    if (!on && backendOnline) console.info('[VOXSHIELD] Demo mode off — real backend active.');
  });
  settingsBtn.addEventListener('click', () => { settingsModal.hidden = false; settingsCloseBtn.focus(); });
  settingsCloseBtn.addEventListener('click', closeSettings);
  settingsModal.addEventListener('click', (e) => { if (e.target === settingsModal) closeSettings(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !settingsModal.hidden) closeSettings(); });
}

function closeSettings() {
  settingsModal.hidden = true; settingsBtn.focus();
  applySettings({ fftSize: parseInt(fftSizeSelect.value, 10), smoothingTimeConstant: parseFloat(smoothingSelect.value) });
}

// ═══════════════════════════════════════════════════════════════
// SHARED UI HELPERS
// ═══════════════════════════════════════════════════════════════

function setOverlayHidden(overlay, hidden) { if (overlay) overlay.classList.toggle('hidden', hidden); }
function updateLevelMeter(bar, valueEl, trackEl, pct, level) {
  if (bar)     bar.style.width     = `${pct}%`;
  if (valueEl) valueEl.textContent = levelToDb(level);
  if (trackEl) trackEl.setAttribute('aria-valuenow', pct);
}
function resetLevelMeter(bar, valueEl, trackEl) { updateLevelMeter(bar, valueEl, trackEl, 0, 0); }
function levelToDb(level) { return level > 0.001 ? `${Math.round(20 * Math.log10(level))} dB` : '—'; }
function hideMicError() { if (micError) micError.hidden = true; }
function showMicError(msg) { if (micError) micError.hidden = false; if (micErrorText) micErrorText.textContent = msg; }
function formatTime(s) { if (!s || isNaN(s)) return '0:00'; return `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`; }
function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

// ═══════════════════════════════════════════════════════════════
// BOOTSTRAP
// ═══════════════════════════════════════════════════════════════

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

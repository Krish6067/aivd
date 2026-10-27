/**
 * audio.js — VOXSHIELD
 *
 * Handles:
 *  - Microphone/device permission and Web Audio API setup
 *  - AnalyserNode for waveform + level data
 *  - Elapsed-time counter
 *  - Device enumeration for Live Call tab
 *  - Clean resource teardown
 */

// ─── Audio State ─────────────────────────────────────────────────────────────

/** @type {AudioContext|null} */
let audioContext = null;

/** @type {MediaStream|null} */
let micStream = null;

/** @type {MediaStreamAudioSourceNode|null} */
let sourceNode = null;

/** @type {AnalyserNode|null} */
let analyserNode = null;

/** @type {Uint8Array|null} */
let timeDomainBuffer = null;

/** Raw microphone frames, consumed by the Live Call WebSocket. */
let streamingProcessor = null;
let silentGain = null;
let pcmFrames = [];
let captureSampleRate = 16000;

/** @type {boolean} Whether the mic is currently active */
let isMonitoring = false;

/** @type {number} Session start timestamp (ms) */
let sessionStartTime = 0;

// ─── Settings ─────────────────────────────────────────────────────────────────

let settings = {
  fftSize: 1024,
  smoothingTimeConstant: 0.8,
};

/**
 * Update audio settings. Takes effect on next startMonitoring() call.
 * @param {{ fftSize?: number, smoothingTimeConstant?: number }} newSettings
 */
export function applySettings(newSettings) {
  settings = { ...settings, ...newSettings };
}

// ─── Device Enumeration ───────────────────────────────────────────────────────

/**
 * Enumerate available audio input devices.
 * Device labels are only populated after getUserMedia permission has been granted.
 *
 * @returns {Promise<MediaDeviceInfo[]>}
 */
export async function enumerateAudioInputDevices() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    return [];
  }
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter(d => d.kind === 'audioinput');
  } catch (_) {
    return [];
  }
}

// ─── Monitoring ───────────────────────────────────────────────────────────────

/**
 * Start microphone monitoring, optionally targeting a specific device.
 *
 * @param {string} [deviceId] - Optional MediaDeviceInfo.deviceId to target
 * @returns {Promise<{ analyser: AnalyserNode, buffer: Uint8Array }>}
 * @throws {Error} with a user-friendly message on failure
 */
export async function startMonitoring(deviceId) {
  if (isMonitoring) {
    throw new Error('Monitoring is already active. Stop it before starting again.');
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      'Your browser does not support microphone access (getUserMedia). ' +
      'Please try a modern browser such as Chrome, Firefox, or Edge.'
    );
  }

  // Build audio constraints — use specific device if provided
  const audioConstraints = deviceId
    ? { deviceId: { exact: deviceId } }
    : true;

  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: audioConstraints,
      video: false,
    });
  } catch (err) {
    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      throw new Error(
        'Microphone permission was denied. ' +
        'Click the lock icon in your browser address bar and allow microphone access, then try again.'
      );
    }
    if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
      throw new Error(
        'No microphone was found on this device. ' +
        'Connect a microphone and try again.'
      );
    }
    if (err.name === 'OverconstrainedError') {
      throw new Error(
        'The selected audio device is no longer available. ' +
        'Refresh the device list and try again.'
      );
    }
    if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
      throw new Error(
        'Your microphone is busy or could not be started. ' +
        'Close other applications using the microphone, then try again.'
      );
    }
    throw new Error(`Microphone error: ${err.message}`);
  }

  // Build Web Audio graph: source → analyser (NOT connected to destination)
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  sourceNode   = audioContext.createMediaStreamSource(micStream);
  analyserNode = audioContext.createAnalyser();

  analyserNode.fftSize               = settings.fftSize;
  analyserNode.smoothingTimeConstant = settings.smoothingTimeConstant;

  sourceNode.connect(analyserNode);
  // Intentionally NOT connecting analyser to destination — no mic playback

  // ScriptProcessor keeps a raw PCM tap separate from the visual analyser.
  // Route it through a muted gain node so browsers continue processing it
  // without playing microphone audio back through the speakers.
  captureSampleRate = audioContext.sampleRate;
  streamingProcessor = audioContext.createScriptProcessor(4096, 1, 1);
  silentGain = audioContext.createGain();
  silentGain.gain.value = 0;
  streamingProcessor.onaudioprocess = (event) => {
    pcmFrames.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    // Cap queued audio at roughly two seconds if the network is unavailable.
    const maxFrames = Math.ceil((captureSampleRate * 2) / 4096);
    if (pcmFrames.length > maxFrames) pcmFrames.splice(0, pcmFrames.length - maxFrames);
  };
  sourceNode.connect(streamingProcessor);
  streamingProcessor.connect(silentGain);
  silentGain.connect(audioContext.destination);

  timeDomainBuffer = new Uint8Array(analyserNode.frequencyBinCount);

  sessionStartTime = Date.now();
  isMonitoring     = true;

  return { analyser: analyserNode, buffer: timeDomainBuffer };
}

/**
 * Stop monitoring and release all audio resources.
 * @returns {{ durationMs: number }}
 */
export function stopMonitoring() {
  const durationMs = isMonitoring ? Date.now() - sessionStartTime : 0;

  isMonitoring = false;

  if (sourceNode) {
    try { sourceNode.disconnect(); } catch (_) {}
    sourceNode = null;
  }
  if (analyserNode) {
    try { analyserNode.disconnect(); } catch (_) {}
    analyserNode = null;
  }
  if (streamingProcessor) {
    try { streamingProcessor.disconnect(); } catch (_) {}
    streamingProcessor.onaudioprocess = null;
    streamingProcessor = null;
  }
  if (silentGain) {
    try { silentGain.disconnect(); } catch (_) {}
    silentGain = null;
  }
  if (micStream) {
    micStream.getTracks().forEach(track => track.stop());
    micStream = null;
  }
  if (audioContext) {
    audioContext.close().catch(() => {});
    audioContext = null;
  }

  timeDomainBuffer = null;
  pcmFrames = [];
  return { durationMs };
}

// ─── Query Helpers ────────────────────────────────────────────────────────────

/** @returns {boolean} */
export function getIsMonitoring() { return isMonitoring; }

/**
 * Get the current audio RMS level normalized to 0–1.
 * @returns {number}
 */
export function getAudioLevel() {
  if (!analyserNode || !timeDomainBuffer) return 0;
  analyserNode.getByteTimeDomainData(timeDomainBuffer);
  let sum = 0;
  for (let i = 0; i < timeDomainBuffer.length; i++) {
    const n = (timeDomainBuffer[i] / 128.0) - 1.0;
    sum += n * n;
  }
  const rms = Math.sqrt(sum / timeDomainBuffer.length);
  return Math.min(rms * 3.5, 1.0);
}

/**
 * Get current waveform time-domain byte data.
 * @returns {Uint8Array|null}
 */
export function getWaveformData() {
  if (!analyserNode || !timeDomainBuffer) return null;
  analyserNode.getByteTimeDomainData(timeDomainBuffer);
  return timeDomainBuffer;
}

/**
 * Return queued microphone PCM resampled to the model's 16 kHz input rate.
 * Audio is removed from memory as soon as it is handed to the WebSocket.
 */
export function consumePcmSamples(targetSampleRate = 16000) {
  if (pcmFrames.length === 0) return null;
  const frames = pcmFrames.splice(0);
  const ratio = captureSampleRate / targetSampleRate;
  const outputLength = frames.reduce((total, frame) => total + Math.floor(frame.length / ratio), 0);
  const output = new Float32Array(outputLength);
  let cursor = 0;
  for (const frame of frames) {
    for (let index = 0; index < frame.length; index += ratio) {
      if (cursor < output.length) output[cursor++] = frame[Math.floor(index)];
    }
  }
  return cursor === output.length ? output : output.slice(0, cursor);
}

/**
 * Get elapsed time as "MM:SS".
 * @returns {string}
 */
export function getElapsedTime() {
  if (!isMonitoring) return '00:00';
  const totalSec = Math.floor((Date.now() - sessionStartTime) / 1000);
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

/** @returns {number} elapsed milliseconds */
export function getElapsedMs() {
  return isMonitoring ? Date.now() - sessionStartTime : 0;
}

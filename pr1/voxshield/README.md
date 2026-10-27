# VOXSHIELD — Frontend Demo

**Real-Time Audio Authenticity Monitor**  
A premium cybersecurity dashboard frontend for AI-generated voice and voice-cloning detection.

> **This is a frontend-only demonstration.** Live waveform visualization uses microphone input for local rendering only — no audio is transmitted or stored. Voice-cloning detection is not active until a genuine anti-spoofing model is integrated.

---

## Project Structure

```
voxshield/
├── index.html          # Dashboard structure and accessible controls
├── css/
│   └── styles.css      # Monochromatic design, animations, responsive layout
├── js/
│   ├── app.js          # UI state, event listeners, session history
│   ├── audio.js        # Web Audio API, mic permissions, level, cleanup
│   ├── waveform.js     # Canvas waveform rendering (live + idle)
│   └── score-ring.js   # SVG score ring, demo mode, animation
├── assets/
│   └── icons/          # (Reserved for future custom icons)
└── README.md
```

---

## Local Setup

The application uses ES6 modules and the Web Audio API, both of which require a local HTTP server (not `file://`).

### Option A — Python (recommended, no installation)

```bash
# From inside the voxshield/ folder:
python -m http.server 8000
```

Then open: **http://localhost:8000**

### Option B — Node.js (`npx serve`)

```bash
npx serve .
```

### Option C — VS Code Live Server

Install the **Live Server** extension, right-click `index.html`, and choose **Open with Live Server**.

---

## Usage

1. Open **http://localhost:8000** in Chrome, Firefox, or Edge.
2. The dashboard loads with the waveform in idle mode and a **DEMO SCORE** of 94%.
3. Click **START MONITORING** — your browser will request microphone permission.
4. Grant permission. The waveform will render your actual microphone input in real time.
5. Speak into the microphone; the waveform and level meter will respond to your voice.
6. Click **STOP MONITORING** to release the microphone. The session is logged in the history panel.
7. Toggle **DEMO MODE** off to see the "Waiting for Analysis" state (for when a real model is connected).
8. Use the ⚙ settings icon to adjust FFT size and smoothing (takes effect on next session).
9. Click **CLEAR** in the session history panel to reset the session log.

---

## Browser Compatibility

| Feature | Chrome | Firefox | Edge | Safari |
|---|---|---|---|---|
| Web Audio API | ✅ | ✅ | ✅ | ✅ |
| getUserMedia | ✅ | ✅ | ✅ | ✅ (14+) |
| ES Modules | ✅ | ✅ | ✅ | ✅ |
| Canvas | ✅ | ✅ | ✅ | ✅ |

> **HTTPS or localhost required** for microphone access. File protocol (`file://`) will not work.

---

## Known Limitations

- **No AI detection** — voice-cloning detection is not implemented. The score ring shows an illustrative demo value.
- **No backend** — no server, database, or WebSocket connection exists in this phase.
- **No audio storage** — microphone audio is never recorded, saved, or transmitted.
- **No authentication** — user accounts and access control are not part of this phase.
- **Session history is in-memory only** — it resets on page reload.

---

## Integrating a Real Model (Future Work)

The frontend is structured to make backend integration straightforward:

1. **Connect a backend** (e.g. FastAPI) via REST or WebSocket.
2. In `audio.js`, stream or send audio chunks to your endpoint.
3. Receive a detection score and call `setScore(score, label)` from `score-ring.js`.
4. Turn off demo mode (`demoToggle.checked = false`) to show live scores.

```js
// Example future integration hook in app.js:
import { setScore } from './score-ring.js';

// When your model returns a result:
setScore(87.4, 'AUTHENTICITY SCORE');
```

---

## Security and Privacy

- Microphone audio is processed locally in the browser using the Web Audio API.
- No audio data is sent to any server in this demo phase.
- No audio is recorded or stored to disk.
- Microphone tracks are explicitly stopped (`track.stop()`) when monitoring ends.
- The AudioContext is closed when monitoring ends to release all system resources.

---

## License

Internal demonstration prototype. Not for distribution.

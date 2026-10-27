# VOXSHIELD Backend — Complete Setup & Usage Guide

## Overview

VOXSHIELD is a real-time AI-generated voice detection system powered by
**AASIST** (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph
Attention Networks, ICASSP 2022).

This backend provides:
- `POST /api/analyze` — upload an audio file and receive a detection score
- `WS /ws/analyze` — stream live audio and receive periodic analysis results
- `GET /api/health` — liveness probe
- `GET /api/model/status` — model readiness and configuration
- Interactive API docs at `http://localhost:8001/docs`

> **Important:** Detection results are probabilistic. AASIST was trained on
> ASVspoof 2019 LA and may not generalise to all modern synthesis methods.
> Do not use results as sole evidence of fraud.

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.11–3.14 | 3.14 tested |
| pip | latest | bundled with Python |
| FFmpeg | any | optional — needed only for MP3/M4A without libsndfile support |

---

## 1 — Create and activate a virtual environment

**Windows PowerShell:**
```powershell
cd c:\work\AIVD\pr1\voxshield-backend

python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
cd /path/to/voxshield-backend
python3 -m venv .venv
source .venv/bin/activate
```

---

## 2 — Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

> GPU inference (optional): after installing the requirements, replace the
> `torch` / `torchaudio` packages with the CUDA build:
> ```
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 3 — Configure environment variables

```powershell
Copy-Item .env.example .env
```

Open `.env` and adjust at minimum:

```ini
# Allow your VOXSHIELD frontend origin
ALLOWED_ORIGINS=http://localhost:8000

# Path to downloaded model weights
MODEL_WEIGHTS_PATH=models/AASIST/weights/AASIST.pth

# cpu | cuda | mps  (leave blank for auto-detect)
DEVICE=
```

---

## 4 — Download AASIST pretrained weights

```powershell
python models/AASIST/download_weights.py
```

Expected output:
```
VOXSHIELD — AASIST Weight Downloader
==================================================
  Downloading https://github.com/clovaai/aasist/...
  ✓ Downloaded X.X MB → models/AASIST/weights/AASIST.pth
```

If the automatic download fails, see `models/AASIST/README.md` for manual
download instructions.

---

## 5 — Start the server

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

The server starts at **http://localhost:8001**.

---

## 6 — Verify the server is healthy

```powershell
curl http://localhost:8001/api/health
```
```json
{"status": "ok", "version": "0.1.0", "timestamp": "..."}
```

---

## 7 — Check model readiness

```powershell
curl http://localhost:8001/api/model/status
```

When the model has loaded:
```json
{
  "readiness": "ready",
  "model_name": "AASIST",
  "device": "cpu",
  "supported_sample_rate": 16000
}
```

If `readiness` is `"not_ready"`, check the `error_detail` field and
re-run the download script.

---

## 8 — Test file upload analysis

```powershell
# Test with a WAV file
curl -X POST http://localhost:8001/api/analyze `
  -F "file=@path\to\your_audio.wav;type=audio/wav"
```

Example response:
```json
{
  "session_id": "...",
  "classification": "likely_authentic",
  "raw_score": 0.874,
  "score_type": "bonafide_probability",
  "score_interpretation": "Score 0.874 ≥ 0.70 — characteristics consistent with natural speech.",
  "model_name": "AASIST",
  "processing_latency_ms": 312.4,
  "audio_duration_seconds": 5.02,
  "sample_rate": 16000,
  "warnings": []
}
```

Full interactive API docs: **http://localhost:8001/docs**

---

## 9 — Test WebSocket streaming

Open your browser console on the VOXSHIELD frontend (`http://localhost:8000`)
and the frontend will connect automatically once you click **Start Call Monitor**.

To test manually with `websocat` (install via `cargo install websocat`):

```bash
websocat ws://localhost:8001/ws/analyze
```
Then type:
```json
{"type": "session_init", "sample_rate": 16000, "channels": 1, "encoding": "pcm_float32"}
```

---

## 10 — Run the test suite

```powershell
# All tests (model-dependent tests auto-skip if weights are missing)
pytest tests/ -v

# Only tests that don't need weights
pytest tests/ -v -k "not real_inference"

# With coverage report
pip install pytest-cov
pytest tests/ --cov=app --cov-report=term-missing
```

---

## 11 — Connect the VOXSHIELD frontend

The frontend at `c:\work\AIVD\pr1\voxshield\` is served on port **8000**.
The backend runs on port **8001**.

CORS is pre-configured for `http://localhost:8000`.

The frontend's `js/app.js` calls:
- `POST http://localhost:8001/api/analyze` for file uploads
- `WS ws://localhost:8001/ws/analyze` for live call monitoring
- `GET http://localhost:8001/api/model/status` on load

Run both servers simultaneously:

```powershell
# Terminal 1 — Frontend
cd c:\work\AIVD\pr1\voxshield
python -m http.server 8000

# Terminal 2 — Backend
cd c:\work\AIVD\pr1\voxshield-backend
.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

---

## Project Structure

```
voxshield-backend/
├── app/
│   ├── main.py                  FastAPI app factory, startup, CORS
│   ├── config.py                Pydantic-settings, env var schema
│   ├── api/
│   │   ├── health.py            GET /api/health, GET /api/model/status
│   │   ├── analysis.py          POST /api/analyze (file upload)
│   │   └── websocket.py         WS /ws/analyze (streaming)
│   ├── schemas/
│   │   └── analysis.py          Pydantic request/response models
│   ├── services/
│   │   ├── audio_processing.py  Decode, validate, resample, tensorise
│   │   ├── model_service.py     AASIST singleton, async inference
│   │   └── streaming_service.py WebSocket session buffer management
│   └── utils/
│       └── validation.py        Magic bytes, size, duration, content guards
├── models/
│   └── AASIST/
│       ├── model_architecture.py AASIST PyTorch architecture (MIT)
│       ├── download_weights.py   Automated weight downloader
│       └── README.md             Model docs and score interpretation
├── tests/
│   ├── conftest.py              Shared fixtures, mock model, audio generators
│   ├── test_health.py           Health + model status
│   ├── test_upload_validation.py Upload validation edge cases
│   ├── test_analysis.py         Inference pipeline + audio processing units
│   └── test_websocket.py        WS protocol + streaming session units
├── datasets/
│   └── README.md               Dataset licensing and access instructions
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md                   ← You are here
```

---

## Score Interpretation

| Softmax P(bonafide) | Classification | Meaning |
|---|---|---|
| ≥ 0.70 | `likely_authentic` | Characteristics consistent with natural speech |
| 0.40–0.70 | `inconclusive` | Ambiguous — longer audio may improve accuracy |
| < 0.40 | `potential_synthetic_speech` | Characteristics associated with synthesised voice |

These thresholds are calibrated against AASIST's EER on ASVspoof 2019 LA.
Re-validate for your deployment context.

---

## Security Notes

This backend is configured for **local development**.

Before deploying to production:

- [ ] Use HTTPS/WSS (reverse proxy: Nginx, Caddy, or cloud load balancer)
- [ ] Restrict `ALLOWED_ORIGINS` to the exact deployed frontend URL
- [ ] Add authentication (API keys, OAuth2, JWT)
- [ ] Enable rate limiting (e.g. slowapi or a gateway)
- [ ] Run behind a firewall; never expose port 8001 directly
- [ ] Use a secrets manager for credentials (not `.env` files)
- [ ] Enable structured logging and monitoring
- [ ] Review and reduce `WS_MAX_SESSIONS` for your hardware capacity
- [ ] Conduct a security review before handling real call audio

---

## Known Limitations

1. **Model generalisation:** AASIST was trained on ASVspoof 2019 LA. It may
   not detect attacks from systems released after 2019 (e.g. VALL-E, Voicebox).
2. **WebSocket audio source:** The browser cannot capture desktop call audio
   (Zoom, Teams, phone) without OS-level virtual audio routing. The frontend
   explains this in the Live Call tab setup guide.
3. **CPU inference latency:** ~100–500 ms per 4-second window on CPU.
   GPU (CUDA) reduces this to ~20–80 ms.
4. **No persistent storage:** Session results are not stored. Integrate a
   database (PostgreSQL, SQLite) if audit logging is required.

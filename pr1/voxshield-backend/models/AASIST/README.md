# AASIST Model — Pretrained Weights

## Model Overview

| Property | Value |
|---|---|
| **Name** | AASIST |
| **Full name** | Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks |
| **Authors** | Jee-weon Jung et al. (Clova AI / NAVER Corp.) |
| **Paper** | [ICASSP 2022 — arXiv:2110.01200](https://arxiv.org/abs/2110.01200) |
| **Source code** | [github.com/clovaai/aasist](https://github.com/clovaai/aasist) |
| **License** | MIT |
| **Training dataset** | ASVspoof 2019 Logical Access (LA) |
| **Reported EER** | 0.83% on ASVspoof 2019 LA eval set |

## Why AASIST?

AASIST was selected for VOXSHIELD because it:
- Has **publicly available pretrained weights** under a permissive license.
- Takes **raw waveform input** (no hand-crafted features required).
- Achieves **state-of-the-art EER** on the standard ASVspoof 2019 benchmark.
- Has **reasonable CPU inference speed** (~100–400 ms per 4-second window).
- Is **well-documented** with a reproducible training recipe.

> **Limitation:** AASIST was trained and evaluated on ASVspoof 2019 LA, which
> contains TTS and voice-conversion attacks from 2019. Its generalisation to
> newer synthesis methods (e.g. VALL-E, Voicebox) has not been independently
> validated. Treat detection results as probabilistic estimates.

## Score Interpretation

| Softmax P(bonafide) | Classification |
|---|---|
| ≥ 0.70 | `likely_authentic` |
| 0.40–0.70 | `inconclusive` |
| < 0.40 | `potential_synthetic_speech` |

These thresholds are derived from the paper's EER analysis on ASVspoof 2019 LA.
**Re-validate on your deployment distribution before production use.**

## Required Input Format

| Property | Value |
|---|---|
| Sample rate | 16 000 Hz |
| Channels | Mono (1) |
| Dtype | float32 |
| Shape | `[batch, samples]` |
| Recommended duration | ≥ 4 seconds |
| Minimum duration | 1 second (accuracy reduced) |
| Normalisation | None (model does not require amplitude normalisation) |

## Downloading Weights

### Option A — Automatic (recommended)

From the `voxshield-backend/` directory:

```powershell
python models/AASIST/download_weights.py
```

### Option B — Manual

1. Visit [github.com/clovaai/aasist → Releases](https://github.com/clovaai/aasist/releases)
2. Download `AASIST.pth`
3. Place it at:
   ```
   models/AASIST/weights/AASIST.pth
   ```
4. Start the server — the model loads automatically at startup.

### Verifying the model loaded

```powershell
curl http://localhost:8001/api/model/status
```

Expected response when ready:
```json
{
  "readiness": "ready",
  "model_name": "AASIST",
  "device": "cpu"
}
```

## Directory Layout

```
models/AASIST/
├── model_architecture.py   ← AASIST PyTorch architecture (MIT, adapted from clovaai)
├── download_weights.py     ← Automated weight downloader
├── weights/
│   └── AASIST.pth          ← Pretrained checkpoint (not in git, download above)
└── README.md               ← This file
```

## Training Your Own Checkpoint

If the pretrained weights are unavailable:

1. **Dataset:** Request access to [ASVspoof 2019](https://www.asvspoof.org/).
   - License: Research use only.
   - ~10 GB LA partition.
2. **Training code:** Clone [clovaai/aasist](https://github.com/clovaai/aasist) and follow their README.
3. **Replace weights path** in `.env`:
   ```
   MODEL_WEIGHTS_PATH=models/AASIST/weights/your_checkpoint.pth
   ```

> Do not use a randomly initialised model for inference — it produces
> meaningless scores. The backend reports `MODEL NOT READY` if no valid
> checkpoint is found.

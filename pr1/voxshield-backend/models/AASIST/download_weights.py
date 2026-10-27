#!/usr/bin/env python3
"""
models/AASIST/download_weights.py — VOXSHIELD Backend

Downloads the official AASIST pretrained weights from the clovaai/aasist
GitHub repository and saves them to models/AASIST/weights/AASIST.pth.

Usage (from voxshield-backend/ directory):
    python models/AASIST/download_weights.py

Requirements: pip install requests (included via requirements.txt)
License of weights: MIT (same as AASIST source code)
Source: https://github.com/clovaai/aasist
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

# ─── Download target ──────────────────────────────────────────────────────────

# Official AASIST pretrained checkpoint (AASIST model, ASVspoof 2019 LA).
# URL points to the GitHub release asset — update if the upstream moves it.
WEIGHTS_URLS = [
    # Primary: GitHub releases
    "https://github.com/clovaai/aasist/releases/download/v1.0.0/AASIST.pth",
    # Mirror: direct raw GitHub (if a weights file is committed to repo)
    "https://raw.githubusercontent.com/clovaai/aasist/main/models/weights/AASIST.pth",
]

WEIGHTS_DIR  = Path(__file__).parent / "weights"
WEIGHTS_FILE = WEIGHTS_DIR / "AASIST.pth"

# SHA-256 of the official checkpoint (verify after download).
# Set to None to skip verification if the hash is unknown.
EXPECTED_SHA256: str | None = "51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0"


# ─── Download logic ───────────────────────────────────────────────────────────


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, chunk_size: int = 65536) -> None:
    import urllib.request

    print(f"  Downloading {url}")
    print(f"  -> {dest}")

    with urllib.request.urlopen(url, timeout=60) as response:
        total = response.headers.get("Content-Length")
        total = int(total) if total else None
        downloaded = 0

        with open(dest, "wb") as out:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:5.1f}% ({downloaded // 1024} KB)", end="", flush=True)
        print()


def main() -> None:
    print("VOXSHIELD — AASIST Weight Downloader")
    print("=" * 50)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if WEIGHTS_FILE.exists():
        size_mb = WEIGHTS_FILE.stat().st_size / (1024 ** 2)
        print(f"  Weights already present: {WEIGHTS_FILE}  ({size_mb:.1f} MB)")
        if EXPECTED_SHA256:
            actual = sha256_of_file(WEIGHTS_FILE)
            if actual == EXPECTED_SHA256:
                print("  SHA-256 checksum: OK")
            else:
                print(f"  WARNING: checksum mismatch!\n"
                      f"    Expected: {EXPECTED_SHA256}\n"
                      f"    Actual:   {actual}")
        return

    last_error: Exception | None = None
    for url in WEIGHTS_URLS:
        try:
            download(url, WEIGHTS_FILE)
            last_error = None
            break
        except Exception as exc:
            print(f"  Failed: {exc}")
            last_error = exc
            if WEIGHTS_FILE.exists():
                WEIGHTS_FILE.unlink()

    if last_error is not None:
        print("\n  ERROR: All download attempts failed.")
        print("\n  Manual download instructions:")
        print("  -----------------------------")
        print("  1. Visit: https://github.com/clovaai/aasist")
        print("  2. Go to Releases → download AASIST.pth")
        print(f"  3. Save to:  {WEIGHTS_FILE}")
        print("  4. Re-run this script to verify the file.\n")
        sys.exit(1)

    # Verify download
    size_mb = WEIGHTS_FILE.stat().st_size / (1024 ** 2)
    print(f"\n  Downloaded {size_mb:.1f} MB -> {WEIGHTS_FILE}")

    if EXPECTED_SHA256:
        actual = sha256_of_file(WEIGHTS_FILE)
        if actual == EXPECTED_SHA256:
            print("  SHA-256 checksum verified.")
        else:
            print(f"  ERROR: Checksum mismatch!\n"
                  f"    Expected: {EXPECTED_SHA256}\n"
                  f"    Actual:   {actual}\n"
                  "  The file may be corrupted. Delete it and re-run this script.")
            sys.exit(1)
    else:
        print(f"  SHA-256: {sha256_of_file(WEIGHTS_FILE)}")
        print("  (Add this hash to EXPECTED_SHA256 in download_weights.py for future verification.)")

    print("\n  Ready. Start the backend with:")
    print("  uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload\n")


if __name__ == "__main__":
    main()

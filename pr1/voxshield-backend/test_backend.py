import requests
import json
import numpy as np
import soundfile as sf
import io

# Create test audio (2 seconds of sine wave)
sr = 16000
t = np.arange(2 * sr) / sr
audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
buf = io.BytesIO()
sf.write(buf, audio, sr, format='WAV', subtype='FLOAT')
buf.seek(0)

# Test 1: File Upload Analysis (POST /api/analyze)
print('=== Test 1: File Upload Analysis ===')
files = {'file': ('test.wav', buf, 'audio/wav')}
r = requests.post('http://localhost:8001/api/analyze', files=files, timeout=30)
print(f'Status: {r.status_code}')
data = r.json()
print(f'Classification: {data["classification"]}')
print(f'Raw Score: {data["raw_score"]:.6f}')
print(f'Interpretation: {data["score_interpretation"][:80]}...')
print()

# Test 2: Health check
print('=== Test 2: Health Check ===')
r = requests.get('http://localhost:8001/api/health', timeout=5)
print(f'Status: {r.status_code}')
print(f'Response: {r.json()}')
print()

# Test 3: Model Status
print('=== Test 3: Model Status ===')
r = requests.get('http://localhost:8001/api/model/status', timeout=5)
print(f'Status: {r.status_code}')
data = r.json()
print(f'Model: {data["model_name"]} v{data["model_version"]}')
print(f'Readiness: {data["readiness"]}')
print(f'Device: {data["device"]}')
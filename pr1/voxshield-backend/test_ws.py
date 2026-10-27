import asyncio
import json
import websockets
import numpy as np

async def test_ws():
    uri = 'ws://localhost:8001/ws/analyze'
    print('=== Test 4: WebSocket Streaming Analysis ===')
    async with websockets.connect(uri) as ws:
        # Send session_init
        await ws.send(json.dumps({
            'type': 'session_init',
            'sample_rate': 16000,
            'channels': 1,
            'encoding': 'pcm_float32'
        }))
        
        # Receive session_started
        msg = await ws.recv()
        data = json.loads(msg)
        print(f'Session started: {data["session_id"]}')
        print(f'Model: {data["model_name"]} v{data["model_version"]}')
        print(f'Min buffer: {data["min_buffer_seconds"]}s')
        print(f'Inference interval: {data["inference_interval_seconds"]}s')
        
        # Send 5 seconds of audio (sine wave)
        sr = 16000
        duration = 5
        t = np.arange(duration * sr) / sr
        audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Send in chunks
        chunk_size = 8000  # 0.5 seconds
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            await ws.send(chunk.tobytes())
            await asyncio.sleep(0.1)
        
        # Wait for analysis results
        print('Waiting for analysis results...')
        for _ in range(3):
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                data = json.loads(msg)
                if data.get('type') == 'analysis_result':
                    print(f'Analysis: {data["classification"]}, Score: {data["raw_score"]:.6f}, Latency: {data["processing_latency_ms"]:.0f}ms')
            except asyncio.TimeoutError:
                print('Timeout')
                break

asyncio.run(test_ws())
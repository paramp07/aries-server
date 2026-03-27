import asyncio
import numpy as np
import requests
import json
import os
import time
from app.ml.audio.yamnet import AudioInference
from app.services.serial_reader import SerialAudioReader
from app.core.broadcast import broadcast

# Configuration
SERIAL_PORT = os.getenv("SERIAL_PORT", "COM3")
BAUD_RATE = int(os.getenv("BAUD_RATE", 921600))
WEBHOOK_URL = os.getenv("NEXTJS_WEBHOOK_URL", "http://localhost:3000/api/audio/webhook")

class InferenceStreamer:
    def __init__(self, port=None, baudrate=None, webhook_url=None):
        # Configuration - Prefer arguments, then environment, then defaults
        self.port = port or os.getenv("SERIAL_PORT", "COM3")
        self.baudrate = baudrate or int(os.getenv("BAUD_RATE", 921600))
        self.webhook_url = webhook_url or os.getenv("NEXTJS_WEBHOOK_URL", "http://localhost:3000/api/audio/webhook")
        
        self.inference = AudioInference()
        self.reader = SerialAudioReader(port=self.port, baudrate=self.baudrate)
        self.target_sr = 16000
        self.window_s = 1.0 # 1 second window for YAMNet
        self.step_s = 0.25 # 0.25 second step
        self.audio_buffer = np.zeros(int(self.target_sr * self.window_s), dtype=np.float32)

    async def run(self):
        print(f"Starting Inference Streamer on {self.reader.port}...")
        self.reader.start()
        
        try:
            while True:
                # Try to get a chunk of audio
                chunk = self.reader.get_audio_chunk(duration_s=self.step_s)
                
                if chunk is not None:
                    # Update rolling buffer
                    self.audio_buffer = np.roll(self.audio_buffer, -len(chunk))
                    self.audio_buffer[-len(chunk):] = chunk
                    
                    # Run inference
                    top_class, human_prob = self.inference.run_inference(self.audio_buffer)
                    
                    # Prepare result with extra fields for Next.js compatibility
                    result = {
                        "type": "audio_inference",
                        "top_sound": top_class,
                        "human_vocal_percent": round(human_prob * 100, 2),
                        "value": float(human_prob), # for AcousticAnalysis.jsx
                        "confidence": float(human_prob),
                        "timestamp": time.time()
                    }
                    
                    # Log to console
                    print(f"[{result['top_sound']}] Human Vocal: {result['human_vocal_percent']}%")
                    
                    # 1. Send to internal bridge (Solves cross-process issue)
                    # This relays data from this process to the main server process
                    asyncio.create_task(self.send_to_bridge(result))
                    
                    # 2. Send via Webhook to Next.js (Optional / External)
                    if self.webhook_url and (("localhost:3000" not in self.webhook_url) or os.getenv("ENABLE_WEBHOOK")):
                        asyncio.create_task(self.send_webhook(result))
                
                # Sleep a bit to prevent CPU pinning
                await asyncio.sleep(0.05)
                
        except KeyboardInterrupt:
            print("Stopping Streamer...")
        finally:
            self.reader.stop()

    async def send_to_bridge(self, data):
        """Sends data to the main server process via an internal API call."""
        try:
            loop = asyncio.get_event_loop()
            # USE 127.0.0.1 instead of localhost for Windows reliability
            # Increased timeout to 0.5s to handle high CPU load during inference
            await loop.run_in_executor(
                None, 
                lambda: requests.post("http://127.0.0.1:8000/ws/audio/data", json=data, timeout=0.5)
            )
        except Exception as e:
            # Silent fail if server is not up yet, but we could log it during debug
            # print(f"Bridge error: {e}")
            pass 

    async def send_webhook(self, data):
        try:
            # Using loop.run_in_executor for synchronous requests call
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.post(self.webhook_url, json=data, timeout=1.0)
            )
            if response.status_code == 200:
                print(f" → Webhook delivered to {self.webhook_url}")
            else:
                print(f" ! Webhook failed with status: {response.status_code}")
        except Exception as e:
            # Webhook might fail if client is not running, we don't want to crash
            pass

if __name__ == "__main__":
    streamer = InferenceStreamer()
    asyncio.run(streamer.run())

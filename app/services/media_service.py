import asyncio
import numpy as np
import time
from typing import List
from fastapi import WebSocket
from app.ml.audio.yamnet import AudioInference
from app.api.ws.audio import broadcast_audio

class MediaService:
    def __init__(self):
        self.inference = AudioInference()
        self.target_sr = 16000
        self.window_s = 1.0  # 1 second window for YAMNet
        self.audio_buffer = np.zeros(int(self.target_sr * self.window_s), dtype=np.float32)
        
        # Accumulator for triggering inference
        self.samples_since_last_inference = 0
        self.inference_threshold = 4000  # 0.25 seconds of new audio triggers inference
        
        # Latest video frame for MJPEG streaming
        self.latest_frame = None
        self.frame_event = asyncio.Event()
        
        # Binary audio clients (Next.js)
        self.audio_clients: List[WebSocket] = []

    def update_video_frame(self, frame_bytes: bytes):
        """Update the latest JPEG frame received from ESP32."""
        self.latest_frame = frame_bytes
        self.frame_event.set()
        self.frame_event.clear()

    async def add_audio_chunk(self, pcm_bytes: bytes):
        """Add raw 16-bit PCM bytes to the buffer and run inference."""
        
        # 1. Broadcast raw bytes to any listening web clients
        await self.broadcast_raw_audio(pcm_bytes)
        
        # 2. Convert to float32 for AI
        audio_data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 3. Update rolling buffer
        self.audio_buffer = np.roll(self.audio_buffer, -len(audio_data))
        self.audio_buffer[-len(audio_data):] = audio_data
        
        # 4. Trigger inference based on accumulated samples
        self.samples_since_last_inference += len(audio_data)
        
        if self.samples_since_last_inference >= self.inference_threshold:
            self.samples_since_last_inference = 0
            
            # Run inference on the full 1-second rolling buffer
            top_class, human_prob = self.inference.run_inference(self.audio_buffer)
            
            result = {
                "type": "audio_inference",
                "top_sound": top_class,
                "human_vocal_percent": round(human_prob * 100, 2),
                "value": float(human_prob),
                "confidence": float(human_prob),
                "timestamp": time.time()
            }
            
            # Log to console for verification
            print(f"👂 AI Listening: {top_class:<15} (Vocal: {result['human_vocal_percent']}%)")
            
            # Broadcast to the existing JSON audio result channel
            await broadcast_audio(result)

    async def broadcast_raw_audio(self, pcm_bytes: bytes):
        """Send raw binary PCM to connected Next.js audio clients."""
        if not self.audio_clients:
            return
            
        disconnected = []
        for client in self.audio_clients:
            try:
                await client.send_bytes(pcm_bytes)
            except Exception:
                disconnected.append(client)
        
        for client in disconnected:
            if client in self.audio_clients:
                self.audio_clients.remove(client)

    async def get_video_generator(self):
        """Generator for FastAPI StreamingResponse (MJPEG)."""
        while True:
            await self.frame_event.wait()
            if self.latest_frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + self.latest_frame + b'\r\n')

# Singleton instance
media_service = MediaService()

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List
import json

router = APIRouter()

# Store all connected Audio clients (Next.js)
audio_clients: List[WebSocket] = []

class AudioData(BaseModel):
    type: str # Always "audio_inference"
    top_sound: str
    human_vocal_percent: float
    value: float
    confidence: float
    timestamp: float

async def broadcast_audio(data: dict):
    """Internal helper to broadcast audio results to all connected audio clients."""
    disconnected = []
    for client in audio_clients:
        try:
            await client.send_json(data)
        except Exception:
            disconnected.append(client)
    
    for client in disconnected:
        if client in audio_clients:
            audio_clients.remove(client)
    
    if len(audio_clients) > 0:
        print(f"📡 Audio AI Broadcast to {len(audio_clients)} clients: {data.get('top_sound')}")

# Internal helper for easy reference
app_post_bridge_dummy = None 

@router.post("/data")
async def receive_audio_ingestion(data: AudioData):
    """Bridge endpoint for the AI process to send results."""
    # print(f"📥 Bridge received audio data: {data.top_sound}")
    await broadcast_audio(data.model_dump())
    return {"status": "relayed"}

@router.websocket("")
async def audio_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for Next.js clients only interested in audio."""
    await websocket.accept()
    audio_clients.append(websocket)
    print("Next.js Audio Client connected")
    
    try:
        while True:
            await websocket.receive_text() # keep alive
    except WebSocketDisconnect:
        if websocket in audio_clients:
            audio_clients.remove(websocket)
        print("Next.js Audio Client disconnected")

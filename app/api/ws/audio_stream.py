from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.media_service import media_service

router = APIRouter()

@router.websocket("")
async def audio_feed_endpoint(websocket: WebSocket):
    """
    Binary PCM Audio Feed for Next.js.
    Playback via Web Audio API.
    """
    await websocket.accept()
    media_service.audio_clients.append(websocket)
    print("Next.js Binary Audio Client connected")
    
    try:
        while True:
            await websocket.receive_text() # keep alive
    except WebSocketDisconnect:
        if websocket in media_service.audio_clients:
            media_service.audio_clients.remove(websocket)
        print("Next.js Binary Audio Client disconnected")

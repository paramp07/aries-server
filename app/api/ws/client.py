# /ws/client — JSON stream to Next.js
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.core.broadcast import clients

router = APIRouter()

@router.websocket("")
async def nextjs_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    print("Next.js client connected")

    try:
        while True:
            await websocket.receive_text()  # keep connection alive
    except WebSocketDisconnect:
        clients.remove(websocket)
        print("Next.js client disconnected")

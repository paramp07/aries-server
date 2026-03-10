# /ws — CBOR ingestion from ESP32
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.core.decoder import decode_sensor_cbor
from app.core.broadcast import broadcast

router = APIRouter()

@router.websocket("")
async def esp32_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("ESP32 connected")

    try:
        while True:
            data = await websocket.receive_bytes()
            print(f"Raw bytes: {data.hex()}")
            try:
                decoded = decode_sensor_cbor(data, decimals=4)
                print("Decoded:", decoded)
                await broadcast(decoded)
            except Exception as e:
                print(f"Decode error: {e}")

    except WebSocketDisconnect:
        print("ESP32 disconnected")

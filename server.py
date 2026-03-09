from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from decoder import decode_sensor_cbor
from typing import List

app = FastAPI()

# store all connected Next.js clients
clients: List[WebSocket] = []

# broadcast data to all connected Next.js clients
async def broadcast(data: dict):
    disconnected = []
    for client in clients:
        try:
            await client.send_json(data)
        except Exception:
            disconnected.append(client)
    for client in disconnected:
        clients.remove(client)


# main WebSocket endpoint for ESP32 to receive CBOR data
@app.websocket("/ws")
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


# WebSocket endpoint for Next.js clients to receive decoded data
@app.websocket("/ws/client")
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
# /ws — CBOR ingestion from ESP32
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.core.decoder import decode_sensor_cbor
from app.core.broadcast import broadcast
from app.ml.vision.thermal import processor as thermal_processor

router = APIRouter()

# websocket endpoint for the esp32 to send CBOR data
@router.websocket("")
async def esp32_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("ESP32 connected")

    try:
        while True:
            data = await websocket.receive_bytes()
            try:
                # 1. Decode CBOR
                decoded = decode_sensor_cbor(data, decimals=4)
                
                # 2. Process Thermal Frame if present
                if 'mlx90640' in decoded:
                    metrics = thermal_processor.process_frame(decoded['mlx90640'])
                    decoded['thermalAnalytics'] = metrics
                    
                    # Diagnostic print to check range
                    grid = np.array(decoded['mlx90640'])
                    valid = grid[grid > 0]
                    if valid.size > 0:
                        print(f"THERMAL STATS: Min={valid.min():.1f} Max={valid.max():.1f} Avg={valid.mean():.1f} (Size: {grid.shape})")
                
                # 3. Broadcast to all clients
                await broadcast(decoded)
                
            except Exception as e:
                print(f"Decode error: {e}")

    except WebSocketDisconnect:
        print("ESP32 disconnected")

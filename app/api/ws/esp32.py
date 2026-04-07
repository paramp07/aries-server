# /ws — CBOR ingestion from ESP32
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.core.decoder import decode_sensor_cbor
from app.core.broadcast import broadcast
from app.ml.vision.thermal import processor as thermal_processor
from app.services.media_service import media_service

router = APIRouter()

# websocket endpoint for the esp32 to send CBOR data
@router.websocket("")
async def esp32_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("ESP32 connected")

    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                continue

            # Header is 1 byte
            packet_type = data[0]
            payload = data[1:]

            try:
                if packet_type == 0x01: # CBOR (Sensors/Thermal)
                    # 1. Decode CBOR
                    decoded = decode_sensor_cbor(payload, decimals=4)
                    
                    # 2. Process Thermal Frame if present
                    if 'mlx90640' in decoded:
                        metrics = thermal_processor.process_frame(decoded['mlx90640'])
                        decoded['thermalAnalytics'] = metrics
                    
                    # 3. Broadcast to all clients
                    await broadcast(decoded)
                
                elif packet_type == 0x02: # MJPEG (Camera)
                    media_service.update_video_frame(payload)
                
                elif packet_type == 0x03: # PCM (Audio)
                    await media_service.add_audio_chunk(payload)
                    
            except Exception as e:
                print(f"Media routing error: {e}")

    except WebSocketDisconnect:
        print("ESP32 disconnected")

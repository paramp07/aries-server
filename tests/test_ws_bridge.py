import asyncio
import websockets
import cbor2
import numpy as np
import json

async def test_bridge():
    # Testing against localhost since the server is running in Docker mapped to 8000
    uri_client = "ws://localhost:8000/ws/client"
    uri_esp32 = "ws://localhost:8000/ws"

    print(f"--- WebSocket Bridge Test ---")
    
    try:
        print("Connecting to Next.js client endpoint...")
        async with websockets.connect(uri_client) as client_ws:
            print("Connecting to ESP32 endpoint...")
            async with websockets.connect(uri_esp32) as esp32_ws:
                
                # Create mock thermal packet (768 float32 values)
                mock_pixels = np.random.uniform(20.0, 35.0, 768).astype('<f4')
                packet = {
                    'mlx90640': mock_pixels.tobytes(),
                    'bmp280': {'temperature': 25.0, 'pressure': 1013.25},
                    'hs3003': {'temperature': 25.0, 'humidity': 50.0},
                    'mq2': {'rawValue': 200}
                }
                cbor_msg = cbor2.dumps(packet)

                print("Sending mock CBOR packet from 'ESP32'...")
                await esp32_ws.send(cbor_msg)

                print("Waiting for broadcast to 'Next.js'...")
                try:
                    # Wait for the broadcasted JSON message
                    # We might get some initial messages if the server is already streaming, 
                    # so we'll look for one that has our mock data or just any valid broadcast.
                    response = await asyncio.wait_for(client_ws.recv(), timeout=2.0)
                    data = json.loads(response)
                    
                    print("\n[SUCCESS] Server received and broadcasted data!")
                    print(f"Keys present: {list(data.keys())}")
                    
                    if 'thermalAnalytics' in data:
                        print(f"Analytics -> Avg Temp: {data['thermalAnalytics']['avg_temp']}°C, FPS: {data['thermalAnalytics']['fps']}")
                    
                    if 'mlx90640' in data:
                        grid = data['mlx90640']
                        print(f"Thermal Grid -> {len(grid)} rows x {len(grid[0])} columns")
                        
                except asyncio.TimeoutError:
                    print("\n[FAILED] Timeout: Server did not broadcast data to the client.")
                except Exception as e:
                    print(f"\n[ERROR] during reception: {e}")
                    
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Could not connect to server: {e}")
        print("Make sure the server is running (docker compose up).")

if __name__ == "__main__":
    try:
        asyncio.run(test_bridge())
    except KeyboardInterrupt:
        pass

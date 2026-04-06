import cbor2
import numpy as np
import time
from app.core.decoder import decode_sensor_cbor
from app.ml.vision.thermal import processor as thermal_processor

def test_integration():
    # 1. Create mock thermal data (768 float32s)
    # 32x24 = 768
    mock_pixels = np.random.uniform(20.0, 35.0, 768).astype('<f4')
    thermal_bytes = mock_pixels.tobytes()
    
    # 2. Create mock CBOR packet
    packet = {
        'mlx90640': thermal_bytes,
        'bmp280': {'temperature': 22.5, 'pressure': 101325.0},
        'hs3003': {'temperature': 23.0, 'humidity': 45.0},
        'mq2': {'rawValue': 450}
    }
    cbor_data = cbor2.dumps(packet)
    
    print("--- Testing Decoder ---")
    decoded = decode_sensor_cbor(cbor_data)
    
    assert 'mlx90640' in decoded
    assert len(decoded['mlx90640']) == 24
    assert len(decoded['mlx90640'][0]) == 32
    assert decoded['bmp280']['temperature'] == 22.5
    print("Decoder OK!")
    
    print("\n--- Testing Thermal Processor ---")
    # Simulate a few frames for FPS
    for i in range(5):
        metrics = thermal_processor.process_frame(decoded['mlx90640'])
        print(f"Frame {i+1} Metrics: {metrics}")
        time.sleep(0.05) # 20Hz simulate
        
    assert 'avg_temp' in metrics
    assert 'fps' in metrics
    print("Thermal Processor OK!")
    
    print("\nVerification Successful!")

if __name__ == "__main__":
    test_integration()

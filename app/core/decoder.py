import cbor2
import numpy as np
import math

def clean_value(val):
    """Recursively clean NaNs/Infs from data to ensure JSON compatibility."""
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return val
    if isinstance(val, dict):
        return {k: clean_value(v) for k, v in val.items()}
    if isinstance(val, list):
        # Optimization: Skip large lists (like the thermal grid) 
        # which have already been cleaned via numpy.nan_to_num
        if len(val) > 100:
            return val
        return [clean_value(v) for v in val]
    return val

def repair_dead_pixels(data_2d):
    """
    Scans for NaNs or extreme values and replaces them with 
    the average of their 8 neighbors.
    """
    fixed_data = data_2d.copy()
    rows, cols = data_2d.shape  # Usually (24, 32)
    
    for y in range(rows):
        for x in range(cols):
            val = fixed_data[y, x]
            
            # 1. Define what counts as a "dead" pixel (NaNs, 0.0, or impossible)
            if not np.isfinite(val) or val <= 0.0 or val > 300.0:
                neighbors = []
                
                # 2. Look at 8 surrounding pixels
                for ny in range(max(0, y-1), min(rows, y+2)):
                    for nx in range(max(0, x-1), min(cols, x+2)):
                        if nx == x and ny == y: continue 
                        
                        n_val = data_2d[ny, nx]
                        if np.isfinite(n_val) and n_val > 0:
                            neighbors.append(n_val)
                
                # 3. Replace with average (or a baseline like 25.0 if no neighbors)
                fixed_data[y, x] = np.mean(neighbors) if neighbors else 25.0
                
    return fixed_data

def decode_sensor_cbor(data: bytes, round_values: bool = True, decimals: int = 2) -> dict:
    decoded = cbor2.loads(data)
    
    def r(val):
        return round(val, decimals) if round_values else val

    payload = {}

    # 1. Thermal Pixels (32x24)
    if 'mlx90640' in decoded:
        raw_bytes = decoded['mlx90640']
        # Enforce exact size for 24x32 float32 grid (3072 bytes)
        if len(raw_bytes) >= 3072:
            raw_bytes = raw_bytes[:3072]
            pixels = np.frombuffer(raw_bytes, dtype='<f4').reshape(24, 32)
            
            # --- APPLY REPAIR ---
            repaired = repair_dead_pixels(pixels)
            payload['mlx90640'] = repaired.tolist()
        else:
            print(f"WARNING: Thermal buffer too small: {len(raw_bytes)} bytes")

    # 2. BMP280
    if 'bmp280' in decoded:
        payload['bmp280'] = {
            "temperature": r(decoded['bmp280'].get('temperature', 0)),
            "pressure": r(decoded['bmp280'].get('pressure', 0)),
        }

    # 3. HS3003
    if 'hs3003' in decoded:
        payload['hs3003'] = {
            "temperature": r(decoded['hs3003'].get('temperature', 0)),
            "humidity": r(decoded['hs3003'].get('humidity', 0)),
        }

    # 4. MQ2
    if 'mq2' in decoded:
        payload['mq2'] = {
            "rawValue": decoded['mq2'].get('rawValue', 0)
        }

    # Final robust cleaning of all NaNs/Infs for JSON safety
    return clean_value(payload)
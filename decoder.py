import cbor2

def decode_sensor_cbor(data: bytes, round_values: bool = True, decimals: int = 2) -> dict:
    decoded = cbor2.loads(data)
    
    def r(val):
        return round(val, decimals) if round_values else val

    return {
        "bmp280": {
            "temperature": r(decoded["bmp280"]["temperature"]),
            "pressure": r(decoded["bmp280"]["pressure"]),
        },
        "hs3003": {
            "temperature": r(decoded["hs3003"]["temperature"]),
            "humidity": r(decoded["hs3003"]["humidity"]),
        }
    }
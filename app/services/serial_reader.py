import serial
import numpy as np
import threading
import time

class SerialAudioReader:
    def __init__(self, port, baudrate=921600, sample_rate=16000, channels=1):
        self.port = port
        self.baudrate = baudrate
        self.sample_rate = sample_rate
        self.channels = channels
        self.ser = None
        self.running = False
        self.buffer = bytearray()
        self.lock = threading.Lock()

    def start(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self.running = True
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            print(f"Serial reader started on {self.port} at {self.baudrate} baud.")
        except Exception as e:
            print(f"Failed to open serial port {self.port}: {e}")
            raise

    def stop(self):
        self.running = False
        if self.ser:
            self.ser.close()

    def _read_loop(self):
        while self.running:
            if self.ser.in_waiting > 0:
                data = self.ser.read(self.ser.in_waiting)
                with self.lock:
                    self.buffer.extend(data)
            else:
                time.sleep(0.01)

    def get_audio_chunk(self, duration_s=0.25):
        """
        Returns a float32 numpy array of audio samples for the requested duration.
        """
        num_samples = int(self.sample_rate * duration_s)
        num_bytes = num_samples * 2 * self.channels # 16-bit = 2 bytes

        with self.lock:
            if len(self.buffer) < num_bytes:
                return None
            
            raw_data = self.buffer[:num_bytes]
            self.buffer = self.buffer[num_bytes:]

        # Convert to int16 numpy array
        audio_int16 = np.frombuffer(raw_data, dtype=np.int16)
        
        # Convert to mono if needed (averaging channels)
        if self.channels > 1:
            audio_int16 = audio_int16.reshape(-1, self.channels).mean(axis=1).astype(np.int16)
        
        # Convert to float32 normalized
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        return audio_float32

if __name__ == "__main__":
    # Test block
    import sys
    port = sys.argv[1] if len(sys.argv) > 1 else "COM4"
    reader = SerialAudioReader(port=port)
    try:
        reader.start()
        while True:
            chunk = reader.get_audio_chunk()
            if chunk is not None:
                print(f"Read chunk of size {len(chunk)}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        reader.stop()

import time
import numpy as np

class ThermalProcessor:
    def __init__(self):
        self.last_frame_time = None
        self.fps = 0.0
        
    def process_frame(self, pixels_list):
        """
        Processes a thermal frame (list of lists) and returns metrics.
        """
        pixels = np.array(pixels_list)
        
        # Calculate Average Temp
        avg_temp = float(np.mean(pixels))
        
        # Calculate FPS
        current_time = time.time()
        if self.last_frame_time is not None:
            dt = current_time - self.last_frame_time
            if dt > 0:
                # Simple smoothing
                new_fps = 1.0 / dt
                self.fps = self.fps * 0.9 + new_fps * 0.1
        
        self.last_frame_time = current_time
        
        return {
            "avg_temp": round(avg_temp, 2),
            "fps": round(self.fps, 1),
            "min_temp": round(float(np.min(pixels)), 2),
            "max_temp": round(float(np.max(pixels)), 2)
        }

# Singleton processor for common use cases
processor = ThermalProcessor()

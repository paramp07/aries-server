#!/usr/bin/env python3
import asyncio
import websockets
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue
import time
import sys

# Configuration
WS_URL = "ws://localhost:8000/ws/client"

class WSDataReceiver:
    def __init__(self, url=WS_URL):
        self.url = url
        self.running = True
        self.data_queue = queue.Queue(maxsize=10)
        self.latest_packet = None

    async def _connect_and_listen(self):
        while self.running:
            try:
                print(f"Connecting to {self.url}...")
                async with websockets.connect(self.url) as websocket:
                    print("✓ Connected to Aries Server")
                    while self.running:
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        if data.get('type') == 'heartbeat':
                            continue
                            
                        if not self.data_queue.full():
                            self.data_queue.put_nowait(data)
                        self.latest_packet = data
            except Exception as e:
                print(f"Connection error: {e}. Retrying in 2s...")
                await asyncio.sleep(2)

    def start(self):
        def run_async():
            asyncio.run(self._connect_and_listen())
        
        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()

class ThermalVisualizer:
    def __init__(self):
        self.receiver = WSDataReceiver()
        self.receiver.start()
        
        self.running = True
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0
        
        # UI Setup
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.patch.set_facecolor('#0a0a0a')
        
        # Initial black frame
        self.im = self.ax.imshow(np.zeros((24, 32)), cmap='magma', interpolation='gaussian')
        self.ax.set_axis_off()
        self.cbar = plt.colorbar(self.im, ax=self.ax)
        self.cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
        
        self.title_text = self.ax.text(0.5, 1.05, "Aries Thermal Stream", 
                                     transform=self.ax.transAxes, color='white', 
                                     ha='center', fontsize=12, fontweight='bold')
        
        self.stats_text = self.ax.text(0.02, 0.02, "", 
                                     transform=self.ax.transAxes, color='cyan', 
                                     fontfamily='monospace', fontsize=9)

        self.ani = FuncAnimation(self.fig, self._update, interval=30, blit=False)

    def _update(self, frame_num):
        try:
            data = self.receiver.data_queue.get_nowait()
        except queue.Empty:
            return [self.im]

        if 'mlx90640' in data:
            grid = np.array(data['mlx90640'])
            
            # Use 1% and 99% percentiles for scaling to handle noise/distortions better
            valid_pixels = grid[grid > 0]
            if valid_pixels.size > 0:
                vmin = np.percentile(valid_pixels, 2)
                vmax = np.percentile(valid_pixels, 98)
                self.im.set_clim(vmin=vmin, vmax=vmax)
            
            self.im.set_data(grid)
            
            # FPS Calculation
            now = time.time()
            dt = now - self.last_time
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            self.last_time = now
            
            # Update Stats
            analytics = data.get('thermalAnalytics', {})
            bmp = data.get('bmp280', {})
            stats = f"FPS: {self.fps:.1f} | Avg: {analytics.get('avg_temp', 0):.1f}°C | BMP: {bmp.get('temperature', 0):.1f}°C"
            self.stats_text.set_text(stats)

        return [self.im, self.stats_text]

    def show(self):
        plt.show()

if __name__ == '__main__':
    print("Starting Aries WebSocket Visualizer...")
    print("Make sure your server (run.py) is running on localhost:8000")
    ThermalVisualizer().show()
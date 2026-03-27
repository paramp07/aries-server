import uvicorn
import multiprocessing
import os
import asyncio
from app.services.inference_streamer import InferenceStreamer

def start_audio_streamer():
    """Starts the Audio AI inference streamer in a separate process."""
    # Ensure PYTHONPATH is set so imports work
    os.environ["PYTHONPATH"] = os.getcwd()
    port = os.getenv("SERIAL_PORT", "COM3")
    streamer = InferenceStreamer(port=port)
    asyncio.run(streamer.run())

if __name__ == "__main__":
    print("Starting Aries AI Audio Streamer...")
    audio_process = multiprocessing.Process(target=start_audio_streamer, daemon=True)
    audio_process.start()

    print("Starting Aries Server...")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # Disabled to avoid multiprocessing recursion on Windows
    )

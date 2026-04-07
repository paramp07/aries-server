import uvicorn
import os

if __name__ == "__main__":
    # Ensure PYTHONPATH is set so app modules are found correctly
    os.environ["PYTHONPATH"] = os.getcwd()
    
    print("Starting Aries AI Multimedia Server...")
    print("Video Feed:   http://0.0.0.0:8000/video_feed")
    print("Audio AI:     ws://0.0.0.0:8000/ws/audio")
    print("Audio Stream: ws://0.0.0.0:8000/ws/audio_feed")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )

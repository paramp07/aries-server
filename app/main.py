# entrypoint, mounts all routers
from fastapi import FastAPI
from app.api.ws import esp32, client, audio

app = FastAPI()

# Mount routers
app.include_router(esp32.router, prefix="/ws", tags=["websocket-esp32"])
app.include_router(client.router, prefix="/ws/client", tags=["websocket-client"])
app.include_router(audio.router, prefix="/ws/audio", tags=["audio-ai"]) # Bridge is at /ws/audio/data, WS is at /ws/audio

@app.get("/")
def read_root():
    return {"message": "Aries Server is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    # run_server.py logic bundled in main here for convenience
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)

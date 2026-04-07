from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.services.media_service import media_service

router = APIRouter()

@router.get("")
async def video_feed():
    """
    MJPEG Video Stream Endpoint.
    Usage in Next.js: <img src="http://server-ip:8000/video_feed" />
    """
    return StreamingResponse(
        media_service.get_video_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

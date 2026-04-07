# broadcast logic with diagnostics
import asyncio
import time
from typing import List
from fastapi import WebSocket

# store all connected Next.js clients
clients: List[WebSocket] = []

# broadcast data to all connected Next.js clients
async def broadcast(data: dict):
    if not clients:
        return

    disconnected = []
    for client in clients:
        try:
            await client.send_json(data)
        except Exception:
            disconnected.append(client)
            
    for client in disconnected:
        clients.remove(client)

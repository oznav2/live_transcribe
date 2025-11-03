"""WebSocket helper utilities for safe message sending."""
import json
import logging
from typing import Optional, Dict, Any
from fastapi import WebSocket
from fastapi.websockets import WebSocketState

logger = logging.getLogger(__name__)


async def safe_ws_send(websocket: WebSocket, data: dict) -> bool:
    """
    Safely send JSON data through WebSocket with connection state check.
    
    Args:
        websocket: WebSocket connection
        data: Dictionary to send as JSON
        
    Returns:
        bool: True if sent successfully, False if connection closed
    """
    try:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json(data)
            return True
        else:
            logger.debug(f"WebSocket not connected (state: {websocket.client_state}), skipping message")
            return False
    except Exception as e:
        logger.debug(f"Failed to send WebSocket message: {e}")
        return False
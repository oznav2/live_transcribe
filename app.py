"""Main application entry point - Modular refactored version."""
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config.settings import PORT
from core.lifespan import lifespan
from api.routes import router as api_router
from api.websocket import websocket_transcribe, websocket_translate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Live Transcription Service",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount API routes
app.include_router(api_router)

# Register WebSocket endpoints
app.add_websocket_route("/ws/transcribe", websocket_transcribe)
app.add_websocket_route("/ws/translate", websocket_translate)

if __name__ == "__main__":
    # Create static directory if it doesn't exist
    Path("static").mkdir(exist_ok=True)
    
    # Run the server
    port = PORT
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
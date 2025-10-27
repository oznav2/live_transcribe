"""
Live Audio Streaming Transcription Application
Inspired by Vibe - Uses FFmpeg + Whisper for real-time transcription
"""
import asyncio
import logging
import os
import subprocess
import tempfile
import threading
import queue
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import whisper
try:
    from whispercpp import Whisper as WhisperCpp
    WHISPER_CPP_AVAILABLE = True
except ImportError:
    WHISPER_CPP_AVAILABLE = False
    WhisperCpp = None
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    try:
        logger.info(f"Loading Whisper model: {MODEL_SIZE}")
        load_model(MODEL_SIZE)
        logger.info("Default Whisper model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load default Whisper model: {e}")
        raise
    
    yield
    
    # Shutdown (if needed)
    logger.info("Application shutting down")

# Initialize FastAPI app with lifespan
app = FastAPI(title="Live Transcription Service", version="1.0.0", lifespan=lifespan)

# Global Whisper models
whisper_models = {}
current_model = None
current_model_name = None
MODEL_SIZE = os.getenv("WHISPER_MODEL", "ivrit-large-v3-turbo")  # Default to Ivrit Hebrew model

# Model configurations
MODEL_CONFIGS = {
    "tiny": {"type": "openai", "name": "tiny"},
    "base": {"type": "openai", "name": "base"},
    "small": {"type": "openai", "name": "small"},
    "medium": {"type": "openai", "name": "medium"},
    "large": {"type": "openai", "name": "large"},
    "ivrit-large-v3-turbo": {"type": "ggml", "path": os.getenv("IVRIT_MODEL_PATH", "models/ivrit-whisper-large-v3-turbo.bin")}
}

# Audio processing configuration
CHUNK_DURATION = 5  # seconds - process audio in 5-second chunks
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
CHANNELS = 1  # Mono audio


def load_model(model_name: str):
    """Load a model based on its configuration"""
    global current_model, current_model_name
    
    if model_name == current_model_name and current_model is not None:
        return current_model
    
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    config = MODEL_CONFIGS[model_name]
    
    if config["type"] == "openai":
        logger.info(f"Loading OpenAI Whisper model: {config['name']}")
        model = whisper.load_model(config["name"])
    elif config["type"] == "ggml":
        if not WHISPER_CPP_AVAILABLE:
            raise ValueError("whisper-cpp-python is not available for GGML models")
        logger.info(f"Loading GGML model from: {config['path']}")
        model = WhisperCpp.from_pretrained(config["path"])
    else:
        raise ValueError(f"Unknown model type: {config['type']}")
    
    current_model = model
    current_model_name = model_name
    return model


class TranscriptionRequest(BaseModel):
    url: str
    language: Optional[str] = None


class AudioStreamProcessor:
    """Processes audio streams from URLs using FFmpeg and transcribes with Whisper"""
    
    def __init__(self, url: str, language: Optional[str] = None, model_name: str = "large"):
        self.url = url
        self.language = language
        self.model_name = model_name
        self.model = None
        self.is_running = False
        self.audio_queue = queue.Queue(maxsize=10)
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        
    def start_ffmpeg_stream(self):
        """Start FFmpeg process to stream audio from URL"""
        try:
            # FFmpeg command to extract audio and convert to PCM WAV format
            # This works with m3u8, direct video URLs, and audio URLs
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', self.url,           # Input URL
                '-f', 's16le',             # PCM signed 16-bit little-endian
                '-acodec', 'pcm_s16le',    # Audio codec
                '-ar', str(SAMPLE_RATE),   # Sample rate 16kHz
                '-ac', str(CHANNELS),      # Mono audio
                '-loglevel', 'error',      # Only show errors
                'pipe:1'                   # Output to stdout
            ]
            
            logger.info(f"Starting FFmpeg stream for URL: {self.url}")
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")
            return False
    
    def read_audio_chunks(self):
        """Read audio data from FFmpeg in chunks"""
        if not self.ffmpeg_process:
            return
        
        # Calculate chunk size: CHUNK_DURATION seconds of audio
        chunk_size = int(SAMPLE_RATE * CHANNELS * 2 * CHUNK_DURATION)  # 2 bytes per sample
        
        try:
            while self.is_running:
                audio_data = self.ffmpeg_process.stdout.read(chunk_size)
                
                if not audio_data:
                    logger.info("FFmpeg stream ended")
                    break
                
                # Put audio chunk in queue for processing
                try:
                    self.audio_queue.put(audio_data, timeout=1)
                except queue.Full:
                    logger.warning("Audio queue full, skipping chunk")
                    
        except Exception as e:
            logger.error(f"Error reading audio chunks: {e}")
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop the audio streaming and cleanup"""
        self.is_running = False
        
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ffmpeg_process.kill()
            except Exception as e:
                logger.error(f"Error stopping FFmpeg: {e}")


async def transcribe_audio_stream(websocket: WebSocket, processor: AudioStreamProcessor):
    """Transcribe audio chunks and send results via WebSocket"""
    
    # Load the model for this processor
    try:
        model = load_model(processor.model_name)
        processor.model = model
        model_config = MODEL_CONFIGS[processor.model_name]
    except Exception as e:
        await websocket.send_json({"error": f"Failed to load model {processor.model_name}: {str(e)}"})
        return
    
    try:
        while processor.is_running or not processor.audio_queue.empty():
            try:
                # Get audio chunk from queue (with timeout)
                audio_data = processor.audio_queue.get(timeout=1)
                
                # Save chunk to temporary WAV file for Whisper
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                    temp_path = temp_audio.name
                    
                    # Write WAV header + audio data
                    import wave
                    with wave.open(temp_path, 'wb') as wav_file:
                        wav_file.setnchannels(CHANNELS)
                        wav_file.setsampwidth(2)  # 16-bit
                        wav_file.setframerate(SAMPLE_RATE)
                        wav_file.writeframes(audio_data)
                
                try:
                    # Transcribe audio chunk
                    logger.info(f"Transcribing audio chunk ({len(audio_data)} bytes)")
                    
                    if model_config["type"] == "openai":
                        # Use OpenAI Whisper
                        result = model.transcribe(
                            temp_path,
                            language=processor.language,
                            fp16=False,  # CPU compatibility
                            verbose=False
                        )
                        transcription_text = result.get('text', '').strip()
                        detected_language = result.get('language', 'unknown')
                    elif model_config["type"] == "ggml":
                        # Use whisper.cpp
                        result = model.transcribe(temp_path)
                        transcription_text = result.strip() if isinstance(result, str) else result.get('text', '').strip()
                        detected_language = processor.language or 'he'  # Default to Hebrew for Ivrit model
                    
                    # Send transcription to client
                    if transcription_text:
                        await websocket.send_json({
                            "type": "transcription",
                            "text": transcription_text,
                            "language": detected_language
                        })
                        logger.info(f"Sent transcription: {transcription_text[:100]}...")
                    
                finally:
                    # Cleanup temp file
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                        
            except queue.Empty:
                # No audio data available, continue waiting
                await asyncio.sleep(0.1)
                continue
            except Exception as e:
                logger.error(f"Error transcribing chunk: {e}")
                await websocket.send_json({"error": str(e)})
                
        # Stream finished
        await websocket.send_json({"type": "complete", "message": "Transcription complete"})
        logger.info("Transcription stream completed")
        
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        await websocket.send_json({"error": str(e)})


@app.get("/", response_class=HTMLResponse)
async def get_home():
    """Serve the main web interface"""
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """WebSocket endpoint for live transcription"""
    await websocket.accept()
    processor = None
    audio_thread = None
    
    try:
        # Receive transcription request
        data = await websocket.receive_json()
        url = data.get("url")
        language = data.get("language")
        model_name = data.get("model", "large")  # Default to large model
        
        if not url:
            await websocket.send_json({"error": "URL is required"})
            return
        
        logger.info(f"Starting transcription for URL: {url} with model: {model_name}")
        await websocket.send_json({"type": "status", "message": "Starting audio stream..."})
        
        # Create audio processor
        processor = AudioStreamProcessor(url, language, model_name)
        
        # Start FFmpeg stream
        if not processor.start_ffmpeg_stream():
            await websocket.send_json({"error": "Failed to start audio stream"})
            return
        
        processor.is_running = True
        
        # Start audio reading thread
        audio_thread = threading.Thread(target=processor.read_audio_chunks, daemon=True)
        audio_thread.start()
        
        await websocket.send_json({"type": "status", "message": "Stream started, transcribing..."})
        
        # Start transcription
        await transcribe_audio_stream(websocket, processor)
        
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        # Cleanup
        if processor:
            processor.stop()
        if audio_thread and audio_thread.is_alive():
            audio_thread.join(timeout=5)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "whisper_model": current_model_name or MODEL_SIZE,
        "model_loaded": current_model is not None
    }


if __name__ == "__main__":
    # Create static directory if it doesn't exist
    Path("static").mkdir(exist_ok=True)
    
    # Run the server
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

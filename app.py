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
import hashlib
import shutil
import re
from pathlib import Path
from typing import Optional, List, Tuple
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

import whisper
import aiohttp
import torch

# Check if whisper.cpp CLI is available
WHISPER_CPP_PATH = os.getenv("WHISPER_CPP_PATH", "/app/whisper.cpp/build/bin/whisper-cli")
WHISPER_CPP_AVAILABLE = os.path.exists(WHISPER_CPP_PATH)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from starlette.websockets import WebSocketState
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl
import uvicorn

# Deepgram configuration
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
try:
    from deepgram import DeepgramClient
    # Map events for Deepgram SDK v3/v4
    DG_EVENT_OPEN = DG_EVENT_CLOSE = DG_EVENT_ERROR = DG_EVENT_MESSAGE = None
    try:
        # v3 style events
        from deepgram.core.events import EventType as _DGEventEnum
        DG_EVENT_OPEN = getattr(_DGEventEnum, "OPEN", None)
        DG_EVENT_MESSAGE = getattr(_DGEventEnum, "MESSAGE", None)
        DG_EVENT_CLOSE = getattr(_DGEventEnum, "CLOSE", None)
        DG_EVENT_ERROR = getattr(_DGEventEnum, "ERROR", None)
    except Exception:
        # v4 style events
        try:
            from deepgram.clients.common.v1.websocket_events import WebSocketEvents
            # TitleCase names in v4
            DG_EVENT_OPEN = getattr(WebSocketEvents, "Open", None)
            DG_EVENT_CLOSE = getattr(WebSocketEvents, "Close", None)
            DG_EVENT_ERROR = getattr(WebSocketEvents, "Error", None)
        except Exception:
            pass
        try:
            from deepgram.clients.listen.enums import LiveTranscriptionEvents as _DGLiveEvents
            # Prefer 'Transcript'; fall back to other possible names
            DG_EVENT_MESSAGE = (
                getattr(_DGLiveEvents, "Transcript", None)
                or getattr(_DGLiveEvents, "TranscriptReceived", None)
                or getattr(_DGLiveEvents, "Results", None)
            )
        except Exception:
            pass
    DEEPGRAM_AVAILABLE = all(e is not None for e in [DG_EVENT_OPEN, DG_EVENT_MESSAGE, DG_EVENT_CLOSE, DG_EVENT_ERROR])
except Exception:
    DEEPGRAM_AVAILABLE = False
    # logger is not yet initialized at this point; use print to avoid NameError
    print("Deepgram SDK not available or incompatible version.")

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

        # Initialize audio cache
        init_cache_dir()
        logger.info(f"Audio cache initialized (enabled: {CACHE_ENABLED})")

        # Initialize capture directory
        init_capture_dir()
        logger.info("Capture directory initialized for first-60s mode")
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
    "ivrit-large-v3-turbo": {"type": "ggml", "path": os.getenv("IVRIT_MODEL_PATH", "models/ivrit-whisper-large-v3-turbo.bin")},
    "deepgram": {"type": "deepgram", "model": "nova-2", "language": "en"}
}

# Audio processing configuration
CHUNK_DURATION = 5   # seconds - very short for fast real-time processing
CHUNK_OVERLAP = 1    # seconds - minimal overlap
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
CHANNELS = 1  # Mono audio
AUDIO_QUEUE_SIZE = 50  # Large queue to handle bursts

# Feature flags for optional parallel chunking on yt-dlp downloads
USE_PARALLEL_TRANSCRIPTION = os.getenv("USE_PARALLEL_TRANSCRIPTION", "false").lower() == "true"
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "2"))
YTDLP_CHUNK_SECONDS = int(os.getenv("YTDLP_CHUNK_SECONDS", str(CHUNK_DURATION)))
YTDLP_CHUNK_OVERLAP = int(os.getenv("YTDLP_CHUNK_OVERLAP", str(CHUNK_OVERLAP)))

# Audio caching configuration
CACHE_DIR = Path("cache/audio")
CACHE_MAX_AGE_HOURS = 24  # Clean cache older than 24 hours
CACHE_ENABLED = os.getenv("AUDIO_CACHE_ENABLED", "true").lower() == "true"

# Capture configuration for first-60s feature
CAPTURE_DIR = Path("cache/captures")
CAPTURES = {}

def init_capture_dir():
    try:
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"Failed to create capture dir: {e}")


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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")
        try:
            model = whisper.load_model(config["name"], device=device)
        except Exception as e:
            logger.warning(f"Failed to load model on {device}: {e}. Falling back to CPU.")
            model = whisper.load_model(config["name"], device="cpu")
    elif config["type"] == "ggml":
        if not WHISPER_CPP_AVAILABLE:
            raise ValueError(f"whisper.cpp CLI not found at: {WHISPER_CPP_PATH}")
        logger.info(f"Loading GGML model from: {config['path']}")
        # For GGML models, we store the path and use whisper.cpp CLI
        model = {"type": "ggml_cli", "path": config["path"], "whisper_cpp_path": WHISPER_CPP_PATH}
    else:
        raise ValueError(f"Unknown model type: {config['type']}")

    current_model = model
    current_model_name = model_name
    return model


def should_use_ytdlp(url: str) -> bool:
    """Determine if URL should use yt-dlp instead of direct FFmpeg streaming"""
    # Use yt-dlp for known video platforms and complex URLs
    ytdlp_patterns = [
        'youtube.com', 'youtu.be', 'vimeo.com', 'dailymotion.com',
        'facebook.com', 'twitter.com', 'twitch.tv', 'tiktok.com'
    ]
    return any(pattern in url.lower() for pattern in ytdlp_patterns)


def download_audio_with_ytdlp(url: str, language: Optional[str] = None) -> Optional[str]:
    """
    Download and normalize audio from URL using yt-dlp
    Returns path to normalized WAV file or None on failure
    """
    try:
        # Create temporary file for download
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            output_path = temp_file.name

        # yt-dlp command with audio extraction and normalization
        cmd = [
            'yt-dlp',
            '--extract-audio',  # Extract audio only
            '--audio-format', 'wav',  # Output as WAV
            '--audio-quality', '0',  # Best quality
            '--postprocessor-args', f'ffmpeg:-ar 16000 -ac 1 -c:a pcm_s16le',  # Normalize to Whisper format
            '--no-playlist',  # Don't download playlists
            '--quiet',  # Suppress output
            '--no-warnings',  # Suppress warnings
            '-o', output_path,
            url
        ]

        logger.info(f"Downloading audio from URL with yt-dlp: {url}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0 and os.path.exists(output_path):
            logger.info(f"Successfully downloaded and normalized audio: {output_path}")
            return output_path
        else:
            logger.error(f"yt-dlp failed: {result.stderr}")
            if os.path.exists(output_path):
                os.unlink(output_path)
            return None

    except subprocess.TimeoutExpired:
        logger.error("yt-dlp download timeout after 5 minutes")
        return None


def get_audio_duration_seconds(audio_path: str) -> Optional[float]:
    """
    Use ffprobe to get total audio duration in seconds. Returns None if failed.
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
        else:
            logger.error(f"ffprobe failed: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"ffprobe error: {e}")
        return None


def split_audio_into_chunks(audio_path: str, chunk_seconds: int, overlap_seconds: int) -> Tuple[str, List[Tuple[int, str]]]:
    """
    Split the given audio file into chunked WAV files suitable for Whisper.
    Returns a tuple of (temp_dir, [(index, chunk_path), ...]). Caller must clean temp_dir.
    Uses -ss and -t for sliding-window segmentation to achieve overlap.
    """
    duration = get_audio_duration_seconds(audio_path)
    if not duration:
        raise RuntimeError("Unable to determine audio duration for chunking")

    temp_dir = tempfile.mkdtemp(prefix='chunks_')
    chunks: List[Tuple[int, str]] = []

    # Compute starts with overlap (next chunk starts chunk_seconds - overlap_seconds ahead)
    step = max(1, chunk_seconds - overlap_seconds)
    index = 0
    start = 0
    while start < duration:
        out_path = os.path.join(temp_dir, f"chunk_{index:04d}.wav")
        # Normalize to 16kHz mono PCM s16le for Whisper
        cmd = [
            'ffmpeg',
            '-ss', str(start),
            '-i', audio_path,
            '-t', str(chunk_seconds),
            '-ar', '16000',
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            '-loglevel', 'error',
            '-y',
            out_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(out_path):
            chunks.append((index, out_path))
            index += 1
            start += step
        else:
            # Stop if we fail to segment further
            logger.error(f"ffmpeg chunking failed at {start}s: {result.stderr}")
            break

    if not chunks:
        raise RuntimeError("No chunks were created from audio file")

    return temp_dir, chunks


def transcribe_chunk(model_config: dict, model, chunk_path: str, language: Optional[str]) -> Tuple[int, str, str]:
    """
    Transcribe a single chunk and return (index, text, detected_language).
    Index is parsed from chunk filename suffix.
    """
    # Extract index from filename like chunk_0001.wav
    try:
        basename = os.path.basename(chunk_path)
        idx_str = os.path.splitext(basename)[0].split('_')[-1]
        index = int(idx_str)
    except Exception:
        index = 0

    transcription_text = ""
    detected_language = language or 'unknown'

    try:
        if model_config["type"] == "openai":
            use_fp16 = torch.cuda.is_available()
            try:
                result = model.transcribe(chunk_path, language=language, fp16=use_fp16, verbose=False)
            except Exception:
                result = model.transcribe(chunk_path, language=language, fp16=False, verbose=False)
            transcription_text = result.get('text', '').strip()
            detected_language = result.get('language', detected_language)
        elif model_config["type"] == "ggml":
            threads = os.getenv("WHISPER_CPP_THREADS", "4")
            cmd = [
                model["whisper_cpp_path"],
                '-m', model['path'],
                '-f', chunk_path,
                '-nt',
                '-t', threads,
                '-bs', '1',
                '--no-prints'
            ]
            if torch.cuda.is_available():
                cmd.append('-fa')
            else:
                cmd.append('-ng')
            if language:
                cmd.extend(['-l', language])
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                out = result.stdout.strip() if result.stdout else ""
                lines = [line.strip() for line in out.split('\n') if line.strip()]
                content_lines = [
                    line for line in lines
                    if not line.startswith('[')
                    and '%]' not in line
                    and not line.startswith('whisper_')
                ]
                transcription_text = ' '.join(content_lines)
                detected_language = language or 'he'
            else:
                transcription_text = ""
    except Exception as e:
        logger.error(f"Chunk transcription error ({chunk_path}): {e}")

    return index, transcription_text, detected_language


# ============================================================================
# Audio Chunk Caching
# Note: Caching is only used for local Whisper/Ivrit models during FFmpeg
# normalization. Deepgram uses its own separate transcription path and does
# not use this caching mechanism.
# ============================================================================

def init_cache_dir():
    """Initialize cache directory and clean old files"""
    if not CACHE_ENABLED:
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Clean old cache files
    cutoff_time = datetime.now() - timedelta(hours=CACHE_MAX_AGE_HOURS)
    cleaned_count = 0

    for cache_file in CACHE_DIR.glob("*.wav"):
        try:
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_time < cutoff_time:
                cache_file.unlink()
                cleaned_count += 1
        except Exception as e:
            logger.warning(f"Failed to clean cache file {cache_file}: {e}")

    if cleaned_count > 0:
        logger.info(f"Cleaned {cleaned_count} old cache files")


def generate_cache_key(audio_data: bytes, sample_rate: int, channels: int) -> str:
    """Generate cache key from audio data and parameters"""
    hasher = hashlib.sha256()
    hasher.update(audio_data)
    hasher.update(f"{sample_rate}:{channels}".encode())
    return hasher.hexdigest()


def get_cached_audio(cache_key: str) -> Optional[str]:
    """Get cached normalized audio file if it exists"""
    if not CACHE_ENABLED:
        return None

    cache_path = CACHE_DIR / f"{cache_key}.wav"
    if cache_path.exists():
        logger.debug(f"Cache hit for {cache_key}")
        return str(cache_path)
    return None


def save_to_cache(cache_key: str, audio_path: str) -> None:
    """Save normalized audio to cache"""
    if not CACHE_ENABLED:
        return

    try:
        cache_path = CACHE_DIR / f"{cache_key}.wav"
        shutil.copy2(audio_path, cache_path)
        logger.debug(f"Cached audio as {cache_key}")
    except Exception as e:
        logger.warning(f"Failed to cache audio: {e}")


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
        self.audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_SIZE)
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
        """Read audio data from FFmpeg in chunks with overlap for context preservation"""
        if not self.ffmpeg_process:
            return

        # Calculate chunk size and overlap size in bytes
        chunk_size = int(SAMPLE_RATE * CHANNELS * 2 * CHUNK_DURATION)  # 2 bytes per sample
        overlap_size = int(SAMPLE_RATE * CHANNELS * 2 * CHUNK_OVERLAP)

        overlap_buffer = b''  # Store overlap from previous chunk

        try:
            while self.is_running:
                # Read new audio data (accounting for overlap)
                new_data_size = chunk_size - len(overlap_buffer)
                audio_data = self.ffmpeg_process.stdout.read(new_data_size)

                if not audio_data:
                    # Stream ended - process remaining overlap if it exists
                    if overlap_buffer and len(overlap_buffer) >= SAMPLE_RATE * CHANNELS * 2:
                        try:
                            self.audio_queue.put_nowait(overlap_buffer)
                        except queue.Full:
                            logger.warning("Audio queue full, dropped final chunk")
                    logger.info("FFmpeg stream ended")
                    break

                # Combine overlap with new data
                full_chunk = overlap_buffer + audio_data

                # Save overlap for next iteration (last CHUNK_OVERLAP seconds)
                if len(full_chunk) >= overlap_size:
                    overlap_buffer = full_chunk[-overlap_size:]
                else:
                    overlap_buffer = full_chunk

                # Put audio chunk in queue for processing.
                # Block briefly to reduce drops; if still full, drop oldest then retry once.
                try:
                    self.audio_queue.put(full_chunk, timeout=0.5)
                except queue.Full:
                    try:
                        self.audio_queue.get_nowait()
                        self.audio_queue.put(full_chunk, timeout=0.2)
                        logger.warning("Audio queue full; evicted oldest chunk and enqueued new one")
                    except Exception:
                        logger.warning("Audio queue saturated; skipping chunk after retry")

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
    """
    Transcribe audio chunks and send results via WebSocket

    Note: This function is only used for local Whisper/Ivrit models with FFmpeg streaming.
    Deepgram transcription uses a separate function (transcribe_with_deepgram) and does not
    use this audio caching mechanism.
    """

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

                # Generate cache key for this audio chunk
                cache_key = generate_cache_key(audio_data, SAMPLE_RATE, CHANNELS)

                # Check cache first
                cached_path = get_cached_audio(cache_key)
                if cached_path:
                    temp_path = cached_path
                    logger.debug("Using cached normalized audio")
                else:
                    # Not in cache - normalize with FFmpeg
                    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as temp_audio:
                        temp_raw = temp_audio.name
                        temp_audio.write(audio_data)

                    # Normalize to 16kHz mono WAV (required by Whisper)
                    temp_path = temp_raw.replace('.raw', '.wav')
                    normalize_cmd = [
                        'ffmpeg',
                        '-f', 's16le',  # Input: signed 16-bit PCM
                        '-ar', str(SAMPLE_RATE),  # Input sample rate
                        '-ac', str(CHANNELS),  # Input channels
                        '-i', temp_raw,
                        '-ar', '16000',  # Whisper requires 16kHz
                        '-ac', '1',  # Mono
                        '-c:a', 'pcm_s16le',  # 16-bit PCM
                        '-y',  # Overwrite
                        temp_path
                    ]
                    # Run FFmpeg asynchronously to avoid blocking
                    norm_process = await asyncio.create_subprocess_exec(
                        *normalize_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    _, norm_stderr = await norm_process.communicate()

                    if norm_process.returncode != 0:
                        error_msg = norm_stderr.decode() if norm_stderr else "Unknown error"
                        logger.error(f"FFmpeg normalization failed: {error_msg}")
                        os.unlink(temp_raw)
                        continue

                    os.unlink(temp_raw)  # Clean up raw file

                    # Save to cache for future use
                    save_to_cache(cache_key, temp_path)
                
                try:
                    # Transcribe audio chunk
                    logger.info(f"Transcribing audio chunk ({len(audio_data)} bytes)")
                    
                    if model_config["type"] == "openai":
                        # Use OpenAI Whisper
                        use_fp16 = torch.cuda.is_available()
                        try:
                            result = model.transcribe(
                                temp_path,
                                language=processor.language,
                                fp16=use_fp16,
                                verbose=False
                            )
                        except Exception as e:
                            logger.warning(f"GPU/FP16 transcribe failed: {e}. Retrying with fp16=False.")
                            result = model.transcribe(
                                temp_path,
                                language=processor.language,
                                fp16=False,
                                verbose=False
                            )
                        transcription_text = result.get('text', '').strip()
                        detected_language = result.get('language', 'unknown')
                    elif model_config["type"] == "ggml":
                        # Use whisper.cpp CLI - output text directly instead of JSON
                        # The -oj flag has issues with mixed output, so we use plain text output
                        threads = os.getenv("WHISPER_CPP_THREADS", "4")
                        cmd = [
                            model["whisper_cpp_path"],
                            "-m", model["path"],
                            "-f", temp_path,
                            "-nt",  # No timestamps in output (plain text)
                            "-t", threads,  # Use env-configured threads (default 4)
                            "-bs", "1",  # Beam size 1 (greedy decoding - FASTEST)
                            "--no-prints"  # Suppress debug output to stderr
                        ]
                        # Enable GPU offload and flash-attn when CUDA is available, else disable GPU explicitly
                        if torch.cuda.is_available():
                            cmd.append("-fa")
                        else:
                            cmd.append("-ng")
                        if processor.language:
                            cmd.extend(["-l", processor.language])

                        # Run command asynchronously to avoid blocking the event loop
                        process = await asyncio.create_subprocess_exec(
                            *cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        stdout, stderr = await process.communicate()

                        if process.returncode == 0:
                            # whisper.cpp outputs text directly to stdout
                            # Clean up the output - remove extra whitespace and newlines
                            transcription_text = stdout.decode('utf-8').strip() if stdout else ""

                            # Remove any debug/status lines that might be present
                            lines = [line.strip() for line in transcription_text.split('\n') if line.strip()]
                            # Filter out lines that look like debug output (contain brackets, percentages, etc.)
                            content_lines = [
                                line for line in lines
                                if not line.startswith('[')
                                and not '%]' in line
                                and not line.startswith('whisper_')
                            ]
                            transcription_text = ' '.join(content_lines)

                            logger.debug(f"Extracted transcription: '{transcription_text[:100]}...'")
                        else:
                            error_msg = stderr.decode('utf-8') if stderr else "Unknown error"
                            logger.error(f"whisper.cpp error: {error_msg}")
                            transcription_text = ""
                        detected_language = processor.language or 'he'  # Default to Hebrew for Ivrit model
                    
                    # Send transcription to client
                    if transcription_text:
                        await websocket.send_json({
                            "type": "transcription",
                            "text": transcription_text,
                            "language": detected_language
                        })
                        logger.info(f"✓ Sent transcription ({len(transcription_text)} chars): {transcription_text[:100]}...")
                    else:
                        logger.warning("⚠ No transcription text extracted from audio chunk")
                    
                finally:
                    # Cleanup temp file
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                        
            except queue.Empty:
                # No audio data available; send heartbeat to keep UI responsive
                await asyncio.sleep(0.1)
                try:
                    await websocket.send_json({"type": "status", "message": "Waiting for audio..."})
                except Exception:
                    pass
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


async def transcribe_with_deepgram(websocket: WebSocket, url: str, language: Optional[str] = None):
    """Transcribe audio stream using Deepgram live transcription (matches user's sample logic)"""
    if not DEEPGRAM_AVAILABLE:
        await websocket.send_json({"error": "Deepgram SDK not available"})
        return

    try:
        # Track connection state to avoid sending after close
        ws_open = True
        dg_closed_event = asyncio.Event()

        # Helper: robustly extract transcript from Deepgram payloads
        def extract_deepgram_transcript(message) -> str:
            try:
                if isinstance(message, dict):
                    alts = message.get('channel', {}).get('alternatives', [])
                    if alts:
                        return alts[0].get('transcript', '')
                # v4 typed objects
                if hasattr(message, 'channel') and hasattr(message.channel, 'alternatives'):
                    alt0 = message.channel.alternatives[0] if message.channel.alternatives else None
                    if alt0 is not None:
                        return getattr(alt0, 'transcript', '') or ''
            except Exception:
                pass
            return ''

        # Helper: print-transcript compatible method
        def print_transcript(json_data):
            alts = json_data.get('channel', {}).get('alternatives', []) if isinstance(json_data, dict) else []
            transcript = alts[0].get('transcript', '') if alts else ''
            print(transcript)

        # Initialize Deepgram client
        client = DeepgramClient(api_key=DEEPGRAM_API_KEY) if DEEPGRAM_API_KEY else DeepgramClient()
        # Apply user-requested params
        TIME_LIMIT = int(os.getenv("DEEPGRAM_TIME_LIMIT", "30"))
        TRANSCRIPT_ONLY = os.getenv("DEEPGRAM_TRANSCRIPT_ONLY", "true").lower() == "true"
        PARAMS = {
            "punctuate": True,
            "numerals": True,
            "model": os.getenv("DEEPGRAM_MODEL", "nova-3"),
            "language": os.getenv("DEEPGRAM_LANGUAGE", "en-US")
        }

        # Create a websocket client (SDK v4)
        connection = client.listen.websocket.v("1")
        # Event handlers
        # Listen for the connection to open
        connection.on(DG_EVENT_OPEN, lambda *_: logger.info("Deepgram connection opened"))

        # Listen for any transcripts received from Deepgram
        def on_message(*args, **kwargs):
            # Deepgram v4 emits handler(self, result=..., **kwargs)
            payload = kwargs.get('result') if 'result' in kwargs else (args[1] if len(args) >= 2 else None)
            sentence = extract_deepgram_transcript(payload) if payload is not None else ''

            if TRANSCRIPT_ONLY:
                if isinstance(payload, dict):
                    print_transcript(payload)
                else:
                    print(sentence)
            else:
                try:
                    print(payload)
                except Exception:
                    print(sentence)

            if sentence:
                try:
                    if ws_open and websocket.client_state == WebSocketState.CONNECTED and not dg_closed_event.is_set():
                        asyncio.create_task(websocket.send_json({
                            "type": "transcription",
                            "text": sentence,
                            "language": language or os.getenv("DEEPGRAM_LANGUAGE", "en-US")
                        }))
                        logger.info(f"✓ Sent Deepgram transcription: {sentence[:100]}...")
                except RuntimeError:
                    pass

        connection.on(DG_EVENT_MESSAGE, on_message)
        # On close, mark closed but don't send over websocket (function will handle finalization)
        # Listen for the connection to close
        def on_close(*_):
            try:
                print('✅ Transcription complete! Connection closed. ✅')
            except Exception:
                pass
            finally:
                try:
                    dg_closed_event.set()
                except Exception:
                    pass
        connection.on(DG_EVENT_CLOSE, on_close)

        # On error, try to notify if still connected, then mark closed
        def on_error(*args, **kwargs):
            err = kwargs.get('error') if 'error' in kwargs else (args[1] if len(args) >= 2 else None)
            try:
                if ws_open and websocket.client_state == WebSocketState.CONNECTED and not dg_closed_event.is_set():
                    asyncio.create_task(websocket.send_json({"error": f"Deepgram error: {err}"}))
            except RuntimeError:
                pass
            finally:
                try:
                    dg_closed_event.set()
                except Exception:
                    pass
        connection.on(DG_EVENT_ERROR, on_error)

        # Surface current Deepgram config to the client for visibility
        try:
            cfg_msg = (
                f"Deepgram connected: model={PARAMS['model']}, lang={PARAMS['language']}, "
                f"tier={(os.getenv('DEEPGRAM_TIER') or 'default')}, limit={TIME_LIMIT}s, "
                f"transcript_only={TRANSCRIPT_ONLY}"
            )
            await websocket.send_json({"type": "status", "message": cfg_msg})
        except Exception:
            await websocket.send_json({"type": "status", "message": "Deepgram connection established, streaming audio..."})

        # Start listening with options
        try:
            from deepgram.clients.listen.v1.websocket.options import LiveOptions
        except Exception as e:
            await websocket.send_json({"error": f"Deepgram SDK options missing: {e}"})
            return

        # Build options from PARAMS and app audio settings
        opts_kwargs = {
            "model": PARAMS["model"],
            "language": PARAMS["language"],
            "punctuate": PARAMS["punctuate"],
            "numerals": PARAMS["numerals"],
            "encoding": "linear16",
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "interim_results": True,
        }
        # Tier may not be allowed on some accounts; include only if set via env
        dg_tier = os.getenv("DEEPGRAM_TIER")
        if dg_tier:
            opts_kwargs["tier"] = dg_tier
        options = LiveOptions(**opts_kwargs)

        # Attempt to start the Deepgram websocket; surface 403 clearly
        try:
            connection.start(options)
        except Exception as e:
            msg = str(e)
            if "403" in msg:
                await websocket.send_json({
                    "error": "Deepgram rejected websocket (HTTP 403). Check API key, model/tier permissions, and account access for live streaming."
                })
            else:
                await websocket.send_json({"error": f"Failed to start Deepgram websocket: {msg}"})
            return

        # Stream audio to Deepgram via ffmpeg to ensure linear16 PCM, avoiding 410/format issues
        await websocket.send_json({"type": "status", "message": "Streaming audio to Deepgram via ffmpeg..."})

        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-user_agent", "Mozilla/5.0",
            "-i", url,
            "-vn",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ac", str(CHANNELS),
            "-ar", str(SAMPLE_RATE),
            "pipe:1",
        ]

        proc = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        start_ts = asyncio.get_event_loop().time()
        time_limit = float(TIME_LIMIT)

        try:
            while True:
                if time_limit != float("inf"):
                    elapsed = asyncio.get_event_loop().time() - start_ts
                    if elapsed >= time_limit:
                        break

                data = await proc.stdout.read(4096)
                if not data:
                    break
                # Stream raw PCM bytes to Deepgram
                connection.send(data)
        finally:
            try:
                proc.terminate()
                await proc.wait()
            except Exception:
                pass

        # Signal finish
        try:
            connection.finish()
        except Exception:
            pass

        # Wait briefly for Deepgram close event, then finalize
        try:
            # Give Deepgram up to 1s to emit close
            try:
                await asyncio.wait_for(dg_closed_event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            # Send completion if still connected
            if ws_open and websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({"type": "complete", "message": "Transcription complete"})
            # Close websocket gracefully
            if websocket.client_state == WebSocketState.CONNECTED:
                ws_open = False
                try:
                    await websocket.close()
                except Exception:
                    pass
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Deepgram transcription error: {e}")
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({"error": f"Deepgram error: {str(e)}"})
        except RuntimeError:
            pass


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
        model_name = data.get("model", "ivrit-large-v3-turbo")
        capture_mode = data.get("captureMode", "full")

        if not url:
            await websocket.send_json({"error": "URL is required"})
            return

        logger.info(f"Starting transcription for URL: {url} with model: {model_name}")

        # Check if user selected Deepgram
        if model_name == "deepgram":
            await transcribe_with_deepgram(websocket, url, language)
            return

        # If capture_mode is first60, perform a 60s capture to WAV and gate transcription
        if capture_mode == "first60":
            await websocket.send_json({"type": "status", "message": "Capturing first 60 seconds (m4a via yt-dlp)..."})
            try:
                # Use yt-dlp to extract audio as m4a and limit to first 60 seconds
                capture_id = str(uuid4())
                output_path = str(CAPTURE_DIR / f"capture_{capture_id}.m4a")
                # yt-dlp progress is printed to stderr; use --newline for line-based updates
                ytdlp_cmd = [
                    'yt-dlp',
                    '--newline',
                    '--extract-audio',
                    '--audio-format', 'm4a',
                    '--audio-quality', '0',
                    '--download-sections', '*00:00-01:00',
                    '--no-playlist',
                    '--no-warnings',
                    '-o', output_path,
                    url
                ]

                proc = await asyncio.create_subprocess_exec(
                    *ytdlp_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                # Read progress lines and emit to UI
                progress_re = re.compile(r"\[download\]\s+(\d+(?:\.\d+)?)%")
                while True:
                    line = await proc.stderr.readline()
                    if not line:
                        break
                    text = line.decode(errors='ignore').strip()
                    m = progress_re.search(text)
                    if m:
                        pct = float(m.group(1))
                        try:
                            await websocket.send_json({"type": "status", "message": f"Capture progress: {pct:.1f}%"})
                        except Exception:
                            pass

                await proc.wait()
                if proc.returncode != 0 or not os.path.exists(output_path):
                    err_out = await proc.stderr.read()
                    err = err_out.decode(errors='ignore') if err_out else "Unknown yt-dlp error"
                    await websocket.send_json({"error": f"Capture failed: {err}"})
                    return

                # Store capture metadata and notify client
                CAPTURES[capture_id] = {
                    "path": output_path,
                    "created_at": datetime.utcnow().isoformat(),
                    "url": url,
                }
                await websocket.send_json({"type": "capture_ready", "capture_id": capture_id})

                # Wait for further messages to transcribe the capture
                while True:
                    try:
                        msg = await websocket.receive_json()
                    except WebSocketDisconnect:
                        return
                    action = msg.get("action")
                    if action == "transcribe_capture":
                        cid = msg.get("capture_id")
                        lang2 = msg.get("language", language)
                        model2 = msg.get("model", model_name)
                        info = CAPTURES.get(cid)
                        if not info or not os.path.exists(info["path"]):
                            await websocket.send_json({"error": "Captured file not found"})
                            return

                        await websocket.send_json({"type": "status", "message": "Transcribing captured audio..."})
                        # Route based on model
                        try:
                            model = load_model(model2)
                            model_config = MODEL_CONFIGS[model2]
                            if model_config["type"] == "openai":
                                use_fp16 = torch.cuda.is_available()
                                try:
                                    result = model.transcribe(info["path"], language=lang2, fp16=use_fp16, verbose=False)
                                except Exception:
                                    result = model.transcribe(info["path"], language=lang2, fp16=False, verbose=False)
                                text = result.get('text', '').strip()
                                det_lang = result.get('language', lang2 or 'unknown')
                            elif model_config["type"] == "ggml":
                                # whisper.cpp CLI prefers WAV; convert m4a to WAV temporarily if needed
                                input_path = info["path"]
                                temp_wav = None
                                if not input_path.lower().endswith('.wav'):
                                    try:
                                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
                                            temp_wav = tf.name
                                        ff_cmd = [
                                            'ffmpeg','-y','-i', input_path,
                                            '-vn','-ac','1','-ar','16000','-c:a','pcm_s16le',
                                            temp_wav
                                        ]
                                        ff_res = subprocess.run(ff_cmd, capture_output=True, text=True)
                                        if ff_res.returncode == 0:
                                            input_path = temp_wav
                                        else:
                                            logger.error(f"ffmpeg convert error: {ff_res.stderr}")
                                    except Exception as ce:
                                        logger.error(f"Temporary WAV conversion failed: {ce}")

                                cmd = [
                                    model["whisper_cpp_path"],
                                    '-m', model["path"],
                                    '-f', input_path,
                                    '-nt',
                                    '-t', '4',
                                    '-bs', '1',
                                    '--no-prints'
                                ]
                                if lang2:
                                    cmd.extend(['-l', lang2])
                                res = subprocess.run(cmd, capture_output=True, text=True)
                                if res.returncode == 0:
                                    raw = res.stdout.strip()
                                    lines = [l.strip() for l in raw.split('\n') if l.strip()]
                                    content = [l for l in lines if not l.startswith('[') and '%]' not in l and not l.startswith('whisper_')]
                                    text = ' '.join(content)
                                else:
                                    logger.error(f"whisper.cpp error: {res.stderr}")
                                    text = ""
                                det_lang = lang2 or 'he'
                                # Cleanup temporary WAV if created
                                if temp_wav:
                                    try:
                                        os.unlink(temp_wav)
                                    except Exception:
                                        pass
                            elif model_config["type"] == "deepgram":
                                # Use Deepgram v3 file transcription API
                                try:
                                    client = DeepgramClient(api_key=DEEPGRAM_API_KEY) if DEEPGRAM_API_KEY else DeepgramClient()
                                    model = os.getenv("DEEPGRAM_MODEL", "nova-3")

                                    with open(info["path"], "rb") as f:
                                        response = client.listen.v1.media.transcribe_file(
                                            request=f.read(),
                                            model=model
                                        )
                                    # Extract transcript
                                    try:
                                        text = response.results.channels[0].alternatives[0].transcript.strip()
                                    except Exception:
                                        text = ""
                                    det_lang = lang2 or os.getenv("DEEPGRAM_LANGUAGE", "en-US")
                                except Exception as e:
                                    await websocket.send_json({"error": f"Deepgram error: {e}"})
                                    return
                            else:
                                await websocket.send_json({"error": "Unsupported model type"})
                                return

                            if text:
                                await websocket.send_json({"type": "transcription", "text": text, "language": det_lang})
                            await websocket.send_json({"type": "complete", "message": "Transcription complete"})
                        finally:
                            try:
                                if os.path.exists(info["path"]):
                                    os.unlink(info["path"])
                                CAPTURES.pop(cid, None)
                            except Exception:
                                pass
                        return

            except Exception as e:
                logger.error(f"Capture flow error: {e}")
                await websocket.send_json({"error": str(e)})
                return

        # Check if we should use yt-dlp or direct streaming
        if should_use_ytdlp(url):
            await websocket.send_json({"type": "status", "message": "Downloading audio with yt-dlp..."})

            # Download entire file first with yt-dlp
            audio_file = download_audio_with_ytdlp(url, language)
            if not audio_file:
                await websocket.send_json({"error": "Failed to download audio from URL"})
                return

            # Decide path: parallel chunking (flagged) vs single-file transcription (default)
            if USE_PARALLEL_TRANSCRIPTION:
                await websocket.send_json({"type": "status", "message": "Chunking audio and starting parallel transcription..."})
                try:
                    model = load_model(model_name)
                    model_config = MODEL_CONFIGS[model_name]

                    # Split into chunks with overlap
                    temp_dir, chunks = split_audio_into_chunks(
                        audio_file,
                        chunk_seconds=YTDLP_CHUNK_SECONDS,
                        overlap_seconds=YTDLP_CHUNK_OVERLAP
                    )

                    # Transcribe chunks in parallel threads
                    results: List[Tuple[int, str, str]] = []
                    with ThreadPoolExecutor(max_workers=max(1, PARALLEL_WORKERS)) as executor:
                        futures = {
                            executor.submit(transcribe_chunk, model_config, model, path, language): idx
                            for idx, path in chunks
                        }
                        for future in as_completed(futures):
                            try:
                                res = future.result()
                                results.append(res)
                            except Exception as e:
                                logger.error(f"Parallel transcription worker failed: {e}")

                    # Order results by index and stream partials to client
                    for idx, text, det_lang in sorted(results, key=lambda r: r[0]):
                        if text:
                            await websocket.send_json({
                                "type": "transcription",
                                "text": text,
                                "language": det_lang
                            })

                    await websocket.send_json({"type": "complete", "message": "Transcription complete"})

                finally:
                    # Cleanup chunk directory and original file
                    try:
                        if 'temp_dir' in locals() and os.path.isdir(temp_dir):
                            shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception:
                        pass
                    if os.path.exists(audio_file):
                        os.unlink(audio_file)

                return

            # Transcribe the downloaded file directly (default path)
            await websocket.send_json({"type": "status", "message": "Transcribing downloaded audio..."})

            try:
                model = load_model(model_name)
                model_config = MODEL_CONFIGS[model_name]

                if model_config["type"] == "openai":
                    use_fp16 = torch.cuda.is_available()
                    try:
                        result = model.transcribe(audio_file, language=language, fp16=use_fp16, verbose=False)
                    except Exception as e:
                        logger.warning(f"GPU/FP16 ytdlp transcribe failed: {e}. Retrying with fp16=False.")
                        result = model.transcribe(audio_file, language=language, fp16=False, verbose=False)
                    transcription_text = result.get('text', '').strip()
                    detected_language = result.get('language', 'unknown')
                elif model_config["type"] == "ggml":
                    # Use whisper.cpp CLI - output text directly instead of JSON
                    cmd = [
                        model["whisper_cpp_path"],
                        "-m", model["path"],
                        "-f", audio_file,
                        "-nt",  # No timestamps in output (plain text)
                        "-t", "4",  # Use 4 threads for faster processing
                        "-bs", "1",  # Beam size 1 (greedy decoding - FASTEST)
                        "--no-prints"
                    ]
                    if language:
                        cmd.extend(["-l", language])

                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        # whisper.cpp outputs text directly to stdout
                        transcription_text = result.stdout.strip()

                        # Remove any debug/status lines
                        lines = [line.strip() for line in transcription_text.split('\n') if line.strip()]
                        content_lines = [
                            line for line in lines
                            if not line.startswith('[')
                            and not '%]' in line
                            and not line.startswith('whisper_')
                        ]
                        transcription_text = ' '.join(content_lines)
                        logger.debug(f"yt-dlp transcription: '{transcription_text[:100]}...'")
                    else:
                        logger.error(f"whisper.cpp error: {result.stderr}")
                        transcription_text = ""
                    detected_language = language or 'he'  # Default to Hebrew for Ivrit model

                if transcription_text:
                    await websocket.send_json({
                        "type": "transcription",
                        "text": transcription_text,
                        "language": detected_language
                    })

                await websocket.send_json({"type": "complete", "message": "Transcription complete"})

            finally:
                # Cleanup downloaded file
                if os.path.exists(audio_file):
                    os.unlink(audio_file)

            return

        # Otherwise, use direct FFmpeg streaming (existing code)
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


@app.get("/gpu")
async def gpu_diagnostics():
    """GPU diagnostics endpoint"""
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    cuda_version = torch.version.cuda if hasattr(torch.version, "cuda") else None
    devices = []
    for idx in range(device_count):
        try:
            props = torch.cuda.get_device_properties(idx)
            devices.append({
                "index": idx,
                "name": props.name,
                "total_memory_mb": round(props.total_memory / (1024 * 1024), 2),
                "multi_processor_count": props.multi_processor_count,
                "major": props.major,
                "minor": props.minor,
            })
        except Exception as e:
            devices.append({"index": idx, "error": str(e)})

    return {
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "device_count": device_count,
        "devices": devices,
        "env": {
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
            "NVIDIA_VISIBLE_DEVICES": os.getenv("NVIDIA_VISIBLE_DEVICES"),
        }
    }


@app.get("/api/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    if not CACHE_ENABLED:
        return {"enabled": False}

    try:
        cache_files = list(CACHE_DIR.glob("*.wav"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "enabled": True,
            "file_count": len(cache_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "max_age_hours": CACHE_MAX_AGE_HOURS
        }
    except Exception as e:
        return {"enabled": True, "error": str(e)}


@app.post("/api/cache/clear")
async def clear_cache():
    """Clear all cached audio files"""
    if not CACHE_ENABLED:
        return {"enabled": False}

    try:
        deleted_count = 0
        for cache_file in CACHE_DIR.glob("*.wav"):
            cache_file.unlink()
            deleted_count += 1

        return {"success": True, "deleted": deleted_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

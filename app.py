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
import time
from pathlib import Path
from typing import Optional, List, Tuple
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

# Primary: Import faster_whisper for Ivrit CT2 models (recommended)
try:
    import faster_whisper
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("ERROR: faster_whisper not available - Ivrit CT2 models will not work")
    print("Install with: pip install faster-whisper>=1.1.1")

# Optional: Import openai-whisper (not recommended - may conflict with faster-whisper)
try:
    import whisper
    OPENAI_WHISPER_AVAILABLE = True
    if FASTER_WHISPER_AVAILABLE:
        print("WARNING: Both openai-whisper and faster-whisper are installed. This may cause conflicts.")
        print("Recommendation: Use only faster-whisper for better performance and to avoid conflicts.")
except ImportError:
    OPENAI_WHISPER_AVAILABLE = False
    # This is actually preferred - we don't need openai-whisper

import aiohttp
import torch

# Try to import ivrit package
try:
    import ivrit
    IVRIT_PACKAGE_AVAILABLE = True
except ImportError:
    IVRIT_PACKAGE_AVAILABLE = False
    print("ivrit package not available - Advanced Ivrit features disabled")

# Try to import pyannote for speaker diarization
try:
    from pyannote.audio import Pipeline
    import torchaudio
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("pyannote.audio not available - Speaker diarization disabled")
    print("Install with: pip install pyannote.audio torchaudio")

# whisper.cpp/GGML support has been removed in favor of faster_whisper/CT2 models
# which provide better performance and quality for Hebrew transcription
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

# Log dependency availability at startup
logger.info("=" * 60)
logger.info("Dependency Status:")
logger.info(f"  OpenAI Whisper: {'✓ Available' if OPENAI_WHISPER_AVAILABLE else '✗ Not Available'}")
logger.info(f"  Faster Whisper: {'✓ Available' if FASTER_WHISPER_AVAILABLE else '✗ Not Available'}")
logger.info(f"  Ivrit Package: {'✓ Available' if IVRIT_PACKAGE_AVAILABLE else '✗ Not Available'}")
logger.info(f"  Pyannote Audio: {'✓ Available' if PYANNOTE_AVAILABLE else '✗ Not Available (Diarization disabled)'}")
# Whisper.cpp/GGML support removed - using faster_whisper instead
logger.info(f"  Deepgram SDK: {'✓ Available' if DEEPGRAM_AVAILABLE else '✗ Not Available'}")
logger.info(f"  CUDA: {'✓ Available' if torch.cuda.is_available() else '✗ Not Available'}")
logger.info("=" * 60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    
    # Critical dependency check
    if not FASTER_WHISPER_AVAILABLE and not OPENAI_WHISPER_AVAILABLE and not (DEEPGRAM_AVAILABLE and DEEPGRAM_API_KEY):
        logger.error("=" * 60)
        logger.error("CRITICAL: No transcription backend available!")
        logger.error("Please install one of the following:")
        logger.error("  • faster-whisper: pip install faster-whisper>=1.1.1")
        logger.error("  • openai-whisper: pip install openai-whisper")
        logger.error("  • Configure Deepgram API with DEEPGRAM_API_KEY")
        logger.error("=" * 60)
    
    try:
        if MODEL_CONFIGS:
            logger.info(f"Loading default model: {MODEL_SIZE}")
            try:
                load_model(MODEL_SIZE)
                logger.info(f"Default model '{MODEL_SIZE}' loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load default model '{MODEL_SIZE}': {e}")
                # Try to load any available model
                for model_name in MODEL_CONFIGS.keys():
                    if model_name != MODEL_SIZE:
                        try:
                            logger.info(f"Trying alternative model: {model_name}")
                            load_model(model_name)
                            logger.info(f"Successfully loaded alternative model: {model_name}")
                            break
                        except Exception as e2:
                            logger.warning(f"Failed to load {model_name}: {e2}")
                            continue
        else:
            logger.warning("No models available at startup. Models can be loaded on-demand via API.")

        # Initialize audio cache
        init_cache_dir()
        logger.info(f"Audio cache initialized (enabled: {CACHE_ENABLED})")

        # Initialize capture directory
        init_capture_dir()
        logger.info("Capture directory initialized for first-60s mode")
        
        # Initialize download cache directory
        init_download_cache_dir()
        logger.info("Download cache initialized for URL-based caching")
    except Exception as e:
        logger.error(f"Critical startup error: {e}")
        # Don't raise - allow the app to start even without models
        logger.warning("Application starting without pre-loaded models. Models will be loaded on-demand.")

    yield

    # Shutdown (if needed)
    logger.info("Application shutting down")

# Initialize FastAPI app with lifespan
app = FastAPI(title="Live Transcription Service", version="1.0.0", lifespan=lifespan)

# Global Whisper models
whisper_models = {}
current_model = None
current_model_name = None
# Global diarization pipeline cache
diarization_pipeline = None
diarization_pipeline_lock = threading.Lock()

# Default model configuration - always use ivrit-ct2 with faster_whisper
MODEL_SIZE = os.getenv("WHISPER_MODEL", "whisper-v3-turbo")  # Default to multilingual model
logger.info(f"Default model: {MODEL_SIZE} (using faster_whisper with CT2 format)")

# Model configurations
MODEL_CONFIGS = {}

# Primary models - faster_whisper CT2 models (always available)
if FASTER_WHISPER_AVAILABLE:
    MODEL_CONFIGS.update({
        # Primary Ivrit model - best for Hebrew
        "ivrit-ct2": {
            "type": "faster_whisper",
            "name": os.getenv("IVRIT_MODEL_NAME", "ivrit-ai/whisper-large-v3-turbo-ct2"),
            "device": os.getenv("IVRIT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
            "compute_type": os.getenv("IVRIT_COMPUTE_TYPE", "float16" if torch.cuda.is_available() else "int8")
        },
        # Alternative name for the same model
        "ivrit-v3-turbo": {
            "type": "faster_whisper",
            "name": "ivrit-ai/whisper-large-v3-turbo-ct2",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "compute_type": "float16" if torch.cuda.is_available() else "int8"
        },
        # General Whisper v3 turbo model
        "whisper-v3-turbo": {
            "type": "faster_whisper",
            "name": "large-v3-turbo",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "compute_type": "float16" if torch.cuda.is_available() else "int8"
        }
    })
else:
    logger.error("faster_whisper is not available! Please install it with: pip install faster-whisper")

# Optional: OpenAI Whisper models (NOT RECOMMENDED - may conflict with faster-whisper)
if OPENAI_WHISPER_AVAILABLE and not FASTER_WHISPER_AVAILABLE:
    # Only add OpenAI models if faster-whisper is not available
    logger.warning("Using openai-whisper models. Consider switching to faster-whisper for better performance.")
    MODEL_CONFIGS.update({
        "tiny": {"type": "openai", "name": "tiny"},
        "base": {"type": "openai", "name": "base"},
        "small": {"type": "openai", "name": "small"},
        "medium": {"type": "openai", "name": "medium"},
        "large": {"type": "openai", "name": "large"},
    })
elif OPENAI_WHISPER_AVAILABLE and FASTER_WHISPER_AVAILABLE:
    # Both are available - skip OpenAI models to avoid conflicts
    logger.info("Skipping OpenAI Whisper models to avoid conflicts with faster-whisper")

# Optional: Deepgram API (if configured)
if DEEPGRAM_AVAILABLE and DEEPGRAM_API_KEY:
    MODEL_CONFIGS["deepgram"] = {"type": "deepgram", "model": "nova-2", "language": "en"}

# Log available models
if MODEL_CONFIGS:
    logger.info(f"Available models: {', '.join(MODEL_CONFIGS.keys())}")
    # Ensure the default model is available
    if MODEL_SIZE not in MODEL_CONFIGS:
        logger.warning(f"Default model '{MODEL_SIZE}' not available. Available models: {', '.join(MODEL_CONFIGS.keys())}")
        # Try to select first available model
        if MODEL_CONFIGS:
            MODEL_SIZE = list(MODEL_CONFIGS.keys())[0]
            logger.info(f"Switching to available model: {MODEL_SIZE}")
else:
    logger.error("No models available! Please install faster-whisper or configure alternatives.")
    # Critical: Ensure faster_whisper is available for Ivrit models
    if not FASTER_WHISPER_AVAILABLE:
        logger.error("CRITICAL: faster-whisper is not installed. Install with: pip install faster-whisper>=1.1.1")

# Audio processing configuration
CHUNK_DURATION = 5   # seconds - very short for fast real-time processing
CHUNK_OVERLAP = 1    # seconds - minimal overlap
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
CHANNELS = 1  # Mono audio
AUDIO_QUEUE_SIZE = 200  # Large queue to handle slow models (Ivrit, large models)

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

# URL-based download cache to avoid re-downloads when switching models
DOWNLOAD_CACHE_DIR = Path("cache/downloads")
URL_DOWNLOADS = {}  # Maps URL hash to downloaded file path

def init_capture_dir():
    try:
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"Failed to create capture dir: {e}")


def init_download_cache_dir():
    """Initialize download cache directory for URL-based audio caching"""
    try:
        DOWNLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Clean old downloads (older than 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        for cache_file in DOWNLOAD_CACHE_DIR.glob("*.wav"):
            try:
                file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_time < cutoff_time:
                    cache_file.unlink()
                    # Remove from URL_DOWNLOADS if present
                    for url_hash, path in list(URL_DOWNLOADS.items()):
                        if path == str(cache_file):
                            del URL_DOWNLOADS[url_hash]
            except Exception as e:
                logger.warning(f"Failed to clean old download cache: {e}")
    except Exception as e:
        logger.warning(f"Failed to create download cache dir: {e}")


def get_url_hash(url: str) -> str:
    """Generate a unique hash for a URL to use as cache key"""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def get_cached_download(url: str) -> Optional[str]:
    """Check if we have a cached download for this URL"""
    url_hash = get_url_hash(url)
    
    # Check in-memory cache first
    if url_hash in URL_DOWNLOADS:
        cached_path = URL_DOWNLOADS[url_hash]
        if os.path.exists(cached_path):
            logger.info(f"Found cached download for URL (in-memory): {cached_path}")
            return cached_path
        else:
            # File was deleted, remove from cache
            del URL_DOWNLOADS[url_hash]
    
    # Check disk cache
    cache_pattern = f"{url_hash}_*.wav"
    for cached_file in DOWNLOAD_CACHE_DIR.glob(cache_pattern):
        if cached_file.exists():
            URL_DOWNLOADS[url_hash] = str(cached_file)
            logger.info(f"Found cached download for URL (disk): {cached_file}")
            return str(cached_file)
    
    return None


def save_download_to_cache(url: str, audio_file: str) -> str:
    """Save a downloaded audio file to URL cache and return the cached path"""
    try:
        url_hash = get_url_hash(url)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cached_filename = f"{url_hash}_{timestamp}.wav"
        cached_path = DOWNLOAD_CACHE_DIR / cached_filename
        
        # If the file is already in the cache directory, just update the mapping
        if DOWNLOAD_CACHE_DIR in Path(audio_file).parents:
            URL_DOWNLOADS[url_hash] = audio_file
            logger.info(f"Audio file already in cache directory: {audio_file}")
            return audio_file
        
        # Copy the file to cache directory
        shutil.copy2(audio_file, cached_path)
        URL_DOWNLOADS[url_hash] = str(cached_path)
        logger.info(f"Cached download for URL: {cached_path}")
        
        # Clean up the original temp file if it's outside cache dir
        try:
            if os.path.exists(audio_file) and DOWNLOAD_CACHE_DIR not in Path(audio_file).parents:
                os.unlink(audio_file)
                temp_dir = os.path.dirname(audio_file)
                if temp_dir and '/tmp' in temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            logger.debug(f"Failed to clean up temp file: {e}")
        
        return str(cached_path)
    except Exception as e:
        logger.error(f"Failed to cache download: {e}")
        return audio_file  # Return original file on cache failure


def load_model(model_name: str):
    """Load a model based on its configuration"""
    global current_model, current_model_name

    if model_name == current_model_name and current_model is not None:
        return current_model

    if model_name not in MODEL_CONFIGS:
        # Provide helpful error message about available models
        available = list(MODEL_CONFIGS.keys())
        if not available:
            raise ValueError(f"No models are available. Please install faster-whisper, openai-whisper, or configure Deepgram API.")
        raise ValueError(f"Unknown model: {model_name}. Available models: {', '.join(available)}")

    config = MODEL_CONFIGS[model_name]

    if config["type"] == "openai":
        if not OPENAI_WHISPER_AVAILABLE:
            # List alternative models
            alternatives = [m for m, c in MODEL_CONFIGS.items() if c["type"] != "openai"]
            alt_msg = f" Try one of these instead: {', '.join(alternatives)}" if alternatives else ""
            raise ValueError(f"openai-whisper is not installed. Cannot load OpenAI Whisper models.{alt_msg}")
        
        logger.info(f"Loading OpenAI Whisper model: {config['name']}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")
        try:
            model = whisper.load_model(config["name"], device=device)
        except Exception as e:
            logger.warning(f"Failed to load model on {device}: {e}. Falling back to CPU.")
            model = whisper.load_model(config["name"], device="cpu")
    
    elif config["type"] == "faster_whisper":
        if not FASTER_WHISPER_AVAILABLE:
            raise ValueError(
                "faster_whisper is not installed. Cannot load Ivrit CT2 models.\n"
                "Install with: pip install faster-whisper>=1.1.1\n"
                "Or use Docker: docker-compose -f docker-compose.ivrit.yml up"
            )
        
        model_name_or_path = config.get("name", "ivrit-ai/whisper-large-v3-turbo-ct2")
        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        compute_type = config.get("compute_type", "float16" if device == "cuda" else "int8")
        
        logger.info(f"Loading faster_whisper model: {model_name_or_path}")
        logger.info(f"Device: {device}, Compute type: {compute_type}")
        
        try:
            # Load the faster_whisper model
            model = faster_whisper.WhisperModel(
                model_name_or_path,
                device=device,
                compute_type=compute_type,
                num_workers=1,
                download_root="/root/.cache/whisper"
            )
            # Wrap in a dict to maintain consistency with other model types
            model = {
                "type": "faster_whisper",
                "model": model,
                "config": config
            }
            logger.info(f"Successfully loaded faster_whisper model: {model_name_or_path}")
        except Exception as e:
            logger.error(f"Failed to load faster_whisper model: {e}")
            # Fallback to CPU with int8
            if device == "cuda":
                logger.info("Attempting CPU fallback with int8 compute type...")
                try:
                    model = faster_whisper.WhisperModel(
                        model_name_or_path,
                        device="cpu",
                        compute_type="int8",
                        num_workers=1
                    )
                    model = {
                        "type": "faster_whisper",
                        "model": model,
                        "config": config
                    }
                    logger.info("Successfully loaded model on CPU")
                except Exception as e2:
                    raise ValueError(f"Failed to load faster_whisper model on both GPU and CPU: {e2}")
            else:
                raise
    
    else:
        raise ValueError(f"Unknown model type: {config['type']}")

    current_model = model
    current_model_name = model_name
    return model


def get_diarization_pipeline():
    """Load and cache the pyannote diarization pipeline"""
    global diarization_pipeline
    
    if not PYANNOTE_AVAILABLE:
        logger.warning("Pyannote not available - diarization disabled")
        return None
    
    with diarization_pipeline_lock:
        if diarization_pipeline is None:
            try:
                logger.info("Loading pyannote diarization pipeline...")
                # Try to load the Ivrit-optimized model first
                try:
                    diarization_pipeline = Pipeline.from_pretrained(
                        "ivrit-ai/pyannote-speaker-diarization-3.1",
                        use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
                    )
                except Exception as e:
                    logger.warning(f"Failed to load Ivrit diarization model: {e}")
                    # Fall back to the standard model
                    diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
                    )
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    diarization_pipeline.to(torch.device("cuda"))
                
                logger.info("Diarization pipeline loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load diarization pipeline: {e}")
                return None
    
    return diarization_pipeline


async def transcribe_with_diarization(
    model, 
    model_config: dict, 
    audio_file: str, 
    language: Optional[str],
    websocket: WebSocket,
    model_name: str = None
) -> Tuple[List[dict], str]:
    """
    Transcribe audio with speaker diarization.
    Returns (diarized_segments, detected_language)
    
    Each segment contains:
    - start: start time in seconds
    - end: end time in seconds  
    - speaker: speaker label (SPEAKER_1, SPEAKER_2, etc.)
    - text: transcribed text
    """
    import time
    
    logger.info(f"Starting transcription with diarization for {audio_file}")
    
    # Send status update
    await websocket.send_json({
        "type": "status",
        "message": "Starting transcription with speaker diarization..."
    })
    
    # Step 1: Run diarization to get speaker segments
    pipeline = get_diarization_pipeline()
    if pipeline is None:
        logger.warning("Diarization pipeline not available, falling back to regular transcription")
        # Fall back to regular transcription
        transcript, lang = await transcribe_with_incremental_output(
            model, model_config, audio_file, language, websocket, model_name
        )
        return [{"text": transcript, "speaker": "SPEAKER_1"}], lang
    
    try:
        # Run diarization
        await websocket.send_json({
            "type": "status", 
            "message": "Analyzing speakers in audio..."
        })
        
        diarization = pipeline(audio_file)
        
        # Convert diarization output to segments
        speaker_segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker
            })
        
        # Renumber speakers to SPEAKER_1, SPEAKER_2, etc.
        speaker_mapping = {}
        speaker_counter = 1
        for segment in speaker_segments:
            if segment["speaker"] not in speaker_mapping:
                speaker_mapping[segment["speaker"]] = f"SPEAKER_{speaker_counter}"
                speaker_counter += 1
            segment["speaker"] = speaker_mapping[segment["speaker"]]
        
        logger.info(f"Found {len(speaker_mapping)} speakers in audio")
        
        # Step 2: Transcribe with timestamps
        await websocket.send_json({
            "type": "status",
            "message": f"Transcribing audio with {len(speaker_mapping)} speakers..."
        })
        
        # Use faster_whisper for transcription with word timestamps
        if model_config["type"] == "faster_whisper":
            # Extract the actual WhisperModel from the wrapper dict
            if isinstance(model, dict) and "model" in model:
                fw_model = model["model"]
            else:
                fw_model = model
            
            # Determine language
            default_lang = None
            if model_name and "ivrit" in model_name.lower():
                default_lang = "he"
            
            logger.info(f"Running faster_whisper transcription with word timestamps for diarization")
            
            # Transcribe with word-level timestamps for better alignment
            # Note: faster_whisper.transcribe returns a generator
            segments_generator, info = fw_model.transcribe(
                audio_file,
                language=language or default_lang,
                word_timestamps=True,  # Enable word timestamps for better alignment
                beam_size=int(os.getenv("IVRIT_BEAM_SIZE", "5")),
                best_of=5,
                patience=1,
                temperature=0
            )
            
            # Convert segments to list with timestamps and show progress
            transcription_segments = []
            segment_count = 0
            start_time_transcribe = time.time()
            
            for segment in segments_generator:
                transcription_segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                })
                segment_count += 1
                
                # Send progress update every 10 segments
                if segment_count % 10 == 0:
                    elapsed = time.time() - start_time_transcribe
                    await websocket.send_json({
                        "type": "status",
                        "message": f"Transcribing: {segment_count} segments processed ({elapsed:.1f}s elapsed)..."
                    })
            
            detected_language = info.language if info else (language or 'unknown')
            logger.info(f"Transcription complete. Detected language: {detected_language}, Segments: {len(transcription_segments)}")
            
        else:
            # Fallback for OpenAI whisper
            result = model.transcribe(
                audio_file,
                language=language,
                verbose=False,
                word_timestamps=True
            )
            
            transcription_segments = []
            for segment in result.get("segments", []):
                transcription_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip()
                })
            
            detected_language = result.get("language", language or 'unknown')
        
        # Step 3: Align transcription with speaker segments
        diarized_segments = []
        
        for speaker_seg in speaker_segments:
            speaker_text = []
            
            # Find transcription segments that overlap with this speaker segment
            for trans_seg in transcription_segments:
                # Check for overlap
                overlap_start = max(speaker_seg["start"], trans_seg["start"])
                overlap_end = min(speaker_seg["end"], trans_seg["end"])
                
                if overlap_end > overlap_start:
                    # There's overlap - this text belongs to this speaker
                    # Calculate overlap percentage
                    trans_duration = trans_seg["end"] - trans_seg["start"]
                    overlap_duration = overlap_end - overlap_start
                    
                    if trans_duration > 0:
                        overlap_pct = overlap_duration / trans_duration
                        
                        # Only include if significant overlap (>50%)
                        if overlap_pct > 0.5:
                            speaker_text.append(trans_seg["text"])
            
            # Combine text for this speaker segment
            if speaker_text:
                combined_text = " ".join(speaker_text)
                diarized_segments.append({
                    "start": speaker_seg["start"],
                    "end": speaker_seg["end"],
                    "speaker": speaker_seg["speaker"],
                    "text": combined_text
                })
        
        # Send incremental results
        total_segments = len(diarized_segments)
        for i, segment in enumerate(diarized_segments):
            # Format timestamp
            start_time = timedelta(seconds=int(segment["start"]))
            end_time = timedelta(seconds=int(segment["end"]))
            
            # Format as [HH:MM:SS-HH:MM:SS] SPEAKER_X: "text"
            formatted_output = (
                f"[{str(start_time).split('.')[0]}-{str(end_time).split('.')[0]}] "
                f"{segment['speaker']}: \"{segment['text']}\""
            )
            
            await websocket.send_json({
                "type": "transcription_chunk",
                "text": formatted_output,
                "chunk_index": i,
                "total_chunks": total_segments,
                "is_final": i == total_segments - 1,
                "speaker": segment["speaker"],
                "start": segment["start"],
                "end": segment["end"]
            })
            
            # Small delay to prevent overwhelming the client
            if i % 5 == 0:
                await asyncio.sleep(0.1)
        
        logger.info(f"Diarization complete: {len(diarized_segments)} segments")
        return diarized_segments, detected_language
        
    except Exception as e:
        logger.error(f"Diarization failed: {e}", exc_info=True)
        # Fall back to regular transcription
        await websocket.send_json({
            "type": "status",
            "message": "Diarization failed, falling back to regular transcription..."
        })
        
        # Call the function directly (it's in the same file)
        transcript, lang = await transcribe_with_incremental_output(
            model, model_config, audio_file, language, websocket, model_name
        )
        return [{"text": transcript, "speaker": "SPEAKER_1"}], lang


def should_use_ytdlp(url: str) -> bool:
    """Determine if URL should use yt-dlp instead of direct FFmpeg streaming"""
    # Use yt-dlp for known video platforms and complex URLs
    ytdlp_patterns = [
        'youtube.com', 'youtu.be', 'vimeo.com', 'dailymotion.com',
        'facebook.com', 'twitter.com', 'twitch.tv', 'tiktok.com',
        'instagram.com', 'reddit.com'
    ]
    
    # Special handling for YouTube URLs with various formats
    youtube_patterns = [
        r'youtube\.com/watch\?v=',
        r'youtu\.be/',
        r'youtube\.com/embed/',
        r'youtube\.com/v/',
        r'm\.youtube\.com'
    ]
    
    url_lower = url.lower()
    
    # Check standard patterns
    if any(pattern in url_lower for pattern in ytdlp_patterns):
        return True
    
    # Check YouTube regex patterns
    import re
    for pattern in youtube_patterns:
        if re.search(pattern, url_lower):
            return True
    
    return False


async def download_audio_with_ffmpeg(url: str, format: str = 'wav', duration: int = 60, websocket = None, use_cache: bool = True) -> Optional[str]:
    """
    Download audio using ffmpeg directly (proven working method from test_deepgram.py)
    Now async with progress monitoring support and URL caching

    Args:
        url: URL to download
        format: Audio format (wav or m4a)
        duration: Duration in seconds to download (default: 60, 0 = complete file)
        websocket: Optional WebSocket connection for progress updates
        use_cache: Whether to check/use cached downloads
    """
    
    # Check cache first if enabled and duration is 0 (complete file)
    if use_cache and duration == 0:
        cached_file = get_cached_download(url)
        if cached_file:
            if websocket:
                file_size_mb = os.path.getsize(cached_file) / (1024 * 1024)
                await websocket.send_json({
                    "type": "status",
                    "message": f"✅ Using cached audio file ({file_size_mb:.1f} MB)"
                })
            return cached_file
    if duration == 0:
        logger.info(f"Downloading COMPLETE audio file with ffmpeg (format: {format})...")
    else:
        logger.info(f"Downloading audio with ffmpeg (format: {format}, duration: {duration}s)...")

    try:
        temp_dir = tempfile.mkdtemp()
        audio_file = os.path.join(temp_dir, f'audio.{format}')
        progress_file = os.path.join(temp_dir, 'progress.txt')

        # Use ffmpeg directly with loudnorm filter for optimal audio quality
        cmd = [
            'ffmpeg',
            '-i', url,
        ]

        # Only add duration limit if specified (0 = complete file)
        if duration > 0:
            cmd.extend(['-t', str(duration)])

        cmd.extend([
            '-vn',  # No video
            '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',  # Loudness normalization filter
            '-acodec', 'pcm_s16le' if format == 'wav' else 'aac',  # Codec based on format
            '-ar', '44100',  # 44.1kHz sample rate
            '-ac', '2',  # Stereo
            '-progress', progress_file,  # Progress output to file
            '-y',  # Overwrite output
            audio_file
        ])

        logger.info(f"Running: ffmpeg -i {url[:50]}... -t {duration} ...")

        # Run ffmpeg asynchronously (following pattern from line 1374)
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Monitor progress if websocket provided
        if websocket:
            import time
            start_time = time.time()
            last_update = 0
            last_file_size = 0
            estimated_total_size = None

            # Create monitoring task
            async def monitor_progress():
                nonlocal last_update

                while True:
                    await asyncio.sleep(0.5)  # Non-blocking sleep

                    # Check if process completed
                    if process.returncode is not None:
                        break

                    # Read progress file
                    if os.path.exists(progress_file):
                        try:
                            with open(progress_file, 'r') as f:
                                lines = f.readlines()
                                progress_data = {}
                                for line in lines:
                                    if '=' in line:
                                        key, value = line.strip().split('=', 1)
                                        progress_data[key] = value

                                # Extract time progress
                                if 'out_time_ms' in progress_data and os.path.exists(audio_file):
                                    try:
                                        current_seconds = int(progress_data['out_time_ms']) / 1000000  # Convert to seconds
                                        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
                                        elapsed = time.time() - start_time

                                        # Calculate speed
                                        speed_mbps = file_size_mb / elapsed if elapsed > 0 else 0

                                        # For unknown duration (duration=0), estimate from file size growth
                                        if duration == 0:
                                            # Estimate total duration based on current time vs file size
                                            # Show indeterminate progress with file size and speed only
                                            percent = min((file_size_mb / 200) * 100, 95) if file_size_mb < 200 else 95  # Cap at 95%
                                            eta_seconds = 0  # Unknown
                                            target_duration = 0  # Unknown
                                        else:
                                            # Known duration - calculate accurate progress
                                            percent = min((current_seconds / duration) * 100, 99)
                                            remaining_seconds = duration - current_seconds
                                            eta_seconds = int((remaining_seconds / current_seconds * elapsed)) if current_seconds > 0 else 0
                                            target_duration = duration

                                        # Send progress update (throttle to every 1 second)
                                        if time.time() - last_update >= 1.0:
                                            try:
                                                await websocket.send_json({
                                                    "type": "download_progress",
                                                    "percent": round(percent, 1),
                                                    "downloaded_mb": round(file_size_mb, 2),
                                                    "speed_mbps": round(speed_mbps, 2),
                                                    "eta_seconds": eta_seconds,
                                                    "current_time": round(current_seconds, 1),
                                                    "target_duration": target_duration
                                                })
                                                last_update = time.time()
                                            except Exception as e:
                                                logger.debug(f"Progress update failed: {e}")
                                    except (ValueError, ZeroDivisionError) as e:
                                        logger.debug(f"Progress parsing error: {e}")
                        except Exception as e:
                            logger.debug(f"Progress file read error: {e}")

            # Start monitoring task
            monitor_task = asyncio.create_task(monitor_progress())

            # Wait for process to complete
            stdout_output, stderr_output = await process.communicate()

            # Cancel monitoring task
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            # Send 100% complete
            if os.path.exists(audio_file):
                file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
                await websocket.send_json({
                    "type": "download_progress",
                    "percent": 100,
                    "downloaded_mb": round(file_size_mb, 2),
                    "speed_mbps": 0,
                    "eta_seconds": 0,
                    "current_time": 0,
                    "target_duration": 0
                })
        else:
            # No websocket, just wait for completion
            stdout_output, stderr_output = await process.communicate()

        result = process

        # Clean up progress file
        try:
            if os.path.exists(progress_file):
                os.unlink(progress_file)
        except:
            pass

        if result.returncode != 0:
            # Extract the actual error from stderr (now properly captured via communicate())
            stderr_lines = stderr_output.split('\n') if stderr_output else []
            error_line = None
            for line in stderr_lines:
                if 'error' in line.lower() or 'http error' in line.lower():
                    error_line = line.strip()
                    break

            if error_line:
                logger.error(f"❌ ffmpeg download failed: {error_line}")
            else:
                logger.error(f"❌ ffmpeg download failed with return code {result.returncode}")

            # Check for common error patterns
            stderr_full = stderr_output.lower() if stderr_output else ""
            if '410' in stderr_full or 'gone' in stderr_full:
                logger.error("💡 URL has expired. Please get a fresh URL from the source.")
            elif '403' in stderr_full or 'forbidden' in stderr_full:
                logger.error("💡 Access denied. The URL may require authentication or be geo-restricted.")
            elif '404' in stderr_full or 'not found' in stderr_full:
                logger.error("💡 URL not found. Please verify the URL is correct.")
            elif 'unsupported' in stderr_full or 'invalid data' in stderr_full:
                logger.error("💡 Audio format not supported or file is corrupted.")

            # Fallback: Try with simpler settings if loudnorm fails
            logger.info("🔄 Trying fallback method without loudnorm...")
            fallback_cmd = [
                'ffmpeg',
                '-i', url,
            ]

            # Only add duration limit if specified (0 = complete file)
            if duration > 0:
                fallback_cmd.extend(['-t', str(duration)])

            fallback_cmd.extend([
                '-vn',
                '-acodec', 'pcm_s16le' if format == 'wav' else 'aac',
                '-ar', '16000',  # 16kHz for better compatibility
                '-ac', '1',  # Mono for better compatibility
                '-y',
                audio_file
            ])

            result = subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                logger.error(f"❌ ffmpeg fallback also failed: {result.stderr[:500]}")

                # Provide user-friendly error summary
                if '410' in result.stderr or 'Gone' in result.stderr:
                    logger.error("🔴 DOWNLOAD FAILED - URL has expired. Get a fresh URL from the source.")
                elif '403' in result.stderr or 'Forbidden' in result.stderr:
                    logger.error("🔴 DOWNLOAD FAILED - Access denied. URL may require authentication.")
                elif '404' in result.stderr:
                    logger.error("🔴 DOWNLOAD FAILED - URL not found. Verify the URL is correct.")
                else:
                    logger.error("🔴 DOWNLOAD FAILED - Unable to download audio from URL.")

                shutil.rmtree(temp_dir, ignore_errors=True)
                return None

        if os.path.exists(audio_file):
            file_size = os.path.getsize(audio_file)
            logger.info(f"Successfully downloaded audio: {audio_file} ({file_size / 1024:.1f} KB)")
            # Cache the download if duration is 0 (complete file) and caching is enabled
            if use_cache and duration == 0:
                audio_file = save_download_to_cache(url, audio_file)
            return audio_file
        else:
            logger.error(f"Audio file not created: {audio_file}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None

    except subprocess.TimeoutExpired:
        logger.error("ffmpeg download timeout after 5 minutes")
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        return None
    except Exception as e:
        logger.error(f"ffmpeg exception: {e}")
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        return None


def download_audio_with_ytdlp(url: str, language: Optional[str] = None, format: str = 'wav', use_cache: bool = True) -> Optional[str]:
    """
    Download and normalize audio from URL using yt-dlp with caching support
    Returns path to audio file or None on failure

    Args:
        url: URL to download from
        language: Optional language hint
        format: Output format ('wav' for Whisper, 'm4a' for Deepgram/general use)
        use_cache: Whether to check/use cached downloads
    """
    
    # Check cache first if enabled
    if use_cache:
        cached_file = get_cached_download(url)
        if cached_file:
            logger.info(f"Using cached download from yt-dlp: {cached_file}")
            return cached_file
    try:
        # Create temporary directory and file path for download
        # yt-dlp needs a template path without extension (it adds the extension)
        temp_dir = tempfile.mkdtemp()
        base_filename = os.path.join(temp_dir, 'audio')

        # Base yt-dlp command with critical HLS handling flags
        cmd = [
            'yt-dlp',
            '-x',  # Extract audio
            '--audio-format', format,
            '--audio-quality', '0',  # Best quality
            '--downloader', 'ffmpeg',  # Explicitly use ffmpeg downloader
            # Removed --hls-use-mpegts (causes malformed MPEG-TS format that FFmpeg can't read)
            '--no-playlist',
            '--no-warnings',
            '-o', base_filename + '.%(ext)s',  # Let yt-dlp add the extension
        ]

        # Add format-specific postprocessor args for wav (Whisper compatibility)
        if format == 'wav':
            cmd.extend(['--postprocessor-args', 'ffmpeg:-ar 16000 -ac 1 -c:a pcm_s16le'])

        cmd.append(url)

        # Expected output path after yt-dlp processes
        output_path = f"{base_filename}.{format}"

        logger.info(f"Downloading audio from URL with yt-dlp (format={format}): {url}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0 and os.path.exists(output_path):
            logger.info(f"Successfully downloaded audio: {output_path}")
            # Cache the download if caching is enabled
            if use_cache:
                output_path = save_download_to_cache(url, output_path)
            return output_path
        else:
            logger.error(f"yt-dlp failed: {result.stderr}")
            # Cleanup temp directory
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
            return None

    except subprocess.TimeoutExpired:
        logger.error("yt-dlp download timeout after 5 minutes")
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        return None
    except Exception as e:
        logger.error(f"yt-dlp exception: {e}")
        try:
            import shutil
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        return None


async def download_with_fallback(url: str, language: Optional[str] = None, format: str = 'wav', websocket: WebSocket = None, use_cache: bool = True) -> Optional[str]:
    """
    Smart download with automatic fallback chain:
    1. Try yt-dlp with cookies/user-agent
    2. Try ffmpeg direct download
    3. Return None if all methods fail
    
    Args:
        url: URL to download from
        language: Optional language hint
        format: Output format ('wav' or 'm4a')
        websocket: WebSocket for progress updates
        use_cache: Whether to check/use cached downloads
        
    Returns:
        Path to downloaded audio file or None
    """
    
    # Try yt-dlp first (better for YouTube and complex sites)
    if websocket:
        await websocket.send_json({
            "type": "status",
            "message": "📥 Attempting download with yt-dlp..."
        })
    
    audio_file = await download_audio_with_ytdlp_async(url, language, format, websocket, use_cache)
    
    if audio_file:
        return audio_file
    
    # yt-dlp failed, try ffmpeg fallback
    logger.warning(f"yt-dlp failed for {url}, trying ffmpeg fallback...")
    
    if websocket:
        await websocket.send_json({
            "type": "status",
            "message": "⚠️ yt-dlp failed, trying alternative download method (ffmpeg)..."
        })
    
    try:
        # Try ffmpeg as fallback (works for direct streams)
        audio_file = await download_audio_with_ffmpeg(url, format=format, duration=0, websocket=websocket, use_cache=use_cache)
        
        if audio_file:
            logger.info(f"Successfully downloaded with ffmpeg fallback: {audio_file}")
            if websocket:
                await websocket.send_json({
                    "type": "status",
                    "message": "✓ Download successful using alternative method"
                })
            return audio_file
        else:
            logger.error(f"Both yt-dlp and ffmpeg failed for {url}")
            # Both methods failed - provide helpful error message
            if websocket and 'youtube.com' in url or 'youtu.be' in url:
                await websocket.send_json({
                    "type": "status",
                    "message": "⚠️ YouTube download failed. This video may require authentication or have restrictions. Try a different video or update yt-dlp."
                })
            return None
            
    except Exception as e:
        logger.error(f"ffmpeg fallback also failed: {e}")
        return None


async def download_audio_with_ytdlp_async(url: str, language: Optional[str] = None, format: str = 'wav', websocket: WebSocket = None, use_cache: bool = True) -> Optional[str]:
    """
    Async version of download_audio_with_ytdlp with WebSocket progress updates
    Returns path to audio file or None on failure
    
    Args:
        url: URL to download from
        language: Optional language hint
        format: Output format ('wav' for Whisper, 'm4a' for general use)
        websocket: WebSocket for sending progress updates
        use_cache: Whether to check/use cached downloads
    """
    
    # Check cache first if enabled
    if use_cache:
        cached_file = get_cached_download(url)
        if cached_file:
            logger.info(f"Using cached download from yt-dlp: {cached_file}")
            if websocket:
                await websocket.send_json({
                    "type": "status",
                    "message": "✓ Using cached audio file"
                })
            return cached_file
    
    try:
        # Create temporary directory and file path for download
        temp_dir = tempfile.mkdtemp()
        base_filename = os.path.join(temp_dir, 'audio')
        
        # Simplified yt-dlp command matching manual usage (more reliable)
        cmd = [
            'yt-dlp',
            '-x',  # Extract audio
            '--audio-format', format,
            '--audio-quality', '0',  # Best quality
            '--no-playlist',
            '--newline',  # Output progress on separate lines for parsing
            '-o', base_filename + '.%(ext)s',
        ]
        
        # Try to use cookies if available (helps with bot detection)
        cookies_file = os.path.expanduser('~/.config/yt-dlp/cookies.txt')
        if os.path.exists(cookies_file):
            cmd.extend(['--cookies', cookies_file])
            logger.info("Using cookies file for YouTube authentication")
        
        cmd.append(url)
        
        # Expected output path
        output_path = f"{base_filename}.{format}"
        
        logger.info(f"Downloading audio from URL with yt-dlp (async, format={format}): {url}")
        
        if websocket:
            await websocket.send_json({
                "type": "status",
                "message": "📥 Starting download with yt-dlp..."
            })
        
        # Run yt-dlp asynchronously
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        
        # Read output line by line and send progress updates
        last_update_time = time.time()
        has_error = False
        error_messages = []
        
        while True:
            line = await process.stdout.readline()
            if not line:
                break
                
            line_str = line.decode('utf-8').strip()
            
            # Detect critical errors (but don't alert UI yet - let fallback handle it)
            if 'ERROR:' in line_str:
                has_error = True
                error_messages.append(line_str)
                logger.error(f"yt-dlp error: {line_str}")
            
            # Parse progress from yt-dlp output
            # Format: [download]  45.3% of 12.34MiB at 1.23MiB/s ETA 00:05
            if '[download]' in line_str and '%' in line_str:
                try:
                    # Extract percentage
                    if 'of' in line_str:
                        parts = line_str.split()
                        percent_str = [p for p in parts if '%' in p][0].replace('%', '')
                        percent = float(percent_str)
                        
                        # Extract size info if available
                        size_info = ""
                        total_size = ""
                        if 'of' in line_str and ('MiB' in line_str or 'GiB' in line_str or 'KiB' in line_str):
                            of_idx = parts.index('of')
                            if of_idx + 1 < len(parts):
                                total_size = parts[of_idx + 1]
                                size_info = f" of {total_size}"
                        
                        # Extract speed if available (e.g., "at 1.23MiB/s")
                        speed_info = ""
                        speed_mbps = "..."
                        if 'at' in line_str and '/s' in line_str:
                            try:
                                at_idx = parts.index('at')
                                if at_idx + 1 < len(parts):
                                    speed_str = parts[at_idx + 1]
                                    speed_info = f" at {speed_str}"
                                    speed_mbps = speed_str
                            except (ValueError, IndexError):
                                pass
                        
                        # Extract ETA if available (e.g., "ETA 00:05" or "ETA Unknown")
                        eta_info = ""
                        eta_seconds = 0
                        if 'ETA' in line_str:
                            try:
                                eta_idx = parts.index('ETA')
                                if eta_idx + 1 < len(parts):
                                    eta_str = parts[eta_idx + 1]
                                    eta_info = f" (ETA: {eta_str})"
                                    
                                    # Parse ETA to seconds (format: MM:SS or HH:MM:SS)
                                    if eta_str not in ['Unknown', '?', '--:--']:
                                        time_parts = eta_str.split(':')
                                        if len(time_parts) == 2:  # MM:SS
                                            eta_seconds = int(time_parts[0]) * 60 + int(time_parts[1])
                                        elif len(time_parts) == 3:  # HH:MM:SS
                                            eta_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
                            except (ValueError, IndexError):
                                pass
                        
                        # Send update every 2 seconds to avoid overwhelming
                        current_time = time.time()
                        if websocket and (current_time - last_update_time) >= 1:  # Increased frequency to 1s
                            await websocket.send_json({
                                "type": "download_progress",
                                "percent": int(percent),
                                "downloaded_mb": f"{percent:.1f}%{size_info}",
                                "speed_mbps": speed_mbps,
                                "eta_seconds": eta_seconds,
                                "message": f"Downloading: {percent:.1f}%{size_info}{speed_info}{eta_info}"
                            })
                            last_update_time = current_time
                            
                except Exception as e:
                    logger.debug(f"Failed to parse progress: {e}")
            
            # Log other important messages
            elif any(keyword in line_str.lower() for keyword in ['error', 'warning', 'failed']):
                logger.warning(f"yt-dlp: {line_str}")
        
        # Wait for process to complete with timeout
        try:
            await asyncio.wait_for(process.wait(), timeout=60)  # 60 second timeout after output stops
        except asyncio.TimeoutError:
            logger.warning("yt-dlp process timeout - killing process")
            try:
                process.kill()
                await process.wait()
            except:
                pass
            has_error = True
            error_messages.append("Process timeout")
        
        # Check if we collected any errors during processing
        if has_error and error_messages:
            error_summary = "\n".join(error_messages[:3])  # Show first 3 errors
            logger.error(f"yt-dlp failed with errors:\n{error_summary}")
            logger.info("Triggering fallback to ffmpeg due to yt-dlp errors")
            # Don't send error to UI yet - let fallback mechanism try ffmpeg first
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
            return None
        
        # Check return code and file existence
        logger.info(f"yt-dlp completed with return code: {process.returncode}, file exists: {os.path.exists(output_path)}")
        
        if process.returncode == 0 and os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"Successfully downloaded audio: {output_path} ({file_size_mb:.1f} MB)")
            
            if websocket:
                await websocket.send_json({
                    "type": "status",
                    "message": f"✓ Download complete ({file_size_mb:.1f} MB)"
                })
            
            # Cache the download if caching is enabled
            if use_cache:
                output_path = save_download_to_cache(url, output_path)
            
            return output_path
        else:
            # Download failed - log but don't send error to UI (let fallback try)
            error_msg = f"yt-dlp failed with return code: {process.returncode}, file exists: {os.path.exists(output_path)}"
            logger.warning(f"{error_msg} - will trigger fallback to ffmpeg")
            
            # Cleanup temp directory
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
            return None
            
    except asyncio.TimeoutError:
        logger.warning("yt-dlp download timeout - will try fallback")
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        return None
        
    except Exception as e:
        logger.warning(f"yt-dlp async exception: {e} - will try fallback")
        try:
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
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


def calculate_progress_metrics(audio_duration: float, elapsed_time: float, processed_chunks: int = 0, total_chunks: int = 0) -> dict:
    """
    Calculate detailed progress metrics for transcription.
    
    Args:
        audio_duration: Total audio duration in seconds
        elapsed_time: Time elapsed since start in seconds
        processed_chunks: Number of chunks processed (for chunked processing)
        total_chunks: Total number of chunks (for chunked processing)
    
    Returns:
        Dictionary with percentage, ETA, speed, and other metrics
    """
    if total_chunks > 0 and processed_chunks > 0:
        # Chunk-based progress
        percentage = (processed_chunks / total_chunks) * 100
        avg_chunk_time = elapsed_time / processed_chunks
        remaining_chunks = total_chunks - processed_chunks
        eta = remaining_chunks * avg_chunk_time
        
        # Calculate processing speed
        processed_duration = (processed_chunks / total_chunks) * audio_duration
        speed = processed_duration / elapsed_time if elapsed_time > 0 else 0
    else:
        # Time-based estimation (fallback)
        # Use conservative estimate of processing speed
        if elapsed_time < 5:
            # Not enough data, use very conservative estimate
            estimated_speed = 0.3  # 0.3x realtime
        else:
            # Adaptive estimation based on model typical performance
            estimated_speed = 0.5  # 0.5x realtime as default
        
        processed_duration = min(elapsed_time * estimated_speed, audio_duration * 0.95)
        percentage = min((processed_duration / audio_duration) * 100, 95)
        remaining_duration = audio_duration - processed_duration
        eta = remaining_duration / estimated_speed if estimated_speed > 0 else 0
        speed = estimated_speed
    
    return {
        "percentage": round(min(percentage, 99.9), 1),
        "eta_seconds": max(int(eta), 0),
        "speed": round(speed, 2),
        "processed_duration": round(processed_duration, 1) if 'processed_duration' in locals() else 0,
        "audio_duration": round(audio_duration, 1)
    }


def split_audio_for_incremental(audio_path: str, chunk_seconds: int = 60, overlap_seconds: int = 5) -> Tuple[str, List[str]]:
    """
    Split audio file into chunks for incremental transcription.
    Returns tuple of (temp_dir, list of chunk file paths)
    """
    duration = get_audio_duration_seconds(audio_path)
    if not duration:
        raise RuntimeError("Unable to determine audio duration for chunking")
    
    temp_dir = tempfile.mkdtemp(prefix='incremental_chunks_')
    chunks = []
    
    # Calculate chunk positions
    step = max(1, chunk_seconds - overlap_seconds)
    position = 0
    index = 0
    
    while position < duration:
        chunk_path = os.path.join(temp_dir, f"chunk_{index:04d}.wav")
        actual_chunk_duration = min(chunk_seconds, duration - position)
        
        # Extract chunk using ffmpeg
        cmd = [
            'ffmpeg',
            '-ss', str(position),
            '-i', audio_path,
            '-t', str(actual_chunk_duration),
            '-ar', '16000',
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            '-loglevel', 'error',
            '-y',
            chunk_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(chunk_path):
            chunks.append(chunk_path)
            index += 1
            position += step
        else:
            logger.error(f"Failed to create chunk at position {position}s: {result.stderr}")
            break
    
    return temp_dir, chunks


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


async def transcribe_with_incremental_output(
    model, model_config: dict, audio_file: str, language: Optional[str], 
    websocket: WebSocket, model_name: str = None, chunk_seconds: int = 60
) -> Tuple[str, str]:
    """
    Transcribe audio with incremental output and detailed progress.
    
    Args:
        model: The loaded transcription model
        model_config: Model configuration dictionary
        audio_file: Path to audio file
        language: Language code or None for auto-detect
        websocket: WebSocket connection for progress updates
        model_name: Name of the model (needed for language detection logic)
        chunk_seconds: Size of audio chunks in seconds
        
    Returns:
        (full_transcript, detected_language)
    """
    import time
    start_time = time.time()
    
    # Get audio duration for progress calculation
    audio_duration = get_audio_duration_seconds(audio_file)
    if not audio_duration:
        logger.warning("Could not determine audio duration, using fallback progress")
        audio_duration = 0
    
    # Send initial status with audio duration
    await websocket.send_json({
        "type": "transcription_status",
        "message": f"Starting transcription of {audio_duration:.1f}s audio...",
        "audio_duration": audio_duration,
        "elapsed_seconds": 0
    })
    
    # For short audio (< 2 minutes), process as single chunk
    if audio_duration and audio_duration < 120:
        logger.info(f"Short audio ({audio_duration}s), processing as single chunk")
        
        # Transcribe entire file
        if model_config["type"] == "faster_whisper":
            # Use faster_whisper for Ivrit CT2 models
            fw_model = model["model"] if isinstance(model, dict) else model
            
            def run_fw_transcription():
                try:
                    # For Ivrit models, use Hebrew as default only if it's an Ivrit model
                    # For general models like whisper-v3-turbo, let it auto-detect
                    default_lang = None
                    if model_name and "ivrit" in model_name.lower():
                        # Ivrit models are Hebrew-optimized, default to Hebrew if no language specified
                        default_lang = "he"
                    
                    segments, info = fw_model.transcribe(
                        audio_file,
                        language=language or default_lang,  # None means auto-detect
                        beam_size=int(os.getenv("IVRIT_BEAM_SIZE", "5")),
                        best_of=5,
                        patience=1,
                        length_penalty=1,
                        temperature=0,
                        compression_ratio_threshold=2.4,
                        log_prob_threshold=-1.0,
                        no_speech_threshold=0.6,
                        word_timestamps=False
                    )
                    # Collect segments into text
                    text_parts = []
                    for segment in segments:
                        text_parts.append(segment.text)
                    return {
                        'text': ' '.join(text_parts),
                        'language': info.language if info else (language or 'he')
                    }
                except Exception as e:
                    logger.error(f"faster_whisper transcription failed: {e}")
                    raise
            
            # Run with progress monitoring
            loop = asyncio.get_event_loop()
            task = loop.run_in_executor(None, run_fw_transcription)
            
            while not task.done():
                await asyncio.sleep(2)
                elapsed = time.time() - start_time
                metrics = calculate_progress_metrics(audio_duration, elapsed)
                
                await websocket.send_json({
                    "type": "transcription_progress",
                    "audio_duration": audio_duration,
                    "percentage": metrics["percentage"],
                    "eta_seconds": metrics["eta_seconds"],
                    "speed": metrics["speed"],
                    "elapsed_seconds": int(elapsed)
                })
            
            result = await task
            transcript = result.get('text', '').strip()
            detected_language = result.get('language', language or 'he')
            
        elif model_config["type"] == "openai":
            def run_transcription():
                use_fp16 = torch.cuda.is_available()
                try:
                    return model.transcribe(audio_file, language=language, fp16=use_fp16, verbose=False)
                except Exception as e:
                    logger.warning(f"GPU/FP16 failed: {e}. Retrying with fp16=False.")
                    return model.transcribe(audio_file, language=language, fp16=False, verbose=False)
            
            # Run with progress monitoring
            loop = asyncio.get_event_loop()
            task = loop.run_in_executor(None, run_transcription)
            
            while not task.done():
                await asyncio.sleep(2)
                elapsed = time.time() - start_time
                metrics = calculate_progress_metrics(audio_duration, elapsed)
                
                await websocket.send_json({
                    "type": "transcription_progress",
                    "audio_duration": audio_duration,
                    "percentage": metrics["percentage"],
                    "eta_seconds": metrics["eta_seconds"],
                    "speed": metrics["speed"],
                    "elapsed_seconds": int(elapsed)
                })
            
            result = await task
            transcript = result.get('text', '').strip()
            detected_language = result.get('language', language or 'unknown')
            
            # Send final transcript
            if transcript:
                await websocket.send_json({
                    "type": "transcription_chunk",
                    "text": transcript,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "is_final": True
                })
            
            return transcript, detected_language
        

            
            if transcript:
                await websocket.send_json({
                    "type": "transcription_chunk",
                    "text": transcript,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "is_final": True
                })
            
            return transcript, detected_language
    
    # For longer audio, use chunked processing for incremental output
    logger.info(f"Long audio ({audio_duration}s), using chunked processing")
    
    try:
        # Split audio into chunks
        temp_dir, chunk_files = split_audio_for_incremental(audio_file, chunk_seconds, overlap_seconds=5)
        total_chunks = len(chunk_files)
        
        logger.info(f"Split audio into {total_chunks} chunks")
        
        transcripts = []
        detected_language = language or 'unknown'
        
        for i, chunk_file in enumerate(chunk_files):
            chunk_start = time.time()
            
            # Transcribe chunk
            if model_config["type"] == "faster_whisper":
                # Use faster_whisper for chunks
                fw_model = model["model"] if isinstance(model, dict) else model
                try:
                    # For Ivrit models, use Hebrew as default only if it's an Ivrit model
                    default_lang = None
                    if model_name and "ivrit" in model_name.lower():
                        # Ivrit models are Hebrew-optimized, default to Hebrew if no language specified
                        default_lang = "he"
                    
                    segments, info = fw_model.transcribe(
                        chunk_file,
                        language=language or default_lang,  # None means auto-detect
                        beam_size=5,
                        best_of=5,
                        patience=1,
                        temperature=0,
                        compression_ratio_threshold=2.4,
                        no_speech_threshold=0.6,
                        word_timestamps=False
                    )
                    text_parts = []
                    for segment in segments:
                        text_parts.append(segment.text)
                    chunk_text = ' '.join(text_parts).strip()
                    if i == 0 and info and info.language:
                        detected_language = info.language
                except Exception as e:
                    logger.error(f"faster_whisper chunk transcription failed: {e}")
                    chunk_text = ""
            
            elif model_config["type"] == "openai":
                use_fp16 = torch.cuda.is_available()
                try:
                    result = model.transcribe(chunk_file, language=language, fp16=use_fp16, verbose=False)
                except Exception:
                    result = model.transcribe(chunk_file, language=language, fp16=False, verbose=False)
                
                chunk_text = result.get('text', '').strip()
                if i == 0 and result.get('language'):
                    detected_language = result.get('language')
            

            
            # Send incremental result
            if chunk_text:
                await websocket.send_json({
                    "type": "transcription_chunk",
                    "text": chunk_text,
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                    "is_final": i == total_chunks - 1
                })
                transcripts.append(chunk_text)
            
            # Calculate progress metrics
            chunk_duration = time.time() - chunk_start
            elapsed = time.time() - start_time
            
            # Calculate ETA based on average chunk processing time
            avg_time_per_chunk = elapsed / (i + 1)
            remaining_chunks = total_chunks - (i + 1)
            eta_seconds = int(avg_time_per_chunk * remaining_chunks)
            percentage = int(((i + 1) / total_chunks) * 100)
            
            # Send detailed progress update
            await websocket.send_json({
                "type": "transcription_progress",
                "audio_duration": audio_duration,
                "percentage": percentage,
                "eta_seconds": eta_seconds,
                "speed": f"{avg_time_per_chunk:.1f}s/chunk",
                "elapsed_seconds": int(elapsed),
                "chunks_processed": i + 1,
                "total_chunks": total_chunks,
                "message": f"Transcribing: {percentage}% (chunk {i+1}/{total_chunks}, ETA: {eta_seconds}s)"
            })
            
            logger.info(f"Processed chunk {i+1}/{total_chunks} in {chunk_duration:.1f}s (avg: {avg_time_per_chunk:.1f}s/chunk, ETA: {eta_seconds}s)")
        
        # Cleanup chunk files
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        
        # Join transcripts with space
        full_transcript = ' '.join(transcripts)
        return full_transcript, detected_language
        
    except Exception as e:
        logger.error(f"Incremental transcription error: {e}")
        # Fallback to single-file transcription
        logger.info("Falling back to single-file transcription")
        
        if model_config["type"] == "openai":
            use_fp16 = torch.cuda.is_available()
            try:
                result = model.transcribe(audio_file, language=language, fp16=use_fp16, verbose=False)
            except Exception:
                result = model.transcribe(audio_file, language=language, fp16=False, verbose=False)
            
            transcript = result.get('text', '').strip()
            detected_language = result.get('language', language or 'unknown')
        else:
            transcript = ""
            detected_language = language or 'unknown'
        
        if transcript:
            await websocket.send_json({
                "type": "transcription_chunk",
                "text": transcript,
                "chunk_index": 0,
                "total_chunks": 1,
                "is_final": True
            })
        
        return transcript, detected_language


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
            if not OPENAI_WHISPER_AVAILABLE:
                raise ValueError("openai-whisper is not installed. Cannot transcribe with OpenAI Whisper models.")
            
            use_fp16 = torch.cuda.is_available()
            try:
                result = model.transcribe(chunk_path, language=language, fp16=use_fp16, verbose=False)
            except Exception:
                result = model.transcribe(chunk_path, language=language, fp16=False, verbose=False)
            transcription_text = result.get('text', '').strip()
            detected_language = result.get('language', detected_language)

        elif model_config["type"] == "faster_whisper":
            if not FASTER_WHISPER_AVAILABLE:
                raise ValueError("faster_whisper is not installed. Cannot transcribe with faster_whisper models.")
            
            # Handle wrapped faster_whisper model structure
            if isinstance(model, dict) and model.get("type") == "faster_whisper":
                actual_model = model["model"]
            else:
                actual_model = model
            
            # Use faster_whisper model for transcription
            segments, info = actual_model.transcribe(chunk_path, language=language)
            transcription_text = ' '.join([segment.text for segment in segments]).strip()
            detected_language = info.language if hasattr(info, 'language') else (language or 'unknown')
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
                # Use backpressure: wait longer for queue space instead of dropping chunks
                try:
                    # First try non-blocking
                    self.audio_queue.put_nowait(full_chunk)
                except queue.Full:
                    # Queue is full - apply backpressure by waiting up to 5 seconds
                    # This slows down audio reading to match transcription speed
                    try:
                        logger.debug(f"Audio queue full ({self.audio_queue.qsize()}/{AUDIO_QUEUE_SIZE}), applying backpressure...")
                        self.audio_queue.put(full_chunk, timeout=5.0)
                        logger.debug("Chunk enqueued after backpressure wait")
                    except queue.Full:
                        # Only drop if queue is still full after 5 seconds
                        # This means transcription is extremely slow
                        try:
                            self.audio_queue.get_nowait()
                            self.audio_queue.put_nowait(full_chunk)
                            logger.warning(f"⚠️ Audio queue saturated ({AUDIO_QUEUE_SIZE} chunks). Evicted oldest chunk. Consider using a faster model.")
                        except Exception:
                            logger.error("❌ Audio queue critically full; dropping chunk. Transcription cannot keep up.")

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
                    elif model_config["type"] == "faster_whisper":
                        # Use faster_whisper for transcription
                        # Handle wrapped model structure
                        if isinstance(model, dict) and model.get("type") == "faster_whisper":
                            fw_model = model["model"]
                        else:
                            fw_model = model
                        
                        try:
                            segments, info = fw_model.transcribe(
                                temp_path,
                                language=processor.language or "he",
                                beam_size=int(os.getenv("IVRIT_BEAM_SIZE", "5")),
                                best_of=5,
                                patience=1,
                                temperature=0,
                                compression_ratio_threshold=2.4,
                                no_speech_threshold=0.6,
                                word_timestamps=False
                            )
                            text_parts = []
                            for segment in segments:
                                text_parts.append(segment.text)
                            transcription_text = ' '.join(text_parts).strip()
                            detected_language = info.language if hasattr(info, 'language') else (processor.language or 'he')
                        except Exception as e:
                            logger.error(f"faster_whisper transcription failed: {e}")
                            transcription_text = ""
                            detected_language = processor.language or 'he'

                    
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


async def transcribe_vod_with_deepgram(websocket: WebSocket, url: str, language: Optional[str] = None):
    """Transcribe VOD (Video On Demand) content using Deepgram's pre-recorded API"""
    if not DEEPGRAM_AVAILABLE:
        await websocket.send_json({"error": "Deepgram SDK not available"})
        return

    # Initialize Deepgram client
    client = DeepgramClient(api_key=DEEPGRAM_API_KEY) if DEEPGRAM_API_KEY else DeepgramClient()
    model = os.getenv("DEEPGRAM_MODEL", "nova-3")
    lang = language or os.getenv("DEEPGRAM_LANGUAGE", "en-US")

    try:
        # OPTIMIZATION: Try sending URL directly to Deepgram first (much faster!)
        # Deepgram can fetch and transcribe URLs directly without us downloading
        await websocket.send_json({"type": "status", "message": "📡 Step 1/3: Attempting direct URL transcription via Deepgram API..."})

        try:
            # Determine if we should use language detection
            use_detect_language = not language or language == "auto"

            # Use Deepgram's URL-based transcription (recommended approach)
            if use_detect_language:
                logger.info("Using automatic language detection (detect_language=True)")
                response = client.listen.v1.media.transcribe_url(
                    url=url,
                    model=model,
                    detect_language=True,
                    punctuate=True,
                    smart_format=True
                )
            else:
                response = client.listen.v1.media.transcribe_url(
                    url=url,
                    model=model,
                    language=language,
                    punctuate=True,
                    smart_format=True
                )

            # Extract transcript and metadata from response
            transcript = response.results.channels[0].alternatives[0].transcript.strip()
            # Safely extract request_id from metadata (it's an object, not a dict)
            request_id = None
            try:
                if hasattr(response, 'metadata') and response.metadata:
                    request_id = getattr(response.metadata, 'request_id', None)
            except Exception:
                pass

            if transcript:
                response_data = {
                    "type": "transcription",
                    "text": transcript,
                    "language": lang
                }
                # Include request_id for tracking/debugging if available
                if request_id:
                    response_data["request_id"] = request_id
                    logger.info(f"✓ Sent complete Deepgram transcription ({len(transcript)} chars) [request_id: {request_id}]")
                else:
                    logger.info(f"✓ Sent complete Deepgram transcription ({len(transcript)} chars)")

                await websocket.send_json(response_data)
                await websocket.send_json({"type": "complete", "message": "Transcription complete"})
                return
            else:
                logger.warning("No transcript received from Deepgram URL method")

        except Exception as url_error:
            # URL method failed (maybe URL not directly accessible by Deepgram)
            logger.warning(f"Deepgram URL method failed: {url_error}, falling back to download method")
            await websocket.send_json({"type": "status", "message": "⚠️ Direct URL method failed. Switching to download method..."})

        # FALLBACK: Download with ffmpeg (proven working method) and upload to Deepgram
        await websocket.send_json({"type": "status", "message": "⬇️ Step 2/3: Downloading audio from URL using ffmpeg..."})

        # Use ffmpeg download method - download complete file for better transcription and caching
        # Using duration=0 for complete file which enables caching
        audio_file = await download_audio_with_ffmpeg(url, format='wav', duration=0, websocket=websocket)
        if not audio_file:
            # Provide detailed error message based on logs
            error_detail = "Failed to download audio from URL.\n\n"
            error_detail += "Common causes:\n"
            error_detail += "  • URL has expired (HTTP 410) - Get a fresh URL\n"
            error_detail += "  • Access denied (HTTP 403) - URL may require authentication\n"
            error_detail += "  • URL not found (HTTP 404) - Verify the URL is correct\n"
            error_detail += "  • Network/connection issues\n\n"
            error_detail += "Check server logs for specific error details."

            await websocket.send_json({"error": error_detail})
            return

        # Notify download success
        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        await websocket.send_json({"type": "status", "message": f"✅ Audio downloaded successfully ({file_size_mb:.1f} MB)"})

        await websocket.send_json({"type": "status", "message": "🚀 Step 3/3: Uploading to Deepgram and transcribing..."})

        try:
            # Check file size before uploading
            file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
            if file_size_mb > 100:
                logger.warning(f"Large audio file detected ({file_size_mb:.1f}MB). This may take longer to process.")
                await websocket.send_json({"type": "status", "message": f"Processing large file ({file_size_mb:.1f}MB), please wait..."})

            with open(audio_file, "rb") as f:
                audio_data = f.read()

            # Determine if we should use language detection
            use_detect_language = not language or language == "auto"

            if use_detect_language:
                logger.info("Using automatic language detection (detect_language=True)")
                response = client.listen.v1.media.transcribe_file(
                    request=audio_data,
                    model=model,
                    detect_language=True,
                    punctuate=True,
                    smart_format=True
                )
            else:
                response = client.listen.v1.media.transcribe_file(
                    request=audio_data,
                    model=model,
                    language=language,
                    punctuate=True,
                    smart_format=True
                )

            transcript = response.results.channels[0].alternatives[0].transcript.strip()

            # Extract additional metadata for better user feedback
            request_id = None
            detected_language = None
            confidence = None

            try:
                if hasattr(response, 'metadata') and response.metadata:
                    request_id = getattr(response.metadata, 'request_id', None)
            except Exception:
                pass

            try:
                # Get detected language and confidence
                if response.results.channels[0].alternatives:
                    alternative = response.results.channels[0].alternatives[0]
                    if hasattr(alternative, 'confidence'):
                        confidence = alternative.confidence
                    if hasattr(alternative, 'detected_language'):
                        detected_language = alternative.detected_language
            except Exception:
                pass

            if transcript:
                # Log detailed success info
                log_parts = [f"✓ Transcription complete ({len(transcript)} chars)"]
                if detected_language:
                    log_parts.append(f"Language: {detected_language}")
                if confidence:
                    log_parts.append(f"Confidence: {confidence:.2f}")
                if request_id:
                    log_parts.append(f"Request ID: {request_id}")
                logger.info(" | ".join(log_parts))

                # Send transcript in chunks to avoid WebSocket message size limits
                # Split into sentences or chunks of ~500 characters
                chunk_size = 500
                transcript_chunks = []

                # Try to split by sentences first (. ! ?)
                sentences = []
                current_sentence = ""
                for char in transcript:
                    current_sentence += char
                    if char in '.!?' and len(current_sentence) > 50:
                        sentences.append(current_sentence.strip())
                        current_sentence = ""
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())

                # Group sentences into chunks
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                        current_chunk += (" " if current_chunk else "") + sentence
                    else:
                        if current_chunk:
                            transcript_chunks.append(current_chunk)
                        current_chunk = sentence
                if current_chunk:
                    transcript_chunks.append(current_chunk)

                # If no chunks created (no sentence breaks), split by character count
                if not transcript_chunks:
                    for i in range(0, len(transcript), chunk_size):
                        transcript_chunks.append(transcript[i:i + chunk_size])

                # Send each chunk
                logger.info(f"Sending transcript in {len(transcript_chunks)} chunks")
                await websocket.send_json({"type": "status", "message": f"📝 Transcription complete! Sending {len(transcript)} characters..."})

                for i, chunk in enumerate(transcript_chunks):
                    response_data = {
                        "type": "transcription",
                        "text": chunk,
                        "language": detected_language or lang
                    }
                    if request_id:
                        response_data["request_id"] = request_id
                    await websocket.send_json(response_data)
                    # Small delay between chunks to ensure proper delivery
                    await asyncio.sleep(0.1)
            else:
                # Provide detailed feedback when no transcript is returned
                error_msg = "⚠️ No transcript received from Deepgram"

                # Provide possible causes
                possible_causes = []
                if file_size_mb < 0.1:
                    possible_causes.append("Audio file is very small (< 100KB) - may contain no speech")
                possible_causes.extend([
                    "No speech detected in the audio",
                    "Audio contains only music/silence",
                    "Audio quality too poor to transcribe",
                    "Language not well-supported by Deepgram"
                ])

                detailed_msg = f"{error_msg}\n\nPossible causes:\n" + "\n".join(f"  • {cause}" for cause in possible_causes)
                logger.warning(detailed_msg)

                await websocket.send_json({
                    "type": "status",
                    "message": "⚠️ No speech detected in audio. The file may contain only music, silence, or unclear speech."
                })

            await websocket.send_json({"type": "complete", "message": "Transcription complete"})

        except Exception as e:
            error_msg = str(e)
            # Handle specific payload size errors
            if "413" in error_msg or "payload too large" in error_msg.lower() or "entity too large" in error_msg.lower():
                logger.error(f"Deepgram payload too large error: {e}")
                await websocket.send_json({
                    "error": "Audio file too large for direct upload. Try using a URL instead or use a smaller file."
                })
            else:
                logger.error(f"Deepgram transcription error: {e}")
                await websocket.send_json({"error": f"Deepgram transcription failed: {str(e)}"})

        finally:
            # Cleanup downloaded file and its temp directory (only if not in cache)
            try:
                if audio_file and os.path.exists(audio_file):
                    # Only delete if not in download cache
                    if DOWNLOAD_CACHE_DIR not in Path(audio_file).parents:
                        os.unlink(audio_file)
                        temp_dir = os.path.dirname(audio_file)
                        if temp_dir and os.path.exists(temp_dir) and '/tmp' in temp_dir:
                            shutil.rmtree(temp_dir, ignore_errors=True)
                    else:
                        logger.debug(f"Keeping cached Deepgram download: {audio_file}")
            except Exception as e:
                logger.debug(f"Cleanup warning: {e}")

    except Exception as e:
        logger.error(f"VOD transcription error: {e}")
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({"error": str(e)})
        except:
            pass


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
        # Only return final transcripts, not interim results
        def extract_deepgram_transcript(message) -> str:
            try:
                if isinstance(message, dict):
                    # Check if this is a final transcript (not interim)
                    is_final = message.get('is_final', True)  # Default to True if not specified
                    if not is_final:
                        return ''  # Skip interim results

                    alts = message.get('channel', {}).get('alternatives', [])
                    if alts:
                        return alts[0].get('transcript', '')
                # v4 typed objects
                if hasattr(message, 'channel') and hasattr(message.channel, 'alternatives'):
                    # Check if this is a final transcript
                    is_final = getattr(message, 'is_final', True)
                    if not is_final:
                        return ''  # Skip interim results

                    alt0 = message.channel.alternatives[0] if message.channel.alternatives else None
                    if alt0 is not None:
                        return getattr(alt0, 'transcript', '') or ''
            except Exception:
                pass
            return ''


        # Initialize Deepgram client
        client = DeepgramClient(api_key=DEEPGRAM_API_KEY) if DEEPGRAM_API_KEY else DeepgramClient()
        # Apply user-requested params
        # TIME_LIMIT: Set to 0 or negative for unlimited streaming, or positive number for time limit in seconds
        TIME_LIMIT = int(os.getenv("DEEPGRAM_TIME_LIMIT", "3600"))
        TRANSCRIPT_ONLY = os.getenv("DEEPGRAM_TRANSCRIPT_ONLY", "true").lower() == "true"
        PARAMS = {
            "punctuate": True,
            "numerals": True,
            "model": os.getenv("DEEPGRAM_MODEL", "nova-3"),
            "language": os.getenv("DEEPGRAM_LANGUAGE", "en-US")
        }

        # Create a websocket client (SDK v4)
        connection = client.listen.websocket.v("1")

        # Capture the event loop reference BEFORE defining callbacks
        # The Deepgram SDK runs callbacks in a different thread, so we need to capture the loop here
        loop = asyncio.get_event_loop()

        # Event handlers
        # Listen for the connection to open
        connection.on(DG_EVENT_OPEN, lambda *_: logger.info("Deepgram connection opened"))

        # Listen for any transcripts received from Deepgram
        def on_message(*args, **kwargs):
            # Deepgram v4 emits handler(self, result=..., **kwargs)
            payload = kwargs.get('result') if 'result' in kwargs else (args[1] if len(args) >= 2 else None)
            sentence = extract_deepgram_transcript(payload) if payload is not None else ''

            # Only process and log non-empty transcripts to avoid log spam from interim results
            if sentence:
                # Optional: print transcript to console if TRANSCRIPT_ONLY mode
                if TRANSCRIPT_ONLY:
                    print(sentence)

                try:
                    if ws_open and websocket.client_state == WebSocketState.CONNECTED and not dg_closed_event.is_set():
                        # Schedule the coroutine on the main event loop from this callback thread
                        asyncio.run_coroutine_threadsafe(
                            websocket.send_json({
                                "type": "transcription",
                                "text": sentence,
                                "language": language or os.getenv("DEEPGRAM_LANGUAGE", "en-US")
                            }),
                            loop
                        )
                        logger.info(f"✓ Sent Deepgram transcription: {sentence[:100]}...")
                except Exception as e:
                    logger.error(f"Failed to send Deepgram transcription: {e}")

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

        # Determine language parameter for streaming
        # WebSocket doesn't support detect_language, so use 'multi' for auto-detection
        stream_language = language if language and language != "auto" else "multi"

        if stream_language == "multi":
            logger.info("Using multi-language support for automatic language detection in streaming")
        else:
            logger.info(f"Using language for streaming: {stream_language}")

        # Build options from PARAMS and app audio settings
        opts_kwargs = {
            "model": PARAMS["model"],
            "language": stream_language,
            "punctuate": PARAMS["punctuate"],
            "numerals": PARAMS["numerals"],
            "encoding": "linear16",
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "interim_results": True,
            "endpointing": False,  # Disable auto-closure on silence detection
            "vad_turnoff": 5000,   # 5 second VAD timeout (prevent premature closure)
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
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-reconnect", "1",              # Reconnect on connection failures
            "-reconnect_streamed", "1",     # Reconnect for streamed inputs
            "-reconnect_delay_max", "5",    # Max reconnect delay
            "-user_agent", "Mozilla/5.0",
            "-i", url,
            "-vn",                          # No video
            "-f", "s16le",                  # PCM signed 16-bit little-endian
            "-acodec", "pcm_s16le",         # Audio codec
            "-ac", str(CHANNELS),           # Mono
            "-ar", str(SAMPLE_RATE),        # 16kHz sample rate
            "pipe:1",
        ]

        proc = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        start_ts = asyncio.get_event_loop().time()
        # Support unlimited streaming: if TIME_LIMIT <= 0, stream indefinitely
        time_limit = float(TIME_LIMIT) if TIME_LIMIT > 0 else float("inf")

        try:
            bytes_sent = 0

            while True:
                # Check time limit only if it's not infinite
                if time_limit != float("inf"):
                    elapsed = asyncio.get_event_loop().time() - start_ts
                    if elapsed >= time_limit:
                        logger.info(f"Deepgram stream reached time limit of {TIME_LIMIT} seconds")
                        break

                data = await proc.stdout.read(4096)
                if not data:
                    # Check if ffmpeg process exited with an error
                    stderr_output = await proc.stderr.read()
                    if stderr_output:
                        logger.error(f"FFmpeg error: {stderr_output.decode('utf-8', errors='ignore')}")
                    logger.info(f"FFmpeg stream ended after {bytes_sent} bytes")
                    break

                # Stream raw PCM bytes to Deepgram as fast as possible
                # No throttling needed - Deepgram handles buffering
                # With endpointing disabled and high VAD timeout, connection stays open
                connection.send(data)
                bytes_sent += len(data)
        finally:
            try:
                proc.terminate()
                await proc.wait()
            except Exception:
                pass

        # Signal finish to Deepgram
        try:
            connection.finish()
            logger.info("Sent finish signal to Deepgram, waiting for final transcriptions...")
        except Exception:
            pass

        # Wait for Deepgram to finish processing all buffered audio
        # Calculate estimated processing time based on audio sent
        # Deepgram typically processes audio faster than real-time, but we need to account for:
        # - Network latency for sending results back
        # - Processing queue time
        # Use generous timeout: min 10s, max 60s, or estimated audio duration
        estimated_audio_seconds = bytes_sent / (SAMPLE_RATE * CHANNELS * 2) if bytes_sent > 0 else 0
        processing_timeout = min(max(estimated_audio_seconds + 10, 10), 60)

        logger.info(f"Waiting up to {processing_timeout:.1f}s for Deepgram to process {estimated_audio_seconds:.1f}s of audio")

        try:
            # Wait for Deepgram close event with appropriate timeout
            try:
                await asyncio.wait_for(dg_closed_event.wait(), timeout=processing_timeout)
                logger.info("Deepgram closed connection after processing")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for Deepgram close after {processing_timeout}s")
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
        model_name = data.get("model", "whisper-v3-turbo")  # Default to multilingual model
        capture_mode = data.get("captureMode", "full")
        enable_diarization = data.get("diarization", False)  # Speaker diarization flag

        if not url:
            await websocket.send_json({"error": "URL is required"})
            return

        logger.info(f"Starting transcription for URL: {url} with model: {model_name}")

        # Warn user if using a slow model
        slow_models = ["large", "medium"]
        if any(slow_model in model_name for slow_model in slow_models):
            await websocket.send_json({
                "type": "status",
                "message": f"⚠️ Using {model_name} - This model provides excellent quality but processes slowly. Consider using Deepgram for faster results."
            })

        # Check if user selected Deepgram
        if model_name == "deepgram":
            # For VOD content (complete videos), use pre-recorded API for better results
            # For true live streams, use streaming API
            # Detection: if URL is yt-dlp compatible or contains VOD patterns, treat as pre-recorded
            is_vod = should_use_ytdlp(url) or any(pattern in url.lower() for pattern in ['.mp4', '.mp3', '.wav', '.m4a', 'video-', '/media/'])

            if is_vod:
                logger.info(f"Detected VOD content, using pre-recorded API for better accuracy")
                await transcribe_vod_with_deepgram(websocket, url, language)
            else:
                logger.info(f"Detected live stream, using streaming API")
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

                            elif model_config["type"] == "faster_whisper":
                                # Use faster_whisper for transcription
                                fw_model = model["model"] if isinstance(model, dict) else model
                                
                                # Determine default language
                                default_lang = None
                                if model2 and "ivrit" in model2.lower():
                                    default_lang = "he"
                                
                                segments, info = fw_model.transcribe(
                                    info["path"],
                                    language=lang2 or default_lang,
                                    beam_size=5,
                                    best_of=5,
                                    patience=1,
                                    temperature=0
                                )
                                
                                text_parts = []
                                for segment in segments:
                                    text_parts.append(segment.text.strip())
                                text = " ".join(text_parts)
                                det_lang = info.language if info else (lang2 or 'unknown')
                                
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
                                await websocket.send_json({"error": f"Unsupported model type: {model_config.get('type')}"})
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
            # Download entire file first with automatic fallback (yt-dlp → ffmpeg)
            audio_file = await download_with_fallback(url, language, websocket=websocket)
            if not audio_file:
                await websocket.send_json({"error": "Failed to download audio from URL. All download methods failed."})
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
                    # Only delete if not in cache
                    if os.path.exists(audio_file) and DOWNLOAD_CACHE_DIR not in Path(audio_file).parents:
                        os.unlink(audio_file)

                return

            # Transcribe the downloaded file with incremental output and detailed progress
            await websocket.send_json({"type": "status", "message": "Transcribing downloaded audio..."})

            try:
                model = load_model(model_name)
                model_config = MODEL_CONFIGS[model_name]
                
                # Use the new incremental transcription function
                transcription_text, detected_language = await transcribe_with_incremental_output(
                    model, model_config, audio_file, language, websocket, model_name
                )

                # Send complete message
                await websocket.send_json({"type": "complete", "message": "Transcription complete"})

            finally:
                # Cleanup downloaded file (only if not in cache)
                if os.path.exists(audio_file) and DOWNLOAD_CACHE_DIR not in Path(audio_file).parents:
                    os.unlink(audio_file)
                    logger.debug(f"Cleaned up temporary file: {audio_file}")
                elif os.path.exists(audio_file):
                    logger.debug(f"Keeping cached file: {audio_file}")

            return

        # For Whisper/Ivrit models with VOD content: Download complete file first, then transcribe
        # This prevents queue overflow and data loss with slow models
        # Only use real-time streaming for actual live streams
        is_vod = should_use_ytdlp(url) or any(pattern in url.lower() for pattern in ['.mp4', '.mp3', '.wav', '.m4a', 'video-', '/media/'])

        if is_vod:
            # VOD Mode: Download complete file first, then batch transcribe
            await websocket.send_json({"type": "status", "message": "📥 Downloading complete audio file for batch transcription..."})

            try:
                # Download complete audio with automatic fallback (yt-dlp → ffmpeg)
                if should_use_ytdlp(url):
                    audio_file = await download_with_fallback(url, language, format='wav', websocket=websocket)
                else:
                    # Use ffmpeg for direct URLs - download complete file (no duration limit for VOD)
                    audio_file = await download_audio_with_ffmpeg(url, format='wav', duration=0, websocket=websocket)  # 0 = no limit

                if not audio_file:
                    await websocket.send_json({"error": "Failed to download audio file. All download methods failed. Please check the URL or try a different video."})
                    return

                file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
                await websocket.send_json({"type": "status", "message": f"✅ Downloaded {file_size_mb:.1f} MB. Starting batch transcription..."})

                # Batch transcribe with incremental output and detailed progress
                model = load_model(model_name)
                model_config = MODEL_CONFIGS[model_name]

                # Use diarization if requested
                if enable_diarization:
                    await websocket.send_json({"type": "status", "message": "🎭 Transcribing with speaker diarization..."})
                    diarized_segments, detected_language = await transcribe_with_diarization(
                        model, model_config, audio_file, language, websocket, model_name
                    )
                    # Combine all text for the transcript variable
                    transcript = " ".join([seg.get("text", "") for seg in diarized_segments])
                else:
                    # Use the new incremental transcription function for better UX
                    transcript, detected_language = await transcribe_with_incremental_output(
                        model, model_config, audio_file, language, websocket, model_name, chunk_seconds=60
                    )

                # Send completion message
                if transcript:
                    await websocket.send_json({"type": "complete", "message": "Transcription complete"})
                else:
                    await websocket.send_json({"type": "status", "message": "⚠️ No speech detected in audio file"})
                    await websocket.send_json({"type": "complete", "message": "Transcription complete (no speech detected)"})

                # Cleanup (only delete if not in download cache)
                try:
                    if DOWNLOAD_CACHE_DIR not in Path(audio_file).parents:
                        os.unlink(audio_file)
                    else:
                        logger.debug(f"Keeping cached file: {audio_file}")
                except:
                    pass

                return

            except Exception as e:
                logger.error(f"Batch transcription error: {e}")
                await websocket.send_json({"error": f"Transcription failed: {str(e)}"})
                return

        # Otherwise, use direct FFmpeg streaming (for true live streams only)
        await websocket.send_json({"type": "status", "message": "Starting real-time audio stream..."})

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


@app.get("/api/download-cache/stats")
async def download_cache_stats():
    """Get download cache statistics"""
    try:
        cache_files = list(DOWNLOAD_CACHE_DIR.glob("*.wav"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        # Get cache details
        cache_details = []
        for cache_file in cache_files:
            cache_details.append({
                "filename": cache_file.name,
                "size_mb": round(cache_file.stat().st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(cache_file.stat().st_mtime).isoformat(),
                "url_hash": cache_file.name.split('_')[0] if '_' in cache_file.name else 'unknown'
            })
        
        return {
            "enabled": True,
            "file_count": len(cache_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "max_age_hours": 24,
            "cache_directory": str(DOWNLOAD_CACHE_DIR),
            "files": cache_details
        }
    except Exception as e:
        return {"enabled": True, "error": str(e)}


@app.post("/api/download-cache/clear")
async def clear_download_cache():
    """Clear all cached download files"""
    try:
        deleted_count = 0
        deleted_size = 0
        
        for cache_file in DOWNLOAD_CACHE_DIR.glob("*.wav"):
            deleted_size += cache_file.stat().st_size
            cache_file.unlink()
            deleted_count += 1
        
        # Clear in-memory cache
        URL_DOWNLOADS.clear()
        
        return {
            "success": True, 
            "deleted_files": deleted_count,
            "freed_space_mb": round(deleted_size / (1024 * 1024), 2)
        }
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

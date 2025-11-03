"""Application constants."""
from pathlib import Path
from config.settings import AUDIO_CACHE_ENABLED

# Cache Directories
CACHE_DIR = Path("cache/audio")
DOWNLOAD_CACHE_DIR = Path("cache/downloads")
CAPTURE_DIR = Path("cache/captures")
CACHE_MAX_AGE_HOURS = 24  # Clean cache older than 24 hours
CACHE_ENABLED = AUDIO_CACHE_ENABLED

# Audio Configuration
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
CHANNELS = 1  # Mono audio
CHUNK_DURATION = 5   # seconds - very short for fast real-time processing
CHUNK_OVERLAP = 1    # seconds - minimal overlap
AUDIO_QUEUE_SIZE = 200  # Large queue to handle slow models (Ivrit, large models)
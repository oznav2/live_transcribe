"""Application configuration sourced from environment variables.

Ensures `.env` is loaded before any settings are read. Only lightweight
parsing is performed here (ints, bools). Heavier logic and device probing
belongs in other modules.
"""
from dotenv import load_dotenv
load_dotenv()  # Must run before any env lookups

import os
from typing import Optional


def parse_bool (value: str) -> bool:
	"""Strict boolean parsing for environment flags."""
	return str(value).strip().lower() in {'1', 'true', 't', 'yes', 'y'}


# Deepgram API
DEEPGRAM_API_KEY: str = os.getenv('DEEPGRAM_API_KEY', '')

# Base model selection
WHISPER_MODEL: str = os.getenv('WHISPER_MODEL', 'whisper-v3-turbo')

# Parallel transcription and yt-dlp chunking
USE_PARALLEL_TRANSCRIPTION: bool = parse_bool(
	os.getenv('USE_PARALLEL_TRANSCRIPTION', 'false')
)
PARALLEL_WORKERS: int = int(os.getenv('PARALLEL_WORKERS', '2'))
YTDLP_CHUNK_SECONDS: int = int(os.getenv('YTDLP_CHUNK_SECONDS', '5'))
YTDLP_CHUNK_OVERLAP: int = int(os.getenv('YTDLP_CHUNK_OVERLAP', '1'))

# Audio caching
AUDIO_CACHE_ENABLED: bool = parse_bool(os.getenv('AUDIO_CACHE_ENABLED', 'true'))

# Ivrit/faster-whisper CT2 config (defaults kept simple; device/compute_type
# fallbacks should be handled in the caller to avoid importing heavy libs here)
IVRIT_MODEL_NAME: str = os.getenv(
	'IVRIT_MODEL_NAME', 'ivrit-ai/whisper-large-v3-turbo-ct2'
)
IVRIT_DEVICE: Optional[str] = os.getenv('IVRIT_DEVICE', '')
IVRIT_COMPUTE_TYPE: Optional[str] = os.getenv('IVRIT_COMPUTE_TYPE', '')

# Deepgram streaming settings
DEEPGRAM_TIME_LIMIT: int = int(os.getenv('DEEPGRAM_TIME_LIMIT', '3600'))
DEEPGRAM_TRANSCRIPT_ONLY: bool = parse_bool(
	os.getenv('DEEPGRAM_TRANSCRIPT_ONLY', 'true')
)
DEEPGRAM_MODEL: str = os.getenv('DEEPGRAM_MODEL', 'nova-3')
DEEPGRAM_LANGUAGE: str = os.getenv('DEEPGRAM_LANGUAGE', 'en-US')

# Server port
PORT: int = int(os.getenv('PORT', '8000'))

# Device configuration (for compatibility)
DEVICE: str = 'cuda' if os.getenv('CUDA_VISIBLE_DEVICES') else 'cpu'
COMPUTE_TYPE: str = 'float16' if DEVICE == 'cuda' else 'int8'

# Service flags
USE_DEEPGRAM: bool = bool(DEEPGRAM_API_KEY)
GROQ_API_KEY: str = os.getenv('GROQ_API_KEY', '')
USE_GROQ: bool = bool(GROQ_API_KEY)

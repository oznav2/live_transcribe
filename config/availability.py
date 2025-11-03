"""Provider availability and model config assembly.

This module detects SDK/library availability and assembles MODEL_CONFIGS.
Avoids importing heavy ML libraries at module import time.
"""
import os
import logging
from typing import Dict, Any

from config.settings import (
    DEEPGRAM_API_KEY,
    WHISPER_MODEL,
    IVRIT_MODEL_NAME,
    IVRIT_DEVICE,
    IVRIT_COMPUTE_TYPE,
    DEEPGRAM_MODEL,
    DEEPGRAM_LANGUAGE,
)

logger = logging.getLogger(__name__)

# Detect library availability using lightweight checks
try:
    import faster_whisper
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logger.warning("faster_whisper not available - Ivrit CT2 models will not work")

try:
    import whisper
    OPENAI_WHISPER_AVAILABLE = True
    if FASTER_WHISPER_AVAILABLE:
        logger.warning("Both openai-whisper and faster-whisper are installed. This may cause conflicts.")
except ImportError:
    OPENAI_WHISPER_AVAILABLE = False

try:
    from pyannote.audio import Pipeline
    import torchaudio
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logger.warning("pyannote.audio not available - Speaker diarization disabled")

# Deepgram availability
try:
    from deepgram import DeepgramClient
    DEEPGRAM_AVAILABLE = bool(DEEPGRAM_API_KEY)
except ImportError:
    DEEPGRAM_AVAILABLE = False
    DeepgramClient = None

# Check if CUDA is available (lightweight check)
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False
    torch = None

# Default model
MODEL_SIZE = WHISPER_MODEL

# Model configurations
MODEL_CONFIGS: Dict[str, Any] = {}

# Primary models - faster_whisper CT2 models (always available)
if FASTER_WHISPER_AVAILABLE:
    MODEL_CONFIGS.update({
        # Primary Ivrit model - best for Hebrew
        "ivrit-ct2": {
            "type": "faster_whisper",
            "name": IVRIT_MODEL_NAME,
            "device": IVRIT_DEVICE or ("cuda" if CUDA_AVAILABLE else "cpu"),
            "compute_type": IVRIT_COMPUTE_TYPE or ("float16" if CUDA_AVAILABLE else "int8"),
        },
        # Alternative name for the same model
        "ivrit-v3-turbo": {
            "type": "faster_whisper",
            "name": "ivrit-ai/whisper-large-v3-turbo-ct2",
            "device": "cuda" if CUDA_AVAILABLE else "cpu",
            "compute_type": "float16" if CUDA_AVAILABLE else "int8"
        },
        # General Whisper v3 turbo model
        "whisper-v3-turbo": {
            "type": "faster_whisper",
            "name": "large-v3-turbo",
            "device": "cuda" if CUDA_AVAILABLE else "cpu",
            "compute_type": "float16" if CUDA_AVAILABLE else "int8"
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
    MODEL_CONFIGS["deepgram"] = {"type": "deepgram", "model": DEEPGRAM_MODEL, "language": DEEPGRAM_LANGUAGE}

# Ensure the default model is available
if MODEL_CONFIGS and MODEL_SIZE not in MODEL_CONFIGS:
    logger.warning(f"Default model '{MODEL_SIZE}' not available. Available models: {', '.join(MODEL_CONFIGS.keys())}")
    # Try to select first available model
    if MODEL_CONFIGS:
        MODEL_SIZE = list(MODEL_CONFIGS.keys())[0]
        logger.info(f"Switching to available model: {MODEL_SIZE}")
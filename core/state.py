"""Global state management for the application.

This module centralizes all global variables to ensure single source of truth
and prevent duplication across modules.
"""
import threading
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Model caches - store loaded models to avoid reloading
whisper_models: Dict[str, Any] = {}

# Diarization pipeline (singleton)
diarization_pipeline: Optional[Any] = None

# Thread safety locks for model loading
model_lock: threading.Lock = threading.Lock()
diarization_pipeline_lock: threading.Lock = threading.Lock()

# Thread pool executor for CPU-bound tasks (created at runtime)
executor: Optional[ThreadPoolExecutor] = None

# Runtime application state
cached_index_html: Optional[str] = None

# Capture tracking
CAPTURES: Dict[str, Any] = {}

# URL Downloads tracking - Maps URL hash to downloaded file path
URL_DOWNLOADS: Dict[str, Any] = {}

# Current model tracking
current_model: Optional[Any] = None
current_model_name: Optional[str] = None
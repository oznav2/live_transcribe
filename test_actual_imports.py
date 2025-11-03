#!/usr/bin/env python3
"""
Test actual imports as they are used in the modular application.
"""

import sys
import traceback
from pathlib import Path

# Add the webapp directory to Python path
sys.path.insert(0, '/home/user/webapp')

def test_import(import_stmt, description=""):
    """Test an import statement."""
    try:
        exec(import_stmt)
        print(f"✓ {import_stmt}")
        return True
    except Exception as e:
        print(f"✗ {import_stmt}")
        print(f"    Error: {e}")
        return False

def main():
    """Test all actual imports from the application."""
    
    print("=" * 70)
    print("TESTING ACTUAL APPLICATION IMPORTS")
    print("=" * 70)
    
    failed = []
    
    # Test app.py imports
    print("\n1. Testing app.py imports:")
    print("-" * 40)
    imports = [
        "import logging",
        "from pathlib import Path",
        "import uvicorn",
        "from fastapi import FastAPI",
        "from fastapi.staticfiles import StaticFiles",
        "from config.settings import PORT",
        "from core.lifespan import lifespan",
        "from api.routes import router as api_router",
        "from api.websocket import websocket_transcribe",
    ]
    for imp in imports:
        if not test_import(imp):
            failed.append(imp)
    
    # Test config module imports
    print("\n2. Testing config module imports:")
    print("-" * 40)
    
    # From config/settings.py
    test_import("from config.settings import WHISPER_MODEL, DEVICE, COMPUTE_TYPE")
    test_import("from config.settings import DEEPGRAM_API_KEY, USE_DEEPGRAM")
    test_import("from config.settings import GROQ_API_KEY, USE_GROQ")
    test_import("from config.settings import AUDIO_CACHE_ENABLED, PORT")
    
    # From config/constants.py
    test_import("from config.constants import UPLOAD_FOLDER, MAX_CONTENT_LENGTH")
    test_import("from config.constants import CHUNK_SIZE, SAMPLE_RATE, CHANNELS")
    test_import("from config.constants import CACHE_ENABLED")
    
    # From config/availability.py  
    test_import("from config.availability import IS_DIARIZATION_AVAILABLE")
    test_import("from config.availability import MODEL_SIZE, AUTH_TOKEN")
    test_import("from config.availability import WHISPER_AVAILABLE, FASTER_WHISPER_AVAILABLE")
    
    # Test core module imports
    print("\n3. Testing core module imports:")
    print("-" * 40)
    
    # From core/state.py
    test_import("from core.state import whisper_models, diarization_pipeline")
    test_import("from core.state import model_lock, diarization_pipeline_lock")
    test_import("from core.state import current_model, current_model_name")
    test_import("from core.state import CAPTURES, URL_DOWNLOADS")
    
    # From core/lifespan.py
    test_import("from core.lifespan import lifespan")
    
    # Test models module imports
    print("\n4. Testing models module imports:")
    print("-" * 40)
    
    test_import("from models.loader import load_model, load_diarization_pipeline")
    
    # Test utils module imports (with actual file names)
    print("\n5. Testing utils module imports:")
    print("-" * 40)
    
    test_import("from utils.cache import get_cached_download, save_download_to_cache")
    test_import("from utils.cache import generate_cache_key, get_cached_audio, save_to_cache")
    test_import("from utils.validators import is_youtube_url, should_use_ytdlp")
    test_import("from utils.helpers import format_duration, format_view_count")
    test_import("from utils.websocket_helpers import safe_ws_send")
    
    # Test services module imports (with actual file names)
    print("\n6. Testing services module imports:")
    print("-" * 40)
    
    test_import("from services.audio_processor import AudioStreamProcessor")
    test_import("from services.audio_processor import download_audio_with_ffmpeg")
    test_import("from services.transcription import transcribe_with_incremental_output")
    test_import("from services.transcription import TranscriptionService")
    test_import("from services.diarization import transcribe_with_diarization")
    test_import("from services.video_metadata import get_youtube_metadata")
    
    # Test api module imports
    print("\n7. Testing api module imports:")
    print("-" * 40)
    
    test_import("from api.routes import router")
    test_import("from api.websocket import websocket_transcribe")
    
    # Summary
    print("\n" + "=" * 70)
    if failed:
        print(f"FAILED IMPORTS ({len(failed)}):")
        for imp in failed:
            print(f"  - {imp}")
        print("\nThese imports need to be fixed.")
        return 1
    else:
        print("ALL IMPORTS SUCCESSFUL!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
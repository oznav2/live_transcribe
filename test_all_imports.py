#!/usr/bin/env python3
"""
Comprehensive import test script to identify all import issues after modular refactoring.
This script tests all imports without actually running the application logic.
"""

import sys
import traceback
from pathlib import Path

# Add the webapp directory to Python path
sys.path.insert(0, '/home/user/webapp')

def test_import(module_name, items=None):
    """Test importing a module and optionally specific items from it."""
    try:
        if items:
            # Test importing specific items
            import_stmt = f"from {module_name} import {', '.join(items)}"
            exec(import_stmt)
            print(f"✓ {import_stmt}")
        else:
            # Test importing the whole module
            import_stmt = f"import {module_name}"
            exec(import_stmt)
            print(f"✓ {import_stmt}")
        return True
    except ImportError as e:
        print(f"✗ {module_name}: {e}")
        return False
    except Exception as e:
        print(f"✗ {module_name}: Unexpected error: {e}")
        traceback.print_exc()
        return False

def main():
    """Test all module imports systematically."""
    
    print("=" * 60)
    print("TESTING ALL MODULE IMPORTS")
    print("=" * 60)
    
    failed_imports = []
    
    # Test config modules
    print("\n1. Testing config modules:")
    print("-" * 40)
    
    # config.settings
    if not test_import("config.settings", [
        "WHISPER_MODEL", "DEVICE", "COMPUTE_TYPE", "DEEPGRAM_API_KEY",
        "USE_DEEPGRAM", "GROQ_API_KEY", "USE_GROQ", "AUDIO_CACHE_ENABLED"
    ]):
        failed_imports.append("config.settings")
    
    # config.constants
    if not test_import("config.constants", [
        "UPLOAD_FOLDER", "MAX_CONTENT_LENGTH", "CHUNK_SIZE", "SAMPLE_RATE",
        "CHANNELS", "DEFAULT_VAD_THRESHOLD", "DEFAULT_MIN_SPEECH_DURATION",
        "DEFAULT_MIN_SILENCE_DURATION", "DEFAULT_SPEECH_PAD", "CACHE_ENABLED"
    ]):
        failed_imports.append("config.constants")
    
    # config.availability
    if not test_import("config.availability", [
        "IS_DIARIZATION_AVAILABLE", "AUTH_TOKEN", "WHISPER_AVAILABLE",
        "FASTER_WHISPER_AVAILABLE", "USE_FASTER_WHISPER", "MODEL_SIZE"
    ]):
        failed_imports.append("config.availability")
    
    # Test core modules
    print("\n2. Testing core modules:")
    print("-" * 40)
    
    # core.state
    if not test_import("core.state", [
        "whisper_models", "diarization_pipeline", "model_lock",
        "diarization_pipeline_lock", "current_model", "current_model_name",
        "clients", "processing_tasks", "chunk_processing_buffers"
    ]):
        failed_imports.append("core.state")
    
    # core.lifespan
    if not test_import("core.lifespan", ["lifespan"]):
        failed_imports.append("core.lifespan")
    
    # Test models module
    print("\n3. Testing models module:")
    print("-" * 40)
    
    if not test_import("models.loader", [
        "load_model", "load_diarization_pipeline"
    ]):
        failed_imports.append("models.loader")
    
    # Test utils modules
    print("\n4. Testing utils modules:")
    print("-" * 40)
    
    if not test_import("utils.audio", [
        "calculate_audio_fingerprint", "generate_cache_key",
        "get_cached_transcription", "save_transcription_to_cache",
        "get_audio_duration", "process_audio_chunk", "save_uploaded_file"
    ]):
        failed_imports.append("utils.audio")
    
    if not test_import("utils.transcription", [
        "transcribe_chunk_with_whisper", "transcribe_with_groq",
        "transcribe_with_deepgram"
    ]):
        failed_imports.append("utils.transcription")
    
    if not test_import("utils.diarization", ["perform_diarization"]):
        failed_imports.append("utils.diarization")
    
    if not test_import("utils.response", ["format_transcription_response"]):
        failed_imports.append("utils.response")
    
    if not test_import("utils.websocket", ["register_client", "unregister_client"]):
        failed_imports.append("utils.websocket")
    
    # Test services module
    print("\n5. Testing services module:")
    print("-" * 40)
    
    if not test_import("services.audio_stream", ["AudioStreamProcessor"]):
        failed_imports.append("services.audio_stream")
    
    # Test api module
    print("\n6. Testing api module:")
    print("-" * 40)
    
    if not test_import("api.routes"):
        failed_imports.append("api.routes")
    
    if not test_import("api.websocket_handlers"):
        failed_imports.append("api.websocket_handlers")
    
    # Test main app
    print("\n7. Testing main app:")
    print("-" * 40)
    
    if not test_import("app"):
        failed_imports.append("app")
    
    # Summary
    print("\n" + "=" * 60)
    if failed_imports:
        print(f"FAILED IMPORTS ({len(failed_imports)}):")
        for module in failed_imports:
            print(f"  - {module}")
        print("\nPlease fix these import issues before running the application.")
        return 1
    else:
        print("ALL IMPORTS SUCCESSFUL!")
        print("The modular refactoring appears to be complete.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
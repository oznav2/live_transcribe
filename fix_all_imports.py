#!/usr/bin/env python3
"""
Fix all import issues after modular refactoring.
This script adds missing exports and fixes import references.
"""

import os
import re
from pathlib import Path

def fix_config_settings():
    """Add missing exports to config/settings.py"""
    print("Fixing config/settings.py...")
    
    # Read current content
    with open('config/settings.py', 'r') as f:
        content = f.read()
    
    # Add missing variables at the end if not present
    additions = []
    
    if 'DEVICE' not in content:
        additions.append("\n# Device configuration (for compatibility)")
        additions.append("DEVICE: str = 'cuda' if os.getenv('CUDA_VISIBLE_DEVICES') else 'cpu'")
    
    if 'COMPUTE_TYPE' not in content:
        additions.append("COMPUTE_TYPE: str = 'float16' if DEVICE == 'cuda' else 'int8'")
    
    if 'USE_DEEPGRAM' not in content:
        additions.append("\n# Service flags")
        additions.append("USE_DEEPGRAM: bool = bool(DEEPGRAM_API_KEY)")
    
    if 'GROQ_API_KEY' not in content:
        additions.append("GROQ_API_KEY: str = os.getenv('GROQ_API_KEY', '')")
    
    if 'USE_GROQ' not in content:
        additions.append("USE_GROQ: bool = bool(GROQ_API_KEY)")
    
    if additions:
        with open('config/settings.py', 'a') as f:
            f.write('\n' + '\n'.join(additions) + '\n')
        print(f"  Added {len([a for a in additions if not a.startswith('#')])} missing exports")

def fix_config_constants():
    """Add missing exports to config/constants.py"""
    print("Fixing config/constants.py...")
    
    # Read current content
    with open('config/constants.py', 'r') as f:
        content = f.read()
    
    # Add missing constants
    additions = []
    
    if 'UPLOAD_FOLDER' not in content:
        additions.append("\n# Upload configuration")
        additions.append("UPLOAD_FOLDER = Path('uploads')")
        additions.append("UPLOAD_FOLDER.mkdir(exist_ok=True)")
    
    if 'MAX_CONTENT_LENGTH' not in content:
        additions.append("MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size")
    
    if '\nCHUNK_SIZE' not in content and 'CHUNK_SIZE =' not in content:
        additions.append("\n# Streaming configuration")
        additions.append("CHUNK_SIZE = 8192  # Bytes for streaming")
    
    if 'DEFAULT_VAD_THRESHOLD' not in content:
        additions.append("\n# VAD configuration")
        additions.append("DEFAULT_VAD_THRESHOLD = 0.5")
        additions.append("DEFAULT_MIN_SPEECH_DURATION = 0.5")
        additions.append("DEFAULT_MIN_SILENCE_DURATION = 1.0")
        additions.append("DEFAULT_SPEECH_PAD = 0.5")
    
    if additions:
        with open('config/constants.py', 'a') as f:
            f.write('\n' + '\n'.join(additions) + '\n')
        print(f"  Added {len([a for a in additions if not a.startswith('#')])} missing constants")

def fix_config_availability():
    """Add missing exports to config/availability.py"""
    print("Fixing config/availability.py...")
    
    # Read current content
    with open('config/availability.py', 'r') as f:
        content = f.read()
    
    # Add missing exports at the end
    additions = []
    
    if 'IS_DIARIZATION_AVAILABLE' not in content:
        additions.append("\n# Export availability flags")
        additions.append("IS_DIARIZATION_AVAILABLE = PYANNOTE_AVAILABLE")
    
    if 'AUTH_TOKEN' not in content:
        additions.append("AUTH_TOKEN = os.getenv('HF_TOKEN', '')")
    
    if 'WHISPER_AVAILABLE' not in content:
        additions.append("WHISPER_AVAILABLE = OPENAI_WHISPER_AVAILABLE or FASTER_WHISPER_AVAILABLE")
    
    if 'USE_FASTER_WHISPER' not in content:
        additions.append("USE_FASTER_WHISPER = FASTER_WHISPER_AVAILABLE")
    
    if additions:
        with open('config/availability.py', 'a') as f:
            f.write('\n' + '\n'.join(additions) + '\n')
        print(f"  Added {len([a for a in additions if not a.startswith('#')])} missing exports")

def fix_core_state():
    """Add missing state variables to core/state.py"""
    print("Fixing core/state.py...")
    
    # Read current content
    with open('core/state.py', 'r') as f:
        content = f.read()
    
    # Add missing state variables
    additions = []
    
    if 'clients' not in content:
        additions.append("\n# WebSocket client tracking")
        additions.append("clients: Dict[str, Any] = {}")
    
    if 'processing_tasks' not in content:
        additions.append("\n# Task tracking")
        additions.append("processing_tasks: Dict[str, Any] = {}")
    
    if 'chunk_processing_buffers' not in content:
        additions.append("chunk_processing_buffers: Dict[str, Any] = {}")
    
    if additions:
        with open('core/state.py', 'a') as f:
            f.write('\n' + '\n'.join(additions) + '\n')
        print(f"  Added {len([a for a in additions if not a.startswith('#')])} missing state variables")

def fix_models_loader():
    """Fix models/loader.py exports"""
    print("Fixing models/loader.py...")
    
    # Read current content
    with open('models/loader.py', 'r') as f:
        content = f.read()
    
    # Check if load_diarization_pipeline exists, if not rename or add alias
    if 'def load_diarization_pipeline' not in content:
        # Add alias to get_diarization_pipeline
        additions = [
            "\n# Alias for compatibility",
            "load_diarization_pipeline = get_diarization_pipeline"
        ]
        
        with open('models/loader.py', 'a') as f:
            f.write('\n' + '\n'.join(additions) + '\n')
        print("  Added load_diarization_pipeline alias")

def create_missing_modules():
    """Create any missing module files that are expected"""
    print("Checking for missing expected modules...")
    
    # These modules appear to be referenced but might be named differently
    module_mappings = {
        'utils/audio.py': 'utils/cache.py',  # If audio.py doesn't exist but is expected
        'utils/transcription.py': None,  # Might not exist
        'utils/diarization.py': None,  # Might not exist
        'utils/response.py': None,  # Might not exist
        'api/websocket_handlers.py': 'api/websocket.py',  # Might be renamed
        'services/audio_stream.py': 'services/audio_processor.py',  # Might be renamed
    }
    
    for expected, actual in module_mappings.items():
        if not Path(expected).exists() and actual and Path(actual).exists():
            print(f"  Note: {expected} doesn't exist, but {actual} does")

def main():
    """Run all fixes"""
    print("=" * 60)
    print("FIXING ALL IMPORT ISSUES")
    print("=" * 60)
    
    # Change to webapp directory
    os.chdir('/home/user/webapp')
    
    # Fix each module
    fix_config_settings()
    fix_config_constants()
    fix_config_availability()
    fix_core_state()
    fix_models_loader()
    create_missing_modules()
    
    print("\n" + "=" * 60)
    print("All fixes applied!")
    print("=" * 60)

if __name__ == "__main__":
    main()
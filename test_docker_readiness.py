#!/usr/bin/env python3
"""
Comprehensive test to ensure the modular application is ready for Docker deployment.
Tests all critical imports and verifies the application structure.
"""

import sys
import os
from pathlib import Path

# Set up environment
os.environ.setdefault('DEEPGRAM_API_KEY', '')
os.environ.setdefault('HF_TOKEN', '')
os.environ.setdefault('PORT', '8000')

# Add webapp to path
sys.path.insert(0, '/home/user/webapp')

def test_docker_readiness():
    """Comprehensive Docker readiness test"""
    print("=" * 70)
    print("DOCKER DEPLOYMENT READINESS TEST")
    print("=" * 70)
    
    errors = []
    warnings = []
    
    # 1. Check directory structure
    print("\n1. Checking directory structure...")
    required_dirs = ['config', 'core', 'models', 'utils', 'services', 'api', 'static']
    for dir_name in required_dirs:
        if Path(dir_name).is_dir():
            print(f"  ✓ {dir_name}/ exists")
        else:
            errors.append(f"Missing directory: {dir_name}/")
            print(f"  ✗ {dir_name}/ missing")
    
    # 2. Check critical files
    print("\n2. Checking critical files...")
    critical_files = [
        'app.py',
        'requirements.txt',
        'Dockerfile',
        'static/css/styles.css',
        'config/__init__.py',
        'core/__init__.py',
        'models/__init__.py',
        'utils/__init__.py',
        'services/__init__.py',
        'api/__init__.py'
    ]
    for file_path in critical_files:
        if Path(file_path).is_file():
            print(f"  ✓ {file_path}")
        else:
            errors.append(f"Missing file: {file_path}")
            print(f"  ✗ {file_path}")
    
    # 3. Test imports without ML dependencies
    print("\n3. Testing core imports (no ML dependencies)...")
    try:
        # Config modules
        from config import settings, constants, availability
        print("  ✓ Configuration modules")
        
        # Core modules
        from core import state, lifespan
        print("  ✓ Core modules")
        
        # Models module
        from models import loader
        print("  ✓ Models module")
        
        # Utils modules
        from utils import cache, validators, helpers, websocket_helpers
        print("  ✓ Utils modules")
        
        # Services modules
        from services import audio_processor, video_metadata
        print("  ✓ Basic services")
        
        # API modules
        from api import routes
        print("  ✓ API routes")
        
        # Main app
        import app
        print("  ✓ Main application")
        
    except ImportError as e:
        errors.append(f"Import error: {e}")
        print(f"  ✗ Import failed: {e}")
    
    # 4. Test critical exports
    print("\n4. Testing critical exports...")
    try:
        from config.settings import PORT, WHISPER_MODEL, DEVICE, COMPUTE_TYPE
        from config.settings import DEEPGRAM_API_KEY, USE_DEEPGRAM, GROQ_API_KEY, USE_GROQ
        from config.settings import AUDIO_CACHE_ENABLED
        print("  ✓ Settings exports")
        
        from config.constants import CACHE_ENABLED, UPLOAD_FOLDER, MAX_CONTENT_LENGTH
        from config.constants import CHUNK_SIZE, SAMPLE_RATE, CHANNELS
        print("  ✓ Constants exports")
        
        from config.availability import MODEL_SIZE, IS_DIARIZATION_AVAILABLE
        from config.availability import AUTH_TOKEN, WHISPER_AVAILABLE, USE_FASTER_WHISPER
        print("  ✓ Availability exports")
        
        from core.state import whisper_models, current_model, current_model_name
        from core.state import model_lock, diarization_pipeline, diarization_pipeline_lock
        print("  ✓ State exports")
        
        from models.loader import load_model, load_diarization_pipeline
        print("  ✓ Model loader exports")
        
    except ImportError as e:
        errors.append(f"Export error: {e}")
        print(f"  ✗ Export failed: {e}")
    
    # 5. Test ML-dependent imports (with proper error handling)
    print("\n5. Testing ML-dependent modules...")
    try:
        from services import transcription, diarization
        print("  ✓ ML services can be imported")
    except ImportError as e:
        if 'torch' in str(e) or 'whisper' in str(e) or 'faster_whisper' in str(e):
            warnings.append(f"ML libraries not installed: {e}")
            print(f"  ⚠ ML libraries not installed (OK for Docker build)")
        else:
            errors.append(f"Unexpected import error: {e}")
            print(f"  ✗ Unexpected error: {e}")
    
    try:
        from api import websocket
        print("  ✓ WebSocket module can be imported")
    except ImportError as e:
        if 'torch' in str(e) or 'whisper' in str(e):
            warnings.append(f"ML libraries needed for websocket: {e}")
            print(f"  ⚠ ML libraries needed (OK for Docker build)")
        else:
            errors.append(f"WebSocket import error: {e}")
            print(f"  ✗ WebSocket error: {e}")
    
    # 6. Check FastAPI app
    print("\n6. Checking FastAPI application...")
    try:
        import app
        if hasattr(app, 'app'):
            from fastapi import FastAPI
            if isinstance(app.app, FastAPI):
                print("  ✓ FastAPI instance created")
                routes = [r.path for r in app.app.routes]
                print(f"  ✓ {len(routes)} routes registered")
            else:
                errors.append("app.app is not a FastAPI instance")
                print("  ✗ Invalid FastAPI instance")
        else:
            errors.append("app.app not found")
            print("  ✗ FastAPI app not found")
    except Exception as e:
        errors.append(f"FastAPI check failed: {e}")
        print(f"  ✗ FastAPI check failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if errors:
        print(f"\n❌ CRITICAL ERRORS ({len(errors)}):")
        for error in errors:
            print(f"  • {error}")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for warning in warnings:
            print(f"  • {warning}")
    
    if not errors:
        print("\n✅ DOCKER DEPLOYMENT READY!")
        print("The application has been successfully modularized.")
        print("All critical imports are working correctly.")
        print("\nNext steps:")
        print("1. Commit all changes to the Modular branch")
        print("2. Push to GitHub")
        print("3. Build and test the Docker container")
        return 0
    else:
        print("\n❌ NOT READY FOR DOCKER DEPLOYMENT")
        print("Please fix the critical errors listed above.")
        return 1

if __name__ == "__main__":
    sys.exit(test_docker_readiness())
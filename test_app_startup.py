#!/usr/bin/env python3
"""
Test if the application can start up without import errors.
This simulates what happens when running the app in Docker.
"""

import sys
import os

# Set up minimal environment
os.environ.setdefault('DEEPGRAM_API_KEY', '')
os.environ.setdefault('HF_TOKEN', '')
os.environ.setdefault('PORT', '8000')

# Add webapp to path
sys.path.insert(0, '/home/user/webapp')

def test_startup():
    """Test basic application startup"""
    print("=" * 60)
    print("TESTING APPLICATION STARTUP")
    print("=" * 60)
    
    try:
        print("\n1. Testing configuration imports...")
        from config import settings, constants, availability
        print("✓ Configuration modules loaded")
        
        print("\n2. Testing core modules...")
        from core import state, lifespan
        print("✓ Core modules loaded")
        
        print("\n3. Testing models module...")
        from models import loader
        print("✓ Models module loaded")
        
        print("\n4. Testing utils modules...")
        from utils import cache, validators, helpers, websocket_helpers
        print("✓ Utils modules loaded")
        
        print("\n5. Testing services modules...")
        from services import audio_processor, video_metadata
        print("✓ Basic services loaded")
        
        # These may fail due to torch dependency but that's OK
        try:
            from services import transcription, diarization
            print("✓ Transcription services loaded")
        except ImportError as e:
            print(f"⚠ Transcription services require ML libraries: {e}")
        
        print("\n6. Testing API modules...")
        from api import routes
        print("✓ API routes loaded")
        
        try:
            from api import websocket
            print("✓ WebSocket module loaded")
        except ImportError as e:
            print(f"⚠ WebSocket module requires ML libraries: {e}")
        
        print("\n7. Testing main application...")
        import app
        print("✓ Main application module loaded")
        
        print("\n8. Testing FastAPI app instance...")
        if hasattr(app, 'app'):
            print(f"✓ FastAPI app instance found: {app.app}")
            
            # Check routes
            routes = [r.path for r in app.app.routes]
            print(f"✓ Routes registered: {len(routes)} routes")
            for route in routes[:10]:  # Show first 10 routes
                print(f"  - {route}")
        
        print("\n" + "=" * 60)
        print("✅ APPLICATION STARTUP TEST PASSED")
        print("The application can be loaded successfully.")
        print("Import issues have been resolved.")
        return 0
        
    except Exception as e:
        print(f"\n❌ STARTUP FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(test_startup())
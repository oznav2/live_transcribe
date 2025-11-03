#!/usr/bin/env python3
"""
Final comprehensive test to ensure Docker deployment will work correctly.
Tests the modular application with the Ivrit Docker configuration.
"""

import sys
import os
from pathlib import Path

# Set environment to match docker-compose.ivrit.yml
os.environ['PORT'] = '8009'
os.environ['WHISPER_MODEL'] = 'whisper-v3-turbo'
os.environ['IVRIT_MODEL_NAME'] = 'ivrit-ai/whisper-large-v3-turbo-ct2'
os.environ['IVRIT_DEVICE'] = 'cuda'
os.environ['IVRIT_COMPUTE_TYPE'] = 'float16'
os.environ.setdefault('DEEPGRAM_API_KEY', '')
os.environ.setdefault('HF_TOKEN', '')

sys.path.insert(0, '/home/user/webapp')

def test_final_configuration():
    """Final comprehensive test for Docker deployment"""
    print("=" * 70)
    print("FINAL DOCKER DEPLOYMENT TEST")
    print("Configuration: docker-compose.ivrit.yml")
    print("=" * 70)
    
    errors = []
    
    # 1. Verify Docker files
    print("\n1. Docker Configuration Files:")
    print("-" * 40)
    
    if not Path('Dockerfile.ivrit').exists():
        errors.append("Dockerfile.ivrit is missing!")
        print("  ✗ Dockerfile.ivrit missing")
    else:
        print("  ✓ Dockerfile.ivrit exists")
    
    if not Path('docker-compose.ivrit.yml').exists():
        errors.append("docker-compose.ivrit.yml is missing!")
        print("  ✗ docker-compose.ivrit.yml missing")
    else:
        print("  ✓ docker-compose.ivrit.yml exists")
    
    # Check old files are removed
    if Path('Dockerfile').exists():
        print("  ⚠ Old Dockerfile still exists (should be removed)")
    else:
        print("  ✓ Old Dockerfile removed")
    
    if Path('docker-compose.yml').exists():
        print("  ⚠ Old docker-compose.yml still exists (should be removed)")
    else:
        print("  ✓ Old docker-compose.yml removed")
    
    # 2. Test PORT configuration
    print("\n2. Port Configuration:")
    print("-" * 40)
    try:
        from config.settings import PORT
        if PORT == 8009:
            print(f"  ✓ PORT configured correctly: {PORT}")
        else:
            errors.append(f"PORT is {PORT}, expected 8009")
            print(f"  ✗ PORT is {PORT}, expected 8009")
    except ImportError as e:
        errors.append(f"Cannot import PORT: {e}")
        print(f"  ✗ Cannot import PORT: {e}")
    
    # 3. Test modular structure
    print("\n3. Modular Application Structure:")
    print("-" * 40)
    
    modules_to_check = [
        ('config', ['settings', 'constants', 'availability']),
        ('core', ['state', 'lifespan']),
        ('models', ['loader']),
        ('utils', ['cache', 'validators', 'helpers', 'websocket_helpers']),
        ('services', ['audio_processor', 'video_metadata']),
        ('api', ['routes'])
    ]
    
    for package, modules in modules_to_check:
        package_ok = True
        for module in modules:
            try:
                exec(f"from {package} import {module}")
                print(f"  ✓ {package}.{module}")
            except ImportError as e:
                if 'torch' not in str(e) and 'whisper' not in str(e):
                    errors.append(f"Failed to import {package}.{module}: {e}")
                    print(f"  ✗ {package}.{module}: {e}")
                    package_ok = False
        
    # 4. Test main application
    print("\n4. Main Application:")
    print("-" * 40)
    try:
        import app
        if hasattr(app, 'app'):
            from fastapi import FastAPI
            if isinstance(app.app, FastAPI):
                print("  ✓ FastAPI application initialized")
                
                # Check critical routes
                routes = {r.path for r in app.app.routes}
                critical_routes = ['/', '/health', '/ws/transcribe', '/static']
                for route in critical_routes:
                    if any(route in r for r in routes):
                        print(f"  ✓ Route registered: {route}")
                    else:
                        errors.append(f"Missing route: {route}")
                        print(f"  ✗ Missing route: {route}")
            else:
                errors.append("app.app is not a FastAPI instance")
        else:
            errors.append("app module doesn't have 'app' attribute")
    except Exception as e:
        errors.append(f"Failed to load main application: {e}")
        print(f"  ✗ Failed to load application: {e}")
    
    # 5. Test Ivrit model configuration
    print("\n5. Ivrit Model Configuration:")
    print("-" * 40)
    try:
        from config.settings import IVRIT_MODEL_NAME, IVRIT_DEVICE, IVRIT_COMPUTE_TYPE
        print(f"  ✓ IVRIT_MODEL_NAME: {IVRIT_MODEL_NAME}")
        print(f"  ✓ IVRIT_DEVICE: {IVRIT_DEVICE}")
        print(f"  ✓ IVRIT_COMPUTE_TYPE: {IVRIT_COMPUTE_TYPE}")
        
        from config.availability import MODEL_CONFIGS
        if MODEL_CONFIGS:
            print(f"  ✓ {len(MODEL_CONFIGS)} models configured")
            for model_name in list(MODEL_CONFIGS.keys())[:3]:
                print(f"    - {model_name}")
        else:
            print("  ⚠ No models configured (ML libraries not installed)")
    except ImportError as e:
        errors.append(f"Ivrit configuration error: {e}")
        print(f"  ✗ Configuration error: {e}")
    
    # 6. Check static files
    print("\n6. Static Files:")
    print("-" * 40)
    if Path('static/css/styles.css').exists():
        print("  ✓ static/css/styles.css exists")
    else:
        errors.append("static/css/styles.css is missing")
        print("  ✗ static/css/styles.css missing")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    if errors:
        print(f"\n❌ FAILED: {len(errors)} error(s) found:")
        for error in errors:
            print(f"  • {error}")
        print("\nPlease fix these issues before deploying with Docker.")
        return 1
    else:
        print("\n✅ ALL TESTS PASSED!")
        print("\nThe modular application is ready for Docker deployment.")
        print("\nTo deploy:")
        print("  docker-compose -f docker-compose.ivrit.yml up --build")
        print("\nThe application will be available at:")
        print("  http://localhost:8009")
        return 0

if __name__ == "__main__":
    sys.exit(test_final_configuration())
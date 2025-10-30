#!/usr/bin/env python3
"""
Test script for Ivrit AI models integration
Tests both faster_whisper and fallback GGML models
"""

import os
import sys
import time
import tempfile
import subprocess
from pathlib import Path

# Test if required packages are available
try:
    import faster_whisper
    print("✓ faster_whisper is installed")
    FW_AVAILABLE = True
except ImportError:
    print("✗ faster_whisper is NOT installed")
    FW_AVAILABLE = False

try:
    import torch
    print(f"✓ PyTorch {torch.__version__} is installed")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("✗ PyTorch is NOT installed")

try:
    import ivrit
    print("✓ ivrit package is installed")
except ImportError:
    print("✗ ivrit package is NOT installed")

print("\n" + "="*50)
print("Testing Ivrit Models")
print("="*50)

def create_test_audio():
    """Create a simple test audio file using ffmpeg"""
    # Create a 5-second silent audio file for testing
    test_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    cmd = [
        'ffmpeg', '-f', 'lavfi', '-i', 'anullsrc=r=16000:cl=mono',
        '-t', '5', '-y', test_file.name
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        print(f"Created test audio file: {test_file.name}")
        return test_file.name
    except Exception as e:
        print(f"Failed to create test audio: {e}")
        return None

def test_faster_whisper_model(model_name="ivrit-ai/whisper-large-v3-turbo-ct2"):
    """Test loading and using a faster_whisper model"""
    if not FW_AVAILABLE:
        print("Skipping faster_whisper test - not installed")
        return False
    
    print(f"\nTesting faster_whisper model: {model_name}")
    print("-" * 40)
    
    try:
        # Determine device and compute type
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        print(f"Loading model on {device} with {compute_type} precision...")
        start_time = time.time()
        
        model = faster_whisper.WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root="/root/.cache/whisper"
        )
        
        load_time = time.time() - start_time
        print(f"✓ Model loaded successfully in {load_time:.2f} seconds")
        
        # Test transcription with a dummy audio file
        test_audio = create_test_audio()
        if test_audio:
            print("Testing transcription...")
            start_time = time.time()
            
            segments, info = model.transcribe(
                test_audio,
                language="he",
                beam_size=5,
                best_of=5,
                temperature=0
            )
            
            # Process segments
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text)
            
            transcribe_time = time.time() - start_time
            result_text = ' '.join(text_parts).strip()
            
            print(f"✓ Transcription completed in {transcribe_time:.2f} seconds")
            print(f"  Detected language: {info.language if info else 'unknown'}")
            print(f"  Result: '{result_text}' (empty expected for silent audio)")
            
            # Clean up
            os.unlink(test_audio)
            
        return True
        
    except Exception as e:
        print(f"✗ Failed to test model: {e}")
        return False

def test_all_ivrit_models():
    """Test all configured Ivrit models"""
    models_to_test = [
        "ivrit-ai/whisper-large-v3-turbo-ct2",
        "large-v3-turbo"
    ]
    
    results = {}
    for model_name in models_to_test:
        success = test_faster_whisper_model(model_name)
        results[model_name] = success
    
    return results

def check_whisper_cpp():
    """Check if whisper.cpp is available for GGML fallback"""
    whisper_cpp_path = os.getenv("WHISPER_CPP_PATH", "/app/whisper.cpp/build/bin/whisper-cli")
    ggml_model_path = os.getenv("IVRIT_MODEL_PATH", "/app/models/ivrit-whisper-large-v3-turbo.bin")
    
    print("\nChecking GGML/whisper.cpp fallback:")
    print("-" * 40)
    
    if os.path.exists(whisper_cpp_path):
        print(f"✓ whisper.cpp found at: {whisper_cpp_path}")
    else:
        print(f"✗ whisper.cpp NOT found at: {whisper_cpp_path}")
    
    if os.path.exists(ggml_model_path):
        print(f"✓ GGML model found at: {ggml_model_path}")
    else:
        print(f"✗ GGML model NOT found at: {ggml_model_path}")

if __name__ == "__main__":
    # Run tests
    if FW_AVAILABLE:
        results = test_all_ivrit_models()
        
        print("\n" + "="*50)
        print("Test Results Summary")
        print("="*50)
        for model, success in results.items():
            status = "✓ PASSED" if success else "✗ FAILED"
            print(f"{model}: {status}")
    else:
        print("\n⚠️  Cannot test Ivrit models - faster_whisper not installed")
        print("Please build with Dockerfile.ivrit to include all dependencies")
    
    # Check fallback options
    check_whisper_cpp()
    
    print("\n" + "="*50)
    print("Testing complete!")
    print("="*50)
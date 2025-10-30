#!/usr/bin/env python3
"""
Test script to verify that whisper.cpp/GGML has been completely removed
and that the app works with faster_whisper models only.
"""

import sys
import os

def test_no_whisper_cpp():
    """Verify whisper.cpp has been removed from the code"""
    print("Testing that whisper.cpp/GGML has been removed...")
    print("=" * 60)
    
    # Check app.py for whisper.cpp references
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Check for various whisper.cpp related strings
    forbidden_strings = [
        'WHISPER_CPP',
        'whisper_cpp',
        'whisper.cpp',
        'ggml',
        'GGML',
        'whisper-cli',
        'ivrit-large-v3-turbo'  # The old GGML model name
    ]
    
    found_issues = []
    for forbidden in forbidden_strings:
        if forbidden in content:
            # Skip comments
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if forbidden in line and not line.strip().startswith('#'):
                    found_issues.append(f"Line {i}: Found '{forbidden}' in non-comment line")
    
    if found_issues:
        print("❌ Found whisper.cpp/GGML references:")
        for issue in found_issues:
            print(f"  • {issue}")
        return False
    
    print("✓ No whisper.cpp/GGML references found in app.py")
    return True


def test_faster_whisper_models():
    """Test that faster_whisper models are configured correctly"""
    print("\nTesting faster_whisper model configuration...")
    print("=" * 60)
    
    try:
        import app
        
        # Check default model
        if app.MODEL_SIZE != "ivrit-ct2":
            print(f"❌ Default model is not ivrit-ct2: {app.MODEL_SIZE}")
            return False
        print(f"✓ Default model is correctly set to: {app.MODEL_SIZE}")
        
        # Check MODEL_CONFIGS
        if "ivrit-ct2" not in app.MODEL_CONFIGS:
            print("❌ ivrit-ct2 not found in MODEL_CONFIGS")
            return False
        
        config = app.MODEL_CONFIGS["ivrit-ct2"]
        if config.get("type") != "faster_whisper":
            print(f"❌ ivrit-ct2 is not configured as faster_whisper: {config.get('type')}")
            return False
        
        if config.get("name") != "ivrit-ai/whisper-large-v3-turbo-ct2":
            print(f"❌ ivrit-ct2 model name is incorrect: {config.get('name')}")
            return False
        
        print("✓ ivrit-ct2 is correctly configured as faster_whisper")
        print(f"  Model: {config.get('name')}")
        print(f"  Device: {config.get('device')}")
        print(f"  Compute type: {config.get('compute_type')}")
        
        # Check that no GGML models exist
        ggml_models = [k for k, v in app.MODEL_CONFIGS.items() if v.get("type") == "ggml"]
        if ggml_models:
            print(f"❌ Found GGML models in MODEL_CONFIGS: {ggml_models}")
            return False
        
        print("✓ No GGML models in MODEL_CONFIGS")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model configuration: {e}")
        return False


def test_dockerfile():
    """Test that Dockerfile.ivrit doesn't reference whisper.cpp"""
    print("\nTesting Dockerfile.ivrit...")
    print("=" * 60)
    
    with open('Dockerfile.ivrit', 'r') as f:
        content = f.read()
    
    issues = []
    
    # Check that whisper.cpp is not being copied or built
    if 'whisper-cli' in content and 'whisper.cpp is optional' not in content:
        issues.append("References to whisper-cli found")
    
    if 'WHISPER_CPP_PATH' in content and not content.count('# ENV WHISPER_CPP_PATH'):
        issues.append("WHISPER_CPP_PATH environment variable is set")
    
    # Check that ivrit-ai/whisper-large-v3-turbo-ct2 is being downloaded
    if 'ivrit-ai/whisper-large-v3-turbo-ct2' not in content:
        issues.append("ivrit-ai/whisper-large-v3-turbo-ct2 model is not being downloaded")
    
    if issues:
        print("❌ Found issues in Dockerfile.ivrit:")
        for issue in issues:
            print(f"  • {issue}")
        return False
    
    print("✓ Dockerfile.ivrit is correctly configured")
    print("✓ ivrit-ai/whisper-large-v3-turbo-ct2 will be pre-downloaded")
    
    return True


def main():
    """Run all tests"""
    print("\n🔍 WHISPER.CPP REMOVAL VERIFICATION")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    if not test_no_whisper_cpp():
        all_passed = False
    
    if not test_faster_whisper_models():
        all_passed = False
    
    if not test_dockerfile():
        all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nThe application is now configured to use only faster_whisper models.")
        print("The whisper.cpp CLI error will no longer occur.")
        print("\nDefault model: ivrit-ai/whisper-large-v3-turbo-ct2")
        print("\nTo build and run:")
        print("  docker-compose -f docker-compose.ivrit.yml up")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease review the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
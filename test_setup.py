#!/usr/bin/env python3
"""
Simple test script to verify the environment setup
"""
import subprocess
import sys

def test_python_version():
    """Check Python version"""
    print("✓ Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+")
        return False

def test_ffmpeg():
    """Check if FFmpeg is installed"""
    print("\n✓ Checking FFmpeg installation...")
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"  ✓ {version_line} - OK")
            return True
        else:
            print("  ✗ FFmpeg not working properly")
            return False
    except FileNotFoundError:
        print("  ✗ FFmpeg not found in PATH")
        print("  → Install FFmpeg: apt-get install ffmpeg (Ubuntu/Debian)")
        return False
    except Exception as e:
        print(f"  ✗ Error checking FFmpeg: {e}")
        return False

def test_imports():
    """Test if required Python packages can be imported"""
    print("\n✓ Checking Python packages...")
    packages = [
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('whisper', 'OpenAI Whisper'),
        ('torch', 'PyTorch'),
    ]
    
    all_ok = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"  ✓ {name} - OK")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            print(f"    → Install: pip install {package}")
            all_ok = False
    
    return all_ok

def test_files():
    """Check if required files exist"""
    print("\n✓ Checking project files...")
    files = [
        'app.py',
        'requirements.txt',
        'Dockerfile',
        'docker-compose.yml',
        'static/index.html',
    ]
    
    all_ok = True
    for file in files:
        try:
            with open(file, 'r') as f:
                print(f"  ✓ {file} - OK")
        except FileNotFoundError:
            print(f"  ✗ {file} - MISSING")
            all_ok = False
    
    return all_ok

def main():
    print("=" * 60)
    print("Live Transcription App - Environment Check")
    print("=" * 60)
    
    results = []
    results.append(("Python Version", test_python_version()))
    results.append(("FFmpeg", test_ffmpeg()))
    results.append(("Python Packages", test_imports()))
    results.append(("Project Files", test_files()))
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10} {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All checks passed! You're ready to run the application.")
        print("\nNext steps:")
        print("  1. Build Docker image: docker-compose build")
        print("  2. Start application: docker-compose up")
        print("  3. Open browser: http://localhost:8009")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

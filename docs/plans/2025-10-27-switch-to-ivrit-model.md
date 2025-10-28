# Switch to Ivrit Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Configure the application to use the Ivrit Whisper model (ivrit-large-v3-turbo) instead of the default base/large OpenAI Whisper model.

**Architecture:** The application already has model configuration support with both OpenAI and GGML model types. The Ivrit model (548MB GGML binary) exists at `models/ivrit-whisper-large-v3-turbo.bin`. We need to ensure environment variables, startup configuration, and default settings all point to the Ivrit model.

**Tech Stack:**
- FastAPI (web framework)
- whisper-cpp-python (GGML model loader)
- Python 3.11
- Docker

---

## Task 1: Update Environment Configuration

**Files:**
- Modify: `.env:8`

**Step 1: Update .env to use ivrit model**

Change line 8 in `.env` from:
```bash
WHISPER_MODEL=large
```

To:
```bash
WHISPER_MODEL=ivrit-large-v3-turbo
```

**Step 2: Verify the change**

Run: `cat .env | grep WHISPER_MODEL`
Expected output: `WHISPER_MODEL=ivrit-large-v3-turbo`

**Step 3: Commit the change**

```bash
git add .env
git commit -m "config: switch default model to ivrit-large-v3-turbo"
```

---

## Task 2: Fix Model Size Reference in app.py

**Files:**
- Modify: `app.py:60`
- Modify: `app.py:41`

**Issue:** Line 60 has a default value of "large" in `os.getenv()` which overrides missing env vars. The startup log shows "Loading Whisper model: base" but code says "large" - this suggests docker-compose or another config is setting it.

**Step 1: Update default model in app.py**

Change line 60 from:
```python
MODEL_SIZE = os.getenv("WHISPER_MODEL", "large")  # Default to large model (Ivrit model requires different approach)
```

To:
```python
MODEL_SIZE = os.getenv("WHISPER_MODEL", "ivrit-large-v3-turbo")  # Default to Ivrit Hebrew model
```

**Step 2: Update startup log message**

The log currently says "Loading default Whisper model" which is unclear about which model type. Change line 41 from:
```python
logger.info(f"Loading default Whisper model: {MODEL_SIZE}")
```

To:
```python
logger.info(f"Loading Whisper model: {MODEL_SIZE}")
```

**Step 3: Verify the changes**

Run: `grep -n "MODEL_SIZE = os.getenv" app.py`
Expected: Should show the updated line with "ivrit-large-v3-turbo"

**Step 4: Commit the changes**

```bash
git add app.py
git commit -m "fix: set default model to ivrit-large-v3-turbo in app.py"
```

---

## Task 3: Update docker-compose.yml Configuration

**Files:**
- Modify: `docker-compose.yml` (need to verify if it exists and what it sets)

**Step 1: Check if docker-compose.yml exists**

Run: `cat docker-compose.yml 2>/dev/null || echo "File not found"`

**Step 2: If docker-compose.yml exists, update WHISPER_MODEL**

Look for environment section and ensure it sets:
```yaml
environment:
  - WHISPER_MODEL=ivrit-large-v3-turbo
```

Or remove the WHISPER_MODEL override entirely to use the default from app.py.

**Step 3: Verify docker-compose configuration**

Run: `grep -A 5 "environment:" docker-compose.yml`
Expected: Should show WHISPER_MODEL set to ivrit-large-v3-turbo or not set at all

**Step 4: Commit if changed**

```bash
git add docker-compose.yml
git commit -m "config: update docker-compose to use ivrit model"
```

---

## Task 4: Verify whisper-cpp-python Installation

**Files:**
- Verify: `requirements.txt:9`

**Step 1: Check whisper-cpp-python is in requirements**

Run: `grep whispercpp requirements.txt`
Expected output: `whispercpp==0.0.17`

**Step 2: Verify the library can load GGML models**

Create a test script to verify the library works:
```bash
python3 << 'EOF'
try:
    from whispercpp import Whisper as WhisperCpp
    print("✓ whispercpp import successful")
    import os
    model_path = "models/ivrit-whisper-large-v3-turbo.bin"
    if os.path.exists(model_path):
        print(f"✓ Model file exists: {model_path}")
        file_size_mb = os.path.getsize(model_path) / (1024*1024)
        print(f"✓ Model size: {file_size_mb:.1f} MB")
    else:
        print(f"✗ Model file not found: {model_path}")
except ImportError as e:
    print(f"✗ Failed to import whispercpp: {e}")
EOF
```

Expected output:
```
✓ whispercpp import successful
✓ Model file exists: models/ivrit-whisper-large-v3-turbo.bin
✓ Model size: 548.0 MB
```

**Step 3: No commit needed (verification only)**

---

## Task 5: Test Model Loading Locally

**Files:**
- Test: `app.py:78-103` (load_model function)

**Step 1: Create a test script to load the model**

```bash
python3 << 'EOF'
import os
import logging
logging.basicConfig(level=logging.INFO)

# Set the environment variable
os.environ['WHISPER_MODEL'] = 'ivrit-large-v3-turbo'

# Import after setting env var
from app import load_model, MODEL_CONFIGS

print(f"Model configs available: {list(MODEL_CONFIGS.keys())}")
print(f"Attempting to load: {os.environ['WHISPER_MODEL']}")

try:
    model = load_model('ivrit-large-v3-turbo')
    print("✓ Model loaded successfully!")
    print(f"✓ Model type: {type(model)}")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
EOF
```

Expected output:
```
Model configs available: ['tiny', 'base', 'small', 'medium', 'large', 'ivrit-large-v3-turbo']
Attempting to load: ivrit-large-v3-turbo
INFO:app:Loading GGML model from: /app/models/ivrit-whisper-large-v3-turbo.bin
✓ Model loaded successfully!
✓ Model type: <class 'whispercpp.Whisper'>
```

**Step 2: If errors occur, check the model path**

The config in `app.py:69` specifies:
```python
"ivrit-large-v3-turbo": {"type": "ggml", "path": "/app/models/ivrit-whisper-large-v3-turbo.bin"}
```

This is the Docker container path. For local testing, the path should be relative:
```python
"ivrit-large-v3-turbo": {"type": "ggml", "path": "models/ivrit-whisper-large-v3-turbo.bin"}
```

**Step 3: Update the model path to work both locally and in Docker**

Modify `app.py:69`:
```python
"ivrit-large-v3-turbo": {"type": "ggml", "path": os.getenv("IVRIT_MODEL_PATH", "models/ivrit-whisper-large-v3-turbo.bin")}
```

And update `Dockerfile:35-36` to add:
```dockerfile
ENV WHISPER_MODEL=ivrit-large-v3-turbo
ENV IVRIT_MODEL_PATH=/app/models/ivrit-whisper-large-v3-turbo.bin
```

**Step 4: Commit the path fix**

```bash
git add app.py Dockerfile
git commit -m "fix: use configurable path for ivrit model (local + docker)"
```

---

## Task 6: Rebuild and Test Docker Container

**Files:**
- Test: Docker container startup

**Step 1: Rebuild the Docker image**

Run: `docker-compose build`
Expected: Build should complete without errors

**Step 2: Start the container**

Run: `docker-compose up`
Expected startup logs:
```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
2025-10-27 XX:XX:XX,XXX - app - INFO - Loading Whisper model: ivrit-large-v3-turbo
2025-10-27 XX:XX:XX,XXX - app - INFO - Loading GGML model from: /app/models/ivrit-whisper-large-v3-turbo.bin
2025-10-27 XX:XX:XX,XXX - app - INFO - Default Whisper model loaded successfully
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8009 (Press CTRL+C to quit)
```

**Step 3: Verify the model is actually the Ivrit model**

Check the logs for "Loading GGML model" instead of "Loading OpenAI Whisper model"

**Step 4: Test transcription with Hebrew audio (optional but recommended)**

Send a test WebSocket message with a Hebrew audio stream URL to verify the model works correctly.

**Step 5: No commit (verification only)**

---

## Task 7: Update Documentation

**Files:**
- Modify: `.env:1-8` (add comment about Ivrit model)
- Modify: `README.md` or `QUICKSTART.md` (if they mention model selection)

**Step 1: Add helpful comment to .env**

Update `.env` to document the Ivrit model:
```bash
# Whisper Model Configuration
# Options: tiny, base, small, medium, large, ivrit-large-v3-turbo
#
# OpenAI Models:
#   tiny   - Fastest, least accurate (~1GB RAM)
#   base   - Good balance (~1GB RAM)
#   small  - Better accuracy (~2GB RAM)
#   medium - High accuracy (~5GB RAM)
#   large  - Best accuracy (~10GB RAM)
#
# GGML Models:
#   ivrit-large-v3-turbo - Hebrew-optimized Whisper model (~548MB)
WHISPER_MODEL=ivrit-large-v3-turbo

# Server Configuration
PORT=8009
```

**Step 2: Verify the change**

Run: `head -20 .env`
Expected: Should show the updated documentation

**Step 3: Commit the documentation**

```bash
git add .env
git commit -m "docs: add ivrit model to environment variable documentation"
```

---

## Task 8: Final Verification

**Files:**
- Verify: Full application stack

**Step 1: Stop any running containers**

Run: `docker-compose down`

**Step 2: Clean rebuild**

Run: `docker-compose build --no-cache`

**Step 3: Start fresh**

Run: `docker-compose up`

**Step 4: Verify startup logs show Ivrit model**

Expected log line:
```
Loading GGML model from: /app/models/ivrit-whisper-large-v3-turbo.bin
```

**Step 5: Check health endpoint**

Run: `curl http://localhost:8009/health`
Expected output should include:
```json
{
  "status": "healthy",
  "whisper_model": "ivrit-large-v3-turbo",
  "model_loaded": true
}
```

Note: The health endpoint in `app.py:341-348` currently references undefined `whisper_model` variable. This needs fixing.

**Step 6: Fix health endpoint**

Modify `app.py:341-348`:
```python
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "whisper_model": current_model_name or MODEL_SIZE,
        "model_loaded": current_model is not None
    }
```

**Step 7: Commit health endpoint fix**

```bash
git add app.py
git commit -m "fix: health endpoint to use current_model_name"
```

**Step 8: Verify health endpoint after fix**

Restart container and test:
```bash
docker-compose restart
curl http://localhost:8009/health
```

Expected:
```json
{
  "status": "healthy",
  "whisper_model": "ivrit-large-v3-turbo",
  "model_loaded": true
}
```

---

## Success Criteria

- ✅ `.env` file specifies `WHISPER_MODEL=ivrit-large-v3-turbo`
- ✅ `app.py` default model is `ivrit-large-v3-turbo`
- ✅ Docker container starts without errors
- ✅ Startup logs show: "Loading GGML model from: /app/models/ivrit-whisper-large-v3-turbo.bin"
- ✅ Health endpoint returns correct model name
- ✅ Model path works both locally and in Docker
- ✅ Documentation updated to explain Ivrit model option

## Rollback Plan

If the Ivrit model fails to load or produces poor results:

1. Revert `.env`: `WHISPER_MODEL=large`
2. Revert `app.py:60`: `MODEL_SIZE = os.getenv("WHISPER_MODEL", "large")`
3. Rebuild: `docker-compose build && docker-compose up`

All changes are configuration-only, no breaking changes to code structure.

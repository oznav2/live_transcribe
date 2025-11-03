# 🎙️ Live Audio Stream Transcription - Project Summary

## 📋 Project Overview

**Live Audio Stream Transcription** is a production-ready, real-time audio transcription application with a **newly refactored modular architecture**. It processes audio from various sources (URLs, streams, video platforms) and transcribes them using multiple AI engines. Built with FastAPI, it supports cloud-based (Deepgram Nova-3) and local (OpenAI Whisper, Faster-Whisper, Ivrit) transcription models with comprehensive progress tracking, speaker diarization, and GPU acceleration.

**Version:** 3.0 (Modular Architecture)
**Language:** Python 3.11+
**Framework:** FastAPI with WebSocket support
**Architecture:** Modular with 17 specialized modules
**Previous Monolith:** 3,618 lines → **Current Entry Point:** 42 lines (98.8% reduction!)

---

## 🏗️ Modular Architecture (v3.0)

### Architectural Transformation

| Aspect | Before (v2.0) | After (v3.0) |
|--------|---------------|--------------|
| **Structure** | Monolithic (single file) | Modular (17 modules) |
| **Main File** | 3,618 lines | 42 lines |
| **Maintainability** | Difficult | Excellent |
| **Testability** | Complex | Simple (unit testable) |
| **Scalability** | Limited | Highly scalable |
| **Code Organization** | Mixed concerns | Single responsibility |
| **Breaking Changes** | - | Zero (100% compatible) |

### Module Structure

```
webapp/
├── app.py (42 lines)           # Minimal entry point
├── api/                        # API Layer
│   ├── routes.py (220)         # REST endpoints
│   └── websocket.py (344)      # WebSocket handler
├── config/                     # Configuration
│   ├── settings.py             # Environment vars
│   ├── constants.py            # App constants
│   └── availability.py         # Library detection
├── core/                       # Core Components
│   ├── state.py                # Global state
│   └── lifespan.py (82)        # Startup/shutdown
├── models/                     # Model Management
│   └── loader.py (180)         # Thread-safe loading
├── services/                   # Business Logic
│   ├── audio_processor.py (670)    # Audio processing
│   ├── transcription.py (1208)     # Transcription
│   ├── diarization.py (241)        # Speaker identification
│   └── video_metadata.py (71)      # Metadata extraction
└── utils/                      # Utilities
    ├── validators.py           # Input validation
    ├── helpers.py              # Helper functions
    ├── websocket_helpers.py    # WebSocket utils
    └── cache.py (242)          # Cache management
```

---

## 🔄 System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│  (Real-time progress, transcription view, model selection)  │
└──────────────────────┬──────────────────────────────────────┘
                       │ WebSocket (ws://...)
┌──────────────────────▼──────────────────────────────────────┐
│                   FastAPI Application                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │             api/websocket.py (344 lines)              │ │
│  │  • WebSocket endpoint management                      │ │
│  │  • Request routing (VOD vs Live)                      │ │
│  │  • Capture mode handling                              │ │
│  │  • Progress coordination                              │ │
│  └──────────────┬─────────────────────────────────────────┘ │
│                 │                                            │
│  ┌──────────────▼────────────┐  ┌──────────────────────┐   │
│  │ services/transcription.py │  │ services/            │   │
│  │       (1208 lines)        │  │ audio_processor.py   │   │
│  │ • transcribe_incremental  │  │     (670 lines)      │   │
│  │ • transcribe_chunk        │  │ • download_audio     │   │
│  │ • transcribe_audio_stream │  │ • AudioStreamProcessor│   │
│  │ • transcribe_vod_deepgram │  │ • split_audio_chunks │   │
│  │ • transcribe_with_deepgram│  │ • FFmpeg management  │   │
│  └────────────────────────────┘  └──────────────────────┘   │
│                                                              │
│  ┌────────────────────┐  ┌───────────────┐  ┌────────────┐ │
│  │ services/          │  │ models/       │  │ utils/     │ │
│  │ diarization.py     │  │ loader.py     │  │ cache.py   │ │
│  │ • Speaker labels   │  │ • Thread-safe │  │ • SHA256   │ │
│  │ • Hebrew support   │  │ • Model cache │  │ • 24h TTL  │ │
│  │ • Overlap calc     │  │ • GPU detect  │  │ • Stats    │ │
│  └────────────────────┘  └───────────────┘  └────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. Client Request → WebSocket Connection
2. URL Analysis → Route Selection:
   a. YouTube/Video → yt-dlp download → Transcription
   b. Direct Stream → FFmpeg stream → Real-time transcription
   c. Deepgram → VOD or Live API selection
3. Model Loading → Thread-safe cache check → Load if needed
4. Audio Processing:
   a. Download with progress tracking
   b. Split into chunks if needed
   c. Cache normalized audio
5. Transcription:
   a. Incremental output for long files
   b. Real-time streaming for live content
   c. Speaker diarization if enabled
6. Response → WebSocket messages → UI updates
```

---

## 📚 Core Modules Documentation

### 1. **app.py** - Entry Point (42 lines)

Minimal FastAPI application that:
- Initializes FastAPI with lifespan handler
- Mounts static files directory
- Includes API router
- Registers WebSocket endpoint
- Starts Uvicorn server

### 2. **api/websocket.py** - WebSocket Handler (344 lines)

Main transcription workflow coordinator:

```python
async def websocket_transcribe(websocket: WebSocket):
    # Handles complete transcription workflow
    # Routes to appropriate service based on:
    # - Model selection (Deepgram/Whisper/Ivrit)
    # - URL type (VOD/Live stream)
    # - Capture mode (full/first60)
    # - Diarization enabled/disabled
```

**Key Features:**
- Capture mode for 60-second previews
- Automatic VOD vs live stream detection
- Progress reporting coordination
- Error handling and cleanup

### 3. **services/transcription.py** - Transcription Services (1208 lines)

Core transcription logic for all models:

#### Key Functions:

**`transcribe_with_incremental_output()`** (351 lines)
- Handles long audio with progress tracking
- Splits into chunks for incremental output
- Nested helpers for model-specific logic
- Progress calculation and ETA estimation

**`transcribe_audio_stream()`**
- Real-time streaming transcription
- Audio queue management
- Cache integration
- FFmpeg normalization

**`transcribe_vod_with_deepgram()`**
- Deepgram pre-recorded API
- Automatic URL vs file detection
- Chunk-based response handling

**`transcribe_with_deepgram()`**
- Live streaming with Deepgram
- WebSocket connection management
- Critical: `asyncio.run_coroutine_threadsafe()` for callbacks

### 4. **services/audio_processor.py** - Audio Processing (670 lines)

Handles all audio download and processing:

#### Key Components:

**`AudioStreamProcessor` Class**
```python
class AudioStreamProcessor:
    def __init__(self, url, language, model_name)
    def start_ffmpeg_stream()  # Start FFmpeg subprocess
    def read_audio_chunks()     # Read with overlap
    def stop()                  # Cleanup
```

**`download_audio_with_ffmpeg()`**
- Async download with progress monitoring
- Nested `monitor_progress()` helper
- Error detection (403, 404, 410)

**`split_audio_for_incremental()`**
- Intelligent chunking with overlap
- Preserves context between chunks

### 5. **services/diarization.py** - Speaker Diarization (241 lines)

Speaker identification and labeling:

```python
async def transcribe_with_diarization():
    # 1. Run pyannote diarization pipeline
    # 2. Get speaker segments
    # 3. Transcribe with timestamps
    # 4. Align transcription with speakers
    # 5. Return labeled segments
```

**Features:**
- Hebrew speaker labels (דובר_1, דובר_2) for Ivrit models
- English labels (SPEAKER_1, SPEAKER_2) for others
- Overlap calculation for accurate attribution
- Fallback to regular transcription if diarization fails

### 6. **models/loader.py** - Model Management (180 lines)

Thread-safe model loading with caching:

```python
def load_model(model_name: str):
    global whisper_models, current_model
    
    # Double-check locking pattern
    if model_name in whisper_models:
        return whisper_models[model_name]
    
    with model_lock:
        if model_name not in whisper_models:
            # Load model based on type
            # Cache for future use
            whisper_models[model_name] = model
```

**Supported Models:**
- OpenAI Whisper (tiny, base, small, medium, large)
- Faster-Whisper (all sizes + Ivrit models)
- Whisper.cpp GGML models
- Deepgram (cloud-based)

### 7. **utils/cache.py** - Cache Management (242 lines)

Comprehensive caching system:

```python
# Audio chunk caching
generate_cache_key(audio_data, sample_rate, channels)  # SHA256
get_cached_audio(cache_key)                           # Retrieve
save_to_cache(cache_key, audio_path)                  # Store

# Download caching
get_cached_download(url_hash)                         # Check cache
save_download_to_cache(url_hash, file_path)          # Save result

# Management
init_cache_dir()                                      # Setup
clear_cache()                                         # Cleanup
get_cache_stats()                                     # Statistics
```

### 8. **core/state.py** - Global State Management

Centralized state management:

```python
# Model caches
whisper_models: Dict[str, Any] = {}
current_model: Optional[Any] = None

# Thread safety
model_lock: threading.Lock = threading.Lock()
diarization_pipeline_lock: threading.Lock = threading.Lock()

# Runtime state
cached_index_html: Optional[str] = None
CAPTURES: Dict[str, Any] = {}
URL_DOWNLOADS: Dict[str, Any] = {}
```

---

## 🔧 Technical Implementation Details

### Thread Safety

The application uses several patterns for thread safety:

1. **Double-Check Locking** (Model Loading)
```python
if model_name not in cache:
    with lock:
        if model_name not in cache:
            cache[model_name] = load()
```

2. **Thread Pool Executor** (CPU-bound tasks)
```python
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(task) for task in tasks]
```

3. **Async Context Manager** (Resource management)
```python
async with aiofiles.open() as f:
    content = await f.read()
```

### Async Patterns

1. **Subprocess Management**
```python
process = await asyncio.create_subprocess_exec(
    *cmd,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE
)
stdout, stderr = await process.communicate()
```

2. **WebSocket State Management**
```python
if websocket.client_state == WebSocketState.CONNECTED:
    await websocket.send_json(data)
```

3. **Deepgram Callback Threading**
```python
# Critical: Run coroutine in main event loop from callback thread
asyncio.run_coroutine_threadsafe(
    websocket.send_json(data),
    loop
)
```

### Performance Optimizations

1. **Caching Strategy**
   - SHA256-based deduplication
   - 24-hour TTL with automatic cleanup
   - In-memory model caching
   - Download result caching

2. **Chunk Processing**
   - 60-second chunks with 5-second overlap
   - Parallel processing option
   - Incremental output for user feedback

3. **Progress Tracking**
   - Real-time download progress
   - Transcription progress with ETA
   - Multi-stage indicators

---

## 🚀 API Reference

### REST Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Web UI interface |
| `/health` | GET | Health check with model status |
| `/api/video-info` | POST | YouTube metadata extraction |
| `/gpu` | GET | GPU diagnostics |
| `/api/cache/stats` | GET | Cache statistics |
| `/api/cache/clear` | POST | Clear cache |
| `/api/download-cache/stats` | GET | Download cache stats |
| `/api/download-cache/clear` | POST | Clear downloads |

### WebSocket Protocol

**Endpoint:** `ws://localhost:8009/ws/transcribe`

**Request Format:**
```json
{
    "url": "https://example.com/video.mp4",
    "model": "whisper-v3-turbo",
    "language": "auto",
    "diarization": false,
    "captureMode": "full"
}
```

**Response Types:**
```json
// Status Update
{
    "type": "status",
    "message": "Downloading audio..."
}

// Progress Update
{
    "type": "transcription_progress",
    "percentage": 45,
    "eta_seconds": 30,
    "speed": "2.5x",
    "audio_duration": 180
}

// Transcription Chunk
{
    "type": "transcription_chunk",
    "text": "Transcribed text...",
    "chunk_index": 0,
    "total_chunks": 10,
    "is_final": false
}

// Completion
{
    "type": "complete",
    "message": "Transcription complete"
}

// Error
{
    "type": "error",
    "error": "Error message"
}
```

---

## 🐳 Docker Configuration

### Build Configurations

1. **Standard Build** (`Dockerfile`)
   - Base Python 3.11
   - FFmpeg and system dependencies
   - PyTorch with CUDA 11.8
   - All Python requirements

2. **Ivrit Build** (`Dockerfile.ivrit`)
   - PyTorch 2.4.1 with CUDA 12.1
   - Faster-Whisper optimized
   - Hebrew model pre-download
   - Diarization support

### Docker Compose

```yaml
services:
  app:
    build: .
    ports:
      - "8009:8009"
    environment:
      - WHISPER_MODEL=whisper-v3-turbo
      - DEEPGRAM_API_KEY=${DEEPGRAM_API_KEY}
    volumes:
      - ./cache:/app/cache
      - ./models:/app/models
```

---

## 📈 Performance Metrics

### Processing Speed

| Model | Speed | RTF* | Accuracy |
|-------|-------|------|----------|
| Deepgram Nova-3 | <100ms latency | 0.05x | 95%+ |
| Whisper V3 Turbo | Fast | 0.3x | 93% |
| Whisper Large | Slow | 2-3x | 95% |
| Faster-Whisper Large | Medium | 0.5-1x | 95% |

*RTF = Real-Time Factor (lower is better)

### Resource Usage

| Component | CPU | RAM | GPU |
|-----------|-----|-----|-----|
| Idle | 1% | 200MB | 0% |
| Downloading | 5% | 300MB | 0% |
| Transcribing (CPU) | 80-100% | 2-10GB | 0% |
| Transcribing (GPU) | 20% | 2-10GB | 60-80% |

---

## 🔍 Monitoring & Debugging

### Logging

Comprehensive logging throughout:
```python
logger.info(f"Starting transcription for {url}")
logger.warning(f"Cache miss for {cache_key}")
logger.error(f"Download failed: {error}")
```

### Health Checks

```bash
# Check application health
curl http://localhost:8009/health

# Check GPU status
curl http://localhost:8009/gpu

# Check cache status
curl http://localhost:8009/api/cache/stats
```

### Common Issues & Solutions

1. **Memory Issues**
   - Use smaller models
   - Enable chunk processing
   - Increase Docker memory limit

2. **Slow Performance**
   - Enable GPU acceleration
   - Use Faster-Whisper instead of OpenAI Whisper
   - Reduce chunk overlap

3. **Network Issues**
   - Check URL accessibility
   - Verify Deepgram API key
   - Check firewall settings

---

## 🚦 Future Roadmap

### Planned Features

- [ ] Database integration for transcription history
- [ ] User authentication and multi-tenancy
- [ ] Real-time translation
- [ ] Advanced speaker diarization with names
- [ ] Subtitle generation (SRT, VTT)
- [ ] Batch processing API
- [ ] Kubernetes deployment manifests
- [ ] Prometheus metrics integration

### Architecture Improvements

- [ ] Message queue for job processing
- [ ] Microservices separation
- [ ] GraphQL API option
- [ ] Redis caching layer
- [ ] S3 storage integration

---

## 📝 Conclusion

The Live Audio Stream Transcription application represents a production-ready solution with:

- **Modular Architecture**: Clean, maintainable, and scalable
- **Multiple Transcription Engines**: Flexibility for different use cases
- **Real-time Processing**: Live streaming and progress tracking
- **Production Features**: Caching, error handling, monitoring
- **Zero Breaking Changes**: Smooth migration from monolithic to modular

The transformation from a 3,618-line monolith to a 42-line entry point with 17 specialized modules demonstrates the power of proper architectural design and separation of concerns.

---

**Last Updated:** November 2024
**Version:** 3.0 (Modular Architecture)
**Maintainers:** Development Team
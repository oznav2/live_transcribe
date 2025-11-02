# Live Transcription Service - Project Summary

**Version:** 2.0  
**Last Updated:** 2025-11-02  
**Total Lines of Code:** 3,618 (app.py)  
**Status:** Production Ready

---

## 🎯 Overview

A high-performance, real-time audio transcription service built with FastAPI and Whisper models. Supports live streaming, VOD transcription, speaker diarization, and YouTube video metadata extraction. Optimized for Hebrew (Ivrit) and multilingual content with bilingual UI support.

---

## 🏗️ Architecture

### Technology Stack

**Backend:**
- **Framework:** FastAPI 0.109.0 with WebSocket support
- **Transcription:** 
  - faster-whisper (primary, CT2 models)
  - openai-whisper (fallback)
  - Deepgram SDK 5.2.0 (cloud option)
- **Audio Processing:** FFmpeg, PyAudio, yt-dlp 2024.10.7
- **Speaker Diarization:** pyannote.audio 3.x (optional)
- **Async:** asyncio with ThreadPoolExecutor
- **Python Version:** 3.11

**Frontend:**
- **UI Framework:** Vanilla JavaScript with Tailwind CSS
- **WebSocket:** Real-time bidirectional communication
- **RTL Support:** Hebrew (right-to-left) and English (left-to-right)
- **Responsive:** Mobile-friendly design

**Infrastructure:**
- **Containerization:** Docker with multi-stage builds
- **GPU Support:** CUDA 11.8/12.1 for accelerated transcription
- **Caching:** Multi-level caching (audio, downloads, models)
- **Health Checks:** Built-in health and diagnostics endpoints

---

## 🎨 Key Features

### 1. **Multi-Source Audio Input**
- ✅ Live audio streams (HLS, m3u8)
- ✅ YouTube videos (all formats)
- ✅ Direct video/audio URLs
- ✅ Vimeo, Dailymotion, Twitter, etc.
- ✅ First 60-second capture mode

### 2. **Transcription Models**
- **Ivrit CT2:** `ivrit-ai/whisper-large-v3-turbo-ct2` (Hebrew-optimized)
- **Whisper V3 Turbo:** `large-v3-turbo` (multilingual)
- **Deepgram Nova-2:** Cloud-based (best for English)
- **Model Selection:** Runtime model switching

### 3. **Speaker Diarization**
- ✅ Automatic speaker detection
- ✅ Speaker labeling (SPEAKER_1, SPEAKER_2, etc.)
- ✅ Hebrew labels (דובר_1, דובר_2) for Ivrit models
- ✅ Timestamp-aligned speaker attribution
- ✅ Pyannote.audio integration

### 4. **YouTube Video Metadata**
- ✅ Auto-detection of YouTube URLs
- ✅ Video title, channel, duration, views
- ✅ Thumbnail display
- ✅ Bilingual UI (Hebrew RTL / English LTR)
- ✅ Dynamic language switching
- ✅ Debounced API calls (500ms)

### 5. **Real-Time Progress Tracking**
- ✅ Download progress with ETA
- ✅ Transcription progress by chunk
- ✅ 100% completion guarantee
- ✅ Speed metrics (MB/s, chunks/s)
- ✅ Non-blocking async updates

### 6. **Caching System**
- **Audio Cache:** SHA256-based deduplication
- **Download Cache:** URL-based with 1-hour TTL
- **Model Cache:** In-memory loaded models
- **HTML Cache:** Static assets cached at startup

### 7. **Bilingual Interface**
- **Hebrew Mode:** RTL layout, Hebrew labels
- **English Mode:** LTR layout, English labels
- **Auto-detection:** Based on model/language selection
- **Dynamic Switching:** Instant language changes

---

## 📂 Project Structure

```
/home/user/webapp/
├── app.py                          # Main application (3,618 lines)
├── static/
│   └── index.html                  # Web UI with bilingual support
├── requirements.txt                # Python dependencies
├── requirements.ivrit.txt          # Ivrit-specific dependencies
├── Dockerfile                      # Standard Docker build
├── Dockerfile.ivrit                # Ivrit-optimized Docker build
├── .env                            # Environment variables (not in repo)
├── cache/                          # Runtime cache directories
│   ├── audio/                      # Audio deduplication cache
│   ├── downloads/                  # URL download cache
│   └── captures/                   # First-60s captures
├── logs/                           # Application logs
└── docs/                           # Documentation
    ├── PROJECT_SUMMARY.md          # This file
    ├── API.md                      # API documentation
    ├── APP_PY_ANALYSIS.md          # Function-level analysis
    ├── FEATURE_VIDEO_METADATA.md   # Video metadata feature docs
    ├── FIXES_*.md                  # Bug fix documentation
    └── plans/                      # Implementation plans
```

---

## 🔧 Core Components

### 1. **Application Lifecycle**
- **Startup:** `lifespan()` async context manager
  - Load default model
  - Initialize cache directories
  - Cache index.html
  - Setup logging
- **Runtime:** FastAPI with uvicorn ASGI server
- **Shutdown:** Graceful cleanup

### 2. **Model Management**
- **Thread-Safe Loading:** Double-check locking pattern
- **Global Cache:** Single model instance per type
- **Dynamic Switching:** Load models on-demand
- **GPU/CPU Fallback:** Automatic device selection

### 3. **Audio Pipeline**
```
Input URL/Stream
    ↓
should_use_ytdlp() → Route decision
    ↓
download_with_fallback() → yt-dlp → ffmpeg
    ↓
Cached? → get_cached_download()
    ↓
split_audio_for_incremental() → Chunks
    ↓
transcribe_with_incremental_output() → Executor
    ↓
WebSocket Updates → UI
    ↓
Diarization? → transcribe_with_diarization()
    ↓
Final Results
```

### 4. **WebSocket Protocol**
- **Connection:** `/ws/transcribe`
- **Messages:**
  - `status`: Status updates
  - `download_progress`: Download metrics
  - `transcription_progress`: Transcription metrics
  - `transcription_chunk`: Incremental text
  - `cached_file`: Cache hit notification
  - `complete`: Job completion
  - `error`: Error messages

### 5. **Async Architecture**
- **Event Loop:** Non-blocking operations
- **Executors:** ThreadPoolExecutor for CPU-bound tasks
- **Subprocess:** Async subprocess for FFmpeg/yt-dlp
- **WebSocket:** Bidirectional real-time communication
- **State Management:** WebSocketState checks

---

## 🚀 Performance Optimizations

### Recent Improvements (2025-10-31)
1. ✅ Removed 146 lines of dead code
2. ✅ Fixed 5 blocking I/O operations
3. ✅ Added thread-safe model loading
4. ✅ Moved diarization to executor
5. ✅ Cached index.html at startup
6. ✅ Added WebSocket state checks
7. ✅ Real-time progress updates
8. ✅ 100% completion guarantee

### Performance Metrics
- **Model Loading:** < 5 seconds (cached)
- **Download:** Streaming with progress
- **Transcription:** Real-time with chunking
- **Diarization:** Parallel processing
- **Memory:** ~2GB (base) + model size
- **CPU:** Multi-threaded (configurable workers)
- **GPU:** CUDA acceleration supported

---

## 🔐 Security

### Input Validation
- ✅ URL format validation
- ✅ File path sanitization
- ✅ Command injection prevention
- ✅ WebSocket state verification

### Resource Limits
- ✅ Timeout protection (10s metadata, 5min download)
- ✅ File size limits
- ✅ Queue size limits
- ✅ Rate limiting (recommended, not implemented)

### Error Handling
- ✅ Graceful degradation
- ✅ Exception catching
- ✅ Detailed logging
- ✅ User-friendly error messages

---

## 🌍 Internationalization

### Supported Languages
- **Primary:** Hebrew (he)
- **Multilingual:** 50+ languages via Whisper models
- **UI Languages:** Hebrew (RTL) and English (LTR)

### Hebrew Optimization
- **Model:** Ivrit AI Whisper Large V3 Turbo CT2
- **Speaker Labels:** דובר_1, דובר_2, דובר_3...
- **RTL Support:** Full right-to-left UI
- **Font Rendering:** Hebrew-optimized fonts

---

## 📊 API Endpoints

### Public Endpoints
- `GET /` - Web UI (HTML)
- `GET /health` - Health check
- `GET /gpu` - GPU diagnostics
- `POST /api/video-info` - YouTube metadata
- `GET /api/cache/stats` - Cache statistics
- `POST /api/cache/clear` - Clear audio cache
- `GET /api/download-cache/stats` - Download cache stats
- `POST /api/download-cache/clear` - Clear download cache
- `WS /ws/transcribe` - WebSocket transcription

### WebSocket API
See [API.md](API.md) for detailed documentation.

---

## 🐳 Docker Deployment

### Two Dockerfile Variants

#### 1. **Dockerfile** (Standard)
- **Base:** Python 3.11-slim
- **Models:** openai-whisper + whisper.cpp (GGML)
- **GPU:** CUDA 11.8 support
- **Size:** ~5GB
- **Use Case:** General purpose

#### 2. **Dockerfile.ivrit** (Recommended)
- **Base:** PyTorch 2.4.1 + CUDA 12.1
- **Models:** faster-whisper (CT2) + Ivrit AI
- **Pre-cached:** Ivrit models downloaded during build
- **Size:** ~8GB
- **Use Case:** Hebrew-optimized, production

### Build & Run

```bash
# Build Ivrit-optimized image
docker build -f Dockerfile.ivrit -t transcription-ivrit .

# Run with GPU
docker run --gpus all -p 8009:8009 -e DEEPGRAM_API_KEY=xxx transcription-ivrit

# Run without GPU
docker run -p 8009:8009 -e IVRIT_DEVICE=cpu transcription-ivrit
```

---

## 🧪 Testing

### Manual Testing
- ✅ YouTube URLs (various formats)
- ✅ Live streams (HLS)
- ✅ Direct video/audio files
- ✅ Diarization (2+ speakers)
- ✅ Hebrew and English content
- ✅ Model switching
- ✅ Cache hit/miss
- ✅ Error conditions

### Automated Testing
- ⚠️ Unit tests not implemented
- ⚠️ Integration tests not implemented
- ⚠️ Load testing not performed

---

## 📝 Environment Variables

### Required
```bash
DEEPGRAM_API_KEY=xxx               # For Deepgram transcription
```

### Optional
```bash
# Model Configuration
WHISPER_MODEL=whisper-v3-turbo     # Default model
IVRIT_MODEL_NAME=ivrit-ai/...      # Ivrit model path
IVRIT_DEVICE=cuda                  # cuda or cpu
IVRIT_COMPUTE_TYPE=float16         # Model precision
IVRIT_BEAM_SIZE=5                  # Beam search size

# Caching
AUDIO_CACHE_ENABLED=true           # Enable audio cache
CACHE_MAX_SIZE_MB=1000             # Max cache size

# Performance
PARALLEL_WORKERS=4                 # Parallel chunk workers
YTDLP_CHUNK_SECONDS=60             # Chunk size

# Server
PORT=8009                          # Server port
HOST=0.0.0.0                       # Bind address
```

---

## 📚 Documentation

### Available Documents
- **PROJECT_SUMMARY.md** - This file
- **API.md** - Complete API documentation
- **APP_PY_ANALYSIS.md** - Function-level code analysis
- **FEATURE_VIDEO_METADATA.md** - Video metadata feature
- **FIXES_COMPLETED_*.md** - Bug fix documentation
- **PLAN_*.md** - Implementation plans
- **README.md** - Quick start guide
- **QUICKSTART.md** - Setup instructions
- **DEPLOYMENT.md** - Deployment guide

---

## 🔮 Roadmap

### Planned Features
- [ ] Unit test suite
- [ ] Rate limiting per IP
- [ ] User authentication
- [ ] Multi-user support
- [ ] Transcript export (SRT, VTT)
- [ ] Real-time translation
- [ ] Playlist support
- [ ] Audio quality selection

### Known Limitations
- No batch processing
- No speaker name customization
- No transcript editing
- No audio preprocessing options
- Single-instance only (no clustering)

---

## 📊 Statistics

### Code Metrics
- **Total Lines:** 3,618 (app.py)
- **Functions:** 48
- **Classes:** 3
- **API Endpoints:** 9
- **WebSocket Endpoints:** 1
- **Dependencies:** 25 packages

### Model Support
- **Whisper Models:** 8 variants
- **Ivrit Models:** 3 variants
- **Deepgram:** Nova-2, Nova-3
- **Diarization:** Pyannote 3.1

### Platform Support
- **OS:** Linux (Docker)
- **Python:** 3.11+
- **GPU:** NVIDIA CUDA 11.8+
- **RAM:** 4GB minimum, 8GB recommended
- **Storage:** 10GB for models + cache

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

Proprietary - All rights reserved

---

## 🙏 Acknowledgments

- **OpenAI Whisper** - Speech recognition models
- **Ivrit AI** - Hebrew-optimized models
- **Deepgram** - Cloud transcription API
- **Pyannote.audio** - Speaker diarization
- **FFmpeg** - Audio processing
- **yt-dlp** - Video download

---

**Maintained by:** oznav2  
**Repository:** https://github.com/oznav2/live_transcribe  
**Status:** Active Development

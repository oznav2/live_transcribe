# Comprehensive Analysis of app.py

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Dependencies and Imports](#dependencies-and-imports)
4. [Global Configuration](#global-configuration)
5. [Core Functions Analysis](#core-functions-analysis)
6. [Data Flow](#data-flow)
7. [Error Handling](#error-handling)
8. [Performance Optimizations](#performance-optimizations)
9. [Security Considerations](#security-considerations)

---

## Overview

**File**: `app.py`  
**Lines**: ~2,740  
**Functions**: 31 (13 async, 18 sync)  
**Primary Purpose**: Live audio streaming transcription service using faster_whisper models with WebSocket support

### Key Features
- Real-time audio transcription via WebSocket
- Multiple transcription backends (faster_whisper, Deepgram)
- URL-based audio download and caching
- Incremental transcription output
- Progress tracking with ETA
- Multi-format audio support

---

## Architecture

### Application Stack
```
┌─────────────────┐
│   FastAPI App   │ ← WebSocket & REST endpoints
├─────────────────┤
│ Transcription   │ ← faster_whisper (primary)
│    Engines      │ ← Deepgram API (optional)
├─────────────────┤
│ Audio Pipeline  │ ← FFmpeg, yt-dlp
├─────────────────┤
│ Caching Layer   │ ← URL cache, Audio cache
└─────────────────┘
```

### Model Hierarchy
1. **Primary**: `faster_whisper` with CT2 models (Ivrit)
2. **Optional**: Deepgram API for cloud transcription
3. **Removed**: GGML/whisper.cpp support (deprecated)

---

## Dependencies and Imports

### Critical Dependencies
```python
# Primary transcription engine
faster_whisper (REQUIRED) - CT2 optimized models
torch - GPU/CPU computation
ivrit - Hebrew language support

# Web framework
fastapi - Async web framework
uvicorn - ASGI server
websockets - Real-time communication

# Audio processing
ffmpeg-python - Audio manipulation
yt-dlp - YouTube/video downloading
pydub - Audio format conversion

# Optional
deepgram-sdk - Cloud transcription API
```

### Import Order and Availability Checks
1. **faster_whisper** - Primary, checked first (FASTER_WHISPER_AVAILABLE)
2. **whisper** - Legacy, not recommended (OPENAI_WHISPER_AVAILABLE)
3. **ivrit** - Hebrew support package (IVRIT_PACKAGE_AVAILABLE)
4. **deepgram** - Cloud API (DEEPGRAM_AVAILABLE)

**Important**: The app prioritizes faster_whisper and warns if openai-whisper is also installed due to conflicts.

---

## Global Configuration

### Model Configuration
```python
MODEL_SIZE = os.getenv("WHISPER_MODEL", "ivrit-ct2")  # Default model
MODEL_CONFIGS = {}  # Dynamically built based on available libraries
```

### Audio Processing Parameters
```python
CHUNK_DURATION = 5        # seconds - for real-time processing
CHUNK_OVERLAP = 1         # seconds - overlap between chunks
SAMPLE_RATE = 16000      # Hz - Whisper standard
CHANNELS = 1             # Mono audio
AUDIO_QUEUE_SIZE = 200   # Queue size for streaming
```

### Caching Configuration
```python
CACHE_DIR = Path("cache/audio")          # Normalized audio cache
DOWNLOAD_CACHE_DIR = Path("cache/downloads")  # URL download cache
CAPTURE_DIR = Path("cache/captures")     # First-60s captures
CACHE_MAX_AGE_HOURS = 24                # Cache expiration
```

---

## Core Functions Analysis

### 1. Application Lifecycle

#### `lifespan(app: FastAPI)` [Lines 122-179]
- **Type**: Async context manager
- **Purpose**: Initialize app resources at startup
- **Key Actions**:
  1. Checks for critical dependencies (faster_whisper)
  2. Attempts to load default model
  3. Initializes cache directories
  4. Falls back gracefully if model loading fails
- **Error Handling**: Logs warnings but doesn't crash if models unavailable

### 2. Cache Management Functions

#### `init_capture_dir()` [Lines 282-288]
- **Purpose**: Create directory for first-60s captures
- **Error Handling**: Logs warning on failure

#### `init_download_cache_dir()` [Lines 289-309]
- **Purpose**: Initialize download cache and clean old files
- **Cleanup**: Removes files older than 24 hours
- **Dictionary Management**: Updates URL_DOWNLOADS mapping

#### `get_url_hash(url: str)` [Lines 310-314]
- **Purpose**: Generate unique 16-char hash for URL caching
- **Algorithm**: SHA256 truncated to 16 chars

#### `get_cached_download(url: str)` [Lines 315-339]
- **Purpose**: Check for cached audio downloads
- **Lookup Order**:
  1. In-memory cache (URL_DOWNLOADS dict)
  2. Disk cache (DOWNLOAD_CACHE_DIR)
- **Returns**: Path to cached file or None

#### `save_download_to_cache(url: str, audio_file: str)` [Lines 340-374]
- **Purpose**: Save downloaded audio to cache
- **Features**:
  - Timestamp-based naming
  - Automatic cleanup of temp files
  - Updates in-memory mapping

### 3. Model Management

#### `load_model(model_name: str)` [Lines 375-468]
- **Purpose**: Load and configure transcription models
- **Model Types Handled**:
  1. `faster_whisper` - Primary, with GPU/CPU fallback
  2. `openai` - Legacy support (not recommended)
  3. `deepgram` - API-based (no local loading)
- **Caching**: Reuses loaded models (current_model)
- **Error Handling**: 
  - Falls back to CPU if GPU fails
  - Provides helpful error messages with installation instructions
- **Return**: Model object (dict-wrapped for faster_whisper)

### 4. Audio Download Functions

#### `should_use_ytdlp(url: str)` [Lines 469-501]
- **Purpose**: Determine if URL requires yt-dlp
- **Supported Platforms**: YouTube, Vimeo, Facebook, Twitter, etc.
- **Pattern Matching**: Regex for various YouTube URL formats

#### `download_audio_with_ffmpeg(url: str, ...)` [Lines 502-764]
- **Type**: Async
- **Purpose**: Download audio using FFmpeg directly
- **Features**:
  - URL caching support
  - Progress monitoring via WebSocket
  - Bandwidth calculation
  - ETA estimation
  - Loudness normalization
- **Parameters**:
  - `format`: 'wav' or 'm4a'
  - `duration`: 0 for complete file
  - `use_cache`: Enable URL-based caching
- **Progress Updates**: Sends detailed progress via WebSocket

#### `download_audio_with_ytdlp(url: str, ...)` [Lines 765-848]
- **Purpose**: Download using yt-dlp for complex URLs
- **Features**:
  - Cookie support
  - Format selection
  - Audio extraction
  - Caching support
- **Error Handling**: Falls back to direct FFmpeg on failure

### 5. Audio Processing Functions

#### `get_audio_duration_seconds(audio_path: str)` [Lines 849-871]
- **Purpose**: Extract audio duration using ffprobe
- **Returns**: Duration in seconds or None
- **Error Handling**: Returns None on failure

#### `calculate_progress_metrics(...)` [Lines 872-919]
- **Purpose**: Calculate transcription progress metrics
- **Returns**: Dictionary with:
  - `percentage`: 0-100
  - `eta_seconds`: Estimated time remaining
  - `eta_formatted`: Human-readable ETA
  - `speed`: Processing speed multiplier

#### `split_audio_for_incremental(...)` [Lines 920-966]
- **Purpose**: Split audio for incremental transcription
- **Method**: Uses FFmpeg segment muxer
- **Returns**: Tuple of (temp_dir, chunk_paths)
- **Overlap**: Configurable overlap between chunks

#### `split_audio_into_chunks(...)` [Lines 967-1014]
- **Purpose**: Alternative chunking method
- **Returns**: List of (timestamp, chunk_path) tuples
- **Use Case**: Parallel transcription

### 6. Transcription Functions

#### `transcribe_with_incremental_output(...)` [Lines 1015-1276]
- **Type**: Async
- **Purpose**: Main transcription with incremental output
- **Supported Models**: 
  - faster_whisper (primary)
  - openai whisper (legacy)
- **Features**:
  - Progress updates with ETA
  - Incremental chunk processing
  - Language detection
  - WebSocket updates
- **Optimization**: Processes chunks sequentially with overlap

#### `transcribe_chunk(...)` [Lines 1277-1332]
- **Purpose**: Transcribe single audio chunk
- **Model Handling**:
  - Unwraps faster_whisper dict structure
  - Handles different model types
- **Returns**: Tuple of (index, text, language)
- **Error Handling**: Returns empty text on failure

#### `transcribe_audio_stream(...)` [Lines 1513-1679]
- **Type**: Async
- **Purpose**: Real-time streaming transcription
- **Process**:
  1. Load model for processor
  2. Process audio queue
  3. Normalize audio with FFmpeg
  4. Cache normalized audio
  5. Transcribe and send results
- **Model Support**: faster_whisper, openai whisper

### 7. Deepgram Integration

#### `transcribe_vod_with_deepgram(...)` [Lines 1680-1952]
- **Type**: Async
- **Purpose**: Transcribe VOD content using Deepgram
- **Process**:
  1. Download complete audio
  2. Upload to Deepgram
  3. Stream results back
- **Features**: Progress tracking, language detection

#### `transcribe_with_deepgram(...)` [Lines 1953-2239]
- **Type**: Async
- **Purpose**: Live stream transcription with Deepgram
- **Features**: Real-time streaming, reconnection support

### 8. WebSocket Endpoints

#### `websocket_transcribe(websocket: WebSocket)` [Lines 2247-2599]
- **Type**: Async WebSocket endpoint
- **Path**: `/ws/transcribe`
- **Purpose**: Main transcription WebSocket endpoint
- **Features**:
  - Model selection
  - Capture modes (full, first60)
  - Progress updates
  - Multiple backend support
- **Process Flow**:
  1. Receive configuration
  2. Select transcription method
  3. Handle capture mode
  4. Process transcription
  5. Clean up resources

### 9. REST API Endpoints

#### `get_home()` [Lines 2240-2246]
- **Path**: `/`
- **Purpose**: Serve main HTML interface

#### `health_check()` [Lines 2600-2609]
- **Path**: `/health`
- **Purpose**: Health check endpoint
- **Returns**: Status, model info

#### `gpu_diagnostics()` [Lines 2610-2642]
- **Path**: `/gpu`
- **Purpose**: GPU information and diagnostics
- **Returns**: CUDA availability, device info

#### `cache_stats()` [Lines 2643-2662]
- **Path**: `/api/cache/stats`
- **Purpose**: Cache statistics
- **Returns**: File count, total size

#### `clear_cache()` [Lines 2663-2679]
- **Path**: `/api/cache/clear`
- **Method**: POST
- **Purpose**: Clear audio cache

#### `download_cache_stats()` [Lines 2680-2708]
- **Path**: `/api/download-cache/stats`
- **Purpose**: Download cache statistics

#### `clear_download_cache()` [Lines 2709-2740]
- **Path**: `/api/download-cache/clear`
- **Method**: POST
- **Purpose**: Clear download cache

### 10. Helper Classes

#### `TranscriptionRequest` [Lines 1396-1399]
- **Type**: Pydantic model
- **Fields**: url, language

#### `AudioStreamProcessor` [Lines 1401-1512]
- **Purpose**: Process audio streams from URLs
- **Features**:
  - FFmpeg streaming
  - Queue management
  - Thread-based processing
- **Methods**:
  - `start()`: Begin audio streaming
  - `stop()`: Stop streaming
  - `_process_audio_stream()`: Core processing loop

---

## Data Flow

### Transcription Pipeline
```
URL Input → Download (FFmpeg/yt-dlp) → Cache Check → 
Audio Processing → Chunking → Model Transcription → 
WebSocket Output → Client
```

### Caching Flow
```
URL → Hash → Check Memory Cache → Check Disk Cache → 
Download if Missing → Save to Cache → Return Path
```

### Model Loading Flow
```
Model Request → Check if Loaded → Load if Needed → 
Wrap in Dict (faster_whisper) → Cache Model → Return
```

---

## Error Handling

### Graceful Degradation
1. **Model Loading**: Falls back to CPU if GPU fails
2. **Missing Dependencies**: App starts without models, loads on-demand
3. **Download Failures**: Falls back from yt-dlp to direct FFmpeg
4. **Cache Failures**: Continues without caching

### Error Messages
- Provides installation instructions when dependencies missing
- Clear error messages via WebSocket
- Detailed logging for debugging

### WebSocket Error Handling
- Catches `WebSocketDisconnect`
- Sends error JSON messages
- Cleans up resources on disconnect

---

## Performance Optimizations

### Caching Strategy
1. **URL Cache**: Avoids re-downloading same content
2. **Audio Cache**: Caches normalized audio chunks
3. **Model Cache**: Reuses loaded models
4. **In-Memory Cache**: Fast lookups for URLs

### Async Processing
- Non-blocking I/O for downloads
- Async subprocess execution
- Concurrent chunk processing (optional)

### Memory Management
- Streaming processing for large files
- Queue-based audio streaming
- Cleanup of temporary files
- Cache expiration (24 hours)

### Model Optimizations
- GPU acceleration when available
- Optimized compute types (float16/int8)
- Batch processing disabled for real-time
- Beam size configuration

---

## Security Considerations

### Input Validation
- URL validation before download
- File size limits (implicit via streaming)
- Temporary file cleanup

### Resource Limits
- Queue size limits (200 chunks)
- Cache expiration
- Process timeouts

### External Services
- Deepgram API key handling
- Cookie support for authenticated downloads

---

## Configuration Options

### Environment Variables
```bash
# Model Configuration
WHISPER_MODEL=ivrit-ct2
IVRIT_MODEL_NAME=ivrit-ai/whisper-large-v3-turbo-ct2
IVRIT_DEVICE=cuda/cpu
IVRIT_COMPUTE_TYPE=float16/int8
IVRIT_BEAM_SIZE=5

# Caching
AUDIO_CACHE_ENABLED=true

# Processing
USE_PARALLEL_TRANSCRIPTION=false
PARALLEL_WORKERS=2

# Deepgram
DEEPGRAM_API_KEY=your_key
DEEPGRAM_MODEL=nova-3
DEEPGRAM_LANGUAGE=en-US
```

---

## Critical Notes

### Model Priority
1. **faster_whisper** is the ONLY recommended library
2. **openai-whisper** causes conflicts and should NOT be installed
3. **Ivrit models** are CT2 format for faster_whisper only

### Performance Characteristics
- faster_whisper: 2-4x faster than openai-whisper
- GPU processing: ~10x faster than CPU
- Caching: Eliminates redundant downloads
- Incremental output: Better UX for long audio

### Known Limitations
- No support for GGML models (removed)
- Requires faster_whisper for Ivrit models
- Memory usage scales with audio length
- WebSocket timeout on very long transcriptions

---

## Maintenance Notes

### Adding New Models
1. Add to MODEL_CONFIGS dictionary
2. Ensure model type handler exists
3. Update WebSocket model selection
4. Test model loading and transcription

### Debugging
- Check logs for dependency availability
- Monitor GPU memory usage
- Verify cache directory permissions
- Test WebSocket connectivity

### Future Improvements
- Implement chunked upload for large files
- Add support for more audio formats
- Implement user authentication
- Add transcription history/database

---

## Detailed Function Interactions

### Critical Function Chains

#### 1. WebSocket Transcription Flow
```python
websocket_transcribe() 
    → load_model()
    → download_audio_with_ffmpeg() / download_audio_with_ytdlp()
    → get_audio_duration_seconds()
    → transcribe_with_incremental_output()
        → split_audio_for_incremental()
        → transcribe_chunk() [multiple calls]
        → calculate_progress_metrics()
```

#### 2. Streaming Transcription Flow
```python
websocket_transcribe()
    → AudioStreamProcessor()
    → processor.start()
    → transcribe_audio_stream()
        → load_model()
        → generate_cache_key()
        → get_cached_audio() / save_to_cache()
        → model.transcribe()
```

#### 3. Cache Management Flow
```python
download_audio_with_ffmpeg()
    → get_url_hash()
    → get_cached_download()
    → save_download_to_cache()
    → URL_DOWNLOADS[hash] = path
```

### Model Handling Details

#### faster_whisper Model Structure
```python
# Model is wrapped in a dictionary for consistency
model = {
    "type": "faster_whisper",
    "model": faster_whisper.WhisperModel(...),  # Actual model
    "config": config  # Original config dict
}

# Unwrapping in transcribe functions:
if isinstance(model, dict) and model.get("type") == "faster_whisper":
    actual_model = model["model"]
```

#### Model Selection Logic
1. Check if model already loaded (`current_model_name`)
2. Validate model exists in `MODEL_CONFIGS`
3. Check dependencies (FASTER_WHISPER_AVAILABLE)
4. Load with appropriate device (GPU/CPU)
5. Cache loaded model

### WebSocket Message Protocol

#### Client → Server Messages
```json
{
    "url": "https://...",
    "language": "he",  // Optional
    "model": "ivrit-ct2",  // Default
    "captureMode": "full"  // or "first60"
}
```

#### Server → Client Messages

**Status Messages**:
```json
{
    "type": "status",
    "message": "Downloading audio..."
}
```

**Progress Messages**:
```json
{
    "type": "transcription_progress",
    "audio_duration": 300,
    "percentage": 45.5,
    "eta_seconds": 120,
    "speed": 1.5,
    "elapsed_seconds": 90
}
```

**Transcription Output**:
```json
{
    "type": "transcription",
    "text": "Transcribed text...",
    "language": "he"
}
```

**Incremental Output**:
```json
{
    "type": "transcription_chunk",
    "text": "Chunk text...",
    "chunk_index": 0,
    "total_chunks": 10,
    "is_final": false
}
```

**Error Messages**:
```json
{
    "error": "Error description..."
}
```

### Performance Metrics

#### Memory Usage Patterns
- **Model Loading**: ~2-4GB for large models
- **Audio Buffer**: AUDIO_QUEUE_SIZE * chunk_size
- **Cache**: Configurable, cleaned after 24 hours
- **FFmpeg Process**: Minimal, streaming-based

#### Processing Speed Factors
1. **Model Type**: faster_whisper > openai-whisper
2. **Device**: GPU (10x) > CPU
3. **Compute Type**: float16 > int8 (quality vs speed)
4. **Chunk Size**: Smaller = more responsive, larger = more efficient
5. **Beam Size**: Lower = faster, higher = better quality

### Error Recovery Mechanisms

#### Download Failures
1. Try yt-dlp with cookies
2. Fall back to direct FFmpeg
3. Return error to client

#### Model Loading Failures
1. Try GPU first
2. Fall back to CPU
3. Try alternative compute type
4. Load alternative model from MODEL_CONFIGS

#### Transcription Failures
1. Catch exceptions in transcribe_chunk
2. Return empty text (continue processing)
3. Log error for debugging

### Thread Safety

#### Shared Resources
- `current_model`: Protected by GIL, single assignment
- `URL_DOWNLOADS`: Dictionary updates are atomic
- `CAPTURES`: Dictionary updates for capture mode
- Cache files: Use unique names with timestamps

#### Async Considerations
- WebSocket per-connection isolation
- AudioStreamProcessor has dedicated thread
- FFmpeg subprocesses are independent
- File I/O uses unique temporary files

## Testing Considerations

### Unit Test Targets
1. `get_url_hash()` - Deterministic output
2. `calculate_progress_metrics()` - Pure function
3. `should_use_ytdlp()` - Pattern matching
4. Model unwrapping logic

### Integration Test Scenarios
1. Complete transcription flow
2. Cache hit/miss scenarios
3. Model switching
4. Error recovery paths
5. WebSocket connection handling

### Load Testing Parameters
- Concurrent WebSocket connections
- Large file processing
- Cache performance under load
- Memory usage over time

## Deployment Checklist

### Prerequisites
- [ ] faster-whisper installed
- [ ] FFmpeg available in PATH
- [ ] CUDA drivers (for GPU)
- [ ] Sufficient disk space for cache
- [ ] Model files downloaded

### Environment Setup
```bash
# Required
export WHISPER_MODEL=ivrit-ct2
export IVRIT_MODEL_NAME=ivrit-ai/whisper-large-v3-turbo-ct2

# Optional
export DEEPGRAM_API_KEY=your_key
export AUDIO_CACHE_ENABLED=true
```

### Docker Deployment
```bash
docker-compose -f docker-compose.ivrit.yml up
```

### Health Checks
- `/health` - Basic health status
- `/gpu` - GPU availability
- `/api/cache/stats` - Cache status

## Conclusion

The application is a robust, production-ready transcription service optimized for Hebrew language support using state-of-the-art CT2 models via faster_whisper. It provides real-time transcription with comprehensive caching, progress tracking, and multiple fallback options. The removal of GGML/whisper.cpp support has simplified the codebase and eliminated potential conflicts, resulting in better performance and maintainability.

### Key Strengths
1. **Performance**: Optimized for speed with faster_whisper
2. **Reliability**: Multiple fallback mechanisms
3. **Scalability**: Efficient caching and streaming
4. **Maintainability**: Clean separation of concerns
5. **User Experience**: Real-time progress and incremental output

### Architecture Highlights
- **Async-first**: Non-blocking I/O throughout
- **Cache-heavy**: Multiple caching layers
- **Fail-safe**: Graceful degradation
- **Modular**: Clear function boundaries
- **Observable**: Comprehensive logging
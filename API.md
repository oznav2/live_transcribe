# Live Transcription Service - API Documentation

**Version:** 2.0  
**Last Updated:** 2025-11-02  
**Base URL:** `http://localhost:8009`

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [REST API Endpoints](#rest-api-endpoints)
4. [WebSocket API](#websocket-api)
5. [Data Models](#data-models)
6. [Error Handling](#error-handling)
7. [Rate Limits](#rate-limits)
8. [Examples](#examples)

---

## 🎯 Overview

The Live Transcription Service provides both REST and WebSocket APIs for real-time audio transcription, speaker diarization, and YouTube video metadata extraction.

### Key Features
- Real-time transcription via WebSocket
- YouTube video metadata extraction
- Multiple model support (Whisper, Ivrit, Deepgram)
- Speaker diarization
- Progress tracking
- Caching system
- Health monitoring

### Content Types
- **Request:** `application/json`
- **Response:** `application/json`
- **WebSocket:** JSON messages

---

## 🔐 Authentication

### Deepgram API Key (Optional)
Required only for Deepgram-based transcription.

**Environment Variable:**
```bash
DEEPGRAM_API_KEY=your_api_key_here
```

**No authentication required** for other endpoints.

---

## 🌐 REST API Endpoints

### 1. Get Home Page

**Endpoint:** `GET /`  
**Description:** Serves the web UI (cached HTML)

**Response:**
- **Content-Type:** `text/html`
- **Status:** 200 OK

**Example:**
```bash
curl http://localhost:8009/
```

---

### 2. Health Check

**Endpoint:** `GET /health`  
**Description:** Returns service health status

**Response:**
```json
{
  "status": "healthy",
  "whisper_model": "whisper-v3-turbo",
  "model_loaded": true
}
```

**Status Codes:**
- `200 OK` - Service is healthy
- `500 Internal Server Error` - Service is unhealthy

**Example:**
```bash
curl http://localhost:8009/health
```

---

### 3. YouTube Video Metadata

**Endpoint:** `POST /api/video-info`  
**Description:** Fetch metadata for YouTube videos

**Request Body:**
```json
{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

**Response (Success):**
```json
{
  "success": true,
  "data": {
    "title": "Video Title",
    "channel": "Channel Name",
    "duration_seconds": 1234,
    "duration_formatted": "20:34",
    "view_count": 1234567,
    "view_count_formatted": "1.2M",
    "thumbnail": "https://i.ytimg.com/...",
    "is_youtube": true
  }
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": "Not a YouTube URL"
}
```

**Error Messages:**
- `"Invalid URL format"` - URL doesn't start with http/https
- `"Not a YouTube URL"` - URL is not from YouTube
- `"Failed to fetch video information"` - Extraction failed
- `"Internal server error"` - Server error

**Status Codes:**
- `200 OK` - Always returns 200 with success/error in body

**Supported YouTube URL Formats:**
- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://youtube.com/embed/VIDEO_ID`
- `https://youtube.com/v/VIDEO_ID`
- `https://m.youtube.com/watch?v=VIDEO_ID`

**Timeout:** 10 seconds

**Example:**
```bash
curl -X POST http://localhost:8009/api/video-info \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

---

### 4. GPU Diagnostics

**Endpoint:** `GET /gpu`  
**Description:** Returns GPU information and availability

**Response:**
```json
{
  "cuda_available": true,
  "device_count": 1,
  "cuda_version": "12.1",
  "devices": [
    {
      "index": 0,
      "name": "NVIDIA GeForce RTX 3090",
      "total_memory_mb": 24576.0,
      "multi_processor_count": 82,
      "major": 8,
      "minor": 6
    }
  ],
  "current_device": 0
}
```

**Status Codes:**
- `200 OK` - Always returns 200

**Example:**
```bash
curl http://localhost:8009/gpu
```

---

### 5. Audio Cache Statistics

**Endpoint:** `GET /api/cache/stats`  
**Description:** Returns audio cache statistics

**Response:**
```json
{
  "enabled": true,
  "cache_dir": "/app/cache/audio",
  "file_count": 42,
  "total_size_mb": 1234.56,
  "max_size_mb": 5000,
  "cache_hit_rate": 0.75
}
```

**Status Codes:**
- `200 OK` - Always returns 200

**Example:**
```bash
curl http://localhost:8009/api/cache/stats
```

---

### 6. Clear Audio Cache

**Endpoint:** `POST /api/cache/clear`  
**Description:** Clears all cached audio files

**Response:**
```json
{
  "status": "success",
  "message": "Cache cleared successfully",
  "files_deleted": 42,
  "space_freed_mb": 1234.56
}
```

**Status Codes:**
- `200 OK` - Cache cleared
- `500 Internal Server Error` - Clear failed

**Example:**
```bash
curl -X POST http://localhost:8009/api/cache/clear
```

---

### 7. Download Cache Statistics

**Endpoint:** `GET /api/download-cache/stats`  
**Description:** Returns download cache statistics

**Response:**
```json
{
  "enabled": true,
  "cache_dir": "/app/cache/downloads",
  "file_count": 15,
  "total_size_mb": 567.89,
  "cached_urls": 15,
  "oldest_entry": "2025-11-01T10:30:00Z",
  "newest_entry": "2025-11-02T14:45:00Z"
}
```

**Status Codes:**
- `200 OK` - Always returns 200

**Example:**
```bash
curl http://localhost:8009/api/download-cache/stats
```

---

### 8. Clear Download Cache

**Endpoint:** `POST /api/download-cache/clear`  
**Description:** Clears all cached downloads

**Response:**
```json
{
  "status": "success",
  "message": "Download cache cleared successfully",
  "files_deleted": 15,
  "space_freed_mb": 567.89
}
```

**Status Codes:**
- `200 OK` - Cache cleared
- `500 Internal Server Error` - Clear failed

**Example:**
```bash
curl -X POST http://localhost:8009/api/download-cache/clear
```

---

## 🔌 WebSocket API

### Transcription WebSocket

**Endpoint:** `WS /ws/transcribe`  
**Description:** Real-time audio transcription with bidirectional communication

### Connection

**Protocol:** WebSocket  
**URL:** `ws://localhost:8009/ws/transcribe`

**JavaScript Example:**
```javascript
const ws = new WebSocket('ws://localhost:8009/ws/transcribe');

ws.onopen = () => {
    console.log('Connected to transcription service');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Message:', data);
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onclose = () => {
    console.log('Connection closed');
};
```

---

### Client → Server Messages

#### 1. Start Transcription

**Message Type:** Initial connection message

**Payload:**
```json
{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "language": "he",
  "model": "ivrit-ct2",
  "captureMode": "full",
  "diarization": true
}
```

**Fields:**
- `url` (required): Audio/video URL or stream URL
- `language` (optional): Language code (e.g., "he", "en", "es")
- `model` (required): Model name
  - `"deepgram"` - Deepgram Nova-2
  - `"whisper-v3-turbo"` - Whisper Large V3 Turbo
  - `"ivrit-ct2"` - Ivrit AI Whisper V3 Turbo CT2
  - `"ivrit-v3-turbo"` - Ivrit V3 Turbo (alternative)
- `captureMode` (optional): `"full"` or `"first60"`
  - `"full"` - Transcribe entire audio
  - `"first60"` - Capture and transcribe first 60 seconds only
- `diarization` (optional): Boolean, enable speaker diarization

**Example:**
```javascript
ws.send(JSON.stringify({
    url: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
    language: 'en',
    model: 'whisper-v3-turbo',
    captureMode: 'full',
    diarization: false
}));
```

#### 2. Transcribe Captured Audio

**Message Type:** After capture_ready received

**Payload:**
```json
{
  "action": "transcribe_capture",
  "capture_id": "uuid-string",
  "language": "he",
  "model": "ivrit-ct2"
}
```

**Used in:** First-60s capture mode

---

### Server → Client Messages

#### 1. Status Update

**Type:** `status`  
**Description:** General status messages

**Payload:**
```json
{
  "type": "status",
  "message": "Starting transcription..."
}
```

**Example Messages:**
- `"Starting transcription..."`
- `"📥 Attempting download with yt-dlp..."`
- `"Analyzing speakers in audio..."`
- `"Transcribing downloaded audio..."`

---

#### 2. Download Progress

**Type:** `download_progress`  
**Description:** Real-time download progress updates

**Payload:**
```json
{
  "type": "download_progress",
  "percent": 45.3,
  "downloaded_mb": 12.34,
  "speed_mbps": 1.23,
  "eta_seconds": 30,
  "current_time": 123.45,
  "target_duration": 300
}
```

**Fields:**
- `percent`: Download percentage (0-100)
- `downloaded_mb`: Megabytes downloaded
- `speed_mbps`: Download speed in MB/s
- `eta_seconds`: Estimated time to completion
- `current_time`: Current audio time (for duration-limited downloads)
- `target_duration`: Target audio duration

**Update Frequency:** ~1 second

---

#### 3. Cached File Notification

**Type:** `cached_file`  
**Description:** Notification that cached file is being used

**Payload:**
```json
{
  "type": "cached_file",
  "message": "✅ Audio file already downloaded (123.45 MB) - Using cache, skipping download",
  "file_size_mb": 123.45,
  "skipped_download": true,
  "cached_path": "audio_abc123.wav"
}
```

**When Sent:** When URL has been previously downloaded and cached

---

#### 4. Transcription Progress

**Type:** `transcription_progress`  
**Description:** Real-time transcription progress

**Payload:**
```json
{
  "type": "transcription_progress",
  "audio_duration": 300.0,
  "percentage": 45,
  "eta_seconds": 120,
  "speed": "2.3s/chunk",
  "elapsed_seconds": 150,
  "chunks_processed": 9,
  "total_chunks": 20,
  "message": "Transcribing: 45% (chunk 9/20, ETA: 120s)"
}
```

**Fields:**
- `audio_duration`: Total audio duration in seconds
- `percentage`: Transcription percentage (0-100)
- `eta_seconds`: Estimated time to completion
- `speed`: Processing speed (e.g., "2.3s/chunk")
- `elapsed_seconds`: Time elapsed since start
- `chunks_processed`: Number of chunks processed
- `total_chunks`: Total number of chunks
- `message`: Human-readable progress message

**Update Frequency:** After each chunk

---

#### 5. Transcription Chunk

**Type:** `transcription_chunk`  
**Description:** Incremental transcription text (sent as each chunk completes)

**Payload (Without Diarization):**
```json
{
  "type": "transcription_chunk",
  "text": "This is the transcribed text from this chunk.",
  "chunk_index": 3,
  "total_chunks": 20,
  "is_final": false
}
```

**Payload (With Diarization):**
```json
{
  "type": "transcription_chunk",
  "text": "[00:12-00:15] SPEAKER_1: \"Hello, how are you?\"",
  "chunk_index": 3,
  "total_chunks": 20,
  "is_final": false,
  "speaker": "SPEAKER_1",
  "start": 12.0,
  "end": 15.0
}
```

**Hebrew Diarization:**
```json
{
  "type": "transcription_chunk",
  "text": "[00:12-00:15] דובר_1: \"שלום, מה שלומך?\"",
  "speaker": "דובר_1",
  "start": 12.0,
  "end": 15.0
}
```

**Fields:**
- `text`: Transcribed text (with timestamps if diarization enabled)
- `chunk_index`: Current chunk index (0-based)
- `total_chunks`: Total number of chunks
- `is_final`: Boolean, true if last chunk
- `speaker` (diarization only): Speaker label
- `start` (diarization only): Start time in seconds
- `end` (diarization only): End time in seconds

---

#### 6. Transcription Status

**Type:** `transcription_status`  
**Description:** Indeterminate progress with elapsed time

**Payload:**
```json
{
  "type": "transcription_status",
  "message": "Starting transcription of 300.0s audio...",
  "audio_duration": 300.0,
  "elapsed_seconds": 5
}
```

**When Sent:** At start of transcription, before chunking begins

---

#### 7. Capture Ready

**Type:** `capture_ready`  
**Description:** First 60 seconds captured, ready to transcribe

**Payload:**
```json
{
  "type": "capture_ready",
  "capture_id": "uuid-string"
}
```

**Next Step:** Client sends `transcribe_capture` message

---

#### 8. Complete

**Type:** `complete`  
**Description:** Transcription completed successfully

**Payload:**
```json
{
  "type": "complete",
  "message": "Transcription complete"
}
```

**After This:** Connection remains open, can be closed by client

---

#### 9. Error

**Type:** `error`  
**Description:** Error occurred during transcription

**Payload:**
```json
{
  "error": "Failed to download audio from URL"
}
```

**Common Error Messages:**
- `"URL is required"`
- `"Failed to load model"`
- `"Failed to download audio from URL"`
- `"All download methods failed"`
- `"Deepgram error: [details]"`
- `"Capture failed: [details]"`

**After Error:** Connection typically closes

---

### WebSocket Flow Examples

#### Example 1: Simple YouTube Transcription

```javascript
// 1. Connect
const ws = new WebSocket('ws://localhost:8009/ws/transcribe');

// 2. Send request
ws.onopen = () => {
    ws.send(JSON.stringify({
        url: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        model: 'whisper-v3-turbo'
    }));
};

// 3. Receive messages
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'status':
            console.log('Status:', data.message);
            break;
        case 'download_progress':
            console.log(`Download: ${data.percent}% (${data.speed_mbps} MB/s)`);
            break;
        case 'transcription_chunk':
            console.log('Text:', data.text);
            break;
        case 'transcription_progress':
            console.log(`Progress: ${data.percentage}%`);
            break;
        case 'complete':
            console.log('Done!');
            ws.close();
            break;
    }
};
```

#### Example 2: Hebrew with Diarization

```javascript
ws.send(JSON.stringify({
    url: 'https://example.com/hebrew-audio.mp4',
    language: 'he',
    model: 'ivrit-ct2',
    diarization: true
}));

// Receive diarized output with Hebrew speaker labels
// "[00:12-00:15] דובר_1: \"שלום\""
// "[00:15-00:18] דובר_2: \"מה נשמע?\""
```

#### Example 3: First 60 Seconds Capture

```javascript
// Send capture request
ws.send(JSON.stringify({
    url: 'https://example.com/long-video.mp4',
    model: 'whisper-v3-turbo',
    captureMode: 'first60'
}));

// Wait for capture_ready
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'capture_ready') {
        // Transcribe the captured 60 seconds
        ws.send(JSON.stringify({
            action: 'transcribe_capture',
            capture_id: data.capture_id,
            model: 'whisper-v3-turbo'
        }));
    }
};
```

---

## 📦 Data Models

### VideoInfoRequest

```typescript
{
  url: string  // YouTube video URL
}
```

### VideoInfoResponse

```typescript
{
  success: boolean
  data?: {
    title: string
    channel: string
    duration_seconds: number
    duration_formatted: string      // "MM:SS" or "HH:MM:SS"
    view_count: number
    view_count_formatted: string    // "1.2M", "45K", etc.
    thumbnail: string               // URL
    is_youtube: boolean
  }
  error?: string
}
```

### HealthResponse

```typescript
{
  status: "healthy" | "unhealthy"
  whisper_model: string
  model_loaded: boolean
}
```

### GPUResponse

```typescript
{
  cuda_available: boolean
  device_count: number
  cuda_version: string | null
  devices: Array<{
    index: number
    name: string
    total_memory_mb: number
    multi_processor_count: number
    major: number
    minor: number
  }>
  current_device: number
}
```

### CacheStatsResponse

```typescript
{
  enabled: boolean
  cache_dir: string
  file_count: number
  total_size_mb: number
  max_size_mb: number
  cache_hit_rate?: number
}
```

---

## ⚠️ Error Handling

### HTTP Error Codes

- `200 OK` - Request successful
- `400 Bad Request` - Invalid request parameters
- `404 Not Found` - Endpoint not found
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Service temporarily unavailable

### WebSocket Error Handling

**Connection Errors:**
- Network disconnection
- Server restart
- Timeout

**Processing Errors:**
- Invalid URL
- Download failure
- Model loading failure
- Transcription error

**Error Recovery:**
- Client should reconnect on disconnection
- Exponential backoff for retries
- Show user-friendly error messages

---

## 🚦 Rate Limits

**Current Implementation:** No rate limiting

**Recommendations:**
- Implement per-IP rate limiting
- Limit concurrent WebSocket connections
- Throttle API requests

**Future Implementation:**
```
- 10 requests/minute per IP (video-info)
- 5 concurrent transcriptions per IP
- 100 requests/hour per IP (cache operations)
```

---

## 📝 Examples

### Python Client Example

```python
import asyncio
import websockets
import json

async def transcribe(url, model='whisper-v3-turbo'):
    uri = 'ws://localhost:8009/ws/transcribe'
    
    async with websockets.connect(uri) as websocket:
        # Send transcription request
        await websocket.send(json.dumps({
            'url': url,
            'model': model
        }))
        
        # Receive messages
        async for message in websocket:
            data = json.loads(message)
            
            if data.get('type') == 'transcription_chunk':
                print(f"Text: {data['text']}")
            elif data.get('type') == 'complete':
                print("Transcription complete!")
                break
            elif 'error' in data:
                print(f"Error: {data['error']}")
                break

# Run
asyncio.run(transcribe('https://www.youtube.com/watch?v=VIDEO_ID'))
```

### cURL Examples

```bash
# Health check
curl http://localhost:8009/health

# Video info
curl -X POST http://localhost:8009/api/video-info \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'

# GPU diagnostics
curl http://localhost:8009/gpu

# Cache stats
curl http://localhost:8009/api/cache/stats

# Clear cache
curl -X POST http://localhost:8009/api/cache/clear
```

---

## 🔗 Additional Resources

- [Project Summary](PROJECT_SUMMARY.md)
- [Function Analysis](APP_PY_ANALYSIS.md)
- [Video Metadata Feature](FEATURE_VIDEO_METADATA.md)
- [GitHub Repository](https://github.com/oznav2/live_transcribe)

---

**API Version:** 2.0  
**Last Updated:** 2025-11-02  
**Maintained by:** oznav2

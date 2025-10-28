# API Documentation

The Live Audio Stream Transcription application provides both a WebSocket API for real-time transcription and REST API endpoints for health monitoring and cache management.

## Base URL

```
http://localhost:8009
```

## REST API Endpoints

### 1. Health Check

Check if the application is running and the Whisper model is loaded.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "whisper_model": "ivrit-large-v3-turbo",
  "model_loaded": true
}
```

**Example**:
```bash
curl http://localhost:8009/health
```

---

### 2. Cache Statistics

Get information about the audio cache system (local models only).

**Endpoint**: `GET /api/cache/stats`

**Response**:
```json
{
  "enabled": true,
  "file_count": 127,
  "total_size_mb": 45.3,
  "max_age_hours": 24
}
```

**Response (Cache Disabled)**:
```json
{
  "enabled": false
}
```

**Example**:
```bash
curl http://localhost:8009/api/cache/stats
```

**Notes**:
- Audio caching only applies to local Whisper/Ivrit models
- Deepgram transcription does not use local caching
- Cache automatically cleans files older than 24 hours

---

### 3. Clear Cache

Clear all cached audio files.

**Endpoint**: `POST /api/cache/clear`

**Response**:
```json
{
  "success": true,
  "deleted": 127
}
```

**Response (Cache Disabled)**:
```json
{
  "enabled": false
}
```

**Example**:
```bash
curl -X POST http://localhost:8009/api/cache/clear
```

---

### 4. Home Page

Serves the web interface.

**Endpoint**: `GET /`

**Response**: HTML page

---

## WebSocket API

### WebSocket Transcription

Real-time transcription via WebSocket connection with support for multiple transcription engines.

**Endpoint**: `WS /ws/transcribe`

#### Connection Flow

1. **Connect to WebSocket**:
   ```javascript
   const ws = new WebSocket('ws://localhost:8009/ws/transcribe');
   ```

2. **Send transcription request**:
   ```javascript
   ws.send(JSON.stringify({
     url: "https://example.com/audio.m3u8",  // Audio/video URL
     language: "en",                          // Optional - language code
     model: "deepgram"                        // Optional - transcription model
   }));
   ```

   **Model Options**:
   - `"deepgram"` - Deepgram Nova-2 (cloud, fastest, requires API key)
   - `"ivrit-large-v3-turbo"` - Ivrit Hebrew model (local, default)
   - `"tiny"`, `"base"`, `"small"`, `"medium"`, `"large"` - OpenAI Whisper models (local)

   **Supported URL types**:
   - **Video platforms** (auto-downloaded with yt-dlp):
     - YouTube: `https://youtube.com/watch?v=...` or `https://youtu.be/...`
     - Vimeo: `https://vimeo.com/...`
     - TikTok, Facebook, Twitter, Twitch, Dailymotion
   - **Streaming URLs** (direct FFmpeg streaming):
     - M3U8/HLS: `https://example.com/stream.m3u8`
     - HTTP/HTTPS audio/video streams
   - **Direct media files**: MP3, MP4, WAV, AAC, etc.

3. **Receive messages**:

   **Status Update**:
   ```json
   {
     "type": "status",
     "message": "Starting audio stream..."
   }
   ```

   **Transcription Result**:
   ```json
   {
     "type": "transcription",
     "text": "This is the transcribed text from the audio chunk.",
     "language": "en"
   }
   ```

   **Completion**:
   ```json
   {
     "type": "complete",
     "message": "Transcription complete"
   }
   ```

   **Error**:
   ```json
   {
     "error": "Error message describing what went wrong"
   }
   ```

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| url | string | Yes | Audio/video stream URL (m3u8, YouTube, mp4, mp3, etc.) |
| language | string | No | Language code (e.g., "en", "es", "fr", "he"). Auto-detect if omitted |
| model | string | No | Transcription model. Default: "ivrit-large-v3-turbo" |

#### Available Models

| Model | Type | Speed | Best For |
|-------|------|-------|----------|
| deepgram | Cloud | Fastest (<100ms) | Production, real-time transcription |
| ivrit-large-v3-turbo | Local | Medium | Hebrew content (default) |
| tiny | Local | Very Fast | Testing, demos |
| base | Local | Fast | General purpose |
| small | Local | Medium | Balanced accuracy/speed |
| medium | Local | Slow | High accuracy |
| large | Local | Very Slow | Maximum accuracy |

**Note**:
- Deepgram requires `DEEPGRAM_API_KEY` in environment variables
- Local models use audio caching (60% CPU reduction on repeated content)
- Cloud models (Deepgram) do not use local caching

#### Supported Language Codes

- `en` - English
- `he` - Hebrew (optimized with Ivrit model)
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ru` - Russian
- `ja` - Japanese
- `ko` - Korean
- `zh` - Chinese
- `ar` - Arabic
- `hi` - Hindi
- And 50+ more languages supported by Whisper

#### Example Usage (JavaScript)

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8009/ws/transcribe');

// Handle connection open
ws.onopen = () => {
  console.log('Connected');

  // Send transcription request with model selection
  ws.send(JSON.stringify({
    url: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',  // YouTube URL
    language: 'en',
    model: 'deepgram'  // Use Deepgram for fastest transcription
  }));
};

// Handle incoming messages
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'status') {
    console.log('Status:', data.message);
  } else if (data.type === 'transcription') {
    console.log('Transcription:', data.text);
    // Update UI with transcription
  } else if (data.type === 'complete') {
    console.log('Complete!');
  } else if (data.error) {
    console.error('Error:', data.error);
  }
};

// Handle errors
ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

// Handle connection close
ws.onclose = () => {
  console.log('Disconnected');
};
```

#### Example Usage (Python)

```python
import asyncio
import websockets
import json

async def transcribe():
    uri = "ws://localhost:8009/ws/transcribe"

    async with websockets.connect(uri) as websocket:
        # Send request with model selection
        await websocket.send(json.dumps({
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "language": "en",
            "model": "ivrit-large-v3-turbo"  # Use local Ivrit model
        }))
        
        # Receive messages
        async for message in websocket:
            data = json.loads(message)
            
            if data.get("type") == "transcription":
                print(f"Transcription: {data['text']}")
            elif data.get("type") == "complete":
                print("Complete!")
                break
            elif data.get("error"):
                print(f"Error: {data['error']}")
                break

# Run
asyncio.run(transcribe())
```

---

## Error Responses

### Common Errors

1. **Invalid URL**:
   ```json
   {
     "error": "URL is required"
   }
   ```

2. **Stream Connection Failed**:
   ```json
   {
     "error": "Failed to start audio stream"
   }
   ```

3. **Transcription Error**:
   ```json
   {
     "error": "Error transcribing chunk: [specific error]"
   }
   ```

4. **Model Not Loaded**:
   ```json
   {
     "error": "Whisper model not loaded"
   }
   ```

---

## Rate Limiting

Currently, there are no rate limits. However, only one transcription can run per WebSocket connection at a time.

---

## CORS

CORS is enabled for all origins in development. Configure appropriate CORS settings for production use.

---

## Performance Notes

### Audio Processing
- **Chunk Duration**: 5 seconds (optimized for real-time)
- **Chunk Overlap**: 1 second (prevents word cutoff at boundaries)
- **Audio Queue Size**: 50 chunks (handles bursts)
- **Processing Mode**: Asynchronous (non-blocking)

### Caching (Local Models Only)
- **Cache Type**: SHA256-based deduplication
- **Cache Location**: `cache/audio/`
- **Cache Lifetime**: 24 hours (automatic cleanup)
- **CPU Savings**: ~60% on repeated content
- **Applies To**: Whisper and Ivrit models only (not Deepgram)

### Model-Specific Behavior

**Deepgram (Cloud)**:
- Ultra-low latency (<100ms)
- No local caching needed
- Requires internet connection and API key
- Best for production real-time transcription

**Local Models (Whisper/Ivrit)**:
- Processing on local hardware
- Audio caching enabled by default
- No API key required
- Processing time varies by model size

## Connection Lifecycle

1. **WebSocket Connect**: Client connects to `/ws/transcribe`
2. **Send Request**: Client sends URL, language, and model selection
3. **Status Updates**: Server sends status messages during processing
4. **Transcription Stream**: Server sends transcription chunks as they complete
5. **Completion**: Server sends completion message
6. **Auto-Close**: Connection closes after completion or error

**Client can disconnect at any time to stop transcription**

---

## Future API Endpoints (Planned)

- `POST /api/transcribe` - REST endpoint for batch transcription
- `GET /api/status/:job_id` - Check transcription job status
- `GET /api/models` - List available models and their status
- `POST /api/upload` - Upload local audio files for transcription
- `GET /api/cache/files` - List individual cached files

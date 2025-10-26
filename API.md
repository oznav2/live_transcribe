# API Documentation

The Live Audio Stream Transcription application provides both a WebSocket API for real-time transcription and REST API endpoints.

## Base URL

```
http://localhost:8000
```

## Endpoints

### 1. Health Check

Check if the application is running and the Whisper model is loaded.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "whisper_model": "base",
  "model_loaded": true
}
```

**Example**:
```bash
curl http://localhost:8000/health
```

---

### 2. Home Page

Serves the web interface.

**Endpoint**: `GET /`

**Response**: HTML page

---

### 3. WebSocket Transcription

Real-time transcription via WebSocket connection.

**Endpoint**: `WS /ws/transcribe`

#### Connection Flow

1. **Connect to WebSocket**:
   ```javascript
   const ws = new WebSocket('ws://localhost:8000/ws/transcribe');
   ```

2. **Send transcription request**:
   ```javascript
   ws.send(JSON.stringify({
     url: "https://example.com/audio.m3u8",
     language: "en"  // Optional
   }));
   ```

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
| url | string | Yes | Audio/video stream URL (m3u8, mp4, mp3, etc.) |
| language | string | No | Language code (e.g., "en", "es", "fr"). Auto-detect if omitted |

#### Supported Language Codes

- `en` - English
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
- And 50+ more...

#### Example Usage (JavaScript)

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/transcribe');

// Handle connection open
ws.onopen = () => {
  console.log('Connected');
  
  // Send transcription request
  ws.send(JSON.stringify({
    url: 'https://example.com/video.m3u8',
    language: 'en'
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
    uri = "ws://localhost:8000/ws/transcribe"
    
    async with websockets.connect(uri) as websocket:
        # Send request
        await websocket.send(json.dumps({
            "url": "https://example.com/audio.mp3",
            "language": "en"
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

## Notes

- The WebSocket connection remains open throughout the transcription process
- Audio is processed in chunks (default: 5 seconds)
- Transcription results are sent as they become available
- The connection closes automatically when transcription is complete
- You can close the connection at any time to stop transcription

---

## Future API Endpoints (Planned)

- `POST /api/transcribe` - REST endpoint for batch transcription
- `GET /api/status/:job_id` - Check transcription job status
- `GET /api/models` - List available Whisper models
- `POST /api/upload` - Upload local audio files

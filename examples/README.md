# Example Clients

This directory contains example client implementations that demonstrate how to use the Live Audio Stream Transcription API programmatically.

## Available Examples

### 1. Python Client (`example_client.py`)

A Python client using `websockets` library.

**Installation:**
```bash
pip install websockets
```

**Usage:**
```bash
# Interactive mode
python example_client.py

# Direct URL
python example_client.py https://example.com/audio.mp3

# With language specification
python example_client.py https://example.com/audio.mp3 en
```

**Features:**
- Interactive mode with example URLs
- Command-line argument support
- Full transcription collection
- Word and segment counting
- Error handling

---

### 2. JavaScript/Node.js Client (`example_client.js`)

A Node.js client using the `ws` library.

**Installation:**
```bash
npm install ws
```

**Usage:**
```bash
# With URL
node example_client.js https://example.com/audio.mp3

# With language specification
node example_client.js https://example.com/audio.mp3 en
```

**Features:**
- Command-line interface
- Full transcription collection
- Word and segment counting
- Error handling
- Can be used as a module

---

## Example URLs for Testing

### Direct Audio Files
```
https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3
```

### HLS Streams (m3u8)
```
https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8
```

### Direct Video Files
```
https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4
```

---

## Usage Patterns

### Python

```python
import asyncio
import websockets
import json

async def transcribe(url):
    async with websockets.connect('ws://localhost:8000/ws/transcribe') as ws:
        await ws.send(json.dumps({"url": url}))
        
        async for message in ws:
            data = json.loads(message)
            if data.get("type") == "transcription":
                print(data["text"])

asyncio.run(transcribe("https://example.com/audio.mp3"))
```

### JavaScript

```javascript
const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:8000/ws/transcribe');

ws.on('open', () => {
    ws.send(JSON.stringify({ url: 'https://example.com/audio.mp3' }));
});

ws.on('message', (data) => {
    const message = JSON.parse(data);
    if (message.type === 'transcription') {
        console.log(message.text);
    }
});
```

---

## Common Use Cases

### 1. Batch Processing

Process multiple URLs sequentially:

```python
urls = [
    "https://example.com/audio1.mp3",
    "https://example.com/audio2.mp3",
    "https://example.com/audio3.mp3"
]

for url in urls:
    await transcribe_stream(url)
```

### 2. Save to File

Save transcription to a text file:

```python
async def transcribe_and_save(url, output_file):
    transcription = []
    # ... receive transcription segments ...
    
    with open(output_file, 'w') as f:
        f.write(' '.join(transcription))
```

### 3. Real-time Processing

Process transcription segments as they arrive:

```python
async def process_segment(text):
    # Analyze sentiment, extract keywords, etc.
    sentiment = analyze_sentiment(text)
    keywords = extract_keywords(text)
    # ...
```

---

## Troubleshooting

### Connection Refused

**Error:** `Connection refused` or `ECONNREFUSED`

**Solution:** Ensure the server is running:
```bash
docker-compose up
```

### WebSocket Timeout

**Error:** WebSocket connection times out

**Solution:** Check if the URL is accessible and the stream is live.

### Authentication Error

If authentication is enabled on the server, add API key:

```javascript
ws.on('open', () => {
    ws.send(JSON.stringify({
        url: 'https://example.com/audio.mp3',
        api_key: 'your-api-key'  // If authentication is enabled
    }));
});
```

---

## Next Steps

1. **Customize**: Modify the examples for your specific use case
2. **Extend**: Add features like progress bars, retry logic, etc.
3. **Integrate**: Use in your applications, scripts, or workflows
4. **Deploy**: Deploy your client application alongside the server

---

For more information, see:
- [API Documentation](../API.md)
- [Main README](../README.md)

# 🎙️ Live Audio Stream Transcription

A real-time audio transcription application that streams audio from URLs (m3u8, video links, audio files) and transcribes them live using multiple transcription engines: **Deepgram Nova-2** (cloud), **Ivrit** (Hebrew-optimized), or **OpenAI Whisper** (local). Inspired by [Vibe](https://github.com/thewh1teagle/vibe).

## ✨ Features

### Core Transcription
- **Live Streaming Transcription**: Transcribe audio as it streams, no need to download the entire file first
- **Multiple Transcription Engines**:
  - **Deepgram Nova-2**: Ultra-fast cloud-based transcription with <100ms latency
  - **Ivrit Large V3 Turbo**: Hebrew language optimized local model
  - **OpenAI Whisper**: Local transcription with multiple model sizes (tiny, base, small, medium, large)
- **YouTube & Video Platform Support**: Download and transcribe from YouTube, Vimeo, Facebook, Twitter, TikTok, and more using yt-dlp
- **Multiple Format Support**: Works with m3u8 (HLS), direct video URLs, audio files, and streaming media
- **Audio Caching**: Intelligent caching system reduces CPU usage by 60% for repeated content (local models only)

### User Experience
- **Real-time Display**: See transcription results live as they're processed
- **Language Support**: Auto-detect or manually specify from 50+ languages
- **Easy Export**: Copy to clipboard or download as text file
- **Model Selection**: Choose your transcription model from the UI
- **Beautiful Web UI**: Modern, responsive interface with real-time updates and dark theme

### Performance & DevOps
- **Extreme Performance Mode**: 5-second chunks with 1-second overlap for fast processing
- **Async Processing**: Non-blocking subprocess execution for true real-time updates
- **Cache Management**: Built-in API endpoints to monitor and manage audio cache
- **Docker Ready**: Simple deployment with Docker and docker-compose
- **Health Monitoring**: Built-in health check endpoints

## 🚀 Quick Start with Docker

### Prerequisites

- Docker and Docker Compose installed
- At least 2GB RAM (4GB+ recommended for better models)
- Internet connection (for initial model download)

### 1. Clone or Download

```bash
git clone <your-repo-url>
cd webapp
```

### 2. Quick Start Script (Recommended)

```bash
# Make start script executable (if not already)
chmod +x start.sh

# Run quick start
./start.sh
```

### 2. Manual Build and Run

```bash
# Build and start the container
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

### 3. Access the Application

Open your browser and navigate to:
```
http://localhost:8000
```

## 📚 Documentation

- **[Quick Start Guide](QUICKSTART.md)**: Fast setup guide for new users
- **[API Documentation](API.md)**: Complete WebSocket and REST API reference
- **[Deployment Guide](DEPLOYMENT.md)**: Deploy to various cloud platforms
- **[Project Summary](PROJECT_SUMMARY.md)**: Comprehensive project overview and method documentation
- **[Contributing](CONTRIBUTING.md)**: Guidelines for contributors

## 🎯 How to Use

1. **Select Model**: Choose your transcription engine:
   - **Deepgram Nova-2**: Fastest, cloud-based, best for real-time (requires API key in .env)
   - **Ivrit Large V3 Turbo**: Hebrew-optimized local model
   - **Whisper Models**: Local processing with various size/accuracy tradeoffs
2. **Enter URL**: Paste an m3u8, video, or audio URL into the input field
3. **Select Language** (Optional): Choose the audio language or leave as "Auto-detect"
4. **Click "Start Transcription"**: The application will begin streaming and transcribing
5. **Watch Live Results**: Transcription appears in real-time as audio is processed
6. **Export**: Use "Copy Text" or "Download" buttons to save your transcription

### Supported URL Types

- **Video Platforms** (automatically downloaded with yt-dlp):
  - YouTube: `https://www.youtube.com/watch?v=...` or `https://youtu.be/...`
  - Vimeo: `https://vimeo.com/...`
  - Facebook, Twitter, Twitch, TikTok, Dailymotion

- **HLS Streams (m3u8)**: Live streams and VOD (direct FFmpeg streaming)
  ```
  https://example.com/video/playlist.m3u8
  ```

- **Direct Video URLs**: MP4, WebM, etc. (direct FFmpeg streaming)
  ```
  https://example.com/video.mp4
  ```

- **Direct Audio URLs**: MP3, WAV, AAC, etc. (direct FFmpeg streaming)
  ```
  https://example.com/audio.mp3
  ```

**Note**: The application automatically detects the URL type and uses the appropriate method (yt-dlp for video platforms, FFmpeg for direct streams).

## ⚙️ Configuration

### Model Selection

Choose a transcription engine based on your needs:

| Model | Type | RAM | Speed | Accuracy | Best For |
|-------|------|-----|-------|----------|----------|
| **Deepgram Nova-2** | Cloud | N/A | **Fastest** (<100ms) | Best | Production, real-time |
| **Ivrit Large V3 Turbo** | Local | ~10GB | Medium | Best | Hebrew content |
| tiny | Local | ~1GB | Very Fast | Basic | Testing, demos |
| base | Local | ~1GB | Fast | Good | General purpose |
| small | Local | ~2GB | Medium | Better | High accuracy |
| medium | Local | ~5GB | Slow | High | Critical apps |
| large | Local | ~10GB | Very Slow | Best | Maximum accuracy |

**Note**: Audio caching (60% CPU reduction) is only available for local Whisper/Ivrit models.

### Environment Configuration

Edit `.env` file or `docker-compose.yml`:

```yaml
environment:
  - WHISPER_MODEL=ivrit-large-v3-turbo  # Default local model
  - DEEPGRAM_API_KEY=your_api_key_here  # Required for Deepgram
  - AUDIO_CACHE_ENABLED=true            # Enable audio caching (default: true)
  - PORT=8009                            # Application port
```

### Port Configuration

To use a different port, edit `docker-compose.yml`:
```yaml
ports:
  - "9000:8000"  # Access on port 9000
```

## 🛠️ Development Setup (Without Docker)

### Prerequisites

- Python 3.11+
- FFmpeg installed and in PATH
- pip

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Access at: `http://localhost:8000`

## 📁 Project Structure

```
live_transcribe/
├── app.py                  # Main FastAPI application (~800 lines)
│   ├── Model Management    # Whisper, Ivrit, Deepgram loaders
│   ├── URL Handling        # yt-dlp integration, URL detection
│   ├── Audio Caching       # SHA256-based cache system
│   ├── Stream Processing   # FFmpeg stream handling
│   ├── Transcription       # Local & cloud transcription
│   └── API Endpoints       # WebSocket & REST APIs
├── static/
│   └── index.html         # Modern web UI with dark theme
├── cache/                 # Audio cache directory (gitignored)
│   └── audio/            # Normalized audio chunks
├── examples/
│   ├── example_client.py  # Python WebSocket client
│   └── example_client.js  # Node.js WebSocket client
├── docs/
│   ├── README.md          # This file
│   ├── API.md             # API documentation
│   ├── DEPLOYMENT.md      # Deployment guide
│   ├── QUICKSTART.md      # Quick start guide
│   ├── PROJECT_SUMMARY.md # Complete project overview
│   └── CONTRIBUTING.md    # Contribution guidelines
├── requirements.txt       # Python dependencies
├── Dockerfile            # Optimized multi-stage build
├── docker-compose.yml    # Docker orchestration
├── start.sh              # Quick start script
├── test_setup.py         # Environment validator
├── .env                  # Environment variables (gitignored)
└── .gitignore            # Git ignore rules
```

## 🔧 Advanced Configuration

### Audio Caching

The application includes an intelligent caching system for local models:

```python
# In app.py or via environment variables
CACHE_ENABLED = True              # Enable/disable caching
CACHE_MAX_AGE_HOURS = 24         # Auto-cleanup old cache files
```

**Cache Management API**:
- `GET /api/cache/stats` - View cache statistics
- `POST /api/cache/clear` - Clear all cached files

**Benefits**:
- 60% CPU reduction on repeated content
- Automatic SHA256-based deduplication
- 24-hour automatic cleanup
- Only applies to local Whisper/Ivrit models (not Deepgram)

### Custom Chunk Duration

The application uses optimized chunk settings:
```python
CHUNK_DURATION = 5   # seconds - fast real-time processing
CHUNK_OVERLAP = 1    # seconds - context preservation
```

**Trade-offs**:
- Smaller chunks = faster updates, more processing overhead
- Larger chunks = slower updates, more efficient processing
- Overlap prevents word cutoff at chunk boundaries

### GPU Acceleration

For NVIDIA GPUs with CUDA support:

1. Use `nvidia/cuda` base image in Dockerfile
2. Install PyTorch with CUDA support
3. Set environment variable in docker-compose.yml:
   ```yaml
   environment:
     - CUDA_VISIBLE_DEVICES=0
   ```

## 🐛 Troubleshooting

### Model Download Issues

**Problem**: First startup is slow
**Solution**: The Whisper model downloads on first run (~150MB - 3GB depending on model). Subsequent starts are fast as models are cached.

### FFmpeg Errors

**Problem**: "FFmpeg not found"
**Solution**: Ensure FFmpeg is installed in Docker container (already included in Dockerfile)

### WebSocket Connection Failed

**Problem**: Cannot connect to transcription service
**Solution**: Check if port 8000 is available and not blocked by firewall

### High Memory Usage

**Problem**: Container uses too much RAM
**Solution**: Use a smaller Whisper model (tiny or base)

### Stream URL Not Working

**Problem**: URL doesn't stream
**Solution**: 
- Verify URL is accessible
- Check if the stream requires authentication
- Try a direct video/audio URL instead of playlist URLs

## 📊 Performance Tips

1. **Model Selection**: Start with `base` model, upgrade if needed
2. **Chunk Duration**: 5-10 seconds is optimal for most use cases
3. **Hardware**: 
   - Minimum: 2GB RAM, 2 CPU cores
   - Recommended: 4GB+ RAM, 4+ CPU cores
   - Optimal: GPU with CUDA support

## 🌍 Supported Languages

Auto-detection works for:
English, Spanish, French, German, Italian, Portuguese, Russian, Japanese, Korean, Chinese, Arabic, Hindi, and 50+ more languages.

## 📝 License

This project is open source. Feel free to use and modify as needed.

## 🙏 Credits

- **OpenAI Whisper**: State-of-the-art speech recognition model
- **FFmpeg**: Audio/video processing
- **FastAPI**: Modern Python web framework
- **Vibe**: Original inspiration for this project

## 🔗 Useful Links

- [OpenAI Whisper GitHub](https://github.com/openai/whisper)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 🚀 Deployment

### Docker Hub

```bash
# Build and tag
docker build -t yourusername/live-transcription:latest .

# Push to Docker Hub
docker push yourusername/live-transcription:latest
```

### Cloud Platforms

This application can be deployed to:
- **AWS ECS/EKS**: Using Docker container
- **Google Cloud Run**: Direct Docker deployment
- **DigitalOcean App Platform**: Docker or Dockerfile
- **Railway**: One-click Docker deployment
- **Render**: Docker-based deployment
- **Fly.io**: Dockerfile deployment

## 📈 Recent Updates (v1.1)

### Completed Features
- ✅ **yt-dlp Integration**: Download and transcribe from YouTube, Vimeo, TikTok, and other video platforms
- ✅ **Deepgram Cloud Transcription**: Ultra-fast cloud-based transcription with <100ms latency
- ✅ **Audio Caching System**: SHA256-based caching for normalized audio chunks (60% CPU reduction)
- ✅ **Cache Management API**: REST endpoints to monitor and clear cache
- ✅ **Async Subprocess Execution**: Non-blocking FFmpeg and whisper.cpp calls for true real-time updates
- ✅ **Extreme Performance Mode**: 5-second chunks with 1-second overlap and greedy decoding
- ✅ **Model Selection UI**: User-selectable transcription models (Deepgram, Whisper, Ivrit)

### Roadmap

- [ ] Support for local file uploads
- [ ] Speaker diarization (identify multiple speakers)
- [ ] Subtitle generation (SRT, VTT formats)
- [ ] Translation support
- [ ] API key authentication
- [ ] Queue system for multiple simultaneous transcriptions
- [ ] Recording capability for live streams

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

**Built with ❤️ using Python, FastAPI, Whisper AI, and FFmpeg**

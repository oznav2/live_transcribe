# 🎙️ Live Audio Stream Transcription

A real-time audio transcription application that streams audio from URLs (m3u8, video links, audio files) and transcribes them live using OpenAI's Whisper AI model. Inspired by [Vibe](https://github.com/thewh1teagle/vibe).

## ✨ Features

- **Live Streaming Transcription**: Transcribe audio as it streams, no need to download the entire file first
- **Multiple Format Support**: Works with m3u8 (HLS), direct video URLs, audio files, and more
- **Real-time Display**: See transcription results live as they're processed
- **Language Support**: Auto-detect or manually specify from 50+ languages
- **Whisper AI**: Powered by OpenAI's state-of-the-art Whisper model
- **Easy Export**: Copy to clipboard or download as text file
- **Docker Ready**: Simple deployment with Docker and docker-compose
- **Beautiful Web UI**: Modern, responsive interface with real-time updates

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

- **[API Documentation](API.md)**: Complete WebSocket and REST API reference
- **[Deployment Guide](DEPLOYMENT.md)**: Deploy to various cloud platforms
- **[Contributing](CONTRIBUTING.md)**: Guidelines for contributors

## 🎯 How to Use

1. **Enter URL**: Paste an m3u8, video, or audio URL into the input field
2. **Select Language** (Optional): Choose the audio language or leave as "Auto-detect"
3. **Click "Start Transcription"**: The application will begin streaming and transcribing
4. **Watch Live Results**: Transcription appears in real-time as audio is processed
5. **Export**: Use "Copy Text" or "Download" buttons to save your transcription

### Supported URL Types

- **HLS Streams (m3u8)**: Live streams and VOD
  ```
  https://example.com/video/playlist.m3u8
  ```
- **Direct Video URLs**: MP4, WebM, etc.
  ```
  https://example.com/video.mp4
  ```
- **Direct Audio URLs**: MP3, WAV, AAC, etc.
  ```
  https://example.com/audio.mp3
  ```
- **YouTube** (requires yt-dlp integration - see Advanced Configuration)

## ⚙️ Configuration

### Whisper Model Selection

Choose a model based on your hardware and accuracy needs:

| Model | RAM | Speed | Accuracy |
|-------|-----|-------|----------|
| tiny | ~1GB | Fastest | Basic |
| base | ~1GB | Fast | Good ⭐ (Default) |
| small | ~2GB | Medium | Better |
| medium | ~5GB | Slow | High |
| large | ~10GB | Slowest | Best |

Edit `docker-compose.yml`:
```yaml
environment:
  - WHISPER_MODEL=small  # Change to your preferred model
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
webapp/
├── app.py                 # Main FastAPI application with WebSocket support
├── static/
│   └── index.html        # Beautiful web interface with real-time updates
├── requirements.txt      # Python dependencies
├── Dockerfile           # Optimized Docker image configuration
├── docker-compose.yml   # Docker Compose setup with volume management
├── start.sh            # Quick start script
├── test_setup.py       # Environment validation script
├── .env.example        # Environment variables template
├── .gitignore          # Git ignore rules
├── README.md           # This file
├── API.md              # Complete API documentation
├── DEPLOYMENT.md       # Detailed deployment guide
└── CONTRIBUTING.md     # Contribution guidelines
```

## 🔧 Advanced Configuration

### YouTube Support

To add YouTube support, update `requirements.txt`:
```txt
yt-dlp==2023.12.30
```

And modify FFmpeg command in `app.py` to use yt-dlp for YouTube URLs.

### Custom Chunk Duration

Edit `app.py` to adjust processing chunk size:
```python
CHUNK_DURATION = 5  # seconds (default: 5)
```

Smaller chunks = faster updates but more processing overhead
Larger chunks = slower updates but more efficient

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

## 📈 Roadmap

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

# 🚀 Quick Start Guide

Get the Live Audio Stream Transcription app running in **under 5 minutes**!

## Prerequisites Check

✅ Docker installed? Run: `docker --version`  
✅ Docker Compose installed? Run: `docker-compose --version`  
✅ At least 2GB free RAM  
✅ Internet connection (for initial model download)  

---

## 🎯 3 Simple Steps

### Step 1: Get the Code

```bash
# Clone the repository
git clone <your-repo-url>
cd webapp
```

### Step 2: Start the Application

**Option A: Quick Start Script (Recommended)**
```bash
chmod +x start.sh
./start.sh
```

**Option B: Manual Start**
```bash
docker-compose up --build
```

### Step 3: Open Your Browser

Navigate to: **http://localhost:8009**

---

## 🎬 Your First Transcription

1. **Copy this test URL**:
   ```
   https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3
   ```

2. **Paste it** into the input field

3. **Click** "Start Transcription"

4. **Watch** the live transcription appear! 🎉

---

## 📊 What's Happening Behind the Scenes?

When you start the app for the first time:

1. ⬇️ **Docker downloads** the base Python image (~300MB)
2. 📦 **Installs dependencies** (FFmpeg, Python packages)
3. 🤖 **Downloads Whisper model** (~150MB for base model)
4. 🚀 **Starts the server** on port 8000

**First start**: ~5-10 minutes (depending on internet speed)  
**Subsequent starts**: ~10-30 seconds ⚡

---

## 🔧 Useful Commands

### Check if Running
```bash
curl http://localhost:8009/health
```

### View Logs
```bash
docker-compose logs -f
```

### Stop Application
```bash
docker-compose down
```

### Restart Application
```bash
docker-compose restart
```

### Stop and Remove Everything
```bash
docker-compose down -v
```

---

## 🎛️ Quick Configuration

### Use a Different Whisper Model

Edit `docker-compose.yml`:
```yaml
environment:
  - WHISPER_MODEL=small  # Options: tiny, base, small, medium, large
```

Then restart:
```bash
docker-compose down
docker-compose up --build
```

### Use a Different Port

Edit `docker-compose.yml`:
```yaml
ports:
  - "9000:8000"  # Access at http://localhost:9000
```

---

## 🧪 Test URLs

Try these sample URLs:

### Music (No Speech)
```
https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3
```

### HLS Stream (m3u8)
```
https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8
```

### Your Own URL
Any direct link to:
- `.mp3`, `.wav`, `.m4a` (audio files)
- `.mp4`, `.webm` (video files)
- `.m3u8` (HLS streams)

---

## 💡 Pro Tips

1. **Language Selection**: Auto-detect works great, but specifying the language improves accuracy

2. **Export Options**: 
   - Click "Copy Text" to copy to clipboard
   - Click "Download" to save as .txt file

3. **Multiple Transcriptions**: Stop the current one before starting a new one

4. **Performance**: 
   - `base` model: Good balance of speed and accuracy (default)
   - `small` model: Better accuracy, needs 2GB RAM
   - `tiny` model: Fastest, less accurate

---

## ❓ Troubleshooting

### "Connection Refused"
**Problem**: Can't access http://localhost:8000  
**Solution**: Check if Docker is running: `docker ps`

### "Port Already in Use"
**Problem**: Port 8000 is taken  
**Solution**: Either:
- Stop the other service using port 8000
- Change port in `docker-compose.yml`

### "Out of Memory"
**Problem**: Docker container crashes  
**Solution**: 
- Use smaller model: `WHISPER_MODEL=tiny`
- Increase Docker memory limit in Docker settings

### First Start is Very Slow
**Expected**: Model download takes time on first run  
**Solution**: Be patient! Subsequent starts are fast

---

## 📚 Next Steps

- 📖 Read the full [README.md](README.md)
- 🔌 Explore the [API Documentation](API.md)
- 🚀 Learn about [Deployment Options](DEPLOYMENT.md)
- 💻 Check out [Example Clients](examples/README.md)

---

## 🆘 Need Help?

1. Check the [README.md](README.md) for detailed documentation
2. Run the environment test: `python test_setup.py`
3. View logs: `docker-compose logs`
4. Open an issue on GitHub

---

## 🎉 Success!

If you see the web interface and can transcribe audio, you're all set!

**Congratulations on getting started with Live Audio Stream Transcription!** 🎙️

---

**Estimated time to complete**: ⏱️ **5-10 minutes** (first time)

**What you've accomplished**:
- ✅ Installed and configured the application
- ✅ Started a Docker-based transcription service
- ✅ Transcribed your first audio stream
- ✅ Learned basic commands and configuration

**You're now ready to transcribe any audio or video stream!** 🚀

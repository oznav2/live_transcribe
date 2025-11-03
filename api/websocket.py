"""WebSocket endpoint for real-time transcription."""
import asyncio
import logging
import os
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import WebSocket, WebSocketDisconnect

from config.availability import MODEL_CONFIGS, DEEPGRAM_AVAILABLE, OPENAI_WHISPER_AVAILABLE
from config.settings import (
    DEEPGRAM_API_KEY, USE_PARALLEL_TRANSCRIPTION, PARALLEL_WORKERS,
    YTDLP_CHUNK_SECONDS, YTDLP_CHUNK_OVERLAP
)
from config.constants import CAPTURE_DIR, DOWNLOAD_CACHE_DIR
from core.state import CAPTURES, current_model, current_model_name
from models.loader import load_model
from services.audio_processor import (
    AudioStreamProcessor, download_with_fallback, split_audio_into_chunks
)
from services.transcription import (
    transcribe_with_incremental_output, transcribe_chunk,
    transcribe_audio_stream, transcribe_vod_with_deepgram,
    transcribe_with_deepgram
)
from services.diarization import transcribe_with_diarization
from utils.validators import should_use_ytdlp

logger = logging.getLogger(__name__)

# Import libraries conditionally
if DEEPGRAM_AVAILABLE:
    from deepgram import DeepgramClient

if OPENAI_WHISPER_AVAILABLE:
    import torch


async def websocket_transcribe(websocket: WebSocket):
    """WebSocket endpoint for live transcription"""
    await websocket.accept()
    processor = None
    audio_thread = None

    try:
        # Receive transcription request
        data = await websocket.receive_json()
        url = data.get("url")
        language = data.get("language")
        model_name = data.get("model", "whisper-v3-turbo")  # Default to multilingual model
        capture_mode = data.get("captureMode", "full")
        enable_diarization = data.get("diarization", False)  # Speaker diarization flag

        if not url:
            await websocket.send_json({"error": "URL is required"})
            return

        logger.info(f"Starting transcription for URL: {url} with model: {model_name}")

        # Warn user if using a slow model
        slow_models = ["large", "medium"]
        if any(slow_model in model_name for slow_model in slow_models):
            await websocket.send_json({
                "type": "status",
                "message": f"⚠️ Using {model_name} - This model provides excellent quality but processes slowly. Consider using Deepgram for faster results."
            })

        # Check if user selected Deepgram
        if model_name == "deepgram":
            # For VOD content (complete videos), use pre-recorded API for better results
            # For true live streams, use streaming API
            # Detection: if URL is yt-dlp compatible or contains VOD patterns, treat as pre-recorded
            is_vod = should_use_ytdlp(url) or any(pattern in url.lower() for pattern in ['.mp4', '.mp3', '.wav', '.m4a', 'video-', '/media/'])

            if is_vod:
                logger.info(f"Detected VOD content, using pre-recorded API for better accuracy")
                await transcribe_vod_with_deepgram(websocket, url, language)
            else:
                logger.info(f"Detected live stream, using streaming API")
                await transcribe_with_deepgram(websocket, url, language)
            return

        # If capture_mode is first60, perform a 60s capture to WAV and gate transcription
        if capture_mode == "first60":
            await websocket.send_json({"type": "status", "message": "Capturing first 60 seconds (m4a via yt-dlp)..."})
            try:
                # Use yt-dlp to extract audio as m4a and limit to first 60 seconds
                capture_id = str(uuid4())
                output_path = str(CAPTURE_DIR / f"capture_{capture_id}.m4a")
                # yt-dlp progress is printed to stderr; use --newline for line-based updates
                ytdlp_cmd = [
                    'yt-dlp',
                    '--newline',
                    '--extract-audio',
                    '--audio-format', 'm4a',
                    '--audio-quality', '0',
                    '--download-sections', '*00:00-01:00',
                    '--no-playlist',
                    '--no-warnings',
                    '-o', output_path,
                    url
                ]

                proc = await asyncio.create_subprocess_exec(
                    *ytdlp_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                # Read progress lines and emit to UI
                progress_re = re.compile(r"\[download\]\s+(\d+(?:\.\d+)?)%")
                while True:
                    line = await proc.stderr.readline()
                    if not line:
                        break
                    text = line.decode(errors='ignore').strip()
                    m = progress_re.search(text)
                    if m:
                        pct = float(m.group(1))
                        try:
                            await websocket.send_json({"type": "status", "message": f"Capture progress: {pct:.1f}%"})
                        except Exception:
                            pass

                await proc.wait()
                if proc.returncode != 0 or not os.path.exists(output_path):
                    err_out = await proc.stderr.read()
                    err = err_out.decode(errors='ignore') if err_out else "Unknown yt-dlp error"
                    await websocket.send_json({"error": f"Capture failed: {err}"})
                    return

                # Store capture metadata and notify client
                CAPTURES[capture_id] = {
                    "path": output_path,
                    "created_at": datetime.utcnow().isoformat(),
                    "url": url,
                }
                await websocket.send_json({"type": "capture_ready", "capture_id": capture_id})

                # Wait for further messages to transcribe the capture
                while True:
                    try:
                        msg = await websocket.receive_json()
                    except WebSocketDisconnect:
                        return
                    action = msg.get("action")
                    if action == "transcribe_capture":
                        cid = msg.get("capture_id")
                        lang2 = msg.get("language", language)
                        model2 = msg.get("model", model_name)
                        info = CAPTURES.get(cid)
                        if not info or not os.path.exists(info["path"]):
                            await websocket.send_json({"error": "Captured file not found"})
                            return

                        await websocket.send_json({"type": "status", "message": "Transcribing captured audio..."})
                        # Route based on model
                        try:
                            model = load_model(model2)
                            model_config = MODEL_CONFIGS[model2]
                            if model_config["type"] == "openai":
                                use_fp16 = torch.cuda.is_available()
                                try:
                                    result = model.transcribe(info["path"], language=lang2, fp16=use_fp16, verbose=False)
                                except Exception:
                                    result = model.transcribe(info["path"], language=lang2, fp16=False, verbose=False)
                                text = result.get('text', '').strip()
                                det_lang = result.get('language', lang2 or 'unknown')

                            elif model_config["type"] == "faster_whisper":
                                # Use faster_whisper for transcription
                                fw_model = model["model"] if isinstance(model, dict) else model
                                
                                # Determine default language
                                default_lang = None
                                if model2 and "ivrit" in model2.lower():
                                    default_lang = "he"
                                
                                segments, info = fw_model.transcribe(
                                    info["path"],
                                    language=lang2 or default_lang,
                                    beam_size=5,
                                    best_of=5,
                                    patience=1,
                                    temperature=0
                                )
                                
                                text_parts = []
                                for segment in segments:
                                    text_parts.append(segment.text.strip())
                                text = " ".join(text_parts)
                                det_lang = info.language if info else (lang2 or 'unknown')
                                
                            elif model_config["type"] == "deepgram":
                                # Use Deepgram v3 file transcription API
                                try:
                                    client = DeepgramClient(api_key=DEEPGRAM_API_KEY) if DEEPGRAM_API_KEY else DeepgramClient()
                                    model = os.getenv("DEEPGRAM_MODEL", "nova-3")

                                    # Read file in executor to avoid blocking
                                    loop = asyncio.get_event_loop()
                                    
                                    def read_capture_file():
                                        with open(info["path"], "rb") as f:
                                            return f.read()
                                    
                                    audio_data = await loop.run_in_executor(None, read_capture_file)
                                    response = client.listen.v1.media.transcribe_file(
                                        request=audio_data,
                                        model=model
                                    )
                                    # Extract transcript
                                    try:
                                        text = response.results.channels[0].alternatives[0].transcript.strip()
                                    except Exception:
                                        text = ""
                                    det_lang = lang2 or os.getenv("DEEPGRAM_LANGUAGE", "en-US")
                                except Exception as e:
                                    await websocket.send_json({"error": f"Deepgram error: {e}"})
                                    return
                            else:
                                await websocket.send_json({"error": f"Unsupported model type: {model_config.get('type')}"})
                                return

                            if text:
                                await websocket.send_json({"type": "transcription", "text": text, "language": det_lang})
                            await websocket.send_json({"type": "complete", "message": "Transcription complete"})
                        finally:
                            try:
                                if os.path.exists(info["path"]):
                                    os.unlink(info["path"])
                                CAPTURES.pop(cid, None)
                            except Exception:
                                pass
                        return

            except Exception as e:
                logger.error(f"Capture flow error: {e}")
                await websocket.send_json({"error": str(e)})
                return

        # Check if we should use yt-dlp or direct streaming
        if should_use_ytdlp(url):
            # Notify user that download is starting
            await websocket.send_json({
                "type": "status",
                "message": "📥 Starting download from URL..."
            })
            
            # Download entire file first with automatic fallback (yt-dlp → ffmpeg)
            audio_file = await download_with_fallback(url, language, websocket=websocket)
            if not audio_file:
                await websocket.send_json({"error": "Failed to download audio from URL. All download methods failed."})
                return

            # Decide path: parallel chunking (flagged) vs single-file transcription (default)
            if USE_PARALLEL_TRANSCRIPTION:
                await websocket.send_json({"type": "status", "message": "Chunking audio and starting parallel transcription..."})
                try:
                    model = load_model(model_name)
                    model_config = MODEL_CONFIGS[model_name]

                    # Split into chunks with overlap
                    temp_dir, chunks = split_audio_into_chunks(
                        audio_file,
                        chunk_seconds=YTDLP_CHUNK_SECONDS,
                        overlap_seconds=YTDLP_CHUNK_OVERLAP
                    )

                    # Transcribe chunks in parallel threads
                    results: List[Tuple[int, str, str]] = []
                    with ThreadPoolExecutor(max_workers=max(1, PARALLEL_WORKERS)) as executor:
                        futures = {
                            executor.submit(transcribe_chunk, model_config, model, path, language): idx
                            for idx, path in chunks
                        }
                        for future in as_completed(futures):
                            try:
                                res = future.result()
                                results.append(res)
                            except Exception as e:
                                logger.error(f"Parallel transcription worker failed: {e}")

                    # Order results by index and stream partials to client
                    for idx, text, det_lang in sorted(results, key=lambda r: r[0]):
                        if text:
                            await websocket.send_json({
                                "type": "transcription",
                                "text": text,
                                "language": det_lang
                            })

                    await websocket.send_json({"type": "complete", "message": "Transcription complete"})

                finally:
                    # Cleanup chunk directory and original file
                    try:
                        if 'temp_dir' in locals() and os.path.isdir(temp_dir):
                            shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception:
                        pass
                    # Only delete if not in cache
                    if os.path.exists(audio_file) and DOWNLOAD_CACHE_DIR not in Path(audio_file).parents:
                        os.unlink(audio_file)

                return

            # Transcribe the downloaded file with incremental output and detailed progress
            try:
                model = load_model(model_name)
                model_config = MODEL_CONFIGS[model_name]
                
                # Check if diarization is requested
                if enable_diarization:
                    await websocket.send_json({"type": "status", "message": "🎭 Transcribing with speaker diarization..."})
                    logger.info(f"Starting diarization for {audio_file}")
                    
                    diarized_segments, detected_language = await transcribe_with_diarization(
                        model, model_config, audio_file, language, websocket, model_name
                    )
                    
                    # Diarization already sends incremental chunks, just send completion
                    await websocket.send_json({"type": "complete", "message": "Transcription with diarization complete"})
                else:
                    # Use regular incremental transcription
                    await websocket.send_json({"type": "status", "message": "Transcribing downloaded audio..."})
                    
                    transcription_text, detected_language = await transcribe_with_incremental_output(
                        model, model_config, audio_file, language, websocket, model_name
                    )

                    # Send complete message
                    await websocket.send_json({"type": "complete", "message": "Transcription complete"})

            finally:
                # Cleanup downloaded file (only if not in cache)
                if os.path.exists(audio_file) and DOWNLOAD_CACHE_DIR not in Path(audio_file).parents:
                    os.unlink(audio_file)
                    logger.debug(f"Cleaned up temporary file: {audio_file}")
                elif os.path.exists(audio_file):
                    logger.debug(f"Keeping cached file: {audio_file}")

            return

        # Otherwise, use direct FFmpeg streaming (for true live streams only)
        await websocket.send_json({"type": "status", "message": "Starting real-time audio stream..."})

        # Create audio processor
        processor = AudioStreamProcessor(url, language, model_name)

        # Start FFmpeg stream
        if not processor.start_ffmpeg_stream():
            await websocket.send_json({"error": "Failed to start audio stream"})
            return

        processor.is_running = True

        # Start audio reading thread
        audio_thread = threading.Thread(target=processor.read_audio_chunks, daemon=True)
        audio_thread.start()

        await websocket.send_json({"type": "status", "message": "Stream started, transcribing..."})

        # Start transcription
        await transcribe_audio_stream(websocket, processor)

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        # Cleanup
        if processor:
            processor.stop()
        if audio_thread and audio_thread.is_alive():
            audio_thread.join(timeout=5)
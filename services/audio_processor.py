"""Audio processing services including download, chunking, and streaming."""
import asyncio
import logging
import os
import subprocess
import tempfile
import shutil
import re
import time
import queue
from pathlib import Path
from typing import Optional, List, Tuple
from fastapi import WebSocket

from config.constants import (
    SAMPLE_RATE,
    CHANNELS,
    CHUNK_DURATION,
    CHUNK_OVERLAP,
    AUDIO_QUEUE_SIZE,
)
from utils.cache import get_cached_download, save_download_to_cache

logger = logging.getLogger(__name__)


async def download_audio_with_ffmpeg(url: str, format: str = 'wav', duration: int = 60, websocket = None, use_cache: bool = True) -> Optional[str]:
    """
    Download audio using ffmpeg directly (proven working method from test_deepgram.py)
    Now async with progress monitoring support and URL caching

    Args:
        url: URL to download
        format: Audio format (wav or m4a)
        duration: Duration in seconds to download (default: 60, 0 = complete file)
        websocket: Optional WebSocket connection for progress updates
        use_cache: Whether to check/use cached downloads
    """
    
    # Check cache first if enabled and duration is 0 (complete file)
    if use_cache and duration == 0:
        cached_file = get_cached_download(url)
        if cached_file:
            if websocket:
                file_size_mb = os.path.getsize(cached_file) / (1024 * 1024)
                # Send special cached_file message type for prominent UI display
                await websocket.send_json({
                    "type": "cached_file",
                    "message": f"✅ Audio file already downloaded ({file_size_mb:.1f} MB) - Using cache, skipping download",
                    "file_size_mb": round(file_size_mb, 2),
                    "skipped_download": True,
                    "cached_path": os.path.basename(cached_file)
                })
            logger.info(f"Using cached audio file: {cached_file} ({file_size_mb:.1f} MB)")
            return cached_file
    if duration == 0:
        logger.info(f"Downloading COMPLETE audio file with ffmpeg (format: {format})...")
    else:
        logger.info(f"Downloading audio with ffmpeg (format: {format}, duration: {duration}s)...")

    try:
        temp_dir = tempfile.mkdtemp()
        audio_file = os.path.join(temp_dir, f'audio.{format}')
        progress_file = os.path.join(temp_dir, 'progress.txt')

        # Use ffmpeg directly with loudnorm filter for optimal audio quality
        cmd = [
            'ffmpeg',
            '-i', url,
        ]

        # Only add duration limit if specified (0 = complete file)
        if duration > 0:
            cmd.extend(['-t', str(duration)])

        cmd.extend([
            '-vn',  # No video
            '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',  # Loudness normalization filter
            '-acodec', 'pcm_s16le' if format == 'wav' else 'aac',  # Codec based on format
            '-ar', '44100',  # 44.1kHz sample rate
            '-ac', '2',  # Stereo
            '-progress', progress_file,  # Progress output to file
            '-y',  # Overwrite output
            audio_file
        ])

        logger.info(f"Running: ffmpeg -i {url[:50]}... -t {duration} ...")

        # Run ffmpeg asynchronously (following pattern from line 1374)
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Monitor progress if websocket provided
        if websocket:
            start_time = time.time()
            last_update = 0
            last_file_size = 0
            estimated_total_size = None

            # Create monitoring task
            async def monitor_progress():
                nonlocal last_update

                while True:
                    await asyncio.sleep(0.5)  # Non-blocking sleep

                    # Check if process completed
                    if process.returncode is not None:
                        break

                    # Read progress file (using executor to avoid blocking)
                    if os.path.exists(progress_file):
                        try:
                            # Run blocking file read in executor
                            loop = asyncio.get_event_loop()
                            
                            def read_progress_file():
                                with open(progress_file, 'r') as f:
                                    return f.readlines()
                            
                            lines = await loop.run_in_executor(None, read_progress_file)
                            progress_data = {}
                            for line in lines:
                                if '=' in line:
                                    key, value = line.strip().split('=', 1)
                                    progress_data[key] = value

                            # Extract time progress
                            if 'out_time_ms' in progress_data and os.path.exists(audio_file):
                                try:
                                    current_seconds = int(progress_data['out_time_ms']) / 1000000  # Convert to seconds
                                    file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
                                    elapsed = time.time() - start_time

                                    # Calculate speed
                                    speed_mbps = file_size_mb / elapsed if elapsed > 0 else 0

                                    # For unknown duration (duration=0), estimate from file size growth
                                    if duration == 0:
                                        # Estimate total duration based on current time vs file size
                                        # Show indeterminate progress with file size and speed only
                                        percent = min((file_size_mb / 200) * 100, 95) if file_size_mb < 200 else 95  # Cap at 95%
                                        eta_seconds = 0  # Unknown
                                        target_duration = 0  # Unknown
                                    else:
                                        # Known duration - calculate accurate progress
                                        percent = min((current_seconds / duration) * 100, 99)
                                        remaining_seconds = duration - current_seconds
                                        eta_seconds = int((remaining_seconds / current_seconds * elapsed)) if current_seconds > 0 else 0
                                        target_duration = duration

                                    # Send progress update (throttle to every 1 second)
                                    if time.time() - last_update >= 1.0:
                                        try:
                                            await websocket.send_json({
                                                "type": "download_progress",
                                                "percent": round(percent, 1),
                                                "downloaded_mb": round(file_size_mb, 2),
                                                "speed_mbps": round(speed_mbps, 2),
                                                "eta_seconds": eta_seconds,
                                                "current_time": round(current_seconds, 1),
                                                "target_duration": target_duration
                                            })
                                            last_update = time.time()
                                        except Exception as e:
                                            logger.debug(f"Progress update failed: {e}")
                                except (ValueError, ZeroDivisionError) as e:
                                    logger.debug(f"Progress parsing error: {e}")
                        except Exception as e:
                            logger.debug(f"Progress file read error: {e}")

            # Start monitoring task
            monitor_task = asyncio.create_task(monitor_progress())

            # Wait for process to complete
            stdout_output, stderr_output = await process.communicate()

            # Cancel monitoring task
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            # Send 100% complete
            if os.path.exists(audio_file):
                file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
                await websocket.send_json({
                    "type": "download_progress",
                    "percent": 100,
                    "downloaded_mb": round(file_size_mb, 2),
                    "speed_mbps": 0,
                    "eta_seconds": 0,
                    "current_time": 0,
                    "target_duration": 0
                })
        else:
            # No websocket, just wait for completion
            stdout_output, stderr_output = await process.communicate()

        result = process

        # Clean up progress file
        try:
            if os.path.exists(progress_file):
                os.unlink(progress_file)
        except:
            pass

        if result.returncode != 0:
            # Extract the actual error from stderr (now properly captured via communicate())
            stderr_lines = stderr_output.split('\n') if stderr_output else []
            error_line = None
            for line in stderr_lines:
                if 'error' in line.lower() or 'http error' in line.lower():
                    error_line = line.strip()
                    break

            if error_line:
                logger.error(f"❌ ffmpeg download failed: {error_line}")
            else:
                logger.error(f"❌ ffmpeg download failed with return code {result.returncode}")

            # Check for common error patterns
            stderr_full = stderr_output.lower() if stderr_output else ""
            if '410' in stderr_full or 'gone' in stderr_full:
                logger.error("💡 URL has expired. Please get a fresh URL from the source.")
            elif '403' in stderr_full or 'forbidden' in stderr_full:
                logger.error("💡 Access denied. The URL may require authentication or be geo-restricted.")
            elif '404' in stderr_full or 'not found' in stderr_full:
                logger.error("💡 URL not found. Please verify the URL is correct.")
            elif 'unsupported' in stderr_full or 'invalid data' in stderr_full:
                logger.error("💡 Audio format not supported or file is corrupted.")

            # Fallback: Try with simpler settings if loudnorm fails
            logger.info("🔄 Trying fallback method without loudnorm...")
            fallback_cmd = [
                'ffmpeg',
                '-i', url,
            ]

            # Only add duration limit if specified (0 = complete file)
            if duration > 0:
                fallback_cmd.extend(['-t', str(duration)])

            fallback_cmd.extend([
                '-vn',
                '-acodec', 'pcm_s16le' if format == 'wav' else 'aac',
                '-ar', '16000',  # 16kHz for better compatibility
                '-ac', '1',  # Mono for better compatibility
                '-y',
                audio_file
            ])

            # Use async subprocess to avoid blocking event loop
            fallback_process = await asyncio.create_subprocess_exec(
                *fallback_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout_fallback, stderr_fallback = await asyncio.wait_for(
                    fallback_process.communicate(),
                    timeout=300
                )
                result = fallback_process
            except asyncio.TimeoutError:
                logger.error("❌ ffmpeg fallback timeout after 5 minutes")
                fallback_process.kill()
                await fallback_process.wait()
                shutil.rmtree(temp_dir, ignore_errors=True)
                return None

            if result.returncode != 0:
                stderr_fallback_str = stderr_fallback.decode() if stderr_fallback else ""
                logger.error(f"❌ ffmpeg fallback also failed: {stderr_fallback_str[:500]}")

                # Provide user-friendly error summary
                if '410' in stderr_fallback_str or 'Gone' in stderr_fallback_str:
                    logger.error("🔴 DOWNLOAD FAILED - URL has expired. Get a fresh URL from the source.")
                elif '403' in stderr_fallback_str or 'Forbidden' in stderr_fallback_str:
                    logger.error("🔴 DOWNLOAD FAILED - Access denied. URL may require authentication.")
                elif '404' in stderr_fallback_str:
                    logger.error("🔴 DOWNLOAD FAILED - URL not found. Verify the URL is correct.")
                else:
                    logger.error("🔴 DOWNLOAD FAILED - Unable to download audio from URL.")

                shutil.rmtree(temp_dir, ignore_errors=True)
                return None

        if os.path.exists(audio_file):
            file_size = os.path.getsize(audio_file)
            logger.info(f"Successfully downloaded audio: {audio_file} ({file_size / 1024:.1f} KB)")
            # Cache the download if duration is 0 (complete file) and caching is enabled
            if use_cache and duration == 0:
                audio_file = save_download_to_cache(url, audio_file)
            return audio_file
        else:
            logger.error(f"Audio file not created: {audio_file}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None

    except subprocess.TimeoutExpired:
        logger.error("ffmpeg download timeout after 5 minutes")
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        return None
    except Exception as e:
        logger.error(f"ffmpeg exception: {e}")
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        return None

async def download_with_fallback(url: str, language: Optional[str] = None, format: str = 'wav', websocket: WebSocket = None, use_cache: bool = True) -> Optional[str]:
    """
    Smart download with automatic fallback chain:
    1. Try yt-dlp with cookies/user-agent
    2. Try ffmpeg direct download
    3. Return None if all methods fail
    
    Args:
        url: URL to download from
        language: Optional language hint
        format: Output format ('wav' or 'm4a')
        websocket: WebSocket for progress updates
        use_cache: Whether to check/use cached downloads
        
    Returns:
        Path to downloaded audio file or None
    """
    
    # Try yt-dlp first (better for YouTube and complex sites)
    if websocket:
        await websocket.send_json({
            "type": "status",
            "message": "📥 Attempting download with yt-dlp..."
        })
    
    audio_file = await download_audio_with_ytdlp_async(url, language, format, websocket, use_cache)
    
    if audio_file:
        return audio_file
    
    # yt-dlp failed, try ffmpeg fallback
    logger.warning(f"yt-dlp failed for {url}, trying ffmpeg fallback...")
    
    if websocket:
        await websocket.send_json({
            "type": "status",
            "message": "⚠️ yt-dlp failed, trying alternative download method (ffmpeg)..."
        })
    
    try:
        # Try ffmpeg as fallback (works for direct streams)
        audio_file = await download_audio_with_ffmpeg(url, format=format, duration=0, websocket=websocket, use_cache=use_cache)
        
        if audio_file:
            logger.info(f"Successfully downloaded with ffmpeg fallback: {audio_file}")
            if websocket:
                await websocket.send_json({
                    "type": "status",
                    "message": "✓ Download successful using alternative method"
                })
            return audio_file
        else:
            logger.error(f"Both yt-dlp and ffmpeg failed for {url}")
            return None
            
    except Exception as e:
        logger.error(f"Error in ffmpeg fallback: {e}")
        return None


async def download_audio_with_ytdlp_async(url: str, language: Optional[str] = None, format: str = 'wav', websocket: WebSocket = None, use_cache: bool = True) -> Optional[str]:
    """Async version of download_audio_with_ytdlp with WebSocket progress updates"""
    
    # Check cache first if enabled
    if use_cache:
        cached_file = get_cached_download(url)
        if cached_file:
            logger.info(f"Using cached download from yt-dlp: {cached_file}")
            if websocket:
                file_size_mb = os.path.getsize(cached_file) / (1024 * 1024)
                await websocket.send_json({
                    "type": "cached_file",
                    "message": f"✅ Audio file already downloaded ({file_size_mb:.1f} MB) - Using cache, skipping download",
                })
            return cached_file
    
    try:
        temp_dir = tempfile.mkdtemp()
        base_filename = os.path.join(temp_dir, 'audio')
        
        cmd = [
            'yt-dlp',
            '-x',  # Extract audio
            '--audio-format', format,
            '--audio-quality', '0',
            '--no-playlist',
            '--newline',
            '-o', base_filename + '.%(ext)s',
        ]
        
        cookies_file = os.path.expanduser('~/.config/yt-dlp/cookies.txt')
        if os.path.exists(cookies_file):
            cmd.extend(['--cookies', cookies_file])
        
        cmd.append(url)
        output_path = f"{base_filename}.{format}"
        
        logger.info(f"Downloading audio from URL with yt-dlp: {url}")
        
        # Run yt-dlp asynchronously
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        
        # Process output for progress updates
        last_progress_time = 0
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            line_str = line.decode('utf-8').strip()
            
            # Parse yt-dlp progress output
            if websocket and '[download]' in line_str:
                try:
                    # Parse percentage (e.g., "[download]  45.2% of 123.45MiB at 2.34MiB/s ETA 00:45")
                    if '%' in line_str:
                        parts = line_str.split()
                        for i, part in enumerate(parts):
                            if '%' in part:
                                percent = float(part.replace('%', ''))
                                
                                # Parse size if available
                                size_mb = 0
                                if 'of' in parts and i+1 < len(parts)-1:
                                    size_str = parts[parts.index('of') + 1]
                                    if 'MiB' in size_str or 'MB' in size_str:
                                        size_mb = float(size_str.replace('MiB', '').replace('MB', ''))
                                
                                # Parse speed if available
                                speed_mbps = 0
                                for j, p in enumerate(parts):
                                    if 'MiB/s' in p or 'MB/s' in p:
                                        speed_mbps = float(p.replace('MiB/s', '').replace('MB/s', ''))
                                        break
                                
                                # Parse ETA if available
                                eta_seconds = 0
                                if 'ETA' in parts:
                                    eta_idx = parts.index('ETA') + 1
                                    if eta_idx < len(parts):
                                        eta_str = parts[eta_idx]
                                        if ':' in eta_str:
                                            time_parts = eta_str.split(':')
                                            if len(time_parts) == 2:
                                                eta_seconds = int(time_parts[0]) * 60 + int(time_parts[1])
                                            elif len(time_parts) == 3:
                                                eta_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
                                
                                # Throttle updates to every 0.5 seconds
                                current_time = time.time()
                                if current_time - last_progress_time >= 0.5:
                                    await websocket.send_json({
                                        "type": "download_progress",
                                        "percent": round(percent, 1),
                                        "downloaded_mb": round(size_mb * percent / 100, 2),
                                        "speed_mbps": round(speed_mbps, 2),
                                        "eta_seconds": eta_seconds
                                    })
                                    last_progress_time = current_time
                                break
                except Exception as e:
                    logger.debug(f"Failed to parse yt-dlp progress: {e}")
            
            # Log errors
            if 'ERROR:' in line_str:
                logger.error(f"yt-dlp error: {line_str}")
        
        await process.wait()
        
        if process.returncode == 0 and os.path.exists(output_path):
            # Cache the download if enabled
            if use_cache:
                output_path = save_download_to_cache(url, output_path)
            return output_path
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None
            
    except Exception as e:
        logger.error(f"Error downloading with yt-dlp: {e}")
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        return None


def get_audio_duration_seconds(audio_path: str) -> float:
    """Get duration of audio file in seconds using ffprobe"""
    try:
        cmd = [
            'ffprobe',
            '-i', audio_path,
            '-show_entries', 'format=duration',
            '-v', 'quiet',
            '-of', 'csv=p=0'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            return duration
    except Exception as e:
        logger.warning(f"Failed to get audio duration: {e}")
    return 0.0


def calculate_progress_metrics(audio_duration: float, elapsed_time: float, processed_chunks: int = 0, total_chunks: int = 0) -> dict:
    """
    Calculate transcription progress metrics
    
    Args:
        audio_duration: Total audio duration in seconds
        elapsed_time: Time elapsed since transcription started
        processed_chunks: Number of chunks processed (for incremental mode)
        total_chunks: Total number of chunks (for incremental mode)
    
    Returns:
        Dictionary with progress metrics
    """
    metrics = {
        'audio_duration': audio_duration,
        'elapsed_time': elapsed_time,
        'processing_speed': 0.0,
        'estimated_time_remaining': 0.0,
        'percent_complete': 0.0
    }
    
    if total_chunks > 0:
        # Chunk-based progress
        metrics['percent_complete'] = (processed_chunks / total_chunks) * 100
        if processed_chunks > 0 and elapsed_time > 0:
            avg_chunk_time = elapsed_time / processed_chunks
            remaining_chunks = total_chunks - processed_chunks
            metrics['estimated_time_remaining'] = avg_chunk_time * remaining_chunks
    elif audio_duration > 0 and elapsed_time > 0:
        # Time-based progress estimation
        metrics['processing_speed'] = audio_duration / elapsed_time
        metrics['percent_complete'] = min((elapsed_time / audio_duration) * 100, 100)
    
    return metrics


def split_audio_for_incremental(audio_path: str, chunk_seconds: int = 60, overlap_seconds: int = 5) -> Tuple[str, List[str]]:
    """Split audio file into chunks for incremental transcription"""
    try:
        import librosa
        import soundfile as sf
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Calculate chunk and overlap samples
        chunk_samples = chunk_seconds * sr
        overlap_samples = overlap_seconds * sr
        
        # Create temp directory for chunks
        temp_dir = tempfile.mkdtemp()
        chunk_paths = []
        
        # Split into chunks
        start = 0
        chunk_idx = 0
        while start < len(y):
            end = min(start + chunk_samples, len(y))
            chunk = y[start:end]
            
            # Save chunk
            chunk_path = os.path.join(temp_dir, f"chunk_{chunk_idx:04d}.wav")
            sf.write(chunk_path, chunk, sr)
            chunk_paths.append(chunk_path)
            
            # Move to next chunk with overlap
            start += chunk_samples - overlap_samples
            chunk_idx += 1
        
        logger.info(f"Split audio into {len(chunk_paths)} chunks")
        return temp_dir, chunk_paths
        
    except ImportError:
        logger.warning("librosa not available, falling back to ffmpeg for splitting")
        return split_audio_into_chunks(audio_path, chunk_seconds, overlap_seconds)
    except Exception as e:
        logger.error(f"Error splitting audio: {e}")
        return "", []


def split_audio_into_chunks(audio_path: str, chunk_seconds: int = 60, overlap_seconds: int = 5) -> Tuple[str, List[str]]:
    """Split audio file into chunks using ffmpeg (fallback method)"""
    try:
        temp_dir = tempfile.mkdtemp()
        chunk_paths = []
        
        # Get total duration
        duration = get_audio_duration_seconds(audio_path)
        if duration == 0:
            return "", []
        
        # Calculate chunks
        start = 0
        chunk_idx = 0
        while start < duration:
            chunk_duration = min(chunk_seconds, duration - start)
            
            chunk_path = os.path.join(temp_dir, f"chunk_{chunk_idx:04d}.wav")
            cmd = [
                'ffmpeg',
                '-i', audio_path,
                '-ss', str(start),
                '-t', str(chunk_duration),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',
                chunk_path
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            chunk_paths.append(chunk_path)
            
            start += chunk_seconds - overlap_seconds
            chunk_idx += 1
        
        logger.info(f"Split audio into {len(chunk_paths)} chunks using ffmpeg")
        return temp_dir, chunk_paths
        
    except Exception as e:
        logger.error(f"Error splitting audio with ffmpeg: {e}")
        return "", []


class AudioStreamProcessor:
    """Processes audio streams from URLs using FFmpeg and transcribes with Whisper"""
    
    def __init__(self, url: str, language: Optional[str] = None, model_name: str = "large"):
        self.url = url
        self.language = language
        self.model_name = model_name
        self.model = None
        self.is_running = False
        self.audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_SIZE)
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        
    def start_ffmpeg_stream(self):
        """Start FFmpeg process to stream audio from URL"""
        try:
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', self.url,
                '-f', 's16le',
                '-acodec', 'pcm_s16le',
                '-ar', str(SAMPLE_RATE),
                '-ac', str(CHANNELS),
                '-loglevel', 'error',
                'pipe:1'
            ]
            
            logger.info(f"Starting FFmpeg stream for URL: {self.url}")
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")
            return False
    
    def read_audio_chunks(self):
        """Read audio data from FFmpeg in chunks with overlap for context preservation"""
        if not self.ffmpeg_process:
            return

        chunk_size = int(SAMPLE_RATE * CHANNELS * 2 * CHUNK_DURATION)
        overlap_size = int(SAMPLE_RATE * CHANNELS * 2 * CHUNK_OVERLAP)
        overlap_buffer = b''

        try:
            while self.is_running:
                new_data_size = chunk_size - len(overlap_buffer)
                audio_data = self.ffmpeg_process.stdout.read(new_data_size)

                if not audio_data:
                    if overlap_buffer and len(overlap_buffer) >= SAMPLE_RATE * CHANNELS * 2:
                        try:
                            self.audio_queue.put_nowait(overlap_buffer)
                        except queue.Full:
                            logger.warning("Audio queue full, dropped final chunk")
                    logger.info("FFmpeg stream ended")
                    break

                full_chunk = overlap_buffer + audio_data

                if len(full_chunk) >= overlap_size:
                    overlap_buffer = full_chunk[-overlap_size:]
                else:
                    overlap_buffer = full_chunk

                try:
                    self.audio_queue.put_nowait(full_chunk)
                except queue.Full:
                    try:
                        logger.debug(f"Audio queue full, applying backpressure...")
                        self.audio_queue.put(full_chunk, timeout=5.0)
                    except queue.Full:
                        logger.warning("Audio queue saturated, dropping chunk")

        except Exception as e:
            logger.error(f"Error reading audio chunks: {e}")
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop the audio streaming and cleanup"""
        self.is_running = False
        
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ffmpeg_process.kill()
            except Exception as e:
                logger.error(f"Error stopping FFmpeg: {e}")

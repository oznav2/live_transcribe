#!/usr/bin/env python3
"""
Test script for Deepgram transcription methods.
Mimics the exact logic from app.py for VOD transcription.
Enhanced with comprehensive audio analysis and diagnostics.
"""

import os
import sys
import tempfile
import subprocess
import logging
import json
import re
import numpy as np
import asyncio
import websockets
import pyaudio
import time
import wave
from pathlib import Path
from deepgram import DeepgramClient

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Load API key from .env
from dotenv import load_dotenv
load_dotenv()

DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')

# Default test URL for streaming tests
DEFAULT_TEST_URL = "https://video-nss.xhcdn.com/meTMFVJ3c6429qU1fZkTUg==,1761760800/media=hls4/multi=256x144:144p:,426x240:240p:,854x480:480p:,1280x720:720p:,1920x1080:1080p:/026/391/223/1080p.av1.mp4.m3u8"


def is_streaming_url(url: str) -> bool:
    """
    Detect if URL is a streaming/HLS URL that Deepgram cannot process directly.
    These URLs typically contain .m3u8 files or streaming indicators.
    """
    url_lower = url.lower()
    streaming_indicators = [
        '.m3u8',
        '/hls/',
        '/live/',
        '/stream/',
        'manifest',
        'playlist.m3u8',
        'index.m3u8',
        'chunklist',
        '/dash/',
        '.mpd'
    ]
    
    for indicator in streaming_indicators:
        if indicator in url_lower:
            logger.info(f"Detected streaming URL indicator: {indicator}")
            return True
    
    return False
if not DEEPGRAM_API_KEY:
    logger.error("Error: DEEPGRAM_API_KEY not found in .env")
    sys.exit(1)


def download_audio_with_ytdlp(url: str, format: str = 'wav', duration: int = 30) -> str | None:
    """
    Download audio using ffmpeg directly with loudnorm filter for optimal audio quality
    
    Args:
        url: URL to download
        format: Audio format (default: wav for better compatibility)
        duration: Duration in seconds to download (default: 30)
    """
    logger.info(f"\nDownloading audio with ffmpeg (format: {format}, duration: {duration}s)...")

    try:
        temp_dir = tempfile.mkdtemp()
        audio_file = os.path.join(temp_dir, f'audio.{format}')

        # Use ffmpeg directly with the exact command provided by user
        # ffmpeg -i 'URL' -vn -af "loudnorm=I=-16:TP=-1.5:LRA=11" -acodec pcm_s16le -ar 44100 -ac 2 output.wav
        cmd = [
            'ffmpeg',
            '-i', url,
            '-t', str(duration),  # Limit duration
            '-vn',  # No video
            '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',  # Loudness normalization filter
            '-acodec', 'pcm_s16le',  # PCM 16-bit little endian
            '-ar', '44100',  # 44.1kHz sample rate
            '-ac', '2',  # Stereo
            '-y',  # Overwrite output
            audio_file
        ]

        logger.info(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"ffmpeg download failed: {result.stderr}")
            
            # Fallback: Try with simpler settings if loudnorm fails
            logger.info("Trying fallback method without loudnorm...")
            fallback_cmd = [
                'ffmpeg',
                '-i', url,
                '-t', str(duration),
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',  # 16kHz for better compatibility with Deepgram
                '-ac', '1',  # Mono for better compatibility
                '-y',
                audio_file
            ]
            
            logger.info(f"Fallback command: {' '.join(fallback_cmd)}")
            fallback_result = subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=300)
            
            if fallback_result.returncode != 0:
                logger.error(f"Fallback ffmpeg also failed: {fallback_result.stderr}")
                return None

        if os.path.exists(audio_file):
            file_size = os.path.getsize(audio_file)
            logger.info(f"Downloaded audio: {audio_file} ({file_size / 1024 / 1024:.2f} MB)")
            return audio_file
        else:
            logger.error(f"Audio file not found: {audio_file}")
            return None

    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timeout after 300 seconds")
        return None
    except Exception as e:
        logger.error(f"Download error: {e}")
        return None


def transcribe_url_method(url: str, language: str = "auto", model: str = "nova-3") -> dict:
    """
    Method 1: Direct URL transcription (Deepgram fetches the file)
    """
    logger.info(f"\nMethod 1: Direct URL transcription")
    logger.info(f"URL: {url}")
    logger.info(f"Language: {language}")
    logger.info(f"Model: {model}")

    try:
        client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

        logger.info("Sending request to Deepgram...")
        
        # Use detect_language=True for automatic language detection
        if language == "auto":
            logger.info("Using automatic language detection (detect_language=True)")
            response = client.listen.v1.media.transcribe_url(
                url=url,
                model=model,
                detect_language=True,
                punctuate=True,
                smart_format=True
            )
        else:
            response = client.listen.v1.media.transcribe_url(
                url=url,
                model=model,
                language=language,
                punctuate=True,
                smart_format=True
            )

        transcript = response.results.channels[0].alternatives[0].transcript.strip()

        request_id = None
        try:
            if hasattr(response, 'metadata') and response.metadata:
                request_id = getattr(response.metadata, 'request_id', None)
        except (AttributeError, TypeError):
            pass

        logger.info(f"Success!")
        logger.info(f"Request ID: {request_id}")
        logger.info(f"Transcript length: {len(transcript)} characters")

        # Display the actual transcript
        logger.info(f"✅ URL Method Transcript Result:")
        if transcript:
            logger.info(f"   {transcript[:500]}")
            if len(transcript) > 500:
                logger.info(f"   ... [showing first 500 of {len(transcript)} characters]")
        else:
            logger.warning("   ⚠️  EMPTY TRANSCRIPT - No speech detected or parsing failed")

        return {
            "success": True,
            "method": "url",
            "transcript": transcript,
            "request_id": request_id,
            "length": len(transcript),
            "model": model
        }

    except Exception as e:
        logger.error(f"Failed: {e}")
        return {
            "success": False,
            "method": "url",
            "error": str(e),
            "model": model
        }


def analyze_audio_data(audio_data: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> dict:
    """Analyze raw audio data for speech detection and quality assessment."""
    try:
        # Convert bytes to numpy array
        if sample_width == 2:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
        elif sample_width == 4:
            audio_array = np.frombuffer(audio_data, dtype=np.int32)
        else:
            audio_array = np.frombuffer(audio_data, dtype=np.uint8)
        
        # Convert to float for analysis
        if sample_width == 2:
            audio_float = audio_array.astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio_float = audio_array.astype(np.float32) / 2147483648.0
        else:
            audio_float = audio_array.astype(np.float32) / 128.0
        
        # Basic audio statistics
        duration = len(audio_array) / (sample_rate * channels)
        rms = np.sqrt(np.mean(audio_float ** 2))
        peak = np.max(np.abs(audio_float))
        
        # Convert to dB
        rms_db = 20 * np.log10(rms + 1e-10)
        peak_db = 20 * np.log10(peak + 1e-10)
        
        # Detect silence (very low RMS)
        is_silent = rms_db < -40.0
        
        # Simple speech detection heuristics
        # 1. Check for dynamic range (speech has varying amplitude)
        dynamic_range = peak_db - rms_db
        
        # 2. Check for frequency content (basic spectral analysis)
        # Calculate zero crossing rate (speech typically has moderate ZCR)
        zero_crossings = np.sum(np.diff(np.sign(audio_float)) != 0)
        zcr = zero_crossings / len(audio_float)
        
        # Speech detection heuristics
        has_speech_indicators = (
            not is_silent and
            rms_db > -35.0 and  # Minimum signal level
            dynamic_range > 6.0 and  # Some dynamic range
            0.01 < zcr < 0.8  # Reasonable zero crossing rate for speech (relaxed for high-frequency content)
        )
        
        return {
            "duration_seconds": duration,
            "rms_db": rms_db,
            "peak_db": peak_db,
            "dynamic_range_db": dynamic_range,
            "zero_crossing_rate": zcr,
            "is_silent": is_silent,
            "likely_contains_speech": has_speech_indicators,
            "sample_count": len(audio_array),
            "analysis_summary": f"RMS: {rms_db:.1f}dB, Peak: {peak_db:.1f}dB, ZCR: {zcr:.3f}, Speech: {'Yes' if has_speech_indicators else 'No'}"
        }
    except Exception as e:
        return {
            "error": f"Audio analysis failed: {str(e)}",
            "duration_seconds": 0,
            "likely_contains_speech": False
        }


def ffprobe_info(audio_file: str) -> dict:
    """
    Inspect audio file with ffprobe and log key metadata.
    """
    logger.info("\nRunning ffprobe to inspect audio metadata...")
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-print_format', 'json',
            '-show_streams',
            '-show_format',
            audio_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.warning(f"ffprobe failed: {result.stderr.strip()}")
            return {}
        data = json.loads(result.stdout)

        # Extract and log concise summary
        fmt = data.get('format', {})
        streams = data.get('streams', [])
        audio_stream = next((s for s in streams if s.get('codec_type') == 'audio'), None)

        logger.info("ffprobe summary:")
        logger.info(f"- container: {fmt.get('format_name')} | duration: {fmt.get('duration')}s | bit_rate: {fmt.get('bit_rate')} bps")
        if audio_stream:
            logger.info(
                f"- audio codec: {audio_stream.get('codec_name')} | sample_rate: {audio_stream.get('sample_rate')} Hz | channels: {audio_stream.get('channels')} | layout: {audio_stream.get('channel_layout')} | bit_rate: {audio_stream.get('bit_rate')} bps"
            )
            # Additional codec details that might affect ASR
            profile = audio_stream.get('profile')
            if profile:
                logger.info(f"- codec profile: {profile}")
        else:
            logger.warning("- No audio stream detected by ffprobe")

        return data
    except subprocess.TimeoutExpired:
        logger.warning("ffprobe timed out after 60 seconds")
        return {}
    except Exception as e:
        logger.warning(f"ffprobe error: {e}")
        return {}


def ffmpeg_volumedetect(audio_file: str) -> dict:
    """
    Use ffmpeg volumedetect to check audio levels (mean/max).
    """
    logger.info("\nChecking audio levels with ffmpeg volumedetect...")
    try:
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-nostats',
            '-i', audio_file,
            '-filter:a', 'volumedetect',
            '-f', 'null', '-'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        stderr = result.stderr or ""

        # Parse mean_volume and max_volume from stderr
        mean_match = re.search(r"mean_volume:\s*([\-\d\.]+)\s*dB", stderr)
        max_match = re.search(r"max_volume:\s*([\-\d\.]+)\s*dB", stderr)
        mean_db = float(mean_match.group(1)) if mean_match else None
        max_db = float(max_match.group(1)) if max_match else None

        logger.info(f"volumedetect: mean_volume={mean_db} dB | max_volume={max_db} dB")
        if mean_db is not None and mean_db < -35:
            logger.warning("- Audio appears very quiet (mean < -35 dB). This can cause empty transcripts.")

        return {"mean_volume_db": mean_db, "max_volume_db": max_db}
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg volumedetect timed out after 120 seconds")
        return {}
    except Exception as e:
        logger.warning(f"ffmpeg volumedetect error: {e}")
        return {}


def ffmpeg_silencedetect(audio_file: str, noise_threshold: float = -30.0, duration_threshold: float = 0.5) -> dict:
    """
    Use ffmpeg silencedetect to analyze speech activity vs silence.
    """
    logger.info(f"\nAnalyzing speech activity with silencedetect (threshold: {noise_threshold} dB, min duration: {duration_threshold}s)...")
    try:
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-nostats',
            '-i', audio_file,
            '-filter:a', f'silencedetect=noise={noise_threshold}dB:duration={duration_threshold}',
            '-f', 'null', '-'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        stderr = result.stderr or ""

        # Parse silence periods
        silence_starts = re.findall(r"silence_start:\s*([\d\.]+)", stderr)
        silence_ends = re.findall(r"silence_end:\s*([\d\.]+)", stderr)
        silence_durations = re.findall(r"silence_duration:\s*([\d\.]+)", stderr)

        total_silence = sum(float(d) for d in silence_durations)
        silence_periods = len(silence_starts)

        logger.info(f"silencedetect: {silence_periods} silence periods | total silence: {total_silence:.2f}s")
        
        # Get total duration from ffprobe for speech ratio calculation
        ffprobe_data = ffprobe_info(audio_file)
        total_duration = 0
        if ffprobe_data.get('format', {}).get('duration'):
            total_duration = float(ffprobe_data['format']['duration'])
            speech_duration = total_duration - total_silence
            speech_ratio = speech_duration / total_duration if total_duration > 0 else 0
            logger.info(f"- total duration: {total_duration:.2f}s | speech: {speech_duration:.2f}s ({speech_ratio:.1%})")
            
            if speech_ratio < 0.1:
                logger.warning("- Very low speech ratio (<10%). Audio may be mostly music/noise.")
            elif speech_ratio < 0.3:
                logger.warning("- Low speech ratio (<30%). Mixed content detected.")

        return {
            "silence_periods": silence_periods,
            "total_silence_duration": total_silence,
            "total_duration": total_duration,
            "speech_duration": total_duration - total_silence if total_duration > 0 else 0,
            "speech_ratio": (total_duration - total_silence) / total_duration if total_duration > 0 else 0
        }
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg silencedetect timed out after 120 seconds")
        return {}
    except Exception as e:
        logger.warning(f"ffmpeg silencedetect error: {e}")
        return {}


def ffmpeg_spectral_analysis(audio_file: str) -> dict:
    """
    Extract spectral features that might indicate speech vs music/noise.
    """
    logger.info("\nPerforming spectral analysis...")
    try:
        # Extract spectral centroid and other features
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-nostats',
            '-i', audio_file,
            '-filter:a', 'aformat=channel_layouts=mono,aresample=16000,astats=metadata=1:reset=1',
            '-f', 'null', '-'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        stderr = result.stderr or ""

        # Parse audio statistics
        rms_match = re.search(r"RMS level dB:\s*([\-\d\.]+)", stderr)
        peak_match = re.search(r"Peak level dB:\s*([\-\d\.]+)", stderr)
        
        # Safely convert to float, handling cases where value is just "-" or invalid
        rms_db = None
        if rms_match:
            try:
                rms_value = rms_match.group(1).strip()
                if rms_value != "-" and rms_value != "":
                    rms_db = float(rms_value)
            except (ValueError, AttributeError):
                pass
        
        peak_db = None
        if peak_match:
            try:
                peak_value = peak_match.group(1).strip()
                if peak_value != "-" and peak_value != "":
                    peak_db = float(peak_value)
            except (ValueError, AttributeError):
                pass

        logger.info(f"spectral analysis: RMS={rms_db} dB | Peak={peak_db} dB")
        
        # Analyze frequency content using showfreqs
        cmd_freq = [
            'ffmpeg',
            '-hide_banner',
            '-nostats',
            '-i', audio_file,
            '-filter:a', 'aformat=channel_layouts=mono,aresample=16000,showfreqs=mode=line:fscale=log',
            '-f', 'null', '-',
            '-t', '10'  # Analyze first 10 seconds only
        ]
        result_freq = subprocess.run(cmd_freq, capture_output=True, text=True, timeout=60)
        
        # Simple heuristic: speech typically has energy in 300-3400 Hz range
        # This is a basic analysis - more sophisticated would require actual FFT
        
        return {
            "rms_db": rms_db,
            "peak_db": peak_db,
            "analysis_note": "Basic spectral analysis completed"
        }
    except subprocess.TimeoutExpired:
        logger.warning("spectral analysis timed out")
        return {}
    except Exception as e:
        logger.warning(f"spectral analysis error: {e}")
        return {}


def test_audio_preprocessing_variants(audio_file: str) -> list:
    """
    Create different preprocessed versions of the audio to test which works best.
    """
    logger.info("\nCreating audio preprocessing variants...")
    variants = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Variant 1: Standard WAV 16k mono
        wav_standard = os.path.join(temp_dir, 'standard.wav')
        cmd1 = ['ffmpeg', '-y', '-i', audio_file, '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', wav_standard]
        if subprocess.run(cmd1, capture_output=True, timeout=60).returncode == 0:
            variants.append(("standard_wav_16k", wav_standard))
        
        # Variant 2: Normalized audio (loudnorm)
        wav_normalized = os.path.join(temp_dir, 'normalized.wav')
        cmd2 = ['ffmpeg', '-y', '-i', audio_file, '-filter:a', 'loudnorm', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', wav_normalized]
        if subprocess.run(cmd2, capture_output=True, timeout=60).returncode == 0:
            variants.append(("normalized_wav_16k", wav_normalized))
        
        # Variant 3: High-pass filtered (remove low frequency noise)
        wav_highpass = os.path.join(temp_dir, 'highpass.wav')
        cmd3 = ['ffmpeg', '-y', '-i', audio_file, '-filter:a', 'highpass=f=80', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', wav_highpass]
        if subprocess.run(cmd3, capture_output=True, timeout=60).returncode == 0:
            variants.append(("highpass_wav_16k", wav_highpass))
        
        # Variant 4: Noise reduction (basic)
        wav_denoise = os.path.join(temp_dir, 'denoise.wav')
        cmd4 = ['ffmpeg', '-y', '-i', audio_file, '-filter:a', 'afftdn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', wav_denoise]
        if subprocess.run(cmd4, capture_output=True, timeout=60).returncode == 0:
            variants.append(("denoised_wav_16k", wav_denoise))
        
        logger.info(f"Created {len(variants)} preprocessing variants")
        return variants
        
    except Exception as e:
        logger.warning(f"Error creating preprocessing variants: {e}")
        return variants


def reencode_to_wav_16k_mono(audio_file: str) -> str | None:
    """
    Re-encode the input to 16 kHz mono WAV (PCM s16le) using minimal ffmpeg flags
    to avoid audio corruption.
    """
    logger.info("\nRe-encoding to WAV 16k mono (pcm_s16le) with minimal flags...")
    try:
        temp_dir = tempfile.mkdtemp()
        out_file = os.path.join(temp_dir, 'audio.wav')
        
        # Use minimal ffmpeg flags to avoid audio corruption
        cmd = [
            'ffmpeg', '-y',
            '-i', audio_file,
            '-ar', '16000',                  # 16kHz sample rate
            '-ac', '1',                      # Force mono
            '-f', 'wav',                     # Force WAV format
            out_file
        ]
        
        logger.info(f"🔧 Running ffmpeg with minimal flags for audio processing...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            logger.error(f"ffmpeg re-encode failed: {result.stderr.strip()}")
            return None
        if os.path.exists(out_file):
            size = os.path.getsize(out_file)
            logger.info(f"✅ WAV created with minimal processing: {out_file} ({size / 1024 / 1024:.2f} MB)")
            return out_file
        logger.error("WAV output not found after re-encode")
        return None
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg re-encode timed out after 180 seconds")
        return None
    except Exception as e:
        logger.error(f"re-encode error: {e}")
        return None


def transcribe_file_method(audio_file: str, language: str = "auto", model: str = "nova-3", test_variants: bool = False) -> dict:
    """
    Method 2: Upload local file transcription with comprehensive analysis
    """
    logger.info(f"\nMethod 2: File upload transcription")
    logger.info(f"File: {audio_file}")
    logger.info(f"Language: {language}")
    logger.info(f"Model: {model}")

    try:
        file_size = os.path.getsize(audio_file)
        logger.info(f"File size: {file_size / 1024 / 1024:.2f} MB")

        if file_size > 100 * 1024 * 1024:
            logger.warning("Warning: File > 100MB may hit payload limits")

        # Comprehensive diagnostics
        ffprobe_data = ffprobe_info(audio_file)
        vol_data = ffmpeg_volumedetect(audio_file)
        silence_data = ffmpeg_silencedetect(audio_file)
        spectral_data = ffmpeg_spectral_analysis(audio_file)

        client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

        with open(audio_file, 'rb') as f:
            buffer_data = f.read()

        logger.info("Uploading to Deepgram...")
        logger.info(f"Buffer size: {len(buffer_data)} bytes ({len(buffer_data) / 1024 / 1024:.2f} MB)")

        # Create client with longer timeout
        from deepgram.core.request_options import RequestOptions
        request_options = RequestOptions(timeout_in_seconds=300)  # 5 minutes

        # First attempt: use automatic language detection with requested model
        logger.info(f"Attempt 1: Transcribe original file with detect_language=True, model='{model}'")
        
        if language == "auto":
            logger.info("Using automatic language detection (detect_language=True)")
            response = client.listen.v1.media.transcribe_file(
                request=buffer_data,
                model=model,
                detect_language=True,
                punctuate=True,
                smart_format=True,
                request_options=request_options
            )
        else:
            response = client.listen.v1.media.transcribe_file(
                request=buffer_data,
                model=model,
                language=language,
                punctuate=True,
                smart_format=True,
                request_options=request_options
            )

        # Debug: Check response structure
        logger.info(f"Response type: {type(response)}")
        logger.info(f"Response has results: {hasattr(response, 'results')}")
        if hasattr(response, 'results'):
            logger.info(f"Channels: {len(response.results.channels) if response.results.channels else 0}")
            if response.results.channels:
                logger.info(f"Alternatives: {len(response.results.channels[0].alternatives) if response.results.channels[0].alternatives else 0}")
                alt = response.results.channels[0].alternatives[0]
                logger.info(f"Transcript: '{alt.transcript}'")
                logger.info(f"Confidence: {alt.confidence}")
                logger.info(f"Words count: {len(alt.words) if alt.words else 0}")
                if hasattr(response.results.channels[0], 'detected_language'):
                    logger.info(f"Detected language: {response.results.channels[0].detected_language}")

        # Check metadata for additional info
        if hasattr(response, 'metadata'):
            logger.info(f"Metadata: {response.metadata}")
            if hasattr(response.metadata, 'duration'):
                logger.info(f"Audio duration from Deepgram: {response.metadata.duration}s")

        transcript = response.results.channels[0].alternatives[0].transcript.strip()
        best_result = {"transcript": transcript, "confidence": response.results.channels[0].alternatives[0].confidence, "method": f"original_{model}"}

        # If empty or very short, try different approaches
        if len(transcript) < 50:  # Less than 50 characters may be insufficient for full audio
            logger.warning(f"Transcript short ({len(transcript)} chars). Trying enhanced approaches to get full audio...")
            
            # Try different model if using nova-3
            if model == "nova-3":
                logger.info("Attempt 2: Trying enhanced model...")
                try:
                    if language == "auto":
                        response_enhanced = client.listen.v1.media.transcribe_file(
                            request=buffer_data,
                            model="enhanced",
                            detect_language=True,
                            punctuate=True,
                            smart_format=True,
                            request_options=request_options
                        )
                    else:
                        response_enhanced = client.listen.v1.media.transcribe_file(
                            request=buffer_data,
                            model="enhanced",
                            language=language,
                            punctuate=True,
                            smart_format=True,
                            request_options=request_options
                        )
                    alt_enhanced = response_enhanced.results.channels[0].alternatives[0]
                    transcript_enhanced = alt_enhanced.transcript.strip()
                    logger.info(f"Enhanced model transcript length: {len(transcript_enhanced)} | confidence: {alt_enhanced.confidence}")
                    if len(transcript_enhanced) > len(transcript):
                        transcript = transcript_enhanced
                        response = response_enhanced
                        best_result = {"transcript": transcript, "confidence": alt_enhanced.confidence, "method": "enhanced"}
                except Exception as e:
                    logger.warning(f"Enhanced model failed: {e}")
            
            # WAV fallback with multiple variants (always try if transcript is short)
            if test_variants or len(transcript) < 50:
                variants = test_audio_preprocessing_variants(audio_file)
                for variant_name, variant_file in variants:
                    logger.info(f"Attempt: Testing {variant_name}...")
                    try:
                        with open(variant_file, 'rb') as vf:
                            variant_buf = vf.read()
                        
                        if language == "auto":
                            response_variant = client.listen.v1.media.transcribe_file(
                                request=variant_buf,
                                model=model,
                                detect_language=True,
                                punctuate=True,
                                smart_format=True,
                                request_options=request_options
                            )
                        else:
                            response_variant = client.listen.v1.media.transcribe_file(
                                request=variant_buf,
                                model=model,
                                language=language,
                                punctuate=True,
                                smart_format=True,
                                request_options=request_options
                            )
                        alt_variant = response_variant.results.channels[0].alternatives[0]
                        transcript_variant = alt_variant.transcript.strip()
                        logger.info(f"{variant_name} transcript length: {len(transcript_variant)} | confidence: {alt_variant.confidence}")
                        if len(transcript_variant) > len(transcript):
                            transcript = transcript_variant
                            response = response_variant
                            best_result = {"transcript": transcript, "confidence": alt_variant.confidence, "method": variant_name}
                    except Exception as e:
                        logger.warning(f"{variant_name} failed: {e}")
            else:
                # Standard WAV fallback
                wav_file = reencode_to_wav_16k_mono(audio_file)
                if wav_file:
                    with open(wav_file, 'rb') as wf:
                        wav_buf = wf.read()
                    logger.info(f"Attempt: Transcribe WAV fallback with detect_language={language == 'auto'}, model='{model}'")
                    
                    if language == "auto":
                        response_wav = client.listen.v1.media.transcribe_file(
                            request=wav_buf,
                            model=model,
                            detect_language=True,
                            punctuate=True,
                            smart_format=True,
                            request_options=request_options
                        )
                    else:
                        response_wav = client.listen.v1.media.transcribe_file(
                            request=wav_buf,
                            model=model,
                            language=language,
                            punctuate=True,
                            smart_format=True,
                            request_options=request_options
                        )
                    alt_wav = response_wav.results.channels[0].alternatives[0]
                    transcript_wav = alt_wav.transcript.strip()
                    logger.info(f"WAV transcript length: {len(transcript_wav)} | confidence: {alt_wav.confidence}")
                    if len(transcript_wav) > len(transcript):
                        transcript = transcript_wav
                        response = response_wav
                        best_result = {"transcript": transcript, "confidence": alt_wav.confidence, "method": f"wav_{model}"}

        request_id = None
        try:
            if hasattr(response, 'metadata') and response.metadata:
                request_id = getattr(response.metadata, 'request_id', None)
        except (AttributeError, TypeError):
            pass

        logger.info(f"Success!")
        logger.info(f"Best result from: {best_result['method']}")
        logger.info(f"Request ID: {request_id}")
        logger.info(f"Transcript length: {len(transcript)} characters")
        logger.info(f"Confidence: {best_result['confidence']}")

        # Display the actual transcript
        logger.info(f"✅ File Upload Method Transcript Result:")
        if transcript:
            logger.info(f"   {transcript[:500]}")
            if len(transcript) > 500:
                logger.info(f"   ... [showing first 500 of {len(transcript)} characters]")
        else:
            logger.warning("   ⚠️  EMPTY TRANSCRIPT - No speech detected in audio")

        return {
            "success": True,
            "method": "file_upload",
            "transcript": transcript,
            "request_id": request_id,
            "length": len(transcript),
            "confidence": best_result['confidence'],
            "best_method": best_result['method'],
            "model": model,
            "diagnostics": {
                "ffprobe": ffprobe_data,
                "volume": vol_data,
                "silence": silence_data,
                "spectral": spectral_data
            }
        }

    except Exception as e:
        logger.error(f"Failed: {e}")
        return {
            "success": False,
            "method": "file_upload",
            "error": str(e),
            "model": model
        }


async def stream_audio_file(audio_file: str, language: str = "auto", model: str = "nova-3", encoding: str = "linear16", sample_rate: int = None, channels: int = None) -> str:
    """
    Stream an audio file to Deepgram for real-time transcription.
    Based on the official Deepgram streaming example.
    """
    logger.info(f"🎵 Starting live streaming transcription of file: {audio_file}")
    logger.info(f"   Model: {model}, Language: {language}, Encoding: {encoding}")
    
    # Mimic sending a real-time stream by sending this many seconds of audio at a time
    REALTIME_RESOLUTION = 0.250
    encoding_samplewidth_map = {"linear16": 2, "mulaw": 1}
    
    try:
        # Read the audio file
        # For WAV files, we need to extract raw PCM data without the header
        if audio_file.lower().endswith('.wav'):
            try:
                with wave.open(audio_file, 'rb') as wav_file:
                    # Get WAV file parameters
                    wav_sample_rate = wav_file.getframerate()
                    wav_channels = wav_file.getnchannels()
                    wav_sample_width = wav_file.getsampwidth()
                    
                    # Read ALL raw PCM data for streaming
                    data = wav_file.readframes(wav_file.getnframes())
                    
                    # Use detected parameters if not provided
                    if sample_rate is None:
                        sample_rate = wav_sample_rate
                    if channels is None:
                        channels = wav_channels
                    sample_width = wav_sample_width
                    
                    logger.info(f"📊 WAV file info: {wav_sample_rate}Hz, {wav_channels} channel(s), {wav_sample_width} bytes/sample")
                    logger.info(f"🔧 Using streaming parameters: {sample_rate}Hz, {channels} channel(s), encoding: {encoding}")
                    logger.info(f"📊 Raw PCM data size: {len(data):,} bytes")
                    
                    # Read first 5 seconds of raw PCM data for analysis
                    frames_per_second = wav_sample_rate * wav_channels
                    max_frames = frames_per_second * 5  # 5 seconds
                    analysis_data = data[:max_frames * wav_sample_width]
            except Exception as e:
                logger.error(f"❌ Error reading WAV file: {e}")
                # Fallback to reading entire file
                with open(audio_file, "rb") as f:
                    data = f.read()
                analysis_data = data[:sample_rate * channels * sample_width * 5]  # First 5 seconds
        else:
            # For non-WAV files, read the entire file
            with open(audio_file, "rb") as f:
                data = f.read()
            analysis_data = data[:sample_rate * channels * sample_width * 5]  # First 5 seconds
        
        logger.info(f"📊 Audio data size for streaming: {len(data):,} bytes")
        
        # Ensure we have valid parameters (fallback to defaults if still None)
        if sample_rate is None:
            sample_rate = 16000
            logger.warning("⚠️  Using fallback sample rate: 16000 Hz")
        if channels is None:
            channels = 1
            logger.warning("⚠️  Using fallback channels: 1 (mono)")
        
        # Analyze audio data before streaming
        # For analysis, we use the actual WAV file's parameters, not the streaming parameters
        streaming_sample_width = encoding_samplewidth_map.get(encoding, 2)
        logger.info(f"🔍 Analyzing audio data (sample_rate={wav_sample_rate}, channels={wav_channels}, sample_width={wav_sample_width})...")
        
        audio_analysis = analyze_audio_data(analysis_data, wav_sample_rate, wav_channels, wav_sample_width)
        
        logger.info(f"📈 Audio Analysis: {audio_analysis.get('analysis_summary', 'Analysis failed')}")
        
        if 'error' in audio_analysis:
            logger.warning(f"⚠️  Audio analysis error: {audio_analysis['error']}")
        elif audio_analysis.get('is_silent', False):
            logger.warning("⚠️  WARNING: Audio appears to be silent or very quiet!")
        elif not audio_analysis.get('likely_contains_speech', False):
            logger.warning("⚠️  WARNING: Audio may not contain detectable speech!")
            logger.warning(f"    - RMS Level: {audio_analysis.get('rms_db', 'N/A'):.1f} dB")
            logger.warning(f"    - Dynamic Range: {audio_analysis.get('dynamic_range_db', 'N/A'):.1f} dB")
            logger.warning(f"    - Zero Crossing Rate: {audio_analysis.get('zero_crossing_rate', 'N/A'):.3f}")
        else:
            logger.info("✅ Audio analysis suggests speech is present")
        
        # Build WebSocket URL
        url = f"wss://api.deepgram.com/v1/listen?encoding={encoding}&sample_rate={sample_rate}&channels={channels}"
        if language == "auto":
            # WebSocket streaming doesn't support detect_language parameter
            # For now, default to multi-language support by adding common languages
            # This allows the model to detect Spanish, English, etc.
            logger.info("ℹ️  Using multi-language support for automatic language detection in streaming")
            url += "&language=multi"
        else:
            url += f"&language={language}"
        # Always add model parameter to ensure consistency
        url += f"&model={model}"
        
        # Add additional streaming parameters
        url += "&punctuate=true&interim_results=true&endpointing=300"
        
        logger.info(f"🔗 Connecting to Deepgram streaming API...")
        
        # Create headers for authorization
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}"
        }
        
        async with websockets.connect(url, additional_headers=headers) as ws:
            logger.info("🟢 Successfully opened streaming connection")
            
            # Store transcription results
            transcripts = []
            final_transcript = ""
            
            async def sender(ws):
                logger.info("🟢 Ready to stream audio data")
                nonlocal data
                
                # Calculate chunk size for real-time simulation
                sample_width = encoding_samplewidth_map.get(encoding, 2)
                byte_rate = sample_width * sample_rate * channels
                chunk_size = int(byte_rate * REALTIME_RESOLUTION)
                
                chunk_count = 0
                while len(data):
                    chunk, data = data[:chunk_size], data[chunk_size:]
                    chunk_count += 1
                    
                    # Simulate real-time by waiting
                    await asyncio.sleep(REALTIME_RESOLUTION)
                    
                    # Send the data
                    await ws.send(chunk)
                    
                    if chunk_count % 20 == 0:  # Log every 5 seconds
                        logger.info(f"   Streamed {chunk_count * REALTIME_RESOLUTION:.1f}s of audio...")
                
                # Close the stream
                await ws.send(json.dumps({"type": "CloseStream"}))
                logger.info("🟢 Successfully closed stream, waiting for final results")
                return
            
            async def receiver(ws):
                nonlocal transcripts, final_transcript
                first_message = True
                message_count = 0
                
                async for msg in ws:
                    message_count += 1
                    if first_message:
                        logger.info("🟢 Successfully receiving transcription results")
                        first_message = False

                    try:
                        res = json.loads(msg)

                        # Extract transcript from Deepgram response
                        channel = res.get("channel", {})
                        alternatives = channel.get("alternatives", [])

                        if alternatives:
                            transcript = alternatives[0].get("transcript", "")
                            confidence = alternatives[0].get("confidence", 0.0)
                            # FIX: is_final is at root level, not in channel
                            is_final = res.get("is_final", False)

                            if transcript.strip():
                                if is_final:
                                    logger.info(f"📝 Final: {transcript} (confidence: {confidence:.3f})")
                                    transcripts.append({
                                        "transcript": transcript,
                                        "confidence": confidence,
                                        "is_final": True
                                    })
                                    final_transcript += transcript + " "
                                else:
                                    logger.info(f"📝 Interim: {transcript}")

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse message: {msg[:100]}... Error: {e}")

                logger.info(f"✅ Streaming finished, processed {message_count} messages")
                return
            
            # Run sender and receiver concurrently
            functions = [
                asyncio.ensure_future(sender(ws)),
                asyncio.ensure_future(receiver(ws)),
            ]
            await asyncio.gather(*functions)
            
            # Return only the transcript text
            return final_transcript.strip()
            
    except Exception as e:
        logger.error(f"❌ Live streaming error: {str(e)}")
        return ""


async def stream_microphone(duration: int = 10, language: str = "auto", model: str = "nova-3") -> str:
    """
    Stream audio from microphone to Deepgram for real-time transcription.
    """
    logger.info(f"🎤 Starting live microphone streaming for {duration} seconds")
    logger.info(f"   Model: {model}, Language: {language}")
    
    # Audio configuration
    RATE = 16000
    CHUNK = 1024
    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    
    try:
        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        
        # Check if microphone is available
        try:
            stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            logger.info("🟢 Microphone initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize microphone: {str(e)}")
            audio.terminate()
            return ""
        
        # Build WebSocket URL
        url = f"wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate={RATE}&channels={CHANNELS}"
        if language != "auto":
            url += f"&language={language}"
        if model != "nova-3":
            url += f"&model={model}"
        
        # Add additional streaming parameters
        url += "&punctuate=true&interim_results=true&endpointing=300"
        
        logger.info(f"🔗 Connecting to Deepgram streaming API...")
        
        # Create headers for authorization
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}"
        }
        
        async with websockets.connect(url, additional_headers=headers) as ws:
            logger.info("🟢 Successfully opened streaming connection")
            logger.info(f"🎤 Recording for {duration} seconds... Speak now!")
            
            # Store transcription results
            transcripts = []
            final_transcript = ""
            
            async def sender(ws):
                nonlocal stream, audio
                start_time = time.time()
                
                try:
                    while time.time() - start_time < duration:
                        # Read audio data from microphone
                        data = stream.read(CHUNK, exception_on_overflow=False)
                        await ws.send(data)
                        await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
                    
                    # Close the stream
                    await ws.send(json.dumps({"type": "CloseStream"}))
                    logger.info("🟢 Recording completed, processing final results...")
                    
                finally:
                    # Clean up audio resources
                    stream.stop_stream()
                    stream.close()
                    audio.terminate()
                
                return
            
            async def receiver(ws):
                nonlocal transcripts, final_transcript
                first_message = True
                
                async for msg in ws:
                    if first_message:
                        logger.info("🟢 Successfully receiving transcription results")
                        first_message = False
                    
                    try:
                        res = json.loads(msg)

                        # Validate response structure
                        if "channel" not in res:
                            logger.warning(f"⚠️  Response missing 'channel' key. Available keys: {list(res.keys())}")
                            logger.debug(f"Full response: {json.dumps(res, indent=2)}")
                            continue

                        # Extract transcript from Deepgram response
                        channel = res.get("channel", {})
                        alternatives = channel.get("alternatives", [])

                        if not alternatives:
                            logger.debug(f"No alternatives found in channel")
                            continue

                        if alternatives:
                            transcript = alternatives[0].get("transcript", "")
                            confidence = alternatives[0].get("confidence", 0.0)
                            # FIX: is_final is at root level, not in channel
                            is_final = res.get("is_final", False)

                            if transcript.strip():
                                if is_final:
                                    logger.info(f"📝 Final: {transcript} (confidence: {confidence:.3f})")
                                    transcripts.append({
                                        "transcript": transcript,
                                        "confidence": confidence,
                                        "is_final": True
                                    })
                                    final_transcript += transcript + " "
                                else:
                                    logger.info(f"📝 Interim: {transcript}")
                            else:
                                logger.debug(f"Empty transcript in response (is_final={is_final})")

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse message: {msg[:100]}... Error: {e}")
                
                return
            
            # Run sender and receiver concurrently
            functions = [
                asyncio.ensure_future(sender(ws)),
                asyncio.ensure_future(receiver(ws)),
            ]
            await asyncio.gather(*functions)
            
            # Return only the transcript text
            return final_transcript.strip()
            
    except Exception as e:
        logger.error(f"❌ Live microphone streaming error: {str(e)}")
        # Clean up audio resources in case of error
        try:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            if 'audio' in locals():
                audio.terminate()
        except:
            pass
        
        return ""


def test_live_streaming(audio_source: str, language: str = "auto", model: str = "nova-3", duration: int = 10) -> str:
    """
    Test live streaming functionality with either a file, URL, or microphone.
    
    Args:
        audio_source: Either a file path, URL, or "microphone"
        language: Language code or "auto"
        model: Deepgram model to use
        duration: Duration in seconds for microphone recording
    """
    logger.info("=" * 80)
    logger.info("🎵 DEEPGRAM LIVE STREAMING TEST")
    logger.info("=" * 80)
    
    if audio_source.lower() == "microphone" or audio_source.lower() == "mic":
        # Test microphone streaming
        transcript = asyncio.run(stream_microphone(duration, language, model))
    else:
        # Check if it's a URL or local file
        if audio_source.startswith(('http://', 'https://')):
            logger.info(f"🌐 Downloading audio from URL for live streaming: {audio_source}")
            # Download the audio file first (limited to 30 seconds for testing)
            downloaded_file = download_audio_with_ytdlp(audio_source, duration=30)
            if not downloaded_file:
                logger.error(f"❌ Failed to download audio from URL: {audio_source}")
                return ""
            audio_source = downloaded_file
            logger.info(f"✅ Downloaded audio file: {audio_source}")

        # Test file streaming
        if not os.path.exists(audio_source):
            logger.error(f"❌ Audio file not found: {audio_source}")
            return ""

        # Use the audio file directly without re-encoding to prevent corruption
        logger.info(f"📁 Using audio file directly for streaming: {audio_source}")

        transcript = asyncio.run(stream_audio_file(audio_source, language, model))

    # Display the final transcript result
    logger.info(f"\n✅ Live Streaming Final Transcript Result:")
    if transcript:
        logger.info(f"   Length: {len(transcript)} characters")
        logger.info(f"   {transcript[:500]}")
        if len(transcript) > 500:
            logger.info(f"   ... [showing first 500 of {len(transcript)} characters]")
    else:
        logger.warning("   ⚠️  EMPTY TRANSCRIPT - Check WebSocket logs above for details")

    return transcript


def test_all_methods(url: str, language: str = "auto", model: str = "nova-3", duration: int = 30) -> dict:
    """
    Test all transcription methods with the SAME audio file for fair comparison.
    
    Order:
    1. File Upload (BASELINE - Most Reliable)
    2. Live Streaming (using same file)
    3. Direct URL Streaming (if applicable)
    
    Args:
        url: Audio file URL or local file path
        language: Language code or "auto"
        model: Deepgram model to use
        duration: Duration limit for downloads (seconds)
    """
    logger.info("=" * 80)
    logger.info("🎯 DEEPGRAM ALL METHODS COMPARISON TEST")
    logger.info("=" * 80)
    
    results = {
        "url": url,
        "language": language,
        "model": model,
        "methods": {},
        "summary": {}
    }
    
    # Step 1: Get the audio file (download once, use everywhere)
    logger.info("\n" + "=" * 60)
    logger.info("📥 STEP 1: PREPARING AUDIO FILE")
    logger.info("=" * 60)
    
    audio_file = None
    if url.startswith(('http://', 'https://')):
        logger.info(f"🌐 Downloading audio from URL (duration: {duration}s)...")
        audio_file = download_audio_with_ytdlp(url, duration=duration)
        if not audio_file:
            logger.error("❌ Download failed - cannot proceed with tests")
            return {
                "url": url,
                "language": language,
                "model": model,
                "methods": {},
                "summary": {"error": "Download failed", "success_rate": 0}
            }
        logger.info(f"✅ Audio downloaded: {audio_file}")
    else:
        # Local file
        if os.path.exists(url):
            audio_file = url
            logger.info(f"✅ Using local file: {audio_file}")
        else:
            logger.error(f"❌ Local file not found: {url}")
            return {
                "url": url,
                "language": language,
                "model": model,
                "methods": {},
                "summary": {"error": "File not found", "success_rate": 0}
            }
    
    # Method 1: File Upload (BASELINE - Most Reliable)
    logger.info("\n" + "=" * 60)
    logger.info("📁 METHOD 1: FILE UPLOAD (BASELINE)")
    logger.info("=" * 60)
    logger.info("ℹ️  Using Deepgram File API - most reliable method")
    
    results["methods"]["file_upload"] = transcribe_file_method(audio_file, language, model, test_variants=False)
    
    # Show baseline result immediately
    baseline_transcript = ""
    baseline_confidence = 0
    if results["methods"]["file_upload"].get("success", False):
        baseline_transcript = results["methods"]["file_upload"].get("transcript", "").strip()
        baseline_confidence = results["methods"]["file_upload"].get("confidence", 0)
        logger.info(f"✅ BASELINE RESULT:")
        logger.info(f"   Confidence: {baseline_confidence:.2f}")
        logger.info(f"   Transcript: {baseline_transcript}")
        logger.info(f"   📋 This will be used as reference for comparison")
    else:
        logger.error("❌ BASELINE FAILED - File upload method failed!")
        error = results["methods"]["file_upload"].get("error", "Unknown error")
        logger.error(f"   Error: {error}")
    
    # Method 2: Live Streaming (using same file)
    logger.info("\n" + "=" * 60)
    logger.info("🔄 METHOD 2: LIVE STREAMING")
    logger.info("=" * 60)
    logger.info("ℹ️  Using same audio file with WebSocket streaming")
    
    transcript = asyncio.run(stream_audio_file(audio_file, language, model))
    results["methods"]["live_streaming"] = {
        "success": bool(transcript),
        "method": "live_streaming",
        "transcript": transcript,
        "confidence": 1.0 if transcript else 0.0,
        "model": model,
        "character_count": len(transcript) if transcript else 0
    }
    
    # Method 3: Direct URL Transcription (if URL provided)
    if url.startswith(('http://', 'https://')):
        logger.info("\n" + "=" * 60)
        logger.info("🌐 METHOD 3: DIRECT URL TRANSCRIPTION")
        logger.info("=" * 60)
        logger.info("ℹ️  Using Deepgram URL API - sends URL directly without downloading")

        try:
            url_result = transcribe_url_method(url, language, model)

            if url_result.get("success"):
                transcript = url_result.get("transcript", "")
                confidence = url_result.get("confidence", 0.0)

                logger.info(f"\n✅ Direct URL Transcription Result:")
                logger.info(f"   Confidence: {confidence:.2f}")
                if transcript:
                    logger.info(f"   Transcript: {transcript[:500]}")
                    if len(transcript) > 500:
                        logger.info(f"   ... [showing first 500 of {len(transcript)} characters]")
                else:
                    logger.warning("   ⚠️  EMPTY TRANSCRIPT")

                results["methods"]["direct_url"] = {
                    "success": True,
                    "method": "direct_url",
                    "transcript": transcript,
                    "confidence": confidence,
                    "model": model
                }
            else:
                error_msg = url_result.get("error", "Unknown error")
                logger.error(f"❌ Direct URL transcription failed: {error_msg}")
                results["methods"]["direct_url"] = {
                    "success": False,
                    "method": "direct_url",
                    "error": error_msg,
                    "model": model
                }
        except Exception as e:
            logger.error(f"❌ Direct URL transcription error: {str(e)}")
            results["methods"]["direct_url"] = {
                "success": False,
                "method": "direct_url",
                "error": str(e),
                "model": model
            }
    
    # Clean up downloaded file (only if we downloaded it)
    if url.startswith(('http://', 'https://')) and audio_file:
        try:
            if os.path.exists(audio_file):
                os.remove(audio_file)
                logger.info(f"🗑️  Cleaned up downloaded file: {audio_file}")
        except Exception as e:
            logger.warning(f"⚠️  Could not clean up file {audio_file}: {e}")
    
    # Generate comparison summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 COMPARISON RESULTS")
    logger.info("=" * 60)
    
    successful_methods = []
    failed_methods = []
    
    # Show baseline first
    if results["methods"]["file_upload"].get("success", False):
        successful_methods.append("file_upload")
        logger.info(f"✅ FILE UPLOAD (BASELINE): Success")
        logger.info(f"   Confidence: {baseline_confidence:.2f}")
        logger.info(f"   Transcript: {baseline_transcript}")
    else:
        failed_methods.append("file_upload")
        error = results["methods"]["file_upload"].get("error", "Unknown error")
        logger.info(f"❌ FILE UPLOAD (BASELINE): Failed - {error}")
    
    # Compare other methods to baseline
    for method_name, result in results["methods"].items():
        if method_name == "file_upload":
            continue  # Already shown above
            
        if result.get("success", False):
            successful_methods.append(method_name)
            transcript = result.get("transcript", "").strip()
            confidence = result.get("confidence", 0)
            
            # Compare to baseline
            matches_baseline = transcript.lower().strip() == baseline_transcript.lower().strip()
            match_indicator = "✅ MATCHES BASELINE" if matches_baseline else "⚠️  DIFFERS FROM BASELINE"
            
            logger.info(f"✅ {method_name.upper()}: Success (confidence: {confidence:.2f})")
            logger.info(f"   {match_indicator}")
            logger.info(f"   Transcript: {transcript}")
            
            if not matches_baseline and baseline_transcript:
                logger.info(f"   📋 Expected: {baseline_transcript}")
        else:
            failed_methods.append(method_name)
            error = result.get("error", "Unknown error")
            logger.info(f"❌ {method_name.upper()}: Failed - {error}")
    
    results["summary"] = {
        "successful_methods": successful_methods,
        "failed_methods": failed_methods,
        "total_methods": len(results["methods"]),
        "success_rate": len(successful_methods) / len(results["methods"]) if results["methods"] else 0,
        "baseline_transcript": baseline_transcript
    }
    
    logger.info(f"\n🎯 Overall Success Rate: {len(successful_methods)}/{len(results['methods'])} methods")
    
    return results


def test_transcription(url: str, language: str = "auto", model: str = "nova-3", test_variants: bool = False):
    """
    Test complete transcription flow with enhanced diagnostics
    """
    logger.info("=" * 80)
    logger.info("ENHANCED DEEPGRAM TRANSCRIPTION TEST")
    logger.info("=" * 80)
    logger.info(f"URL: {url}")
    logger.info(f"Language: {language}")
    logger.info(f"Model: {model}")
    logger.info(f"Test variants: {test_variants}")

    # Check if this is a streaming URL that Deepgram cannot process directly
    if is_streaming_url(url):
        logger.info("\n" + "=" * 80)
        logger.info("STREAMING URL DETECTED - SKIPPING DIRECT URL METHOD")
        logger.info("=" * 80)
        logger.info("This appears to be an HLS/streaming URL (.m3u8, /hls/, etc.)")
        logger.info("Deepgram cannot process streaming URLs directly.")
        logger.info("Proceeding directly to download + upload method...")
        result1 = {"success": False, "method": "url", "error": "Streaming URL detected - skipped direct transcription"}
    else:
        result1 = transcribe_url_method(url, language, model)

    if result1["success"] and result1["length"] > 10:  # Only consider success if we got substantial content
        logger.info("\n" + "=" * 80)
        logger.info("URL METHOD SUCCEEDED")
        logger.info("=" * 80)
        logger.info(f"Request ID: {result1.get('request_id', 'N/A')}")
        logger.info(f"Transcript ({result1['length']} chars):")
        logger.info("-" * 80)
        logger.info(result1["transcript"][:500])
        if result1["length"] > 500:
            logger.info(f"\n... [truncated, showing first 500 of {result1['length']} characters]")
        return result1

    logger.info("\n" + "=" * 80)
    logger.info("URL method failed or insufficient, trying download + upload fallback...")
    logger.info("=" * 80)

    audio_file = download_audio_with_ytdlp(url, format='m4a')

    if not audio_file:
        logger.error("\n" + "=" * 80)
        logger.error("BOTH METHODS FAILED")
        logger.error("=" * 80)
        return {
            "success": False,
            "error": "Both URL and download methods failed"
        }

    result2 = transcribe_file_method(audio_file, language, model, test_variants)

    try:
        os.remove(audio_file)
        os.rmdir(os.path.dirname(audio_file))
    except (OSError, FileNotFoundError):
        pass

    logger.info("\n" + "=" * 80)
    if result2["success"]:
        logger.info("FILE UPLOAD METHOD SUCCEEDED")
        logger.info("=" * 80)
        logger.info(f"Best method: {result2.get('best_method', 'N/A')}")
        logger.info(f"Request ID: {result2.get('request_id', 'N/A')}")
        logger.info(f"Confidence: {result2.get('confidence', 'N/A')}")
        logger.info(f"Transcript ({result2['length']} chars):")
        logger.info("-" * 80)
        logger.info(result2["transcript"][:500])
        if result2["length"] > 500:
            logger.info(f"\n... [truncated, showing first 500 of {result2['length']} characters]")
    else:
        logger.error("FILE UPLOAD METHOD ALSO FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {result2.get('error', 'Unknown')}")

    return result2


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Check if this is a live streaming command
        if sys.argv[1].lower() in ['stream', 'live', 'streaming']:
            # Parse arguments - use default URL if not provided
            if len(sys.argv) < 3:
                audio_source = DEFAULT_TEST_URL
                logger.info(f"🎯 No URL provided, using default test URL")
            else:
                audio_source = sys.argv[2]
            test_language = "auto"
            test_model = "nova-3"
            duration = 10
            
            # Parse named arguments
            i = 3
            while i < len(sys.argv):
                if sys.argv[i] == '--language' and i + 1 < len(sys.argv):
                    test_language = sys.argv[i + 1]
                    i += 2
                elif sys.argv[i] == '--model' and i + 1 < len(sys.argv):
                    test_model = sys.argv[i + 1]
                    i += 2
                elif sys.argv[i] == '--duration' and i + 1 < len(sys.argv):
                    try:
                        duration = int(sys.argv[i + 1])
                    except ValueError:
                        logger.error(f"Invalid duration value: {sys.argv[i + 1]}")
                        sys.exit(1)
                    i += 2
                else:
                    # Support positional arguments for backward compatibility
                    if i == 3:  # language
                        test_language = sys.argv[i]
                    elif i == 4:  # model
                        test_model = sys.argv[i]
                    elif i == 5:  # duration
                        try:
                            duration = int(sys.argv[i])
                        except ValueError:
                            logger.error(f"Invalid duration value: {sys.argv[i]}")
                            sys.exit(1)
                    i += 1
            
            transcript = test_live_streaming(audio_source, test_language, test_model, duration)
            if transcript:
                logger.info(f"📝 Transcript: {transcript}")
                sys.exit(0)
            else:
                logger.error("❌ No transcript received")
                sys.exit(1)
        
        # Check if this is an all methods comparison command
        elif sys.argv[1].lower() in ['all', 'compare', 'methods']:
            # Parse arguments - use default URL if not provided
            if len(sys.argv) < 3:
                test_url = DEFAULT_TEST_URL
                logger.info(f"🎯 No URL provided, using default test URL")
            else:
                test_url = sys.argv[2]
            test_language = "auto"
            test_model = "nova-3"
            duration = 30
            
            # Parse named arguments
            i = 3
            while i < len(sys.argv):
                if sys.argv[i] == '--language' and i + 1 < len(sys.argv):
                    test_language = sys.argv[i + 1]
                    i += 2
                elif sys.argv[i] == '--model' and i + 1 < len(sys.argv):
                    test_model = sys.argv[i + 1]
                    i += 2
                elif sys.argv[i] == '--duration' and i + 1 < len(sys.argv):
                    try:
                        duration = int(sys.argv[i + 1])
                    except ValueError:
                        logger.error(f"Invalid duration value: {sys.argv[i + 1]}")
                        sys.exit(1)
                    i += 2
                else:
                    # Support positional arguments for backward compatibility
                    if i == 3:  # language
                        test_language = sys.argv[i]
                    elif i == 4:  # model
                        test_model = sys.argv[i]
                    elif i == 5:  # duration
                        try:
                            duration = int(sys.argv[i])
                        except ValueError:
                            logger.error(f"Invalid duration value: {sys.argv[i]}")
                            sys.exit(1)
                    i += 1
            
            result = test_all_methods(test_url, test_language, test_model, duration)
            success = result["summary"]["success_rate"] > 0
            sys.exit(0 if success else 1)
        
        # Regular transcription mode
        test_url = sys.argv[1]
        test_language = sys.argv[2] if len(sys.argv) > 2 else "auto"
        test_model = sys.argv[3] if len(sys.argv) > 3 else "nova-3"
        test_variants_flag = len(sys.argv) > 4 and sys.argv[4].lower() in ['true', '1', 'yes', 'variants']
        
        result = test_transcription(test_url, test_language, test_model, test_variants_flag)
        sys.exit(0 if result["success"] else 1)
    else:
        logger.error("Usage:")
        logger.error("  Regular transcription: python test_deepgram.py <URL> [language] [model] [test_variants]")
        logger.error("  Live streaming: python test_deepgram.py stream [audio_source] [language] [model] [duration]")
        logger.error("  All methods comparison: python test_deepgram.py all [URL] [language] [model] [duration]")
        logger.error("")
        logger.error("Regular transcription:")
        logger.error("  URL: Audio file URL or local file path")
        logger.error("  Language: auto (default, uses detect_language=True), en, es, fr, etc.")
        logger.error("  Models: nova-3 (default), enhanced, base")
        logger.error("  test_variants: true/false (default: false)")
        logger.error("")
        logger.error("Live streaming:")
        logger.error("  audio_source: file path, URL, or 'microphone'/'mic' (optional, uses default test URL if not provided)")
        logger.error("  language: auto (default), en, es, fr, etc.")
        logger.error("  model: nova-3 (default), enhanced, base")
        logger.error("  duration: seconds for microphone recording (default: 10)")
        logger.error("")
        logger.error("All methods comparison:")
        logger.error("  URL: Audio file URL or local file path (optional, uses default test URL if not provided)")
        logger.error("  language: auto (default), en, es, fr, etc.")
        logger.error("  model: nova-3 (default), enhanced, base")
        logger.error("  duration: seconds for download duration limit (default: 30)")
        logger.error("  Tests: 1) Download + Upload (File API), 2) Download + Live Streaming")
        logger.error("")
        logger.error("Examples:")
        logger.error("  python test_deepgram.py 'https://example.com/audio.mp3'")
        logger.error("  python test_deepgram.py stream                    # Uses default test URL")
        logger.error("  python test_deepgram.py stream microphone")
        logger.error("  python test_deepgram.py stream audio.wav es enhanced")
        logger.error("  python test_deepgram.py stream 'https://example.com/audio.mp3' --language auto --model nova-3")
        logger.error("  python test_deepgram.py all                      # Compare all methods with default URL")
        logger.error("  python test_deepgram.py all 'https://example.com/audio.mp3' auto nova-3 60")
        sys.exit(1)

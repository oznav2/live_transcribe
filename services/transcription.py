"""Transcription services for various models and streaming."""
import asyncio
import logging
import os
import time
import json
import tempfile
import shutil
import subprocess
import queue
import threading
from typing import Optional, Tuple, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from config.settings import (
    DEEPGRAM_API_KEY, DEEPGRAM_MODEL, DEEPGRAM_LANGUAGE,
    DEEPGRAM_TIME_LIMIT, DEEPGRAM_TRANSCRIPT_ONLY
)
from config.availability import FASTER_WHISPER_AVAILABLE, OPENAI_WHISPER_AVAILABLE, DEEPGRAM_AVAILABLE, MODEL_CONFIGS
from config.constants import SAMPLE_RATE, CHANNELS
from core.state import whisper_models, model_lock, executor
from models.loader import load_model
from services.audio_processor import (
    AudioStreamProcessor,
    get_audio_duration_seconds,
    calculate_progress_metrics,
    split_audio_for_incremental,
)
from utils.websocket_helpers import safe_ws_send
from utils.cache import generate_cache_key, get_cached_audio, save_to_cache, DOWNLOAD_CACHE_DIR
from utils.validators import should_use_ytdlp, sanitize_url
from services.audio_processor import download_audio_with_ffmpeg
from utils.cleantext import (
    clean_transcription_text,
    IncrementalDeduplicator,
    remove_repeated_word_sequences,
    split_into_sentences,
    remove_consecutive_duplicate_sentences,
)

logger = logging.getLogger(__name__)

# Import model libraries conditionally
if FASTER_WHISPER_AVAILABLE:
    import faster_whisper
    
if OPENAI_WHISPER_AVAILABLE:
    try:
        import whisper
        import torch
    except ImportError:
        whisper = None
        torch = None
        OPENAI_WHISPER_AVAILABLE = False
    
if DEEPGRAM_AVAILABLE:
	# Import Deepgram client
	try:
		from deepgram import DeepgramClient
	except ImportError:
		DeepgramClient = None

	# Map events for Deepgram SDK v3/v4 (matching original robust pattern)
	DG_EVENT_OPEN = DG_EVENT_CLOSE = DG_EVENT_ERROR = DG_EVENT_MESSAGE = None
	try:
		# v3 style events
		from deepgram.core.events import EventType as _DGEventEnum
		DG_EVENT_OPEN = getattr(_DGEventEnum, "OPEN", None)
		DG_EVENT_MESSAGE = getattr(_DGEventEnum, "MESSAGE", None)
		DG_EVENT_CLOSE = getattr(_DGEventEnum, "CLOSE", None)
		DG_EVENT_ERROR = getattr(_DGEventEnum, "ERROR", None)
	except Exception:
		# v4 style events
		try:
			from deepgram.clients.common.v1.websocket_events import WebSocketEvents
			# TitleCase names in v4
			DG_EVENT_OPEN = getattr(WebSocketEvents, "Open", None)
			DG_EVENT_CLOSE = getattr(WebSocketEvents, "Close", None)
			DG_EVENT_ERROR = getattr(WebSocketEvents, "Error", None)
		except Exception:
			pass
		try:
			from deepgram.clients.listen.enums import LiveTranscriptionEvents as _DGLiveEvents
			# Prefer 'Transcript'; fall back to other possible names
			DG_EVENT_MESSAGE = (
				getattr(_DGLiveEvents, "Transcript", None)
				or getattr(_DGLiveEvents, "TranscriptReceived", None)
				or getattr(_DGLiveEvents, "Results", None)
			)
		except Exception:
			pass


async def transcribe_with_incremental_output(
    model, model_config: dict, audio_file: str, language: Optional[str], 
    websocket: WebSocket, model_name: str = None, chunk_seconds: int = 60
) -> Tuple[str, str]:
    """
    Transcribe audio with incremental output and detailed progress.
    
    Args:
        model: The loaded transcription model
        model_config: Model configuration dictionary
        audio_file: Path to audio file
        language: Language code or None for auto-detect
        websocket: WebSocket connection for progress updates
        model_name: Name of the model (needed for language detection logic)
        chunk_seconds: Size of audio chunks in seconds
        
    Returns:
        (full_transcript, detected_language)
    """
    import time
    start_time = time.time()
    
    # Get audio duration for progress calculation
    audio_duration = get_audio_duration_seconds(audio_file)
    if not audio_duration:
        logger.warning("Could not determine audio duration, using fallback progress")
        audio_duration = 0
    
    # Send initial status with audio duration
    await websocket.send_json({
        "type": "transcription_status",
        "message": f"Starting transcription of {audio_duration:.1f}s audio...",
        "audio_duration": audio_duration,
        "elapsed_seconds": 0
    })
    
    # For short audio (< 2 minutes), process as single chunk
    if audio_duration and audio_duration < 120:
        logger.info(f"Short audio ({audio_duration}s), processing as single chunk")
        
        # Transcribe entire file
        if model_config["type"] == "faster_whisper":
            # Use faster_whisper for Ivrit CT2 models
            fw_model = model["model"] if isinstance(model, dict) else model
            
            def run_fw_transcription():
                try:
                    # For Ivrit models, use Hebrew as default only if it's an Ivrit model
                    # For general models like whisper-v3-turbo, let it auto-detect
                    default_lang = None
                    if model_name and "ivrit" in model_name.lower():
                        # Ivrit models are Hebrew-optimized, default to Hebrew if no language specified
                        default_lang = "he"
                    
                    segments, info = fw_model.transcribe(
                        audio_file,
                        language=language or default_lang,  # None means auto-detect
                        beam_size=int(os.getenv("IVRIT_BEAM_SIZE", "1")),  # Reduced from 5 to 1 for speed
                        best_of=1,  # Reduced from 5 to 1 for speed
                        patience=1,
                        length_penalty=1,
                        temperature=0,
                        compression_ratio_threshold=2.4,
                        log_prob_threshold=-1.0,
                        no_speech_threshold=0.6,
                        word_timestamps=False,
                        vad_filter=True,  # Enable VAD for faster processing
                        vad_parameters=dict(
                            threshold=0.5,
                            min_speech_duration_ms=250,
                            max_speech_duration_s=float('inf'),
                            min_silence_duration_ms=2000,
                            speech_pad_ms=400
                        )
                    )
                    # Collect segments into text
                    text_parts = []
                    for segment in segments:
                        text_parts.append(segment.text)
                    return {
                        'text': ' '.join(text_parts),
                        'language': info.language if info else (language or 'he')
                    }
                except Exception as e:
                    logger.error(f"faster_whisper transcription failed: {e}")
                    raise
            
            # Run with progress monitoring
            loop = asyncio.get_event_loop()
            task = loop.run_in_executor(None, run_fw_transcription)
            
            while not task.done():
                await asyncio.sleep(2)
                elapsed = time.time() - start_time
                metrics = calculate_progress_metrics(audio_duration, elapsed)

                await websocket.send_json({
                    "type": "transcription_progress",
                    "audio_duration": audio_duration,
                    "percentage": metrics["percent_complete"],
                    "eta_seconds": metrics["estimated_time_remaining"],
                    "speed": metrics["processing_speed"],
                    "elapsed_seconds": int(elapsed)
                })

            result = await task
            transcript = result.get('text', '').strip()
            detected_language = result.get('language', language or 'he')

        elif model_config["type"] == "openai":
            def run_transcription():
                use_fp16 = torch.cuda.is_available()
                try:
                    return model.transcribe(audio_file, language=language, fp16=use_fp16, verbose=False)
                except Exception as e:
                    logger.warning(f"GPU/FP16 failed: {e}. Retrying with fp16=False.")
                    return model.transcribe(audio_file, language=language, fp16=False, verbose=False)

            # Run with progress monitoring
            loop = asyncio.get_event_loop()
            task = loop.run_in_executor(None, run_transcription)

            while not task.done():
                await asyncio.sleep(2)
                elapsed = time.time() - start_time
                metrics = calculate_progress_metrics(audio_duration, elapsed)

                await websocket.send_json({
                    "type": "transcription_progress",
                    "audio_duration": audio_duration,
                    "percentage": metrics["percent_complete"],
                    "eta_seconds": metrics["estimated_time_remaining"],
                    "speed": metrics["processing_speed"],
                    "elapsed_seconds": int(elapsed)
                })
            
            result = await task
            transcript = result.get('text', '').strip()
            detected_language = result.get('language', language or 'unknown')
            # Apply deduplication to single-chunk transcript before sending
            if transcript:
                try:
                    # Single chunk: apply robust cleaning
                    transcript = remove_repeated_word_sequences(transcript, min_sequence_length=5)
                    sentences = split_into_sentences(transcript)
                    sentences = remove_consecutive_duplicate_sentences(sentences)
                    transcript = ''.join(sentences)
                except Exception:
                    # Fail-safe: leave transcript as-is
                    pass
            
            # Send final transcript
            if transcript:
                await websocket.send_json({
                    "type": "transcription_chunk",
                    "text": transcript,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "is_final": True
                })
            
            return transcript, detected_language
        

            
            if transcript:
                await websocket.send_json({
                    "type": "transcription_chunk",
                    "text": transcript,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "is_final": True
                })
            
            return transcript, detected_language
    
    # For longer audio, use chunked processing for incremental output
    logger.info(f"Long audio ({audio_duration}s), using chunked processing")
    
    try:
        # Split audio into chunks (no overlap to avoid duplication in transcript)
        temp_dir, chunk_files = split_audio_for_incremental(audio_file, chunk_seconds, overlap_seconds=0)
        total_chunks = len(chunk_files)
        
        logger.info(f"Split audio into {total_chunks} chunks")
        
        transcripts = []
        # Initialize stateful deduplicator to handle overlap across chunk boundaries
        deduper = IncrementalDeduplicator(context_window=20, min_sequence_length=5)
        detected_language = language or 'unknown'
        
        for i, chunk_file in enumerate(chunk_files):
            chunk_start = time.time()
            
            # Send progress BEFORE processing (so user sees it immediately)
            # Calculate preliminary progress
            elapsed_so_far = time.time() - start_time
            if i > 0:
                avg_time_per_chunk = elapsed_so_far / i
                remaining_chunks = total_chunks - i
                eta_seconds = int(avg_time_per_chunk * remaining_chunks)
            else:
                eta_seconds = 0  # Unknown for first chunk
            
            percentage_starting = int((i / total_chunks) * 100)
            
            # Send "starting chunk" progress
            await websocket.send_json({
                "type": "transcription_progress",
                "audio_duration": audio_duration,
                "percentage": percentage_starting,
                "eta_seconds": eta_seconds,
                "speed": f"{avg_time_per_chunk:.1f}s/chunk" if i > 0 else "calculating...",
                "elapsed_seconds": int(elapsed_so_far),
                "chunks_processed": i,
                "total_chunks": total_chunks,
                "message": f"Processing chunk {i+1}/{total_chunks}..."
            })
            
            # Transcribe chunk (run in executor to avoid blocking event loop)
            loop = asyncio.get_event_loop()
            
            if model_config["type"] == "faster_whisper":
                # Use faster_whisper for chunks
                fw_model = model["model"] if isinstance(model, dict) else model
                
                def transcribe_fw_chunk():
                    try:
                        # For Ivrit models, use Hebrew as default only if it's an Ivrit model
                        default_lang = None
                        if model_name and "ivrit" in model_name.lower():
                            # Ivrit models are Hebrew-optimized, default to Hebrew if no language specified
                            default_lang = "he"
                        
                        segments, info = fw_model.transcribe(
                            chunk_file,
                            language=language or default_lang,  # None means auto-detect
                            beam_size=1,  # Reduced for speed
                            best_of=1,  # Reduced for speed
                            patience=1,
                            temperature=0,
                            compression_ratio_threshold=2.4,
                            no_speech_threshold=0.6,
                            word_timestamps=False,
                            vad_filter=True,  # Enable VAD for speed
                            vad_parameters=dict(
                                threshold=0.5,
                                min_speech_duration_ms=250,
                                min_silence_duration_ms=1000
                            )
                        )
                        text_parts = []
                        for segment in segments:
                            text_parts.append(segment.text)
                        chunk_text = ' '.join(text_parts).strip()
                        return chunk_text, info
                    except Exception as e:
                        logger.error(f"faster_whisper chunk transcription failed: {e}")
                        return "", None
                
                # Run in executor (non-blocking)
                chunk_text, info = await loop.run_in_executor(None, transcribe_fw_chunk)
                
                if i == 0 and info and info.language:
                    detected_language = info.language
            
            elif model_config["type"] == "openai":
                def transcribe_openai_chunk():
                    use_fp16 = torch.cuda.is_available()
                    try:
                        result = model.transcribe(chunk_file, language=language, fp16=use_fp16, verbose=False)
                    except Exception:
                        result = model.transcribe(chunk_file, language=language, fp16=False, verbose=False)
                    return result
                
                # Run in executor (non-blocking)
                result = await loop.run_in_executor(None, transcribe_openai_chunk)
                
                chunk_text = result.get('text', '').strip()
                if i == 0 and result.get('language'):
                    detected_language = result.get('language')
            
            else:
                chunk_text = ""
            

            
            # Send incremental result (apply stateful per-chunk deduplication)
            if chunk_text:
                # Boundary-aware and intra-chunk deduplication
                try:
                    cleaned_chunk = deduper.process_chunk(chunk_text)
                    # Apply repeated word sequence removal for robustness
                    cleaned_chunk = remove_repeated_word_sequences(cleaned_chunk, min_sequence_length=5)
                    # Apply sentence-level consecutive duplicate removal
                    sentences = split_into_sentences(cleaned_chunk)
                    sentences = remove_consecutive_duplicate_sentences(sentences)
                    cleaned_chunk = ''.join(sentences)
                except Exception as e:
                    logger.debug(f"Chunk deduplication encountered issue, using original: {e}")
                    cleaned_chunk = chunk_text

                if cleaned_chunk:
                    await websocket.send_json({
                        "type": "transcription_chunk",
                        "text": cleaned_chunk,
                        "chunk_index": i,
                        "total_chunks": total_chunks,
                        "is_final": i == total_chunks - 1
                    })
                    transcripts.append(cleaned_chunk)
            
            # Calculate progress metrics
            chunk_duration = time.time() - chunk_start
            elapsed = time.time() - start_time
            
            # Calculate ETA based on average chunk processing time
            avg_time_per_chunk = elapsed / (i + 1)
            remaining_chunks = total_chunks - (i + 1)
            eta_seconds = int(avg_time_per_chunk * remaining_chunks)
            percentage = int(((i + 1) / total_chunks) * 100)
            
            # Send detailed progress update
            await websocket.send_json({
                "type": "transcription_progress",
                "audio_duration": audio_duration,
                "percentage": percentage,
                "eta_seconds": eta_seconds,
                "speed": f"{avg_time_per_chunk:.1f}s/chunk",
                "elapsed_seconds": int(elapsed),
                "chunks_processed": i + 1,
                "total_chunks": total_chunks,
                "message": f"Transcribing: {percentage}% (chunk {i+1}/{total_chunks}, ETA: {eta_seconds}s)"
            })
            
            logger.info(f"Processed chunk {i+1}/{total_chunks} in {chunk_duration:.1f}s (avg: {avg_time_per_chunk:.1f}s/chunk, ETA: {eta_seconds}s)")
        
        # Send final 100% completion message
        final_elapsed = time.time() - start_time
        await websocket.send_json({
            "type": "transcription_progress",
            "audio_duration": audio_duration,
            "percentage": 100,
            "eta_seconds": 0,
            "speed": f"{final_elapsed/total_chunks:.1f}s/chunk",
            "elapsed_seconds": int(final_elapsed),
            "chunks_processed": total_chunks,
            "total_chunks": total_chunks,
            "message": f"Transcription complete! Processed {total_chunks} chunks in {int(final_elapsed)}s"
        })
        logger.info(f"Transcription complete: {total_chunks} chunks in {final_elapsed:.1f}s")
        
        # Cleanup chunk files
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        
        # Join transcripts with space
        full_transcript = ' '.join(transcripts)

        # Apply text deduplication to remove any duplicate sentences or word sequences
        full_transcript = clean_transcription_text(full_transcript)

        return full_transcript, detected_language
        
    except Exception as e:
        logger.error(f"Incremental transcription error: {e}")
        # Fallback to single-file transcription
        logger.info("Falling back to single-file transcription")
        
        if model_config["type"] == "openai":
            use_fp16 = torch.cuda.is_available()
            try:
                result = model.transcribe(audio_file, language=language, fp16=use_fp16, verbose=False)
            except Exception:
                result = model.transcribe(audio_file, language=language, fp16=False, verbose=False)
            
            transcript = result.get('text', '').strip()
            detected_language = result.get('language', language or 'unknown')
        else:
            transcript = ""
            detected_language = language or 'unknown'
        
        if transcript:
            # Apply dedup to fallback single transcript
            try:
                transcript = remove_repeated_word_sequences(transcript, min_sequence_length=5)
                sentences = split_into_sentences(transcript)
                sentences = remove_consecutive_duplicate_sentences(sentences)
                transcript = ''.join(sentences)
            except Exception:
                pass
            await websocket.send_json({
                "type": "transcription_chunk",
                "text": transcript,
                "chunk_index": 0,
                "total_chunks": 1,
                "is_final": True
            })
        
        return transcript, detected_language


def transcribe_chunk(model_config: dict, model, chunk_path: str, language: Optional[str]) -> Tuple[int, str, str]:
    """
    Transcribe a single chunk and return (index, text, detected_language).
    Index is parsed from chunk filename suffix.
    """
    # Extract index from filename like chunk_0001.wav
    try:
        basename = os.path.basename(chunk_path)
        idx_str = os.path.splitext(basename)[0].split('_')[-1]
        index = int(idx_str)
    except Exception:
        index = 0

    transcription_text = ""
    detected_language = language or 'unknown'

    try:
        if model_config["type"] == "openai":
            if not OPENAI_WHISPER_AVAILABLE:
                raise ValueError("openai-whisper is not installed. Cannot transcribe with OpenAI Whisper models.")
            
            use_fp16 = torch.cuda.is_available()
            try:
                result = model.transcribe(chunk_path, language=language, fp16=use_fp16, verbose=False)
            except Exception:
                result = model.transcribe(chunk_path, language=language, fp16=False, verbose=False)
            transcription_text = result.get('text', '').strip()
            detected_language = result.get('language', detected_language)

        elif model_config["type"] == "faster_whisper":
            if not FASTER_WHISPER_AVAILABLE:
                raise ValueError("faster_whisper is not installed. Cannot transcribe with faster_whisper models.")
            
            # Handle wrapped faster_whisper model structure
            if isinstance(model, dict) and model.get("type") == "faster_whisper":
                actual_model = model["model"]
            else:
                actual_model = model
            
            # Use faster_whisper model for transcription with optimized parameters
            segments, info = actual_model.transcribe(
                chunk_path, 
                language=language,
                beam_size=1,  # Fast beam search
                best_of=1,
                vad_filter=True,  # Enable VAD for speed
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=1000
                )
            )
            transcription_text = ' '.join([segment.text for segment in segments]).strip()
            detected_language = info.language if hasattr(info, 'language') else (language or 'unknown')
    except Exception as e:
        logger.error(f"Chunk transcription error ({chunk_path}): {e}")

    return index, transcription_text, detected_language


async def transcribe_audio_stream(websocket: WebSocket, processor: AudioStreamProcessor):
    """
    Transcribe audio chunks and send results via WebSocket

    Note: This function is only used for local Whisper/Ivrit models with FFmpeg streaming.
    Deepgram transcription uses a separate function (transcribe_with_deepgram) and does not
    use this audio caching mechanism.
    """

    # Load the model for this processor
    try:
        model = load_model(processor.model_name)
        processor.model = model
        model_config = MODEL_CONFIGS[processor.model_name]
    except Exception as e:
        await websocket.send_json({"error": f"Failed to load model {processor.model_name}: {str(e)}"})
        return
    
    # Initialize deduplicator for this streaming session
    deduper = IncrementalDeduplicator(context_window=20, min_sequence_length=5)

    try:
        while processor.is_running or not processor.audio_queue.empty():
            try:
                # Get audio chunk from queue (with timeout)
                audio_data = processor.audio_queue.get(timeout=1)

                # Generate cache key for this audio chunk
                cache_key = generate_cache_key(audio_data, SAMPLE_RATE, CHANNELS)

                # Check cache first
                cached_path = get_cached_audio(cache_key)
                if cached_path:
                    temp_path = cached_path
                    logger.debug("Using cached normalized audio")
                else:
                    # Not in cache - normalize with FFmpeg
                    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as temp_audio:
                        temp_raw = temp_audio.name
                        temp_audio.write(audio_data)

                    # Normalize to 16kHz mono WAV (required by Whisper)
                    temp_path = temp_raw.replace('.raw', '.wav')
                    normalize_cmd = [
                        'ffmpeg',
                        '-f', 's16le',  # Input: signed 16-bit PCM
                        '-ar', str(SAMPLE_RATE),  # Input sample rate
                        '-ac', str(CHANNELS),  # Input channels
                        '-i', temp_raw,
                        '-ar', '16000',  # Whisper requires 16kHz
                        '-ac', '1',  # Mono
                        '-c:a', 'pcm_s16le',  # 16-bit PCM
                        '-y',  # Overwrite
                        temp_path
                    ]
                    # Run FFmpeg asynchronously to avoid blocking
                    norm_process = await asyncio.create_subprocess_exec(
                        *normalize_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    _, norm_stderr = await norm_process.communicate()

                    if norm_process.returncode != 0:
                        error_msg = norm_stderr.decode() if norm_stderr else "Unknown error"
                        logger.error(f"FFmpeg normalization failed: {error_msg}")
                        os.unlink(temp_raw)
                        continue

                    os.unlink(temp_raw)  # Clean up raw file

                    # Save to cache for future use
                    save_to_cache(cache_key, temp_path)
                
                try:
                    # Transcribe audio chunk
                    logger.info(f"Transcribing audio chunk ({len(audio_data)} bytes)")
                    
                    if model_config["type"] == "openai":
                        # Use OpenAI Whisper
                        use_fp16 = torch.cuda.is_available()
                        try:
                            result = model.transcribe(
                                temp_path,
                                language=processor.language,
                                fp16=use_fp16,
                                verbose=False
                            )
                        except Exception as e:
                            logger.warning(f"GPU/FP16 transcribe failed: {e}. Retrying with fp16=False.")
                            result = model.transcribe(
                                temp_path,
                                language=processor.language,
                                fp16=False,
                                verbose=False
                            )
                        transcription_text = result.get('text', '').strip()
                        detected_language = result.get('language', 'unknown')
                    elif model_config["type"] == "faster_whisper":
                        # Use faster_whisper for transcription
                        # Handle wrapped model structure
                        if isinstance(model, dict) and model.get("type") == "faster_whisper":
                            fw_model = model["model"]
                        else:
                            fw_model = model
                        
                        try:
                            segments, info = fw_model.transcribe(
                                temp_path,
                                language=processor.language or "he",
                                beam_size=int(os.getenv("IVRIT_BEAM_SIZE", "5")),
                                best_of=5,
                                patience=1,
                                temperature=0,
                                compression_ratio_threshold=2.4,
                                no_speech_threshold=0.6,
                                word_timestamps=False
                            )
                            text_parts = []
                            for segment in segments:
                                text_parts.append(segment.text)
                            transcription_text = ' '.join(text_parts).strip()
                            detected_language = info.language if hasattr(info, 'language') else (processor.language or 'he')
                        except Exception as e:
                            logger.error(f"faster_whisper transcription failed: {e}")
                            transcription_text = ""
                            detected_language = processor.language or 'he'

                    
                    # Send transcription to client (apply per-chunk dedup)
                    if transcription_text:
                        try:
                            cleaned_chunk = deduper.process_chunk(transcription_text)
                            cleaned_chunk = remove_repeated_word_sequences(cleaned_chunk, min_sequence_length=5)
                            sentences = split_into_sentences(cleaned_chunk)
                            sentences = remove_consecutive_duplicate_sentences(sentences)
                            cleaned_chunk = ''.join(sentences)
                        except Exception as e:
                            logger.debug(f"Stream chunk dedup failed, using original: {e}")
                            cleaned_chunk = transcription_text

                        if cleaned_chunk:
                            await websocket.send_json({
                                "type": "transcription",
                                "text": cleaned_chunk,
                                "language": detected_language
                            })
                            logger.info(f"✓ Sent transcription ({len(cleaned_chunk)} chars): {cleaned_chunk[:100]}...")
                    else:
                        logger.warning("⚠ No transcription text extracted from audio chunk")
                    
                finally:
                    # Cleanup temp file
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                        
            except queue.Empty:
                # No audio data available; send heartbeat to keep UI responsive
                await asyncio.sleep(0.1)
                try:
                    await websocket.send_json({"type": "status", "message": "Waiting for audio..."})
                except Exception:
                    pass
                continue
            except Exception as e:
                logger.error(f"Error transcribing chunk: {e}")
                await websocket.send_json({"error": str(e)})
                
        # Stream finished
        await websocket.send_json({"type": "complete", "message": "Transcription complete"})
        logger.info("Transcription stream completed")
        
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        await websocket.send_json({"error": str(e)})


async def transcribe_vod_with_deepgram(websocket: WebSocket, url: str, language: Optional[str] = None):
    """Transcribe VOD (Video On Demand) content using Deepgram's pre-recorded API"""
    if not DEEPGRAM_AVAILABLE:
        await websocket.send_json({"error": "Deepgram SDK not available"})
        return

    # Initialize Deepgram client
    if DeepgramClient is None:
        await websocket.send_json({
            "error": "DeepgramClient not available in installed SDK. Please upgrade to deepgram SDK v4."
        })
        return

    # Helper to detect auth errors robustly
    def _is_auth_error(err_msg: str) -> bool:
        msg = (err_msg or "").lower()
        return (
            "401" in msg or
            "unauthorized" in msg or
            "invalid credentials" in msg or
            "invalid auth" in msg or
            "missing api key" in msg or
            "no api key" in msg or
            "forbidden" in msg
        )

    # If API key missing, return error (no fallback when Deepgram is explicitly selected)
    if not DEEPGRAM_API_KEY:
        await websocket.send_json({
            "error": "Deepgram API key not configured. Please set DEEPGRAM_API_KEY in your .env file."
        })
        return

    client = DeepgramClient(api_key=DEEPGRAM_API_KEY) if DEEPGRAM_API_KEY else DeepgramClient()
    model = os.getenv("DEEPGRAM_MODEL", "nova-3")
    lang = language or os.getenv("DEEPGRAM_LANGUAGE", "en-US")

    # Sanitize potential pasted links (zero-width chars/backticks)
    url = sanitize_url(url)

    try:
        # OPTIMIZATION: Try sending URL directly to Deepgram first (much faster!)
        # Deepgram can fetch and transcribe URLs directly without us downloading
        await websocket.send_json({"type": "status", "message": "📡 Step 1/3: Attempting direct URL transcription via Deepgram API..."})

        try:
            # Determine if we should use language detection
            use_detect_language = not language or language == "auto"

            # Use Deepgram's URL-based transcription (recommended approach)
            if use_detect_language:
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

            # Extract transcript and metadata from response
            transcript = response.results.channels[0].alternatives[0].transcript.strip()

            # Apply text deduplication to remove any duplicate sentences or word sequences
            transcript = clean_transcription_text(transcript)

            # Safely extract request_id from metadata (it's an object, not a dict)
            request_id = None
            try:
                if hasattr(response, 'metadata') and response.metadata:
                    request_id = getattr(response.metadata, 'request_id', None)
            except Exception:
                pass

            if transcript:
                response_data = {
                    "type": "transcription",
                    "text": transcript,
                    "language": lang
                }
                # Include request_id for tracking/debugging if available
                if request_id:
                    response_data["request_id"] = request_id
                    logger.info(f"✓ Sent complete Deepgram transcription ({len(transcript)} chars) [request_id: {request_id}]")
                else:
                    logger.info(f"✓ Sent complete Deepgram transcription ({len(transcript)} chars)")

                await websocket.send_json(response_data)
                await websocket.send_json({"type": "complete", "message": "Transcription complete"})
                return
            else:
                logger.warning("No transcript received from Deepgram URL method")

        except Exception as url_error:
            # Detect auth errors and fail immediately (no fallback when Deepgram is explicitly selected)
            err_text = str(url_error)
            if _is_auth_error(err_text):
                logger.error(f"Deepgram authentication failed: {err_text}")
                await websocket.send_json({
                    "error": f"Deepgram authentication failed: {err_text}\n\nPlease check:\n  • DEEPGRAM_API_KEY is correctly set in .env\n  • API key is valid and not expired\n  • Account has sufficient credits"
                })
                return
            # URL method failed (maybe URL not directly accessible by Deepgram)
            logger.warning(f"Deepgram URL method failed: {url_error}, falling back to download method")
            await websocket.send_json({"type": "status", "message": "⚠️ Direct URL method failed. Switching to download method..."})

        # FALLBACK: Download with robust method (yt-dlp preferred for YouTube) and upload to Deepgram
        await websocket.send_json({"type": "status", "message": "⬇️ Step 2/3: Downloading audio from URL (yt-dlp/ffmpeg fallback)..."})

        # Prefer yt-dlp for YouTube/shorts; falls back to ffmpeg automatically
        from services.audio_processor import download_with_fallback
        audio_file = await download_with_fallback(url, language=language, format='wav', websocket=websocket, use_cache=True)
        if not audio_file:
            # Provide detailed error message based on logs
            error_detail = "Failed to download audio from URL.\n\n"
            error_detail += "Common causes:\n"
            error_detail += "  • URL has expired (HTTP 410) - Get a fresh URL\n"
            error_detail += "  • Access denied (HTTP 403) - URL may require authentication\n"
            error_detail += "  • URL not found (HTTP 404) - Verify the URL is correct\n"
            error_detail += "  • Network/connection issues\n\n"
            error_detail += "Check server logs for specific error details."

            await websocket.send_json({"error": error_detail})
            return

        # Notify download success
        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        await websocket.send_json({"type": "status", "message": f"✅ Audio downloaded successfully ({file_size_mb:.1f} MB)"})

        await websocket.send_json({"type": "status", "message": "🚀 Step 3/3: Uploading to Deepgram and transcribing..."})

        try:
            # Check file size before uploading
            file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
            if file_size_mb > 100:
                logger.warning(f"Large audio file detected ({file_size_mb:.1f}MB). This may take longer to process.")
                await websocket.send_json({"type": "status", "message": f"Processing large file ({file_size_mb:.1f}MB), please wait..."})

            # Read file in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            
            def read_audio_file():
                with open(audio_file, "rb") as f:
                    return f.read()
            
            audio_data = await loop.run_in_executor(None, read_audio_file)

            # Determine if we should use language detection
            use_detect_language = not language or language == "auto"

            if use_detect_language:
                logger.info("Using automatic language detection (detect_language=True)")
                response = client.listen.v1.media.transcribe_file(
                    request=audio_data,
                    model=model,
                    detect_language=True,
                    punctuate=True,
                    smart_format=True
                )
            else:
                response = client.listen.v1.media.transcribe_file(
                    request=audio_data,
                    model=model,
                    language=language,
                    punctuate=True,
                    smart_format=True
                )

            transcript = response.results.channels[0].alternatives[0].transcript.strip()

            # Extract additional metadata for better user feedback
            request_id = None
            detected_language = None
            confidence = None

            try:
                if hasattr(response, 'metadata') and response.metadata:
                    request_id = getattr(response.metadata, 'request_id', None)
            except Exception:
                pass

            try:
                # Get detected language and confidence
                if response.results.channels[0].alternatives:
                    alternative = response.results.channels[0].alternatives[0]
                    if hasattr(alternative, 'confidence'):
                        confidence = alternative.confidence
                    if hasattr(alternative, 'detected_language'):
                        detected_language = alternative.detected_language
            except Exception:
                pass

            if transcript:
                # Apply text deduplication to remove any duplicate sentences or word sequences
                transcript = clean_transcription_text(transcript)

                # Log detailed success info
                log_parts = [f"✓ Transcription complete ({len(transcript)} chars)"]
                if detected_language:
                    log_parts.append(f"Language: {detected_language}")
                if confidence:
                    log_parts.append(f"Confidence: {confidence:.2f}")
                if request_id:
                    log_parts.append(f"Request ID: {request_id}")
                logger.info(" | ".join(log_parts))

                # Send transcript in chunks to avoid WebSocket message size limits
                # Split into sentences or chunks of ~500 characters
                chunk_size = 500
                transcript_chunks = []

                # Try to split by sentences first (. ! ?)
                sentences = []
                current_sentence = ""
                for char in transcript:
                    current_sentence += char
                    if char in '.!?' and len(current_sentence) > 50:
                        sentences.append(current_sentence.strip())
                        current_sentence = ""
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())

                # Group sentences into chunks
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                        current_chunk += (" " if current_chunk else "") + sentence
                    else:
                        if current_chunk:
                            transcript_chunks.append(current_chunk)
                        current_chunk = sentence
                if current_chunk:
                    transcript_chunks.append(current_chunk)

                # If no chunks created (no sentence breaks), split by character count
                if not transcript_chunks:
                    for i in range(0, len(transcript), chunk_size):
                        transcript_chunks.append(transcript[i:i + chunk_size])

                # Send each chunk
                logger.info(f"Sending transcript in {len(transcript_chunks)} chunks")
                await websocket.send_json({"type": "status", "message": f"📝 Transcription complete! Sending {len(transcript)} characters..."})

                # Initialize deduper for per-chunk cleaning across VOD chunks
                vod_deduper = IncrementalDeduplicator(context_window=20, min_sequence_length=5)

                for i, chunk in enumerate(transcript_chunks):
                    try:
                        cleaned_chunk = vod_deduper.process_chunk(chunk)
                        cleaned_chunk = remove_repeated_word_sequences(cleaned_chunk, min_sequence_length=5)
                        sentences = split_into_sentences(cleaned_chunk)
                        sentences = remove_consecutive_duplicate_sentences(sentences)
                        cleaned_chunk = ''.join(sentences)
                    except Exception:
                        cleaned_chunk = chunk
                    response_data = {
                        "type": "transcription",
                        "text": cleaned_chunk,
                        "language": detected_language or lang
                    }
                    if request_id:
                        response_data["request_id"] = request_id
                    await websocket.send_json(response_data)
                    # Small delay between chunks to ensure proper delivery
                    await asyncio.sleep(0.1)
            else:
                # Provide detailed feedback when no transcript is returned
                error_msg = "⚠️ No transcript received from Deepgram"

                # Provide possible causes
                possible_causes = []
                if file_size_mb < 0.1:
                    possible_causes.append("Audio file is very small (< 100KB) - may contain no speech")
                possible_causes.extend([
                    "No speech detected in the audio",
                    "Audio contains only music/silence",
                    "Audio quality too poor to transcribe",
                    "Language not well-supported by Deepgram"
                ])

                detailed_msg = f"{error_msg}\n\nPossible causes:\n" + "\n".join(f"  • {cause}" for cause in possible_causes)
                logger.warning(detailed_msg)

                await websocket.send_json({
                    "type": "status",
                    "message": "⚠️ No speech detected in audio. The file may contain only music, silence, or unclear speech."
                })

            await websocket.send_json({"type": "complete", "message": "Transcription complete"})

        except Exception as e:
            error_msg = str(e)
            # Handle specific payload size errors
            if "413" in error_msg or "payload too large" in error_msg.lower() or "entity too large" in error_msg.lower():
                logger.error(f"Deepgram payload too large error: {e}")
                await websocket.send_json({
                    "error": "Audio file too large for direct upload. Try using a URL instead or use a smaller file."
                })
            elif _is_auth_error(error_msg):
                logger.error(f"Deepgram authentication failed on file upload: {e}")
                await websocket.send_json({
                    "error": f"Deepgram authentication failed: {error_msg}\n\nPlease check:\n  • DEEPGRAM_API_KEY is correctly set in .env\n  • API key is valid and not expired\n  • Account has sufficient credits"
                })
            else:
                logger.error(f"Deepgram transcription error: {e}")
                await websocket.send_json({"error": f"Deepgram transcription failed: {str(e)}"})

        finally:
            # Cleanup downloaded file and its temp directory (only if not in cache)
            try:
                if audio_file and os.path.exists(audio_file):
                    # Only delete if not in download cache
                    if DOWNLOAD_CACHE_DIR not in Path(audio_file).parents:
                        os.unlink(audio_file)
                        temp_dir = os.path.dirname(audio_file)
                        if temp_dir and os.path.exists(temp_dir) and '/tmp' in temp_dir:
                            shutil.rmtree(temp_dir, ignore_errors=True)
                    else:
                        logger.debug(f"Keeping cached Deepgram download: {audio_file}")
            except Exception as e:
                logger.debug(f"Cleanup warning: {e}")

    except Exception as e:
        logger.error(f"VOD transcription error: {e}")
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({"error": str(e)})
        except:
            pass


async def transcribe_with_deepgram(websocket: WebSocket, url: str, language: Optional[str] = None):
    """Transcribe audio stream using Deepgram live transcription (matches user's sample logic)"""
    if not DEEPGRAM_AVAILABLE:
        await websocket.send_json({"error": "Deepgram SDK not available"})
        return

    try:
        # Track connection state to avoid sending after close
        ws_open = True
        dg_closed_event = asyncio.Event()

        # Helper: robustly extract transcript from Deepgram payloads
        # Only return final transcripts, not interim results
        def extract_deepgram_transcript(message) -> str:
            try:
                if isinstance(message, dict):
                    # Check if this is a final transcript (not interim)
                    is_final = message.get('is_final', True)  # Default to True if not specified
                    if not is_final:
                        return ''  # Skip interim results

                    alts = message.get('channel', {}).get('alternatives', [])
                    if alts:
                        return alts[0].get('transcript', '')
                # v4 typed objects
                if hasattr(message, 'channel') and hasattr(message.channel, 'alternatives'):
                    # Check if this is a final transcript
                    is_final = getattr(message, 'is_final', True)
                    if not is_final:
                        return ''  # Skip interim results

                    alt0 = message.channel.alternatives[0] if message.channel.alternatives else None
                    if alt0 is not None:
                        return getattr(alt0, 'transcript', '') or ''
            except Exception:
                pass
            return ''


        # Initialize Deepgram client
        if DeepgramClient is None:
            await websocket.send_json({
                "error": "DeepgramClient not available in installed SDK. Please upgrade to deepgram SDK v4."
            })
            return

        client = DeepgramClient(api_key=DEEPGRAM_API_KEY) if DEEPGRAM_API_KEY else DeepgramClient()
        # Apply user-requested params
        # TIME_LIMIT: Set to 0 or negative for unlimited streaming, or positive number for time limit in seconds
        TIME_LIMIT = DEEPGRAM_TIME_LIMIT
        TRANSCRIPT_ONLY = DEEPGRAM_TRANSCRIPT_ONLY
        PARAMS = {
            "punctuate": True,
            "numerals": True,
            "model": DEEPGRAM_MODEL,
            "language": DEEPGRAM_LANGUAGE,
        }

        # Create a websocket client (SDK v4)
        connection = client.listen.websocket.v("1")

        # Capture the event loop reference BEFORE defining callbacks
        # The Deepgram SDK runs callbacks in a different thread, so we need to capture the loop here
        loop = asyncio.get_event_loop()

        # Event handlers
        # Initialize deduper for Deepgram live session (thread-safe usage via callback)
        dg_deduper = IncrementalDeduplicator(context_window=20, min_sequence_length=5)
        dedup_lock = threading.Lock()
        # Listen for the connection to open
        connection.on(DG_EVENT_OPEN, lambda *_: logger.info("Deepgram connection opened"))

        # Listen for any transcripts received from Deepgram
        def on_message(*args, **kwargs):
            # Deepgram v4 emits handler(self, result=..., **kwargs)
            payload = kwargs.get('result') if 'result' in kwargs else (args[1] if len(args) >= 2 else None)
            sentence = extract_deepgram_transcript(payload) if payload is not None else ''

            # Only process and log non-empty transcripts to avoid log spam from interim results
            if sentence:
                # Optional: print transcript to console if TRANSCRIPT_ONLY mode
                if TRANSCRIPT_ONLY:
                    print(sentence)

                try:
                    if ws_open and websocket.client_state == WebSocketState.CONNECTED and not dg_closed_event.is_set():
                        # Apply stateful, thread-safe deduplication to the sentence
                        try:
                            with dedup_lock:
                                cleaned = dg_deduper.process_chunk(sentence)
                            cleaned = remove_repeated_word_sequences(cleaned, min_sequence_length=5)
                            sents = split_into_sentences(cleaned)
                            sents = remove_consecutive_duplicate_sentences(sents)
                            cleaned = ''.join(sents)
                        except Exception:
                            cleaned = sentence
                        # Schedule the coroutine on the main event loop from this callback thread
                        asyncio.run_coroutine_threadsafe(
                            websocket.send_json({
                                "type": "transcription",
                                "text": cleaned,
                                "language": language or os.getenv("DEEPGRAM_LANGUAGE", "en-US")
                            }),
                            loop
                        )
                        logger.info(f"✓ Sent Deepgram transcription: {cleaned[:100]}...")
                except Exception as e:
                    logger.error(f"Failed to send Deepgram transcription: {e}")

        connection.on(DG_EVENT_MESSAGE, on_message)
        # On close, mark closed but don't send over websocket (function will handle finalization)
        # Listen for the connection to close
        def on_close(*_):
            try:
                print('✅ Transcription complete! Connection closed. ✅')
            except Exception:
                pass
            finally:
                try:
                    dg_closed_event.set()
                except Exception:
                    pass
        connection.on(DG_EVENT_CLOSE, on_close)

        # On error, try to notify if still connected, then mark closed
        def on_error(*args, **kwargs):
            err = kwargs.get('error') if 'error' in kwargs else (args[1] if len(args) >= 2 else None)
            try:
                if ws_open and websocket.client_state == WebSocketState.CONNECTED and not dg_closed_event.is_set():
                    asyncio.create_task(websocket.send_json({"error": f"Deepgram error: {err}"}))
            except RuntimeError:
                pass
            finally:
                try:
                    dg_closed_event.set()
                except Exception:
                    pass
        connection.on(DG_EVENT_ERROR, on_error)

        # Surface current Deepgram config to the client for visibility
        try:
            cfg_msg = (
                f"Deepgram connected: model={PARAMS['model']}, lang={PARAMS['language']}, "
                f"tier={(os.getenv('DEEPGRAM_TIER') or 'default')}, limit={TIME_LIMIT}s, "
                f"transcript_only={TRANSCRIPT_ONLY}"
            )
            await websocket.send_json({"type": "status", "message": cfg_msg})
        except Exception:
            await websocket.send_json({"type": "status", "message": "Deepgram connection established, streaming audio..."})

        # Start listening with options
        try:
            from deepgram.clients.listen.v1.websocket.options import LiveOptions
        except Exception as e:
            await websocket.send_json({"error": f"Deepgram SDK options missing: {e}"})
            return

        # Determine language parameter for streaming
        # WebSocket doesn't support detect_language, so use 'multi' for auto-detection
        stream_language = language if language and language != "auto" else "multi"

        if stream_language == "multi":
            logger.info("Using multi-language support for automatic language detection in streaming")
        else:
            logger.info(f"Using language for streaming: {stream_language}")

        # Build options from PARAMS and app audio settings
        opts_kwargs = {
            "model": PARAMS["model"],
            "language": stream_language,
            "punctuate": PARAMS["punctuate"],
            "numerals": PARAMS["numerals"],
            "encoding": "linear16",
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "interim_results": True,
            "endpointing": False,  # Disable auto-closure on silence detection
            "vad_turnoff": 5000,   # 5 second VAD timeout (prevent premature closure)
        }
        # Tier may not be allowed on some accounts; include only if set via env
        dg_tier = os.getenv("DEEPGRAM_TIER")
        if dg_tier:
            opts_kwargs["tier"] = dg_tier
        options = LiveOptions(**opts_kwargs)

        # Attempt to start the Deepgram websocket; surface 403 clearly
        try:
            connection.start(options)
        except Exception as e:
            msg = str(e)
            if "403" in msg:
                await websocket.send_json({
                    "error": "Deepgram rejected websocket (HTTP 403). Check API key, model/tier permissions, and account access for live streaming."
                })
            else:
                await websocket.send_json({"error": f"Failed to start Deepgram websocket: {msg}"})
            return

        # Stream audio to Deepgram via ffmpeg to ensure linear16 PCM, avoiding 410/format issues
        await websocket.send_json({"type": "status", "message": "Streaming audio to Deepgram via ffmpeg..."})

        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-reconnect", "1",              # Reconnect on connection failures
            "-reconnect_streamed", "1",     # Reconnect for streamed inputs
            "-reconnect_delay_max", "5",    # Max reconnect delay
            "-user_agent", "Mozilla/5.0",
            "-i", url,
            "-vn",                          # No video
            "-f", "s16le",                  # PCM signed 16-bit little-endian
            "-acodec", "pcm_s16le",         # Audio codec
            "-ac", str(CHANNELS),           # Mono
            "-ar", str(SAMPLE_RATE),        # 16kHz sample rate
            "pipe:1",
        ]

        proc = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        start_ts = asyncio.get_event_loop().time()
        # Support unlimited streaming: if TIME_LIMIT <= 0, stream indefinitely
        time_limit = float(TIME_LIMIT) if TIME_LIMIT > 0 else float("inf")

        try:
            bytes_sent = 0

            while True:
                # Check time limit only if it's not infinite
                if time_limit != float("inf"):
                    elapsed = asyncio.get_event_loop().time() - start_ts
                    if elapsed >= time_limit:
                        logger.info(f"Deepgram stream reached time limit of {TIME_LIMIT} seconds")
                        break

                data = await proc.stdout.read(4096)
                if not data:
                    # Check if ffmpeg process exited with an error
                    stderr_output = await proc.stderr.read()
                    if stderr_output:
                        logger.error(f"FFmpeg error: {stderr_output.decode('utf-8', errors='ignore')}")
                    logger.info(f"FFmpeg stream ended after {bytes_sent} bytes")
                    break

                # Stream raw PCM bytes to Deepgram as fast as possible
                # No throttling needed - Deepgram handles buffering
                # With endpointing disabled and high VAD timeout, connection stays open
                connection.send(data)
                bytes_sent += len(data)
        finally:
            try:
                proc.terminate()
                await proc.wait()
            except Exception:
                pass

        # Signal finish to Deepgram
        try:
            connection.finish()
            logger.info("Sent finish signal to Deepgram, waiting for final transcriptions...")
        except Exception:
            pass

        # Wait for Deepgram to finish processing all buffered audio
        # Calculate estimated processing time based on audio sent
        # Deepgram typically processes audio faster than real-time, but we need to account for:
        # - Network latency for sending results back
        # - Processing queue time
        # Use generous timeout: min 10s, max 60s, or estimated audio duration
        estimated_audio_seconds = bytes_sent / (SAMPLE_RATE * CHANNELS * 2) if bytes_sent > 0 else 0
        processing_timeout = min(max(estimated_audio_seconds + 10, 10), 60)

        logger.info(f"Waiting up to {processing_timeout:.1f}s for Deepgram to process {estimated_audio_seconds:.1f}s of audio")

        try:
            # Wait for Deepgram close event with appropriate timeout
            try:
                await asyncio.wait_for(dg_closed_event.wait(), timeout=processing_timeout)
                logger.info("Deepgram closed connection after processing")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for Deepgram close after {processing_timeout}s")
                pass
            # Send completion if still connected
            if ws_open and websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({"type": "complete", "message": "Transcription complete"})
            # Close websocket gracefully
            if websocket.client_state == WebSocketState.CONNECTED:
                ws_open = False
                try:
                    await websocket.close()
                except Exception:
                    pass
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Deepgram transcription error: {e}")
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({"error": f"Deepgram error: {str(e)}"})
        except RuntimeError:
            pass
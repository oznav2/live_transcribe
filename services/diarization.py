"""Speaker diarization service for transcription with speaker identification."""
import asyncio
import logging
import os
import time
from datetime import timedelta
from typing import List, Optional, Tuple

from fastapi import WebSocket
from starlette.websockets import WebSocketState

from config.availability import FASTER_WHISPER_AVAILABLE, OPENAI_WHISPER_AVAILABLE
from models.loader import get_diarization_pipeline
from services.transcription import transcribe_with_incremental_output

logger = logging.getLogger(__name__)

# Import model libraries conditionally
if FASTER_WHISPER_AVAILABLE:
    import faster_whisper
    
if OPENAI_WHISPER_AVAILABLE:
    import whisper
    import torch


async def transcribe_with_diarization(
    model, 
    model_config: dict, 
    audio_file: str, 
    language: Optional[str],
    websocket: WebSocket,
    model_name: str = None
) -> Tuple[List[dict], str]:
    """
    Transcribe audio with speaker diarization.
    Returns (diarized_segments, detected_language)
    
    Each segment contains:
    - start: start time in seconds
    - end: end time in seconds  
    - speaker: speaker label (דובר_1, דובר_2 for Hebrew/Ivrit models; SPEAKER_1, SPEAKER_2 for others)
    - text: transcribed text
    
    Speaker labels are automatically localized based on model:
    - Ivrit models (Hebrew): דובר_1, דובר_2, דובר_3...
    - Other models: SPEAKER_1, SPEAKER_2, SPEAKER_3...
    """
    import time
    
    logger.info(f"Starting transcription with diarization for {audio_file}")
    
    # Send status update (check connection state)
    if websocket.client_state == WebSocketState.CONNECTED:
        await websocket.send_json({
            "type": "status",
            "message": "Starting transcription with speaker diarization..."
        })
    
    # Step 1: Run diarization to get speaker segments
    pipeline = get_diarization_pipeline()
    if pipeline is None:
        logger.warning("Diarization pipeline not available, falling back to regular transcription")
        # Fall back to regular transcription
        transcript, lang = await transcribe_with_incremental_output(
            model, model_config, audio_file, language, websocket, model_name
        )
        return [{"text": transcript, "speaker": "SPEAKER_1"}], lang
    
    try:
        # Run diarization
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({
                "type": "status", 
                "message": "Analyzing speakers in audio..."
            })
        
        # Run blocking diarization in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        diarization = await loop.run_in_executor(None, pipeline, audio_file)
        
        # Convert diarization output to segments
        speaker_segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker
            })
        
        # Detect if using Hebrew (Ivrit) model and set appropriate speaker label prefix
        is_hebrew_model = model_name and "ivrit" in model_name.lower()
        speaker_prefix = "דובר_" if is_hebrew_model else "SPEAKER_"
        
        logger.info(f"Using speaker label prefix: '{speaker_prefix}' (Hebrew model: {is_hebrew_model})")
        
        # Renumber speakers to דובר_1, דובר_2 (Hebrew) or SPEAKER_1, SPEAKER_2 (other languages)
        speaker_mapping = {}
        speaker_counter = 1
        for segment in speaker_segments:
            if segment["speaker"] not in speaker_mapping:
                speaker_mapping[segment["speaker"]] = f"{speaker_prefix}{speaker_counter}"
                speaker_counter += 1
            segment["speaker"] = speaker_mapping[segment["speaker"]]
        
        logger.info(f"Found {len(speaker_mapping)} speakers in audio")
        
        # Step 2: Transcribe with timestamps
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({
                "type": "status",
                "message": f"Transcribing audio with {len(speaker_mapping)} speakers..."
            })
        
        # Use faster_whisper for transcription with word timestamps
        if model_config["type"] == "faster_whisper":
            # Extract the actual WhisperModel from the wrapper dict
            if isinstance(model, dict) and "model" in model:
                fw_model = model["model"]
            else:
                fw_model = model
            
            # Determine language
            default_lang = None
            if model_name and "ivrit" in model_name.lower():
                default_lang = "he"
            
            logger.info(f"Running faster_whisper transcription with word timestamps for diarization")
            
            # Transcribe with word-level timestamps for better alignment
            # Note: faster_whisper.transcribe returns a generator
            segments_generator, info = fw_model.transcribe(
                audio_file,
                language=language or default_lang,
                word_timestamps=True,  # Enable word timestamps for better alignment
                beam_size=int(os.getenv("IVRIT_BEAM_SIZE", "5")),
                best_of=5,
                patience=1,
                temperature=0
            )
            
            # Convert segments to list with timestamps and show progress
            transcription_segments = []
            segment_count = 0
            start_time_transcribe = time.time()
            
            for segment in segments_generator:
                transcription_segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                })
                segment_count += 1
                
                # Send progress update every 10 segments
                if segment_count % 10 == 0:
                    elapsed = time.time() - start_time_transcribe
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_json({
                            "type": "status",
                            "message": f"Transcribing: {segment_count} segments processed ({elapsed:.1f}s elapsed)..."
                        })
            
            detected_language = info.language if info else (language or 'unknown')
            logger.info(f"Transcription complete. Detected language: {detected_language}, Segments: {len(transcription_segments)}")
            
        else:
            # Fallback for OpenAI whisper
            result = model.transcribe(
                audio_file,
                language=language,
                verbose=False,
                word_timestamps=True
            )
            
            transcription_segments = []
            for segment in result.get("segments", []):
                transcription_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip()
                })
            
            detected_language = result.get("language", language or 'unknown')
        
        # Step 3: Align transcription with speaker segments
        diarized_segments = []
        
        for speaker_seg in speaker_segments:
            speaker_text = []
            
            # Find transcription segments that overlap with this speaker segment
            for trans_seg in transcription_segments:
                # Check for overlap
                overlap_start = max(speaker_seg["start"], trans_seg["start"])
                overlap_end = min(speaker_seg["end"], trans_seg["end"])
                
                if overlap_end > overlap_start:
                    # There's overlap - this text belongs to this speaker
                    # Calculate overlap percentage
                    trans_duration = trans_seg["end"] - trans_seg["start"]
                    overlap_duration = overlap_end - overlap_start
                    
                    if trans_duration > 0:
                        overlap_pct = overlap_duration / trans_duration
                        
                        # Only include if significant overlap (>50%)
                        if overlap_pct > 0.5:
                            speaker_text.append(trans_seg["text"])
            
            # Combine text for this speaker segment
            if speaker_text:
                combined_text = " ".join(speaker_text)
                diarized_segments.append({
                    "start": speaker_seg["start"],
                    "end": speaker_seg["end"],
                    "speaker": speaker_seg["speaker"],
                    "text": combined_text
                })
        
        # Send incremental results
        total_segments = len(diarized_segments)
        for i, segment in enumerate(diarized_segments):
            # Format timestamp
            start_time = timedelta(seconds=int(segment["start"]))
            end_time = timedelta(seconds=int(segment["end"]))
            
            # Format as [HH:MM:SS-HH:MM:SS] SPEAKER_X: "text"
            formatted_output = (
                f"[{str(start_time).split('.')[0]}-{str(end_time).split('.')[0]}] "
                f"{segment['speaker']}: \"{segment['text']}\""
            )
            
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({
                    "type": "transcription_chunk",
                    "text": formatted_output,
                    "chunk_index": i,
                    "total_chunks": total_segments,
                    "is_final": i == total_segments - 1,
                    "speaker": segment["speaker"],
                    "start": segment["start"],
                    "end": segment["end"]
                })
            
            # Small delay to prevent overwhelming the client
            if i % 5 == 0:
                await asyncio.sleep(0.1)
        
        logger.info(f"Diarization complete: {len(diarized_segments)} segments")
        return diarized_segments, detected_language
        
    except Exception as e:
        logger.error(f"Diarization failed: {e}", exc_info=True)
        # Fall back to regular transcription
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({
                "type": "status",
                "message": "Diarization failed, falling back to regular transcription..."
            })
        
        # Call the function directly (it's in the same file)
        transcript, lang = await transcribe_with_incremental_output(
            model, model_config, audio_file, language, websocket, model_name
        )
        return [{"text": transcript, "speaker": "SPEAKER_1"}], lang
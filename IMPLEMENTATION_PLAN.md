# Implementation Plan: Enhanced Transcription Feedback & YouTube Support

## Current Issues Identified

1. **Limited Progress Information**
   - Users only see elapsed time, not percentage or ETA
   - No indication of audio file duration
   - No processing speed metrics

2. **No Incremental Output**
   - Users wait for entire transcription to complete
   - No partial results during processing
   - Poor UX for long files

3. **YouTube URL Support**
   - Current pattern matching works but could be more robust
   - Need to ensure proper handling of various YouTube URL formats

## Proposed Solutions

### 1. Enhanced Progress Tracking
**Implementation Strategy:**
- Get audio duration using `get_audio_duration_seconds()`
- Calculate percentage based on processing speed
- Estimate ETA using rolling average of processing rate
- Send detailed progress updates with:
  - Total duration
  - Processed duration  
  - Percentage complete
  - ETA
  - Processing speed (x realtime)

### 2. Incremental Transcription Output
**Two Approaches:**

**Option A: Chunked Processing (Recommended)**
- Split audio into chunks (30-60 second segments)
- Process chunks sequentially
- Send results as each chunk completes
- Maintains context with overlap
- Works with all models

**Option B: Whisper Callback (Limited)**
- OpenAI Whisper doesn't provide native progress callbacks
- Would need to modify the library or use verbose mode parsing
- More complex and fragile

**Selected: Option A - Chunked Processing**

### 3. YouTube URL Enhancement
**Implementation:**
- Already supports youtube.com and youtu.be
- Add support for various URL formats:
  - Standard: `https://www.youtube.com/watch?v=VIDEO_ID`
  - Short: `https://youtu.be/VIDEO_ID`
  - With timestamp: `...&t=123`
  - With playlist: `...&list=...`
- Use yt-dlp's robust URL handling

## Implementation Steps

### Step 1: Add Progress Calculation Utilities
```python
def calculate_progress_metrics(audio_duration, elapsed_time, processed_duration=None):
    """Calculate progress percentage and ETA"""
    if processed_duration:
        percentage = (processed_duration / audio_duration) * 100
        speed = processed_duration / elapsed_time if elapsed_time > 0 else 0
        eta = (audio_duration - processed_duration) / speed if speed > 0 else 0
    else:
        # Estimate based on typical processing speed
        estimated_speed = 0.5  # 0.5x realtime as conservative estimate
        processed_duration = elapsed_time * estimated_speed
        percentage = min((processed_duration / audio_duration) * 100, 99)
        eta = (audio_duration - processed_duration) / estimated_speed
    
    return {
        "percentage": round(percentage, 1),
        "eta_seconds": int(eta),
        "speed": round(speed, 2) if processed_duration else estimated_speed,
        "processed_duration": processed_duration or elapsed_time * estimated_speed
    }
```

### Step 2: Implement Chunked Transcription
```python
async def transcribe_with_chunks(model, model_config, audio_file, language, websocket):
    """Transcribe audio in chunks with incremental output"""
    # Get total duration
    audio_duration = get_audio_duration_seconds(audio_file)
    
    # Split into chunks
    chunk_duration = 60  # 60-second chunks
    overlap = 5  # 5-second overlap
    
    chunks = split_audio_for_incremental(audio_file, chunk_duration, overlap)
    
    results = []
    start_time = time.time()
    
    for i, chunk_file in enumerate(chunks):
        # Transcribe chunk
        chunk_text = transcribe_single_chunk(model, model_config, chunk_file, language)
        
        # Send incremental result
        await websocket.send_json({
            "type": "transcription_chunk",
            "text": chunk_text,
            "chunk_index": i,
            "total_chunks": len(chunks)
        })
        
        # Send progress update
        processed = (i + 1) * chunk_duration
        metrics = calculate_progress_metrics(audio_duration, time.time() - start_time, processed)
        
        await websocket.send_json({
            "type": "transcription_progress",
            "audio_duration": audio_duration,
            "processed_duration": processed,
            "percentage": metrics["percentage"],
            "eta_seconds": metrics["eta_seconds"],
            "speed": metrics["speed"]
        })
        
        results.append(chunk_text)
    
    return " ".join(results)
```

### Step 3: Enhance YouTube URL Support
- Current implementation already handles YouTube URLs well
- Just need to ensure yt-dlp command is optimal
- Add better error handling for various URL formats

## Risk Mitigation

1. **Backward Compatibility**
   - Keep existing transcription methods as fallback
   - Add feature flag for incremental output
   - Maintain same WebSocket message structure

2. **Performance Considerations**
   - Chunk processing adds slight overhead
   - But provides much better UX
   - Can adjust chunk size based on model

3. **Error Handling**
   - Handle chunk failures gracefully
   - Continue with remaining chunks
   - Provide partial results even on failure

## Testing Plan

1. Test with various audio lengths (1min, 10min, 1hour)
2. Test all models (Whisper, Ivrit, Deepgram)
3. Test YouTube URLs in different formats
4. Verify incremental output displays correctly
5. Check progress calculations are accurate
6. Ensure no memory leaks with chunked processing

## Success Criteria

- [ ] Users see percentage complete during transcription
- [ ] Users see ETA for completion
- [ ] Users see incremental results as processing continues
- [ ] YouTube URLs work reliably
- [ ] No degradation in transcription quality
- [ ] No breaking changes to existing functionality
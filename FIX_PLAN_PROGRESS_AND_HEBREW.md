# Fix Plan: Progress Display & Hebrew Speaker Labels

## Problem Analysis

### Issue 1: Progress Not Showing in Real-Time
**Current Behavior:**
- Progress bar shows **AFTER** all chunks are processed
- Users see no feedback during download/transcription
- Stuck at 82% even when complete

**Root Cause:**
Looking at the code flow:
1. **Download Phase**: `download_audio_with_ffmpeg()` sends `download_progress` messages ✅ (WORKING)
2. **Transcription Phase**: The problem is in message ordering and timing

In `transcribe_with_incremental_output()` (lines 1776-1856):
```python
for i, chunk_file in enumerate(chunk_files):
    # Transcribe chunk (BLOCKING SYNC CALL - takes time)
    segments, info = fw_model.transcribe(chunk_file, ...)  # LINE 1790
    
    # Build text from segments (happens AFTER transcription)
    text_parts = []
    for segment in segments:
        text_parts.append(segment.text)
    chunk_text = ' '.join(text_parts).strip()
    
    # Send chunk (LINE 1826 - AFTER all processing)
    await websocket.send_json({
        "type": "transcription_chunk",
        "text": chunk_text,
        ...
    })
    
    # Send progress (LINE 1846 - AFTER chunk is sent)
    await websocket.send_json({
        "type": "transcription_progress",
        "percentage": percentage,
        ...
    })
```

**The Problem:**
1. `fw_model.transcribe()` is a **SYNCHRONOUS** call that blocks the event loop
2. Progress messages are sent AFTER each chunk completes
3. User sees nothing until first chunk finishes
4. All progress batches up and arrives late

**UI Side Issue:**
In `index.html` (line 1144-1157), progress handler works correctly, but messages arrive late.

### Issue 2: 82% Stuck Progress
**Root Cause:**
- Final progress update sends 100% (line 1843: `percentage = int(((i + 1) / total_chunks) * 100)`)
- But for last chunk where `i = total_chunks - 1`: `((total_chunks) / total_chunks) * 100 = 100%`
- However, if there's a delay or websocket closes early, it might show previous value (82%)

### Issue 3: No "File in Cache" Message
**Location:** Line 822-830 in `download_audio_with_ffmpeg()`
```python
if use_cache and duration == 0:
    cached_file = get_cached_download(url)
    if cached_file:
        if websocket:
            file_size_mb = os.path.getsize(cached_file) / (1024 * 1024)
            await websocket.send_json({
                "type": "status",
                "message": f"✅ Using cached audio file ({file_size_mb:.1f} MB)"
            })
```

**Problem:** Message sent but UI might not show it prominently or user misses it in status bar.

### Issue 4: Hebrew Speaker Labels
**Location:** Line 596-598 in `transcribe_with_diarization()`
```python
if segment["speaker"] not in speaker_mapping:
    speaker_mapping[segment["speaker"]] = f"SPEAKER_{speaker_counter}"
    speaker_counter += 1
```

**Problem:** Hardcoded English "SPEAKER_" labels. Need to detect Hebrew (Ivrit) model and use "דובר_" instead.

---

## Solution Design

### FIX 1: Make Transcription Real-Time Responsive
**Strategy:** Move blocking `fw_model.transcribe()` to executor and send progress DURING processing

**Implementation:**
```python
async def transcribe_with_incremental_output(...):
    for i, chunk_file in enumerate(chunk_files):
        # BEFORE processing - send "starting chunk" message
        await websocket.send_json({
            "type": "transcription_progress",
            "percentage": int((i / total_chunks) * 100),
            "message": f"Processing chunk {i+1}/{total_chunks}...",
            ...
        })
        
        # Run transcription in executor (non-blocking)
        loop = asyncio.get_event_loop()
        
        def transcribe_chunk():
            segments, info = fw_model.transcribe(chunk_file, ...)
            text_parts = [seg.text for seg in segments]
            return ' '.join(text_parts).strip(), info
        
        chunk_text, info = await loop.run_in_executor(None, transcribe_chunk)
        
        # AFTER processing - send chunk result + updated progress
        if chunk_text:
            await websocket.send_json({
                "type": "transcription_chunk",
                "text": chunk_text,
                ...
            })
        
        # Send completion progress
        await websocket.send_json({
            "type": "transcription_progress",
            "percentage": int(((i + 1) / total_chunks) * 100),
            ...
        })
```

**Benefits:**
- User sees "Processing chunk X/Y" immediately
- Event loop stays responsive
- Progress updates in real-time
- No late message batching

### FIX 2: Ensure 100% Progress Always Sent
**Strategy:** Always send final 100% completion after loop

```python
# After loop completes
await websocket.send_json({
    "type": "transcription_progress",
    "percentage": 100,
    "message": "Transcription complete",
    "chunks_processed": total_chunks,
    "total_chunks": total_chunks,
    "eta_seconds": 0
})
```

### FIX 3: Prominent Cache Message
**Strategy:** Send special message type that UI can display prominently

```python
if cached_file:
    file_size_mb = os.path.getsize(cached_file) / (1024 * 1024)
    await websocket.send_json({
        "type": "cached_file",  # New message type
        "message": f"✅ Audio file already downloaded ({file_size_mb:.1f} MB) - Using cache",
        "file_size_mb": file_size_mb,
        "skipped_download": True
    })
```

**UI Handler (index.html):**
```javascript
} else if (data.type === 'cached_file') {
    updateStatus('active', data.message);
    updateStage(1, 'completed');  // Mark download complete
    // Show prominent notification
    showCacheNotification(data.message);
}
```

### FIX 4: Hebrew Speaker Labels for Ivrit Models
**Strategy:** Detect Ivrit model and use Hebrew labels

```python
async def transcribe_with_diarization(
    model, model_config: dict, audio_file: str, language: Optional[str], 
    websocket: WebSocket, model_name: str = None
):
    # Detect if using Hebrew (Ivrit model)
    is_hebrew_model = model_name and "ivrit" in model_name.lower()
    speaker_prefix = "דובר_" if is_hebrew_model else "SPEAKER_"
    
    # ... diarization code ...
    
    # Renumber speakers with language-appropriate labels
    speaker_mapping = {}
    speaker_counter = 1
    for segment in speaker_segments:
        if segment["speaker"] not in speaker_mapping:
            speaker_mapping[segment["speaker"]] = f"{speaker_prefix}{speaker_counter}"
            speaker_counter += 1
        segment["speaker"] = speaker_mapping[segment["speaker"]]
```

---

## Implementation Steps

### Step 1: Fix Transcription Progress (HIGH PRIORITY)
1. Modify `transcribe_with_incremental_output()` (line 1776)
2. Move `fw_model.transcribe()` to executor
3. Send progress BEFORE and AFTER each chunk
4. Add final 100% completion message

### Step 2: Fix Hebrew Speaker Labels (HIGH PRIORITY)
1. Modify `transcribe_with_diarization()` (line 596)
2. Detect Ivrit model from `model_name`
3. Use "דובר_" prefix for Hebrew, "SPEAKER_" for others

### Step 3: Enhance Cache Notification (MEDIUM PRIORITY)
1. Add special "cached_file" message type in backend
2. Add UI handler in index.html
3. Show prominent notification when using cached file

### Step 4: Fix 100% Completion (MEDIUM PRIORITY)
1. Add explicit 100% message after transcription loop
2. Update UI to always show 100% on completion

---

## Testing Plan

### Test Case 1: Real-Time Progress
1. Upload a 5-minute Hebrew audio file
2. Select Ivrit CT2 model
3. **Expected:** 
   - See download progress in real-time
   - See "Processing chunk 1/X..." immediately
   - Progress bar updates smoothly (not in batch)
   - Reaches 100% when complete

### Test Case 2: Cached File
1. Upload same file twice
2. **Expected:**
   - First time: Normal download progress
   - Second time: "✅ Audio file already downloaded (X MB) - Using cache"
   - Download stage skips immediately to transcription

### Test Case 3: Hebrew Speaker Labels
1. Upload audio with 2 speakers
2. Select Ivrit CT2 model
3. Enable diarization
4. **Expected:**
   - Output shows "דובר_1" and "דובר_2"
   - Not "SPEAKER_1" and "SPEAKER_2"

### Test Case 4: Non-Hebrew Diarization
1. Upload English audio with 2 speakers
2. Select whisper-v3-turbo model
3. Enable diarization
4. **Expected:**
   - Output shows "SPEAKER_1" and "SPEAKER_2"

---

## Risk Assessment

### Low Risk Changes:
- ✅ Hebrew speaker labels (simple string replacement)
- ✅ Cache notification enhancement (additive change)
- ✅ Final 100% completion message (additive)

### Medium Risk Changes:
- ⚠️ Moving transcription to executor (changes async flow)
  - **Mitigation:** Test thoroughly with both short and long audio
  - **Fallback:** Keep original code path as fallback

### Breaking Change Risk: **VERY LOW**
- All changes are improvements to existing features
- No API changes
- No model loading changes
- UI changes are additive (new message handlers)

---

## Code Locations to Modify

| Fix | File | Lines | Function | Priority |
|-----|------|-------|----------|----------|
| Real-time progress | app.py | 1776-1856 | `transcribe_with_incremental_output()` | HIGH |
| Hebrew labels | app.py | 596-598 | `transcribe_with_diarization()` | HIGH |
| Cache notification | app.py | 822-830 | `download_audio_with_ffmpeg()` | MEDIUM |
| Cache UI handler | index.html | 1112-1202 | `ws.onmessage` | MEDIUM |
| 100% completion | app.py | 1856 | After transcription loop | MEDIUM |

---

## Summary

**Problems Identified:**
1. ❌ Transcription progress shown AFTER chunks complete (blocking sync calls)
2. ❌ Stuck at 82% (missing final 100% update)
3. ❌ Cache message not prominent enough
4. ❌ English speaker labels for Hebrew audio

**Solutions:**
1. ✅ Move transcription to executor, send progress BEFORE/AFTER each chunk
2. ✅ Add explicit 100% completion message
3. ✅ Add special "cached_file" message type with prominent UI notification
4. ✅ Detect Ivrit model and use "דובר_" prefix

**Impact:**
- Users see real-time progress during download AND transcription
- Clear indication when using cached files
- Proper Hebrew speaker labels for Hebrew audio
- 100% completion always displayed

**Risk:** Low - mostly additive changes, one async refactor with low risk

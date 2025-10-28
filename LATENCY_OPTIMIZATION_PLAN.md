# Live Transcription Latency Optimization Plan

## Executive Summary

After analyzing the current implementation, I've identified the root causes of the 30-second latency issue and developed a comprehensive optimization plan. The main bottlenecks are:

1. **Sequential Processing**: Audio chunks are processed one by one
2. **CPU-Only Processing**: No GPU acceleration despite available RTX 3090
3. **FFmpeg Normalization Overhead**: Each chunk requires separate FFmpeg processing
4. **Large Model Processing**: Using large models without optimization

## Current System Analysis

### ✅ Findings from Analysis

#### 1. Current Implementation (`app.py`)
- **Architecture**: FastAPI + WebSocket streaming
- **Audio Processing**: FFmpeg → 5-second chunks → Queue → Sequential transcription
- **Models**: OpenAI Whisper + Ivrit GGML model via whisper.cpp
- **Caching**: Audio normalization caching implemented
- **Bottleneck**: Sequential processing in `transcribe_audio_stream()`

#### 2. Docker & CUDA Configuration
- **GPU Available**: NVIDIA GeForce RTX 3090 (24GB VRAM)
- **CUDA Version**: 12.9
- **Docker Runtime**: NVIDIA runtime available
- **Current Setup**: CPU-only Docker container
- **Issue**: No GPU support in current Dockerfile

#### 3. Whisper Model Analysis
- **OpenAI Whisper**: Uses PyTorch 2.1.2 (CPU-only installation)
- **GPU Support**: Not enabled (`fp16=False` for CPU compatibility)
- **whisper.cpp**: Has `--no-gpu` option but current build is CPU-only
- **Opportunity**: Can enable GPU acceleration for both

#### 4. whisper.cpp Configuration
- **Current Build**: CPU-only (no CUDA libraries linked)
- **GPU Options**: `--no-gpu` flag available (implies GPU support possible)
- **Performance**: Using 4 threads, beam size 1 (already optimized for speed)
- **Opportunity**: Rebuild with CUDA support

## Optimization Strategy

### Phase 1: Immediate Latency Reduction (Quick Wins)

#### 1.1 Parallel Processing Architecture
**Current**: Sequential chunk processing
**Solution**: Multi-threaded transcription pipeline

```python
# New architecture:
# Audio Stream → Chunk Queue → Multiple Worker Threads → Result Aggregation → WebSocket
```

**Implementation**:
- Create worker thread pool (3-4 threads)
- Parallel transcription of multiple chunks
- Result ordering and streaming
- **Expected Improvement**: 60-70% latency reduction

#### 1.2 Optimized Chunk Processing
**Current**: 5-second chunks with 1-second overlap
**Solution**: Smaller chunks with parallel processing

```python
CHUNK_DURATION = 3   # Reduce from 5 to 3 seconds
CHUNK_OVERLAP = 0.5  # Reduce overlap
WORKER_THREADS = 4   # Parallel processing
```

**Expected Improvement**: 40% faster chunk processing

#### 1.3 FFmpeg Optimization
**Current**: Individual FFmpeg calls per chunk
**Solution**: Batch processing and optimized commands

```bash
# Optimized FFmpeg with hardware acceleration
ffmpeg -hwaccel auto -i input -f s16le -ar 16000 -ac 1 output
```

### Phase 2: GPU Acceleration (Major Performance Boost)

#### 2.1 Docker GPU Support
**Update Dockerfile**:
```dockerfile
# Use CUDA base image aligned with PyTorch cu118
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Install PyTorch with CUDA 11.8 support
RUN pip install torch==2.1.2+cu118 torchaudio==2.1.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

**Update docker-compose.yml**:
```yaml
services:
  transcription-app:
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

#### 2.2 Whisper GPU Acceleration
**OpenAI Whisper GPU Support**:
```python
import torch

# Enable GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model(config["name"], device=device)

# Enable FP16 when using GPU for faster processing
use_fp16 = device == "cuda"
result = model.transcribe(
    temp_path,
    language=processor.language,
    fp16=use_fp16,
    verbose=False,
)
```

#### 2.3 whisper.cpp GPU Build
**Rebuild with CUDA**:
```bash
cd whisper.cpp
cmake -B build -DGGML_CUDA=1 -DGGML_CUBLAS=1
cmake --build build -j --config Release
```

**CLI GPU Flags**:
- `-fa`/`--flash-attn` to enable flash attention (default true)
- `-nfa`/`--no-flash-attn` to disable flash attention
- `-ng`/`--no-gpu` to force CPU mode

**Expected Improvement**: 5-10x faster transcription

### Phase 3: File-Based Processing System (Alternative Approach)

#### 3.1 yt-dlp Integration Enhancement
**Current**: Basic yt-dlp download
**Solution**: Streaming extraction with chunking

```python
class StreamingAudioProcessor:
    def __init__(self, url: str):
        self.url = url
        self.chunk_size = 3  # seconds
        
    async def extract_and_chunk(self):
        # Use yt-dlp to extract audio stream
        # Split into overlapping chunks
        # Queue chunks for parallel processing
```

#### 3.2 Advanced Chunking Strategy
```python
# Smart chunking with voice activity detection
def create_smart_chunks(audio_data):
    # Use VAD to find speech boundaries
    # Create chunks at natural speech breaks
    # Minimize word cutting
```

## Implementation Plan

### 🚀 Phase 1: Quick Wins (1-2 days)
1. **Implement Parallel Processing**
   - Create worker thread pool
   - Modify `transcribe_audio_stream()` for parallel processing
   - Add result ordering mechanism

2. **Optimize Chunk Parameters**
   - Reduce chunk duration to 3 seconds
   - Optimize overlap to 0.5 seconds
   - Test performance improvements

3. **FFmpeg Optimization**
   - Add hardware acceleration flags
   - Optimize audio processing pipeline

### 🔥 Phase 2: GPU Acceleration (2-3 days)
1. **Update Docker Configuration**
   - Modify Dockerfile for CUDA support
   - Update docker-compose.yml for GPU access
   - Install PyTorch with CUDA

2. **Enable Whisper GPU Support**
   - Modify model loading for GPU
   - Enable FP16 processing
   - Test GPU utilization

3. **Rebuild whisper.cpp with CUDA**
   - Compile with CUDA support
   - Update binary in Docker image
   - Test GPU acceleration

### 🎯 Phase 3: Advanced Optimizations (3-4 days)
1. **Implement File-Based Processing**
   - Enhanced yt-dlp integration
   - Smart chunking algorithm
   - Voice activity detection

2. **Advanced Multithreading**
   - Separate threads for audio extraction, processing, transcription
   - Async pipeline with queues
   - Load balancing

## Rollback Strategy

### Backup Current Implementation
```bash
# Create backup branch
git checkout -b backup-original-implementation
git add .
git commit -m "Backup: Original implementation before optimization"

# Create rollback script
cp app.py app.py.backup
cp Dockerfile Dockerfile.backup
cp docker-compose.yml docker-compose.yml.backup
```

### Rollback Procedure
1. **Quick Rollback**: `git checkout backup-original-implementation`
2. **Partial Rollback**: Restore specific files from backup
3. **Docker Rollback**: Use original Docker images

### Testing Strategy
1. **Unit Tests**: Test each component separately
2. **Integration Tests**: Test full pipeline
3. **Performance Tests**: Measure latency improvements
4. **Load Tests**: Test with multiple concurrent streams

## Expected Performance Improvements

| Optimization | Current Latency | Expected Latency | Improvement |
|--------------|----------------|------------------|-------------|
| Baseline | 30 seconds | 30 seconds | 0% |
| Parallel Processing | 30 seconds | 10-12 seconds | 60-65% |
| + GPU Acceleration | 10-12 seconds | 2-3 seconds | 80-85% |
| + Advanced Chunking | 2-3 seconds | 1-2 seconds | 90-95% |

## Risk Assessment

### Low Risk
- Parallel processing implementation
- Chunk parameter optimization
- FFmpeg optimization

### Medium Risk
- Docker GPU configuration
- PyTorch CUDA installation
- whisper.cpp rebuild

### High Risk
- Complete architecture overhaul
- File-based processing system
- Advanced multithreading

## Success Metrics

### Primary Metrics
- **Latency**: Reduce from 30s to <3s
- **Throughput**: Maintain or improve
- **Accuracy**: No degradation in transcription quality

### Secondary Metrics
- **GPU Utilization**: >70% during transcription
- **CPU Usage**: Reduce CPU load
- **Memory Usage**: Efficient memory management
- **Error Rate**: <1% processing errors

## Next Steps

1. **Immediate**: Start with Phase 1 (Parallel Processing)
2. **Week 1**: Complete Phase 1 and begin Phase 2
3. **Week 2**: Complete GPU acceleration
4. **Week 3**: Implement advanced optimizations if needed

## Conclusion

This optimization plan addresses the 30-second latency issue through a phased approach:

1. **Quick wins** with parallel processing (60-65% improvement)
2. **Major boost** with GPU acceleration (additional 5-10x improvement)
3. **Advanced optimizations** for near real-time performance

## Plan Addendum: File-Based Chunking with yt-dlp + Multithreading (Reversible)

### Goals
- Reduce perceived latency by emitting partial results continuously, not in 30-second batches.
- Avoid breaking current streaming functionality; integrate behind feature flags with immediate rollback.
- Leverage a Vibe-style approach: record/extract audio to local files while transcribing chunk-by-chunk concurrently.

### Design Overview
- Capture: Use `yt-dlp` (or `ffmpeg` via `yt-dlp` stream URL) to extract audio and write it locally as normalized PCM WAV.
- Segment: Split the growing audio file into fixed-size segments (default `5s`), optionally with small overlap.
- Queue: Push segments into a bounded `Queue` for back-pressure.
- Workers: A pool of 3–4 transcription worker threads consume chunks concurrently.
- Ordering: Maintain per-source monotonic segment indices; reorder results before emitting to UI.
- Emit: Stream partial transcripts to the UI as soon as each segment completes.

### Pipeline Details
- Source input: URL or stream provided by the existing API.
- Normalization: `ffmpeg -hide_banner -loglevel error -i <in> -ar 16000 -ac 1 -f wav <out>` (can add `-hwaccel auto`).
- Segmentation options:
  - File-based: `ffmpeg -i <wav> -f segment -segment_time 5 -c copy segments/%03d.wav`
  - Rolling writer + cutter: write a single WAV and cut every 5s via an indexer thread.
- Overlap: 0.5–1.0s if accuracy drops; configurable.

### Multithreading Model
- `RecorderThread`: spawns `yt-dlp`/`ffmpeg` and writes normalized PCM to disk.
- `SegmenterThread`: watches file size/time, emits 5s segments and places them on `transcribe_queue`.
- `TranscribeWorker[n]`: N threads pop from `transcribe_queue`, run model (OpenAI Whisper or whisper.cpp) and push results onto `results_queue` with segment index.
- `AggregatorThread`: orders by segment index and emits to WebSocket/UI immediately.

### Integration Points (app.py)
- Add a new optional path (guarded by env flags) that uses the pipeline above when a URL stream is provided.
- Preserve current path for direct streaming and existing models.
- Detect CUDA availability to choose GPU-enabled settings for Whisper and whisper.cpp.

### Feature Flags (Non-breaking)
- `USE_FILE_PIPELINE` (bool): enable the file-based chunking pipeline (default `false`).
- `CHUNK_SECONDS` (int): segment length, default `5`.
- `CHUNK_OVERLAP_SECONDS` (float): default `0.5`.
- `WORKER_THREADS` (int): transcription workers, default `4`.
- `USE_WHISPER_CPP_GPU` (bool): default `true` when CUDA is detected.
- `USE_WHISPER_TORCH_GPU` (bool): default `true` when CUDA is detected.
- `EMIT_PARTIALS` (bool): stream each segment result immediately, default `true`.

### whisper.cpp CLI Flags
- `-fa`/`--flash-attn` to enable flash attention; `-nfa` to disable.
- `-ng`/`--no-gpu` to force CPU mode if GPU is unstable.
- Threading: tune `-t <N>` and beam size `-bs 1` for speed.

### Ordering & UX
- Each segment carries `(source_id, segment_index)`.
- The aggregator emits transcripts as soon as contiguous indices are available to avoid large delays.
- UI can display a subtle “live” indicator per line to reflect out-of-order arrivals, then reflow on aggregation.

### Monitoring & Back-Pressure
- Bounded queues prevent memory blow-up; if workers fall behind, the segmenter can slow down.
- Log processing times per stage; track GPU utilization and worker throughput.

### Testing Plan (Pre-Implementation)
- Dry-run mode: generate fake segments to validate threading, ordering, and UI emission without model cost.
- File replay: use a known WAV to simulate `yt-dlp` output and validate segment boundaries.
- Model A/B: compare Whisper vs whisper.cpp on identical segments for performance and accuracy.

### Revert Strategy (One-Command Rollback)
- Keep all changes behind flags; default behavior remains existing streaming path.
- To revert: set `USE_FILE_PIPELINE=false` (or remove), and restore previous Docker/image if needed.
- Maintain a backup branch and copy of updated files before implementation.

### Implementation Steps (Next PR)
1. Add env/config flags and stubbed pipeline classes/threads (no functional changes yet).
2. Integrate `yt-dlp`/`ffmpeg` recorder and the segmenter thread.
3. Implement worker pool with safe shutdown and error handling.
4. Wire aggregator to WebSocket/UI emission, preserving current API schema.
5. Add metrics and logs; document usage in README and QUICKSTART.
6. Bench and tune `CHUNK_SECONDS`, overlap, and worker count.

This addendum documents a reversible, non-breaking plan modeled on Vibe’s record-and-transcribe approach, tailored to your current architecture and GPU setup.

The plan maintains backward compatibility and includes comprehensive rollback strategies to ensure system stability.
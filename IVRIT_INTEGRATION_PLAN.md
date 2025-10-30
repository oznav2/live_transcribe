# Ivrit AI Integration Plan

## Current State Analysis

### Current Setup
- **Model**: Using GGML version of Ivrit (whisper.cpp binary format)
- **Quality**: Poor transcription quality reported
- **Runtime**: whisper.cpp CLI with CUDA support
- **Base Image**: Python 3.11 slim with custom CUDA libraries

### Ivrit AI Models Available
1. **ivrit-ai/whisper-large-v3-turbo-ct2** - Hebrew-optimized Whisper model (CTranslate2 format)
2. **ivrit-ai/yi-whisper-large-v3-turbo-ct2** - Yiddish variant
3. **large-v3-turbo** - Standard Whisper turbo model
4. **pyannote-speaker-diarization-3.1** - Speaker diarization
5. **speechbrain/spkrec-ecapa-voxceleb** - Speaker recognition

## Integration Strategy

### Approach 1: Direct Integration (Recommended)
Integrate Ivrit models directly into the application using faster_whisper, avoiding RunPod dependency.

**Advantages:**
- No external API dependency
- Lower latency (local inference)
- Better control over model loading
- Cost-effective (no RunPod fees)

**Requirements:**
- PyTorch 2.4.1 with CUDA 12.1
- faster_whisper for CTranslate2 models
- ivrit[all]==0.1.8 package
- Sufficient GPU memory for models

### Approach 2: RunPod Integration (Alternative)
Use RunPod serverless endpoint as shown in the repository.

**Advantages:**
- Offload compute to RunPod
- Scalable for high load

**Disadvantages:**
- External dependency
- API costs
- Network latency
- Complexity

## Implementation Plan

### Phase 1: Update Base Image
Switch from Python slim to PyTorch base with CUDA 12.1 for better compatibility.

### Phase 2: Install Ivrit Dependencies
- ivrit[all]==0.1.8
- faster_whisper
- torch==2.4.1
- pyannote.audio for diarization
- speechbrain for speaker recognition

### Phase 3: Model Integration in app.py
- Add new model type "ivrit_ct2" for CTranslate2 models
- Implement faster_whisper backend
- Support for multiple Ivrit models
- Optional diarization support

### Phase 4: Docker Optimization
- Multi-stage build to minimize image size
- Pre-download models during build
- Proper CUDA library paths
- Health checks for model availability

## File Changes Required

### 1. Dockerfile
- Switch to PyTorch base image
- Add Ivrit dependencies
- Pre-load models
- Configure LD_LIBRARY_PATH

### 2. requirements.txt
- Add ivrit[all]==0.1.8
- Add faster_whisper
- Update torch to 2.4.1
- Add pyannote.audio
- Add speechbrain

### 3. app.py
- Add IvritCT2Model class
- Implement faster_whisper transcription
- Add model switching logic
- Support diarization option

### 4. docker-compose.yml
- Update environment variables
- Add Ivrit model selection
- Configure GPU memory

## Memory Requirements

### Model Sizes (Approximate)
- whisper-large-v3-turbo-ct2: ~3GB
- yi-whisper-large-v3-turbo-ct2: ~3GB
- pyannote diarization: ~500MB
- speechbrain speaker: ~200MB

### GPU Memory Needed
- Minimum: 8GB VRAM
- Recommended: 12GB+ VRAM
- With diarization: 16GB+ VRAM

## Configuration Options

### Environment Variables
```yaml
IVRIT_MODEL_TYPE: "ct2"  # or "ggml" for old version
IVRIT_MODEL_NAME: "ivrit-ai/whisper-large-v3-turbo-ct2"
IVRIT_ENABLE_DIARIZATION: "false"
IVRIT_DEVICE: "cuda"
IVRIT_COMPUTE_TYPE: "float16"
IVRIT_BEAM_SIZE: 5
```

## Risk Mitigation

1. **Backward Compatibility**: Keep GGML support as fallback
2. **Memory Management**: Implement model unloading when switching
3. **Error Handling**: Graceful degradation if models fail to load
4. **Testing**: Thorough testing with Hebrew audio samples
5. **Monitoring**: Add metrics for model performance

## Performance Expectations

### Speed Improvements
- Current GGML: ~0.3x realtime
- Expected CT2: ~1.0-2.0x realtime
- With GPU optimization: Up to 3x realtime

### Quality Improvements
- Better Hebrew language model
- More accurate transcription
- Optional speaker diarization
- Word-level timestamps

## Rollback Plan

If integration fails:
1. Keep original Dockerfile as Dockerfile.legacy
2. Maintain GGML model support in app.py
3. Environment variable to switch between implementations
4. Quick revert via docker-compose target change
## Base Python runtime stage
FROM python:3.11-slim AS base

# Install system dependencies including FFmpeg, build tools, and Python pip
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    build-essential \
    cmake \
    python3-pip \
    portaudio19-dev \
    && pip3 install yt-dlp \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (PyTorch CUDA 11.8 wheels)
# First install torch with CUDA support, then install other dependencies
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
        torch==2.1.2+cu118 torchaudio==2.1.2+cu118

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Verify PyTorch installation (optional - won't fail build if CUDA unavailable)
RUN python -c "import torch; print('✓ PyTorch installed:', torch.__version__)" || echo "Warning: PyTorch verification failed"

# Verify Deepgram SDK installation
RUN python -c "from deepgram import DeepgramClient; import deepgram; print('✓ Deepgram SDK installed:', deepgram.__version__)"

# Copy application code
COPY app.py .
COPY static/ ./static/

# Create directory for Whisper models and copy the converted Ivrit model
RUN mkdir -p /root/.cache/whisper && \
    mkdir -p /app/models
COPY models/ivrit-whisper-large-v3-turbo.bin /app/models/

# Expose port
EXPOSE 8009

# Environment variables
ENV WHISPER_MODEL=ivrit-large-v3-turbo
ENV IVRIT_MODEL_PATH=/app/models/ivrit-whisper-large-v3-turbo.bin
ENV WHISPER_CPP_PATH=/app/whisper.cpp/build/bin/whisper-cli
ENV PORT=8009
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8009/health')" || exit 1

##############################################
# Runtime (prebuilt): copy locally built artifacts
##############################################
FROM base AS runtime-prebuilt

# Prepare directories
RUN mkdir -p /app/whisper.cpp/build/bin && \
    mkdir -p /app/whisper.cpp/build/src && \
    mkdir -p /app/whisper.cpp/build/ggml/src && \
    mkdir -p /usr/local/cuda/lib64

## Copy locally built whisper.cpp artifacts directly from repo (simpler than ./prebuilt)
## Make sure you have run the host build: `cmake -S whisper.cpp -B whisper.cpp/build -DGGML_CUDA=1 && cmake --build whisper.cpp/build -j`
COPY whisper.cpp/build/bin/whisper-cli /app/whisper.cpp/build/bin/whisper-cli
COPY whisper.cpp/build/src/libwhisper.so* /app/whisper.cpp/build/src/
COPY whisper.cpp/build/ggml/src/libggml*.so* /app/whisper.cpp/build/ggml/src/

# Bundle essential CUDA runtime libs required by ggml CUDA backend
# We use a lightweight CUDA runtime stage to avoid building whisper.cpp in-container
##############################################
# CUDA runtime libs stage (no build, just libs)
##############################################
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS cuda-runtime-libs

##############################################
# Back to runtime-prebuilt to include CUDA libs
##############################################
FROM base AS runtime-prebuilt-with-cuda

# Prepare directories
RUN mkdir -p /app/whisper.cpp/build/bin && \
    mkdir -p /app/whisper.cpp/build/src && \
    mkdir -p /app/whisper.cpp/build/ggml/src && \
    mkdir -p /usr/local/cuda/lib64

# Copy locally built whisper.cpp artifacts directly from repo
COPY whisper.cpp/build/bin/whisper-cli /app/whisper.cpp/build/bin/whisper-cli
COPY whisper.cpp/build/src/libwhisper.so* /app/whisper.cpp/build/src/
COPY whisper.cpp/build/ggml/src/libggml*.so* /app/whisper.cpp/build/ggml/src/

# Copy CUDA runtime libraries needed by ggml CUDA backend
COPY --from=cuda-runtime-libs /usr/local/cuda/lib64/libcudart.so* /usr/local/cuda/lib64/
COPY --from=cuda-runtime-libs /usr/local/cuda/lib64/libnvrtc.so* /usr/local/cuda/lib64/

# Set library path for whisper.cpp and CUDA runtime libs
ENV LD_LIBRARY_PATH=/app/whisper.cpp/build/src:/app/whisper.cpp/build/ggml/src:/usr/local/cuda/lib64

# Expose port and run
EXPOSE 8009
CMD ["python", "app.py"]

##############################################
# CUDA builder stage for whisper.cpp (GGML_CUDA)
##############################################
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS whispercpp-builder

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy local whisper.cpp source to builder
COPY whisper.cpp/ /src/whisper.cpp/

# Build whisper.cpp with CUDA support (no cuBLAS to minimize runtime deps)
RUN cmake -S /src/whisper.cpp -B /build \
      -DGGML_CUDA=1 \
      -DGGML_CUBLAS=0 \
      -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /build -j --config Release

##############################################
# Runtime (built): copy artifacts from builder
##############################################
FROM base AS runtime-built

# Prepare directories
RUN mkdir -p /app/whisper.cpp/build/bin && \
    mkdir -p /app/whisper.cpp/build/src && \
    mkdir -p /app/whisper.cpp/build/ggml/src && \
    mkdir -p /usr/local/cuda/lib64

# Copy CUDA-built whisper.cpp binary and shared libraries from builder
COPY --from=whispercpp-builder /build/bin/whisper-cli /app/whisper.cpp/build/bin/whisper-cli
COPY --from=whispercpp-builder /build/src/libwhisper.so* /app/whisper.cpp/build/src/
COPY --from=whispercpp-builder /build/ggml/src/libggml*.so* /app/whisper.cpp/build/ggml/src/

# Bundle essential CUDA runtime libs required by ggml CUDA backend
COPY --from=whispercpp-builder /usr/local/cuda/lib64/libcudart.so* /usr/local/cuda/lib64/
COPY --from=whispercpp-builder /usr/local/cuda/lib64/libnvrtc.so* /usr/local/cuda/lib64/

# Set library path for whisper.cpp and CUDA runtime
ENV LD_LIBRARY_PATH=/app/whisper.cpp/build/src:/app/whisper.cpp/build/ggml/src:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Expose port and run
EXPOSE 8009
CMD ["python", "app.py"]
